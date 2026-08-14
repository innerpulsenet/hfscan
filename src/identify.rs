//! Signature matching across the current SDR span.
//!
//! The radio already delivers ~192 kHz of IQ — a whole amateur band slice.
//! Occupied regions are mixed to baseband and scored with the same cues a
//! human uses on a waterfall: occupied bandwidth, tonality, tone spacing,
//! envelope keying, and a residual carrier. That is enough to separate
//! CW, PSK31, RTTY, an FT8/FT4 pile-up, SSB voice, and AM without
//! running every decoder on every hertz.

use crate::decoders::cw;
use crate::decoders::psk31;
use crate::bands;
use num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::f32::consts::PI;
use std::sync::Arc;

const AUDIO: f32 = 8_000.0;
const PSD_N: usize = 2048;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Cw,
    Psk31,
    Rtty,
    Ft8,
    Ft4,
    Ssb,
    Am,
    Carrier,
    Unknown,
}

impl Kind {
    pub fn label(self) -> &'static str {
        match self {
            Kind::Cw => "CW",
            Kind::Psk31 => "PSK",
            Kind::Rtty => "RTTY",
            Kind::Ft8 => "FT8",
            Kind::Ft4 => "FT4",
            Kind::Ssb => "SSB",
            Kind::Am => "AM",
            Kind::Carrier => "CAR",
            Kind::Unknown => "?",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Ident {
    /// Offset from the radio centre, Hz.
    pub offset_hz: f32,
    pub bw_hz: f32,
    pub snr_db: f32,
    pub kind: Kind,
    /// 0..1 confidence.
    pub score: f32,
}

/// Classify occupied regions in `spectrum` using a short IQ buffer.
pub fn classify_span(
    iq: &[Complex32],
    fs: f64,
    spectrum: &[f32],
    abs_center: f64,
) -> Vec<Ident> {
    if iq.len() < (fs * 0.25) as usize || spectrum.len() < 16 {
        return Vec::new();
    }
    let mut occ = occupancies(spectrum, fs);
    occ.sort_by(|a, b| b.snr_db.partial_cmp(&a.snr_db).unwrap_or(std::cmp::Ordering::Equal));
    occ.truncate(20);

    let mut fft = Psd::new();
    let mut out = Vec::new();
    for o in occ {
        let audio = mix_decim(iq, fs as f32, o.offset_hz, (fs / AUDIO as f64).round().max(1.0) as usize);
        if audio.len() < (AUDIO * 0.2) as usize {
            continue;
        }
        let abs = abs_center + o.offset_hz as f64;
        let (kind, score) = classify_one(&audio, o.bw_hz, abs, &mut fft);
        if kind == Kind::Unknown && score < 0.35 {
            continue;
        }
        if out
            .iter()
            .any(|i: &Ident| (i.offset_hz - o.offset_hz).abs() < o.bw_hz.max(80.0) * 0.5)
        {
            continue;
        }
        out.push(Ident {
            offset_hz: o.offset_hz,
            bw_hz: o.bw_hz,
            snr_db: o.snr_db,
            kind,
            score,
        });
    }
    out.sort_by(|a, b| {
        a.offset_hz
            .partial_cmp(&b.offset_hz)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

struct Occ {
    offset_hz: f32,
    snr_db: f32,
    bw_hz: f32,
}

fn occupancies(spectrum: &[f32], rate: f64) -> Vec<Occ> {
    let n = spectrum.len();
    let mut sorted = spectrum.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = sorted[n / 2];
    let thr = med + 8.0;
    let dc = n / 2;
    let bin = rate as f32 / n as f32;
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        if spectrum[i] < thr {
            i += 1;
            continue;
        }
        let start = i;
        let mut best: Option<usize> = None;
        while i < n && spectrum[i] >= thr {
            // The LO spike is not a signal, but do not split a wide
            // occupancy that happens to straddle DC.
            if i.abs_diff(dc) > 2 && best.is_none_or(|b| spectrum[i] > spectrum[b]) {
                best = Some(i);
            }
            i += 1;
        }
        let Some(best) = best else {
            continue;
        };
        let bw = ((i - start) as f32 * bin).max(bin);
        let off = (best as f32 - n as f32 / 2.0) * bin;
        out.push(Occ {
            offset_hz: off,
            snr_db: spectrum[best] - med,
            bw_hz: bw,
        });
    }
    out
}

fn classify_one(audio: &[Complex32], coarse_bw: f32, abs_hz: f64, psd: &mut Psd) -> (Kind, f32) {
    let spec = psd.power(audio);
    let fine = features(&spec, AUDIO / spec.len() as f32, audio);

    // Known-mode probes first — they are specific and cheap enough.
    // Keep them on narrow occupancies so a 2 kHz SSB passband with
    // syllabic AM cannot look like keyed CW.
    let psk = psk31::scan_span(audio, AUDIO as f64, &[(0.0, 20.0)]);
    if let Some(h) = psk.iter().max_by(|a, b| {
        a.quality
            .partial_cmp(&b.quality)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        if h.quality > 0.7 && coarse_bw < 250.0 && fine.obw_hz < 250.0 {
            return (Kind::Psk31, h.quality);
        }
    }
    let cw = cw::scan_span(audio, AUDIO as f64, &[(0.0, 20.0)]);
    if let Some(h) = cw.iter().max_by(|a, b| {
        a.quality
            .partial_cmp(&b.quality)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        if h.quality > 0.45 && coarse_bw < 250.0 && fine.obw_hz < 200.0 {
            return (Kind::Cw, h.quality);
        }
    }

    if let Some(sep) = fine.dual_hz {
        if (150.0..220.0).contains(&sep) && fine.env_cv < 0.35 && fine.obw_hz < 600.0 {
            return (Kind::Rtty, 0.75);
        }
    }

    let marker = marker_kind(abs_hz);

    if fine.obw_hz > 1500.0 && fine.n_peaks >= 6 {
        if let Some(sp) = fine.spacing_hz {
            if (5.0..9.0).contains(&sp) || (11.0..14.0).contains(&sp) {
                return (Kind::Ft8, 0.8);
            }
            if (18.0..26.0).contains(&sp) {
                return (Kind::Ft4, 0.78);
            }
        }
        // A 2–3 kHz forest of narrow tones on an FT8 calling frequency.
        if matches!(marker, Some(Kind::Ft8) | Some(Kind::Ft4)) && fine.n_peaks >= 8 {
            return (marker.unwrap(), 0.7);
        }
    }

    if fine.carrier && fine.obw_hz > 3500.0 {
        return (Kind::Am, 0.72);
    }
    if fine.carrier && fine.obw_hz < 200.0 && fine.env_cv < 0.2 {
        return (Kind::Carrier, 0.65);
    }

    let wide = coarse_bw.max(fine.obw_hz);
    if (1200.0..3600.0).contains(&wide)
        && !fine.carrier
        && fine.n_peaks < 8
        && fine.speech > 0.35
    {
        return (Kind::Ssb, 0.62 + 0.2 * fine.speech.min(1.0));
    }
    if (1200.0..3600.0).contains(&wide) && !fine.carrier && fine.tonality < 8.0 {
        return (Kind::Ssb, 0.5);
    }

    if coarse_bw < 150.0 && fine.tonality > 10.0 && fine.env_cv < 0.15 {
        return (Kind::Carrier, 0.5);
    }
    // A lone FT8/FT4 is only ~50–80 Hz of tones; the pile-up test above
    // misses it. Trust the calling-frequency marker when the occupancy
    // is that narrow and not a dead carrier.
    if let Some(k @ (Kind::Ft8 | Kind::Ft4)) = marker {
        if (35.0..140.0).contains(&fine.obw_hz) && !fine.carrier {
            return (k, 0.55);
        }
    }
    // Calling-frequency markers are a last-ditch hint, not a vote.
    if let Some(k) = marker {
        if coarse_bw < 200.0 && fine.n_peaks <= 4 {
            return (k, 0.35);
        }
    }
    (Kind::Unknown, 0.2)
}

/// Compact "3 CW  2 SSB  1 PSK" tally for the status line / spectrum title.
pub fn summary(idents: &[Ident]) -> String {
    const ORDER: [Kind; 9] = [
        Kind::Cw,
        Kind::Psk31,
        Kind::Rtty,
        Kind::Ft8,
        Kind::Ft4,
        Kind::Ssb,
        Kind::Am,
        Kind::Carrier,
        Kind::Unknown,
    ];
    let mut parts = Vec::new();
    for k in ORDER {
        let n = idents.iter().filter(|i| i.kind == k).count();
        if n > 0 {
            parts.push(format!("{n} {}", k.label()));
        }
    }
    parts.join("  ")
}

fn marker_kind(abs_hz: f64) -> Option<Kind> {
    let m = bands::MARKERS
        .iter()
        .find(|m| (m.freq - abs_hz).abs() < 2500.0)?;
    match m.label {
        "FT8" => Some(Kind::Ft8),
        "PSK" => Some(Kind::Psk31),
        "RTTY" => Some(Kind::Rtty),
        _ => None,
    }
}

struct Feat {
    obw_hz: f32,
    tonality: f32,
    n_peaks: usize,
    spacing_hz: Option<f32>,
    dual_hz: Option<f32>,
    carrier: bool,
    env_cv: f32,
    speech: f32,
}

fn features(psd: &[f32], bin_hz: f32, audio: &[Complex32]) -> Feat {
    let n = psd.len();
    let mut sorted = psd.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = sorted[n / 2].max(1e-20);
    let peak = sorted[n - 1];
    // Median×8 is ~9 dB — fine on a noisy band, but a clean two-tone
    // test (or a quiet night) has a tiny median and then FFT leakage
    // inflates occupied bandwidth into "FT8 pile-up" territory.
    // Also require the bin to be within ~13 dB of the peak.
    let thr = (med * 8.0).max(peak * 0.05);

    let signed_hz = |i: usize| -> f32 {
        if i <= n / 2 {
            i as f32 * bin_hz
        } else {
            (i as f32 - n as f32) * bin_hz
        }
    };
    let mut occ_hz: Vec<f32> = Vec::new();
    // (power, signed Hz) of local maxima above the noise floor.
    let mut peaks: Vec<(f32, f32)> = Vec::new();
    for i in 1..n - 1 {
        if psd[i] >= thr {
            occ_hz.push(signed_hz(i));
        }
        if psd[i] > psd[i - 1] && psd[i] >= psd[i + 1] && psd[i] >= thr {
            peaks.push((psd[i], signed_hz(i)));
        }
    }
    occ_hz.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let obw = match (occ_hz.first(), occ_hz.last()) {
        (Some(a), Some(b)) => (b - a).max(bin_hz),
        _ => bin_hz,
    };
    let mut freqs: Vec<f32> = peaks.iter().map(|p| p.1).collect();
    freqs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut diffs: Vec<f32> = freqs
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|d| *d > 2.0)
        .collect();
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Prefer a FT8/FT4-like grid if one is present; otherwise the median gap.
    let ftish: Vec<f32> = diffs
        .iter()
        .copied()
        .filter(|d| (5.0..28.0).contains(d))
        .collect();
    let spacing = if ftish.len() >= 2 {
        Some(ftish[ftish.len() / 2])
    } else if diffs.len() >= 2 {
        Some(diffs[diffs.len() / 2])
    } else {
        None
    };
    peaks.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    // Prefer a 170 Hz pair (RTTY) among the strong peaks. The two
    // loudest bins are often neighbouring keying sidebands of one tone.
    let dual = rtty_sep(&peaks).or_else(|| {
        if peaks.len() >= 2 {
            Some((peaks[0].1 - peaks[1].1).abs())
        } else {
            None
        }
    });

    let dc = psd[0];
    let neigh = psd.get(1).copied().unwrap_or(dc);
    let carrier = dc > neigh * 3.0 && dc > med * 12.0;

    let (env_cv, speech) = envelope_stats(audio);
    Feat {
        obw_hz: obw.min(AUDIO / 2.0),
        tonality: peak / med,
        n_peaks: peaks.len(),
        spacing_hz: spacing,
        dual_hz: dual,
        carrier,
        env_cv,
        speech,
    }
}

fn rtty_sep(peaks: &[(f32, f32)]) -> Option<f32> {
    if peaks.len() < 2 {
        return None;
    }
    let top = peaks[0].0;
    let n = peaks.len().min(10);
    let mut best_sep = None;
    let mut best_weak = 0.0f32;
    for i in 0..n {
        if peaks[i].0 < top * 0.1 {
            continue;
        }
        for j in i + 1..n {
            if peaks[j].0 < top * 0.1 {
                continue;
            }
            let sep = (peaks[i].1 - peaks[j].1).abs();
            if (150.0..230.0).contains(&sep) {
                let weak = peaks[i].0.min(peaks[j].0);
                if weak > best_weak {
                    best_weak = weak;
                    best_sep = Some(sep);
                }
            }
        }
    }
    best_sep
}

fn envelope_stats(audio: &[Complex32]) -> (f32, f32) {
    if audio.is_empty() {
        return (0.0, 0.0);
    }
    let decim = 40usize; // 8 kHz -> 200 Hz
    let mut env = Vec::new();
    let mut acc = 0.0;
    let mut n = 0;
    for s in audio {
        acc += s.norm();
        n += 1;
        if n == decim {
            env.push(acc / decim as f32);
            acc = 0.0;
            n = 0;
        }
    }
    if env.len() < 16 {
        return (0.0, 0.0);
    }
    let mean = env.iter().sum::<f32>() / env.len() as f32;
    let var = env.iter().map(|e| (e - mean) * (e - mean)).sum::<f32>() / env.len() as f32;
    let cv = var.sqrt() / mean.max(1e-9);

    // Syllabic AM lives around 2–8 Hz. A 200 Hz env rate, 64-pt Goertzel bank.
    let n = env.len().min(128);
    let slice = &env[env.len() - n..];
    let mut low = 0.0f32;
    let mut mid = 0.0f32;
    let fs = 200.0f32;
    for k in 1..16 {
        let f = k as f32 * fs / n as f32;
        let mut acc = Complex32::new(0.0, 0.0);
        let w = -2.0 * PI * k as f32 / n as f32;
        for (i, &e) in slice.iter().enumerate() {
            let (sin, cos) = (w * i as f32).sin_cos();
            acc += Complex32::new(e, 0.0) * Complex32::new(cos, sin);
        }
        let p = acc.norm_sqr();
        if (2.0..10.0).contains(&f) {
            low += p;
        } else if (12.0..40.0).contains(&f) {
            mid += p;
        }
    }
    let speech = low / (low + mid).max(1e-20);
    (cv, speech)
}

struct Psd {
    fft: Arc<dyn Fft<f32>>,
    buf: Vec<Complex32>,
    window: Vec<f32>,
}

impl Psd {
    fn new() -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(PSD_N);
        let window: Vec<f32> = (0..PSD_N)
            .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / PSD_N as f32).cos())
            .collect();
        Self {
            fft,
            buf: vec![Complex32::new(0.0, 0.0); PSD_N],
            window,
        }
    }

    fn power(&mut self, audio: &[Complex32]) -> Vec<f32> {
        let hop = PSD_N / 2;
        let mut acc = vec![0.0f32; PSD_N];
        let mut nseg = 0;
        let mut i = 0;
        while i + PSD_N <= audio.len() {
            for k in 0..PSD_N {
                self.buf[k] = audio[i + k] * self.window[k];
            }
            self.fft.process(&mut self.buf);
            for k in 0..PSD_N {
                acc[k] += self.buf[k].norm_sqr();
            }
            nseg += 1;
            i += hop;
        }
        if nseg == 0 {
            return vec![1e-20; PSD_N];
        }
        let s = 1.0 / nseg as f32;
        acc.iter().map(|v| v * s + 1e-20).collect()
    }
}

fn mix_decim(iq: &[Complex32], fs: f32, hz: f32, decim: usize) -> Vec<Complex32> {
    let step = -2.0 * PI * hz / fs;
    let mut phase = 0.0f32;
    let mut acc = Complex32::new(0.0, 0.0);
    let mut n = 0usize;
    let mut out = Vec::with_capacity(iq.len() / decim + 1);
    let scale = 1.0 / decim as f32;
    for &s in iq {
        let (sin, cos) = phase.sin_cos();
        phase += step;
        if phase > PI {
            phase -= 2.0 * PI;
        } else if phase < -PI {
            phase += 2.0 * PI;
        }
        acc += s * Complex32::new(cos, sin);
        n += 1;
        if n == decim {
            out.push(acc * scale);
            acc = Complex32::new(0.0, 0.0);
            n = 0;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keyed_cw(n: usize, fs: f32, wpm: f32, hz: f32) -> Vec<Complex32> {
        let dit = (1.2 / wpm * fs) as usize;
        let mut out = Vec::with_capacity(n);
        let mut on = true;
        let mut phase = 0.0f32;
        let step = 2.0 * PI * hz / fs;
        while out.len() < n {
            for _ in 0..dit {
                if out.len() >= n {
                    break;
                }
                let a = if on { 1.0 } else { 0.0 };
                out.push(Complex32::from_polar(a, phase));
                phase += step;
            }
            on = !on;
        }
        out
    }

    /// Fake display spectrum: energy at `hz`, away from the LO bin.
    fn spec_peak(n: usize, fs: f32, hz: f32, bw_bins: usize) -> Vec<f32> {
        let mut spec = vec![-85.0f32; n];
        let bin = (n as f32 / 2.0 + hz / (fs / n as f32)).round() as isize;
        let half = bw_bins as isize;
        for d in -half..=half {
            let i = (bin + d).clamp(0, n as isize - 1) as usize;
            spec[i] = if d == 0 { -40.0 } else { -52.0 };
        }
        spec
    }

    #[test]
    fn classifies_keyed_cw() {
        let sig = keyed_cw(8000, 8000.0, 20.0, 400.0);
        let spec = spec_peak(256, 8000.0, 400.0, 1);
        // Park off a calling-frequency marker so a failed probe cannot
        // fall through to "this is the PSK watering hole".
        let ids = classify_span(&sig, 8000.0, &spec, 7_020_000.0);
        assert!(
            ids.iter().any(|i| i.kind == Kind::Cw),
            "expected CW, got {ids:?}"
        );
    }

    #[test]
    fn classifies_ssb_like_noise() {
        let mut rng = 0xC0FFEEu32;
        let mut sig = Vec::with_capacity(8000);
        let mut phase = 0.0f32;
        for i in 0..8000 {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            let n = (rng as f32 / u32::MAX as f32) - 0.5;
            // Syllabic AM at 4 Hz, energy spread (no carrier).
            let env = 0.55 + 0.45 * (2.0 * PI * 4.0 * i as f32 / 8000.0).sin();
            phase += 2.0 * PI * 1200.0 / 8000.0;
            sig.push(Complex32::from_polar(n.abs() * env * 0.4, phase + n));
        }
        let mut spec = vec![-85.0f32; 256];
        for b in 80..160 {
            spec[b] = -55.0;
        }
        spec[120] = -50.0;
        let ids = classify_span(&sig, 8000.0, &spec, 14_200_000.0);
        assert!(
            ids.iter().any(|i| i.kind == Kind::Ssb || i.kind == Kind::Unknown),
            "SSB-like should not look like CW/PSK, got {ids:?}"
        );
        assert!(
            !ids.iter().any(|i| i.kind == Kind::Cw || i.kind == Kind::Psk31),
            "SSB misread as digital: {ids:?}"
        );
    }

    #[test]
    fn classifies_rtty_tone_pair() {
        let mut sig = Vec::with_capacity(8000);
        let mut phase = 0.0f32;
        for i in 0..8000 {
            let mark = (i / 176) % 2 == 0; // ~45 baud
            let f = 400.0 + if mark { 85.0 } else { -85.0 };
            phase += 2.0 * PI * f / 8000.0;
            sig.push(Complex32::from_polar(1.0, phase));
        }
        let spec = spec_peak(256, 8000.0, 400.0, 3);
        let ids = classify_span(&sig, 8000.0, &spec, 14_080_000.0);
        assert!(
            ids.iter().any(|i| i.kind == Kind::Rtty),
            "expected RTTY, got {ids:?}"
        );
    }

    #[test]
    fn summary_counts_kinds() {
        let ids = vec![
            Ident {
                offset_hz: 0.0,
                bw_hz: 80.0,
                snr_db: 12.0,
                kind: Kind::Cw,
                score: 0.8,
            },
            Ident {
                offset_hz: 200.0,
                bw_hz: 80.0,
                snr_db: 10.0,
                kind: Kind::Cw,
                score: 0.7,
            },
            Ident {
                offset_hz: 1500.0,
                bw_hz: 2400.0,
                snr_db: 15.0,
                kind: Kind::Ssb,
                score: 0.6,
            },
        ];
        assert_eq!(summary(&ids), "2 CW  1 SSB");
    }
}
