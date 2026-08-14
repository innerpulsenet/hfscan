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
use crate::dsp::mix_decim;
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

    /// Modes that do not exist as amateur traffic outside the HF allocations.
    fn amateur_only(self) -> bool {
        matches!(
            self,
            Kind::Cw | Kind::Psk31 | Kind::Rtty | Kind::Ft8 | Kind::Ft4
        )
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
    let decim = (fs / AUDIO as f64).round().max(1.0) as usize;
    // The achieved audio rate; only equals AUDIO when fs is a multiple.
    let fs_a = (fs / decim as f64) as f32;
    for o in occ {
        let audio = mix_decim(iq, fs as f32, o.offset_hz, decim);
        // The PSD needs a whole FFT frame; on less it would come back flat
        // and every feature computed from it would be garbage.
        if audio.len() < PSD_N {
            continue;
        }
        let abs = abs_center + o.offset_hz as f64;
        let (kind, score) = classify_one(&audio, fs_a, o.bw_hz, abs, &mut fft);
        // Nyquist-edge junk, mixed to audio, often looks like an FT8
        // forest. Ham modes do not live outside the allocations.
        if kind.amateur_only() && !bands::in_amateur(abs) {
            continue;
        }
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
    cluster_ft(out)
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
    // The fftshifted periodogram puts Nyquist at the ends. Those bins
    // ring and get classified as a phantom digital signal ~fs/2 below
    // the dial (e.g. 13978 kHz on a 14074 kHz / 192 kHz span).
    let edge = (n / 50).max(2);
    let usable = |i: usize| i >= edge && i + edge < n && i.abs_diff(dc) > 2;
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
            if usable(i) && best.is_none_or(|b| spectrum[i] > spectrum[b]) {
                best = Some(i);
            }
            i += 1;
        }
        let Some(best) = best else {
            continue;
        };
        let bw = ((i - start) as f32 * bin).max(bin);
        // A run tens of kHz wide is the noise floor, not a mode.
        // (SSB / AM / an FT8 pile-up all sit under ~3–10 kHz.)
        if bw > 12_000.0 {
            continue;
        }
        let off = (best as f32 - n as f32 / 2.0) * bin;
        out.push(Occ {
            offset_hz: off,
            snr_db: spectrum[best] - med,
            bw_hz: bw,
        });
    }
    out
}

fn ft_kind(abs_hz: f64) -> Option<Kind> {
    match bands::ft_mode(abs_hz) {
        Some("FT8") => Some(Kind::Ft8),
        Some("FT4") => Some(Kind::Ft4),
        _ => None,
    }
}

fn classify_one(
    audio: &[Complex32],
    fs_a: f32,
    coarse_bw: f32,
    abs_hz: f64,
    psd: &mut Psd,
) -> (Kind, f32) {
    let spec = psd.power(audio);
    let fine = features(&spec, fs_a / spec.len() as f32, audio, fs_a);
    let ft = ft_kind(abs_hz);

    // One-off modes only outside the FT USB windows — a single FT8 tone
    // looks like a carrier / CW / a 170 Hz neighbour pair (RTTY).
    //
    // Except where the band plan puts a narrowband sub-band inside an FT
    // window, which it does in five places: FT4 shares a dial frequency with
    // 30 m PSK31 and with 20 m RTTY, and sits 500 Hz above 40 m RTTY. Vetoing
    // those outright means those sub-bands can never be classified as
    // anything, so 30 m PSK31 and 20 m RTTY were simply invisible. The
    // confirmations are strong enough to be trusted there: a real FT8 message
    // rendered as complex baseband neither confirms as PSK31 nor frames as
    // Baudot (see the tests of exactly that).
    let shared = bands::narrow_mode(abs_hz).is_some();
    if ft.is_none() || shared {
        let psk = psk31::scan_span(audio, fs_a as f64, &[(0.0, 20.0)]);
        if let Some(h) = psk.iter().max_by(|a, b| {
            a.quality
                .partial_cmp(&b.quality)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            if h.quality > 0.7 && coarse_bw < 250.0 && fine.obw_hz < 250.0 {
                return (Kind::Psk31, h.quality);
            }
        }
        let cw = cw::scan_span(audio, fs_a as f64, &[(0.0, 20.0)]);
        if let Some(h) = cw.iter().max_by(|a, b| {
            a.quality
                .partial_cmp(&b.quality)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            if h.quality > 0.45 && coarse_bw < 250.0 && fine.obw_hz < 200.0 {
                return (Kind::Cw, h.quality);
            }
        }
    }

    // RTTY is rare. Demand a clean 170 Hz pair with almost no extra
    // peaks. Two FT8 stations 170 Hz apart fail the peak-count test
    // and, in the FT8 window, never get here as RTTY.
    if let Some(sep) = fine.dual_hz {
        let clean = (160.0..185.0).contains(&sep)
            && fine.env_cv < 0.30
            && fine.n_peaks <= 8
            && fine.obw_hz < 450.0
            && (shared || !matches!(ft, Some(Kind::Ft8)));
        if clean {
            return (Kind::Rtty, 0.78);
        }
    }

    if let Some(k) = ft {
        let wide = coarse_bw.max(fine.obw_hz);
        if fine.speech > 0.55 && (1800.0..3600.0).contains(&wide) && fine.n_peaks < 5 {
            return (Kind::Ssb, 0.55);
        }
        return (k, 0.74);
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
    (Kind::Unknown, 0.2)
}

/// Collapse neighbouring FT8/FT4 blobs into one pile-up so the spectrum
/// and the activity strip show a single group, not fifty carriers.
pub fn cluster_ft(idents: Vec<Ident>) -> Vec<Ident> {
    let (mut ft, mut rest): (Vec<Ident>, Vec<Ident>) = idents
        .into_iter()
        .partition(|i| matches!(i.kind, Kind::Ft8 | Kind::Ft4));
    ft.sort_by(|a, b| {
        a.kind
            .label()
            .cmp(b.kind.label())
            .then(
                a.offset_hz
                    .partial_cmp(&b.offset_hz)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });
    let mut i = 0;
    while i < ft.len() {
        let mut lo = ft[i].offset_hz - ft[i].bw_hz * 0.5;
        let mut hi = ft[i].offset_hz + ft[i].bw_hz * 0.5;
        let kind = ft[i].kind;
        let mut snr = ft[i].snr_db;
        let mut score = ft[i].score;
        let mut j = i + 1;
        while j < ft.len() && ft[j].kind == kind {
            let flo = ft[j].offset_hz - ft[j].bw_hz * 0.5;
            let fhi = ft[j].offset_hz + ft[j].bw_hz * 0.5;
            if flo - hi > 800.0 {
                break;
            }
            lo = lo.min(flo);
            hi = hi.max(fhi);
            snr = snr.max(ft[j].snr_db);
            score = score.max(ft[j].score);
            j += 1;
        }
        rest.push(Ident {
            offset_hz: (lo + hi) * 0.5,
            bw_hz: (hi - lo).max(80.0),
            snr_db: snr,
            kind,
            score,
        });
        i = j;
    }
    rest.sort_by(|a, b| {
        a.offset_hz
            .partial_cmp(&b.offset_hz)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rest
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

struct Feat {
    obw_hz: f32,
    tonality: f32,
    n_peaks: usize,
    dual_hz: Option<f32>,
    carrier: bool,
    env_cv: f32,
    speech: f32,
}

fn features(psd: &[f32], bin_hz: f32, audio: &[Complex32], fs_a: f32) -> Feat {
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
    peaks.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let dual = rtty_sep(&peaks);

    let dc = psd[0];
    let neigh = psd.get(1).copied().unwrap_or(dc);
    let carrier = dc > neigh * 3.0 && dc > med * 12.0;

    let (env_cv, speech) = envelope_stats(audio, fs_a);
    Feat {
        obw_hz: obw.min(fs_a / 2.0),
        tonality: peak / med,
        n_peaks: peaks.len(),
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
            if (160.0..185.0).contains(&sep) && peaks[j].0 > top * 0.25 {
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

fn envelope_stats(audio: &[Complex32], fs_a: f32) -> (f32, f32) {
    if audio.is_empty() {
        return (0.0, 0.0);
    }
    let decim = (fs_a / 200.0).round().max(1.0) as usize; // audio -> ~200 Hz
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

    // Syllabic AM lives around 2–8 Hz. Goertzel bank on the recent envelope,
    // with the mean removed first — DC leakage through the rectangular
    // window otherwise floods the low band and everything reads as speech.
    let n = env.len().min(128);
    let slice = &env[env.len() - n..];
    let dc = slice.iter().sum::<f32>() / n as f32;
    let mut low = 0.0f32;
    let mut mid = 0.0f32;
    let fs = fs_a / decim as f32;
    // Cover the whole 12–40 Hz reference band, not just the first 15 bins.
    let k_max = ((40.0 * n as f32 / fs).ceil() as usize).min(n / 2);
    for k in 1..=k_max {
        let f = k as f32 * fs / n as f32;
        if f >= 40.0 {
            break;
        }
        let mut acc = Complex32::new(0.0, 0.0);
        let w = -2.0 * PI * k as f32 / n as f32;
        for (i, &e) in slice.iter().enumerate() {
            let (sin, cos) = (w * i as f32).sin_cos();
            acc += Complex32::new(e - dc, 0.0) * Complex32::new(cos, sin);
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
        let ids = classify_span(&sig, 8000.0, &spec, 7_047_000.0);
        assert!(
            ids.iter().any(|i| i.kind == Kind::Rtty),
            "expected RTTY, got {ids:?}"
        );
    }

    #[test]
    fn ignores_nyquist_edge_as_ft8() {
        // Hot bins at the left edge of an fftshifted 192 kHz span are
        // Nyquist, not a signal 96 kHz below the dial.
        let mut spec = vec![-85.0f32; 256];
        spec[0] = -25.0;
        spec[1] = -28.0;
        spec[2] = -32.0;
        spec[3] = -40.0;
        let mut sig = Vec::with_capacity(48_000);
        let mut phase = 0.0f32;
        for _ in 0..48_000 {
            phase += 2.0 * PI * 96_000.0 / 192_000.0;
            sig.push(Complex32::from_polar(0.3, phase));
        }
        let ids = classify_span(&sig, 192_000.0, &spec, 14_074_000.0);
        assert!(
            !ids.iter().any(|i| {
                let f = 14_074_000.0 + i.offset_hz as f64;
                matches!(i.kind, Kind::Ft8 | Kind::Ft4) || f < 14_000_000.0
            }),
            "edge spur must not become out-of-band FT8, got {ids:?}"
        );
    }

    #[test]
    fn ham_modes_stay_inside_amateur_bands() {
        assert!(bands::in_amateur(14_074_000.0));
        assert!(!bands::in_amateur(13_978_700.0));
        assert!(!bands::in_amateur(13_978_700.0 + 200.0));
        assert_eq!(bands::ft_mode(14_075_500.0), Some("FT8"));
        assert_eq!(bands::ft_mode(14_026_000.0), None);
    }

    fn tone(n: usize, fs: f32, hz: f32) -> Vec<Complex32> {
        let mut phase = 0.0f32;
        let step = 2.0 * PI * hz / fs;
        (0..n)
            .map(|_| {
                let s = Complex32::from_polar(1.0, phase);
                phase += step;
                s
            })
            .collect()
    }

    #[test]
    fn ft8_window_tone_is_ft8_not_carrier() {
        let sig = tone(8000, 8000.0, 400.0);
        let spec = spec_peak(256, 8000.0, 400.0, 1);
        let ids = classify_span(&sig, 8000.0, &spec, 14_074_000.0);
        assert!(
            ids.iter().any(|i| i.kind == Kind::Ft8),
            "a tone in the FT8 USB window should be FT8, got {ids:?}"
        );
        assert!(
            !ids.iter().any(|i| i.kind == Kind::Carrier || i.kind == Kind::Rtty),
            "must not call FT8 traffic CAR/RTTY: {ids:?}"
        );
    }

    #[test]
    fn rtty_pair_in_ft8_window_is_not_rtty() {
        let mut sig = Vec::with_capacity(8000);
        let mut phase = 0.0f32;
        for i in 0..8000 {
            let mark = (i / 176) % 2 == 0;
            let f = 400.0 + if mark { 85.0 } else { -85.0 };
            phase += 2.0 * PI * f / 8000.0;
            sig.push(Complex32::from_polar(1.0, phase));
        }
        let spec = spec_peak(256, 8000.0, 400.0, 3);
        let ids = classify_span(&sig, 8000.0, &spec, 14_074_000.0);
        assert!(
            !ids.iter().any(|i| i.kind == Kind::Rtty),
            "170 Hz pair inside the FT8 window is not RTTY, got {ids:?}"
        );
    }

    #[test]
    fn clusters_nearby_ft8() {
        let ids = cluster_ft(vec![
            Ident {
                offset_hz: 200.0,
                bw_hz: 80.0,
                snr_db: 10.0,
                kind: Kind::Ft8,
                score: 0.7,
            },
            Ident {
                offset_hz: 400.0,
                bw_hz: 80.0,
                snr_db: 12.0,
                kind: Kind::Ft8,
                score: 0.8,
            },
            Ident {
                offset_hz: 5000.0,
                bw_hz: 60.0,
                snr_db: 8.0,
                kind: Kind::Cw,
                score: 0.6,
            },
        ]);
        let ft: Vec<_> = ids.iter().filter(|i| i.kind == Kind::Ft8).collect();
        assert_eq!(ft.len(), 1, "two close FT8 blobs should be one pile-up: {ids:?}");
        assert!(ft[0].bw_hz > 200.0, "cluster should span both, bw={}", ft[0].bw_hz);
        assert!(ids.iter().any(|i| i.kind == Kind::Cw));
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
