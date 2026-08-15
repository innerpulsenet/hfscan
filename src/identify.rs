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
    /// Measured mark/space separation for `Kind::Rtty`. 170 Hz is the common
    /// case but 425 and 850 are both in amateur use, and a decoder given the
    /// wrong one frames noise — so the classifier passes on what it measured
    /// rather than letting the decoder assume.
    pub shift_hz: Option<f32>,
}

impl Ident {
    /// Whether an offset falls inside what this signal already explains.
    ///
    /// An FSK signal reaches beyond its own shift: each tone keys on and off,
    /// and the sidebands and squaring cross-terms that produces are what the
    /// narrowband probes latch onto a couple of hundred hertz out. Anything
    /// found in there belongs to this signal, not beside it.
    pub fn covers(&self, offset_hz: f32) -> bool {
        let margin = if self.kind == Kind::Rtty { 150.0 } else { 0.0 };
        (self.offset_hz - offset_hz).abs() <= self.bw_hz * 0.5 + margin
    }
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
        let (kind, score, fsk) = classify_one(&audio, fs_a, o.bw_hz, abs, &mut fft);
        // Nyquist-edge junk, mixed to audio, often looks like an FT8
        // forest. Ham modes do not live outside the allocations.
        if kind.amateur_only() && !bands::in_amateur(abs) {
            continue;
        }
        if kind == Kind::Unknown && score < 0.35 {
            continue;
        }
        // Against the wider of the two footprints, not just this
        // occupancy's. A wide-shift FSK signal appears in the periodogram as
        // two separate occupancies, and the second one — a lone keyed tone —
        // classifies as CW and spawns its own decoder unless the RTTY ident
        // already covering that frequency is allowed to claim it.
        if out.iter().any(|i: &Ident| {
            (i.offset_hz - o.offset_hz).abs() < i.bw_hz.max(o.bw_hz).max(80.0) * 0.5
        }) {
            continue;
        }
        out.push(Ident {
            // An FSK occupancy peaks on whichever tone is busier — mark, for
            // a station that idles there. Report the midpoint instead, which
            // is where a decoder has to sit.
            offset_hz: o.offset_hz + fsk.map_or(0.0, |(_, mid)| mid),
            // An FSK signal occupies its shift plus a bit of skirt, whatever
            // the occupancy detector made of its two halves.
            bw_hz: fsk.map_or(o.bw_hz, |(sep, _)| (sep + 200.0).max(o.bw_hz)),
            snr_db: o.snr_db,
            kind,
            score,
            shift_hz: fsk.map(|(sep, _)| sep),
        });
    }
    // An FSK signal covers its whole shift, and its two tones are each a
    // keyed carrier in their own right — so whatever else was identified
    // inside that span is one of those tones read on its own. This is
    // resolved here rather than by the overlap check above, which runs
    // strongest-occupancy-first and would just as happily let a mark tone's
    // PSK31 verdict suppress the RTTY ident that explains it. Two tones
    // alternating is the stronger claim whichever arrived first.
    let spans: Vec<Ident> = out.iter().filter(|i| i.kind == Kind::Rtty).cloned().collect();
    out.retain(|i| i.kind == Kind::Rtty || !spans.iter().any(|r| r.covers(i.offset_hz)));
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

/// (kind, confidence, mark/space shift and mid-shift offset when it is FSK).
type Verdict = (Kind, f32, Option<(f32, f32)>);

fn classify_one(
    audio: &[Complex32],
    fs_a: f32,
    coarse_bw: f32,
    abs_hz: f64,
    psd: &mut Psd,
) -> Verdict {
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

    // FSK first, because it is the one narrowband mode that announces itself.
    // A signal alternating between two tones cannot be anything else, whereas
    // "looks like BPSK" and "looks like keyed CW" are both things a single
    // RTTY tone satisfies — and whichever probe runs first wins, so running
    // the weaker evidence first is what produced RTTY labelled PSK31.
    let fsk = fsk_shift(audio, fs_a).filter(|_| fine.obw_hz < 1500.0);
    if let Some((sep, mid, share)) = fsk {
        // The same FT-window rule the peak-pair test below uses: inside an
        // FT8/FT4 window only the band plan's own narrowband sub-bands may
        // claim to be something else.
        if shared || !matches!(ft, Some(Kind::Ft8) | Some(Kind::Ft4)) {
            return (Kind::Rtty, (0.60 + 0.35 * share).min(0.95), Some((sep, mid)));
        }
    }

    if ft.is_none() || shared {
        let psk = psk31::scan_span(audio, fs_a as f64, &[(0.0, 20.0)]);
        if let Some(h) = psk.iter().max_by(|a, b| {
            a.quality
                .partial_cmp(&b.quality)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            if h.quality > 0.7 && coarse_bw < 250.0 && fine.obw_hz < 250.0 {
                return (Kind::Psk31, h.quality, None);
            }
        }
        let cw = cw::scan_span(audio, fs_a as f64, &[(0.0, 20.0)]);
        if let Some(h) = cw.iter().max_by(|a, b| {
            a.quality
                .partial_cmp(&b.quality)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            if h.quality > 0.45 && coarse_bw < 250.0 && fine.obw_hz < 200.0 {
                return (Kind::Cw, h.quality, None);
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
            return (Kind::Rtty, 0.78, None);
        }
    }

    if let Some(k) = ft {
        let wide = coarse_bw.max(fine.obw_hz);
        if fine.speech > 0.55 && (1800.0..3600.0).contains(&wide) && fine.n_peaks < 5 {
            return (Kind::Ssb, 0.55, None);
        }
        return (k, 0.74, None);
    }

    if fine.carrier && fine.obw_hz > 3500.0 {
        return (Kind::Am, 0.72, None);
    }
    if fine.carrier && fine.obw_hz < 200.0 && fine.env_cv < 0.2 {
        return (Kind::Carrier, 0.65, None);
    }

    let wide = coarse_bw.max(fine.obw_hz);
    if (1200.0..3600.0).contains(&wide)
        && !fine.carrier
        && fine.n_peaks < 8
        && fine.speech > 0.35
    {
        return (Kind::Ssb, 0.62 + 0.2 * fine.speech.min(1.0), None);
    }
    if (1200.0..3600.0).contains(&wide) && !fine.carrier && fine.tonality < 8.0 {
        return (Kind::Ssb, 0.5, None);
    }

    if coarse_bw < 150.0 && fine.tonality > 10.0 && fine.env_cv < 0.15 {
        return (Kind::Carrier, 0.5, None);
    }
    (Kind::Unknown, 0.2, None)
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
            shift_hz: None,
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

/// Two tones, one signal — measured from where the signal *is*, moment to
/// moment, rather than from two peaks in an averaged spectrum.
///
/// The peak-pair test this backs up needs both tones to show up in a PSD with
/// the weaker within 6 dB of the stronger, and real RTTY does not oblige: a
/// station idles on mark and returns to mark between characters, so mark can
/// be several dB up on space over any averaging window. When the pair test
/// misses, the occupancy falls through to the probes below it and a mark tone
/// keying on and off is read as BPSK — which is how RTTY ends up labelled
/// PSK31 with a decoder attached to it.
///
/// Instantaneous frequency does not have that problem. FSK spends all its
/// time at exactly two frequencies whatever the duty cycle between them, so
/// the histogram is bimodal even when one mode is three times the other.
/// Returns (shift Hz, mid-shift offset Hz, share of the signal's time the two
/// tones account for). The offset matters as much as the shift: an occupancy
/// peaks on whichever tone is busier, and a decoder handed that instead of
/// the midpoint is half a shift off before it starts.
fn fsk_shift(audio: &[Complex32], fs: f32) -> Option<(f32, f32, f32)> {
    // ~4 ms blocks: short against a 45 baud bit (22 ms), long enough that the
    // coherent sum inside one is a frequency estimate rather than a sample.
    let blk = (fs / 250.0).round().max(2.0) as usize;
    if audio.len() < blk * 32 {
        return None;
    }
    const BIN_HZ: f32 = 20.0;
    const SPAN_HZ: f32 = 1200.0;
    let nbins = (2.0 * SPAN_HZ / BIN_HZ) as usize;
    let mut hist = vec![0.0f32; nbins];
    let mut total = 0.0f32;
    let mut i = 1usize;
    while i + blk <= audio.len() {
        let mut acc = Complex32::new(0.0, 0.0);
        for k in 0..blk {
            acc += audio[i + k] * audio[i + k - 1].conj();
        }
        i += blk;
        // The weight is the coherent sum's own magnitude, so a block of noise
        // — whose phase differences cancel — barely votes, and the gaps
        // between characters cannot invent a third tone.
        let w = acc.norm();
        if w < 1e-12 {
            continue;
        }
        let hz = acc.arg() * fs / (2.0 * PI);
        if hz.abs() >= SPAN_HZ {
            continue;
        }
        let b = ((hz + SPAN_HZ) / BIN_HZ) as usize;
        hist[b.min(nbins - 1)] += w;
        total += w;
    }
    if total <= 0.0 {
        return None;
    }
    let bin_hz = |b: usize| (b as f32 + 0.5) * BIN_HZ - SPAN_HZ;
    let mass = |centre: f32| -> f32 {
        hist.iter()
            .enumerate()
            .filter(|(b, _)| (bin_hz(*b) - centre).abs() <= 40.0)
            .map(|(_, v)| *v)
            .sum()
    };
    let top = (0..nbins).max_by(|&a, &b| {
        hist[a].partial_cmp(&hist[b]).unwrap_or(std::cmp::Ordering::Equal)
    })?;
    let a_hz = bin_hz(top);
    // The second tone has to be a separate mode, not the shoulder of the
    // first, so anything within 80 Hz of the winner is the same tone.
    let second = (0..nbins)
        .filter(|&b| (bin_hz(b) - a_hz).abs() > 80.0)
        .max_by(|&x, &y| hist[x].partial_cmp(&hist[y]).unwrap_or(std::cmp::Ordering::Equal))?;
    let b_hz = bin_hz(second);
    let (ma, mb) = (mass(a_hz), mass(b_hz));
    // Bins are 20 Hz; the centroid inside each mode recovers the real tone
    // frequency, which is what makes the shift worth reporting to a decoder.
    let centroid = |centre: f32| -> f32 {
        let (mut num, mut den) = (0.0f32, 0.0f32);
        for (b, &v) in hist.iter().enumerate() {
            let f = bin_hz(b);
            if (f - centre).abs() <= 40.0 {
                num += f * v;
                den += v;
            }
        }
        if den > 0.0 { num / den } else { centre }
    };
    let (a_c, b_c) = (centroid(a_hz), centroid(b_hz));
    let sep = (a_c - b_c).abs();
    // Shifts in amateur use run 170 (most), 425 and 850. Below 100 Hz the
    // "pair" is one tone's own sidebands; above 1000 it is two stations.
    if !(100.0..=1000.0).contains(&sep) {
        return None;
    }
    let share = (ma + mb) / total;
    let weaker = mb / (ma + mb).max(1e-20);
    // A single keyed tone puts everything in one mode; noise puts it
    // everywhere. Only a signal that genuinely alternates between two
    // frequencies has most of its time in two modes with the lesser one
    // holding a real share of it.
    if share < 0.55 || weaker < 0.12 {
        return None;
    }
    Some((sep, (a_c + b_c) * 0.5, share))
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


    /// Realistic RTTY: idle mark, Baudot characters, 45.45 baud / 170 Hz.
    ///
    /// The existing `classifies_rtty_tone_pair` fixture alternates mark and
    /// space every bit, which is the one pattern real RTTY never sends. Real
    /// traffic idles on mark and spends most of its time there, so the space
    /// tone is far weaker in an averaged PSD — which is exactly the case the
    /// two-peak test has to survive.
    fn real_rtty(fs: f32, secs: f32, snr: f32, centre: f32, idle_only: bool) -> Vec<Complex32> {
        const LTRS: &str = "\0E\nA SIU\rDRJNFCKTZLWHYPQOBG\0MXV\0";
        let text = "CQ CQ DE W1AW W1AW K ";
        let sps = fs / 45.45;
        let mut bits: Vec<bool> = vec![true; 60];
        for c in text.chars().take(if idle_only { 0 } else { usize::MAX }) {
            let Some(code) = LTRS.chars().position(|x| x == c) else { continue };
            bits.push(false);
            for b in 0..5 {
                bits.push(code as u8 & (1 << b) != 0);
            }
            bits.push(true);
            bits.push(true);
        }
        bits.extend(std::iter::repeat_n(true, 60));
        let mut rng = 0x1234_5678u32;
        let mut noise = move || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            (rng as f32 / u32::MAX as f32) - 0.5
        };
        let mut out = Vec::new();
        let mut phase = 0.0f32;
        let want = (fs * secs) as usize;
        let mut i = 0usize;
        while out.len() < want {
            let mark = bits[i % bits.len()];
            let f = centre + if mark { 85.0 } else { -85.0 };
            for _ in 0..sps as usize {
                phase += 2.0 * PI * f / fs;
                out.push(
                    Complex32::from_polar(1.0, phase)
                        + Complex32::new(noise(), noise()) * snr,
                );
            }
            i += 1;
        }
        out
    }


    /// The reported failure: RTTY on the waterfall, labelled PSK31, with a
    /// decoder attached producing nonsense.
    ///
    /// The fixture is what makes this a real test — a station that idles on
    /// mark, so its space tone is far weaker than mark over any averaging
    /// window and the two-peak test cannot see it. That is ordinary RTTY, and
    /// it is exactly the case that used to fall through to the PSK31 probe.
    #[test]
    fn real_rtty_is_not_psk31() {
        let sig = real_rtty(8000.0, 3.0, 0.05, 400.0, false);
        let disp = spec_peak(256, 8000.0, 400.0, 4);
        let ids = classify_span(&sig, 8000.0, &disp, 7_047_000.0);
        assert!(
            !ids.iter().any(|i| i.kind == Kind::Psk31),
            "RTTY classified as PSK31: {ids:?}"
        );
        assert!(
            ids.iter().any(|i| i.kind == Kind::Rtty),
            "RTTY was not identified at all: {ids:?}"
        );
    }

    /// The confirmation the misidentification actually turned on. Whatever the
    /// classifier decides, the PSK31 scout runs across the whole span on its
    /// own and raises its own idents — so it has to reject a mark tone too.
    #[test]
    fn psk31_scout_rejects_an_rtty_tone() {
        let sig = real_rtty(8000.0, 3.0, 0.05, 400.0, false);
        for probe in [400.0f64, 485.0, 315.0] {
            let hits = psk31::scan_span(&sig, 8000.0, &[(probe, 20.0)]);
            assert!(
                hits.is_empty(),
                "PSK31 confirmed at {probe} Hz on an RTTY signal: {:?}",
                hits.iter().map(|h| (h.offset_hz, h.quality)).collect::<Vec<_>>()
            );
        }
    }

    /// What the FSK test must and must not fire on. A keyed carrier, a plain
    /// carrier, PSK31 and noise all have to come back None, or the cure is
    /// worse than the disease.
    #[test]
    fn fsk_detector_only_fires_on_two_tones() {
        let rtty = real_rtty(8000.0, 3.0, 0.05, 0.0, false);
        let (sep, mid, _) = fsk_shift(&rtty, 8000.0).expect("170 Hz RTTY is FSK");
        assert!((sep - 170.0).abs() < 25.0, "shift measured as {sep:.0} Hz");
        assert!(mid.abs() < 40.0, "mid-shift offset {mid:.0} Hz should be near zero");

        // A station idling on mark is a carrier, and saying so is correct.
        assert!(fsk_shift(&real_rtty(8000.0, 3.0, 0.05, 0.0, true), 8000.0).is_none());
        assert!(fsk_shift(&keyed_cw(24000, 8000.0, 20.0, 50.0), 8000.0).is_none());
        assert!(fsk_shift(&tone(24000, 8000.0, 50.0), 8000.0).is_none());
        let psk = crate::decoders::tests::gen_psk31_at("CQ DE W1AW ", 8000.0, 0.0, 0.03, 3.0);
        assert!(fsk_shift(&psk, 8000.0).is_none(), "PSK31 read as FSK");
        let mut rng = 0xBEEF_0001u32;
        let noise: Vec<Complex32> = (0..24000)
            .map(|_| {
                let mut f = || {
                    rng ^= rng << 13;
                    rng ^= rng >> 17;
                    rng ^= rng << 5;
                    (rng as f32 / u32::MAX as f32) - 0.5
                };
                Complex32::new(f(), f())
            })
            .collect();
        assert!(fsk_shift(&noise, 8000.0).is_none(), "noise read as FSK");
    }

    /// 425 and 850 Hz shifts are both in amateur use, and measuring the shift
    /// is what lets the decoder be set to it instead of framing noise at 170.
    #[test]
    fn wide_shifts_are_measured_not_assumed() {
        for want in [425.0f32, 850.0] {
            let sig = crate::decoders::tests::gen_rtty_at(
                "CQ DE W1AW ", 8000.0, 0.0, want, 0.03, 3.0,
            );
            let (sep, _, _) = fsk_shift(&sig, 8000.0)
                .unwrap_or_else(|| panic!("{want:.0} Hz shift RTTY not seen as FSK"));
            assert!(
                (sep - want).abs() < 0.1 * want,
                "{want:.0} Hz shift measured as {sep:.0} Hz"
            );
        }
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
                shift_hz: None,
            },
            Ident {
                offset_hz: 400.0,
                bw_hz: 80.0,
                snr_db: 12.0,
                kind: Kind::Ft8,
                score: 0.8,
                shift_hz: None,
            },
            Ident {
                offset_hz: 5000.0,
                bw_hz: 60.0,
                snr_db: 8.0,
                kind: Kind::Cw,
                score: 0.6,
                shift_hz: None,
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
                shift_hz: None,
            },
            Ident {
                offset_hz: 200.0,
                bw_hz: 80.0,
                snr_db: 10.0,
                kind: Kind::Cw,
                score: 0.7,
                shift_hz: None,
            },
            Ident {
                offset_hz: 1500.0,
                bw_hz: 2400.0,
                snr_db: 15.0,
                kind: Kind::Ssb,
                score: 0.6,
                shift_hz: None,
            },
        ];
        assert_eq!(summary(&ids), "2 CW  1 SSB");
    }
}
