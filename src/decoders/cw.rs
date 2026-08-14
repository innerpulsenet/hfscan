//! Morse (CW) decoder with adaptive speed tracking and nearby-tone lock.
//!
//! Envelope detection with hysteresis slices the keying. Dit length is
//! estimated from a short/long cluster of recent marks so a station that
//! speeds up or slows down is followed instead of being decoded as
//! garbage. A passband scout (FFT + keyed-envelope score) finds CW tones
//! near the cursor and mixes the best one to DC; `n` hops to the next.

use super::{CwView, Decoder};
use crate::dsp::{mix_decim, OnePole};
use num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::collections::VecDeque;
use std::f32::consts::PI;
use std::sync::Arc;

const ENV_DECIM: usize = 8; // audio rate -> ~1 kHz envelope rate
const SEARCH_HZ: f32 = 180.0;
const BANDWIDTH: f32 = 400.0;
const FFT_SIZE: usize = 4096;
const SCAN_AUDIO: f32 = 8000.0;
const MARK_HIST: usize = 24;
/// Envelope history for the scope pane (~0.7 s at 1 kHz).
const ENV_HIST: usize = 700;
/// 5–70 WPM. Below that is not Morse; above it the 4 ms envelope lags.
const DIT_MIN_S: f32 = 0.017; // ~70 WPM
const DIT_MAX_S: f32 = 0.24; // ~5 WPM
/// Marks held before the clock is trusted and the copy starts flowing.
/// Enough to cluster a dit and a dah from — about two characters — without
/// making a station that sends only "K" wait forever.
const WARMUP_MARKS: usize = 8;
/// Envelope samples discarded at start-up and after a re-lock, while the
/// filter chain fills. At the 1 kHz envelope rate this is 60 ms — an order
/// of magnitude past the settling time of everything in front of it, and
/// far shorter than the gap before a station starts sending.
const SETTLE_MS: u32 = 60;

/// Post-mix low-pass corner, and how many poles of it.
///
/// The tuning chain hands the decoder a 400 Hz passband, which on a busy CW
/// band holds two or three stations. The envelope detector sums whatever is
/// in front of it, so a neighbour keys the slicer with *its* timing and the
/// copy becomes noise — the single worst thing that can happen to a CW
/// decoder, and invisible on a clean synthetic signal.
///
/// Mixing the wanted tone to DC puts each neighbour at its own offset, where
/// a narrow low-pass can reject it. 150 Hz over four poles is 2.6 dB down at
/// 60 Hz — past the keying sidebands of even a 50 WPM fist — while a station
/// 200 Hz off is 18 dB down and one at 300 Hz is 28 dB down.
const POST_MIX_HZ: f32 = 150.0;
const POST_MIX_POLES: usize = 4;

/// Cascaded complex one-poles: the post-mix channel filter.
struct NarrowLpf {
    z: [Complex32; POST_MIX_POLES],
    a: f32,
}

impl NarrowLpf {
    fn new(fs: f32) -> Self {
        Self {
            z: [Complex32::new(0.0, 0.0); POST_MIX_POLES],
            a: 1.0 - (-2.0 * PI * POST_MIX_HZ / fs).exp(),
        }
    }

    fn process(&mut self, x: Complex32) -> Complex32 {
        let mut v = x;
        for z in self.z.iter_mut() {
            *z += (v - *z) * self.a;
            v = *z;
        }
        v
    }

    fn reset(&mut self) {
        self.z = [Complex32::new(0.0, 0.0); POST_MIX_POLES];
    }
}

/// Read a dit and a dah length out of a run of mark lengths, or decide the
/// run is not Morse at all.
///
/// Keyed Morse puts its marks in two tight clusters about three units apart.
/// Noise sliced by a threshold sitting inside it produces marks with an
/// exponential-looking spread and no such structure, so requiring the
/// structure is what separates a real fist from an empty frequency — the
/// difference between copy and the letters that noise spells.
///
/// Returns `(dit, dah)` in envelope samples.
fn morse_clock(marks: &[f32]) -> Option<(f32, f32)> {
    if marks.len() < 4 {
        return None;
    }
    let mut v: Vec<f32> = marks.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Two-mean cluster, seeded at the extremes.
    let (mut c1, mut c2) = (v[0], v[v.len() - 1]);
    if c2 <= c1 * 1.05 {
        // All one length: dits alone (or dahs alone) is legitimate Morse but
        // gives no ratio to work from, so take it as dits.
        return Some((c1.max(1e-6), 3.0 * c1.max(1e-6)));
    }
    for _ in 0..12 {
        let (mut s1, mut n1, mut s2, mut n2) = (0.0, 0.0, 0.0, 0.0);
        for &x in &v {
            if (x - c1).abs() <= (x - c2).abs() {
                s1 += x;
                n1 += 1.0;
            } else {
                s2 += x;
                n2 += 1.0;
            }
        }
        if n1 > 0.0 {
            c1 = s1 / n1;
        }
        if n2 > 0.0 {
            c2 = s2 / n2;
        }
    }
    let (short, long) = if c1 <= c2 { (c1, c2) } else { (c2, c1) };
    if short < 1e-6 {
        return None;
    }
    // A dah is 3 dits; allow a heavy or light fist, but not anything.
    let r = long / short;
    if !(1.9..=4.4).contains(&r) {
        return None;
    }
    // Every mark must sit close to one of the two lengths. This is the test
    // that actually separates Morse from noise: a keyer (or a hand) quantises
    // its marks, and sliced noise does not, however conveniently eight of its
    // marks may happen to fall into two groups.
    let near = v
        .iter()
        .filter(|&&x| {
            let d = (x - short).abs().min((x - long).abs());
            d <= 0.30 * if (x - short).abs() < (x - long).abs() { short } else { long }
        })
        .count();
    if near * 10 < v.len() * 8 {
        return None;
    }
    Some((short, long))
}

/// Whether the gaps between marks are keyed by the same clock as the marks.
///
/// Morse spaces come in three lengths — one dit between elements, three
/// between characters, seven between words — so the short gaps land on the
/// same dit the marks did. Noise has no reason to oblige, which makes this
/// the cheapest strong check available: it tests a relationship between two
/// independent measurements rather than the shape of either one.
fn gaps_match_clock(gaps: &[f32], dit: f32) -> bool {
    if gaps.len() < 3 || dit <= 0.0 {
        return true;
    }
    let mut v: Vec<f32> = gaps.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // The inter-element gap is the commonest, so the lower half of the
    // distribution is the one to compare against a dit.
    let short = v[v.len() / 4];
    (0.45..=2.2).contains(&(short / dit))
}

/// A CW tone found by the span scout or the in-passband searcher.
#[derive(Clone, Debug)]
pub struct CwHit {
    pub offset_hz: f32,
    pub score: f32,
    pub quality: f32,
}

pub struct CwDecoder {
    fs: f32,
    env_rate: f32,
    smooth: OnePole,
    peak: f32,
    floor: f32,
    peak_decay: f32,
    floor_attack: f32,
    floor_decay: f32,
    decim_ctr: usize,
    key_down: bool,
    run: f32,
    /// Samples the raw slicer has disagreed with `key_down`. An edge is
    /// only committed once this outlasts the debounce, so a QSB dropout
    /// cannot split a dah and a static crash cannot invent a dit.
    pending: f32,
    dit: f32,
    /// Tracked dah length; with `dit` it sets the dit/dah boundary so a
    /// heavy or light fist is classified by the operator's own weighting.
    dah: f32,
    marks: VecDeque<f32>,
    /// Fast-adapt elements remaining after idle or a speed-change snap.
    acquire: u32,
    /// Elements held while the clock is still unknown, oldest first.
    warmup: Vec<(f32, bool)>,
    /// Whether `warmup` is collecting rather than the clock being trusted.
    warming: bool,
    symbol: String,
    text: String,
    idle: f32,
    started: bool,
    quality: f32,
    env_hist: VecDeque<f32>,
    key_hist: VecDeque<bool>,
    on_thr: f32,
    off_thr: f32,
    tune_err: f32,
    prev_mixed: Complex32,
    have_mixed: bool,
    /// Whether the peak/floor trackers have been seeded.
    have_env: bool,
    /// Envelope samples still to discard while the filters fill.
    settle: u32,
    /// Searches to skip after a manual nudge so AFC does not fight the user.
    hold_tune: u32,

    mix_hz: f32,
    mix_phase: f32,
    /// Rejects the neighbours the 400 Hz chain filter lets through.
    post: NarrowLpf,
    locked: bool,
    lock_score: f32,
    hits: Vec<CwHit>,
    search_buf: Vec<Complex32>,
    since_search: usize,
    fft: Arc<dyn Fft<f32>>,
    fft_buf: Vec<Complex32>,
    window: Vec<f32>,
}

impl CwDecoder {
    pub fn new(fs: f64) -> Self {
        let fs = fs as f32;
        let env_rate = fs / ENV_DECIM as f32;
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let window: Vec<f32> = (0..FFT_SIZE)
            .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / FFT_SIZE as f32).cos())
            .collect();
        Self {
            fs,
            env_rate,
            // ~3 ms envelope: still passes a 50 WPM dit (~24 ms) with room
            // to spare, but follows QRQ edges better than 4 ms.
            smooth: OnePole::new(0.003 * fs),
            post: NarrowLpf::new(fs),
            peak: 0.0,
            floor: 0.0,
            peak_decay: 1.0 - (-1.0f32 / (0.6 * env_rate)).exp(),
            floor_attack: 1.0 - (-1.0f32 / (0.35 * env_rate)).exp(),
            floor_decay: 1.0 - (-1.0f32 / (0.05 * env_rate)).exp(),
            decim_ctr: 0,
            key_down: false,
            run: 0.0,
            pending: 0.0,
            dit: 0.06 * env_rate, // start at 20 WPM
            dah: 0.18 * env_rate,
            marks: VecDeque::with_capacity(MARK_HIST + 1),
            acquire: 16,
            warmup: Vec::new(),
            warming: true,
            symbol: String::new(),
            text: String::new(),
            idle: 0.0,
            started: false,
            quality: 0.0,
            env_hist: VecDeque::with_capacity(ENV_HIST + 1),
            key_hist: VecDeque::with_capacity(ENV_HIST + 1),
            on_thr: 0.55,
            off_thr: 0.35,
            tune_err: 0.0,
            prev_mixed: Complex32::new(0.0, 0.0),
            have_mixed: false,
            have_env: false,
            settle: SETTLE_MS,
            hold_tune: 0,
            mix_hz: 0.0,
            mix_phase: 0.0,
            locked: false,
            lock_score: 0.0,
            hits: Vec::new(),
            search_buf: Vec::with_capacity(FFT_SIZE * 2),
            since_search: 0,
            fft,
            fft_buf: vec![Complex32::new(0.0, 0.0); FFT_SIZE],
            window,
        }
    }

    pub fn wpm(&self) -> f32 {
        let dit_ms = self.dit / self.env_rate * 1000.0;
        if dit_ms > 1.0 {
            1200.0 / dit_ms
        } else {
            0.0
        }
    }

    fn mix(&mut self, s: Complex32) -> Complex32 {
        if self.mix_hz.abs() < 0.05 {
            return s;
        }
        let (sin, cos) = self.mix_phase.sin_cos();
        self.mix_phase += -2.0 * PI * self.mix_hz / self.fs;
        if self.mix_phase > PI {
            self.mix_phase -= 2.0 * PI;
        } else if self.mix_phase < -PI {
            self.mix_phase += 2.0 * PI;
        }
        s * Complex32::new(cos, sin)
    }

    fn clamp_dit(&mut self) {
        let lo = DIT_MIN_S * self.env_rate;
        let hi = DIT_MAX_S * self.env_rate;
        self.dit = self.dit.clamp(lo, hi);
        self.dah = self.dah.clamp(2.0 * self.dit, 4.4 * self.dit);
    }

    /// How long the slicer must hold a new state before the edge is real.
    /// Scales with the dit so QRQ is not smeared, floored so a single
    /// noisy envelope sample can never key the decoder.
    fn debounce_env(&self) -> f32 {
        (0.25 * self.dit).clamp(0.006 * self.env_rate, 0.030 * self.env_rate)
    }

    fn push_symbol(&mut self) {
        if self.symbol.is_empty() {
            return;
        }
        let c = morse_lookup(&self.symbol).unwrap_or('*');
        self.text.push(c);
        self.symbol.clear();
    }

    fn on_mark_end(&mut self, len: f32) {
        // Nothing in Morse is longer than a dah. A mark several dahs long is
        // a carrier, a tuner-upper, or the channel filter and the threshold
        // tracker settling at switch-on — classifying it prepends a phantom
        // dah to the first real character.
        if len > 6.0 * self.dah || len > 1.2 * self.env_rate {
            self.symbol.clear();
            self.warmup.clear();
            return;
        }
        // While the clock is still being learned the elements are held, not
        // classified: the dit estimate starts at 20 WPM, so a station sending
        // anything else has its first characters decoded against the wrong
        // ruler and comes out as garbage. Held elements are replayed once the
        // cluster below has a speed to classify them with.
        if self.warming {
            self.warmup.push((len, true));
            if self.warmup.iter().filter(|(_, m)| *m).count() >= WARMUP_MARKS {
                self.flush_warmup();
            }
            return;
        }
        self.classify_mark(len);

        // The structure check has to keep running, not just gate entry. Noise
        // that flukes its way through one warm-up would otherwise be decoded
        // for as long as it lasts, and on an empty frequency that is forever.
        // A real fist stays inside `morse_clock`'s tolerance; noise does not,
        // and dropping back to warm-up costs only the next few elements.
        if self.marks.len() >= WARMUP_MARKS {
            let recent: Vec<f32> = self.marks.iter().copied().collect();
            if morse_clock(&recent).is_none() {
                self.symbol.clear();
                self.warmup.clear();
                self.warming = true;
                self.marks.clear();
            }
        }
    }

    /// Set the clock from the held elements, then replay them.
    ///
    /// Clustering the whole warm-up at once beats tracking through it: the
    /// first characters get the same speed estimate as the rest of the
    /// transmission rather than whatever the tracker had converged to by the
    /// time it reached them.
    fn flush_warmup(&mut self) {
        let marks: Vec<f32> = self
            .warmup
            .iter()
            .filter(|(_, m)| *m)
            .map(|(l, _)| *l)
            .collect();
        let gaps: Vec<f32> = self
            .warmup
            .iter()
            .filter(|(_, m)| !*m)
            .map(|(l, _)| *l)
            .collect();
        let clock = morse_clock(&marks).filter(|(dit, _)| gaps_match_clock(&gaps, *dit));
        let Some((dit, dah)) = clock else {
            // These marks are not Morse — on an empty frequency the slicer
            // keys on the band noise, because `floor` chases the envelope
            // minima rather than its mean and the thresholds end up sitting
            // inside the noise. Rather than emit the letters that noise
            // spells, drop what was held and keep waiting for a real fist.
            // The oldest mark goes so a signal starting mid-buffer can still
            // fill the window.
            if let Some(i) = self.warmup.iter().position(|(_, m)| *m) {
                self.warmup.drain(..=i);
            } else {
                self.warmup.clear();
            }
            return;
        };
        self.dit = dit;
        self.dah = dah;
        self.clamp_dit();
        self.marks.clear();
        for &m in marks.iter().take(MARK_HIST) {
            self.marks.push_back(m);
        }

        let mut held: Vec<(f32, bool)> = std::mem::take(&mut self.warmup);
        // A stray mark or two picked up off the band before the station
        // started, separated from the real copy by more than a word space, is
        // not part of it — and prepending the letter that noise spelled to
        // someone's callsign is the most visible way to be wrong.
        if let Some(cut) = held.iter().position(|(l, m)| !*m && *l > 7.0 * dit)
            && held[..cut].iter().filter(|(_, m)| *m).count() <= 2
        {
            held.drain(..=cut);
        }
        self.warming = false;
        self.acquire = 8;
        for (len, is_mark) in held {
            if is_mark {
                self.classify_mark(len);
            } else {
                self.on_space_end(len);
            }
        }
    }

    /// Classify one mark against the current clock and add it to the symbol.
    fn classify_mark(&mut self, len: f32) {
        let dit = self.dit.max(1e-6);
        let ratio = len / dit;
        // Classify against the midpoint of the *tracked* dit and dah, not a
        // fixed 2.0: a fist with light dahs (or stretched dits) moves the
        // boundary with it instead of straddling it.
        let boundary = ((self.dit + self.dah) / (2.0 * dit)).clamp(1.55, 2.6);
        let is_dah = ratio >= boundary;
        self.symbol.push(if is_dah { '-' } else { '.' });

        self.marks.push_back(len);
        if self.marks.len() > MARK_HIST {
            self.marks.pop_front();
        }

        // Fast while acquiring (after idle or a speed snap), then only
        // trust unambiguous dits and dahs so Farnsworth gaps and a
        // mid-element speed change cannot drag the estimate.
        let alpha = if self.acquire > 0 { 0.42 } else { 0.20 };
        if self.acquire > 0 {
            self.acquire -= 1;
        }
        let dah_r = (self.dah / dit).clamp(2.0, 4.0);
        if !is_dah && ratio < 1.65 {
            self.dit = (1.0 - alpha) * self.dit + alpha * len;
        } else if is_dah && ratio < 5.0 {
            let a = alpha * 0.55;
            self.dah = (1.0 - a) * self.dah + a * len;
            if ratio > boundary * 1.15 {
                // Unambiguous dah: it carries the clock too, scaled by the
                // operator's own dah/dit ratio rather than an assumed 3.
                self.dit = (1.0 - a) * self.dit + a * (len / dah_r);
            }
        }
        self.clamp_dit();

        if self.marks.len() >= 6 {
            self.recluster();
        }

        // Confidence: how cleanly this mark sat in a dit or dah bucket.
        let fit = if is_dah {
            1.0 - (len / self.dah.max(1e-6) - 1.0).abs().min(1.0)
        } else {
            1.0 - (ratio - 1.0).abs().min(1.0)
        };
        self.quality = 0.9 * self.quality + 0.1 * fit;
    }

    /// Two-mean cluster of recent mark lengths. If the short cluster is a
    /// coherent dit and the long one is ~3×, snap toward it when the
    /// operator has changed speed.
    fn recluster(&mut self) {
        let mut v: Vec<f32> = self.marks.iter().copied().collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut c1 = v[0];
        let mut c2 = *v.last().unwrap();
        if (c2 - c1).abs() < 1e-6 {
            return;
        }
        for _ in 0..8 {
            let mut s1 = 0.0;
            let mut n1 = 0.0;
            let mut s2 = 0.0;
            let mut n2 = 0.0;
            for &x in &v {
                if (x - c1).abs() <= (x - c2).abs() {
                    s1 += x;
                    n1 += 1.0;
                } else {
                    s2 += x;
                    n2 += 1.0;
                }
            }
            if n1 > 0.0 {
                c1 = s1 / n1;
            }
            if n2 > 0.0 {
                c2 = s2 / n2;
            }
        }
        let (short, long, n_short) = if c1 <= c2 {
            let n = v.iter().filter(|&&x| (x - c1).abs() <= (x - c2).abs()).count();
            (c1, c2, n)
        } else {
            let n = v.iter().filter(|&&x| (x - c2).abs() <= (x - c1).abs()).count();
            (c2, c1, n)
        };
        if n_short < 3 || short < 1e-6 {
            return;
        }
        let r = long / short;
        // Down to 1.9: a light fist's dahs can sit near 2× the dit.
        if !(1.9..=4.4).contains(&r) {
            return;
        }
        let rel = (short - self.dit).abs() / self.dit.max(1e-6);
        if rel > 0.28 {
            // Speed changed: adopt most of the new dit in one go.
            self.dit = 0.40 * self.dit + 0.60 * short;
            self.dah = 0.40 * self.dah + 0.60 * long;
            self.acquire = 10;
        } else {
            self.dit = 0.82 * self.dit + 0.18 * short;
            self.dah = 0.82 * self.dah + 0.18 * long;
        }
        self.clamp_dit();
    }

    fn on_space_end(&mut self, len: f32) {
        // Held with the marks, so the gaps are read against the same clock
        // the marks around them end up classified with.
        if self.warming {
            if !self.warmup.is_empty() {
                self.warmup.push((len, false));
            }
            return;
        }
        // Spaces never update dit — Farnsworth sending would otherwise
        // drag the estimate out to the character gap.
        if len >= 2.0 * self.dit {
            self.push_symbol();
            if len >= 5.0 * self.dit && !self.text.ends_with(' ') && !self.text.is_empty() {
                self.text.push(' ');
            }
        }
    }

    fn search(&mut self) {
        if self.search_buf.len() < FFT_SIZE {
            return;
        }
        let start = self.search_buf.len() - FFT_SIZE;
        let slice = &self.search_buf[start..];
        for i in 0..FFT_SIZE {
            self.fft_buf[i] = slice[i] * self.window[i];
        }
        self.fft.process(&mut self.fft_buf);

        let n = FFT_SIZE;
        let bin_hz = self.fs / n as f32;
        let max_bin = ((SEARCH_HZ / bin_hz).ceil() as usize).min(n / 2 - 2).max(2);
        let mag = |c: Complex32| c.norm();
        let mut floor = Vec::with_capacity(max_bin * 2);
        for k in 1..=max_bin {
            floor.push(mag(self.fft_buf[k]));
            floor.push(mag(self.fft_buf[n - k]));
        }
        floor.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = floor[floor.len() / 2].max(1e-20);

        let mut peaks: Vec<(usize, f32)> = Vec::new();
        for k in 1..=max_bin {
            let v = mag(self.fft_buf[k]);
            if v >= mag(self.fft_buf[k - 1])
                && v >= mag(self.fft_buf[k + 1])
                && v / med >= 4.0
            {
                peaks.push((k, v / med));
            }
            let kn = n - k;
            let vn = mag(self.fft_buf[kn]);
            if vn >= mag(self.fft_buf[(kn + 1) % n])
                && vn >= mag(self.fft_buf[kn - 1])
                && vn / med >= 4.0
            {
                peaks.push((kn, vn / med));
            }
        }
        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        peaks.truncate(8);

        let mut fresh = Vec::new();
        for (k, ratio) in peaks {
            let hz = if k < n / 2 {
                k as f32 * bin_hz
            } else {
                (k as f32 - n as f32) * bin_hz
            };
            let hz = hz.clamp(-SEARCH_HZ, SEARCH_HZ);
            if let Some((q, _)) = score_cw(slice, self.fs, hz) {
                if fresh.iter().any(|h: &CwHit| (h.offset_hz - hz).abs() < 40.0) {
                    continue;
                }
                fresh.push(CwHit {
                    offset_hz: hz,
                    score: ratio,
                    quality: q,
                });
            }
        }
        fresh.sort_by(|a, b| b.quality.partial_cmp(&a.quality).unwrap_or(std::cmp::Ordering::Equal));

        if fresh.is_empty() {
            return;
        }
        for n in fresh {
            match self
                .hits
                .iter_mut()
                .find(|h| (h.offset_hz - n.offset_hz).abs() < 40.0)
            {
                Some(old) => *old = n,
                None => self.hits.push(n),
            }
        }
        self.hits
            .sort_by(|a, b| b.quality.partial_cmp(&a.quality).unwrap_or(std::cmp::Ordering::Equal));
        self.hits.truncate(8);

        if self.hold_tune > 0 {
            self.hold_tune -= 1;
            return;
        }
        if self.locked {
            if let Some(h) = self
                .hits
                .iter()
                .find(|h| (h.offset_hz - self.mix_hz).abs() < 35.0)
                .cloned()
            {
                self.mix_hz = h.offset_hz;
                self.lock_score = h.score;
                self.locked = true;
                return;
            }
        }
        if let Some(h) = self.hits.first().cloned() {
            if (h.offset_hz - self.mix_hz).abs() > 8.0 {
                self.mix_phase = 0.0;
                self.post.reset();
            }
            self.mix_hz = h.offset_hz;
            self.lock_score = h.score;
            self.locked = true;
        }
    }

    fn clear_lock_state(&mut self) {
        self.mix_hz = 0.0;
        self.mix_phase = 0.0;
        self.post.reset();
        self.locked = false;
        self.lock_score = 0.0;
        self.hits.clear();
        self.search_buf.clear();
        self.since_search = 0;
        self.symbol.clear();
        self.warmup.clear();
        self.warming = true;
        self.key_down = false;
        self.run = 0.0;
        self.pending = 0.0;
        self.started = false;
        self.acquire = 16;
        self.idle = 0.0;
        self.peak = 0.0;
        self.floor = 0.0;
        self.have_env = false;
        self.settle = SETTLE_MS;
        self.quality = 0.0;
        self.env_hist.clear();
        self.key_hist.clear();
        self.tune_err = 0.0;
        self.have_mixed = false;
        self.hold_tune = 0;
        // Keep dit — the next station may be a similar speed.
    }
}

impl Decoder for CwDecoder {
    fn name(&self) -> &'static str {
        "CW"
    }

    fn bandwidth(&self) -> f32 {
        BANDWIDTH
    }

    fn lock_hz(&self) -> f32 {
        self.mix_hz
    }

    fn locked(&self) -> bool {
        self.locked
    }

    fn squelched(&self) -> bool {
        // The passband scout has to keep hearing, or it cannot find a
        // nearby tone when the cursor is sitting on noise.
        false
    }

    fn hop(&mut self) {
        self.clear_lock_state();
    }

    fn next_lock(&mut self, forward: bool) -> Option<f32> {
        if self.hits.len() < 2 {
            return None;
        }
        let mut hs = self.hits.clone();
        hs.sort_by(|a, b| {
            a.offset_hz
                .partial_cmp(&b.offset_hz)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let cur = self.mix_hz;
        let pick = if forward {
            hs.iter()
                .find(|h| h.offset_hz > cur + 35.0)
                .or_else(|| hs.first())
        } else {
            hs.iter()
                .rev()
                .find(|h| h.offset_hz < cur - 35.0)
                .or_else(|| hs.last())
        };
        pick.cloned().map(|h| {
            self.mix_hz = h.offset_hz;
            self.lock_score = h.score;
            self.locked = true;
            self.mix_phase = 0.0;
            self.post.reset();
            self.symbol.clear();
            self.acquire = 12;
            h.offset_hz
        })
    }

    fn candidate_hz(&self) -> Vec<f32> {
        self.hits.iter().map(|h| h.offset_hz).collect()
    }

    fn nudge_lock(&mut self, delta_hz: f32) -> Option<f32> {
        self.mix_hz = (self.mix_hz + delta_hz).clamp(-SEARCH_HZ, SEARCH_HZ);
        self.locked = true;
        self.hold_tune = 6;
        self.tune_err = 0.0;
        Some(self.mix_hz)
    }

    fn cw_view(&self) -> Option<CwView> {
        let span = (self.peak - self.floor).max(1e-9);
        Some(CwView {
            env: self.env_hist.iter().copied().collect(),
            keyed: self.key_hist.iter().copied().collect(),
            on_thr: ((self.on_thr - self.floor) / span).clamp(0.0, 1.0),
            off_thr: ((self.off_thr - self.floor) / span).clamp(0.0, 1.0),
            lock_hz: self.mix_hz,
            tune_err_hz: self.tune_err,
            wpm: self.wpm(),
            quality: self.quality,
            key_down: self.key_down,
            symbol: self.symbol.clone(),
            dit_ms: self.dit / self.env_rate * 1000.0,
            locked: self.locked,
            hits: self.hits.clone(),
        })
    }

    fn process(&mut self, samples: &[Complex32]) -> String {
        self.search_buf.extend_from_slice(samples);
        if self.search_buf.len() > FFT_SIZE * 2 {
            let excess = self.search_buf.len() - FFT_SIZE;
            self.search_buf.drain(..excess);
        }
        self.since_search += samples.len();
        let interval = if self.locked {
            FFT_SIZE * 4
        } else {
            FFT_SIZE
        };
        if self.since_search >= interval && self.search_buf.len() >= FFT_SIZE {
            self.since_search = 0;
            self.search();
        }

        for &raw in samples {
            let mixed = self.mix(raw);
            // Everything downstream — the discriminator as much as the
            // envelope — sees only the wanted channel.
            let s = self.post.process(mixed);
            if self.have_mixed && s.norm() > 1e-9 && self.prev_mixed.norm() > 1e-9 {
                let d = s * self.prev_mixed.conj();
                let inst = d.arg() * self.fs / (2.0 * PI);
                // Only trust the discriminator while the key is down.
                if self.key_down {
                    self.tune_err = 0.92 * self.tune_err + 0.08 * inst.clamp(-80.0, 80.0);
                }
            }
            self.prev_mixed = s;
            self.have_mixed = true;

            let env = self.smooth.process(s.norm());
            self.decim_ctr += 1;
            if self.decim_ctr < ENV_DECIM {
                continue;
            }
            self.decim_ctr = 0;

            // The envelope smoother, the channel filter and the tuning
            // chain in front of them all start at zero, so the first tens of
            // milliseconds are a ramp up to the band noise rather than a
            // measurement of it. Slicing that ramp keys the decoder and
            // prepends a phantom mark to the first real character; seeding
            // the trackers from it is just as wrong, because everything
            // sampled during the ramp reads far below the true floor.
            //
            // So: drop the ramp, then seed both trackers from real noise.
            // With `floor` starting at the noise instead of at zero, the
            // snr_ok guard below can finally tell a signal from the band.
            if self.settle > 0 {
                self.settle -= 1;
                continue;
            }
            if !self.have_env {
                self.peak = env;
                self.floor = env;
                self.have_env = true;
            }

            if env > self.peak {
                self.peak = env;
            } else {
                self.peak += (env - self.peak) * self.peak_decay;
            }
            // The floor is the band, so it is measured from the band: only
            // while the key is up, and as an average rather than a chase of
            // the minima. Tracking minima put it well below the noise, which
            // left the thresholds — struck as fractions of peak-minus-floor
            // — sitting inside the noise, so an empty frequency keyed the
            // slicer and spelled letters. A fast path down remains, or a
            // signal fading out would strand the floor at its level.
            if !self.key_down {
                let a = if env < self.floor {
                    self.floor_decay
                } else {
                    self.floor_attack
                };
                self.floor += (env - self.floor) * a;
            }

            let span = (self.peak - self.floor).max(1e-9);
            let on_thr = self.floor + 0.52 * span;
            let off_thr = self.floor + 0.32 * span;
            // Noise is not flat: the envelope of a Rayleigh-distributed band
            // peaks 2.5-3x its own mean over the peak tracker's window, so a
            // 2.2x gate let an empty frequency key the slicer and spell a
            // phantom letter before every transmission. 2.8x clears that
            // while the post-mix filter's narrower noise bandwidth keeps a
            // real signal well above it — measured flat to 10 dB and 90%+ at
            // 6 dB, where 3.2x and above start costing weak copy.
            let snr_ok = self.peak > 2.8 * self.floor.max(1e-9);

            let next = if self.key_down {
                env > off_thr
            } else {
                env > on_thr && snr_ok
            };
            self.on_thr = on_thr;
            self.off_thr = off_thr;
            let norm = ((env - self.floor) / span).clamp(0.0, 1.0);
            self.env_hist.push_back(norm);
            self.key_hist.push_back(next);
            while self.env_hist.len() > ENV_HIST {
                self.env_hist.pop_front();
                self.key_hist.pop_front();
            }

            if next == self.key_down {
                // Any sub-debounce flicker is folded back into the element
                // it interrupted, so its length is not lost.
                self.run += 1.0 + self.pending;
                self.pending = 0.0;
            } else {
                // Tentative edge: only commit it once the new state has
                // outlived the debounce.
                self.pending += 1.0;
                if self.pending >= self.debounce_env() {
                    let len = self.run;
                    self.run = self.pending;
                    self.pending = 0.0;
                    if self.key_down {
                        if len > 0.010 * self.env_rate {
                            self.on_mark_end(len);
                        }
                    } else if self.started {
                        self.on_space_end(len);
                    }
                    self.started = true;
                    self.idle = 0.0;
                    self.key_down = next;
                }
            }

            if !self.key_down {
                self.idle += 1.0;
                if self.idle > 8.0 * self.dit && !self.symbol.is_empty() {
                    self.push_symbol();
                }
                // A pause this long means the next thing heard is very
                // likely a different station: hold their first elements and
                // re-learn the clock rather than reading them against the
                // last operator's speed.
                if self.idle > 1.6 * self.env_rate {
                    self.acquire = self.acquire.max(12);
                    if !self.warming {
                        self.push_symbol();
                        self.warming = true;
                    }
                }
                // Whatever is still held when the band goes quiet has to be
                // released, or a short transmission is never shown at all.
                if self.warming && !self.warmup.is_empty() && self.idle > 3.0 * self.env_rate {
                    self.flush_warmup();
                    self.warming = true;
                }
            }
        }
        std::mem::take(&mut self.text)
    }

    fn status(&self) -> String {
        let wpm = self.wpm();
        let extra = if self.hits.len() > 1 {
            format!(" +{}", self.hits.len() - 1)
        } else {
            String::new()
        };
        if self.locked && self.mix_hz.abs() > 1.0 {
            format!("{wpm:.0} WPM lock {:+.0}Hz{extra}", self.mix_hz)
        } else {
            format!("{wpm:.0} WPM{extra}")
        }
    }

    fn reset(&mut self) {
        self.text.clear();
        self.marks.clear();
        self.dit = 0.06 * self.env_rate;
        self.dah = 0.18 * self.env_rate;
        self.clear_lock_state();
    }
}

/// Confirm which of `peaks` (offsets from IQ DC, Hz) are keyed CW.
pub fn scan_span(iq: &[Complex32], fs: f64, peaks: &[(f64, f32)]) -> Vec<CwHit> {
    let decim = (fs / SCAN_AUDIO as f64).round().max(1.0) as usize;
    let need = (SCAN_AUDIO * 0.35) as usize;
    if iq.len() / decim < need {
        return Vec::new();
    }
    let mut out = Vec::new();
    for &(off, _) in peaks {
        let audio = mix_decim(iq, fs as f32, off as f32, decim);
        if audio.len() < need {
            continue;
        }
        if let Some((q, _snr)) = score_cw(&audio, SCAN_AUDIO, 0.0) {
            if out
                .iter()
                .any(|h: &CwHit| (h.offset_hz - off as f32).abs() < 40.0)
            {
                continue;
            }
            out.push(CwHit {
                offset_hz: off as f32,
                score: q,
                quality: q,
            });
        }
    }
    out.sort_by(|a, b| {
        a.offset_hz
            .partial_cmp(&b.offset_hz)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

/// Mix `buf` by `hz` and decide whether the envelope is keyed Morse rather
/// than a dead carrier, noise, or PSK31. Returns (quality, snr).
fn score_cw(buf: &[Complex32], fs: f32, hz: f32) -> Option<(f32, f32)> {
    let step = -2.0 * PI * hz / fs;
    let mut phase = 0.0f32;
    let decim = (fs / 1000.0).round().max(1.0) as usize;
    let mut env = Vec::with_capacity(buf.len() / decim + 1);
    let mut acc = 0.0f32;
    let mut n = 0usize;
    // ~180 Hz LPF after the mix so a neighbour (carrier, another CW)
    // does not sit on the envelope and hide the keying.
    let lpf_a = 1.0 - (-2.0 * PI * 180.0 / fs).exp();
    let mut lpf = Complex32::new(0.0, 0.0);
    for &s in buf {
        let (sin, cos) = phase.sin_cos();
        phase += step;
        let mixed = s * Complex32::new(cos, sin);
        lpf += (mixed - lpf) * lpf_a;
        acc += lpf.norm();
        n += 1;
        if n == decim {
            env.push(acc / decim as f32);
            acc = 0.0;
            n = 0;
        }
    }
    if env.len() < 40 {
        return None;
    }

    let mut sorted = env.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let floor = sorted[sorted.len() / 8].max(1e-12);
    let peak = sorted[sorted.len() * 7 / 8];
    let snr = peak / floor;
    if snr < 2.2 {
        return None;
    }

    let thr = floor + 0.45 * (peak - floor);
    let mut on = 0u32;
    let mut trans = 0u32;
    let mut prev = env[0] > thr;
    let mut run = 1u32;
    let mut marks: Vec<u32> = Vec::new();
    for &e in &env[1..] {
        let now = e > thr;
        if now {
            on += 1;
        }
        if now == prev {
            run += 1;
            continue;
        }
        if prev && run >= 2 {
            marks.push(run);
        }
        trans += 1;
        prev = now;
        run = 1;
    }
    if prev && run >= 2 {
        marks.push(run);
    }
    let duty = on as f32 / env.len() as f32;
    // A tuning tone is stuck on; noise chatters. CW sits in between and
    // has several keying edges in a fraction of a second.
    if !(0.06..=0.88).contains(&duty) || trans < 3 {
        return None;
    }

    let mut quality = ((snr - 2.0) / 6.0).clamp(0.2, 1.0);
    quality *= 0.55 + 0.45 * (1.0 - (duty - 0.4).abs());
    if marks.len() >= 3 {
        let mut ms = marks.clone();
        ms.sort_unstable();
        let short = ms[ms.len() / 4].max(1) as f32;
        let long = ms[ms.len() * 3 / 4] as f32;
        let r = long / short;
        if (2.0..=5.0).contains(&r) {
            quality = (quality + 0.25).min(1.0);
        }
    }
    if quality < 0.35 {
        return None;
    }
    Some((quality, snr))
}

fn morse_lookup(sym: &str) -> Option<char> {
    const TABLE: &[(&str, char)] = &[
        (".-", 'A'), ("-...", 'B'), ("-.-.", 'C'), ("-..", 'D'), (".", 'E'),
        ("..-.", 'F'), ("--.", 'G'), ("....", 'H'), ("..", 'I'), (".---", 'J'),
        ("-.-", 'K'), (".-..", 'L'), ("--", 'M'), ("-.", 'N'), ("---", 'O'),
        (".--.", 'P'), ("--.-", 'Q'), (".-.", 'R'), ("...", 'S'), ("-", 'T'),
        ("..-", 'U'), ("...-", 'V'), (".--", 'W'), ("-..-", 'X'), ("-.--", 'Y'),
        ("--..", 'Z'),
        ("-----", '0'), (".----", '1'), ("..---", '2'), ("...--", '3'),
        ("....-", '4'), (".....", '5'), ("-....", '6'), ("--...", '7'),
        ("---..", '8'), ("----.", '9'),
        (".-.-.-", '.'), ("--..--", ','), ("..--..", '?'), (".----.", '\''),
        ("-.-.--", '!'), ("-..-.", '/'), ("-.--.", '('), ("-.--.-", ')'),
        (".-...", '&'), ("---...", ':'), ("-.-.-.", ';'), ("-...-", '='),
        (".-.-.", '+'), ("-....-", '-'), ("..--.-", '_'), (".-..-.", '"'),
        ("...-..-", '$'), (".--.-.", '@'),
    ];
    TABLE.iter().find(|(s, _)| *s == sym).map(|(_, c)| *c)
}
