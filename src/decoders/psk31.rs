//! BPSK31 decoder with automatic nearby-signal identify and calibrate.
//!
//! PSK31 is differentially encoded (a phase reversal is a 0 bit, no reversal a
//! 1), so no carrier recovery loop is needed: comparing the phase of adjacent
//! symbols cancels any modest frequency offset. A slow AFC term mops up the
//! residual rotation. Symbol timing comes from the amplitude dip that the
//! raised-cosine pulse shaping puts at every symbol boundary.
//!
//! The decoder does not need the cursor parked on the carrier. Every half
//! second it squares a block of baseband (wiping the BPSK modulation, leaving
//! a tone at twice the offset), finds that tone, and confirms the candidate
//! looks like PSK31 — BPSK symbols clustered on the real axis, and a reversal
//! rate that is neither a dead carrier nor noise. The internal NCO then mixes
//! the lock to DC. Once locked, a raised-cosine matched filter, Gardner-style
//! dump timing, and the AFC keep the demod calibrated.

use super::callscan::{utc_hhmmss, CallScanner};
use super::{Decoder, FtMessage, PskView};
use std::collections::VecDeque;
use crate::dsp::{mix_decim, Rotator};
use num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::collections::HashMap;
use std::f32::consts::PI;
use std::sync::Arc;

const BAUD: f32 = 31.25;
/// Search window around the cursor. A PSK31 signal anywhere in here is
/// identified and mixed onto DC; the user only has to be nearby.
const SEARCH_HZ: f32 = 180.0;
/// Audio bandwidth the tuning chain should deliver (a little wider than the
/// search so the FIR transition band does not eat the edges).
const BANDWIDTH: f32 = 400.0;
const FFT_SIZE: usize = 4096;
/// z² peak / median — a real BPSK carrier stands well above the floor.
const PEAK_RATIO: f32 = 6.0;
/// Differential-symbol concentration that counts as "this is BPSK", on the
/// calibrated scale below (0.76 raw, the value this gate always used).
const BPSK_QUALITY: f32 = 0.34;
/// Idle PSK31 is almost all reversals; a CW carrier is almost none.
const REV_MIN: f32 = 0.22;
/// Idle PSK31 is continuous reversals (1.0); a CW carrier is ~0.
const REV_MAX: f32 = 1.0;
/// Copy below this confidence is noise being read as varicode. The scanner
/// applies its own, higher floor on top; this one only stops the decoder
/// spotting and printing when there is demonstrably nothing there.
const PRINT_QUALITY: f32 = 0.20;
/// Mean |cos| of a uniformly random phase — what the raw differential-symbol
/// concentration reads on pure noise.
///
/// This is the whole reason a bad PSK31 lock still filled the pane with text.
/// The raw measure is `E|cos θ|`, and for noise θ is uniform, so it settles at
/// 2/π = 0.64, not 0: every threshold below that (the old 0.50 print gate, the
/// old 0.35 lock-drop) was one noise could never fail. Subtracting the noise
/// floor and rescaling gives a number that means what it says — 0 is noise,
/// 1 is a perfectly resolved constellation — and makes a threshold like "only
/// show me copy above 40%" a statement about the signal rather than a
/// statement about arithmetic.
const NOISE_Q: f32 = 2.0 / PI;
/// Symbols the confidence average settles over — about two seconds of copy,
/// and the point at which its own sampling error stops being worth
/// discounting. Matches the 0.03 smoothing it runs at once warmed up.
const Q_WINDOW: usize = 64;

/// Put a raw differential-symbol concentration on the 0 = noise, 1 = clean
/// scale. Only ever applied to an *averaged* raw value: clamping individual
/// symbols would throw away the below-average half and leave noise reading
/// ~0.37 instead of 0.
fn calibrate(raw: f32) -> f32 {
    ((raw - NOISE_Q) / (1.0 - NOISE_Q)).clamp(0.0, 1.0)
}

/// Rough SNR in a 2500 Hz reference bandwidth for a calibrated confidence.
///
/// Differential detection sees a phase error of variance 1/γ for a symbol SNR
/// of γ, so `E|cos| = exp(-1/2γ)` and the confidence inverts to a γ. PSK31
/// occupies 31.25 Hz, so the reference-bandwidth figure is 10log10(γ) - 19 dB.
fn snr_from_quality(q: f32) -> f32 {
    let raw = NOISE_Q + q.clamp(0.0, 1.0) * (1.0 - NOISE_Q);
    let gamma = -1.0 / (2.0 * raw.clamp(1e-3, 0.999).ln());
    (10.0 * gamma.log10() - 19.0).clamp(-24.0, 20.0)
}
const SYM_HIST: usize = 80;
const ENV_HIST: usize = 400;
/// Audio rate `scan_span` decimates radio IQ down to.
const SCAN_AUDIO: f32 = 8000.0;

/// A PSK31 signal found by the span scout or the in-passband searcher.
#[derive(Clone, Debug)]
pub struct PskHit {
    /// Offset from the IQ DC (radio centre for a span scan, cursor for
    /// the decoder's own search), Hz.
    pub offset_hz: f32,
    /// z² peak / median, or BPSK quality when the scout scored a peak.
    pub score: f32,
    pub quality: f32,
}

pub struct Psk31Decoder {
    fs: f32,
    sps: usize,
    idx: usize,
    energy: Vec<f32>,
    since: usize,
    symbol_len: usize,
    acc: Complex32,
    prev: Complex32,
    /// Residual rotation after the search NCO, radians per symbol.
    afc: f32,
    /// Identified carrier offset, Hz, relative to the cursor.
    mix_hz: f32,
    mix_phase: f32,
    locked: bool,
    lock_score: f32,
    code: String,
    pending_zero: bool,
    text: String,
    symbols: u32,
    /// Uncalibrated differential-symbol concentration, averaged. Read it
    /// through `conf()`; on its own it never drops below `NOISE_Q`.
    q_raw: f32,
    /// Symbols folded into `q_raw` since the lock changed, capped. The
    /// average is bias-corrected over these, so a fresh lock reports what it
    /// is actually seeing within a few symbols instead of climbing out of
    /// zero for a second — a second in which the copy floor would be
    /// swallowing the start of every transmission.
    q_n: f32,
    reversals: f32,
    have_prev: bool,
    /// Low-pass after the search mix so the demod sees ~PSK31 bandwidth.
    lpf: Complex32,
    lpf_a: f32,

    search_buf: Vec<Complex32>,
    since_search: usize,
    fft: Arc<dyn Fft<f32>>,
    fft_buf: Vec<Complex32>,
    window: Vec<f32>,

    /// Word scanner for pskreporter spots (`DE CALL CALL`, `CQ CALL`).
    scan: CallScanner,
    /// PSK31 signals identified in the current passband, strongest first.
    hits: Vec<PskHit>,
    varicode: HashMap<&'static str, char>,
    sym_hist: VecDeque<Complex32>,
    env_hist: VecDeque<f32>,
    env_decim: usize,
    hold_tune: u32,

    #[cfg(test)]
    pub(crate) captured_bits: Vec<bool>,
}

impl Psk31Decoder {
    pub fn new(fs: f64) -> Self {
        let fs = fs as f32;
        let sps = (fs / BAUD).round().max(4.0) as usize;
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let window: Vec<f32> = (0..FFT_SIZE)
            .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / FFT_SIZE as f32).cos())
            .collect();
        // ~60 Hz post-mix low-pass: PSK31 occupies ~50 Hz, this keeps the
        // matched filter from seeing neighbouring signals in the search window.
        let lpf_a = 1.0 - (-2.0 * PI * 60.0 / fs).exp();
        let mut varicode = HashMap::with_capacity(128);
        for (i, code) in VARICODE.iter().enumerate() {
            // Only the printable half of the table is mapped. Noise decodes
            // to random varicode, and the control codes it lands on (ESC,
            // BEL, CR, the C0 block) would go straight through to the
            // terminal and corrupt the display. An unmapped code emits
            // nothing at all, which is the honest result for noise anyway.
            let c = i as u8 as char;
            if c == '\n' || (' '..='~').contains(&c) {
                varicode.insert(*code, c);
            }
        }
        Self {
            fs,
            sps,
            idx: 0,
            energy: vec![0.0; sps],
            since: 0,
            symbol_len: sps,
            acc: Complex32::new(0.0, 0.0),
            prev: Complex32::new(0.0, 0.0),
            afc: 0.0,
            mix_hz: 0.0,
            mix_phase: 0.0,
            locked: false,
            lock_score: 0.0,
            code: String::new(),
            pending_zero: false,
            text: String::new(),
            symbols: 0,
            q_raw: 0.0,
            q_n: 0.0,
            reversals: 0.0,
            have_prev: false,
            lpf: Complex32::new(0.0, 0.0),
            lpf_a,
            search_buf: Vec::with_capacity(FFT_SIZE * 2),
            since_search: 0,
            fft,
            fft_buf: vec![Complex32::new(0.0, 0.0); FFT_SIZE],
            window,
            scan: CallScanner::new(),
            hits: Vec::new(),
            varicode,
            sym_hist: VecDeque::with_capacity(SYM_HIST + 1),
            env_hist: VecDeque::with_capacity(ENV_HIST + 1),
            env_decim: 0,
            hold_tune: 0,
            #[cfg(test)]
            captured_bits: Vec::new(),
        }
    }

    /// Copy confidence, 0 = noise and 1 = a perfectly resolved constellation.
    ///
    /// Discounted by how young the average is. A mean of |cos| over `n`
    /// symbols carries a standard error of about 0.85/√n on this scale, so an
    /// average four symbols old routinely reads 40% on pure noise — and a
    /// threshold is only worth anything if noise cannot walk over it. The
    /// discount is measured against the settled window (64 symbols) so it
    /// falls to nothing once the estimate has earned its number, rather than
    /// taxing every reading forever.
    fn conf(&self) -> f32 {
        let young = 0.85 * (1.0 / self.q_n.max(1.0).sqrt() - 1.0 / (Q_WINDOW as f32).sqrt());
        (calibrate(self.q_raw) - young.max(0.0)).clamp(0.0, 1.0)
    }

    fn mix(&mut self, s: Complex32) -> Complex32 {
        let (sin, cos) = self.mix_phase.sin_cos();
        self.mix_phase += -2.0 * PI * self.mix_hz / self.fs;
        if self.mix_phase > PI {
            self.mix_phase -= 2.0 * PI;
        } else if self.mix_phase < -PI {
            self.mix_phase += 2.0 * PI;
        }
        let mixed = s * Complex32::new(cos, sin);
        self.lpf += (mixed - self.lpf) * self.lpf_a;
        self.lpf
    }

    fn retune(&mut self, hz: f32, score: f32) {
        let jump = (hz - self.mix_hz).abs();
        self.mix_hz = hz;
        self.lock_score = score;
        self.locked = true;
        if jump > 2.0 {
            // New carrier: drop half-built varicode and timing so the first
            // symbols of the new lock are not decoded against the old clock.
            self.code.clear();
            self.pending_zero = false;
            self.have_prev = false;
            self.acc = Complex32::new(0.0, 0.0);
            self.since = 0;
            self.symbol_len = self.sps;
            self.afc = 0.0;
            self.lpf = Complex32::new(0.0, 0.0);
            self.energy.iter_mut().for_each(|e| *e = 0.0);
            self.scan.reset();
            // Confidence describes the carrier being demodulated, and this is
            // a different carrier. Carrying the old number over would let a
            // good lock vouch for whatever the search jumped to next.
            self.q_raw = 0.0;
            self.q_n = 0.0;
        }
    }

    /// Square the last FFT_SIZE samples, find every 2×offset peak that looks
    /// like PSK31, and lock the strongest (or stay on the current lock).
    fn search(&mut self) {
        let n = FFT_SIZE;
        if self.search_buf.len() < n {
            return;
        }
        let start = self.search_buf.len() - n;
        let slice = &self.search_buf[start..];
        for i in 0..n {
            let s = slice[i] * self.window[i];
            self.fft_buf[i] = s * s;
        }
        self.fft.process(&mut self.fft_buf);

        let bin_hz = self.fs / n as f32;
        let max_bin = ((2.0 * SEARCH_HZ) / bin_hz).ceil() as usize;
        let max_bin = max_bin.min(n / 2 - 2).max(2);

        let mag = |c: Complex32| c.norm();
        let mut floor = Vec::with_capacity(max_bin * 2 + 1);
        floor.push(mag(self.fft_buf[0]));
        for k in 1..=max_bin {
            floor.push(mag(self.fft_buf[k]));
            floor.push(mag(self.fft_buf[n - k]));
        }
        let mut sorted = floor.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = sorted[sorted.len() / 2].max(1e-20);

        // Local maxima in the z² spectrum — each is a candidate 2×offset.
        let mut peaks: Vec<(usize, f32)> = Vec::new();
        let consider = |k: usize, v: f32, lo: f32, hi: f32, peaks: &mut Vec<(usize, f32)>| {
            if v >= lo && v >= hi && v / med >= PEAK_RATIO {
                peaks.push((k, v / med));
            }
        };
        let v0 = mag(self.fft_buf[0]);
        consider(0, v0, mag(self.fft_buf[1]), mag(self.fft_buf[n - 1]), &mut peaks);
        for k in 1..=max_bin {
            let v = mag(self.fft_buf[k]);
            consider(k, v, mag(self.fft_buf[k - 1]), mag(self.fft_buf[k + 1]), &mut peaks);
            let kn = n - k;
            let vn = mag(self.fft_buf[kn]);
            let lo = mag(self.fft_buf[(kn + n - 1) % n]);
            let hi = mag(self.fft_buf[(kn + 1) % n]);
            consider(kn, vn, lo, hi, &mut peaks);
        }
        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        peaks.truncate(8);

        let mut hits = Vec::new();
        for (k, ratio) in peaks {
            let hz = interp_hz(&self.fft_buf, k, n, bin_hz).clamp(-SEARCH_HZ, SEARCH_HZ);
            let hz = refine_hz(slice, self.fs, hz, self.sps);
            if let Some((q, _)) = confirm_psk(slice, self.fs, hz, self.sps) {
                if hits.iter().any(|h: &PskHit| (h.offset_hz - hz).abs() < 12.0) {
                    continue;
                }
                // Last and strictest: is it keying at 31.25 baud? Checked once
                // per surviving candidate rather than inside `refine_hz`,
                // which evaluates the cheap confirmations seven times over.
                //
                // Given the whole buffer rather than the FFT frame the search
                // used: resolving a 31.25 Hz line apart from a 45.45 Hz one
                // takes a second of envelope, and half a second of it reads a
                // copyable signal as unkeyed.
                if keys_at_other_baud(&self.search_buf, self.fs, hz) {
                    continue;
                }
                hits.push(PskHit {
                    offset_hz: hz,
                    score: ratio,
                    quality: q,
                });
            }
        }
        hits.sort_by(|a, b| b.quality.partial_cmp(&a.quality).unwrap_or(std::cmp::Ordering::Equal));

        if hits.is_empty() {
            // Idle between characters often produces no fresh peak. Keep
            // the last confirmed set so next_lock still has somewhere to go.
            return;
        }

        for n in hits {
            match self
                .hits
                .iter_mut()
                .find(|h| (h.offset_hz - n.offset_hz).abs() < 12.0)
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
                .find(|h| (h.offset_hz - self.mix_hz).abs() < 8.0)
                .cloned()
            {
                self.retune(h.offset_hz, h.score);
                return;
            }
        }
        if let Some(h) = self.hits.first().cloned() {
            self.retune(h.offset_hz, h.score);
            self.locked = true;
        }
    }

    fn clear_lock_state(&mut self) {
        self.mix_hz = 0.0;
        self.mix_phase = 0.0;
        self.afc = 0.0;
        self.locked = false;
        self.lock_score = 0.0;
        self.have_prev = false;
        self.acc = Complex32::new(0.0, 0.0);
        self.since = 0;
        self.symbol_len = self.sps;
        self.lpf = Complex32::new(0.0, 0.0);
        self.energy.iter_mut().for_each(|e| *e = 0.0);
        self.search_buf.clear();
        self.since_search = 0;
        self.hits.clear();
        self.code.clear();
        self.pending_zero = false;
        self.scan.reset();
        self.q_raw = 0.0;
        self.q_n = 0.0;
        self.reversals = 0.0;
        self.sym_hist.clear();
        self.env_hist.clear();
        self.env_decim = 0;
        self.hold_tune = 0;
    }

    fn on_symbol(&mut self, sym: Complex32) {
        self.sym_hist.push_back(sym);
        while self.sym_hist.len() > SYM_HIST {
            self.sym_hist.pop_front();
        }
        self.symbols = self.symbols.wrapping_add(1);
        if !self.have_prev {
            self.prev = sym;
            self.have_prev = true;
            return;
        }
        let d = sym * self.prev.conj();
        self.prev = sym;
        if d.norm() < 1e-12 {
            return;
        }
        // De-rotate by the tracked residual carrier offset.
        let derot = d * Complex32::from_polar(1.0, -self.afc);
        let bit = derot.re >= 0.0;

        // Confidence: how close the symbol sits to the real axis after removing
        // the modulation. Averaged raw, then calibrated on read — a single
        // symbol says almost nothing, and clamping one is what would bias the
        // average (see `calibrate`).
        let q = derot.re.abs() / derot.norm().max(1e-12);
        let a = 0.03f32.max(1.0 / (self.q_n + 1.0));
        self.q_raw += a * (q - self.q_raw);
        self.q_n = (self.q_n + 1.0).min(Q_WINDOW as f32);
        self.reversals = 0.97 * self.reversals + 0.03 * if bit { 0.0 } else { 1.0 };

        // Strip the BPSK modulation, then nudge the AFC toward the leftover
        // phase. Fold a little of that residual into mix_hz so a slow drift
        // is calibrated out of the search NCO rather than fighting it.
        let resid = if bit { derot } else { -derot };
        self.afc += 0.03 * resid.arg();
        self.afc = self.afc.clamp(-0.8, 0.8);
        let resid_hz = self.afc * BAUD / (2.0 * PI);
        if resid_hz.abs() > 0.4 {
            self.mix_hz = (self.mix_hz + 0.15 * resid_hz).clamp(-SEARCH_HZ, SEARCH_HZ);
            self.afc *= 0.85;
        }

        // Drop a stale lock if the demod has been garbage for a while.
        // Identification (vs CW / cross-terms) is search's job.
        if self.locked && self.conf() < 0.10 && self.symbols > 120 {
            self.locked = false;
        }

        self.push_bit(bit);
    }

    fn push_bit(&mut self, bit: bool) {
        #[cfg(test)]
        self.captured_bits.push(bit);
        if bit {
            if self.pending_zero {
                self.code.push('0');
                self.pending_zero = false;
            }
            self.code.push('1');
            if self.code.len() > 12 {
                self.code.clear(); // never a valid varicode; resync
            }
        } else if self.pending_zero {
            // "00" terminates a character.
            if !self.code.is_empty() {
                if let Some(c) = self.varicode.get(self.code.as_str()).copied() {
                    if self.locked && self.conf() >= PRINT_QUALITY {
                        self.text.push(c);
                        self.scan.push(c);
                    }
                }
                self.code.clear();
            }
            self.pending_zero = false;
        } else {
            self.pending_zero = true;
        }
    }

}

/// True PSK31 after mixing `hz` to DC: BPSK on the real axis, a plausible
/// reversal rate, energy concentrated near DC (rejects the z² cross-term of
/// two nearby signals), and — when the stream looks like idle — a 31.25 Hz
/// envelope notch (rejects a CW carrier parked half a baud off).
fn confirm_psk(buf: &[Complex32], fs: f32, hz: f32, sps: usize) -> Option<(f32, f32)> {
    let (q, rev) = score_bpsk(buf, fs, hz, sps);
    if q < BPSK_QUALITY || !(REV_MIN..=REV_MAX).contains(&rev) {
        return None;
    }
    if dc_focus(buf, fs, hz) < 0.25 {
        return None;
    }
    // Idle-like reversal rate is also what a CW tone at ±baud/2 produces.
    // Real PSK31 idle has a deep raised-cosine notch; the CW ghost does not.
    if rev > 0.82 && env_notch(buf, fs, hz, sps) < 0.28 {
        return None;
    }
    Some((q, rev))
}

/// How much more of the mixed energy sits inside ±20 Hz than noise alone
/// would put there. A real PSK31 carrier lives in that window; the z² cross
/// term of two neighbours lands near ±baud, out in the 20-50 Hz ring.
///
/// The measure is taken against a noise baseline rather than as a raw
/// fraction. Every sample is hard-limited to unit magnitude first (so the
/// raised-cosine amplitude notches cannot fake a DC peak when the carrier is
/// really a baud away), and hard-limiting noise spreads it evenly across the
/// bins — five of the thirteen counted are the ±20 Hz window, so pure noise
/// scores 0.385, not 0. Comparing a raw fraction against a fixed threshold
/// therefore tests signal-to-noise, not concentration, and rejected exactly
/// the weak signals PSK31 exists to work: at 6 dB a perfectly good carrier
/// scored 0.52 against a 0.70 bar. Subtracting the baseline leaves 0 for
/// noise and 1 for a clean carrier, whatever the SNR.
fn dc_focus(buf: &[Complex32], fs: f32, hz: f32) -> f32 {
    // Half a second where there is that much: sixteen symbols rather than
    // four, which is the difference between an estimate and a guess.
    let n_win = buf.len().min(4096);
    if n_win < 1024 {
        return 1.0;
    }
    let n = n_win;
    let mut osc = Rotator::new(-2.0 * PI * hz / fs);
    let start = buf.len() - n;
    // Channel-filter before hard-limiting. The span scout hands this the
    // full 8 kHz of decimated audio, so a 31 Hz signal that is a healthy
    // 10 dB in its own bandwidth is 11 dB *below* the noise across that
    // audio — and a limiter fed that measures the noise's phase, not the
    // carrier's. Filtering first is what lets the anti-AM limiting do its
    // job (the raised-cosine notches are at 31 Hz, well inside this) while
    // leaving a weak carrier standing.
    let a = 1.0 - (-2.0 * PI * 60.0 / fs).exp();
    let mut lp = [Complex32::new(0.0, 0.0); 3];
    let mut mixed = vec![Complex32::new(0.0, 0.0); n];
    for i in 0..n {
        let mut m = buf[start + i] * osc.next();
        for z in lp.iter_mut() {
            *z += (m - *z) * a;
            m = *z;
        }
        // Strip AM so idle-PSK sidebands cannot fake a DC peak when the
        // carrier is actually sitting at ±baud.
        let nm = m.norm();
        mixed[i] = if nm > 1e-12 { m / nm } else { m };
    }
    let bin_hz = fs / n as f32;
    let mut low = 0.0f32;
    let mut high = 0.0f32;
    let (mut n_low, mut n_high) = (0usize, 0usize);
    // Only a few DFT bins are needed (|f| ≤ 50 Hz) — but both signs: a
    // residual offset a few hertz *below* DC puts the carrier's energy in
    // the negative-frequency bins, and ignoring those rejects real PSK31.
    let k_max = ((50.0 / bin_hz).ceil() as usize).min(n / 2);
    for k in 0..=k_max {
        for &bin in &[k, n - k] {
            if bin == n || (k == 0 && bin != 0) {
                continue; // N-0 aliases bin 0; count DC once
            }
            let mut acc = Complex32::new(0.0, 0.0);
            let mut w = Rotator::new(-2.0 * PI * bin as f32 / n as f32);
            for &s in mixed.iter() {
                acc += s * w.next();
            }
            let p = acc.norm_sqr();
            let f = k as f32 * bin_hz;
            if f <= 20.0 {
                low += p;
                n_low += 1;
            } else if f <= 50.0 {
                high += p;
                n_high += 1;
            }
        }
    }
    let raw = low / (low + high).max(1e-20);
    // Bins counted: DC plus +/-7.8 and +/-15.6 Hz in the window, against
    // +/-23.4 through +/-46.9 Hz outside it.
    let baseline = n_low as f32 / (n_low + n_high).max(1) as f32;
    ((raw - baseline) / (1.0 - baseline).max(1e-6)).clamp(0.0, 1.0)
}

/// The rate at which the candidate is keying, and how far that line stands
/// out of its own envelope spectrum.
///
/// This is the one feature that is *definitionally* PSK31 rather than merely
/// consistent with it. Everything else `confirm_psk` measures — energy at DC,
/// symbols on the real axis, a plausible reversal rate — a keyed carrier can
/// satisfy, which is precisely how an RTTY mark tone was being confirmed as
/// PSK31 and handed to the demodulator: RTTY idles on mark, so its mark tone
/// alone is a carrier switching on and off, and switching on and off is what
/// a BPSK detector reads as symbols.
///
/// A signal that carries information has to change state, and the rate it
/// changes at is the mode. PSK31's raised-cosine pulses take the envelope to
/// zero at every phase reversal, putting a line at 31.25 Hz; RTTY at 45.45
/// baud puts one at 45.45 (or 22.7 for the alternating `RY` pattern); an
/// unkeyed carrier has none at all.
///
/// Returns (strongest line Hz, its peak/median, the same ratio measured at
/// 31.25 Hz) — the caller needs both the winner and PSK31's own line, because
/// "there is a line at 31.25" and "nothing else is louder" are different
/// claims and a keyed carrier can accidentally satisfy the first.
pub(crate) fn baud_line(buf: &[Complex32], fs: f32, hz: f32) -> (f32, f32, f32) {
    // Two seconds is sixty PSK31 symbols and ninety RTTY bits — enough for a
    // clock line to be a line rather than a smear.
    let n_win = buf.len().min((fs * 2.0) as usize);
    if n_win < (fs * 0.5) as usize {
        return (0.0, 0.0, 0.0);
    }
    let start = buf.len() - n_win;
    // Channel filter first: the envelope has to be this signal's keying and
    // not a neighbour's leaking through.
    // The same ~60 Hz channel filter the demodulator runs. Narrower raises
    // PSK31's own line, but it raises every other mode's leakage at 31.25 Hz
    // by more — measured, a 35 Hz six-pole filter improves the absolute
    // reading and halves the separation, which is the wrong trade for a test
    // whose entire job is telling the modes apart.
    let a = 1.0 - (-2.0 * PI * 60.0 / fs).exp();
    let mut lp = [Complex32::new(0.0, 0.0); 3];
    let dec = (fs / 500.0).round().max(1.0) as usize;
    let mut env: Vec<f32> = Vec::with_capacity(n_win / dec + 1);
    let mut osc = Rotator::new(-2.0 * PI * hz / fs);
    let mut acc = 0.0f32;
    let mut k = 0usize;
    for i in 0..n_win {
        let mut m = buf[start + i] * osc.next();
        for z in lp.iter_mut() {
            *z += (m - *z) * a;
            m = *z;
        }
        acc += m.norm();
        k += 1;
        if k == dec {
            env.push(acc / dec as f32);
            acc = 0.0;
            k = 0;
        }
    }
    let n = env.len();
    if n < 64 {
        return (0.0, 0.0, 0.0);
    }
    let env_fs = fs / dec as f32;
    let mean = env.iter().sum::<f32>() / n as f32;
    // Hann against the mean: an unwindowed rectangle smears a strong line
    // across the whole band and every candidate then looks keyed.
    let win: Vec<f32> = env
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let w = 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos();
            (e - mean) * w
        })
        .collect();
    // 15–70 Hz covers PSK31's 31.25, RTTY's 22.7 and 45.45, and the 75 baud
    // and 100 baud lines that would say "not PSK31" just as clearly.
    // A second of envelope resolves ~1 Hz, so half-hertz steps sample the
    // Hann main lobe without paying for detail the window cannot deliver.
    let line_at = |f: f32| -> f32 {
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        let w = -2.0 * PI * f / env_fs;
        for (i, &v) in win.iter().enumerate() {
            let (s, c) = (w * i as f32).sin_cos();
            re += v * c;
            im += v * s;
        }
        (re * re + im * im).sqrt()
    };
    let mut best = (0.0f32, 0.0f32);
    let mut mags: Vec<f32> = Vec::with_capacity(111);
    let mut f = 15.0f32;
    while f <= 70.0 {
        let mag = line_at(f);
        mags.push(mag);
        if mag > best.1 {
            best = (f, mag);
        }
        f += 0.5;
    }
    mags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = mags[mags.len() / 2].max(1e-20);
    (best.0, best.1 / med, line_at(BAUD) / med)
}

/// Reject a candidate whose envelope says it is keying at some *other* rate.
///
/// Framed as a veto rather than a requirement, deliberately. Requiring a
/// strong 31.25 Hz line would be the stronger statement, but the line's
/// absolute level depends on how the audio reached here — the span scout's
/// decimation costs about 4 dB against the decoder's own filtered baseband —
/// and a threshold tuned on one path silently stops identifying real signals
/// on the other. What survives that difference is the *comparison*: when a
/// signal is keying at 17 or 20 or 45.5 Hz, that line towers over 31.25, and
/// no genuine PSK31 signal does that.
///
/// Measured over one second, PSK31 at 10 dB reads 5.6 at 31.25 Hz through the
/// decoder and 3.2 through the span scout, with nothing else louder. An RTTY
/// mark tone reads 2.0 there against 12.3 at 17 Hz; mid-shift, 0.1 against
/// 5.2 at 45.5 Hz; the RY pattern 1.1 against 25.2; keyed CW 2.4 against 7.9.
/// A three-to-one margin sits in the gap with room on both sides.
fn keys_at_other_baud(buf: &[Complex32], fs: f32, hz: f32) -> bool {
    let (_, peak, at_baud) = baud_line(buf, fs, hz);
    // Too little audio to have measured anything; the other confirmations
    // stand on their own rather than being vetoed by a non-measurement.
    if peak <= 0.0 {
        return false;
    }
    at_baud < 3.0 && peak > 3.0 * at_baud
}

fn env_notch(buf: &[Complex32], fs: f32, hz: f32, sps: usize) -> f32 {
    if sps == 0 {
        return 0.0;
    }
    let step = -2.0 * PI * hz / fs;
    let mut phase = 0.0f32;
    let mut acc = vec![0.0f32; sps];
    let mut cnt = vec![0u32; sps];
    let mut i = 0usize;
    // ~60 Hz low-pass after the mix: the notch belongs to the candidate
    // alone, and a neighbour elsewhere in the audio would otherwise fill
    // it in and get real PSK31 rejected as a CW ghost.
    let lpf_a = 1.0 - (-2.0 * PI * 60.0 / fs).exp();
    let mut lpf = Complex32::new(0.0, 0.0);
    for &s in buf {
        let (sin, cos) = phase.sin_cos();
        phase += step;
        if phase > PI {
            phase -= 2.0 * PI;
        } else if phase < -PI {
            phase += 2.0 * PI;
        }
        lpf += (s * Complex32::new(cos, sin) - lpf) * lpf_a;
        acc[i] += lpf.norm();
        cnt[i] += 1;
        i += 1;
        if i == sps {
            i = 0;
        }
    }
    let mut lo = f32::MAX;
    let mut hi = 0.0f32;
    for j in 0..sps {
        if cnt[j] == 0 {
            continue;
        }
        let v = acc[j] / cnt[j] as f32;
        lo = lo.min(v);
        hi = hi.max(v);
    }
    if hi < 1e-12 {
        return 0.0;
    }
    (hi - lo) / hi
}

/// Mix `buf` by `hz` and measure BPSK quality plus phase-reversal rate over
/// a handful of symbols. Used to tell PSK31 from a CW carrier or noise.
///
/// The quality returned is calibrated (0 = noise, 1 = clean), so it can be
/// read against the same scale as the demodulator's own confidence. The
/// comparisons that pick the best clock phase stay on the raw value: the
/// calibration floors at zero, and everything about a noise candidate is
/// below that floor.
fn score_bpsk(buf: &[Complex32], fs: f32, hz: f32, sps: usize) -> (f32, f32) {
    if buf.len() < sps * 8 {
        return (0.0, 0.0);
    }
    // Mix once, then try four symbol-clock phases and keep the best: an
    // integrate-and-dump straddling the reversals scores a real signal as
    // noise, and the search has no other timing recovery.
    let step = -2.0 * PI * hz / fs;
    let mut phase = 0.0f32;
    let mixed: Vec<Complex32> = buf
        .iter()
        .map(|&s| {
            let (sin, cos) = phase.sin_cos();
            phase += step;
            if phase > PI {
                phase -= 2.0 * PI;
            } else if phase < -PI {
                phase += 2.0 * PI;
            }
            s * Complex32::new(cos, sin)
        })
        .collect();
    let mut best = (0.0f32, 0.0f32);
    for off in [0, sps / 4, sps / 2, 3 * sps / 4] {
        let mut prev = Complex32::new(0.0, 0.0);
        let mut have = false;
        let mut q_sum = 0.0;
        let mut n = 0u32;
        let mut revs = 0u32;
        let mut i = off;
        while i + sps <= mixed.len() {
            let mut acc = Complex32::new(0.0, 0.0);
            for k in 0..sps {
                acc += mixed[i + k];
            }
            i += sps;
            if !have {
                prev = acc;
                have = true;
                continue;
            }
            let d = acc * prev.conj();
            prev = acc;
            let nrm = d.norm();
            if nrm < 1e-12 {
                continue;
            }
            q_sum += d.re.abs() / nrm;
            if d.re < 0.0 {
                revs += 1;
            }
            n += 1;
        }
        if n > 0 {
            let q = q_sum / n as f32;
            if q > best.0 {
                best = (q, revs as f32 / n as f32);
            }
        }
    }
    (calibrate(best.0), best.1)
}

impl Decoder for Psk31Decoder {
    fn name(&self) -> &'static str {
        "PSK31"
    }

    fn bandwidth(&self) -> f32 {
        BANDWIDTH
    }

    fn lock_hz(&self) -> f32 {
        self.mix_hz + self.afc * BAUD / (2.0 * PI)
    }

    fn locked(&self) -> bool {
        self.locked
    }

    fn squelched(&self) -> bool {
        // Search and lock must keep running; a squelch on the cursor would
        // hide a nearby PSK31 the scout is supposed to find.
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
                .find(|h| h.offset_hz > cur + 8.0)
                .or_else(|| hs.first())
        } else {
            hs.iter()
                .rev()
                .find(|h| h.offset_hz < cur - 8.0)
                .or_else(|| hs.last())
        };
        pick.cloned().map(|h| {
            self.retune(h.offset_hz, h.score);
            self.locked = true;
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
        self.afc = 0.0;
        Some(self.lock_hz())
    }

    fn psk_view(&self) -> Option<PskView> {
        let mut env: Vec<f32> = self.env_hist.iter().copied().collect();
        let mx = env.iter().cloned().fold(1e-9f32, f32::max);
        for e in &mut env {
            *e = (*e / mx).clamp(0.0, 1.0);
        }
        Some(PskView {
            symbols: self.sym_hist.iter().copied().collect(),
            env,
            lock_hz: self.lock_hz(),
            tune_err_hz: self.afc * BAUD / (2.0 * PI),
            quality: self.conf(),
            reversals: self.reversals,
            locked: self.locked,
            hits: self.hits.clone(),
        })
    }

    fn process(&mut self, samples: &[Complex32]) -> String {
        self.search_buf.extend_from_slice(samples);
        // Trim to the window rather than back to one FFT frame. Draining to
        // FFT_SIZE made the buffer saw between one frame and two, so half the
        // searches saw only half a second — fine for the FFT, which reads the
        // last frame either way, but not for the baud-rate confirmation,
        // which needs a full second to tell 31.25 Hz from 45.45 Hz.
        if self.search_buf.len() > FFT_SIZE * 2 {
            let excess = self.search_buf.len() - FFT_SIZE * 2;
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
            let s = self.mix(raw);
            self.env_decim += 1;
            if self.env_decim >= 8 {
                self.env_decim = 0;
                let nrm = s.norm();
                self.env_hist.push_back(nrm);
                while self.env_hist.len() > ENV_HIST {
                    self.env_hist.pop_front();
                }
            }
            // Raised-cosine weight over the symbol: closer to the matched
            // filter than a rectangular dump, and it emphasises the middle
            // of the pulse where the envelope (and SNR) peaks.
            let n = self.symbol_len.max(1) as f32;
            let x = (self.since as f32 + 0.5) / n;
            let w = 0.5 * (1.0 + (PI * (x - 0.5) * 2.0).cos());
            self.idx = (self.idx + 1) % self.sps;
            self.energy[self.idx] = 0.95 * self.energy[self.idx] + 0.05 * s.norm();
            self.acc += s * w;
            self.since += 1;

            // Dump on a sample countdown rather than a phase match: the symbol
            // period stays ~sps no matter how the timing estimate moves, which
            // is what keeps the bit stream from slipping.
            if self.since >= self.symbol_len {
                let mut sym = self.acc;
                self.acc = Complex32::new(0.0, 0.0);
                self.since = 0;
                // PSK31 is phase-only: normalising the dump removes amplitude
                // fading so the AFC and quality metric stay calibrated.
                let nrm = sym.norm();
                if nrm > 1e-12 {
                    sym /= nrm;
                }
                self.on_symbol(sym);

                // The envelope minimum marks the true symbol boundary; steer the
                // next symbol's length to walk the dump onto it.
                let mut best = 0usize;
                let mut best_v = f32::MAX;
                for (i, &e) in self.energy.iter().enumerate() {
                    if e < best_v {
                        best_v = e;
                        best = i;
                    }
                }
                self.symbol_len = if self.symbols < 3 {
                    self.sps
                } else {
                    let n = self.sps as isize;
                    let err = ((best as isize - self.idx as isize + n / 2).rem_euclid(n)) - n / 2;
                    // Pull hard while acquiring, then only trim, so an
                    // established lock stays steady.
                    let limit = if self.symbols < 60 { n / 8 } else { 2 };
                    (n + err.clamp(-limit, limit)).max(4) as usize
                };
            }
        }
        std::mem::take(&mut self.text)
    }

    fn take_messages(&mut self) -> Vec<FtMessage> {
        let (stamp, snr, hz) = (utc_hhmmss(), snr_from_quality(self.conf()), self.lock_hz());
        self.scan
            .take_calls()
            .into_iter()
            .map(|call| FtMessage {
                stamp: stamp.clone(),
                snr_db: snr,
                dt_sec: 0.0,
                freq_hz: hz,
                text: format!("CQ {call}"),
            })
            .collect()
    }

    fn status(&self) -> String {
        let hz = self.lock_hz();
        let n = self.hits.len();
        let more = if n > 1 { format!(" +{}", n - 1) } else { String::new() };
        let q = self.conf() * 100.0;
        if self.locked {
            format!("lock {hz:+.1}Hz q={q:.0}%{more}")
        } else if self.lock_score > 0.0 {
            format!("near {hz:+.1}Hz q={q:.0}%")
        } else {
            format!("search q={q:.0}%")
        }
    }

    fn confidence(&self) -> Option<f32> {
        Some(self.conf())
    }

    fn speed(&self) -> Option<String> {
        Some("31bd".into())
    }

    fn reset(&mut self) {
        self.text.clear();
        self.scan.reset();
        self.clear_lock_state();
    }
}

/// Confirm which of `peaks` (offsets from IQ DC, Hz) are PSK31. Used by the
/// span-wide scout so `n` / `p` hop between real PSK31 signals, not just
/// energy peaks.
pub fn scan_span(iq: &[Complex32], fs: f64, peaks: &[(f64, f32)]) -> Vec<PskHit> {
    let decim = (fs / SCAN_AUDIO as f64).round().max(1.0) as usize;
    let sps = (SCAN_AUDIO / BAUD).round().max(4.0) as usize;
    let need = sps * 8;
    if iq.len() / decim < need {
        return Vec::new();
    }
    let mut out = Vec::new();
    for &(off, _) in peaks {
        let audio = mix_decim(iq, fs as f32, off as f32, decim);
        if audio.len() < need {
            continue;
        }
        let hz = refine_hz(&audio, SCAN_AUDIO, 0.0, sps);
        if let Some((q, _)) = confirm_psk(&audio, SCAN_AUDIO, hz, sps) {
            if out.iter().any(|h: &PskHit| (h.offset_hz - (off as f32 + hz)).abs() < 30.0)
            {
                continue;
            }
            if keys_at_other_baud(&audio, SCAN_AUDIO, hz) {
                continue;
            }
            out.push(PskHit {
                offset_hz: off as f32 + hz,
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

fn interp_hz(fft: &[Complex32], k: usize, n: usize, bin_hz: f32) -> f32 {
    let mag = |c: Complex32| c.norm();
    let left = if k == 0 {
        mag(fft[n - 1])
    } else {
        mag(fft[k - 1])
    };
    let mid = mag(fft[k]);
    let right = mag(fft[(k + 1) % n]);
    let denom = left - 2.0 * mid + right;
    let frac = if denom.abs() > 1e-12 {
        (0.5 * (left - right) / denom).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    let k_f = k as f32 + frac;
    let two_hz = if k_f <= n as f32 / 2.0 {
        k_f * bin_hz
    } else {
        (k_f - n as f32) * bin_hz
    };
    two_hz * 0.5
}

/// Nudge `hz` by a few tenths of a hertz to maximise BPSK quality.
fn refine_hz(buf: &[Complex32], fs: f32, hz: f32, sps: usize) -> f32 {
    let mut best_hz = hz;
    let mut best_q = confirm_psk(buf, fs, hz, sps).map(|(q, _)| q).unwrap_or(0.0);
    for d in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0] {
        if let Some((q, _)) = confirm_psk(buf, fs, hz + d, sps)
            && q > best_q
        {
            best_q = q;
            best_hz = hz + d;
        }
    }
    best_hz
}

/// Standard PSK31 varicode, indexed by ASCII code point.
#[rustfmt::skip]
pub(crate) const VARICODE: [&str; 128] = [
    "1010101011", "1011011011", "1011101101", "1101110111", "1011101011", "1101011111",
    "1011101111", "1011111101", "1011111111", "11101111",   "11101",      "1101101111",
    "1011011101", "11111",      "1101110101", "1110101011", "1011110111", "1011110101",
    "1110101101", "1110101111", "1101011011", "1101101011", "1101101101", "1101010111",
    "1101111011", "1101111101", "1110110111", "1101010101", "1101011101", "1110111011",
    "1011111011", "1101111111",
    "1",          "111111111",  "101011111",  "111110101",  "111011011",  "1011010101",
    "1010111011", "101111111",  "11111011",   "11110111",   "101101111",  "111011111",
    "1110101",    "110101",     "1010111",    "110101111",
    "10110111",   "10111101",   "11101101",   "11111111",   "101110111",  "101011011",
    "101101011",  "110101101",  "110101011",  "110110111",
    "11110101",   "110111101",  "111101101",  "1010101",    "111010111",  "1010101111",
    "1010111101",
    "1111101",    "11101011",   "10101101",   "10110101",   "1110111",    "11011011",
    "11111101",   "101010101",  "1111111",    "111111101",  "101111101",  "11010111",
    "10111011",   "11011101",   "10101011",   "11010101",   "111011101",  "10101111",
    "1101111",    "1101101",    "101010111",  "110110101",  "101011101",  "101110101",
    "101111011",  "1010101101",
    "111110111",  "111101111",  "111111011",  "1010111111",  "101101101",  "1011011111",
    "1011",       "1011111",    "101111",     "101101",     "11",         "111101",
    "1011011",    "101011",     "1101",       "111101011",  "10111111",   "11011",
    "111011",     "1111",       "111",        "111111",     "110111111",  "10101",
    "10111",      "101",        "110111",     "1111011",    "1101011",    "11011111",
    "1011101",    "111010101",
    "1010110111", "110111011",  "1010110101", "1011010111", "1110110101",
];
