//! Spectrum estimation and the narrowband tuning chain feeding the decoders.

use num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::f32::consts::PI;
use std::sync::Arc;

/// Analysis windows, and the trade they represent.
///
/// A window buys sidelobe suppression with main-lobe width. Sidelobes are
/// leakage: a strong signal smeared across the bins around it, which raises
/// the floor a weak neighbour has to stand out of. Main-lobe width is the
/// opposite concern — a narrowband signal spread over more bins has a lower
/// peak, and every detector here works from peak height over a local floor.
///
/// So the choice is not free in either direction, and which one wins depends
/// on whether the weak signal you care about has a loud neighbour.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Window {
    /// First sidelobe -31 dB, noise bandwidth 1.50 bins. Kept for the bench
    /// that chose against it: the comparison is the justification for the
    /// default, and it is worth being able to re-run rather than trust.
    #[allow(dead_code)]
    Hann,
    /// Four-term Blackman-Harris: sidelobes -92 dB, noise bandwidth 2.00
    /// bins. Sixty dB less leakage, at 1.25 dB of narrowband sensitivity.
    BlackmanHarris,
}

impl Window {
    fn coeffs(self, size: usize) -> Vec<f32> {
        match self {
            Window::Hann => (0..size)
                .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / size as f32).cos())
                .collect(),
            Window::BlackmanHarris => {
                const A: [f32; 4] = [0.358_75, 0.488_29, 0.141_28, 0.011_68];
                (0..size)
                    .map(|i| {
                        let x = 2.0 * PI * i as f32 / size as f32;
                        A[0] - A[1] * x.cos() + A[2] * (2.0 * x).cos()
                            - A[3] * (3.0 * x).cos()
                    })
                    .collect()
            }
        }
    }
}

/// Welch-style averaged periodogram, fftshifted so bin 0 is the lowest frequency.
pub struct Spectrum {
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    size: usize,
    buf: Vec<Complex32>,
    pending: Vec<Complex32>,
}

#[allow(dead_code)]
impl Spectrum {
    pub fn new(size: usize) -> Self {
        Self::with_window(size, Window::BlackmanHarris)
    }

    pub fn with_window(size: usize, win: Window) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(size);
        let window = win.coeffs(size);
        Self {
            fft,
            window,
            size,
            buf: vec![Complex32::new(0.0, 0.0); size],
            pending: Vec::with_capacity(size * 2),
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Average the periodogram over every whole segment available, returning dB.
    ///
    /// Samples are carried across calls so the FFT can be larger than one IQ
    /// block — that is what makes high-resolution views possible. If a full
    /// segment is not yet available `out` is left untouched.
    ///
    /// Segments hop by half an FFT so successive estimates overlap; that
    /// Welch average is what keeps the noise floor from boiling.
    pub fn power_db(&mut self, input: &[Complex32], out: &mut Vec<f32>) {
        self.pending.extend_from_slice(input);
        let hop = (self.size / 2).max(1);
        let nseg = if self.pending.len() >= self.size {
            1 + (self.pending.len() - self.size) / hop
        } else {
            0
        };
        if nseg == 0 {
            if out.is_empty() {
                out.resize(self.size, -140.0);
            }
            return;
        }
        out.clear();
        out.resize(self.size, 0.0);
        let mut acc = vec![0.0f32; self.size];
        for s in 0..nseg {
            let start = s * hop;
            let seg = &self.pending[start..start + self.size];
            for i in 0..self.size {
                self.buf[i] = seg[i] * self.window[i];
            }
            self.fft.process(&mut self.buf);
            for i in 0..self.size {
                acc[i] += self.buf[i].norm_sqr();
            }
        }
        let consumed = nseg * hop;
        if consumed > 0 {
            self.pending.drain(..consumed.min(self.pending.len()));
        }
        let scale = 1.0 / (nseg as f32 * self.size as f32);
        let half = self.size / 2;
        // fftshift while converting to dB
        for i in 0..self.size {
            let src = (i + half) % self.size;
            out[i] = 10.0 * (acc[src] * scale + 1e-20).log10();
        }
    }
}

/// Per-bin minimum-statistics noise estimate. Falls promptly when a quiet
/// observation arrives but rises slowly enough that traffic cannot become its
/// own reference level.
pub struct NoiseFloor {
    bins: Vec<f32>,
}

impl NoiseFloor {
    pub fn new() -> Self { Self { bins: Vec::new() } }

    pub fn update<'a>(&'a mut self, power_db: &[f32]) -> &'a [f32] {
        if self.bins.len() != power_db.len() {
            self.bins = power_db.to_vec();
        } else {
            for (floor, &x) in self.bins.iter_mut().zip(power_db) {
                let a = if x < *floor { 0.25 } else { 0.002 };
                *floor += a * (x - *floor);
            }
        }
        &self.bins
    }

}

/// Frequency-domain smooth. `taps` 1 = none, 3 = [1,2,1], 5 = [1,4,6,4,1].
/// Local maxima are kept so carriers stay sharp while the floor calms down.
pub fn smooth_bins(src: &[f32], taps: usize, out: &mut Vec<f32>) {
    out.clear();
    out.resize(src.len(), -140.0);
    if src.is_empty() {
        return;
    }
    if taps <= 1 || src.len() < 3 {
        out.copy_from_slice(src);
        return;
    }
    let w: &[f32] = if taps >= 5 {
        &[1.0, 4.0, 6.0, 4.0, 1.0]
    } else {
        &[1.0, 2.0, 1.0]
    };
    let r = w.len() / 2;
    for i in 0..src.len() {
        let mut acc = 0.0;
        let mut ww = 0.0;
        for (k, &wk) in w.iter().enumerate() {
            let j = i as isize + k as isize - r as isize;
            if j >= 0 && (j as usize) < src.len() {
                acc += src[j as usize] * wk;
                ww += wk;
            }
        }
        out[i] = acc / ww.max(1e-6);
    }
    for i in 1..src.len().saturating_sub(1) {
        if src[i] >= src[i - 1] && src[i] >= src[i + 1] {
            out[i] = src[i];
        }
    }
}

/// A unit phasor advanced by a fixed angle per sample.
///
/// A complex multiply where `sin_cos` would otherwise be called per sample.
/// The scouts mix the whole span buffer once per candidate frequency, so that
/// call was costing a quarter of a second per scout pass on a busy band —
/// most of the budget between passes, spent on trigonometry.
///
/// A repeated multiply loses magnitude to rounding, so it is renormalised
/// often enough that the drift never reaches the signal: a first-order
/// correction every 1024 samples, which costs one multiply per sample amortised.
pub(crate) struct Rotator {
    cur: Complex32,
    step: Complex32,
    n: u32,
}

impl Rotator {
    pub(crate) fn new(rad_per_sample: f32) -> Self {
        let (sin, cos) = rad_per_sample.sin_cos();
        Self {
            cur: Complex32::new(1.0, 0.0),
            step: Complex32::new(cos, sin),
            n: 0,
        }
    }

    #[inline]
    pub(crate) fn next(&mut self) -> Complex32 {
        let v = self.cur;
        self.cur *= self.step;
        self.n += 1;
        if self.n >= 1024 {
            self.n = 0;
            // Newton step towards |cur| = 1, cheaper than a square root.
            let k = 1.5 - 0.5 * self.cur.norm_sqr();
            self.cur *= k;
        }
        v
    }
}

/// Mix `iq` down by `hz` and decimate by `decim` for the scouts and the
/// signature classifier. The decimation window is triangular (two boxcars
/// back to back), giving a sinc² response — roughly double the stopband dB
/// of a plain block average for one extra accumulator — so a strong
/// neighbour elsewhere in the span aliases far less into the analysis audio.
pub fn mix_decim(iq: &[Complex32], fs: f32, hz: f32, decim: usize) -> Vec<Complex32> {
    let mut osc = Rotator::new(-2.0 * PI * hz / fs);
    let mut out = Vec::with_capacity(iq.len() / decim.max(1) + 1);
    if decim <= 1 {
        for &s in iq {
            out.push(s * osc.next());
        }
        return out;
    }
    let d = decim as f32;
    // Each sample feeds the falling half of the window ending this block and
    // the rising half of the one ending next block; the halves sum to D+1.
    let norm = 1.0 / (d + 1.0);
    let mut fall = Complex32::new(0.0, 0.0);
    let mut rise = Complex32::new(0.0, 0.0);
    let mut r = 0usize;
    for &s in iq {
        let m = s * osc.next();
        fall += m * (1.0 - r as f32 / d);
        rise += m * ((r + 1) as f32 / d);
        r += 1;
        if r == decim {
            out.push(fall * norm);
            fall = rise;
            rise = Complex32::new(0.0, 0.0);
            r = 0;
        }
    }
    out
}

/// Numerically controlled oscillator used to shift a signal of interest to 0 Hz.
pub struct Nco {
    phase: f64,
    step: f64,
}

impl Nco {
    pub fn new() -> Self {
        Self {
            phase: 0.0,
            step: 0.0,
        }
    }

    pub fn set_freq(&mut self, hz: f64, fs: f64) {
        self.step = -2.0 * std::f64::consts::PI * hz / fs;
    }

    pub fn mix(&mut self, input: &[Complex32], out: &mut Vec<Complex32>) {
        out.clear();
        out.reserve(input.len());
        for &s in input {
            let (sin, cos) = self.phase.sin_cos();
            out.push(s * Complex32::new(cos as f32, sin as f32));
            self.phase += self.step;
            if self.phase > std::f64::consts::PI {
                self.phase -= 2.0 * std::f64::consts::PI;
            } else if self.phase < -std::f64::consts::PI {
                self.phase += 2.0 * std::f64::consts::PI;
            }
        }
    }
}

/// Windowed-sinc FIR lowpass that only evaluates the samples it keeps.
pub struct DecimFir {
    taps: Vec<f32>,
    buf: Vec<Complex32>,
    decim: usize,
    fft: Option<Arc<dyn Fft<f32>>>,
    ifft: Option<Arc<dyn Fft<f32>>>,
    response: Vec<Complex32>,
    work: Vec<Complex32>,
    overlap: Vec<Complex32>,
    fft_len: usize,
    fft_phase: usize,
    fft_warm: usize,
}

#[allow(dead_code)]
impl DecimFir {
    pub fn new(cutoff_hz: f32, fs: f32, decim: usize, ntaps: usize) -> Self {
        let ntaps = ntaps | 1; // force odd so there is a center tap
        let mut this = Self {
            taps: lowpass_taps(cutoff_hz, fs, ntaps),
            buf: Vec::with_capacity(ntaps * 4),
            decim,
            fft: None, ifft: None, response: Vec::new(), work: Vec::new(), overlap: Vec::new(), fft_len: 0, fft_phase: 0, fft_warm: 0,
        };
        this.rebuild_fft();
        this
    }

    pub fn set_cutoff(&mut self, cutoff_hz: f32, fs: f32) {
        let n = self.taps.len();
        self.taps = lowpass_taps(cutoff_hz, fs, n);
        self.rebuild_fft();
    }

    /// Redesign at a new length as well as a new cutoff. The sample history
    /// is kept, so changing the filter does not click.
    pub fn set_taps(&mut self, cutoff_hz: f32, fs: f32, ntaps: usize) {
        self.taps = lowpass_taps(cutoff_hz, fs, ntaps | 1);
        self.rebuild_fft();
    }

    pub fn decim(&self) -> usize {
        self.decim
    }

    pub fn process(&mut self, input: &[Complex32], out: &mut Vec<Complex32>) {
        if self.fft.is_some() {
            self.process_fft(input, out);
            return;
        }
        out.clear();
        self.buf.extend_from_slice(input);
        let n = self.taps.len();
        let mut i = 0usize;
        while i + n <= self.buf.len() {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for k in 0..n {
                re += self.buf[i + k].re as f64 * self.taps[k] as f64;
                im += self.buf[i + k].im as f64 * self.taps[k] as f64;
            }
            out.push(Complex32::new(re as f32, im as f32));
            i += self.decim;
        }
        if i > 0 {
            self.buf.drain(..i);
        }
    }

    fn rebuild_fft(&mut self) {
        if self.taps.len() < 64 {
            self.fft = None;
            return;
        }
        self.fft_len = self.taps.len().next_power_of_two();
        if self.fft_len - (self.taps.len() - 1) < 512 {
            self.fft_len *= 2;
        }
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(self.fft_len);
        let ifft = planner.plan_fft_inverse(self.fft_len);
        self.response = vec![Complex32::new(0.0, 0.0); self.fft_len];
        for (dst, &tap) in self.response.iter_mut().zip(&self.taps) { dst.re = tap; }
        fft.process(&mut self.response);
        self.work.resize(self.fft_len, Complex32::new(0.0, 0.0));
        self.overlap.resize(self.taps.len() - 1, Complex32::new(0.0, 0.0));
        self.fft = Some(fft);
        self.ifft = Some(ifft);
        self.buf.clear();
        self.fft_phase = 0;
        self.fft_warm = self.taps.len() - 1;
    }

    fn process_fft(&mut self, input: &[Complex32], out: &mut Vec<Complex32>) {
        out.clear();
        self.buf.extend_from_slice(input);
        let discard = self.taps.len() - 1;
        let hop = self.fft_len - discard;
        while self.buf.len() >= hop {
            self.work.fill(Complex32::new(0.0, 0.0));
            self.work[..discard].copy_from_slice(&self.overlap);
            self.work[discard..discard + hop].copy_from_slice(&self.buf[..hop]);
            self.fft.as_ref().unwrap().process(&mut self.work);
            for (x, h) in self.work.iter_mut().zip(&self.response) { *x *= *h; }
            self.ifft.as_ref().unwrap().process(&mut self.work);
            let scale = 1.0 / self.fft_len as f32;
            for x in &self.work[discard..discard + hop] {
                if self.fft_warm > 0 {
                    self.fft_warm -= 1;
                    continue;
                }
                if self.fft_phase == 0 { out.push(*x * scale); }
                self.fft_phase += 1;
                if self.fft_phase == self.decim { self.fft_phase = 0; }
            }
            if discard <= hop {
                self.overlap.copy_from_slice(&self.buf[hop - discard..hop]);
            } else {
                self.overlap.rotate_left(hop);
                self.overlap[discard - hop..].copy_from_slice(&self.buf[..hop]);
            }
            self.buf.drain(..hop);
        }
    }
}

/// A Blackman-windowed sinc moves from passband to stopband in about
/// `5.5 / ntaps` of the sample rate. Every filter length below is derived
/// from that relationship rather than guessed, because a cutoff far narrower
/// than the transition width is not a filter you designed — it is whatever
/// the window happened to give you.
const TRANSITION_K: f32 = 5.5;

fn transition_hz(fs: f32, ntaps: usize) -> f32 {
    TRANSITION_K * fs / ntaps.max(1) as f32
}

/// Shortest odd tap count whose transition is no wider than `want_hz`.
fn taps_for_transition(fs: f32, want_hz: f32, lo: usize, hi: usize) -> usize {
    let n = (TRANSITION_K * fs / want_hz.max(1e-3)).ceil() as usize;
    n.clamp(lo, hi) | 1
}

fn lowpass_taps(cutoff_hz: f32, fs: f32, ntaps: usize) -> Vec<f32> {
    let fc = (cutoff_hz / fs).clamp(0.0005, 0.49);
    let mid = (ntaps / 2) as f32;
    let mut taps = Vec::with_capacity(ntaps);
    for i in 0..ntaps {
        let x = i as f32 - mid;
        let sinc = if x.abs() < 1e-6 {
            2.0 * fc
        } else {
            (2.0 * PI * fc * x).sin() / (PI * x)
        };
        let xw = 2.0 * PI * i as f32 / (ntaps - 1) as f32;
        let w = 0.358_75 - 0.488_29 * xw.cos() + 0.141_28 * (2.0 * xw).cos()
            - 0.011_68 * (3.0 * xw).cos();
        taps.push(sinc * w);
    }
    let sum: f32 = taps.iter().sum();
    if sum.abs() > 1e-12 {
        taps.iter_mut().for_each(|t| *t /= sum);
    }
    taps
}

/// Radio-rate taps. Only the samples the decimator keeps are evaluated, so
/// this costs `taps × fs_out` multiplies per second regardless of `fs_in`.
const RADIO_TAPS: usize = 2047;
const AUDIO_TAPS_MIN: usize = 129;
const AUDIO_TAPS_MAX: usize = 4095;

/// NCO + decimating lowpass: takes wideband IQ at the radio rate and produces
/// narrowband complex baseband centred on the cursor at the mode's audio rate.
///
/// The two FIRs have distinct jobs. The radio-rate stage is *only* an
/// anti-alias filter: it passes the whole channel flat and is stopped before
/// `fs_out - bw/2`, the lowest frequency that folds into the channel when
/// decimating. It deliberately does not try to realise the channel width
/// itself — an 80 Hz cutoff at 192 kHz is 35× narrower than 511 taps can
/// resolve there, so asking for it yields the window's own shape, not an
/// 80 Hz filter. The audio-rate stage is the channel filter, and its length
/// is sized from the transition the requested bandwidth actually needs.
pub struct DecodeChain {
    nco: Nco,
    fir: DecimFir,
    audio: DecimFir,
    fs_in: f64,
    fs_out: f64,
    mixed: Vec<Complex32>,
    decimated: Vec<Complex32>,
}

impl DecodeChain {
    /// `target_rate` is the audio rate the mode wants. The achieved rate is
    /// `fs_in / round(fs_in / target_rate)`, so it only lands exactly on the
    /// target when the radio rate is an integer multiple of it — which FT8/FT4
    /// require, and the caller enforces.
    pub fn new(fs_in: f64, bandwidth: f32, target_rate: f64) -> Self {
        let decim = (fs_in / target_rate).round().max(1.0) as usize;
        let fs_out = fs_in / decim as f64;
        let mut chain = Self {
            nco: Nco::new(),
            fir: DecimFir::new(fs_out as f32 * 0.5, fs_in as f32, decim, RADIO_TAPS),
            audio: DecimFir::new(bandwidth / 2.0, fs_out as f32, 1, AUDIO_TAPS_MIN),
            fs_in,
            fs_out,
            mixed: Vec::new(),
            decimated: Vec::new(),
        };
        // One place decides both cutoffs and the audio length.
        chain.set_bandwidth(bandwidth);
        chain
    }

    pub fn fs_out(&self) -> f64 {
        self.fs_out
    }

    pub fn set_offset(&mut self, hz: f64) {
        self.nco.set_freq(hz, self.fs_in);
    }

    pub fn set_bandwidth(&mut self, bw: f32) {
        let fs_out = self.fs_out as f32;
        let half = (bw * 0.5).clamp(10.0, fs_out * 0.45);

        // Anti-alias stage. Flat across the channel, stopped by the time
        // the spectrum folds back onto it.
        let tr = transition_hz(self.fs_in as f32, RADIO_TAPS);
        let lo = half + tr * 0.5;
        let hi = (fs_out - half - tr * 0.5).max(lo);
        let radio_cut = (fs_out * 0.5).clamp(lo, hi);
        self.fir.set_cutoff(radio_cut, self.fs_in as f32);

        // Channel stage. FT8/FT4 set the tightest requirement: the chain sits
        // 1600 Hz above the dial and the decoder searches 200–3000 Hz, so the
        // lower skirt has to be flat at 200 Hz and already stopped at 0 Hz —
        // 129 taps could not do that and cost the edges of the waterfall
        // several dB. Ask for a transition that fits inside the setting.
        let want_tr = (0.03 * bw).max(0.01 * fs_out);
        let ntaps = taps_for_transition(fs_out, want_tr, AUDIO_TAPS_MIN, AUDIO_TAPS_MAX);
        self.audio.set_taps(half, fs_out, ntaps);
    }

    pub fn process(&mut self, input: &[Complex32], out: &mut Vec<Complex32>) {
        self.nco.mix(input, &mut self.mixed);
        let mixed = std::mem::take(&mut self.mixed);
        self.fir.process(&mixed, &mut self.decimated);
        self.mixed = mixed;
        self.audio.process(&self.decimated, out);
    }
}

/// Hang AGC for the decoder path. Measures each block, ducks quickly when
/// the signal is hot, then holds and only creeps gain back up after a hang
/// so static crashes and FT8 bursts do not pump the audio.
pub struct SoftAgc {
    gain: f32,
    hang: u32,
    hang_samples: u32,
    attack: f32,
    decay: f32,
}

impl SoftAgc {
    pub fn new(fs: f64) -> Self {
        let fs = fs as f32;
        Self {
            gain: 4.0,
            hang: 0,
            hang_samples: (0.8 * fs) as u32,
            // Per-block blend toward the target. Blocks are ~80 ms of audio.
            attack: 0.45,
            decay: 0.92,
        }
    }

    pub fn reset(&mut self) {
        self.gain = 4.0;
        self.hang = 0;
    }

    pub fn gain(&self) -> f32 {
        self.gain
    }

    pub fn process(&mut self, samples: &mut [Complex32]) {
        if samples.is_empty() {
            return;
        }
        const TARGET: f32 = 0.10;
        const CEILING: f32 = 0.55;

        let mut peak = 0.0f32;
        let mut sum_sq = 0.0f32;
        for s in samples.iter() {
            let n2 = s.norm_sqr();
            sum_sq += n2;
            let mag = n2.sqrt();
            if mag > peak {
                peak = mag;
            }
        }
        let n = samples.len() as f32;
        let rms = (sum_sq / n).sqrt() * self.gain;
        let pk = peak * self.gain;

        if pk > CEILING {
            self.gain *= (CEILING / pk.max(1e-9)).clamp(0.05, 1.0);
            self.hang = self.hang_samples;
        } else if rms > TARGET * 1.15 {
            let want = TARGET / rms.max(1e-9);
            self.gain *= self.attack + (1.0 - self.attack) * want;
            self.hang = self.hang_samples;
        } else if self.hang > 0 {
            self.hang = self.hang.saturating_sub(samples.len() as u32);
        } else if rms > 1e-5 && rms < TARGET * 0.7 {
            let want = (TARGET / rms).min(1.25);
            self.gain *= self.decay + (1.0 - self.decay) * want;
        }
        self.gain = self.gain.clamp(0.08, 60.0);

        let g = self.gain;
        for s in samples.iter_mut() {
            *s *= g;
        }
    }
}

/// One-pole smoother, used for envelopes and AGC-ish level tracking.
#[derive(Clone, Copy)]
pub struct OnePole {
    y: f32,
    a: f32,
}

#[allow(dead_code)]
impl OnePole {
    pub fn new(tau_samples: f32) -> Self {
        Self {
            y: 0.0,
            a: (-1.0 / tau_samples.max(1.0)).exp(),
        }
    }
    pub fn process(&mut self, x: f32) -> f32 {
        self.y = self.a * self.y + (1.0 - self.a) * x;
        self.y
    }
    pub fn value(&self) -> f32 {
        self.y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smooth_bins_keeps_a_peak_and_calms_the_floor() {
        let mut src = vec![-80.0f32; 32];
        src[16] = -20.0;
        src[10] = -70.0;
        src[11] = -68.0;
        src[12] = -71.0;
        let mut out = Vec::new();
        smooth_bins(&src, 5, &mut out);
        assert_eq!(out.len(), src.len());
        assert!((out[16] + 20.0).abs() < 0.01, "peak should be preserved");
        // Neighbours of the floor wiggle are pulled together.
        assert!(out[11] < src[11] + 1.0);
    }

    #[test]
    fn soft_agc_pulls_a_hot_block_down_and_does_not_pump() {
        let mut agc = SoftAgc::new(8000.0);
        let mut hot: Vec<Complex32> = (0..800)
            .map(|_| Complex32::new(0.8, 0.0))
            .collect();
        agc.process(&mut hot);
        let peak = hot.iter().map(|c| c.norm()).fold(0.0f32, f32::max);
        assert!(peak < 0.6, "hot block should be ducked, peak {peak}");

        // After hang, a quiet block must not instantly slam the gain up.
        let gain_after_hot = agc.gain();
        let mut quiet: Vec<Complex32> = (0..800)
            .map(|_| Complex32::new(0.01, 0.0))
            .collect();
        agc.process(&mut quiet);
        assert!(
            agc.gain() <= gain_after_hot * 1.05,
            "gain pumped during hang: {} -> {}",
            gain_after_hot,
            agc.gain()
        );
    }

    #[test]
    fn decode_chain_still_emits_audio() {
        let mut chain = DecodeChain::new(192_000.0, 400.0, 8000.0);
        chain.set_bandwidth(200.0);
        // A quarter second of IQ, fed in the ~16k blocks the radio delivers.
        let input: Vec<Complex32> = (0..48_000)
            .map(|i| Complex32::new((i as f32 * 0.01).sin() * 0.05, 0.0))
            .collect();
        let mut out = Vec::new();
        let mut total = 0usize;
        for c in input.chunks(16_384) {
            chain.process(c, &mut out);
            total += out.len();
        }
        assert!(total > 0, "audio FIR should emit samples");
        // Both FIRs swallow half their length once, at startup. Past that the
        // chain must keep up with the input or audio would drift behind.
        let expect = input.len() / 24;
        assert!(
            total > expect - 1200 && total <= expect,
            "emitted {total} of an expected ~{expect} samples"
        );
    }

    #[test]
    fn overlap_save_matches_direct_decimation() {
        let input: Vec<_> = (0..20_000).map(|i| Complex32::new((i as f32 * 0.013).sin(), (i as f32 * 0.021).cos())).collect();
        let mut fast = DecimFir::new(3500.0, 192_000.0, 24, 4095);
        let taps = fast.taps.clone();
        let mut got = Vec::new();
        for block in input.chunks(3000) { let mut part = Vec::new(); fast.process(block, &mut part); got.extend(part); }
        let mut want = Vec::new();
        for i in (0..=input.len() - taps.len()).step_by(24) {
            want.push(input[i..i + taps.len()].iter().zip(&taps).map(|(x, h)| *x * *h).sum::<Complex32>());
        }
        let err = got.iter().zip(&want).map(|(a,b)| (*a-*b).norm()).fold(0.0f32, f32::max);
        assert!(err < 1e-3, "overlap-save differs by {err}");
    }

    /// Gain of the whole chain at `hz` off centre, relative to DC, in dB.
    fn chain_resp_db(fs_in: f64, bw: f32, target: f64, hz: f64) -> f32 {
        let tone = |hz: f64| -> f32 {
            let mut chain = DecodeChain::new(fs_in, bw, target);
            let n = (fs_in * 0.4) as usize;
            let mut phase = 0.0f64;
            let step = 2.0 * std::f64::consts::PI * hz / fs_in;
            let input: Vec<Complex32> = (0..n)
                .map(|_| {
                    let s = Complex32::from_polar(1.0, phase as f32);
                    phase += step;
                    if phase > std::f64::consts::PI { phase -= 2.0 * std::f64::consts::PI; }
                    s
                })
                .collect();
            let mut out = Vec::new();
            let mut all = Vec::new();
            for c in input.chunks(16384) {
                chain.process(c, &mut out);
                all.extend_from_slice(&out);
            }
            if all.len() < 200 {
                return -200.0;
            }
            let tail = &all[all.len() / 2..];
            let p: f32 = tail.iter().map(|c| c.norm_sqr()).sum::<f32>() / tail.len() as f32;
            10.0 * (p + 1e-30).log10()
        };
        tone(hz) - tone(0.0)
    }

    /// FT8/FT4 put the chain 1600 Hz above the dial and search 200–3000 Hz,
    /// so the passband has to be flat across ±1400 Hz. A filter too soft to
    /// manage that quietly costs the top and bottom of the waterfall several
    /// dB — the stations that were already marginal.
    #[test]
    fn ft8_passband_is_flat_across_the_search_range() {
        for off in [-1400.0, -1200.0, -600.0, 600.0, 1200.0, 1400.0] {
            let r = chain_resp_db(192_000.0, 3000.0, 12_000.0, off);
            assert!(
                r > -1.0,
                "FT8 audio {:.0} Hz is {r:.1} dB down; passband must be flat",
                off + 1600.0
            );
        }
        // ...and still stopped before the real part folds it onto itself.
        let fold = chain_resp_db(192_000.0, 3000.0, 12_000.0, -1750.0);
        assert!(fold < -30.0, "below the dial should be rejected, got {fold:.1} dB");
    }

    /// A filter setting has to mean what it says: the narrow positions are
    /// the ones that dig a signal out of QRM, and they are also the ones a
    /// fixed tap count cannot resolve.
    #[test]
    fn narrow_filters_are_the_width_they_claim() {
        for bw in [80.0f32, 200.0, 500.0] {
            let edge = chain_resp_db(192_000.0, bw, 8000.0, (bw / 2.0) as f64);
            assert!(
                (-9.0..-3.0).contains(&edge),
                "{bw:.0} Hz filter is {edge:.1} dB at its own edge, so it is not {bw:.0} Hz wide"
            );
            // An octave out must be genuinely gone, not merely rolling off.
            let out = chain_resp_db(192_000.0, bw, 8000.0, bw as f64);
            assert!(
                out < -40.0,
                "{bw:.0} Hz filter passes {out:.1} dB at {bw:.0} Hz off centre"
            );
        }
    }

    /// Everything within half a channel of `fs_out` folds straight onto the
    /// signal when the decimator drops samples, and no later filter can
    /// separate it again.
    #[test]
    fn decimation_does_not_alias_into_the_channel() {
        for hz in [7900.0, 8000.0, 8100.0, 16_000.0] {
            let r = chain_resp_db(192_000.0, 400.0, 8000.0, hz);
            assert!(
                r < -80.0,
                "{hz:.0} Hz folds into the channel at {r:.1} dB"
            );
        }
    }

    #[test]
    #[ignore]
    fn bench_fast_convolution() {
        use std::time::Instant;
        let input: Vec<_> = (0..96_000).map(|i| Complex32::new((i as f32 * 0.017).sin(), 0.0)).collect();
        let taps = lowpass_taps(100.0, 8000.0, 1101);
        let at = Instant::now();
        let mut direct = Vec::with_capacity(input.len());
        for i in 0..input.len() - taps.len() {
            direct.push(input[i..i + taps.len()].iter().zip(&taps).map(|(x, h)| *x * *h).sum::<Complex32>());
        }
        let direct_ms = at.elapsed().as_secs_f64() * 1000.0;
        let mut fir = DecimFir::new(100.0, 8000.0, 1, 1101);
        let at = Instant::now();
        let mut out = Vec::new();
        for block in input.chunks(4096) { fir.process(block, &mut out); }
        let fft_ms = at.elapsed().as_secs_f64() * 1000.0;
        println!("1101-tap convolution: direct {direct_ms:.1} ms, overlap-save {fft_ms:.1} ms ({:.1}x faster)", direct_ms / fft_ms);
    }
}

/// Basic receiver front-end cleanup, applied to raw IQ before anything else
/// looks at it.
///
/// A quadrature front end has two defects that are not signals and that no
/// amount of care downstream can undo:
///
/// * a **DC offset** — a constant bias on I and Q, which is a carrier sitting
///   exactly on the local oscillator. It is why every candidate picker here
///   has to blank the bins either side of the LO, and so why a real signal
///   that close cannot be seen.
/// * **IQ imbalance** — the two channels differing slightly in gain or in
///   quadrature, which mirrors a copy of every signal about the LO. The image
///   of a strong station is an ordinary-looking weak one, and detection
///   thresholds low enough to find real weak signals are low enough to find
///   those too.
///
/// The driver corrects both where it can (see `radio`), but support varies by
/// device and by SoapySDR backend, so the same corrections are done here as
/// well. Both are cheap, and both are no-ops on IQ that is already clean.
/// Streaming wideband impulse blanker. It deliberately runs before every
/// channel filter, while an RF impulse is still only a few samples wide.
pub struct NoiseBlanker {
    fs: f32,
    background: f32,
    level: u8,
    tail: usize,
    window_seen: usize,
    window_blanked: usize,
    last_rate: usize,
    inhibited: bool,
    trained: usize,
}

impl NoiseBlanker {
    pub fn new(fs: f64) -> Self {
        Self {
            fs: fs as f32,
            background: 1e-3,
            level: 2,
            tail: 0,
            window_seen: 0,
            window_blanked: 0,
            last_rate: 0,
            inhibited: false,
            trained: 0,
        }
    }

    /// Off, gentle (6x), normal (5x), aggressive (4x).
    pub fn cycle(&mut self) -> &'static str {
        self.level = (self.level + 1) % 4;
        ["off", "gentle", "normal", "aggressive"][self.level as usize]
    }

    pub fn label(&self) -> &'static str {
        ["off", "gentle", "normal", "aggressive"][self.level as usize]
    }

    pub fn blanks_per_second(&self) -> usize {
        self.last_rate
    }

    fn process(&mut self, iq: &mut [Complex32]) {
        let a = 1.0 - (-1.0 / (0.075 * self.fs)).exp();
        let threshold = [f32::INFINITY, 6.0, 5.0, 4.0][self.level as usize];
        for i in 0..iq.len() {
            let mag = iq[i].norm();
            if self.trained == 0 {
                self.background = mag.max(1e-6);
            }
            // Clamping keeps a crash from raising the reference used to
            // recognise the rest of that same crash.
            let observed = if self.trained < self.fs as usize / 4 {
                mag
            } else {
                mag.min((self.background * 2.5).max(1e-6))
            };
            self.background += (observed - self.background) * a;
            let hot = self.level != 0
                && !self.inhibited
                && self.trained >= (self.fs as usize / 4).max(32)
                && mag > threshold * self.background.max(1e-6);
            if hot {
                for back in 0..=2 {
                    if let Some(s) = i.checked_sub(back).and_then(|j| iq.get_mut(j)) {
                        if s.norm_sqr() != 0.0 {
                            *s = Complex32::new(0.0, 0.0);
                            self.window_blanked += 1;
                        }
                    }
                }
                self.tail = 3;
            } else if self.tail > 0 {
                // Raised-cosine-like return to full amplitude prevents the
                // blanking edge itself from becoming a broadband click.
                let weight = [1.0, 0.75, 0.25, 0.0][self.tail];
                iq[i] *= weight;
                self.window_blanked += 1;
                self.tail -= 1;
            }
            self.window_seen += 1;
            self.trained = self.trained.saturating_add(1);
            if self.window_seen >= self.fs as usize {
                self.last_rate = self.window_blanked;
                self.inhibited = self.window_blanked * 50 > self.window_seen;
                self.window_seen = 0;
                self.window_blanked = 0;
            }
        }
    }
}

pub struct FrontEnd {
    fs: f64,
    dc: Complex32,
    dc_a: f32,
    iq_fft: Arc<dyn Fft<f32>>,
    iq_ifft: Arc<dyn Fft<f32>>,
    iq_buf: Vec<Complex32>,
    iq_orig: Vec<Complex32>,
    image: [Complex32; 12],
    /// Samples seen, against the number needed before correcting at all.
    warm: u32,
    settle: u32,
    blanker: NoiseBlanker,
}

impl FrontEnd {
    pub fn new(fs: f64) -> Self {
        let fsf = fs as f32;
        let mut planner = FftPlanner::new();
        Self {
            fs,
            dc: Complex32::new(0.0, 0.0),
            // ~2 Hz corner. Slow enough that it is a bias estimate rather
            // than a filter, so it cannot touch a signal even a few tens of
            // hertz off the LO — the region worth recovering in the first place.
            dc_a: 1.0 - (-2.0 * PI * 2.0 / fsf).exp(),
            iq_fft: planner.plan_fft_forward(4096),
            iq_ifft: planner.plan_fft_inverse(4096),
            iq_buf: vec![Complex32::new(0.0, 0.0); 4096],
            iq_orig: vec![Complex32::new(0.0, 0.0); 4096],
            image: [Complex32::new(0.0, 0.0); 12],
            warm: 0,
            // Two time constants of the estimator above.
            settle: (2.0 * fs) as u32,
            blanker: NoiseBlanker::new(fs),
        }
    }

    /// Forget everything learned — after a retune or a rate change, when the
    /// front end is being asked a different question.
    pub fn reset(&mut self) {
        *self = FrontEnd::new(self.fs);
    }

    /// The correction currently being applied, for display: DC magnitude and
    /// image rejection in dB (higher is better; ~100 means nothing to correct).
    pub fn status(&self) -> (f32, f32) {
        let err = self.image.iter().map(|v| v.norm()).fold(0.0f32, f32::max);
        let rej = if err > 1e-6 {
            -20.0 * err.log10()
        } else {
            100.0
        };
        (self.dc.norm(), rej.min(100.0))
    }

    pub fn cycle_blanker(&mut self) -> &'static str {
        self.blanker.cycle()
    }

    pub fn blanker_status(&self) -> (&'static str, usize) {
        (self.blanker.label(), self.blanker.blanks_per_second())
    }

    pub fn process(&mut self, iq: &mut [Complex32]) {
        for s in iq.iter_mut() {
            // --- DC offset: track the mean and subtract it.
            self.dc += (*s - self.dc) * self.dc_a;
            let v = *s - self.dc;

            *s = v;
        }

        self.warm = self.warm.saturating_add(iq.len() as u32);
        self.correct_images(iq);
        self.blanker.process(iq);
    }

    fn correct_images(&mut self, iq: &mut [Complex32]) {
        const N: usize = 4096;
        const K: usize = 12;
        for block in iq.chunks_exact_mut(N) {
            self.iq_buf.copy_from_slice(block);
            self.iq_fft.process(&mut self.iq_buf);
            self.iq_orig.copy_from_slice(&self.iq_buf);
            let mut cross = [Complex32::new(0.0, 0.0); K];
            let mut paired = [0.0f32; K];
            for k in 1..N {
                let shifted = (k + N / 2) % N;
                let band = (shifted * K / N).min(K - 1);
                let mirror = (N - k) % N;
                cross[band] += self.iq_orig[k] * self.iq_orig[mirror];
                paired[band] += self.iq_orig[k].norm_sqr() + self.iq_orig[mirror].norm_sqr();
            }
            let alpha = (N as f32 / (self.fs as f32 * 2.0)).clamp(0.002, 0.2);
            let mut estimate = [None; K];
            let max_power = paired.iter().copied().fold(0.0f32, f32::max);
            for b in 0..K {
                if paired[b] > (max_power * 1e-3).max(1e-12) {
                    let v = cross[b] / paired[b];
                    if v.norm() < 0.25 {
                        estimate[b] = Some(v);
                    }
                }
            }
            for b in 0..K {
                if let Some(mut target) = estimate[b] {
                    let mut weight = 1.0;
                    for neighbour in [b.checked_sub(1), (b + 1 < K).then_some(b + 1)].into_iter().flatten() {
                        if let Some(v) = estimate[neighbour] {
                            target += v * 0.25;
                            weight += 0.25;
                        }
                    }
                    target /= weight;
                    self.image[b] += (target - self.image[b]) * alpha;
                }
            }
            if self.warm >= self.settle {
                for k in 1..N {
                    let shifted = (k + N / 2) % N;
                    let band = (shifted * K / N).min(K - 1);
                    let a = self.image[band];
                    let mirror = (N - k) % N;
                    self.iq_buf[k] = self.iq_orig[k] - a * self.iq_orig[mirror].conj();
                }
                self.iq_ifft.process(&mut self.iq_buf);
                for (dst, src) in block.iter_mut().zip(&self.iq_buf) { *dst = *src / N as f32; }
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod frontend_tests {
    use super::*;

    pub(crate) fn noise(rng: &mut u32) -> f32 {
        *rng ^= *rng << 13;
        *rng ^= *rng >> 17;
        *rng ^= *rng << 5;
        (*rng as f32 / u32::MAX as f32) - 0.5
    }

    /// Power in a narrow band around `hz`, by direct correlation.
    pub(crate) fn power_at(iq: &[Complex32], fs: f32, hz: f32) -> f32 {
        let n = iq.len().min(200_000);
        let start = iq.len() - n;
        let mut osc = Rotator::new(-2.0 * PI * hz / fs);
        let mut acc = Complex32::new(0.0, 0.0);
        for &s in &iq[start..] {
            acc += s * osc.next();
        }
        (acc / n as f32).norm_sqr()
    }

    /// A band of noise with a signal on it, a DC offset, and a gain and
    /// quadrature error between I and Q — an ordinary uncorrected front end.
    pub(crate) fn dirty_iq(fs: f32, sig_hz: f32, dc: Complex32, gain_err: f32, phase_err: f32) -> Vec<Complex32> {
        let mut rng = 0x1234_5678u32;
        // Long enough to cover the front end's settle time several times over.
        (0..(fs * 10.0) as usize)
            .map(|i| {
                let ph = 2.0 * PI * sig_hz * i as f32 / fs;
                let clean = Complex32::from_polar(0.5, ph)
                    + Complex32::new(noise(&mut rng), noise(&mut rng)) * 0.05;
                // Gain and quadrature error on Q, then the DC bias.
                let q = clean.im * gain_err + clean.re * phase_err;
                Complex32::new(clean.re, q) + dc
            })
            .collect()
    }

    /// Run a long stretch of dirty IQ through in blocks, as the app does,
    /// and return the settled tail of the output. Feeding the corrected
    /// output back in instead would measure a feedback loop, not the
    /// correction.
    pub(crate) fn run_frontend(iq: &[Complex32], fs: f32) -> Vec<Complex32> {
        let mut fe = FrontEnd::new(fs as f64);
        let mut out = Vec::with_capacity(iq.len());
        for chunk in iq.chunks(16_384) {
            let mut buf = chunk.to_vec();
            fe.process(&mut buf);
            out.extend_from_slice(&buf);
        }
        // The last quarter, by which time the estimators have long settled.
        out.split_off(out.len() * 3 / 4)
    }

    fn impulse_case(fs: f32, impulses: bool) -> (Vec<Complex32>, Vec<Complex32>) {
        let mut rng = 0x91ab_cdefu32;
        let mut clean = Vec::with_capacity((fs * 3.0) as usize);
        let mut dirty = Vec::with_capacity(clean.capacity());
        for i in 0..clean.capacity() {
            let tone = Complex32::from_polar(0.015, 2.0 * PI * 1500.0 * i as f32 / fs);
            let noise = Complex32::new(noise(&mut rng), noise(&mut rng)) * 0.025;
            clean.push(tone + noise);
            let crash = impulses && i > fs as usize && i % 1800 < 4;
            dirty.push(tone + noise + if crash { Complex32::new(8.0, -6.0) } else { Complex32::new(0.0, 0.0) });
        }
        (clean, dirty)
    }

    fn error_power(got: &[Complex32], want: &[Complex32]) -> f32 {
        got.iter().zip(want).skip(got.len() / 3).map(|(a, b)| (*a - *b).norm_sqr()).sum::<f32>()
            / (got.len() - got.len() / 3) as f32
    }

    #[test]
    fn noise_blanker_removes_impulses_without_harming_clean_iq() {
        let fs = 48_000.0;
        let (clean, dirty) = impulse_case(fs, true);
        let mut blanked = dirty.clone();
        let mut nb = NoiseBlanker::new(fs as f64);
        for block in blanked.chunks_mut(4096) { nb.process(block); }
        let improvement = 10.0 * (error_power(&dirty, &clean) / error_power(&blanked, &clean)).log10();
        assert!(improvement >= 10.0, "impulse error improved only {improvement:.1} dB");

        let (_, mut untouched) = impulse_case(fs, false);
        let before = power_at(&untouched, fs, 1500.0);
        let mut nb = NoiseBlanker::new(fs as f64);
        for block in untouched.chunks_mut(4096) { nb.process(block); }
        let harm = (10.0 * (before / power_at(&untouched, fs, 1500.0)).log10()).abs();
        assert!(harm < 0.2, "clean tone changed by {harm:.2} dB");
    }

    #[test]
    #[ignore]
    fn bench_noise_blanker() {
        let fs = 48_000.0;
        let (clean, dirty) = impulse_case(fs, true);
        let mut blanked = dirty.clone();
        let mut nb = NoiseBlanker::new(fs as f64);
        for block in blanked.chunks_mut(4096) { nb.process(block); }
        let improvement = 10.0 * (error_power(&dirty, &clean) / error_power(&blanked, &clean)).log10();
        println!("wideband impulse blanker: {improvement:.1} dB error-power improvement, {} blanks/s", nb.blanks_per_second());
    }

    /// The DC offset is what forces the LO bins to be blanked, so removing it
    /// is the difference between a signal near the LO being findable and not.
    #[test]
    fn dc_offset_is_removed() {
        let fs = 48_000.0f32;
        let iq = dirty_iq(fs, 400.0, Complex32::new(0.25, -0.18), 1.0, 0.0);
        let before = power_at(&iq, fs, 0.0);
        let after = power_at(&run_frontend(&iq, fs), fs, 0.0);
        let db = 10.0 * (before / after.max(1e-30)).log10();
        assert!(
            db > 40.0,
            "DC only came down {db:.0} dB ({before:.6} -> {after:.9})"
        );
    }

    /// IQ imbalance mirrors every signal about the LO. With detection
    /// thresholds low enough to find real weak signals, the image of a strong
    /// station is an ordinary-looking weak one — a station that is not there.
    #[test]
    fn iq_imbalance_image_is_suppressed() {
        let fs = 48_000.0f32;
        // 6% gain error and 3 degrees of quadrature error: poor, but the sort
        // of thing an uncorrected direct-conversion front end really does.
        let iq = dirty_iq(fs, 400.0, Complex32::new(0.0, 0.0), 1.06, 0.052);
        let before =
            10.0 * (power_at(&iq, fs, 400.0) / power_at(&iq, fs, -400.0)).log10();
        let out = run_frontend(&iq, fs);
        let sig = power_at(&out, fs, 400.0);
        let after = 10.0 * (sig / power_at(&out, fs, -400.0)).log10();
        assert!(
            after > before + 15.0,
            "image rejection only improved {:.1} dB ({before:.1} -> {after:.1})",
            after - before
        );
        assert!(
            sig > power_at(&iq, fs, 400.0) * 0.5,
            "the wanted signal was attenuated"
        );
    }

    fn frequency_dependent_case(fs: f32) -> (Vec<Complex32>, [f32; 2]) {
        let hz = [4687.5, 16406.25]; // exact 4096-point bins
        let a = [Complex32::new(0.09, 0.025), Complex32::new(-0.065, 0.045)];
        let iq = (0..(fs * 10.0) as usize).map(|i| {
            let mut x = Complex32::new(0.0, 0.0);
            for j in 0..2 {
                let s = Complex32::from_polar(0.35, 2.0 * PI * hz[j] * i as f32 / fs);
                x += s + a[j] * s.conj();
            }
            x
        }).collect();
        (iq, hz)
    }

    fn worst_rejection(iq: &[Complex32], fs: f32, hz: [f32; 2]) -> f32 {
        hz.into_iter().map(|f| 10.0 * (power_at(iq, fs, f) / power_at(iq, fs, -f).max(1e-30)).log10())
            .fold(f32::INFINITY, f32::min)
    }

    fn run_subband_only(iq: &[Complex32], fs: f32) -> Vec<Complex32> {
        let mut fe = FrontEnd::new(fs as f64);
        fe.blanker.level = 0;
        let mut out = Vec::new();
        for chunk in iq.chunks(16_384) {
            let mut block = chunk.to_vec();
            fe.process(&mut block);
            out.extend(block);
        }
        out.split_off(out.len() * 3 / 4)
    }

    #[test]
    fn subband_iq_correction_beats_a_scalar_solve() {
        let fs = 48_000.0;
        let (iq, hz) = frequency_dependent_case(fs);
        let scalar_a = Complex32::new(0.0125, 0.035);
        let scalar: Vec<_> = iq.iter().map(|&x| x - scalar_a * x.conj()).collect();
        let scalar_rej = worst_rejection(&scalar, fs, hz);
        let per_band = worst_rejection(&run_subband_only(&iq, fs), fs, hz);
        assert!(per_band >= scalar_rej + 15.0, "subbands gained only {:.1} dB ({scalar_rej:.1} -> {per_band:.1})", per_band - scalar_rej);
    }

    #[test]
    #[ignore]
    fn bench_frequency_dependent_iq() {
        let fs = 48_000.0;
        let (iq, hz) = frequency_dependent_case(fs);
        let scalar_a = Complex32::new(0.0125, 0.035);
        let scalar: Vec<_> = iq.iter().map(|&x| x - scalar_a * x.conj()).collect();
        let scalar_rej = worst_rejection(&scalar, fs, hz);
        let per_band = worst_rejection(&run_subband_only(&iq, fs), fs, hz);
        println!("frequency-dependent IQ: worst-band scalar {scalar_rej:.1} dB, 12-band {per_band:.1} dB ({:.1} dB gain)", per_band - scalar_rej);
    }

    /// And clean IQ must come out unharmed — the correction has to be a
    /// no-op on a front end that has nothing wrong with it.
    #[test]
    fn clean_iq_is_left_alone() {
        let fs = 48_000.0f32;
        let clean = dirty_iq(fs, 400.0, Complex32::new(0.0, 0.0), 1.0, 0.0);
        let out = run_frontend(&clean, fs);
        let sig = power_at(&out, fs, 400.0);
        let want = power_at(&clean, fs, 400.0);
        assert!(
            (sig / want - 1.0).abs() < 0.05,
            "clean signal changed by {:.1}%",
            (sig / want - 1.0) * 100.0
        );
        let img = 10.0 * (sig / power_at(&out, fs, -400.0)).log10();
        assert!(img > 40.0, "invented an image: {img:.0} dB rejection");
    }
}

#[cfg(test)]
mod frontend_bench {
    use super::*;
    use super::frontend_tests::*;

    #[test]
    #[ignore]
    fn bench_frontend() {
        let fs = 48_000.0f32;
        println!("\n== DC offset removal ==");
        for dc in [0.02f32, 0.10, 0.25] {
            let iq = dirty_iq(fs, 400.0, Complex32::new(dc, -dc * 0.7), 1.0, 0.0);
            let before = power_at(&iq, fs, 0.0);
            let after = power_at(&run_frontend(&iq, fs), fs, 0.0);
            println!(
                "  offset {:.2} of full scale: {:.0} dB down",
                dc,
                10.0 * (before / after.max(1e-30)).log10()
            );
        }

        println!("\n== IQ image rejection (gain error / quadrature error) ==");
        for (g, p) in [(1.01f32, 0.009f32), (1.03, 0.026), (1.06, 0.052), (1.12, 0.105)] {
            let iq = dirty_iq(fs, 400.0, Complex32::new(0.0, 0.0), g, p);
            let before =
                10.0 * (power_at(&iq, fs, 400.0) / power_at(&iq, fs, -400.0)).log10();
            let out = run_frontend(&iq, fs);
            let after =
                10.0 * (power_at(&out, fs, 400.0) / power_at(&out, fs, -400.0)).log10();
            println!(
                "  {:>4.0}% gain, {:>4.1} deg: {:>5.1} dB -> {:>5.1} dB",
                (g - 1.0) * 100.0,
                p.asin().to_degrees(),
                before,
                after
            );
        }
    }
}

#[cfg(test)]
mod frontend_cost {
    use super::*;

    /// The front end runs on every sample of every block, so it has to be
    /// negligible against the 192 kHz stream it sits in front of.
    #[test]
    #[ignore]
    fn bench_frontend_cost() {
        let fs = 192_000.0f64;
        let n = (fs * 4.0) as usize;
        let mut rng = 0x2222_1111u32;
        let mut nz = || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            (rng as f32 / u32::MAX as f32) - 0.5
        };
        let iq: Vec<Complex32> = (0..n).map(|_| Complex32::new(nz(), nz())).collect();
        let mut fe = FrontEnd::new(fs);
        let t = std::time::Instant::now();
        for chunk in iq.chunks(16_384) {
            let mut buf = chunk.to_vec();
            fe.process(&mut buf);
        }
        let el = t.elapsed().as_secs_f64();
        println!(
            "  {:.1} ms for {:.1} s of 192 kHz IQ ({:.2}% of real time)",
            el * 1000.0,
            n as f64 / fs,
            el / (n as f64 / fs) * 100.0
        );
    }
}

#[cfg(test)]
mod window_bench {
    use super::*;

    fn noise(rng: &mut u32) -> f32 {
        *rng ^= *rng << 13;
        *rng ^= *rng >> 17;
        *rng ^= *rng << 5;
        (*rng as f32 / u32::MAX as f32) - 0.5
    }

    /// Peak height over the *local* floor, which is what `scout_peaks` works
    /// from: the median of the surrounding bins, excluding the peak's own
    /// skirts. Measuring against the whole spectrum's median instead misses
    /// the entire effect — one strong signal cannot move the median of 8192
    /// bins, but it can certainly raise the floor beside itself.
    fn peak_over_floor(spec: &[f32], fs: f32, hz: f32) -> f32 {
        const NEAR: isize = 3;
        const CTX: isize = 40;
        let n = spec.len() as isize;
        let bin = fs / n as f32;
        let centre = ((hz / bin) + n as f32 / 2.0).round() as isize;
        let at = |i: isize| spec[i.clamp(0, n - 1) as usize];
        let mut peak = f32::MIN;
        for d in -NEAR..=NEAR {
            peak = peak.max(at(centre + d));
        }
        let mut ctx: Vec<f32> = (-CTX..=CTX)
            .filter(|d| d.abs() > NEAR)
            .map(|d| at(centre + d))
            .collect();
        ctx.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        peak - ctx[ctx.len() / 2]
    }

    fn spectrum_of(iq: &[Complex32], win: Window, size: usize) -> Vec<f32> {
        let mut s = Spectrum::with_window(size, win);
        let mut out = Vec::new();
        s.power_db(iq, &mut out);
        out
    }

    /// Two competing effects, measured against each other: how far a weak
    /// narrowband signal stands out of the floor on its own, and how far it
    /// stands out with a strong station nearby leaking over it.
    #[test]
    #[ignore]
    fn bench_window_tradeoff() {
        let fs = 192_000.0f32;
        let n = 8192;
        let len = n * 8;
        let weak_hz = 2_000.0f32;

        println!("\n== a weak narrowband signal on its own ==");
        println!("{:>10}{:>12}{:>12}", "level", "Hann", "Blackman-H");
        for amp in [0.02f32, 0.01, 0.005] {
            let mut rng = 0x9111_2222u32;
            let iq: Vec<Complex32> = (0..len)
                .map(|i| {
                    let ph = 2.0 * PI * weak_hz * i as f32 / fs;
                    Complex32::from_polar(amp, ph)
                        + Complex32::new(noise(&mut rng), noise(&mut rng)) * 0.01
                })
                .collect();
            let h = peak_over_floor(&spectrum_of(&iq, Window::Hann, n), fs, weak_hz);
            let b =
                peak_over_floor(&spectrum_of(&iq, Window::BlackmanHarris, n), fs, weak_hz);
            println!(
                "{:>10}{:>12}{:>12}",
                format!("{amp:.3}"),
                format!("{h:.1} dB"),
                format!("{b:.1} dB")
            );
        }

        println!("\n== the same weak signal with a strong station nearby ==");
        println!("(weak signal at 0.005; strong one 80 dB above it)");
        println!("(under ~4 bins apart the two are not resolved at all, so");
        println!(" those rows measure the strong signal, not the weak one)");
        println!(
            "{:>10}{:>10}{:>12}{:>12}",
            "spacing", "bins", "Hann", "Blackman-H"
        );
        for sep in [25.0f32, 50.0, 100.0, 200.0, 500.0, 2_000.0] {
            let mut rng = 0x9111_2222u32;
            let iq: Vec<Complex32> = (0..len)
                .map(|i| {
                    let t = i as f32 / fs;
                    let weak = Complex32::from_polar(0.005, 2.0 * PI * weak_hz * t);
                    let strong =
                        Complex32::from_polar(50.0, 2.0 * PI * (weak_hz + sep) * t);
                    weak + strong + Complex32::new(noise(&mut rng), noise(&mut rng)) * 0.01
                })
                .collect();
            let h = peak_over_floor(&spectrum_of(&iq, Window::Hann, n), fs, weak_hz);
            let b =
                peak_over_floor(&spectrum_of(&iq, Window::BlackmanHarris, n), fs, weak_hz);
            println!(
                "{:>10}{:>10}{:>12}{:>12}",
                format!("{sep:.0} Hz"),
                format!("{:.1}", sep / (fs / n as f32)),
                format!("{h:.1} dB"),
                format!("{b:.1} dB")
            );
        }
    }
}

#[cfg(test)]
mod frontend_audit {
    use super::frontend_tests::*;
    use super::*;

    /// SNR of a weak tone measured against the broadband floor around it.
    fn tone_snr_db(iq: &[Complex32], fs: f32, hz: f32) -> f32 {
        let sig = power_at(iq, fs, hz);
        // Floor sampled well away from the tone and its image, avoiding DC.
        let probes = [hz + 3000.0, hz + 5000.0, hz - 4000.0, hz + 7000.0];
        let n: f32 = probes.iter().map(|&f| power_at(iq, fs, f)).sum::<f32>()
            / probes.len() as f32;
        10.0 * (sig / n.max(1e-30)).log10()
    }

    /// Does the front end cost a weak signal anything?
    ///
    /// Each stage is exercised on its own so a loss can be attributed. The
    /// image corrector is the one under suspicion: it round-trips every 4096
    /// samples through an FFT, edits bins, and inverse-transforms with no
    /// window and no overlap, which is a circular convolution and rings at
    /// the block seams.
    #[test]
    #[ignore]
    fn bench_frontend_weak_signal_cost() {
        let fs = 192_000.0f32;
        let hz = 12_000.0f32;
        for (label, gain_err, phase_err, dc) in [
            ("clean input      ", 1.0f32, 0.0f32, Complex32::new(0.0, 0.0)),
            ("1% gain imbalance", 1.01, 0.005, Complex32::new(0.0, 0.0)),
            ("5% gain imbalance", 1.05, 0.03, Complex32::new(0.01, -0.007)),
        ] {
            let mut rng = 0x1357_9bdfu32;
            let n = (fs * 4.0) as usize;
            // A weak tone in noise, then the receiver's imperfections on top.
            let raw: Vec<Complex32> = (0..n)
                .map(|i| {
                    let ph = 2.0 * PI * hz * i as f32 / fs;
                    let clean = Complex32::from_polar(0.004, ph)
                        + Complex32::new(noise(&mut rng), noise(&mut rng)) * 0.05;
                    let q = clean.im * gain_err + clean.re * phase_err;
                    Complex32::new(clean.re, q) + dc
                })
                .collect();
            let before = tone_snr_db(&raw[raw.len() * 3 / 4..], fs, hz);
            let after = tone_snr_db(&run_frontend(&raw, fs), fs, hz);
            println!(
                "  {label}: {before:6.2} dB in -> {after:6.2} dB out   ({:+.2} dB)",
                after - before
            );
        }
    }

    /// The blanker is on by default at "normal". On a band with no impulse
    /// noise it should be doing nothing at all.
    #[test]
    #[ignore]
    fn bench_blanker_on_quiet_band() {
        let fs = 192_000.0f32;
        let hz = 12_000.0f32;
        let mut rng = 0x2468_ace0u32;
        let n = (fs * 4.0) as usize;
        let raw: Vec<Complex32> = (0..n)
            .map(|i| {
                let ph = 2.0 * PI * hz * i as f32 / fs;
                Complex32::from_polar(0.004, ph)
                    + Complex32::new(noise(&mut rng), noise(&mut rng)) * 0.05
            })
            .collect();
        let mut nb = NoiseBlanker::new(fs as f64);
        let mut out = raw.clone();
        for chunk in out.chunks_mut(16_384) {
            nb.process(chunk);
        }
        let before = tone_snr_db(&raw[raw.len() * 3 / 4..], fs, hz);
        let after = tone_snr_db(&out[out.len() * 3 / 4..], fs, hz);
        println!(
            "  blanker 'normal' on clean noise: {before:6.2} -> {after:6.2} dB  ({:+.2} dB), {} blanks/s",
            after - before,
            nb.blanks_per_second()
        );
    }
}
