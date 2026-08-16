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
                        A[0] - A[1] * x.cos() + A[2] * (2.0 * x).cos() - A[3] * (3.0 * x).cos()
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
    acc: Vec<f32>,
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
            acc: vec![0.0f32; size],
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
    /// Returns whether a fresh estimate was produced. A large FFT needs more
    /// samples than one IQ block carries, so `out` is often left holding the
    /// previous estimate — and anything integrating this over time needs to
    /// know that, or it folds the same observation in several times over.
    pub fn power_db(&mut self, input: &[Complex32], out: &mut Vec<f32>) -> bool {
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
            return false;
        }
        out.clear();
        out.resize(self.size, 0.0);
        self.acc.fill(0.0);
        for s in 0..nseg {
            let start = s * hop;
            let seg = &self.pending[start..start + self.size];
            for i in 0..self.size {
                self.buf[i] = seg[i] * self.window[i];
            }
            self.fft.process(&mut self.buf);
            for i in 0..self.size {
                self.acc[i] += self.buf[i].norm_sqr();
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
            out[i] = 10.0 * (self.acc[src] * scale + 1e-20).log10();
        }
        true
    }
}

/// Per-bin minimum-statistics noise estimate. Falls promptly when a quiet
/// observation arrives but rises slowly enough that traffic cannot become its
/// own reference level.
pub struct NoiseFloor {
    /// Raw minimum-following estimate, which sits below the noise level.
    bins: Vec<f32>,
    /// ...and the same thing with the measured bias put back.
    corrected: Vec<f32>,
    bias: f32,
    seen: bool,
}

/// How fast the estimate follows the spectrum down, and back up.
///
/// In seconds, deliberately. These used to be per-*call* blend coefficients,
/// which made the time constant a function of how often `feed` happened to
/// run — and that is set by the sample rate and the block size, so the same
/// numbers meant 43 s of upward memory on a 192 kS/s span and 21 s on a
/// 384 kS/s one. A detection parameter must not change because the operator
/// pressed `b`.
const TAU_DOWN_S: f32 = 0.30;
const TAU_UP_S: f32 = 43.0;

/// How quickly the measured bias correction follows.
const TAU_BIAS_S: f32 = 8.0;

impl NoiseFloor {
    pub fn new() -> Self {
        Self {
            bins: Vec::new(),
            corrected: Vec::new(),
            bias: 0.0,
            seen: false,
        }
    }

    /// Fold in one fresh periodogram covering `dt` seconds.
    ///
    /// Returns the *bias-corrected* floor. An estimator that follows minima
    /// quickly and maxima slowly necessarily settles below the mean of what it
    /// watches — that is what makes it immune to signals, and it is also why
    /// the raw number is not a noise floor. The offset is not a constant: it
    /// grows as the periodogram gets noisier, and the periodogram gets noisier
    /// as the FFT gets larger, because fewer Welch segments fit in a block.
    /// Measured, it runs from about 11 dB at 4096 points to 19 dB at 32768.
    ///
    /// So it is measured rather than assumed. The median across bins of
    /// `observation - floor` is that offset directly: signals occupy few bins
    /// of a span, so the median is still describing noise even on a busy band,
    /// which is the same reasoning the rest of the detection code uses medians
    /// for. That makes the correction self-calibrating across FFT size, sample
    /// rate and block size at once, with no constant to keep in step.
    pub fn update<'a>(&'a mut self, power_db: &[f32], dt: f32) -> &'a [f32] {
        if self.bins.len() != power_db.len() {
            self.bins = power_db.to_vec();
            self.corrected = power_db.to_vec();
            self.bias = 0.0;
            self.seen = false;
            return &self.corrected;
        }
        let dt = dt.clamp(1e-4, 5.0);
        let a_down = 1.0 - (-dt / TAU_DOWN_S).exp();
        let a_up = 1.0 - (-dt / TAU_UP_S).exp();
        for (floor, &x) in self.bins.iter_mut().zip(power_db) {
            let a = if x < *floor { a_down } else { a_up };
            *floor += a * (x - *floor);
        }

        // Subsampled median of the gap, for the same reason `sampled_median`
        // subsamples: a full sort of every bin, every update, buys nothing.
        let mut gaps: Vec<f32> = power_db
            .iter()
            .zip(&self.bins)
            .step_by(7)
            .map(|(x, f)| x - f)
            .collect();
        if !gaps.is_empty() {
            let mid = gaps.len() / 2;
            gaps.select_nth_unstable_by(mid, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            let inst = gaps[mid].max(0.0);
            if !self.seen {
                self.bias = inst;
                self.seen = true;
            } else {
                let a = 1.0 - (-dt / TAU_BIAS_S).exp();
                self.bias += a * (inst - self.bias);
            }
        }

        self.corrected.clear();
        self.corrected
            .extend(self.bins.iter().map(|f| f + self.bias));
        &self.bins
    }

    /// The same estimate with the measured bias put back — an actual noise
    /// level in dB, for the cursor SNR readout and the waterfall's colour
    /// scale.
    ///
    /// Detection deliberately does *not* use this. `scout_peaks` compares a
    /// peak against `max(floor, local_median - 3)`, and with the raw estimate
    /// sitting well below the noise that clamp is what wins, which puts the
    /// candidate gate at roughly zero prominence — permissive on purpose,
    /// because the scouts' mix-down-and-match stage is the real false-alarm
    /// filter and a candidate costs almost nothing. Feeding the corrected
    /// level in instead raises the bar by the whole bias and loses weak
    /// signals the scouts would have confirmed. The two callers want
    /// genuinely different things: one wants a level, the other wants a
    /// reference nothing on the air can pull upwards.
    pub fn level(&self) -> &[f32] {
        &self.corrected
    }
}

/// Frequency-domain smooth. `taps` 1 = none, 3 = [1,2,1], 5 = [1,4,6,4,1].
/// Local maxima are kept so carriers stay sharp while the floor calms down.
pub fn smooth_bins(src: &[f32], taps: usize, out: &mut Vec<f32>) {
    out.clear();
    out.resize(src.len(), -140.0);
    let len = src.len();
    if len == 0 {
        return;
    }
    if taps <= 1 || len < 3 {
        out.copy_from_slice(src);
        return;
    }
    if taps >= 5 && len >= 5 {
        out[0] = (6.0 * src[0] + 4.0 * src[1] + src[2]) / 11.0;
        out[1] = (4.0 * src[0] + 6.0 * src[1] + 4.0 * src[2] + src[3]) / 15.0;
        out[len - 2] =
            (src[len - 4] + 4.0 * src[len - 3] + 6.0 * src[len - 2] + 4.0 * src[len - 1]) / 15.0;
        out[len - 1] = (src[len - 3] + 4.0 * src[len - 2] + 6.0 * src[len - 1]) / 11.0;
        const NORM5: f32 = 1.0 / 16.0;
        for i in 2..len - 2 {
            out[i] = (src[i - 2] + 4.0 * src[i - 1] + 6.0 * src[i] + 4.0 * src[i + 1] + src[i + 2])
                * NORM5;
        }
    } else {
        out[0] = (2.0 * src[0] + src[1]) / 3.0;
        out[len - 1] = (src[len - 2] + 2.0 * src[len - 1]) / 3.0;
        const NORM3: f32 = 0.25;
        for i in 1..len - 1 {
            out[i] = (src[i - 1] + 2.0 * src[i] + src[i + 1]) * NORM3;
        }
    }
    for i in 1..len - 1 {
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

    /// Change the rate without disturbing the phase.
    ///
    /// A decoder tracking a drifting signal retunes its mixer constantly, and
    /// building a fresh `Rotator` for each new rate would restart it at 1+0j —
    /// a phase step in the middle of the very signal the AFC is trying to hold.
    pub(crate) fn set_rate(&mut self, rad_per_sample: f32) {
        let (sin, cos) = rad_per_sample.sin_cos();
        self.step = Complex32::new(cos, sin);
    }

    /// Restart the phase at zero, for a genuine change of signal.
    pub(crate) fn reset_phase(&mut self) {
        self.cur = Complex32::new(1.0, 0.0);
        self.n = 0;
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
#[allow(dead_code)]
pub fn mix_decim(iq: &[Complex32], fs: f32, hz: f32, decim: usize) -> Vec<Complex32> {
    let mut out = Vec::with_capacity(iq.len() / decim.max(1) + 1);
    mix_decim_into(iq, fs, hz, decim, &mut out);
    out
}

/// Zero-allocation variant of `mix_decim` that reuses `out`.
pub fn mix_decim_into(iq: &[Complex32], fs: f32, hz: f32, decim: usize, out: &mut Vec<Complex32>) {
    out.clear();
    let mut osc = Rotator::new(-2.0 * PI * hz / fs);
    if decim <= 1 {
        for &s in iq {
            out.push(s * osc.next());
        }
        return;
    }
    let d = decim as f32;
    let inv_d = 1.0 / d;
    let norm = 1.0 / (d + 1.0);
    let mut weights_fall = Vec::with_capacity(decim);
    let mut weights_rise = Vec::with_capacity(decim);
    for r in 0..decim {
        weights_fall.push(1.0 - r as f32 * inv_d);
        weights_rise.push((r + 1) as f32 * inv_d);
    }
    let mut fall = Complex32::new(0.0, 0.0);
    let mut rise = Complex32::new(0.0, 0.0);
    let mut r = 0usize;
    for &s in iq {
        let m = s * osc.next();
        fall += m * weights_fall[r];
        rise += m * weights_rise[r];
        r += 1;
        if r == decim {
            out.push(fall * norm);
            fall = rise;
            rise = Complex32::new(0.0, 0.0);
            r = 0;
        }
    }
}

/// Numerically controlled oscillator used to shift a signal of interest to 0 Hz.
pub struct Nco {
    cur: Complex32,
    step: Complex32,
    n: u32,
}

impl Nco {
    pub fn new() -> Self {
        Self {
            cur: Complex32::new(1.0, 0.0),
            step: Complex32::new(1.0, 0.0),
            n: 0,
        }
    }

    pub fn set_freq(&mut self, hz: f64, fs: f64) {
        let step_rad = -2.0 * std::f64::consts::PI * hz / fs;
        let (sin, cos) = (step_rad as f32).sin_cos();
        self.step = Complex32::new(cos, sin);
    }

    pub fn mix(&mut self, input: &[Complex32], out: &mut Vec<Complex32>) {
        out.clear();
        out.reserve(input.len());
        for &s in input {
            out.push(s * self.cur);
            self.cur *= self.step;
            self.n += 1;
            if self.n >= 1024 {
                self.n = 0;
                let k = 1.5 - 0.5 * self.cur.norm_sqr();
                self.cur *= k;
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
            fft: None,
            ifft: None,
            response: Vec::new(),
            work: Vec::new(),
            overlap: Vec::new(),
            fft_len: 0,
            fft_phase: 0,
            fft_warm: 0,
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
        for (dst, &tap) in self.response.iter_mut().zip(&self.taps) {
            dst.re = tap;
        }
        fft.process(&mut self.response);
        self.work.resize(self.fft_len, Complex32::new(0.0, 0.0));
        self.overlap
            .resize(self.taps.len() - 1, Complex32::new(0.0, 0.0));
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
            for (x, h) in self.work.iter_mut().zip(&self.response) {
                *x *= *h;
            }
            self.ifft.as_ref().unwrap().process(&mut self.work);
            let scale = 1.0 / self.fft_len as f32;
            for x in &self.work[discard..discard + hop] {
                if self.fft_warm > 0 {
                    self.fft_warm -= 1;
                    continue;
                }
                if self.fft_phase == 0 {
                    out.push(*x * scale);
                }
                self.fft_phase += 1;
                if self.fft_phase == self.decim {
                    self.fft_phase = 0;
                }
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

/// Frame size the channeliser transforms.
///
/// Divisible by 24 and 16 — the decimations a 192 kS/s span uses for its
/// 8 kHz and 12 kHz audio rates — and by 48 and 32 for a 384 kS/s one. That
/// divisibility is what lets a tap decimate by folding the spectrum instead
/// of inverse-transforming the whole frame and throwing away most of it.
const CHAN_N: usize = 3072;

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// One forward transform of the wideband stream, shared by every decoder.
///
/// Each `DecodeChain` mixes and filters the *whole* span independently: an
/// NCO over every input sample, then an overlap-save FIR whose forward FFT is
/// recomputed from scratch. Twenty-four slots therefore ran twenty-four
/// identical 4096-point transforms of the same input, having first spent
/// twenty-four passes of complex multiplies arriving at inputs that differ
/// only by a frequency shift.
///
/// A frequency shift is a rotation of the spectrum, so all of that is one
/// transform and a change of index. The channeliser does it once; each
/// `ChannelTap` picks up the shared spectrum, rotates it to its own centre,
/// multiplies by its filter and inverse-transforms.
pub struct Channelizer {
    fft: Arc<dyn Fft<f32>>,
    buf: Vec<Complex32>,
    /// Forward transforms of every frame produced by the last `push`, laid
    /// end to end.
    spectra: Vec<Complex32>,
    /// Absolute index of the first sample of each frame.
    starts: Vec<u64>,
    next_start: u64,
    hop: usize,
}

impl Channelizer {
    pub fn new(fs_in: f64) -> Self {
        // Frames advance by a whole number of output samples for *every*
        // audio rate in use. A tap decimating by D produces samples at frame
        // offsets 0, D, 2D…; if the hop were not a multiple of D the
        // decimation phase would shift from frame to frame and the output
        // would stop being a uniformly sampled signal.
        let d8 = (fs_in / 8_000.0).round().max(1.0) as usize;
        let d12 = (fs_in / 12_000.0).round().max(1.0) as usize;
        let align = d8 / gcd(d8, d12) * d12;
        let max_hop = CHAN_N - (RADIO_TAPS - 1);
        let hop = (max_hop / align.max(1)).max(1) * align.max(1);
        Self {
            fft: FftPlanner::new().plan_fft_forward(CHAN_N),
            buf: Vec::with_capacity(CHAN_N * 2),
            spectra: Vec::new(),
            starts: Vec::new(),
            next_start: 0,
            hop: hop.min(max_hop).max(1),
        }
    }

    pub fn hop(&self) -> usize {
        self.hop
    }

    pub fn reset(&mut self) {
        self.buf.clear();
        self.spectra.clear();
        self.starts.clear();
        self.next_start = 0;
    }

    /// Transform every whole frame now available. Frames overlap by the
    /// filter length, as overlap-save requires.
    pub fn push(&mut self, input: &[Complex32]) -> usize {
        self.buf.extend_from_slice(input);
        self.spectra.clear();
        self.starts.clear();
        while self.buf.len() >= CHAN_N {
            let at = self.spectra.len();
            self.spectra.extend_from_slice(&self.buf[..CHAN_N]);
            self.fft.process(&mut self.spectra[at..at + CHAN_N]);
            self.starts.push(self.next_start);
            self.buf.drain(..self.hop);
            self.next_start += self.hop as u64;
        }
        self.starts.len()
    }

    pub fn frame(&self, i: usize) -> (&[Complex32], u64) {
        (&self.spectra[i * CHAN_N..(i + 1) * CHAN_N], self.starts[i])
    }
}

/// One decoder's view of the shared spectrum: an NCO and both filter stages,
/// producing the same narrowband baseband a `DecodeChain` would.
pub struct ChannelTap {
    ifft: Arc<dyn Fft<f32>>,
    response: Vec<Complex32>,
    /// Bins where the filter is not effectively zero. A 400 Hz channel is a
    /// hundred-odd bins of a 3072-point frame, so the fold below only has to
    /// touch those — everywhere else it would be adding zeros.
    active: Vec<usize>,
    /// Folded spectrum, `CHAN_N / decim` long: decimation done by aliasing
    /// the (already filtered) spectrum rather than by inverse-transforming
    /// the whole frame and discarding all but every Dth sample.
    folded: Vec<Complex32>,
    decimated: Vec<Complex32>,
    audio_buf: Vec<Complex32>,
    audio: DecimFir,
    fs_in: f64,
    fs_out: f64,
    decim: usize,
    m: usize,
    hop: usize,
    /// Whether decimation is done by folding the spectrum (fast) or by
    /// inverse-transforming the frame and keeping every Dth sample.
    fold: bool,
    dec_phase: usize,
    /// Whole bins of the wanted shift, and the sub-bin remainder.
    k0: isize,
    res_rot: Rotator,
    /// Output samples still to swallow while the filter fills, so the tap
    /// lines up with what a `DecodeChain` would have produced.
    warm: usize,
}

impl ChannelTap {
    pub fn new(fs_in: f64, bandwidth: f32, target_rate: f64, hop: usize) -> Self {
        let decim = (fs_in / target_rate).round().max(1.0) as usize;
        let fs_out = fs_in / decim as f64;
        // Folding needs the frame and the hop to be whole numbers of output
        // samples. Every rate in the band table obliges; a `--rate` override
        // need not, and then the tap inverse-transforms the whole frame and
        // decimates in time instead. Slower, still shares the forward
        // transform, and always available.
        let fold = CHAN_N.is_multiple_of(decim) && hop.is_multiple_of(decim);
        let m = if fold { CHAN_N / decim } else { CHAN_N };
        let mut tap = Self {
            ifft: FftPlanner::new().plan_fft_inverse(m),
            fold,
            dec_phase: 0,
            response: vec![Complex32::new(0.0, 0.0); CHAN_N],
            active: Vec::new(),
            folded: vec![Complex32::new(0.0, 0.0); m],
            decimated: Vec::new(),
            audio_buf: Vec::new(),
            audio: DecimFir::new(bandwidth / 2.0, fs_out as f32, 1, AUDIO_TAPS_MIN),
            fs_in,
            fs_out,
            decim,
            m,
            hop,
            k0: 0,
            res_rot: Rotator::new(0.0),
            warm: if fold {
                (RADIO_TAPS - 1) / decim
            } else {
                RADIO_TAPS - 1
            },
        };
        tap.set_bandwidth(bandwidth);
        tap
    }

    pub fn fs_out(&self) -> f64 {
        self.fs_out
    }

    /// Split the wanted shift into whole bins — free, because rotating the
    /// shared spectrum is only a change of index — and the remainder, which
    /// is applied as an ordinary rotation on the output.
    pub fn set_offset(&mut self, hz: f64) {
        let bin = self.fs_in / CHAN_N as f64;
        self.k0 = (hz / bin).round() as isize;
        let residual = hz - self.k0 as f64 * bin;
        // Advanced once per sample the inverse transform yields — which is an
        // output sample when folding and an input-rate sample when not.
        let per = if self.fold { self.fs_out } else { self.fs_in };
        self.res_rot
            .set_rate(-2.0 * std::f64::consts::PI as f32 * residual as f32 / per as f32);
    }

    pub fn set_bandwidth(&mut self, bw: f32) {
        let fs_out = self.fs_out as f32;
        let half = (bw * 0.5).clamp(10.0, fs_out * 0.45);
        // Identical sizing to `DecodeChain::set_bandwidth`; the two have to
        // realise the same filter or a slot would change character depending
        // on which one happened to drive it.
        let tr = transition_hz(self.fs_in as f32, RADIO_TAPS);
        let lo = half + tr * 0.5;
        let hi = (fs_out - half - tr * 0.5).max(lo);
        let radio_cut = (fs_out * 0.5).clamp(lo, hi);
        let taps = lowpass_taps(radio_cut, self.fs_in as f32, RADIO_TAPS);
        self.response
            .iter_mut()
            .for_each(|v| *v = Complex32::new(0.0, 0.0));
        for (dst, &t) in self.response.iter_mut().zip(&taps) {
            dst.re = t;
        }
        let fft = FftPlanner::new().plan_fft_forward(CHAN_N);
        fft.process(&mut self.response);
        // Everything the filter has already removed contributes nothing to
        // the fold. 100 dB below the passband is far past what the taps'
        // own stopband reaches.
        let peak = self
            .response
            .iter()
            .map(|v| v.norm())
            .fold(0.0f32, f32::max);
        let floor = peak * 1e-5;
        self.active.clear();
        self.active
            .extend((0..CHAN_N).filter(|&k| self.response[k].norm() > floor));

        let want_tr = (0.03 * bw).max(0.01 * fs_out);
        let ntaps = taps_for_transition(fs_out, want_tr, AUDIO_TAPS_MIN, AUDIO_TAPS_MAX);
        self.audio.set_taps(half, fs_out, ntaps);
    }

    /// Filter one shared frame down to this tap's channel, appending audio.
    pub fn process_frame(&mut self, spec: &[Complex32], start: u64, out: &mut Vec<Complex32>) {
        let n = CHAN_N;
        // Rotating the shared spectrum by k0 bins *is* the frequency shift,
        // and folding it onto `m` bins is the decimation: sample the result
        // every D samples and the bins alias exactly this way. Both together
        // reduce a tap to one pass over the hundred-odd bins its filter
        // actually passes, plus a short inverse transform.
        let k0 = self.k0.rem_euclid(n as isize) as usize;
        self.folded
            .iter_mut()
            .for_each(|v| *v = Complex32::new(0.0, 0.0));
        if self.fold {
            for &k in &self.active {
                let src = k + k0;
                let src = if src >= n { src - n } else { src };
                self.folded[k % self.m] += spec[src] * self.response[k];
            }
        } else {
            for &k in &self.active {
                let src = k + k0;
                let src = if src >= n { src - n } else { src };
                self.folded[k] = spec[src] * self.response[k];
            }
        }
        self.ifft.process(&mut self.folded);

        // The rotation was taken about the frame's own origin, but the signal
        // does not restart at each frame: the shift has to be referred to
        // absolute time, which is one phase per frame.
        let ang = -2.0
            * std::f32::consts::PI
            * (self.k0 as f64 * (start % n as u64) as f64 / n as f64) as f32;
        let block_phase = Complex32::from_polar(1.0 / n as f32, ang);

        // Overlap-save: only the tail of the frame is free of wrap-around.
        self.decimated.clear();
        if self.fold {
            let first = (n - self.hop) / self.decim;
            for r in first..self.m {
                let v = self.folded[r] * block_phase * self.res_rot.next();
                if self.warm > 0 {
                    self.warm -= 1;
                    continue;
                }
                self.decimated.push(v);
            }
        } else {
            for i in (n - self.hop)..n {
                let v = self.folded[i] * block_phase * self.res_rot.next();
                if self.warm > 0 {
                    self.warm -= 1;
                    continue;
                }
                if self.dec_phase == 0 {
                    self.decimated.push(v);
                }
                self.dec_phase += 1;
                if self.dec_phase == self.decim {
                    self.dec_phase = 0;
                }
            }
        }
        let decimated = std::mem::take(&mut self.decimated);
        self.audio.process(&decimated, &mut self.audio_buf);
        self.decimated = decimated;
        out.extend_from_slice(&self.audio_buf);
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

/// Audio-domain spectral subtraction and Wiener filtering (Ephraim-Malah /
/// decision-directed a priori SNR estimation) to reduce background hiss.
#[allow(dead_code)]
pub struct AudioNr {
    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    noise_psd: Vec<f32>,
    prev_mag: Vec<f32>,
    in_buf: Vec<Complex32>,
    out_buf: Vec<Complex32>,
    fft_buf: Vec<Complex32>,
    overlap: Vec<Complex32>,
    size: usize,
    hop: usize,
    enabled: bool,
    trained: usize,
}

#[allow(dead_code)]
impl AudioNr {
    pub fn new(size: usize) -> Self {
        let size = size.next_power_of_two().max(128);
        let hop = size / 2;
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(size);
        let ifft = planner.plan_fft_inverse(size);
        let window: Vec<f32> = (0..size)
            .map(|i| (PI * (i as f32 + 0.5) / size as f32).sin())
            .collect();
        Self {
            fft,
            ifft,
            window,
            noise_psd: vec![1e-6; size],
            prev_mag: vec![0.0; size],
            in_buf: Vec::with_capacity(size * 4),
            out_buf: Vec::with_capacity(size * 4),
            fft_buf: vec![Complex32::new(0.0, 0.0); size],
            overlap: vec![Complex32::new(0.0, 0.0); hop],
            size,
            hop,
            enabled: true,
            trained: 0,
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn reset(&mut self) {
        self.noise_psd.fill(1e-6);
        self.prev_mag.fill(0.0);
        self.in_buf.clear();
        self.out_buf.clear();
        self.overlap.fill(Complex32::new(0.0, 0.0));
        self.trained = 0;
    }

    pub fn process(&mut self, input: &mut [Complex32]) {
        if !self.enabled || input.is_empty() {
            return;
        }
        self.in_buf.extend_from_slice(input);

        while self.in_buf.len() >= self.size {
            for i in 0..self.size {
                self.fft_buf[i] = self.in_buf[i] * self.window[i];
            }
            self.fft.process(&mut self.fft_buf);

            let powers: Vec<f32> = self.fft_buf.iter().map(|c| c.norm_sqr()).collect();
            let mut sorted = powers.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let med_pwr = sorted[sorted.len() / 2].max(1e-12);

            // Estimate a posteriori & decision-directed a priori SNR
            const ALPHA: f32 = 0.94;
            const FLOOR_GAIN: f32 = 0.15; // Max 16 dB suppression
            for i in 0..self.size {
                let p = powers[i];
                let noise_ref = p.min(med_pwr * 2.0);
                if self.trained < 16 {
                    self.noise_psd[i] = (self.noise_psd[i] * self.trained as f32 + noise_ref)
                        / (self.trained + 1) as f32;
                } else {
                    self.noise_psd[i] += (noise_ref - self.noise_psd[i]) * 0.05;
                }
                let gamma = p / self.noise_psd[i].max(1e-12);
                let xi = (ALPHA * self.prev_mag[i] / self.noise_psd[i].max(1e-12)
                    + (1.0 - ALPHA) * (gamma - 1.0).max(0.0))
                .max(1e-4);
                let gain = (xi / (1.0 + xi)).clamp(FLOOR_GAIN, 1.0);
                self.fft_buf[i] *= gain;
                self.prev_mag[i] = self.fft_buf[i].norm_sqr();
            }
            self.trained = self.trained.saturating_add(1);

            self.ifft.process(&mut self.fft_buf);
            let norm = 1.0 / (self.size as f32);
            for i in 0..self.hop {
                let s = self.overlap[i] + self.fft_buf[i] * self.window[i] * norm;
                self.out_buf.push(s);
            }
            for i in 0..self.hop {
                self.overlap[i] = self.fft_buf[self.hop + i] * self.window[self.hop + i] * norm;
            }
            self.in_buf.drain(..self.hop);
        }

        let n = self.out_buf.len().min(input.len());
        if n > 0 {
            input[..n].copy_from_slice(&self.out_buf[..n]);
            self.out_buf.drain(..n);
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

    /// Run pure noise through `Spectrum` into `NoiseFloor` at a given rate
    /// and block size, and return (settled floor, true mean) in dB.
    fn settled_floor(fs: f64, block: usize, fft: usize, secs: f64) -> (f32, f32) {
        let mut rng = 0x5eed_1234u32;
        let n = (fs * secs) as usize;
        let iq: Vec<Complex32> = (0..n)
            .map(|_| {
                Complex32::new(
                    frontend_tests::noise(&mut rng),
                    frontend_tests::noise(&mut rng),
                )
            })
            .collect();
        let mut spec = Spectrum::new(fft);
        let mut nf = NoiseFloor::new();
        let mut power = Vec::new();
        let mut dt = 0.0f32;
        let mut floor_mean = 0.0f32;
        // One periodogram is a single noisy draw per bin — at a large FFT
        // only one Welch segment fits in a block — so the reference is
        // averaged over the settled tail rather than read off the last one.
        let mut ref_sum = 0.0f64;
        let mut ref_n = 0usize;
        let total = iq.len() / block;
        for (b, chunk) in iq.chunks(block).enumerate() {
            let fresh = spec.power_db(chunk, &mut power);
            dt += chunk.len() as f32 / fs as f32;
            if fresh {
                nf.update(&power, dt);
                dt = 0.0;
                let f = nf.level();
                floor_mean = f.iter().sum::<f32>() / f.len() as f32;
                if b * 3 >= total * 2 {
                    ref_sum += (power.iter().sum::<f32>() / power.len() as f32) as f64;
                    ref_n += 1;
                }
            }
        }
        (floor_mean, (ref_sum / ref_n.max(1) as f64) as f32)
    }

    /// The tracker's memory must be a length of time, not a number of calls.
    ///
    /// These coefficients used to be applied per `feed`, and `feed` runs once
    /// per IQ block — so doubling the sample rate halved every time constant
    /// and moved a detection threshold that nothing had asked to move.
    #[test]
    fn noise_floor_settles_the_same_at_any_sample_rate() {
        let (a, _) = settled_floor(192_000.0, 16_384, 8192, 60.0);
        let (b, _) = settled_floor(384_000.0, 16_384, 8192, 60.0);
        assert!(
            (a - b).abs() < 0.5,
            "floor settled at {a:.2} dB at 192 kS/s but {b:.2} dB at 384 kS/s"
        );
    }

    /// And once the bias is put back, it must read as the noise level rather
    /// than as some number below it.
    #[test]
    fn noise_floor_reads_the_actual_noise_level() {
        let (floor, mean) = settled_floor(192_000.0, 16_384, 8192, 60.0);
        assert!(
            (floor - mean).abs() < 2.0,
            "floor settled {:.2} dB from the true noise level ({floor:.2} vs {mean:.2}); \
             the bias correction is not doing its job",
            floor - mean
        );
    }

    #[test]
    #[ignore]
    fn bench_noise_floor_bias() {
        for (fs, label) in [(192_000.0f64, "192 kS/s"), (384_000.0, "384 kS/s")] {
            for fft in [4096usize, 8192, 32768] {
                let (floor, mean) = settled_floor(fs, 16_384, fft, 60.0);
                println!(
                    "  {label}, {fft}-point: floor {floor:6.2} dB, mean {mean:6.2} dB, residual {:+.2} dB",
                    floor - mean
                );
            }
        }
    }

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
        let mut hot: Vec<Complex32> = (0..800).map(|_| Complex32::new(0.8, 0.0)).collect();
        agc.process(&mut hot);
        let peak = hot.iter().map(|c| c.norm()).fold(0.0f32, f32::max);
        assert!(peak < 0.6, "hot block should be ducked, peak {peak}");

        // After hang, a quiet block must not instantly slam the gain up.
        let gain_after_hot = agc.gain();
        let mut quiet: Vec<Complex32> = (0..800).map(|_| Complex32::new(0.01, 0.0)).collect();
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
    fn audio_nr_reduces_noise_floor_and_preserves_tone() {
        let mut nr = AudioNr::new(256);
        let fs = 8000.0f32;
        let mut rng = 0x1234_5678u32;
        let mut nz = || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            (rng as f32 / u32::MAX as f32) - 0.5
        };
        // 1 second of noise + tone at 700 Hz
        let n = 8000;
        let tone_hz = 700.0f32;
        let raw: Vec<Complex32> = (0..n)
            .map(|i| {
                let s = Complex32::from_polar(0.5, 2.0 * PI * tone_hz * i as f32 / fs);
                let n = Complex32::new(nz(), nz()) * 0.1;
                s + n
            })
            .collect();
        let mut proc = raw.clone();
        for chunk in proc.chunks_mut(512) {
            nr.process(chunk);
        }
        let tone_pwr_in = frontend_tests::power_at(&raw[4000..], fs, tone_hz);
        let tone_pwr_out = frontend_tests::power_at(&proc[4000..], fs, tone_hz);
        assert!(
            (tone_pwr_out / tone_pwr_in - 1.0).abs() < 0.20,
            "tone altered significantly: {tone_pwr_in} -> {tone_pwr_out}"
        );
    }

    /// A tap has to be the chain it replaces.
    ///
    /// The whole point is that twenty-four decoders can share one transform,
    /// which is only worth anything if what each of them gets out is what it
    /// got before. A tone somewhere in the span, tuned by both paths, has to
    /// come out at the same frequency and the same amplitude.
    #[test]
    fn a_channel_tap_matches_the_chain_it_replaces() {
        let fs = 192_000.0f64;
        let offset = 37_500.0f64;
        let n = 400_000usize;
        let mut rng = 0x77aa_1133u32;
        // A tone at the tuned offset, plus noise and a strong neighbour that
        // both filters have to reject.
        let iq: Vec<Complex32> = (0..n)
            .map(|i| {
                let t = i as f32 / fs as f32;
                Complex32::from_polar(0.3, 2.0 * PI * offset as f32 * t)
                    + Complex32::from_polar(3.0, 2.0 * PI * (offset as f32 + 9000.0) * t)
                    + Complex32::new(
                        frontend_tests::noise(&mut rng),
                        frontend_tests::noise(&mut rng),
                    ) * 0.02
            })
            .collect();

        let mut chain = DecodeChain::new(fs, 400.0, 8000.0);
        chain.set_offset(offset);
        let mut chain_out = Vec::new();
        let mut scratch = Vec::new();
        for c in iq.chunks(16_384) {
            chain.process(c, &mut scratch);
            chain_out.extend_from_slice(&scratch);
        }

        let mut ch = Channelizer::new(fs);
        let mut tap = ChannelTap::new(fs, 400.0, 8000.0, ch.hop());
        tap.set_offset(offset);
        let mut tap_out = Vec::new();
        for c in iq.chunks(16_384) {
            let frames = ch.push(c);
            for f in 0..frames {
                let (spec, start) = ch.frame(f);
                tap.process_frame(spec, start, &mut tap_out);
            }
        }

        // The two buffer internally at different block sizes, so they can
        // finish up to one audio-FIR block apart; what matters is that the
        // tap keeps up with the stream rather than quietly losing samples.
        assert!(
            tap_out.len() as f32 > chain_out.len() as f32 * 0.9,
            "tap emitted {} samples against the chain's {}",
            tap_out.len(),
            chain_out.len()
        );
        // The tuned tone must land at DC in both, at the same level.
        let take = tap_out.len().min(chain_out.len());
        let lo = take / 3;
        let p_chain = frontend_tests::power_at(&chain_out[lo..take], 8000.0, 0.0);
        let p_tap = frontend_tests::power_at(&tap_out[lo..take], 8000.0, 0.0);
        let db = 10.0 * (p_tap / p_chain.max(1e-30)).log10();
        assert!(
            db.abs() < 0.5,
            "tap delivered the tuned tone {db:+.2} dB against the chain ({p_chain:.6} vs {p_tap:.6})"
        );
        // ...and the strong neighbour must be just as absent from both.
        let n_chain = frontend_tests::power_at(&chain_out[lo..take], 8000.0, 1000.0);
        let n_tap = frontend_tests::power_at(&tap_out[lo..take], 8000.0, 1000.0);
        assert!(
            n_tap <= n_chain * 4.0 + 1e-12,
            "tap let through more of the neighbour than the chain ({n_chain:.3e} vs {n_tap:.3e})"
        );
    }

    #[test]
    fn overlap_save_matches_direct_decimation() {
        let input: Vec<_> = (0..20_000)
            .map(|i| Complex32::new((i as f32 * 0.013).sin(), (i as f32 * 0.021).cos()))
            .collect();
        let mut fast = DecimFir::new(3500.0, 192_000.0, 24, 4095);
        let taps = fast.taps.clone();
        let mut got = Vec::new();
        for block in input.chunks(3000) {
            let mut part = Vec::new();
            fast.process(block, &mut part);
            got.extend(part);
        }
        let mut want = Vec::new();
        for i in (0..=input.len() - taps.len()).step_by(24) {
            want.push(
                input[i..i + taps.len()]
                    .iter()
                    .zip(&taps)
                    .map(|(x, h)| *x * *h)
                    .sum::<Complex32>(),
            );
        }
        let err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (*a - *b).norm())
            .fold(0.0f32, f32::max);
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
                    if phase > std::f64::consts::PI {
                        phase -= 2.0 * std::f64::consts::PI;
                    }
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
        assert!(
            fold < -30.0,
            "below the dial should be rejected, got {fold:.1} dB"
        );
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
            assert!(r < -80.0, "{hz:.0} Hz folds into the channel at {r:.1} dB");
        }
    }

    #[test]
    #[ignore]
    fn bench_fast_convolution() {
        use std::time::Instant;
        let input: Vec<_> = (0..96_000)
            .map(|i| Complex32::new((i as f32 * 0.017).sin(), 0.0))
            .collect();
        let taps = lowpass_taps(100.0, 8000.0, 1101);
        let at = Instant::now();
        let mut direct = Vec::with_capacity(input.len());
        for i in 0..input.len() - taps.len() {
            direct.push(
                input[i..i + taps.len()]
                    .iter()
                    .zip(&taps)
                    .map(|(x, h)| *x * *h)
                    .sum::<Complex32>(),
            );
        }
        let direct_ms = at.elapsed().as_secs_f64() * 1000.0;
        let mut fir = DecimFir::new(100.0, 8000.0, 1, 1101);
        let at = Instant::now();
        let mut out = Vec::new();
        for block in input.chunks(4096) {
            fir.process(block, &mut out);
        }
        let fft_ms = at.elapsed().as_secs_f64() * 1000.0;
        println!(
            "1101-tap convolution: direct {direct_ms:.1} ms, overlap-save {fft_ms:.1} ms ({:.1}x faster)",
            direct_ms / fft_ms
        );
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
    window_seen: usize,
    window_blanked: usize,
    window_triggers: usize,
    last_rate: usize,
    inhibited: bool,
    trained: usize,
    /// Which samples of the current block are to be replaced.
    mask: Vec<bool>,
    /// Last sample that survived, so a run reaching a block edge has
    /// something to interpolate from.
    last_good: Complex32,
}

impl NoiseBlanker {
    pub fn new(fs: f64) -> Self {
        Self {
            fs: fs as f32,
            background: 1e-3,
            level: 2,
            window_seen: 0,
            window_blanked: 0,
            window_triggers: 0,
            last_rate: 0,
            inhibited: false,
            trained: 0,
            mask: Vec::new(),
            last_good: Complex32::new(0.0, 0.0),
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
        let n = iq.len();
        if n == 0 {
            return;
        }
        // Detection first, over the untouched block, so the background is
        // never measured against samples the blanker has already altered.
        self.mask.clear();
        self.mask.resize(n, false);
        for (i, sample) in iq.iter().enumerate() {
            let mag = sample.norm();
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
            let trigger = self.level != 0
                && self.trained >= (self.fs as usize / 4).max(32)
                && mag > threshold * self.background.max(1e-6);
            if trigger {
                self.window_triggers += 1;
            }
            if trigger && !self.inhibited {
                // An impulse has skirts: the samples either side of the one
                // that tripped are already contaminated.
                for j in i.saturating_sub(2)..=(i + 2).min(n - 1) {
                    self.mask[j] = true;
                }
            }
            self.window_seen += 1;
            self.trained = self.trained.saturating_add(1);
            if self.window_seen >= self.fs as usize {
                self.last_rate = self.window_blanked;
                self.inhibited = self.window_triggers * 50 > self.window_seen;
                self.window_seen = 0;
                self.window_blanked = 0;
                self.window_triggers = 0;
            }
        }

        // Then replace each run by a straight line between the last good
        // sample before it and the first after. Zeroing a run instead leaves
        // a rectangular hole, and a hole is a step at each end — which the
        // narrow channel filters downstream ring on, turning a four-sample
        // impulse into a many-millisecond thump of a different shape.
        let mut i = 0;
        while i < n {
            if !self.mask[i] {
                i += 1;
                continue;
            }
            let start = i;
            while i < n && self.mask[i] {
                i += 1;
            }
            let end = i;
            let before = if start > 0 {
                iq[start - 1]
            } else {
                self.last_good
            };
            let after = if end < n { iq[end] } else { before };
            let span = (end - start + 1) as f32;
            for (k, j) in (start..end).enumerate() {
                let t = (k + 1) as f32 / span;
                iq[j] = before * (1.0 - t) + after * t;
            }
            self.window_blanked += end - start;
        }
        self.last_good = iq[n - 1];
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
        // Blank *before* estimating the image correction. An impulse is
        // broadband and, by definition, the loudest thing in the block, so
        // leaving it in means the cross-correlation the imbalance estimate is
        // built from is dominated by something that is not a signal at all.
        self.blanker.process(iq);
        self.correct_images(iq);
    }

    fn correct_block(&mut self, block: &mut [Complex32]) {
        const N: usize = 4096;
        const K: usize = 12;
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
                for neighbour in [b.checked_sub(1), (b + 1 < K).then_some(b + 1)]
                    .into_iter()
                    .flatten()
                {
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
            for (dst, src) in block.iter_mut().zip(&self.iq_buf) {
                *dst = *src / N as f32;
            }
        }
    }

    fn correct_images(&mut self, iq: &mut [Complex32]) {
        const N: usize = 4096;
        let mut chunks = iq.chunks_exact_mut(N);
        for block in chunks.by_ref() {
            self.correct_block(block);
        }
        // Whatever is left over is shorter than the transform, and it used to
        // be passed through with no correction at all — silently, and
        // depending on a block size chosen three files away. The per-band
        // solve cannot run on it, but the bands it has already learned can
        // still be applied as their mean, which is exactly the scalar
        // correction this front end used to do for everything.
        let tail = chunks.into_remainder();
        if !tail.is_empty() && self.warm >= self.settle {
            let mut mean = Complex32::new(0.0, 0.0);
            for a in &self.image {
                mean += *a;
            }
            mean /= self.image.len() as f32;
            for s in tail.iter_mut() {
                *s -= mean * s.conj();
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
    pub(crate) fn dirty_iq(
        fs: f32,
        sig_hz: f32,
        dc: Complex32,
        gain_err: f32,
        phase_err: f32,
    ) -> Vec<Complex32> {
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
            dirty.push(
                tone + noise
                    + if crash {
                        Complex32::new(8.0, -6.0)
                    } else {
                        Complex32::new(0.0, 0.0)
                    },
            );
        }
        (clean, dirty)
    }

    fn error_power(got: &[Complex32], want: &[Complex32]) -> f32 {
        got.iter()
            .zip(want)
            .skip(got.len() / 3)
            .map(|(a, b)| (*a - *b).norm_sqr())
            .sum::<f32>()
            / (got.len() - got.len() / 3) as f32
    }

    #[test]
    fn noise_blanker_removes_impulses_without_harming_clean_iq() {
        let fs = 48_000.0;
        let (clean, dirty) = impulse_case(fs, true);
        let mut blanked = dirty.clone();
        let mut nb = NoiseBlanker::new(fs as f64);
        for block in blanked.chunks_mut(4096) {
            nb.process(block);
        }
        let improvement =
            10.0 * (error_power(&dirty, &clean) / error_power(&blanked, &clean)).log10();
        // Interpolating across a blanked run rather than zeroing it is worth
        // about 2 dB of this on its own: a zeroed run is a rectangular hole,
        // and its edges are steps that the narrow channel filters downstream
        // ring on.
        assert!(
            improvement >= 12.0,
            "impulse error improved only {improvement:.1} dB"
        );

        let (_, mut untouched) = impulse_case(fs, false);
        let before = power_at(&untouched, fs, 1500.0);
        let mut nb = NoiseBlanker::new(fs as f64);
        for block in untouched.chunks_mut(4096) {
            nb.process(block);
        }
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
        for block in blanked.chunks_mut(4096) {
            nb.process(block);
        }
        let improvement =
            10.0 * (error_power(&dirty, &clean) / error_power(&blanked, &clean)).log10();
        println!(
            "wideband impulse blanker: {improvement:.1} dB error-power improvement, {} blanks/s",
            nb.blanks_per_second()
        );
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
        let before = 10.0 * (power_at(&iq, fs, 400.0) / power_at(&iq, fs, -400.0)).log10();
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
        let iq = (0..(fs * 10.0) as usize)
            .map(|i| {
                let mut x = Complex32::new(0.0, 0.0);
                for j in 0..2 {
                    let s = Complex32::from_polar(0.35, 2.0 * PI * hz[j] * i as f32 / fs);
                    x += s + a[j] * s.conj();
                }
                x
            })
            .collect();
        (iq, hz)
    }

    fn worst_rejection(iq: &[Complex32], fs: f32, hz: [f32; 2]) -> f32 {
        hz.into_iter()
            .map(|f| 10.0 * (power_at(iq, fs, f) / power_at(iq, fs, -f).max(1e-30)).log10())
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
        assert!(
            per_band >= scalar_rej + 15.0,
            "subbands gained only {:.1} dB ({scalar_rej:.1} -> {per_band:.1})",
            per_band - scalar_rej
        );
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
        println!(
            "frequency-dependent IQ: worst-band scalar {scalar_rej:.1} dB, 12-band {per_band:.1} dB ({:.1} dB gain)",
            per_band - scalar_rej
        );
    }

    /// The same frequency-dependent imbalance, with static crashes on top.
    fn imbalance_with_impulses(fs: f32) -> (Vec<Complex32>, [f32; 2]) {
        let (mut iq, hz) = frequency_dependent_case(fs);
        for (i, s) in iq.iter_mut().enumerate() {
            if i > fs as usize / 10 && i % 1700 < 4 {
                *s += Complex32::new(9.0, -7.0);
            }
        }
        (iq, hz)
    }

    /// The whole front end, blanker included, in the blocks the app uses.
    fn run_full(iq: &[Complex32], fs: f32) -> Vec<Complex32> {
        let mut fe = FrontEnd::new(fs as f64);
        let mut out = Vec::new();
        for chunk in iq.chunks(16_384) {
            let mut block = chunk.to_vec();
            fe.process(&mut block);
            out.extend(block);
        }
        out.split_off(out.len() * 3 / 4)
    }

    /// Order matters: blank, then estimate the imbalance.
    ///
    /// The image estimate is a cross-correlation between each bin and the
    /// conjugate of its mirror, accumulated over whole blocks. An impulse is
    /// broadband and is the loudest thing in the block by a wide margin, so
    /// running the estimator over unblanked IQ builds the correction mostly
    /// out of static crashes — and the correction is then applied to every
    /// sample, all the time, on a band where crashes are what HF sounds like.
    #[test]
    fn impulses_do_not_poison_the_image_estimate() {
        let fs = 48_000.0;
        let (iq, hz) = imbalance_with_impulses(fs);
        let rej = worst_rejection(&run_full(&iq, fs), fs, hz);
        assert!(
            rej >= 40.0,
            "image rejection through static crashes was only {rej:.1} dB"
        );
    }

    #[test]
    #[ignore]
    fn bench_blank_before_image_estimate() {
        let fs = 48_000.0;
        let (iq, hz) = imbalance_with_impulses(fs);
        let rej = worst_rejection(&run_full(&iq, fs), fs, hz);
        let (clean, _) = frequency_dependent_case(fs);
        let clean_rej = worst_rejection(&run_full(&clean, fs), fs, hz);
        println!(
            "worst-band image rejection: {clean_rej:.1} dB without impulses, {rej:.1} dB with them"
        );
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
    use super::frontend_tests::*;
    use super::*;

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
        for (g, p) in [
            (1.01f32, 0.009f32),
            (1.03, 0.026),
            (1.06, 0.052),
            (1.12, 0.105),
        ] {
            let iq = dirty_iq(fs, 400.0, Complex32::new(0.0, 0.0), g, p);
            let before = 10.0 * (power_at(&iq, fs, 400.0) / power_at(&iq, fs, -400.0)).log10();
            let out = run_frontend(&iq, fs);
            let after = 10.0 * (power_at(&out, fs, 400.0) / power_at(&out, fs, -400.0)).log10();
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
            let b = peak_over_floor(&spectrum_of(&iq, Window::BlackmanHarris, n), fs, weak_hz);
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
                    let strong = Complex32::from_polar(50.0, 2.0 * PI * (weak_hz + sep) * t);
                    weak + strong + Complex32::new(noise(&mut rng), noise(&mut rng)) * 0.01
                })
                .collect();
            let h = peak_over_floor(&spectrum_of(&iq, Window::Hann, n), fs, weak_hz);
            let b = peak_over_floor(&spectrum_of(&iq, Window::BlackmanHarris, n), fs, weak_hz);
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
        let n: f32 = probes.iter().map(|&f| power_at(iq, fs, f)).sum::<f32>() / probes.len() as f32;
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
            (
                "clean input      ",
                1.0f32,
                0.0f32,
                Complex32::new(0.0, 0.0),
            ),
            ("1% gain imbalance", 1.01, 0.005, Complex32::new(0.0, 0.0)),
            (
                "5% gain imbalance",
                1.05,
                0.03,
                Complex32::new(0.01, -0.007),
            ),
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
