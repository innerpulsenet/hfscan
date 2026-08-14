//! Spectrum estimation and the narrowband tuning chain feeding the decoders.

use num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::f32::consts::PI;
use std::sync::Arc;

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
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(size);
        // Hann window
        let window: Vec<f32> = (0..size)
            .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / size as f32).cos())
            .collect();
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
    pub fn power_db(&mut self, input: &[Complex32], out: &mut Vec<f32>) {
        self.pending.extend_from_slice(input);
        let nseg = self.pending.len() / self.size;
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
            let seg = &self.pending[s * self.size..(s + 1) * self.size];
            for i in 0..self.size {
                self.buf[i] = seg[i] * self.window[i];
            }
            self.fft.process(&mut self.buf);
            for i in 0..self.size {
                acc[i] += self.buf[i].norm_sqr();
            }
        }
        self.pending.drain(..nseg * self.size);
        let scale = 1.0 / (nseg as f32 * self.size as f32);
        let half = self.size / 2;
        // fftshift while converting to dB
        for i in 0..self.size {
            let src = (i + half) % self.size;
            out[i] = 10.0 * (acc[src] * scale + 1e-20).log10();
        }
    }
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
}

#[allow(dead_code)]
impl DecimFir {
    pub fn new(cutoff_hz: f32, fs: f32, decim: usize, ntaps: usize) -> Self {
        let ntaps = ntaps | 1; // force odd so there is a center tap
        Self {
            taps: lowpass_taps(cutoff_hz, fs, ntaps),
            buf: Vec::with_capacity(ntaps * 4),
            decim,
        }
    }

    pub fn set_cutoff(&mut self, cutoff_hz: f32, fs: f32) {
        let n = self.taps.len();
        self.taps = lowpass_taps(cutoff_hz, fs, n);
    }

    pub fn decim(&self) -> usize {
        self.decim
    }

    pub fn process(&mut self, input: &[Complex32], out: &mut Vec<Complex32>) {
        out.clear();
        self.buf.extend_from_slice(input);
        let n = self.taps.len();
        let mut i = 0usize;
        while i + n <= self.buf.len() {
            let mut acc = Complex32::new(0.0, 0.0);
            for k in 0..n {
                acc += self.buf[i + k] * self.taps[k];
            }
            out.push(acc);
            i += self.decim;
        }
        if i > 0 {
            self.buf.drain(..i);
        }
    }
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
        // Blackman window keeps the stopband clean enough for narrow HF filters
        let w = 0.42 - 0.5 * (2.0 * PI * i as f32 / (ntaps - 1) as f32).cos()
            + 0.08 * (4.0 * PI * i as f32 / (ntaps - 1) as f32).cos();
        taps.push(sinc * w);
    }
    let sum: f32 = taps.iter().sum();
    if sum.abs() > 1e-12 {
        taps.iter_mut().for_each(|t| *t /= sum);
    }
    taps
}

/// NCO + decimating lowpass: takes wideband IQ at the radio rate and produces
/// narrowband complex baseband centred on the cursor at the mode's audio rate.
pub struct DecodeChain {
    nco: Nco,
    fir: DecimFir,
    fs_in: f64,
    fs_out: f64,
    mixed: Vec<Complex32>,
}

impl DecodeChain {
    /// `target_rate` is the audio rate the mode wants. The achieved rate is
    /// `fs_in / round(fs_in / target_rate)`, so it only lands exactly on the
    /// target when the radio rate is an integer multiple of it — which FT8/FT4
    /// require, and the caller enforces.
    pub fn new(fs_in: f64, bandwidth: f32, target_rate: f64) -> Self {
        let decim = (fs_in / target_rate).round().max(1.0) as usize;
        let fs_out = fs_in / decim as f64;
        Self {
            nco: Nco::new(),
            fir: DecimFir::new(bandwidth / 2.0, fs_in as f32, decim, 255),
            fs_in,
            fs_out,
            mixed: Vec::new(),
        }
    }

    pub fn fs_out(&self) -> f64 {
        self.fs_out
    }

    pub fn set_offset(&mut self, hz: f64) {
        self.nco.set_freq(hz, self.fs_in);
    }

    pub fn set_bandwidth(&mut self, bw: f32) {
        self.fir.set_cutoff(bw / 2.0, self.fs_in as f32);
    }

    pub fn process(&mut self, input: &[Complex32], out: &mut Vec<Complex32>) {
        self.nco.mix(input, &mut self.mixed);
        let mixed = std::mem::take(&mut self.mixed);
        self.fir.process(&mixed, out);
        self.mixed = mixed;
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
