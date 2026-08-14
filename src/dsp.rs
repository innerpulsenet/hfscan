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
///
/// A second FIR at the audio rate sharpens the skirt — the radio-rate
/// decimator cannot resolve a 80 Hz cutoff at 192 kHz with a practical
/// tap count, but 129 taps at 8 kHz can.
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
        let audio_cut = (bandwidth / 2.0).min(fs_out as f32 * 0.45);
        Self {
            nco: Nco::new(),
            fir: DecimFir::new(bandwidth / 2.0, fs_in as f32, decim, 255),
            audio: DecimFir::new(audio_cut, fs_out as f32, 1, 129),
            fs_in,
            fs_out,
            mixed: Vec::new(),
            decimated: Vec::new(),
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
        let audio_cut = (bw / 2.0).min(self.fs_out as f32 * 0.45).max(20.0);
        self.audio.set_cutoff(audio_cut, self.fs_out as f32);
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
        let input: Vec<Complex32> = (0..4096)
            .map(|i| Complex32::new((i as f32 * 0.01).sin() * 0.05, 0.0))
            .collect();
        let mut out = Vec::new();
        chain.process(&input, &mut out);
        assert!(!out.is_empty(), "audio FIR should emit samples");
    }
}
