//! Watterson HF channel model, for benchmarking decoders against the band
//! rather than against a laboratory tone.
//!
//! Every synthetic signal in this crate's decoder tests was, until this
//! module, a constant-amplitude carrier in additive Gaussian noise. That is a
//! wire, not an ionosphere, and the difference is not a detail: measured
//! across the live 20m capture, the mark level of a single CW station moves
//! by 12 to 31 dB inside one 60-second over, and which stations the CW
//! decoder copies tracks that number far more closely than it tracks their
//! signal-to-noise ratio. A decoder tuned on flat carriers is tuned for a
//! band nobody transmits on.
//!
//! Watterson is the standard model for exactly this (CCIR Rec. 520). A signal
//! arrives over two or more ionospheric paths of slightly different length;
//! each path's amplitude and phase wander independently, and the receiver
//! hears their sum. Two consequences matter here, and neither can be produced
//! by scaling a carrier up and down:
//!
//! - The sum of two independent complex Gaussian paths is Rayleigh
//!   distributed, so the deep fades are *nulls*, not dips. That is what puts
//!   30 dB of range into one over.
//! - Because the paths differ in delay, the null lands at a different time for
//!   each frequency. Two stations 200 Hz apart in the same passband fade
//!   independently, which is what a real crowded band does and what a single
//!   multiplicative envelope cannot reproduce.
//!
//! The model is shared rather than CW-specific: RTTY and PSK31 have their own
//! ad-hoc fading in the tests, and both should move onto this.

#![allow(dead_code)]

use num_complex::Complex32;
use std::f32::consts::PI;

/// A path count, a differential delay and a Doppler spread — the three
/// numbers that define a Watterson channel.
#[derive(Clone, Copy, Debug)]
pub struct Condition {
    /// Differential delay between the two paths, milliseconds.
    ///
    /// For a signal as narrow as CW this is not about intersymbol
    /// interference — 2 ms against a 60 ms dit is nothing. It sets the
    /// coherence bandwidth, `~1/(2*pi*delay)`, which is what decides whether
    /// two stations in the same passband fade together or independently.
    pub delay_ms: f32,
    /// Doppler spread, Hz: the standard deviation of the Gaussian Doppler
    /// power spectrum, and so the rate at which the fading moves.
    pub doppler_hz: f32,
}

/// CCIR 520 "good" conditions — a quiet path, slow shallow fading.
pub const CCIR_GOOD: Condition = Condition {
    delay_ms: 0.5,
    doppler_hz: 0.1,
};
/// CCIR 520 "moderate" conditions.
pub const CCIR_MODERATE: Condition = Condition {
    delay_ms: 1.0,
    doppler_hz: 0.5,
};
/// CCIR 520 "poor" conditions.
pub const CCIR_POOR: Condition = Condition {
    delay_ms: 2.0,
    doppler_hz: 1.0,
};
/// A disturbed, fluttery path — high-latitude or near-auroral. Included
/// because it is the case that separates a decoder that degrades from one
/// that falls over.
pub const CCIR_FLUTTER: Condition = Condition {
    delay_ms: 1.0,
    doppler_hz: 5.0,
};
/// No channel at all: the flat carrier the old tests used. Kept so a bench
/// can show the flat and faded cases side by side.
pub const FLAT: Condition = Condition {
    delay_ms: 0.0,
    doppler_hz: 0.0,
};

/// xorshift, matching the one the decoder tests use so nothing here needs a
/// dependency and every run is reproducible.
fn next_u32(rng: &mut u32) -> u32 {
    *rng ^= *rng << 13;
    *rng ^= *rng >> 17;
    *rng ^= *rng << 5;
    *rng
}

fn uniform(rng: &mut u32) -> f32 {
    next_u32(rng) as f32 / u32::MAX as f32
}

/// One standard normal sample, Box-Muller.
fn gauss(rng: &mut u32) -> f32 {
    let u1 = uniform(rng).max(1e-7);
    let u2 = uniform(rng);
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// Generate one path's complex fading process, at `fade_rate` samples/s.
///
/// White complex Gaussian shaped by a Gaussian kernel, which is what gives
/// the Gaussian Doppler power spectrum the model calls for. For a kernel with
/// time-domain standard deviation `s`, the *power* spectrum has standard
/// deviation `1/(2*pi*sqrt(2)*s)`; setting that to the wanted spread fixes
/// `s`. Sampling the process at a fixed multiple of the spread makes `s` a
/// constant in samples, so the kernel is short no matter how slow the fading.
fn fading_process(n: usize, fade_rate: f32, doppler_hz: f32, rng: &mut u32) -> Vec<Complex32> {
    let s = fade_rate / (2.0 * PI * std::f32::consts::SQRT_2 * doppler_hz.max(1e-6));
    let half = ((4.0 * s).ceil() as usize).max(1);
    let kernel: Vec<f32> = (0..=2 * half)
        .map(|i| {
            let t = i as f32 - half as f32;
            (-0.5 * (t / s) * (t / s)).exp()
        })
        .collect();
    let norm = kernel.iter().map(|k| k * k).sum::<f32>().sqrt().max(1e-12);

    let total = n + 2 * half;
    // Unit *complex* power: two unit-variance components would carry power
    // two, and the resulting 3 dB would be quietly added to every signal that
    // passed through the channel.
    let c = std::f32::consts::FRAC_1_SQRT_2;
    let white: Vec<Complex32> = (0..total)
        .map(|_| Complex32::new(gauss(rng) * c, gauss(rng) * c))
        .collect();

    (0..n)
        .map(|i| {
            let mut acc = Complex32::new(0.0, 0.0);
            for (k, &w) in kernel.iter().enumerate() {
                acc += white[i + k] * w;
            }
            // Unit mean power per path, so the caller's SNR still means what
            // it says once both paths are summed.
            acc / norm
        })
        .collect()
}

/// Pass `sig` through a two-path Watterson channel.
///
/// Mean power is preserved, so a caller can fade a signal and then add noise
/// for a wanted SNR in the usual way and get the SNR they asked for *on
/// average* — the whole point being that the instantaneous value now moves.
pub fn watterson(sig: &[Complex32], fs: f32, cond: Condition, seed: u32) -> Vec<Complex32> {
    if cond.doppler_hz <= 0.0 {
        return sig.to_vec();
    }
    // The fading is sampled at a fixed multiple of its own spread and
    // interpolated up; a Gaussian spectrum is long dead by 40 sigma.
    let fade_rate = (40.0 * cond.doppler_hz).clamp(20.0, 2000.0);
    let step = fade_rate / fs;
    let n_fade = (sig.len() as f32 * step).ceil() as usize + 2;

    let mut rng = seed | 1;
    let g0 = fading_process(n_fade, fade_rate, cond.doppler_hz, &mut rng);
    let g1 = fading_process(n_fade, fade_rate, cond.doppler_hz, &mut rng);

    let delay = ((cond.delay_ms / 1000.0) * fs).round() as usize;
    // Two equal-power paths summed; the half keeps total power at unity.
    let scale = std::f32::consts::FRAC_1_SQRT_2;

    (0..sig.len())
        .map(|i| {
            let x = i as f32 * step;
            let j = x as usize;
            let frac = x - j as f32;
            let lerp = |g: &[Complex32]| g[j] + (g[j + 1] - g[j]) * frac;
            let direct = sig[i] * lerp(&g0);
            let echo = if i >= delay {
                sig[i - delay] * lerp(&g1)
            } else {
                Complex32::new(0.0, 0.0)
            };
            (direct + echo) * scale
        })
        .collect()
}

/// Add impulsive noise — the static crashes that dominate a real HF band and
/// that Gaussian noise does not contain.
///
/// A crash keys the decoder's threshold tracker far harder than its duration
/// suggests, because a peak-hold takes the peak and then has to decay out of
/// it. That is a failure mode no additive-Gaussian test can produce.
///
/// `per_sec` crashes a second, each a few milliseconds long and `amp` times
/// the signal's own RMS.
pub fn static_crashes(sig: &mut [Complex32], fs: f32, per_sec: f32, amp: f32, seed: u32) {
    if per_sec <= 0.0 || sig.is_empty() {
        return;
    }
    let rms = (sig.iter().map(|s| s.norm_sqr()).sum::<f32>() / sig.len() as f32)
        .sqrt()
        .max(1e-9);
    let mut rng = seed | 1;
    let n_crashes = ((sig.len() as f32 / fs) * per_sec).round() as usize;
    for _ in 0..n_crashes {
        let at = (uniform(&mut rng) * sig.len() as f32) as usize;
        // 1–4 ms, decaying.
        let len = ((1.0 + 3.0 * uniform(&mut rng)) * 1e-3 * fs) as usize;
        let a = rms * amp * (0.3 + uniform(&mut rng));
        for k in 0..len {
            let Some(s) = sig.get_mut(at + k) else { break };
            let env = a * (-(k as f32) / (len as f32 * 0.3)).exp();
            *s += Complex32::new(gauss(&mut rng), gauss(&mut rng)) * env;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The channel must not quietly change the signal level, or every SNR in
    /// every bench that uses it means something different from what it says.
    ///
    /// The claim is about the ensemble, not about one run, and the difference
    /// is physical rather than a testing convenience: at 0.1 Hz of Doppler
    /// spread a ten-second over contains about one fade, so its realised mean
    /// power genuinely can sit several dB off the long-run average. That is
    /// what "the band was down for that over" means. Averaging over
    /// realisations tests the model; normalising each one would delete the
    /// effect being modelled.
    #[test]
    fn the_channel_preserves_mean_power() {
        let sig: Vec<Complex32> = (0..80_000)
            .map(|i| Complex32::from_polar(1.0, 0.05 * i as f32))
            .collect();
        let p_in = sig.iter().map(|s| s.norm_sqr()).sum::<f32>() / sig.len() as f32;
        for cond in [CCIR_GOOD, CCIR_MODERATE, CCIR_POOR, CCIR_FLUTTER] {
            let mut p_sum = 0.0;
            const RUNS: u32 = 24;
            for k in 0..RUNS {
                let out = watterson(&sig, 8000.0, cond, 0x1234_5678 ^ (k * 0x9e37_79b9));
                p_sum += out.iter().map(|s| s.norm_sqr()).sum::<f32>() / out.len() as f32;
            }
            let db = 10.0 * (p_sum / RUNS as f32 / p_in).log10();
            assert!(
                db.abs() < 1.0,
                "{cond:?} shifted mean power by {db:.2} dB over {RUNS} runs; \
                 SNRs would stop meaning anything"
            );
        }
    }

    /// The point of the model: fades are deep, and they get deeper and faster
    /// as conditions worsen. A multiplicative envelope would give a fixed
    /// depth; Rayleigh nulls are what put 30 dB into one over.
    #[test]
    fn worse_conditions_fade_deeper_and_faster() {
        let sig: Vec<Complex32> = (0..480_000)
            .map(|i| Complex32::from_polar(1.0, 0.05 * i as f32))
            .collect();
        let depth = |cond: Condition| {
            let out = watterson(&sig, 8000.0, cond, 0xdead_beef);
            // Envelope percentiles over the whole run, in dB.
            let mut env: Vec<f32> = out.chunks(800).map(|c| c[0].norm()).collect();
            env.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let lo = env[env.len() / 20].max(1e-9);
            let hi = env[env.len() * 19 / 20];
            20.0 * (hi / lo).log10()
        };
        let good = depth(CCIR_GOOD);
        let poor = depth(CCIR_POOR);
        assert!(
            good > 8.0,
            "even a good path should fade by more than {good:.1} dB"
        );
        assert!(
            poor > 15.0,
            "a poor path should show deep nulls, got {poor:.1} dB"
        );
    }

    /// The reason the model has two paths with a delay between them rather
    /// than one envelope: a real band fades one station and not its
    /// neighbour, and a decoder that has to pick between them meets that
    /// every time it is used.
    #[test]
    fn separated_tones_fade_independently() {
        let fs = 8000.0f32;
        let n = 480_000;
        let tone = |hz: f32| -> Vec<Complex32> {
            (0..n)
                .map(|i| Complex32::from_polar(1.0, 2.0 * PI * hz * i as f32 / fs))
                .collect()
        };
        // One channel realisation, both tones through it together.
        let mut both: Vec<Complex32> = tone(0.0)
            .iter()
            .zip(tone(300.0).iter())
            .map(|(a, b)| a + b)
            .collect();
        both = watterson(&both, fs, CCIR_POOR, 0x0bad_f00d);

        // Recover each tone's envelope by mixing it to DC and low-passing.
        let env_of = |hz: f32| -> Vec<f32> {
            let mut z = Complex32::new(0.0, 0.0);
            let a = 1.0 - (-2.0 * PI * 20.0 / fs).exp();
            let mut out = Vec::new();
            for (i, s) in both.iter().enumerate() {
                let m = s * Complex32::from_polar(1.0, -2.0 * PI * hz * i as f32 / fs);
                z += (m - z) * a;
                if i % 800 == 0 {
                    out.push(z.norm());
                }
            }
            out
        };
        let (e0, e1) = (env_of(0.0), env_of(300.0));
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let (m0, m1) = (mean(&e0), mean(&e1));
        let cov: f32 = e0
            .iter()
            .zip(&e1)
            .map(|(a, b)| (a - m0) * (b - m1))
            .sum::<f32>();
        let v0: f32 = e0.iter().map(|a| (a - m0) * (a - m0)).sum();
        let v1: f32 = e1.iter().map(|b| (b - m1) * (b - m1)).sum();
        let r = cov / (v0 * v1).sqrt().max(1e-12);
        assert!(
            r < 0.8,
            "tones 300 Hz apart faded together (r={r:.2}); the delay is not doing its job"
        );
    }
}
