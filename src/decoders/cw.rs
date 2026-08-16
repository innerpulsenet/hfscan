//! Morse (CW) decoder with nearby-tone lock and an HSMM element decoder.
//!
//! Detection and timing are solved jointly by an explicit-duration HMM
//! scored over a grid of dit periods, so a speed change cannot close a
//! loop through a slicer. Stage 2 of `weak-signal-plan-3-cw.md` — in the
//! tree, not yet at that plan's band/flat bars. A passband scout finds CW
//! tones near the cursor and mixes the best one to DC; `n` hops to the next.

use super::callscan::{CallScanner, utc_hhmmss};
use super::cw_hsmm::{HsmmDecoder, MorseEvent};
use super::{CwView, Decoder, FtMessage};
use crate::dsp::{OnePole, Rotator, mix_decim_into};
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
/// Envelope history for the scope pane (~0.7 s at 1 kHz).
const ENV_HIST: usize = 700;
/// 5 WPM upper bound on the matched-filter window.
const DIT_MAX_S: f32 = 0.24;
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

/// Corner frequency per second of dit, once the clock is known.
///
/// 150 Hz is the right answer only for a fist fast enough to need it. A
/// four-pole 150 Hz filter passes about 147 Hz of noise, while the keying it
/// has to carry occupies roughly `2/T` — 33 Hz at 20 WPM. That gap is the
/// single largest weak-signal loss in this decoder: measured, narrowing the
/// filter onto the tracked clock takes 20 WPM copy at 0 dB from 15% to 60%,
/// and 25 WPM from 20% to 65%, with no cost at any SNR above it.
///
/// The constant is a bandwidth-per-baud, not a frequency: `K / dit_seconds`
/// gives 37 Hz at 18 WPM and 73 Hz at 35 WPM, which is what the sweep across
/// both speeds asked for. Measured at 2.0, 2.5 and 3.0; 2.5 was the value
/// that cost nothing at 20 dB and gained the most at 0.
const POST_MIX_K: f32 = 2.5;
/// The narrowest the filter is ever allowed to get.
///
/// This is a safety floor, not a tuning parameter, and it is set by a failure
/// mode rather than by a measurement. Narrowing onto the tracked clock closes
/// a loop through the very estimator that sets it: if the filter is ever too
/// narrow for the fist actually being sent, the keying smears, the marks
/// merge, the clock reads *slower* — and a slower clock asks for a narrower
/// filter still. A station going from 15 to 32 WPM drove exactly that spiral
/// down to a 7.9 WPM estimate with no way back out.
///
/// The floor is wide enough that the loop can only ever run in the safe
/// direction — widening — for any fist the tracker will follow.
///
/// 70 Hz was the original guess. It can come down, but only as far as the
/// speed tracker's own reach allows, and the two have to be moved together:
/// every fixed-speed cell in `bench_cw_score` wants the narrowest filter its
/// keying will fit through, so the score rises steadily as the floor drops,
/// while `cw_follows_a_speed_change` fails abruptly once the filter can no
/// longer pass a fist the tracker is still trying to follow. Sweeping the two
/// against each other, with `RECLUSTER_MARKS` shortened so the tracker keeps
/// up: 55 and 60 Hz both pass and score 90.0, 50 Hz fails, 70 Hz passes and
/// scores 89.6.
///
/// 60 Hz is the best of those and leaves a margin over the 50 Hz failure.
/// That margin is the point: the spiral has no way out once it starts, so
/// being slightly too wide costs a fraction of a dB and being slightly too
/// narrow costs the station.
const POST_MIX_MIN_HZ: f32 = 60.0;

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

    /// Retune the corner, keeping the filter state so the change does not
    /// click. Only called when the HSMM's winning period has moved materially.
    fn set_corner(&mut self, hz: f32, fs: f32) {
        self.a = 1.0 - (-2.0 * PI * hz / fs).exp();
    }
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
    matched: VecDeque<f32>,
    matched_sum: f32,
    decim_ctr: usize,
    key_down: bool,
    dit: f32,
    hsmm: HsmmDecoder,
    symbol: String,
    text: String,
    /// Word scanner for pskreporter spots (`CQ ... CALL`, `DE CALL CALL`).
    scan: CallScanner,
    quality: f32,
    env_hist: VecDeque<f32>,
    key_hist: VecDeque<bool>,
    on_thr: f32,
    off_thr: f32,
    tune_err: f32,
    /// Mean envelope well inside a mark, and well inside a space.
    mark_env: f32,
    space_env: f32,
    prev_mixed: Complex32,
    have_mixed: bool,
    /// Envelope samples still to discard while the filters fill.
    settle: u32,
    /// Searches to skip after a manual nudge so AFC does not fight the user.
    hold_tune: u32,

    mix_hz: f32,
    /// Mixer that brings the locked tone to DC. A rotator rather than a
    /// `sin_cos` per audio sample, retuned in place when the lock moves.
    mix_rot: Rotator,
    mix_rate_hz: f32,
    /// Rejects the neighbours the 400 Hz chain filter lets through, and the
    /// noise outside the keying bandwidth once the clock is known.
    post: NarrowLpf,
    /// Corner currently programmed into `post`.
    post_hz: f32,
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
            smooth: OnePole::new(0.003 * fs),
            matched: VecDeque::new(),
            matched_sum: 0.0,
            post: NarrowLpf::new(fs),
            post_hz: POST_MIX_HZ,
            decim_ctr: 0,
            key_down: false,
            dit: 0.06 * env_rate, // start at 20 WPM
            hsmm: HsmmDecoder::new(env_rate),
            symbol: String::new(),
            text: String::new(),
            scan: CallScanner::new(),
            quality: 0.0,
            env_hist: VecDeque::with_capacity(ENV_HIST + 1),
            key_hist: VecDeque::with_capacity(ENV_HIST + 1),
            on_thr: 0.55,
            off_thr: 0.35,
            tune_err: 0.0,
            mark_env: 0.0,
            space_env: 0.0,
            prev_mixed: Complex32::new(0.0, 0.0),
            have_mixed: false,
            settle: SETTLE_MS,
            hold_tune: 0,
            mix_hz: 0.0,
            mix_rot: Rotator::new(0.0),
            mix_rate_hz: 0.0,
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
        if dit_ms > 1.0 { 1200.0 / dit_ms } else { 0.0 }
    }

    fn mix(&mut self, s: Complex32) -> Complex32 {
        if self.mix_hz.abs() < 0.05 {
            return s;
        }
        if (self.mix_hz - self.mix_rate_hz).abs() > 0.01 {
            self.mix_rate_hz = self.mix_hz;
            self.mix_rot.set_rate(-2.0 * PI * self.mix_hz / self.fs);
        }
        s * self.mix_rot.next()
    }

    /// Match the post-mix filter to the winning dit period.
    fn update_post_mix(&mut self) {
        let want = if !self.hsmm.have_period() {
            POST_MIX_HZ
        } else {
            let dit_s = (self.dit / self.env_rate).max(1e-3);
            (POST_MIX_K / dit_s).clamp(POST_MIX_MIN_HZ, POST_MIX_HZ)
        };
        if (want - self.post_hz).abs() > 0.08 * self.post_hz {
            self.post_hz = want;
            self.post.set_corner(want, self.fs);
        }
    }

    fn push_symbol(&mut self) {
        if self.symbol.is_empty() {
            return;
        }
        let sym = std::mem::take(&mut self.symbol);
        if self.mark_env < 1.25 * self.space_env.max(1e-9) {
            return;
        }
        if let Some(c) = morse_lookup(&sym) {
            self.text.push(c);
            self.scan.push(c);
        } else if sym.len() <= 7 {
            self.text.push('*');
            self.scan.push('*');
        }
    }

    fn apply_events(&mut self, evs: &[MorseEvent]) {
        // Without a tone lock, mark/space contrast is the only thing that
        // separates a weak station from the HSMM explaining band noise.
        if !self.locked && self.mark_env < 2.4 * self.space_env.max(1e-9) {
            return;
        }
        for ev in evs {
            match ev {
                MorseEvent::Dit => self.symbol.push('.'),
                MorseEvent::Dah => self.symbol.push('-'),
                MorseEvent::CharGap => self.push_symbol(),
                MorseEvent::WordGap => {
                    self.push_symbol();
                    self.scan.push(' ');
                    if !self.text.ends_with(' ') && !self.text.is_empty() {
                        self.text.push(' ');
                    }
                }
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
            if v >= mag(self.fft_buf[k - 1]) && v >= mag(self.fft_buf[k + 1]) && v / med >= 4.0 {
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
                if fresh
                    .iter()
                    .any(|h: &CwHit| (h.offset_hz - hz).abs() < 40.0)
                {
                    continue;
                }
                fresh.push(CwHit {
                    offset_hz: hz,
                    score: ratio,
                    quality: q,
                });
            }
        }
        fresh.sort_by(|a, b| {
            b.quality
                .partial_cmp(&a.quality)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

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
        self.hits.sort_by(|a, b| {
            let dist_a = (a.offset_hz - self.mix_hz).abs();
            let dist_b = (b.offset_hz - self.mix_hz).abs();
            let rank_a = a.quality - (dist_a / 400.0).min(0.20);
            let rank_b = b.quality - (dist_b / 400.0).min(0.20);
            rank_b
                .partial_cmp(&rank_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
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
                self.mix_hz = 0.85 * self.mix_hz + 0.15 * h.offset_hz;
                self.lock_score = h.score;
                self.locked = true;
                return;
            }
        }
        if let Some(h) = self.hits.first().cloned() {
            if (h.offset_hz - self.mix_hz).abs() > 12.0 {
                self.mix_rot.reset_phase();
                self.post.reset();
            }
            self.mix_hz = h.offset_hz;
            self.lock_score = h.score;
            self.locked = true;
        }
    }

    /// SNR in a 2500 Hz reference bandwidth, for the spot report.
    fn spot_snr(&self) -> f32 {
        if self.mark_env <= 0.0 || self.space_env <= 0.0 {
            return -24.0;
        }
        let ratio = (self.mark_env / self.space_env).max(1.0);
        (20.0 * ratio.log10() - 1.05 - 12.3 - 3.2).clamp(-24.0, 20.0)
    }

    fn clear_lock_state(&mut self) {
        self.scan.reset();
        self.mix_hz = 0.0;
        self.mix_rot.reset_phase();
        self.post.reset();
        self.locked = false;
        self.lock_score = 0.0;
        self.post_hz = POST_MIX_HZ;
        self.post.set_corner(POST_MIX_HZ, self.fs);
        self.hits.clear();
        self.search_buf.clear();
        self.since_search = 0;
        self.symbol.clear();
        self.key_down = false;
        self.mark_env = 0.0;
        self.space_env = 0.0;
        self.matched.clear();
        self.matched_sum = 0.0;
        self.settle = SETTLE_MS;
        self.quality = 0.0;
        self.env_hist.clear();
        self.key_hist.clear();
        self.tune_err = 0.0;
        self.have_mixed = false;
        self.hold_tune = 0;
        self.hsmm.reset();
        // Keep dit — the next station may be a similar speed.
    }

    fn step_envelope(&mut self, env: f32) {
        self.update_post_mix();
        self.hsmm.push(env);
        self.dit = self.hsmm.dit();
        self.mark_env = self.hsmm.mu_mark();
        self.space_env = self.hsmm.mu_space();
        let span = (self.mark_env - self.space_env).max(1e-9);
        let norm = ((env - self.space_env) / span).clamp(0.0, 1.0);
        let keyed = env > self.space_env + 0.45 * span;
        self.env_hist.push_back(norm);
        self.key_hist.push_back(keyed);
        while self.env_hist.len() > ENV_HIST {
            self.env_hist.pop_front();
            self.key_hist.pop_front();
        }
        self.on_thr = self.space_env + 0.55 * span;
        self.off_thr = self.space_env + 0.35 * span;
        self.key_down = keyed;
        self.quality = self.hsmm.quality();
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

    fn wants_agc(&self) -> bool {
        false
    }

    fn squelched(&self) -> bool {
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
            self.mix_rot.reset_phase();
            self.post.reset();
            self.symbol.clear();
            self.hsmm.reset();
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
        let span = (self.mark_env - self.space_env).max(1e-9);
        Some(CwView {
            env: self.env_hist.iter().copied().collect(),
            keyed: self.key_hist.iter().copied().collect(),
            on_thr: ((self.on_thr - self.space_env) / span).clamp(0.0, 1.0),
            off_thr: ((self.off_thr - self.space_env) / span).clamp(0.0, 1.0),
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
        let interval = if self.locked { FFT_SIZE * 4 } else { FFT_SIZE };
        if self.since_search >= interval && self.search_buf.len() >= FFT_SIZE {
            self.since_search = 0;
            self.search();
        }

        for &raw in samples {
            let mixed = self.mix(raw);
            let s = self.post.process(mixed);
            if self.have_mixed && s.norm() > 1e-9 && self.prev_mixed.norm() > 1e-9 {
                let d = s * self.prev_mixed.conj();
                let inst = d.arg() * self.fs / (2.0 * PI);
                if self.key_down {
                    self.tune_err = 0.92 * self.tune_err + 0.08 * inst.clamp(-80.0, 80.0);
                }
            }
            self.prev_mixed = s;
            self.have_mixed = true;

            let raw_env = self.smooth.process(s.norm());
            self.decim_ctr += 1;
            if self.decim_ctr < ENV_DECIM {
                continue;
            }
            self.decim_ctr = 0;

            if self.settle > 0 {
                self.settle -= 1;
                continue;
            }

            let matched_n = (0.22 * self.dit)
                .round()
                .clamp(4.0, self.env_rate * DIT_MAX_S) as usize;
            self.matched.push_back(raw_env);
            self.matched_sum += raw_env;
            while self.matched.len() > matched_n {
                self.matched_sum -= self.matched.pop_front().unwrap_or(0.0);
            }
            if self.matched_sum < 0.0 {
                self.matched_sum = self.matched.iter().sum();
            }
            let env = self.matched_sum / self.matched.len().max(1) as f32;
            self.step_envelope(env);
        }
        let evs = self.hsmm.flush();
        self.apply_events(&evs);
        self.quality = self.hsmm.quality();
        self.dit = self.hsmm.dit();
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

    /// Likelihood ratio of the HSMM's best path against an all-space null.
    fn confidence(&self) -> Option<f32> {
        Some(self.quality.clamp(0.0, 1.0))
    }

    fn speed(&self) -> Option<String> {
        let wpm = self.wpm();
        (wpm >= 1.0).then(|| format!("{wpm:.0}wpm"))
    }

    /// Stations that identified themselves since the last call. The scanner
    /// only recognises `CQ` and `DE` announcements, so an exchange in progress
    /// produces nothing — see `callscan` for why that is the right answer.
    fn take_messages(&mut self) -> Vec<FtMessage> {
        let (stamp, snr, hz) = (utc_hhmmss(), self.spot_snr(), self.mix_hz);
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

    fn reset(&mut self) {
        self.text.clear();
        self.dit = 0.06 * self.env_rate;
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
    let mut audio = Vec::with_capacity(iq.len() / decim + 1);
    let mut out = Vec::new();
    for &(off, _) in peaks {
        mix_decim_into(iq, fs as f32, off as f32, decim, &mut audio);
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
    // This is the scout path `Rotator` was written for: it mixes the whole
    // buffer once per candidate, and a `sin_cos` per sample here was costing
    // a quarter of a second per pass on a busy band.
    let mut osc = Rotator::new(-2.0 * PI * hz / fs);
    let decim = (fs / 1000.0).round().max(1.0) as usize;
    let mut env = Vec::with_capacity(buf.len() / decim + 1);
    let mut acc = 0.0f32;
    let mut n = 0usize;
    // ~180 Hz LPF after the mix so a neighbour (carrier, another CW)
    // does not sit on the envelope and hide the keying.
    let lpf_a = 1.0 - (-2.0 * PI * 180.0 / fs).exp();
    let mut lpf = Complex32::new(0.0, 0.0);
    for &s in buf {
        let mixed = s * osc.next();
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
    if marks.len() >= 4 {
        let mut ms = marks.clone();
        ms.sort_unstable();
        let short = ms[ms.len() / 4].max(1) as f32;
        let long = ms[ms.len() * 3 / 4] as f32;
        let r = long / short;
        if (1.75..=5.0).contains(&r) {
            quality = (quality + 0.25).min(1.0);
        } else if r < 1.4 {
            quality *= 0.70;
        }
    }
    if quality < 0.35 {
        return None;
    }
    Some((quality, snr))
}

fn morse_lookup(sym: &str) -> Option<char> {
    const TABLE: &[(&str, char)] = &[
        (".-", 'A'),
        ("-...", 'B'),
        ("-.-.", 'C'),
        ("-..", 'D'),
        (".", 'E'),
        ("..-.", 'F'),
        ("--.", 'G'),
        ("....", 'H'),
        ("..", 'I'),
        (".---", 'J'),
        ("-.-", 'K'),
        (".-..", 'L'),
        ("--", 'M'),
        ("-.", 'N'),
        ("---", 'O'),
        (".--.", 'P'),
        ("--.-", 'Q'),
        (".-.", 'R'),
        ("...", 'S'),
        ("-", 'T'),
        ("..-", 'U'),
        ("...-", 'V'),
        (".--", 'W'),
        ("-..-", 'X'),
        ("-.--", 'Y'),
        ("--..", 'Z'),
        ("-----", '0'),
        (".----", '1'),
        ("..---", '2'),
        ("...--", '3'),
        ("....-", '4'),
        (".....", '5'),
        ("-....", '6'),
        ("--...", '7'),
        ("---..", '8'),
        ("----.", '9'),
        (".-.-.-", '.'),
        ("--..--", ','),
        ("..--..", '?'),
        (".----.", '\''),
        ("-.-.--", '!'),
        ("-..-.", '/'),
        ("-.--.", '('),
        ("-.--.-", ')'),
        (".-...", '&'),
        ("---...", ':'),
        ("-.-.-.", ';'),
        ("-...-", '='),
        (".-.-.", '+'),
        ("-....-", '-'),
        ("..--.-", '_'),
        (".-..-.", '"'),
        ("...-..-", '$'),
        (".--.-.", '@'),
    ];
    TABLE.iter().find(|(s, _)| *s == sym).map(|(_, c)| *c)
}
