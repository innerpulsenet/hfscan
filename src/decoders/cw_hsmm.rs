//! Explicit-duration HMM (HSMM) for CW detection and timing (Stage 2).
//!
//! A period grid replaces the slicer/clock feedback loop: each candidate
//! dit length is scored independently, so a 35 WPM fist cannot drag the
//! estimator into a spiral. See `weak-signal-plan-3-cw.md` §7 for measured
//! scores — this module has not met the band ≥ 60 % / flat ≥ 88 % bars.

use std::f32::consts::E;

const N_STATE: usize = 6;
const MARK_DIT: usize = 0;
const MARK_DAH: usize = 1;
const GAP_ELEM: usize = 2;
const GAP_CHAR: usize = 3;
const GAP_WORD: usize = 4;
const IDLE: usize = 5;

const K_S: [f32; 5] = [1.0, 3.0, 1.0, 3.0, 7.0];
/// GapElement must not reach a character gap (3 T) or the two collapse.
const IS_MARK: [bool; 6] = [true, true, false, false, false, false];

/// Fractional timing jitter in the duration Gaussian.
const JITTER: f32 = 0.25;
const D_LO: f32 = 0.40;
const D_HI: f32 = 2.20;

const NEG: f32 = -1.0e30;

/// 5–50 WPM, matching `DIT_MIN_S` / `DIT_MAX_S` in `cw.rs`.
const DIT_MIN_S: f32 = 0.024;
const DIT_MAX_S: f32 = 0.24;

const WIN_S: f32 = 2.5;
const STEP_S: f32 = 0.60;
const LAG_S: f32 = 0.30;
/// Wait for this much envelope before the first decode, so the period grid
/// sees a whole character rather than three dits that look like a dah.
const MIN_DECODE_S: f32 = 1.15;
/// Trellis rate as a fraction of the 1 kHz envelope. A 50 WPM dit is
/// 12 samples at 500 Hz.
const DECIM: usize = 2;

/// Geometric Idle: leave-rate of about one per second.
fn idle_stay(env_rate: f32) -> f32 {
    let p_leave = (1.0 / env_rate).clamp(1e-4, 0.05);
    (1.0 - p_leave).ln()
}

#[derive(Clone, Copy, Debug)]
pub enum MorseEvent {
    Dit,
    Dah,
    CharGap,
    WordGap,
}

#[derive(Clone, Copy)]
struct Seg {
    state: u8,
    start: u32,
    end: u32,
}

pub struct LevelTracker {
    pub mean: f32,
    pub mu_mark: f32,
    pub mu_space: f32,
    have: bool,
    a_mean: f32,
    a_mark: f32,
    a_space: f32,
}

impl LevelTracker {
    pub fn new(env_rate: f32) -> Self {
        Self {
            mean: 0.0,
            mu_mark: 0.0,
            mu_space: 0.0,
            have: false,
            // Mean follows a fade; mark is faster still; space is the
            // band's noise and barely moves.
            a_mean: 1.0 - (-1.0 / (0.18 * env_rate)).exp(),
            a_mark: 1.0 - (-1.0 / (0.07 * env_rate)).exp(),
            a_space: 1.0 - (-1.0 / (0.70 * env_rate)).exp(),
        }
    }

    pub fn observe(&mut self, env: f32) {
        if !self.have {
            self.mean = env;
            self.mu_mark = env * 1.15;
            self.mu_space = env;
            self.have = true;
            return;
        }
        self.mean += self.a_mean * (env - self.mean);
        if env > self.mean {
            self.mu_mark += self.a_mark * (env - self.mu_mark);
        } else {
            self.mu_space += self.a_space * (env - self.mu_space);
        }
        if self.mu_mark < self.mu_space * 1.04 {
            self.mu_mark = self.mu_space * 1.04;
        }
    }

    pub fn reset(&mut self) {
        self.have = false;
        self.mean = 0.0;
        self.mu_mark = 0.0;
        self.mu_space = 0.0;
    }
}

pub struct HsmmDecoder {
    env_rate: f32,
    env: Vec<f32>,
    origin: u64,
    committed: u64,
    t_dit: f32,
    have_t: bool,
    since_global: u32,
    quality: f32,
    log_a: [[f32; N_STATE]; N_STATE],
    log_stay: f32,
    levels: LevelTracker,
    latest_mark: Vec<bool>,
    latest_origin: u64,
    since_poll: usize,
    /// Last committed event ended a character, so more space is a word gap.
    after_char: bool,
    scratch: Scratch,
}

struct Scratch {
    delta: Vec<f32>,
    back_d: Vec<u16>,
    back_s: Vec<u8>,
    best_in: Vec<f32>,
    best_from: Vec<u8>,
    c_mark: Vec<f32>,
    c_space: Vec<f32>,
    l_mark: Vec<f32>,
    l_space: Vec<f32>,
    mu_m: Vec<f32>,
    mu_s: Vec<f32>,
}

impl Scratch {
    fn new() -> Self {
        Self {
            delta: Vec::new(),
            back_d: Vec::new(),
            back_s: Vec::new(),
            best_in: Vec::new(),
            best_from: Vec::new(),
            c_mark: Vec::new(),
            c_space: Vec::new(),
            l_mark: Vec::new(),
            l_space: Vec::new(),
            mu_m: Vec::new(),
            mu_s: Vec::new(),
        }
    }

    fn ensure(&mut self, n: usize) {
        let cells = (n + 1) * N_STATE;
        grow_f32(&mut self.delta, cells);
        grow_u16(&mut self.back_d, cells);
        grow_u8(&mut self.back_s, cells);
        grow_f32(&mut self.best_in, cells);
        grow_u8(&mut self.best_from, cells);
        grow_f32(&mut self.c_mark, n + 1);
        grow_f32(&mut self.c_space, n + 1);
        grow_f32(&mut self.l_mark, n);
        grow_f32(&mut self.l_space, n);
        grow_f32(&mut self.mu_m, n);
        grow_f32(&mut self.mu_s, n);
    }
}

fn grow_f32(v: &mut Vec<f32>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}
fn grow_u16(v: &mut Vec<u16>, n: usize) {
    if v.len() < n {
        v.resize(n, 0);
    }
}
fn grow_u8(v: &mut Vec<u8>, n: usize) {
    if v.len() < n {
        v.resize(n, 0);
    }
}

fn build_log_a() -> [[f32; N_STATE]; N_STATE] {
    let mut a = [[NEG; N_STATE]; N_STATE];
    for &m in &[MARK_DIT, MARK_DAH] {
        a[m][GAP_ELEM] = 0.62f32.ln();
        a[m][GAP_CHAR] = 0.30f32.ln();
        a[m][GAP_WORD] = 0.06f32.ln();
        a[m][IDLE] = 0.02f32.ln();
    }
    for &g in &[GAP_ELEM, GAP_CHAR, GAP_WORD, IDLE] {
        a[g][MARK_DIT] = 0.55f32.ln();
        a[g][MARK_DAH] = 0.45f32.ln();
    }
    a
}

impl HsmmDecoder {
    pub fn new(env_rate: f32) -> Self {
        let t_dit = 0.06 * env_rate;
        Self {
            env_rate,
            env: Vec::with_capacity((WIN_S * env_rate) as usize + 8),
            origin: 0,
            committed: 0,
            t_dit,
            have_t: false,
            since_global: 99,
            quality: 0.0,
            log_a: build_log_a(),
            log_stay: idle_stay(env_rate),
            levels: LevelTracker::new(env_rate),
            latest_mark: Vec::new(),
            latest_origin: 0,
            since_poll: 0,
            after_char: false,
            scratch: Scratch::new(),
        }
    }

    pub fn dit(&self) -> f32 {
        self.t_dit
    }

    pub fn have_period(&self) -> bool {
        self.have_t
    }

    pub fn quality(&self) -> f32 {
        self.quality
    }

    pub fn mu_mark(&self) -> f32 {
        self.levels.mu_mark
    }

    pub fn mu_space(&self) -> f32 {
        self.levels.mu_space
    }

    pub fn reset(&mut self) {
        self.env.clear();
        self.origin = 0;
        self.committed = 0;
        self.have_t = false;
        self.since_global = 99;
        self.quality = 0.0;
        self.latest_mark.clear();
        self.since_poll = 0;
        self.after_char = false;
        self.levels.reset();
        // Keep t_dit — the next station may be a similar speed.
    }

    pub fn push(&mut self, env: f32) {
        self.levels.observe(env);
        self.env.push(env);
        self.since_poll += 1;
        self.trim();
    }

    fn trim(&mut self) {
        let win = (WIN_S * self.env_rate) as usize;
        let now = self.origin + self.env.len() as u64;
        let keep_from = self.committed.min(now.saturating_sub(win as u64));
        if keep_from > self.origin {
            let drop = (keep_from - self.origin) as usize;
            let drop = drop.min(self.env.len());
            if drop > 0 {
                self.env.drain(..drop);
                self.origin += drop as u64;
            }
        }
    }

    /// Decode newly arrived envelope. `force` commits a silent tail so the
    /// last character of an over is not held for the sliding-window lag.
    pub fn poll(&mut self, force: bool) -> Vec<MorseEvent> {
        let step = (STEP_S * self.env_rate) as usize;
        let min_n = (MIN_DECODE_S * self.env_rate) as usize;
        if self.env.len() < min_n {
            return Vec::new();
        }
        let idle_flush = self.have_t && self.tail_is_idle();
        if !force && !idle_flush && self.since_poll < step {
            return Vec::new();
        }
        self.since_poll = 0;
        self.decode_window(force || idle_flush)
    }

    /// Decode newly arrived envelope. Called once per audio block.
    pub fn flush(&mut self) -> Vec<MorseEvent> {
        let min_n = (MIN_DECODE_S * self.env_rate) as usize;
        if self.env.len() < min_n {
            return Vec::new();
        }
        self.since_poll = 0;
        self.decode_window(true)
    }

    fn decode_window(&mut self, force: bool) -> Vec<MorseEvent> {
        let n_full = self.env.len();
        if n_full < 48 {
            return Vec::new();
        }
        let n = n_full / DECIM;
        if n < 24 {
            return Vec::new();
        }
        self.scratch.ensure(n);

        let mut env_ds = vec![0.0f32; n];
        for i in 0..n {
            let base = i * DECIM;
            let mut s = 0.0;
            for k in 0..DECIM {
                s += self.env[base + k];
            }
            env_ds[i] = s / DECIM as f32;
        }

        seed_window_levels(&env_ds, &mut self.scratch.mu_m, &mut self.scratch.mu_s);
        fill_likelihoods(
            &env_ds,
            &self.scratch.mu_m,
            &self.scratch.mu_s,
            &mut self.scratch.l_mark,
            &mut self.scratch.l_space,
        );

        let contrast =
            mean_slice(&self.scratch.mu_m[..n]) / mean_slice(&self.scratch.mu_s[..n]).max(1e-12);
        let lag = (LAG_S * self.env_rate) as usize;
        let commit_end = if force && self.tail_is_idle() {
            n_full
        } else {
            n_full.saturating_sub(lag).max(1)
        };

        // Unimodal noise still has a 7/8:1/8 ratio of ~2. Real keyed CW,
        // even at 0 dB, sits higher once the matched filter has done its job.
        if contrast < 1.25 {
            self.quality *= 0.80;
            self.advance_commit(commit_end);
            return Vec::new();
        }

        let trellis_rate = self.env_rate / DECIM as f32;
        let hint = hint_dit(
            &env_ds,
            mean_slice(&self.scratch.mu_m[..n]),
            mean_slice(&self.scratch.mu_s[..n]),
        );
        // §5.5: re-estimate the period on a schedule, and search only a
        // narrow local grid between times. A sending station's WPM does not
        // drift on a two-second timescale, and the full coarse grid is the
        // single most expensive thing the decoder does — at every sixth
        // window it was most of the CW budget. Twenty-four windows is ~14 s.
        let need_global = !self.have_t || self.since_global >= 24;
        let mut candidates = if need_global {
            let coarse = period_grid(trellis_rate, true);
            let (t0, _, _) = self.best_period(&coarse, n, hint);
            self.since_global = 0;
            refine_around(t0, trellis_rate)
        } else {
            self.since_global += 1;
            refine_around(self.t_dit / DECIM as f32, trellis_rate)
        };
        if let Some(h) = hint {
            if h >= DIT_MIN_S * trellis_rate && h <= DIT_MAX_S * trellis_rate {
                let h = h.round().max(1.0);
                if !candidates.iter().any(|c| (*c - h).abs() < 0.5) {
                    candidates.push(h);
                }
            }
        }

        let (mut t_ds, mut score, mut path) = self.best_period(&candidates, n, hint);

        // EM pass 1: re-estimate levels from the decoded path, then let the
        // period re-settle locally against the improved levels.
        if !path.is_empty() {
            reest_levels(
                &env_ds,
                &path,
                n,
                &mut self.scratch.mu_m,
                &mut self.scratch.mu_s,
            );
            fill_likelihoods(
                &env_ds,
                &self.scratch.mu_m,
                &self.scratch.mu_s,
                &mut self.scratch.l_mark,
                &mut self.scratch.l_space,
            );
            let em_cands = refine_steps(t_ds, trellis_rate, 1);
            let (t_em, s_em, p_em) = self.best_period(&em_cands, n, hint);
            if s_em > score - 5.0 {
                t_ds = t_em;
                score = s_em;
                path = p_em;
            }
        }

        // EM pass 2: a second level re-estimation, for deep fades where one
        // pass has not yet pulled mu_mark down onto the faded signal.
        if contrast < 8.0 && !path.is_empty() {
            reest_levels(
                &env_ds,
                &path,
                n,
                &mut self.scratch.mu_m,
                &mut self.scratch.mu_s,
            );
            fill_likelihoods(
                &env_ds,
                &self.scratch.mu_m,
                &self.scratch.mu_s,
                &mut self.scratch.l_mark,
                &mut self.scratch.l_space,
            );
            let (s2, p2) = self.viterbi(n, t_ds);
            let s2 = s2 - 3.5 * p2.len() as f32;
            if s2 > score {
                score = s2;
                path = p2;
            }
        }

        let mut t_best = t_ds * DECIM as f32;
        if self.have_t {
            let jump = (t_best - self.t_dit).abs() / self.t_dit.max(1.0);
            if jump > 0.20 {
                let (old_raw, old_path) = self.viterbi(n, self.t_dit / DECIM as f32);
                let old_score = old_raw - 3.5 * old_path.len() as f32;
                if score < old_score {
                    t_best = self.t_dit;
                    score = old_score;
                    path = old_path;
                }
            }
        }
        for seg in &mut path {
            seg.start *= DECIM as u32;
            seg.end *= DECIM as u32;
        }

        let null = self.scratch.c_space[n] + (n.saturating_sub(1) as f32) * self.log_stay;
        let lr = (score - null) / n as f32;
        let inst_q = (1.0 - (-lr * 1.35).exp()).clamp(0.0, 1.0);
        self.quality = if self.quality <= 0.0 {
            inst_q
        } else {
            0.55 * self.quality + 0.45 * inst_q
        };

        let mark_frac = path_mark_frac(&path, n_full);
        let mixed = path_has_dit_and_dah(&path);
        let t_jump = (t_best - self.t_dit).abs() / self.t_dit.max(1.0);
        // An all-dit tail must not steal a period the rest of the over
        // already agreed on (it looks like dahs at 3×).
        let freeze_t = force && self.tail_is_idle() && !mixed && self.have_t;
        if !freeze_t
            && mark_frac >= 0.08
            && inst_q >= 0.18
            && (mixed || (self.have_t && t_jump < 0.22))
        {
            if self.have_t && t_jump < 0.30 {
                self.t_dit = 0.65 * self.t_dit + 0.35 * t_best;
            } else {
                self.t_dit = t_best;
            }
            self.have_t = true;
        }
        self.store_keyed(&path, n_full);

        // A path that barely beats all-space is the HSMM explaining noise.
        if inst_q < 0.14 || mark_frac < 0.06 {
            self.advance_commit(commit_end);
            return Vec::new();
        }
        if contrast < 2.3 && !marks_fit_clock(&path, t_best) {
            self.advance_commit(commit_end);
            return Vec::new();
        }

        let events = self.commit_path(&path, n_full, commit_end);
        events
    }

    fn best_period(
        &mut self,
        candidates: &[f32],
        n: usize,
        hint: Option<f32>,
    ) -> (f32, f32, Vec<Seg>) {
        let mut best_t = candidates.first().copied().unwrap_or(self.t_dit);
        let mut best_s = NEG;
        let mut best_p = Vec::new();
        let prefer = hint.unwrap_or(self.t_dit / DECIM as f32);
        let rate = self.env_rate / DECIM as f32;
        for &t in candidates {
            if t < DIT_MIN_S * rate * 0.95 || t > DIT_MAX_S * rate * 1.05 {
                continue;
            }
            let (score, path) = self.viterbi(n, t);
            // Short T can overfit the envelope with many tiny elements;
            // charge each extra segment so the true period wins ties.
            let mut score = score - 3.5 * path.len() as f32;
            // The 3× reading (dits as dahs) is the usual impostor.
            if let Some(h) = hint {
                if t < 0.55 * h {
                    score -= 40.0;
                }
            }
            let better = score > best_s + 1.0
                || (score > best_s - 1.0 && (t - prefer).abs() < (best_t - prefer).abs());
            if better {
                best_s = score;
                best_t = t;
                best_p = path;
            }
        }
        (best_t, best_s, best_p)
    }

    fn viterbi(&mut self, n: usize, t_dit: f32) -> (f32, Vec<Seg>) {
        let log_a = self.log_a;
        let log_stay = self.log_stay;
        prefix_sum(&self.scratch.l_mark[..n], &mut self.scratch.c_mark);
        prefix_sum(&self.scratch.l_space[..n], &mut self.scratch.c_space);

        let ranges: [(usize, usize); 5] = std::array::from_fn(|s| dur_range(s, t_dit, n));

        let d0 = &mut self.scratch.delta;
        let bd = &mut self.scratch.back_d;
        let bs = &mut self.scratch.back_s;
        let bi = &mut self.scratch.best_in;
        let bf = &mut self.scratch.best_from;
        let cm = &self.scratch.c_mark;
        let cs = &self.scratch.c_space;

        for s in 0..N_STATE {
            d0[s] = NEG;
            bi[s] = NEG;
            bf[s] = IDLE as u8;
        }
        d0[IDLE] = 0.0;
        fill_best_in(0, d0, bi, bf, &log_a);

        for t in 1..=n {
            for s in 0..5 {
                let (dmin, dmax) = ranges[s];
                let c = if IS_MARK[s] { cm } else { cs };
                let mean = K_S[s] * t_dit;
                let sigma = (JITTER * K_S[s] * t_dit).max(1.0);
                let inv = 1.0 / (2.0 * sigma * sigma);
                let ln_s = sigma.ln();
                let mut best = NEG;
                let mut best_d = dmin as u16;
                let mut best_prev = IDLE as u8;
                // §5.6, duration pruning: sample the duration range rather
                // than walking it. The Gaussian is smooth over `d`, so
                // twenty-eight probes locate its peak as well as fifty-six do
                // and cost half as much. Going below about twenty starts to
                // miss the peak on the long states and costs real copy.
                let stride = {
                    let span = dmax.saturating_sub(dmin).max(1);
                    (span / 28).max(1)
                };
                let mut d = dmin;
                while d <= dmax && d <= t {
                    let prev = t - d;
                    let dev = d as f32 - mean;
                    let dc = if s == GAP_WORD && d as f32 >= mean {
                        -ln_s
                    } else {
                        -dev * dev * inv - ln_s
                    };
                    let score = bi[prev * N_STATE + s] + dc + (c[t] - c[prev]);
                    if score > best {
                        best = score;
                        best_d = d as u16;
                        best_prev = bf[prev * N_STATE + s];
                    }
                    d += stride;
                }
                let i = t * N_STATE + s;
                d0[i] = best;
                bd[i] = best_d;
                bs[i] = best_prev;
            }

            // Idle is geometric: one sample, optional self-loop.
            let emit = cs[t] - cs[t - 1];
            let mut best = d0[(t - 1) * N_STATE + IDLE] + log_stay + emit;
            let mut from = IDLE as u8;
            for &m in &[MARK_DIT, MARK_DAH] {
                let sc = d0[(t - 1) * N_STATE + m] + log_a[m][IDLE] + emit;
                if sc > best {
                    best = sc;
                    from = m as u8;
                }
            }
            let i = t * N_STATE + IDLE;
            d0[i] = best;
            bd[i] = 1;
            bs[i] = from;

            fill_best_in(t, d0, bi, bf, &log_a);
        }

        let mut best_s = 0usize;
        let mut best_v = NEG;
        for s in 0..N_STATE {
            let v = d0[n * N_STATE + s];
            if v > best_v {
                best_v = v;
                best_s = s;
            }
        }
        let path = backtrace(n, best_s, bd, bs);
        (best_v, path)
    }

    fn store_keyed(&mut self, path: &[Seg], n: usize) {
        self.latest_mark.clear();
        self.latest_mark.resize(n, false);
        for seg in path {
            if IS_MARK[seg.state as usize] {
                let a = seg.start as usize;
                let b = (seg.end as usize).min(n);
                for x in &mut self.latest_mark[a..b] {
                    *x = true;
                }
            }
        }
        self.latest_origin = self.origin;
    }

    fn commit_path(&mut self, path: &[Seg], n: usize, commit_end: usize) -> Vec<MorseEvent> {
        let rel0 = self.committed.saturating_sub(self.origin) as usize;
        let rel0 = rel0.min(n);
        let commit_end = commit_end.min(n).max(rel0);
        // Only commit through a character/word boundary so a window that
        // ends mid-letter cannot lock a run of dits in as a dah.
        let last_gap = path
            .iter()
            .filter(|seg| {
                let s = seg.state as usize;
                let b = seg.end as usize;
                b <= commit_end
                    && b > rel0
                    && (s == GAP_CHAR
                        || s == GAP_WORD
                        || (s == IDLE && (seg.end - seg.start) as f32 / self.t_dit.max(1.0) >= 2.1))
            })
            .map(|seg| seg.end as usize)
            .max()
            .unwrap_or(rel0);
        let emit_end = last_gap;
        let mut out = Vec::new();
        for seg in path {
            let a = seg.start as usize;
            let b = seg.end as usize;
            if b <= rel0 || a >= emit_end {
                continue;
            }
            if a < rel0 {
                // We only commit through a gap, so a mark that straddles
                // rel0 is the first element of the next character — keep it.
                if !IS_MARK[seg.state as usize] {
                    continue;
                }
            }
            if b > emit_end {
                break;
            }
            match seg.state as usize {
                MARK_DIT => out.push(MorseEvent::Dit),
                MARK_DAH => out.push(MorseEvent::Dah),
                GAP_ELEM | GAP_CHAR | GAP_WORD | IDLE => {
                    let dits = (b.saturating_sub(a)) as f32 / self.t_dit.max(1.0);
                    if dits >= 4.6 {
                        out.push(MorseEvent::WordGap);
                    } else if dits >= 1.9 {
                        out.push(MorseEvent::CharGap);
                    }
                }
                _ => {}
            }
        }
        self.committed = self.origin + emit_end as u64;
        // A word gap split across windows looks like a char gap, then more
        // space. The leftover must not be stripped — it is the word space.
        if self.after_char {
            if let Some(MorseEvent::CharGap) = out.first() {
                out[0] = MorseEvent::WordGap;
            }
        } else {
            while matches!(out.first(), Some(MorseEvent::CharGap)) {
                out.remove(0);
            }
        }
        if matches!(out.last(), Some(MorseEvent::CharGap)) {
            self.after_char = true;
        } else if matches!(out.last(), Some(MorseEvent::WordGap)) {
            self.after_char = false;
        } else if out
            .iter()
            .any(|e| matches!(e, MorseEvent::Dit | MorseEvent::Dah))
        {
            self.after_char = false;
        }
        out
    }

    fn advance_commit(&mut self, commit_end: usize) {
        let rel0 = self.committed.saturating_sub(self.origin) as usize;
        let end = commit_end.max(rel0);
        self.committed = self.origin + end as u64;
    }

    fn tail_is_idle(&self) -> bool {
        // Just longer than a word gap (7 dits).
        let t = self.t_dit.max(8.0);
        let n = (8.5 * t) as usize;
        if self.env.len() < n {
            return false;
        }
        let mid = 0.5 * (self.levels.mu_mark + self.levels.mu_space);
        let tail = &self.env[self.env.len() - n..];
        tail.iter().filter(|&&e| e < mid).count() * 4 >= n * 3
    }
}

fn fill_best_in(
    t: usize,
    delta: &[f32],
    best_in: &mut [f32],
    best_from: &mut [u8],
    log_a: &[[f32; N_STATE]; N_STATE],
) {
    let base = t * N_STATE;
    for s in 0..N_STATE {
        let mut best = NEG;
        let mut from = 0u8;
        for sp in 0..N_STATE {
            let sc = delta[base + sp] + log_a[sp][s];
            if sc > best {
                best = sc;
                from = sp as u8;
            }
        }
        best_in[base + s] = best;
        best_from[base + s] = from;
    }
}

fn backtrace(n: usize, mut s: usize, back_d: &[u16], back_s: &[u8]) -> Vec<Seg> {
    let mut segs = Vec::new();
    let mut t = n;
    while t > 0 {
        let i = t * N_STATE + s;
        let d = (back_d[i] as usize).max(1).min(t);
        let prev = back_s[i] as usize;
        segs.push(Seg {
            state: s as u8,
            start: (t - d) as u32,
            end: t as u32,
        });
        t -= d;
        s = prev.min(N_STATE - 1);
        if segs.len() > n + 2 {
            break;
        }
    }
    segs.reverse();
    coalesce(&segs)
}

fn is_space_state(s: u8) -> bool {
    matches!(s as usize, GAP_ELEM | GAP_CHAR | GAP_WORD | IDLE)
}

fn coalesce(segs: &[Seg]) -> Vec<Seg> {
    let mut out: Vec<Seg> = Vec::new();
    for &s in segs {
        if let Some(last) = out.last_mut() {
            if last.end == s.start
                && (last.state == s.state
                    || (is_space_state(last.state) && is_space_state(s.state)))
            {
                last.end = s.end;
                if is_space_state(s.state) && is_space_state(last.state) {
                    // Relabel by total length later; keep the longer-kind label
                    // so a 3 T gap isn't left as GapElement.
                    if s.state > last.state {
                        last.state = s.state;
                    }
                }
                continue;
            }
        }
        out.push(s);
    }
    out
}

fn dur_range(s: usize, t_dit: f32, n: usize) -> (usize, usize) {
    let k = K_S[s];
    let (lo, hi) = match s {
        GAP_ELEM => (0.45, 1.65),
        GAP_CHAR => (1.80, 5.20),
        GAP_WORD => (4.50, 14.0),
        _ => (D_LO, D_HI),
    };
    let dmin = ((lo * k * t_dit).round() as usize).max(1);
    let mut dmax = ((hi * k * t_dit).round() as usize).max(dmin);
    // For GapChar/GapWord, `hi` is already in dit units, not k-scaled.
    if s == GAP_CHAR || s == GAP_WORD {
        let dmin2 = ((lo * t_dit).round() as usize).max(1);
        let dmax2 = ((hi * t_dit).round() as usize).max(dmin2);
        return (dmin2, dmax2.min(n));
    }
    dmax = dmax.min(n);
    (dmin, dmax)
}

fn prefix_sum(x: &[f32], c: &mut [f32]) {
    c[0] = 0.0;
    for i in 0..x.len() {
        c[i + 1] = c[i] + x[i];
    }
}

fn fill_likelihoods(env: &[f32], mu_m: &[f32], mu_s: &[f32], l_m: &mut [f32], l_s: &mut [f32]) {
    for i in 0..env.len() {
        let span = (mu_m[i] - mu_s[i]).max(1e-9);
        // Normalised Gaussians so a fade cannot inflate the emission.
        let x = (env[i] - mu_s[i]) / span;
        const VAR: f32 = 0.12;
        l_m[i] = -0.5 * (x - 1.0) * (x - 1.0) / VAR;
        l_s[i] = -0.5 * x * x / VAR;
    }
}

fn seed_window_levels(env: &[f32], mu_m: &mut [f32], mu_s: &mut [f32]) {
    let n = env.len();
    if n == 0 {
        return;
    }
    let mut sorted: Vec<f32> = env.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = sorted[n / 8].max(1e-12);
    let hi = sorted[n * 7 / 8].max(lo * 1.05);
    let mid = sorted[n / 2];
    let a_mean = 1.0 - E.powf(-1.0 / (0.16 * n as f32).max(8.0));
    let a_hi = 1.0 - E.powf(-1.0 / (0.08 * n as f32).max(4.0));
    let a_lo = 1.0 - E.powf(-1.0 / (0.40 * n as f32).max(8.0));
    let mut mean = mid;
    let mut mark = hi;
    let mut space = lo;
    for i in 0..n {
        mean += a_mean * (env[i] - mean);
        if env[i] > mean {
            mark += a_hi * (env[i] - mark);
        } else {
            space += a_lo * (env[i] - space);
        }
        if mark < space * 1.05 {
            mark = space * 1.05;
        }
        mu_m[i] = mark;
        mu_s[i] = space;
    }
}

fn reest_levels(env: &[f32], path: &[Seg], n: usize, mu_m: &mut [f32], mu_s: &mut [f32]) {
    let mut mark_on = vec![false; n];
    for seg in path {
        if IS_MARK[seg.state as usize] {
            let a = seg.start as usize;
            let b = (seg.end as usize).min(n);
            for x in &mut mark_on[a..b] {
                *x = true;
            }
        }
    }
    let mut s_sum = 0.0;
    let mut s_n = 0.0;
    for i in 0..n {
        if !mark_on[i] {
            s_sum += env[i];
            s_n += 1.0;
        }
    }
    let space = if s_n > 4.0 {
        s_sum / s_n
    } else {
        mean_slice(&mu_s[..n])
    };
    let mut mark = mean_slice(&mu_m[..n]).max(space * 1.1);
    let a = 0.18f32;
    for i in 0..n {
        if mark_on[i] {
            mark += a * (env[i] - mark);
        }
        mu_m[i] = mark.max(space * 1.05);
        mu_s[i] = space;
    }
}

fn marks_fit_clock(path: &[Seg], t_dit: f32) -> bool {
    let t = t_dit.max(1.0);
    let mut near = 0usize;
    let mut n = 0usize;
    for seg in path {
        if !IS_MARK[seg.state as usize] {
            continue;
        }
        let r = (seg.end - seg.start) as f32 / t;
        n += 1;
        if (r - 1.0).abs() < 0.55 || (r - 3.0).abs() < 1.35 {
            near += 1;
        }
    }
    n >= 2 && near * 10 >= n * 5
}

fn path_has_dit_and_dah(path: &[Seg]) -> bool {
    let mut dit = false;
    let mut dah = false;
    for seg in path {
        match seg.state as usize {
            MARK_DIT => dit = true,
            MARK_DAH => dah = true,
            _ => {}
        }
    }
    dit && dah
}

fn path_mark_frac(path: &[Seg], n: usize) -> f32 {
    if n == 0 {
        return 0.0;
    }
    let mut marks = 0usize;
    for seg in path {
        if IS_MARK[seg.state as usize] {
            marks += (seg.end - seg.start) as usize;
        }
    }
    marks as f32 / n as f32
}

fn mean_slice(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().sum::<f32>() / x.len() as f32
}

fn period_grid(env_rate: f32, coarse: bool) -> Vec<f32> {
    let t_min = DIT_MIN_S * env_rate;
    let t_max = DIT_MAX_S * env_rate;
    let ratio = if coarse { 1.20 } else { 1.10 };
    let mut out = Vec::new();
    let mut t = t_min;
    let mut last_round = 0i32;
    while t <= t_max * 1.001 {
        let r = t.round() as i32;
        if r != last_round && r > 0 {
            out.push(r as f32);
            last_round = r;
        }
        t *= ratio;
    }
    if out.is_empty() {
        out.push((0.06 * env_rate).round().max(1.0));
    }
    out
}

fn hint_dit(env: &[f32], mu_m: f32, mu_s: f32) -> Option<f32> {
    let thr = 0.45 * mu_m + 0.55 * mu_s;
    let mut marks = Vec::new();
    let mut run = 0i32;
    let mut on = false;
    for &e in env {
        if e > thr {
            if !on {
                run = 0;
                on = true;
            }
            run += 1;
        } else if on {
            if run >= 2 {
                marks.push(run as f32);
            }
            on = false;
        }
    }
    if marks.len() < 3 {
        return None;
    }
    marks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(marks[marks.len() / 4].max(1.0))
}

fn refine_around(t: f32, env_rate: f32) -> Vec<f32> {
    refine_steps(t, env_rate, 2)
}

/// `steps` grid points either side of `t`. The EM pass only needs the period
/// to re-settle against the new levels, so it asks for one.
fn refine_steps(t: f32, env_rate: f32, steps: i32) -> Vec<f32> {
    let lo = DIT_MIN_S * env_rate;
    let hi = DIT_MAX_S * env_rate;
    let mut out = Vec::new();
    for k in -steps..=steps {
        let v = (t * 1.12f32.powi(k)).clamp(lo, hi).round();
        if v >= 1.0 && !out.iter().any(|x: &f32| (*x - v).abs() < 0.1) {
            out.push(v);
        }
    }
    if out.is_empty() {
        out.push(t.round().max(1.0));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_paris(t: usize) -> Vec<f32> {
        // PARIS: .--.  .-  .-.  ..  ...
        let letters: &[&str] = &[".--.", ".-", ".-.", "..", "..."];
        let mut env = vec![0.08; t * 8];
        for (li, pat) in letters.iter().enumerate() {
            for (ei, el) in pat.chars().enumerate() {
                let n = if el == '.' { t } else { 3 * t };
                env.extend(std::iter::repeat(1.0).take(n));
                if ei + 1 < pat.len() {
                    env.extend(std::iter::repeat(0.08).take(t));
                }
            }
            let gap = if li + 1 == letters.len() {
                7 * t
            } else {
                3 * t
            };
            env.extend(std::iter::repeat(0.08).take(gap));
        }
        env.extend(std::iter::repeat(0.08).take(t * 10));
        env
    }

    #[test]
    fn viterbi_recovers_paris_and_period() {
        let env_rate = 1000.0;
        let t = 60usize;
        let env = synth_paris(t);
        let mut d = HsmmDecoder::new(env_rate);
        for e in env {
            d.push(e);
        }
        let ev = d.poll(true);
        let mut sym = String::new();
        let mut text = String::new();
        for e in ev {
            match e {
                MorseEvent::Dit => sym.push('.'),
                MorseEvent::Dah => sym.push('-'),
                MorseEvent::CharGap | MorseEvent::WordGap => {
                    text.push_str(&sym);
                    text.push(' ');
                    sym.clear();
                }
            }
        }
        text.push_str(&sym);
        assert!(
            (d.dit() - t as f32).abs() < 12.0,
            "dit {} want {t}",
            d.dit()
        );
        assert!(
            text.contains(".--.") && text.contains("..."),
            "symbols {text:?}"
        );
    }
}
