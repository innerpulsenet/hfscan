//! Morse (CW) decoder with adaptive speed tracking and nearby-tone lock.
//!
//! Envelope detection with hysteresis slices the keying. Dit length is
//! estimated from a short/long cluster of recent marks so a station that
//! speeds up or slows down is followed instead of being decoded as
//! garbage. A passband scout (FFT + keyed-envelope score) finds CW tones
//! near the cursor and mixes the best one to DC; `n` hops to the next.

use super::callscan::{CallScanner, utc_hhmmss};
use super::cwlex::{self, Sym};
use super::{BgCopy, CwView, Decoder, FtMessage};
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
const MARK_HIST: usize = 24;
/// Marks the speed tracker clusters over.
///
/// `MARK_HIST` is sized for the structure check in `morse_clock`, which wants
/// as long a run as it can get before deciding a fist is not Morse. The speed
/// tracker wants the opposite: when an operator changes speed, every mark
/// still in the window from the old fist drags the clusters toward the old
/// clock, and with 24 marks a 15→32 WPM change takes longer to flush than a
/// short over lasts — the tracker settles halfway, at 18 WPM, where the new
/// dahs look like slow dits and hold it there. Clustering over the most
/// recent marks only shortens that to about a word.
const RECLUSTER_MARKS: usize = 10;
/// Envelope history for the scope pane (~0.7 s at 1 kHz).
const ENV_HIST: usize = 700;
/// 5–50 WPM. Below that is not Morse; above it noise ripples chop.
const DIT_MIN_S: f32 = 0.024; // ~50 WPM
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

/// Where the slicer arms and releases, as a fraction of the span between the
/// tracked floor and peak.
///
/// These were 0.52 and 0.32, which are the right numbers if `peak` is the
/// level of a mark. It is not: `peak` attacks instantly and decays over 2.5 s,
/// so it is a max-hold, and a max-hold over a noisy envelope sits well above
/// the marks it is supposed to be measuring. The bias grows as the signal
/// weakens, which is precisely when it hurts — the span inflates, the arming
/// threshold rides up with it, and real marks stop reaching it. Copy did not
/// degrade gracefully at 0 dB so much as fall through the floor.
///
/// Compensating in the coefficients is not elegant, but it is what measures
/// best. Slicing against `mark_env`/`space_env` instead — the levels that are
/// honestly a mark and a space — was tried and is worse (87.4 % against
/// 89.2 %): those settle slowly and only under conditions the slicer itself
/// has to establish first, so they lag exactly when the signal is moving.
///
/// Swept jointly over a four-seed grid, the surface is a broad plateau across
/// roughly 0.38–0.46 arming and 0.28–0.32 releasing, all of it 1.5 to 2.4
/// points above the old pair; these sit in the middle of it. The release
/// threshold must not go much below 0.28 — at 0.24 the score collapses,
/// because a mark that never releases runs into the next one.
const ON_SPAN_K: f32 = 0.40;
const OFF_SPAN_K: f32 = 0.30;

/// The carrier-SNR ladder, as `peak/floor` ratios.
///
/// Three separate commits each added one of these to fix a different bug —
/// the noise gate, the false-'E' bursts, and gating quality on carrier SNR —
/// and none of them mentions the others. They were read once as three
/// inconsistent references to the same quantity, which they are not, so what
/// they actually are is written down here.
///
/// They form a squelch ladder, strict to open and progressively looser to
/// hold, and the ordering `ARM > HOLD > DROP` is the only part that matters:
///
/// - `SNR_ARM` gates the key-up to key-down transition in `step_envelope`,
///   and *only* that one — `OFF_SPAN_K` releases a mark that has already
///   started. Below it no mark ever begins, so nothing downstream ever runs.
/// - `SNR_HOLD` gates emitting a finished letter in `push_symbol`. It is
///   looser than `SNR_ARM` on purpose and is not redundant with it, because
///   `peak` decays over 2.5 s and `floor` attacks over 2 while a letter is
///   being assembled: a letter that armed cleanly can finish under a signal
///   that has since faded, and this decides how far it may fade first.
/// - `SNR_DROP` is where `classify_mark` gives up, discards the clock and
///   returns to warm-up.
///
/// `snr_factor` in `classify_mark` is the same ladder read as a ramp rather
/// than a step, which is why it is derived here rather than written out: it
/// has to reach full quality exactly where the slicer is willing to arm, and
/// an independent literal would silently stop doing that the moment
/// `SNR_ARM` moved.
///
/// None of the three has been swept. They are a plausible ladder that
/// measures no worse than the alternatives tried, not a measured optimum —
/// if they are ever tuned, tune them against `bench_cw_band`, together.
/// How many stations in the passband are decoded at once.
///
/// The passband holds two to four, and the decoder used to take one. Each
/// extra tone costs a full mixer, a four-pole filter, an envelope detector
/// and a slicer over every audio sample — the per-sample chain is the whole
/// cost, since the FFT search is shared and runs once for all of them.
///
/// This is the CPU budget §6 of the plan never set, and the omission it calls
/// "the one that allowed all of this". It is a hard cap rather than a target:
/// `cw_cost_scales_with_tones` measures the real ratio.
pub const MAX_TONES: usize = 4;
/// Characters of background copy held per tone before the oldest are
/// dropped. Roughly two overs' worth; the primary's copy is never capped
/// because it is drained on every `process` call.
const BG_TEXT_MAX: usize = 4096;
/// Closest two tones may sit and still be worth running separately.
///
/// Not a resolution limit — the search resolves better than this — but a
/// usefulness one. The post-mix filter is 60 to 150 Hz wide, so two tones
/// closer than its narrowest setting are listening to the same keying and
/// would produce the same copy twice, once well and once badly. Measured
/// directly: at 40 Hz a station at +50 Hz got both the primary and a
/// background tone, and both transcribed it.
const TONE_SEP_HZ: f32 = 60.0;
/// Searches a tone may go unfound before it is retired.
///
/// The search runs every `FFT_SIZE * 4` samples once locked, so at 8 kHz this
/// is about eight seconds of grace — long enough to ride out a QSB null on a
/// station that is still there, short enough that a station which has stopped
/// sending frees its slot within an over.
const TONE_MISSES: u32 = 4;

const SNR_ARM: f32 = 2.8;
const SNR_HOLD: f32 = 1.8;
const SNR_DROP: f32 = 1.5;
/// The `peak/floor` span over which quality ramps from nothing to full.
const SNR_RAMP: f32 = SNR_ARM - 1.0;
// The ordering is the invariant, so it fails the build rather than the band.
// Inverting any pair silently strands a signal in a state it cannot leave:
// HOLD above ARM discards letters the slicer was willing to start, and DROP
// above HOLD re-warms the clock under a signal still good enough to copy.
const _: () = assert!(SNR_ARM > SNR_HOLD && SNR_HOLD > SNR_DROP && SNR_DROP > 1.0);

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
    /// click. Only called when the tracked speed has moved materially.
    fn set_corner(&mut self, hz: f32, fs: f32) {
        self.a = 1.0 - (-2.0 * PI * hz / fs).exp();
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
fn morse_clock(marks: &[f32], gaps: &[f32]) -> Option<(f32, f32)> {
    if marks.len() < 3 {
        return None;
    }
    let mut v: Vec<f32> = marks.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let (c1_init, c2_init) = (v[0], v[v.len() - 1]);
    if c2_init <= c1_init * 1.55 {
        // Homogeneous mark sequence (all dits or all dahs with human jitter).
        let med = v[v.len() / 2];
        if med < 1e-6 {
            return None;
        }
        let near = v.iter().filter(|&&x| (x - med).abs() <= 0.35 * med).count();
        if near * 10 >= v.len() * 7 {
            // Disambiguate all-dits vs all-dahs using inter-element gaps if available:
            if gaps.len() >= 2 {
                let mut g = gaps.to_vec();
                g.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let short_gap = g[g.len() / 4];
                if short_gap > 0.0 && short_gap <= 0.55 * med {
                    // Marks are dahs (~3 dits), inter-element gaps are dits (~1 dit)
                    let dit = (med / 3.0).max(short_gap);
                    return Some((dit, med));
                }
            }
            return Some((med, 3.0 * med));
        }
    }
    // Two-mean cluster, seeded at the extremes.
    let (mut c1, mut c2) = (c1_init, c2_init);
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
    // A dah is ~3 dits; allow heavy straight key (1.75) up to fast bug (5.0).
    let r = long / short;
    if !(1.75..=5.0).contains(&r) {
        return None;
    }
    // Every mark must sit close to one of the two lengths.
    let near = v
        .iter()
        .filter(|&&x| {
            let d = (x - short).abs().min((x - long).abs());
            d <= 0.38
                * if (x - short).abs() < (x - long).abs() {
                    short
                } else {
                    long
                }
        })
        .count();
    if near * 10 < v.len() * 7 {
        return None;
    }
    Some((short, long))
}

/// Whether the gaps between marks are keyed by the same clock as the marks.
fn gaps_match_clock(gaps: &[f32], dit: f32) -> bool {
    if gaps.len() < 2 || dit <= 0.0 {
        return true;
    }
    let mut v: Vec<f32> = gaps.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // The inter-element gap is the commonest, so the lower half of the
    // distribution is the one to compare against a dit.
    let short = v[v.len() / 4];
    (0.35..=2.8).contains(&(short / dit))
}

/// A CW tone found by the span scout or the in-passband searcher.
#[derive(Clone, Debug)]
pub struct CwHit {
    pub offset_hz: f32,
    pub score: f32,
    pub quality: f32,
}

/// One station's worth of decode: a mixer onto its tone, the filter and
/// envelope behind it, the slicer, the clock and the text it has produced.
///
/// This is everything that used to be `CwDecoder`'s body. It is a separate
/// struct because a 400 Hz passband holds two to four stations and the
/// decoder used to pick one and discard the rest — `CwDecoder` now runs one
/// of these per tone the search finds.
struct Tone {
    /// Stable identity. The primary is tracked by this rather than by its
    /// offset, because offsets move every search and float equality on a
    /// tracked frequency is not an identity.
    id: u32,
    fs: f32,
    env_rate: f32,
    smooth: OnePole,
    matched: VecDeque<f32>,
    matched_sum: f32,
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
    /// The word being assembled, held until a gap so `cwlex` can rescore it
    /// whole. Each character keeps the elements it came from.
    word_buf: Vec<Sym>,
    text: String,
    /// Word scanner for pskreporter spots (`CQ ... CALL`, `DE CALL CALL`).
    scan: CallScanner,
    idle: f32,
    started: bool,
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
    /// Whether the peak/floor trackers have been seeded.
    have_env: bool,
    /// Envelope samples still to discard while the filters fill.
    settle: u32,

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
    /// Searches since this tone was last seen by the search. A tone that
    /// stops being found is retired, which is what bounds the work.
    missed: u32,
}

pub struct CwDecoder {
    fs: f32,
    /// One decoder per tone the search is currently holding, newest last.
    /// `primary` indexes the one the user is listening to.
    tones: Vec<Tone>,
    primary_id: u32,
    next_id: u32,
    locked: bool,
    lock_score: f32,
    /// Searches to skip after a manual nudge so AFC does not fight the user.
    hold_tune: u32,
    hits: Vec<CwHit>,
    /// Confidence a tone must reach before its copy leaves the decoder.
    copy_floor: f32,
    search_buf: Vec<Complex32>,
    since_search: usize,
    fft: Arc<dyn Fft<f32>>,
    fft_buf: Vec<Complex32>,
    window: Vec<f32>,
}

impl Tone {
    fn new(id: u32, fs: f32, offset_hz: f32) -> Self {
        let env_rate = fs / ENV_DECIM as f32;
        Self {
            id,
            fs,
            env_rate,
            smooth: OnePole::new(0.003 * fs),
            matched: VecDeque::new(),
            matched_sum: 0.0,
            post: NarrowLpf::new(fs),
            post_hz: POST_MIX_HZ,
            peak: 0.0,
            floor: 0.0,
            peak_decay: 1.0 - (-1.0f32 / (2.5 * env_rate)).exp(),
            floor_attack: 1.0 - (-1.0f32 / (2.0 * env_rate)).exp(),
            floor_decay: 1.0 - (-1.0f32 / (0.10 * env_rate)).exp(),
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
            word_buf: Vec::new(),
            text: String::new(),
            scan: CallScanner::new(),
            idle: 0.0,
            started: false,
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
            have_env: false,
            settle: SETTLE_MS,
            mix_hz: offset_hz,
            mix_rot: Rotator::new(0.0),
            mix_rate_hz: 0.0,
            missed: 0,
        }
    }

    fn wpm(&self) -> f32 {
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

    /// Match the post-mix filter to the tracked keying speed.
    fn update_post_mix(&mut self) {
        let want = if self.warming {
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

    fn clamp_dit(&mut self) {
        let lo = DIT_MIN_S * self.env_rate;
        let hi = DIT_MAX_S * self.env_rate;
        self.dit = self.dit.clamp(lo, hi);
        self.dah = self.dah.clamp(2.0 * self.dit, 4.4 * self.dit);
    }

    /// How long the slicer must hold a new state before the edge is real.
    fn debounce_env(&self) -> f32 {
        (0.25 * self.dit).clamp(0.006 * self.env_rate, 0.030 * self.env_rate)
    }

    fn push_symbol(&mut self) {
        if self.symbol.is_empty() {
            return;
        }
        let sym = std::mem::take(&mut self.symbol);
        // Only emit symbols if there is carrier SNR
        let snr_ok = self.peak > SNR_HOLD * self.floor.max(1e-9);
        if !snr_ok {
            return;
        }
        if let Some(c) = morse_lookup(&sym) {
            let is_dit = sym == ".";
            if is_dit && self.quality < 0.40 {
                return;
            }
            self.word_buf.push(Sym { ch: c, elems: sym });
        } else {
            // An unmatched pattern still carries evidence: keep the elements
            // so the rescorer can measure against them even though no letter
            // can be named.
            self.word_buf.push(Sym { ch: '*', elems: sym });
            self.quality *= 0.85;
        }
    }

    /// Emit the buffered word, rescored against ham CW's vocabulary.
    ///
    /// Called wherever a word boundary is recognised, before the space is
    /// written, so the transcript stays in order.
    fn flush_word(&mut self) {
        if self.word_buf.is_empty() {
            return;
        }
        let w = std::mem::take(&mut self.word_buf);
        let decoded: String = w.iter().map(|s| s.ch).collect();
        let word = match cwlex::rescore_word(&w) {
            Some(fixed) => fixed.to_string(),
            None => decoded,
        };
        let word = match cwlex::split_prefix(&word) {
            Some((head, rest)) => format!("{head} {rest}"),
            None => word,
        };
        for c in word.chars() {
            self.text.push(c);
            self.scan.push(c);
        }
    }

    fn on_mark_end(&mut self, len: f32) {
        if len > 6.0 * self.dah || len > 1.4 * self.env_rate {
            self.symbol.clear();
            self.warmup.clear();
            return;
        }
        if !self.warming && len < 0.012 * self.env_rate {
            // Discard sub-12ms noise glitch
            return;
        }
        if self.warming {
            self.warmup.push((len, true));
            let n_marks = self.warmup.iter().filter(|(_, m)| *m).count();
            if n_marks >= WARMUP_MARKS {
                self.flush_warmup();
            }
            return;
        }
        self.classify_mark(len);

        if self.marks.len() >= 12 {
            let recent: Vec<f32> = self.marks.iter().copied().collect();
            let gaps: Vec<f32> = Vec::new();
            if morse_clock(&recent, &gaps).is_none() {
                if self.peak < SNR_DROP * self.floor.max(1e-9) || self.quality < 0.05 {
                    self.symbol.clear();
                    self.warmup.clear();
                    self.warming = true;
                    self.marks.clear();
                }
            }
        }
    }

    /// Set the clock from the held elements, then replay them.
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
        let clock = morse_clock(&marks, &gaps).filter(|(dit, _)| gaps_match_clock(&gaps, *dit));
        let (dit, dah) = match clock {
            Some((dit, dah)) => (dit, dah),
            None => {
                if marks.len() >= WARMUP_MARKS {
                    if let Some(i) = self.warmup.iter().position(|(_, m)| *m) {
                        self.warmup.drain(..=i);
                    } else {
                        self.warmup.clear();
                    }
                    return;
                } else if self.dit >= DIT_MIN_S * self.env_rate {
                    (self.dit, self.dah)
                } else {
                    return;
                }
            }
        };
        self.dit = dit;
        self.dah = dah;
        self.clamp_dit();
        self.marks.clear();
        for &m in marks.iter().take(MARK_HIST) {
            self.marks.push_back(m);
        }

        let mut held: Vec<(f32, bool)> = std::mem::take(&mut self.warmup);
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

        if ratio < 0.40 {
            return;
        }

        // Classify against the midpoint of the *tracked* dit and dah, not a
        // fixed 2.0: a fist with light dahs (or stretched dits) moves the
        // boundary with it instead of straddling it.
        let boundary = ((self.dit + self.dah) / (2.0 * dit)).clamp(1.50, 2.75);
        let is_dah = ratio >= boundary;
        self.symbol.push(if is_dah { '-' } else { '.' });

        self.marks.push_back(len);
        if self.marks.len() > MARK_HIST {
            self.marks.pop_front();
        }

        let alpha = if self.acquire > 0 { 0.20 } else { 0.08 };
        if self.acquire > 0 {
            self.acquire -= 1;
        }
        let dah_r = (self.dah / dit).clamp(2.0, 4.2);
        // The lower gate matches the one that rejected the mark outright a few
        // lines up. With the two set differently there was a band of lengths —
        // 0.40 to 0.50 dits — good enough to be decoded as a dit but not good
        // enough to say anything about the clock, and that band is exactly
        // where a real dit lands when the operator has just doubled their
        // speed: against the old clock a 32 WPM dit reads 0.47. The shorter
        // dits were being decoded and then ignored while the same fist's dahs
        // read 1.4 and were taken for slow dits, which pushes the estimate the
        // wrong way. Letting them speak is worth a fifth of a point of score.
        //
        // It is not what rescues a doubled speed on its own — `RECLUSTER_MARKS`
        // is — but it stops this loop pulling against that one.
        if !is_dah && ratio < boundary && ratio > 0.40 {
            self.dit = (1.0 - alpha) * self.dit + alpha * len;
        } else if is_dah && ratio >= boundary && ratio < 5.0 {
            let a = alpha * 0.5;
            self.dah = (1.0 - a) * self.dah + a * len;
            self.dit = (1.0 - a) * self.dit + a * (len / dah_r);
        }
        self.clamp_dit();

        if self.marks.len() >= 6 {
            self.recluster();
        }

        let fit = if is_dah {
            1.0 - (len / self.dah.max(1e-6) - 1.0).abs().min(1.0)
        } else {
            1.0 - (ratio - 1.0).abs().min(1.0)
        };
        let snr_factor = ((self.peak / self.floor.max(1e-9) - 1.0) / SNR_RAMP).clamp(0.0, 1.0);
        let sample_q = fit * snr_factor;

        let dah_count = self.marks.iter().filter(|&&m| m >= 1.7 * self.dit).count();
        let dit_count = self.marks.iter().filter(|&&m| m < 1.7 * self.dit).count();
        let single_type_consistent = if self.marks.len() >= 6 {
            let mean = self.marks.iter().sum::<f32>() / self.marks.len() as f32;
            let variance = self.marks.iter().map(|m| (m - mean).powi(2)).sum::<f32>() / self.marks.len() as f32;
            variance.sqrt() <= 0.38 * mean
        } else {
            true
        };
        let coherent = (dah_count >= 1 && dit_count >= 1) || single_type_consistent;
        let target_q = if coherent { sample_q } else { sample_q * 0.45 };
        let q_alpha = if self.acquire > 0 { 0.28 } else { 0.12 };
        self.quality = (1.0 - q_alpha) * self.quality + q_alpha * target_q;
    }

    /// Two-mean cluster of recent mark lengths. If the short cluster is a
    /// coherent dit and the long one is ~3×, snap toward it when the
    /// operator has changed speed.
    fn recluster(&mut self) {
        let mut v: Vec<f32> = self.marks.iter().rev().take(RECLUSTER_MARKS).copied().collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let c1_init = v[0];
        let c2_init = *v.last().unwrap();
        if (c2_init - c1_init).abs() < 1e-6 {
            return;
        }
        if c2_init <= c1_init * 1.55 {
            let med = v[v.len() / 2];
            if med < 1e-6 {
                return;
            }
            let is_dah_cluster = (med - self.dah).abs() < (med - self.dit).abs() && med > 1.8 * self.dit;
            if is_dah_cluster {
                self.dah = 0.75 * self.dah + 0.25 * med;
                self.clamp_dit();
            }
            return;
        }
        let (mut c1, mut c2) = (c1_init, c2_init);
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
            let n = v
                .iter()
                .filter(|&&x| (x - c1).abs() <= (x - c2).abs())
                .count();
            (c1, c2, n)
        } else {
            let n = v
                .iter()
                .filter(|&&x| (x - c2).abs() <= (x - c1).abs())
                .count();
            (c2, c1, n)
        };
        let n_long = v.len() - n_short;
        if n_short < 2 || n_long < 1 || short < 1e-6 {
            return;
        }
        let r = long / short;
        if !(1.75..=4.8).contains(&r) {
            return;
        }
        let rel = (short - self.dit).abs() / self.dit.max(1e-6);
        if rel > 0.22 {
            // Speed changed: adopt the new clock directly from the measured cluster
            self.dit = short;
            self.dah = long;
            self.acquire = 8;
        } else {
            self.dit = 0.80 * self.dit + 0.20 * short;
            self.dah = 0.80 * self.dah + 0.20 * long;
        }
        self.clamp_dit();
    }

    fn on_space_end(&mut self, len: f32) {
        if self.warming {
            if !self.warmup.is_empty() {
                self.warmup.push((len, false));
            }
            return;
        }
        // Adaptive character boundary: 2.1 dits cleanly separates elements (1 dit) and characters (3 dits)
        if len >= 2.1 * self.dit {
            self.push_symbol();
            // A character gap is 3 dits and a word gap 7, so the honest place
            // to split them is 5. It sat at 5.5, which leant toward calling a
            // word gap a character gap — and the slicer leans the same way:
            // it arms below the middle of the span and releases below that
            // again, so every mark reads a little long and every gap a little
            // short. Two biases in the same direction ran words together on
            // the live capture (`CVA PT6T` came out `CVAPT6T`). Five is the
            // midpoint and measures a shade better than 5.5 as well.
            if len >= 5.0 * self.dit {
                self.flush_word();
                self.scan.push(' ');
                if !self.text.ends_with(' ') && !self.text.is_empty() {
                    self.text.push(' ');
                }
            }
        }
    }

    /// Mix, filter, detect and slice one audio sample for this tone.
    fn feed(&mut self, raw: Complex32) {
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
            return;
        }
        self.decim_ctr = 0;

        if self.settle > 0 {
            self.settle -= 1;
            return;
        }

        let matched_n = (0.35 * self.dit)
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

    /// Move this tone onto a new offset, resetting only what the move
    /// invalidates. The clock is kept — it is the same operator.
    fn retune(&mut self, hz: f32) {
        if (hz - self.mix_hz).abs() > 12.0 {
            self.mix_rot.reset_phase();
            self.post.reset();
        }
        self.mix_hz = hz;
    }
}

impl CwDecoder {
    pub fn new(fs: f64) -> Self {
        let fs = fs as f32;
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let window: Vec<f32> = (0..FFT_SIZE)
            .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / FFT_SIZE as f32).cos())
            .collect();
        Self {
            fs,
            // One tone at DC until the search says otherwise, so a decoder
            // handed audio it never searches behaves as it always did.
            tones: vec![Tone::new(0, fs, 0.0)],
            primary_id: 0,
            next_id: 1,
            locked: false,
            lock_score: 0.0,
            hold_tune: 0,
            hits: Vec::new(),
            copy_floor: 0.0,
            search_buf: Vec::with_capacity(FFT_SIZE * 2),
            since_search: 0,
            fft,
            fft_buf: vec![Complex32::new(0.0, 0.0); FFT_SIZE],
            window,
        }
    }

    pub fn wpm(&self) -> f32 {
        self.cur().wpm()
    }

    /// Index of the tone the user is listening to. `sync_tones` guarantees
    /// the primary is never retired, so the fallback is unreachable in
    /// practice and exists only so this cannot panic.
    fn primary(&self) -> usize {
        self.tones
            .iter()
            .position(|t| t.id == self.primary_id)
            .unwrap_or(0)
    }

    /// The tone the user is listening to. Always present.
    fn cur(&self) -> &Tone {
        &self.tones[self.primary()]
    }

    fn cur_mut(&mut self) -> &mut Tone {
        let i = self.primary();
        &mut self.tones[i]
    }

    /// How many tones are currently running. Never exceeds `MAX_TONES`.
    // Background copy reaches the app through `take_messages` as spots, which
    // is where the value is; these two expose the raw streams for anything
    // that wants them. The TUI does not show background text yet — that is a
    // question about panes and screen room, not about the decoder.
    #[allow(dead_code)]
    pub fn tone_count(&self) -> usize {
        self.tones.len()
    }

    /// Copy from every tone except the one being listened to, drained.
    ///
    /// These are the stations Stage 4 exists to recover: they sit in the same
    /// 400 Hz passband as the primary and used to be discarded unheard.
    /// Returned as `(offset_hz, text)` so a caller can tell them apart.
    pub fn take_background(&mut self) -> Vec<BgCopy> {
        let (i, floor) = (self.primary(), self.copy_floor);
        self.tones
            .iter_mut()
            .enumerate()
            .filter(|(k, t)| *k != i && !t.text.is_empty())
            .map(|(_, t)| BgCopy {
                hz: t.mix_hz,
                quality: t.quality.clamp(0.0, 1.0),
                // Drained whether or not it clears the bar: a tone held back
                // must not accumulate until it happens to pass and then emit
                // a minute of stale copy in one go.
                text: match t.quality >= floor {
                    true => std::mem::take(&mut t.text),
                    false => {
                        t.text.clear();
                        String::new()
                    }
                },
            })
            .filter(|b| !b.text.is_empty())
            .collect()
    }

    /// Start a tone on `hz` if the budget allows and nothing is there yet.
    fn spawn(&mut self, hz: f32) {
        if self.tones.len() >= MAX_TONES
            || self.tones.iter().any(|t| (t.mix_hz - hz).abs() < TONE_SEP_HZ)
        {
            return;
        }
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        self.tones.push(Tone::new(id, self.fs, hz));
    }

    fn mix_hz(&self) -> f32 {
        self.cur().mix_hz
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
        let cur_hz = self.mix_hz();
        self.hits.sort_by(|a, b| {
            let dist_a = (a.offset_hz - cur_hz).abs();
            let dist_b = (b.offset_hz - cur_hz).abs();
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

        // The primary is decided first and alone, and only then are the
        // other stations reconciled. Doing it the other way round lets a tone
        // spawned this same search be adopted as the primary, handing the user
        // a cold slicer that has missed the start of the transmission —
        // measured as "NQ CQ DE ..." where the station sent "CQ CQ DE ...".
        if self.locked
            && let Some(h) = self
                .hits
                .iter()
                .find(|h| (h.offset_hz - cur_hz).abs() < 35.0)
                .cloned()
        {
            let hz = 0.85 * cur_hz + 0.15 * h.offset_hz;
            self.cur_mut().mix_hz = hz;
            self.lock_score = h.score;
            self.sync_tones();
            return;
        }
        if let Some(h) = self.hits.first().cloned() {
            // Once locked, prefer a tone already running on that station: it
            // has a clock and a warm slicer, which is the point of keeping
            // them alive. Before the first lock there is nothing to inherit
            // and the primary simply takes the frequency, as it always did.
            let adopt = self
                .locked
                .then(|| {
                    self.tones
                        .iter()
                        .find(|t| (t.mix_hz - h.offset_hz).abs() < TONE_SEP_HZ)
                        .map(|t| t.id)
                })
                .flatten();
            match adopt {
                Some(id) => self.primary_id = id,
                None => self.cur_mut().retune(h.offset_hz),
            }
            self.lock_score = h.score;
            self.locked = true;
        }
        self.sync_tones();
    }

    /// Match the running tones to what the search just found: retune the ones
    /// that moved, start one for a hit nobody is on, retire the ones whose
    /// station has gone.
    ///
    /// Retirement is what bounds the work, and it is deliberately not
    /// conditional on the tone decoding anything — §7.11 of the plan is a
    /// lockup caused by exactly that coupling, where state was retained until
    /// a decode succeeded and so grew without bound on the signals where it
    /// never did. A tone is retired when the *search* stops seeing it, which
    /// is a judgement made outside the decoder it is judging.
    fn sync_tones(&mut self) {
        // A background tone the primary has since tuned onto is a duplicate:
        // both would transcribe the same station.
        let (primary_id, primary_hz) = (self.primary_id, self.mix_hz());
        self.tones
            .retain(|t| t.id == primary_id || (t.mix_hz - primary_hz).abs() >= TONE_SEP_HZ);
        for t in self.tones.iter_mut() {
            t.missed += 1;
        }
        let hits: Vec<(f32, usize)> = self
            .hits
            .iter()
            .enumerate()
            .map(|(i, h)| (h.offset_hz, i))
            .collect();
        for (hz, rank) in hits {
            match self
                .tones
                .iter()
                .position(|t| (t.mix_hz - hz).abs() < TONE_SEP_HZ)
            {
                Some(i) => {
                    self.tones[i].missed = 0;
                    // The primary's offset is steered by `search` itself, so
                    // leave it alone here — smoothing it in both places pulls
                    // it twice per search and measurably degrades its copy.
                    if self.tones[i].id != self.primary_id
                        && (self.tones[i].mix_hz - hz).abs() > 3.0
                    {
                        self.tones[i].mix_hz = 0.85 * self.tones[i].mix_hz + 0.15 * hz;
                    }
                }
                // Only the strongest `MAX_TONES` hits are ever worth a slot,
                // so a weak hit cannot displace the budget on a busy band.
                None if rank < MAX_TONES && self.tones.len() < MAX_TONES => {
                    self.spawn(hz);
                }
                None => {}
            }
        }
        // The primary is exempt from retirement: the user chose it, and a
        // fade must not silently move them to another station.
        self.tones
            .retain(|t| t.missed <= TONE_MISSES || t.id == primary_id);
        if self.tones.is_empty() {
            self.tones.push(Tone::new(primary_id, self.fs, 0.0));
        }
    }

    fn clear_lock_state(&mut self) {
        self.locked = false;
        self.lock_score = 0.0;
        self.hits.clear();
        self.search_buf.clear();
        self.since_search = 0;
        self.hold_tune = 0;
        // Every tone goes, including the primary: `hop` means the operator has
        // given up on this stretch of band, not that one station faded.
        self.tones.clear();
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        self.tones.push(Tone::new(id, self.fs, 0.0));
        self.primary_id = id;
    }
}

impl Tone {
    /// SNR in a 2500 Hz reference bandwidth, for the spot report.
    ///
    /// `mark_env/space_env` is measured through the post-mix filter, so it is
    /// an SNR in *that* filter's noise bandwidth and has to be referred to
    /// 2500 Hz before anyone else can read it.
    ///
    /// That correction used to be a hardcoded 12.3 dB, which is exactly
    /// `10*log10(2500/147)` — the 147 Hz a four-pole filter at the old fixed
    /// `POST_MIX_HZ` passes. `POST_MIX_K` later made the corner track the
    /// keying, down to `POST_MIX_MIN_HZ`, and nothing here noticed: at 20 WPM
    /// the filter now sits at 60 Hz, the correction should be 16.3 dB, and
    /// every spot went out about 4 dB optimistic. Measured against
    /// synthesised signals it read +1.7 dB at 25 dB SNR rising to +4.8 dB at
    /// 0 dB, against a commit that had calibrated it to 1 dB.
    ///
    /// Deriving it from `post_hz` is what stops that happening again — the
    /// term is not a constant of this decoder, it is a property of a filter
    /// that moves.
    fn spot_snr(&self) -> f32 {
        if self.mark_env <= 0.0 || self.space_env <= 0.0 {
            return -24.0;
        }
        let ratio = (self.mark_env / self.space_env).max(1.0);
        // The four-pole cascade passes 147 Hz of noise at a 150 Hz corner.
        let enbw = self.post_hz * (147.0 / POST_MIX_HZ);
        let to_2500 = 10.0 * (2500.0 / enbw.max(1.0)).log10();
        // What is left is empirical, and stays empirical: `mark_env` carries
        // the noise as well as the signal, so the ratio is (S+N)/N and reads
        // high, most at the weak end where the correction matters most.
        (20.0 * ratio.log10() - 1.05 - to_2500 - 3.2).clamp(-24.0, 20.0)
    }

    /// Return this tone to the state a fresh one starts in, keeping only its
    /// offset and its clock — the next station on the same frequency is
    /// usually the same operator, or at least a similar speed.
    fn reset_decode(&mut self) {
        self.scan.reset();
        self.mix_rot.reset_phase();
        self.post.reset();
        self.post_hz = POST_MIX_HZ;
        self.post.set_corner(POST_MIX_HZ, self.fs);
        self.symbol.clear();
        self.word_buf.clear();
        self.marks.clear();
        self.missed = 0;
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
        self.mark_env = 0.0;
        self.space_env = 0.0;
        self.have_env = false;
        self.matched.clear();
        self.matched_sum = 0.0;
        self.settle = SETTLE_MS;
        self.quality = 0.0;
        self.env_hist.clear();
        self.key_hist.clear();
        self.tune_err = 0.0;
        self.have_mixed = false;
        // Keep dit — the next station may be a similar speed.
    }

    fn step_envelope(&mut self, env: f32) {
        self.update_post_mix();
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
        if !self.key_down {
            let a = if env < self.floor {
                self.floor_decay
            } else if self.pending == 0.0 && self.run > 0.3 * self.dit {
                self.floor_attack
            } else {
                0.0
            };
            self.floor += (env - self.floor) * a;
        }

        let span = (self.peak - self.floor).max(1e-9);
        let on_thr = self.floor + ON_SPAN_K * span;
        let off_thr = self.floor + OFF_SPAN_K * span;
        let snr_ok = self.peak > SNR_ARM * self.floor.max(1e-9);

        let next = if self.key_down {
            env > off_thr
        } else {
            env > on_thr && snr_ok
        };

        self.on_thr = on_thr;
        self.off_thr = off_thr;

        let settled = if self.key_down {
            env > on_thr
        } else {
            env < off_thr
        };
        if settled && self.pending == 0.0 && self.run > 0.4 * self.dit {
            let (level, clean_is_up) = if self.key_down {
                (&mut self.mark_env, true)
            } else {
                (&mut self.space_env, false)
            };
            if *level <= 0.0 {
                *level = env;
            } else {
                let a = if (env > *level) == clean_is_up {
                    0.05
                } else {
                    0.002
                };
                *level += (env - *level) * a;
            }
        }

        let norm = ((env - self.floor) / span).clamp(0.0, 1.0);
        self.env_hist.push_back(norm);
        self.key_hist.push_back(next);
        while self.env_hist.len() > ENV_HIST {
            self.env_hist.pop_front();
            self.key_hist.pop_front();
        }

        if next == self.key_down {
            self.run += 1.0 + self.pending;
            self.pending = 0.0;
        } else {
            self.pending += 1.0;
            if self.pending >= self.debounce_env() {
                let len = self.run;
                self.run = self.pending;
                self.pending = 0.0;
                if self.key_down {
                    if len >= 0.010 * self.env_rate {
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
            if self.idle > 6.0 * self.dit {
                if !self.symbol.is_empty() {
                    self.push_symbol();
                }
                if (self.idle - 6.0 * self.dit) <= 1.0 {
                    self.flush_word();
                    self.scan.push(' ');
                    if !self.text.ends_with(' ') && !self.text.is_empty() {
                        self.text.push(' ');
                    }
                }
            }
            if self.idle > 6.0 * self.env_rate {
                self.acquire = self.acquire.max(12);
                if !self.warming {
                    self.symbol.clear();
                    self.warming = true;
                    self.marks.clear();
                }
            }
            if self.warming && !self.warmup.is_empty() {
                let marks_count = self.warmup.iter().filter(|(_, m)| *m).count();
                if marks_count >= 3 && self.started && self.idle > 6.0 * self.dit {
                    self.flush_warmup();
                } else if self.idle > 1.5 * self.env_rate {
                    self.warmup.clear();
                    self.warming = true;
                }
            }
        }
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
        self.mix_hz()
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
        let cur = self.mix_hz();
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
            // The station may already be running as a background tone, in
            // which case hopping to it inherits a clock and a warm slicer
            // rather than starting from nothing — the point of Stage 4.
            match self.tones.iter().find(|t| (t.mix_hz - h.offset_hz).abs() < 40.0) {
                Some(t) => self.primary_id = t.id,
                None => {
                    let t = self.cur_mut();
                    t.retune(h.offset_hz);
                    t.symbol.clear();
                    t.acquire = 12;
                }
            }
            self.lock_score = h.score;
            self.locked = true;
            h.offset_hz
        })
    }

    fn candidate_hz(&self) -> Vec<f32> {
        self.hits.iter().map(|h| h.offset_hz).collect()
    }

    fn nudge_lock(&mut self, delta_hz: f32) -> Option<f32> {
        let hz = (self.mix_hz() + delta_hz).clamp(-SEARCH_HZ, SEARCH_HZ);
        let t = self.cur_mut();
        t.mix_hz = hz;
        t.tune_err = 0.0;
        self.locked = true;
        self.hold_tune = 6;
        Some(hz)
    }

    fn set_copy_floor(&mut self, floor: f32) {
        self.copy_floor = floor;
    }

    fn take_background(&mut self) -> Vec<BgCopy> {
        CwDecoder::take_background(self)
    }

    fn cw_view(&self) -> Option<CwView> {
        let t = self.cur();
        let span = (t.peak - t.floor).max(1e-9);
        Some(CwView {
            env: t.env_hist.iter().copied().collect(),
            keyed: t.key_hist.iter().copied().collect(),
            on_thr: ((t.on_thr - t.floor) / span).clamp(0.0, 1.0),
            off_thr: ((t.off_thr - t.floor) / span).clamp(0.0, 1.0),
            lock_hz: t.mix_hz,
            tune_err_hz: t.tune_err,
            wpm: t.wpm(),
            quality: t.quality,
            key_down: t.key_down,
            symbol: t.symbol.clone(),
            dit_ms: t.dit / t.env_rate * 1000.0,
            locked: self.locked,
            hits: self.hits.clone(),
            live: self.tones.iter().map(|t| t.mix_hz).collect(),
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

        // Every tone sees every sample. This is the whole cost of Stage 4:
        // the search above is shared, but each station needs its own mixer,
        // filter, envelope and slicer because each is keyed by a different
        // operator at a different speed.
        for &raw in samples {
            for t in self.tones.iter_mut() {
                t.feed(raw);
            }
        }
        // Only the primary's copy goes to the pane. The rest is not thrown
        // away — `take_messages` spots it and `take_background` returns it —
        // but interleaving four stations into one transcript is unreadable.
        let i = self.primary();
        for (k, t) in self.tones.iter_mut().enumerate() {
            // Bounded whether or not anyone ever reads it. §7.11 of the plan
            // is a lockup caused by state that grew until a decode succeeded;
            // this buffer is capped on length alone, so a caller that never
            // drains it costs a fixed amount rather than an increasing one.
            if k != i && t.text.len() > BG_TEXT_MAX {
                let cut = t.text.len() - BG_TEXT_MAX / 2;
                t.text.drain(..cut);
            }
        }
        std::mem::take(&mut self.tones[i].text)
    }

    fn status(&self) -> String {
        let wpm = self.wpm();
        let extra = if self.hits.len() > 1 {
            format!(" +{}", self.hits.len() - 1)
        } else {
            String::new()
        };
        if self.locked && self.mix_hz().abs() > 1.0 {
            format!("{wpm:.0} WPM lock {:+.0}Hz{extra}", self.mix_hz())
        } else {
            format!("{wpm:.0} WPM{extra}")
        }
    }

    /// How cleanly recent marks fell into the dit and dah buckets. Keying that
    /// is really noise through the slicer has random mark lengths, so they
    /// straddle the boundary and this collapses; well-sent CW sits near 1.
    fn confidence(&self) -> Option<f32> {
        Some(self.cur().quality.clamp(0.0, 1.0))
    }

    fn speed(&self) -> Option<String> {
        let wpm = self.wpm();
        (wpm >= 1.0).then(|| format!("{wpm:.0}wpm"))
    }

    /// Stations that identified themselves since the last call. The scanner
    /// only recognises `CQ` and `DE` announcements, so an exchange in progress
    /// produces nothing — see `callscan` for why that is the right answer.
    ///
    /// Every tone is asked, not just the one being listened to. This is where
    /// Stage 4 actually pays: a station calling CQ behind a louder neighbour
    /// used to be discarded unheard, and now it reaches pskreporter with its
    /// own offset and its own SNR.
    fn take_messages(&mut self) -> Vec<FtMessage> {
        let stamp = utc_hhmmss();
        let (mut out, floor) = (Vec::new(), self.copy_floor);
        for t in self.tones.iter_mut() {
            let (snr, hz) = (t.spot_snr(), t.mix_hz);
            let calls = t.scan.take_calls();
            // Each station is held to the bar by its own confidence. Gating
            // the lot on the lock's would let a clean background station be
            // suppressed by a poor primary, and a poor one ride out on a good
            // primary — a spot is a claim about one station, not the passband.
            if t.quality < floor {
                continue;
            }
            out.extend(calls.into_iter().map(|call| FtMessage {
                stamp: stamp.clone(),
                snr_db: snr,
                dt_sec: 0.0,
                freq_hz: hz,
                text: format!("CQ {call}"),
            }));
        }
        out
    }

    fn reset(&mut self) {
        for t in self.tones.iter_mut() {
            t.text.clear();
            t.dit = 0.06 * t.env_rate;
            t.dah = 0.18 * t.env_rate;
            t.reset_decode();
        }
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

/// The Morse alphabet, read in both directions.
///
/// `morse_lookup` decodes; `morse_elements` re-encodes, which is what lets
/// `cwlex` measure how far a mis-copied word is from a real one in the space
/// the errors actually happen in — elements — rather than in characters.
pub(crate) const MORSE: &[(&str, char)] = &[
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

fn morse_lookup(sym: &str) -> Option<char> {
    MORSE.iter().find(|(s, _)| *s == sym).map(|(_, c)| *c)
}

/// The element pattern for a character, or `None` if it has no Morse form.
pub(crate) fn morse_elements(c: char) -> Option<&'static str> {
    let c = c.to_ascii_uppercase();
    MORSE.iter().find(|(_, t)| *t == c).map(|(s, _)| *s)
}
