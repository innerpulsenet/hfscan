//! Baudot RTTY decoder. Default 45.45 baud / 170 Hz shift, which covers almost
//! all amateur HF RTTY.
//!
//! Detection is a pair of matched filters, one per tone, each an
//! integrate-and-dump across exactly one bit — the optimal detector for a
//! constant-envelope tone — dumped on the framer's own bit boundary. Per-tone
//! envelope trackers normalise them (automatic threshold correction), so a
//! selective fade that takes one tone 20 dB down still slices correctly.
//!
//! Tuning is found rather than assumed: a quarter-second transform looks for
//! two tones a known shift apart and centres on their midpoint, scoring each
//! candidate by its *weaker* tone so that a lone carrier or a pile-up of FT8
//! signals cannot pass for an FSK pair however loud it is. A slow AFC trims
//! the residual from there.
//!
//! Shift polarity is detected, not assumed. Both slicings of every bit are
//! framed in parallel and the one that actually frames — start bit a space,
//! stop bit a mark — wins; `r` still forces it either way.

use super::callscan::{utc_hhmmss, CallScanner};
use super::{Decoder, FtMessage};
use crate::dsp::Rotator;
use num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Good frames one polarity must lead the other by before it is believed.
/// A few characters' worth: enough that a run of luck does not decide it,
/// few enough that copy starts almost immediately.
const POLARITY_MARGIN: i64 = 6;
/// ...and the fraction of its frames that must have closed on a mark stop
/// bit. The margin alone is not enough: on an empty frequency the stop-bit
/// check passes about half the time by chance, so both polarities creep up
/// together and a random walk eventually hands one of them a lead. Real
/// Baudot frames essentially every character, so demanding most of them
/// separates a station from a coin toss.
const POLARITY_FRAMED: f32 = 0.72;
/// Frames counted before both tallies are halved. Keeps the ratio a measure
/// of how framing is going now rather than of everything ever seen.
const FRAME_WINDOW: u32 = 64;
/// Samples the coarse acquisition transforms — a quarter second at 8 kHz,
/// eleven bits, which resolves the tone pair to a few hertz.
const ACQ_N: usize = 2048;
/// How far off centre the coarse search looks. The decoder asks for
/// `shift + 2*baud + 100` Hz of passband and the tones sit half a shift
/// either side of centre, so beyond about this the chain filter has already
/// removed what we would be looking for.
const ACQ_RANGE_HZ: f32 = 120.0;

pub struct RttyDecoder {
    fs: f32,
    baud: f32,
    shift: f32,
    samples_per_bit: f32,
    /// Mixers for the tone pair and for the centre correction. Rotators
    /// rather than a `sin_cos` per sample: this loop ran two of them on every
    /// audio sample of every RTTY slot, which is exactly what `Rotator` exists
    /// to avoid.
    tone_rot: Rotator,
    centre_rot: Rotator,
    /// Rate currently programmed into `centre_rot`, so it is only recomputed
    /// when the AFC has actually moved.
    centre_rate_hz: f32,
    /// Sliding correlation of each tone.
    ///
    /// The optimal detector for a constant-envelope tone over a bit is an
    /// integrate-and-dump across exactly that bit. What stood here before was
    /// a one-pole leaky integrator with a time constant of a *third* of a bit,
    /// which is a far wider noise bandwidth than the signal needs and threw
    /// away most of the coherent gain the tone offers.
    mark: ToneCorr,
    space: ToneCorr,
    mark_acc: Complex32,
    space_acc: Complex32,
    mark_peak: f32,
    space_peak: f32,
    /// Normal and reversed slicings of the same discriminator output. Which
    /// one is right is decided by which one frames, not assumed.
    framers: [Framer; 2],
    /// The winning polarity once known, or forced by the user.
    polarity: Option<usize>,
    /// Whether the polarity is still being worked out, or the user has said.
    auto: bool,
    afc_hz: f32,
    /// Coarse centre offset from the tone-pair search, Hz. Kept apart from
    /// `afc_hz`, which stays the narrow ±10 Hz trim it always was.
    centre_hz: f32,
    acq_buf: Vec<Complex32>,
    acq_pos: usize,
    acq_fill: usize,
    fft: Arc<dyn Fft<f32>>,
    fft_buf: Vec<Complex32>,
    /// Mean power in the whole passband, and mean power in the two tone
    /// filters. Their difference is the noise, which is the one way to get an
    /// SNR out of this demodulator that is not defeated by its own
    /// normalisation. Comparing the two tone filters saturates around 18 dB,
    /// where the quieter one stops hearing the band and starts hearing its
    /// neighbour's skirt. The discriminator cannot be used either: the peak
    /// trackers that scale it decay over 1.5 s, so it swings by tens of
    /// percent on the bit pattern alone. Wideband power has neither problem —
    /// the noise there is far above any leakage, and nothing has normalised
    /// it away. Zero means not yet seen.
    band_pwr: f32,
    tone_pwr: f32,
    /// Word scanner for pskreporter spots (`CQ ... CALL`, `DE CALL CALL`).
    /// Fed only from copy that has already survived the polarity vote — the
    /// losing framer's output is the same bits read upside down.
    scan: CallScanner,
}

/// A sliding boxcar correlation against one tone.
///
/// The window is carried as a running sum so the cost is one add and one
/// subtract per sample rather than a pass over the window, and re-summed
/// exactly once per full buffer so f32 rounding cannot accumulate.
struct ToneCorr {
    hist: Vec<Complex32>,
    pos: usize,
    sum: Complex32,
    since_refresh: usize,
}

impl ToneCorr {
    fn new(n: usize) -> Self {
        Self {
            hist: vec![Complex32::new(0.0, 0.0); n.max(2)],
            pos: 0,
            sum: Complex32::new(0.0, 0.0),
            since_refresh: 0,
        }
    }

    fn refresh(&mut self) {
        self.sum = self.hist.iter().sum();
        self.since_refresh = 0;
    }

    fn reset(&mut self) {
        self.hist.iter_mut().for_each(|s| *s = Complex32::new(0.0, 0.0));
        self.pos = 0;
        self.sum = Complex32::new(0.0, 0.0);
        self.since_refresh = 0;
    }

    /// Push one sample and return the mean over the window.
    fn push(&mut self, x: Complex32) -> Complex32 {
        let n = self.hist.len();
        self.sum += x - self.hist[self.pos];
        self.hist[self.pos] = x;
        self.pos = (self.pos + 1) % n;
        self.since_refresh += 1;
        if self.since_refresh >= n {
            self.refresh();
        }
        self.sum / n as f32
    }
}

#[derive(PartialEq)]
enum State {
    Idle,
    Data,
}

/// One start-stop framer, slicing bits one way round.
///
/// A mark and a space are only distinguishable by convention — which sideband
/// the signal arrived on decides whether the high tone is the mark, and the
/// receiver cannot know. What it can know is that Baudot frames every
/// character between a space start bit and a mark stop bit, so the wrong
/// polarity fails to frame almost every character while the right one frames
/// nearly all of them. Running both and counting is a far more reliable
/// answer than any rule about tone order.
struct Framer {
    /// Whether this framer reads the low tone as mark.
    invert: bool,
    state: State,
    counter: f32,
    bits: u8,
    nbits: u32,
    figs: bool,
    text: String,
    good: u32,
    err: u32,
}

impl Framer {
    fn new(invert: bool) -> Self {
        Self {
            invert,
            state: State::Idle,
            counter: 0.0,
            bits: 0,
            nbits: 0,
            figs: false,
            text: String::new(),
            good: 0,
            err: 0,
        }
    }

    fn clear(&mut self) {
        let invert = self.invert;
        *self = Framer::new(invert);
    }

    /// How convincingly this polarity is framing: good characters net of the
    /// failures, which the wrong polarity produces in abundance.
    fn score(&self) -> i64 {
        self.good as i64 - self.err as i64
    }

    /// Fraction of attempted frames that closed on a mark stop bit.
    fn framed(&self) -> f32 {
        let total = self.good + self.err;
        if total == 0 {
            0.0
        } else {
            self.good as f32 / total as f32
        }
    }

    fn feed(&mut self, f: f32, thr: f32, samples_per_bit: f32) {
        let mark = if self.invert { f < thr } else { f > thr };
        match self.state {
            State::Idle => {
                // A start bit is a space; sample it half a bit in to confirm.
                if !mark {
                    self.counter += 1.0;
                    if self.counter >= samples_per_bit * 0.5 {
                        self.state = State::Data;
                        // We are half a bit into the start bit, so counting
                        // one full bit from here lands mid data bit 0.
                        self.counter = 0.0;
                        self.bits = 0;
                        self.nbits = 0;
                    }
                } else {
                    self.counter = 0.0;
                }
            }
            State::Data => {
                self.counter += 1.0;
                if self.counter >= samples_per_bit {
                    self.counter -= samples_per_bit;
                    // The discriminator is now a boxcar across a whole bit, so
                    // this single sample already *is* the average of the bit —
                    // and averaging several of them would fold in windows that
                    // straddle the bit before, which is ISI rather than noise
                    // rejection. Measured, dropping the old mid-bit average
                    // took 10 dB copy from 68% to 86% and restored the first
                    // character at 20 dB.
                    let m = if self.invert { f < thr } else { f > thr };
                    if self.nbits < 5 {
                        // Baudot sends the least significant bit first.
                        if m {
                            self.bits |= 1 << self.nbits;
                        }
                        self.nbits += 1;
                    } else {
                        // Stop bit must be mark; otherwise we lost framing.
                        if m {
                            let code = self.bits;
                            self.emit(code);
                            self.good = self.good.saturating_add(1);
                        } else {
                            self.err = self.err.saturating_add(1);
                        }
                        if self.good + self.err >= FRAME_WINDOW {
                            self.good /= 2;
                            self.err /= 2;
                        }
                        self.state = State::Idle;
                        self.counter = 0.0;
                    }
                }
            }
        }
    }

    fn emit(&mut self, code: u8) {
        match code {
            0x1F => self.figs = false, // LTRS
            0x1B => self.figs = true,  // FIGS
            0x04 => {
                // USOS (Unshift On Space): revert to LTRS on space to prevent
                // FIGS lockup from noise or lost shift characters.
                self.figs = false;
                self.text.push(' ');
            }
            0x08 => self.text.push('\n'), // CR
            0x02 => {}                     // LF (CR already breaks the line)
            0x00 => {}
            _ => {
                let table = if self.figs { FIGS } else { LTRS };
                let c = table[code as usize];
                if c != '\0' {
                    self.text.push(c);
                }
            }
        }
    }
}

#[allow(dead_code)]
impl RttyDecoder {
    /// One bit of samples — the integration length a matched filter for a
    /// constant-envelope tone over one bit has, and the whole point of the
    /// change. Shorter trades the coherent gain straight back: measured at
    /// 0.8, 0.6 and 0.5 of a bit, copy at 10 dB fell 86% → 73% → 59% → 64%.
    fn corr_len(fs: f32, baud: f32) -> usize {
        (fs / baud).round().max(2.0) as usize
    }

    pub fn new(fs: f64) -> Self {
        let fs = fs as f32;
        let baud = 45.45;
        Self {
            fs,
            baud,
            shift: 170.0,
            samples_per_bit: fs / baud,
            tone_rot: Rotator::new(2.0 * std::f32::consts::PI * 170.0 * 0.5 / fs),
            centre_rot: Rotator::new(0.0),
            centre_rate_hz: 0.0,
            mark: ToneCorr::new(Self::corr_len(fs, baud)),
            space: ToneCorr::new(Self::corr_len(fs, baud)),
            mark_acc: Complex32::new(0.0, 0.0),
            space_acc: Complex32::new(0.0, 0.0),
            mark_peak: 1e-6,
            space_peak: 1e-6,
            framers: [Framer::new(false), Framer::new(true)],
            polarity: None,
            auto: true,
            afc_hz: 0.0,
            centre_hz: 0.0,
            acq_buf: vec![Complex32::new(0.0, 0.0); ACQ_N],
            acq_pos: 0,
            acq_fill: 0,
            fft: FftPlanner::new().plan_fft_forward(ACQ_N),
            fft_buf: vec![Complex32::new(0.0, 0.0); ACQ_N],
            band_pwr: 0.0,
            tone_pwr: 0.0,
            scan: CallScanner::new(),
        }
    }

    /// Find the centre of the mark/space pair in the raw passband.
    ///
    /// A one-bit correlator has its first null 45 Hz off tune, so it cannot
    /// find a signal the cursor is not already close to — and the phase-advance
    /// AFC that used to do the finding cannot help, because its discriminant
    /// is only meaningful once the tone is already inside the main lobe. Made
    /// wide and fast enough to escape that, it stopped being a tuning loop and
    /// started chasing whatever was loudest, framing FT8 traffic as Baudot.
    ///
    /// So the coarse step asks the question that actually identifies RTTY:
    /// where are *two* tones a known shift apart? Scoring a candidate centre
    /// by the weaker of its two tones means a single carrier, a CW signal or
    /// an FT8 pile-up scores nothing however loud it is — only a genuine FSK
    /// pair does. One transform per quarter second, and only while the framer
    /// is not already holding.
    fn acquire_centre(&mut self) {
        let n = ACQ_N;
        for k in 0..n {
            self.fft_buf[k] = self.acq_buf[(self.acq_pos + k) % n];
        }
        // Hann, so a strong tone does not smear across the whole search band.
        for (k, v) in self.fft_buf.iter_mut().enumerate() {
            let w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * k as f32 / n as f32).cos();
            *v *= w;
        }
        self.fft.process(&mut self.fft_buf);

        let bin_hz = self.fs / n as f32;
        let at = |hz: f32| -> f32 {
            let b = (hz / bin_hz).round() as isize;
            let idx = b.rem_euclid(n as isize) as usize;
            self.fft_buf[idx].norm_sqr()
        };
        let half = self.shift * 0.5;
        let steps = (ACQ_RANGE_HZ / bin_hz).round() as isize;
        let mut best = (0.0f32, self.centre_hz);
        for step in -steps..=steps {
            let d = step as f32 * bin_hz;
            // The weaker tone is the score: both must be there.
            let score = at(d + half).min(at(d - half));
            if score > best.0 {
                best = (score, d);
            }
        }
        // A pair has to stand out of the passband, or there is nothing to
        // tune to and the last centre is as good a guess as a noise peak.
        let mut floor: Vec<f32> = (0..n).step_by(7).map(|i| self.fft_buf[i].norm_sqr()).collect();
        floor.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = floor[floor.len() / 2].max(1e-20);
        if best.0 > 6.0 * med {
            self.centre_hz = best.1;
        }
    }

    /// Which framer to believe, and whether that is settled.
    fn chosen(&self) -> Option<usize> {
        if let Some(p) = self.polarity {
            return Some(p);
        }
        let (a, b) = (self.framers[0].score(), self.framers[1].score());
        let (lead, other) = if a >= b { (0usize, b) } else { (1usize, a) };
        let fr = &self.framers[lead];
        if fr.score() - other >= POLARITY_MARGIN && fr.framed() >= POLARITY_FRAMED {
            Some(lead)
        } else {
            None
        }
    }

    pub fn set_baud(&mut self, baud: f32) {
        self.baud = baud;
        self.samples_per_bit = self.fs / baud;
    }

    pub fn set_shift(&mut self, shift: f32) {
        self.shift = shift;
        self.tone_rot
            .set_rate(2.0 * std::f32::consts::PI * shift * 0.5 / self.fs);
    }

    /// SNR in a 2500 Hz reference bandwidth, for the spot report.
    ///
    /// The tones hold the signal and the rest of the passband holds the noise,
    /// so `band_pwr - tone_pwr` is the noise in whatever bandwidth reaches the
    /// decoder — the tuning chain's, or the sample rate if that is narrower —
    /// and scaling it to 2500 Hz gives the figure the reporting convention
    /// wants.
    fn spot_snr(&self) -> f32 {
        if self.band_pwr <= 0.0 || self.tone_pwr <= 0.0 {
            // Nothing measured yet. Cannot happen once a call has been
            // decoded, which takes far longer than these take to seed, but
            // guessing the top of the range would be the wrong way to be wrong.
            return 0.0;
        }
        let noise = self.band_pwr - self.tone_pwr;
        if noise <= 0.0 {
            // The tones account for everything in the passband.
            return 20.0;
        }
        let bw = Decoder::bandwidth(self).min(self.fs);
        let gamma = self.tone_pwr / (noise * 2500.0 / bw);
        (10.0 * gamma.log10()).clamp(-24.0, 20.0)
    }
}

impl Decoder for RttyDecoder {
    fn name(&self) -> &'static str {
        "RTTY"
    }

    fn bandwidth(&self) -> f32 {
        self.shift + 2.0 * self.baud + 100.0
    }

    fn process(&mut self, samples: &[Complex32]) -> String {
        for &s in samples {
            self.acq_buf[self.acq_pos] = s;
            self.acq_pos = (self.acq_pos + 1) % ACQ_N;
            self.acq_fill += 1;
            if self.acq_fill >= ACQ_N {
                self.acq_fill = 0;
                let fr = &self.framers[self.chosen().unwrap_or(0)];
                if !(fr.good + fr.err >= 8 && fr.framed() >= 0.6) {
                    self.acquire_centre();
                }
            }
            let want = self.centre_hz + self.afc_hz;
            if (want - self.centre_rate_hz).abs() > 0.05 {
                self.centre_rate_hz = want;
                self.centre_rot
                    .set_rate(-2.0 * std::f32::consts::PI * want / self.fs);
            }
            let s = s * self.centre_rot.next();
            // One phasor serves both tones: the mark sits as far below the
            // centre as the space sits above it.
            let t = self.tone_rot.next();
            let mark_mix = s * t.conj();
            let space_mix = s * t;
            let old_mark = self.mark_acc;
            let old_space = self.space_acc;
            self.mark_acc = self.mark.push(mark_mix);
            self.space_acc = self.space.push(space_mix);
            let em = self.mark_acc.norm_sqr();
            let es = self.space_acc.norm_sqr();
            let decay = 1.0 - 1.0 / (self.fs * 1.5);
            self.mark_peak = (self.mark_peak * decay).max(em);
            self.space_peak = (self.space_peak * decay).max(es);
            let f = em / self.mark_peak.max(1e-9) - es / self.space_peak.max(1e-9);
            // Averaged over a few characters, so the pair describes the
            // transmission rather than the bit going past.
            let pa = 1.0 / (40.0 * self.samples_per_bit);
            for (level, v) in [
                (&mut self.band_pwr, s.norm_sqr()),
                (&mut self.tone_pwr, em + es),
            ] {
                if *level <= 0.0 {
                    *level = v;
                } else {
                    *level += (v - *level) * pa;
                }
            }
            let dominant = if em > es { (self.mark_acc, old_mark) } else { (self.space_acc, old_space) };
            if dominant.0.norm_sqr() > 1e-8 && dominant.1.norm_sqr() > 1e-8 {
                let residual = (dominant.0 * dominant.1.conj()).arg() * self.fs / (2.0 * std::f32::consts::PI);
                self.afc_hz = (self.afc_hz + 0.00005 * residual.clamp(-20.0, 20.0)).clamp(-10.0, 10.0);
            }

            // Both polarities are framed; only one of them is the station.
            let (thr, spb) = (0.0, self.samples_per_bit);
            for fr in self.framers.iter_mut() {
                fr.feed(f, thr, spb);
            }
        }

        // Nothing is released until the polarity is settled — the losing
        // framer's output is the same bits read upside down, so emitting
        // before the vote is in means emitting garbage half the time. Once
        // settled, everything that framer has buffered comes out at once,
        // so the start of the transmission is not lost to the wait.
        let Some(p) = self.chosen() else {
            // Don't let an undecided buffer grow without bound on noise.
            for fr in self.framers.iter_mut() {
                if fr.text.len() > 4096 {
                    fr.text.clear();
                }
            }
            return String::new();
        };
        if self.auto && self.polarity.is_none() {
            self.polarity = Some(p);
        }
        let other = 1 - p;
        self.framers[other].text.clear();

        // Framing is checked continuously, not just to get started. A slot
        // outlives the station it was pointed at, and once the transmission
        // ends the framer is slicing noise — which frames about half the
        // time and spells Baudot as readily as anything else.
        let fr = &self.framers[p];
        if fr.good + fr.err >= 8 && fr.framed() < 0.5 {
            for f in self.framers.iter_mut() {
                f.clear();
            }
            if self.auto {
                self.polarity = None;
            }
            // Whatever announcement was half-heard belonged to a station that
            // is no longer there; joining it to the next one invents a call.
            self.scan.reset();
            return String::new();
        }
        let out = std::mem::take(&mut self.framers[p].text);
        for c in out.chars() {
            self.scan.push(c);
        }
        out
    }

    fn status(&self) -> String {
        let fr = &self.framers[self.chosen().unwrap_or(0)];
        let total = fr.good + fr.err;
        let good = if total > 0 {
            100.0 * fr.good as f32 / total as f32
        } else {
            0.0
        };
        let total = self.centre_hz + self.afc_hz;
        let off = if total.abs() > 5.0 {
            format!(" {total:+.0}Hz")
        } else {
            String::new()
        };
        let pol = match (self.chosen(), self.auto) {
            (None, _) => "?",
            (Some(0), true) => "NOR",
            (Some(_), true) => "REV",
            (Some(0), false) => "NOR*",
            (Some(_), false) => "REV*",
        };
        format!(
            "{:.2} baud/{:.0}Hz {pol}{off} {:.0}% framed",
            self.baud, self.shift, good
        )
    }

    /// Snap to the nearest shift in amateur use rather than to whatever the
    /// histogram measured: the standard shifts are 170, 425 and 850 Hz, and a
    /// demodulator run at the measured 431 Hz would put its matched filters
    /// slightly off both tones for no reason. Anything not near a standard
    /// shift is left alone — better the 170 Hz default than a wrong guess.
    fn set_shift(&mut self, hz: f32) {
        let snapped = [170.0f32, 425.0, 850.0]
            .into_iter()
            .min_by(|a, b| {
                (a - hz)
                    .abs()
                    .partial_cmp(&(b - hz).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(170.0);
        if (snapped - hz).abs() <= 0.15 * snapped {
            RttyDecoder::set_shift(self, snapped);
        }
    }

    /// Framing success, rescaled so noise reads zero.
    ///
    /// A Baudot framer fed noise still finds a start bit and a valid stop bit
    /// about half the time — that is the whole reason `process` throws the
    /// buffer away below 50% rather than below zero. So half is the floor
    /// here too, and only what is framed above chance counts as copy.
    /// Too few frames to judge is reported as no opinion rather than as no
    /// confidence. The first characters of a transmission arrive on the first
    /// two or three frames — calling that zero would hold back exactly the
    /// copy the decoder waited for its polarity vote to release.
    fn confidence(&self) -> Option<f32> {
        let fr = &self.framers[self.chosen()?];
        if fr.good + fr.err < 4 {
            return None;
        }
        Some(((fr.framed() - 0.5) * 2.0).clamp(0.0, 1.0))
    }

    fn speed(&self) -> Option<String> {
        Some(format!("{:.0}bd", self.baud))
    }

    /// Stations that identified themselves since the last call.
    ///
    /// The spot carries the mark tone, which is the frequency a RTTY contact
    /// is logged and reported on. The cursor sits midway between the tones —
    /// `process` mixes mark and space symmetrically about it — so the mark is
    /// half a shift above it.
    ///
    /// The signal report comes from `spot_snr`.
    fn take_messages(&mut self) -> Vec<FtMessage> {
        let calls = self.scan.take_calls();
        if calls.is_empty() {
            return Vec::new();
        }
        let snr = self.spot_snr();
        let stamp = utc_hhmmss();
        let hz = 0.5 * self.shift;
        calls
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

    /// `r` forces the polarity and stops the detector second-guessing it.
    fn toggle(&mut self) {
        let cur = self.chosen().unwrap_or(0);
        let next = 1 - cur;
        self.auto = false;
        self.polarity = Some(next);
        for fr in self.framers.iter_mut() {
            fr.clear();
        }
        self.scan.reset();
    }

    fn reset(&mut self) {
        for fr in self.framers.iter_mut() {
            fr.clear();
        }
        self.scan.reset();
        // A reset means a new signal, so the polarity is an open question
        // again — unless the user answered it, in which case it stays theirs.
        if self.auto {
            self.polarity = None;
        }
        self.tone_rot.reset_phase();
        self.centre_rot.reset_phase();
        self.centre_rate_hz = 0.0;
        self.mark.reset();
        self.space.reset();
        self.centre_hz = 0.0;
        self.acq_buf.iter_mut().for_each(|s| *s = Complex32::new(0.0, 0.0));
        self.acq_pos = 0;
        self.acq_fill = 0;
        self.mark_acc = Complex32::new(0.0, 0.0);
        self.space_acc = Complex32::new(0.0, 0.0);
        self.mark_peak = 1e-6;
        self.space_peak = 1e-6;
        self.band_pwr = 0.0;
        self.tone_pwr = 0.0;
    }
}

// ITA2 tables indexed by the 5-bit code (LSB = first bit on the wire).
#[rustfmt::skip]
const LTRS: [char; 32] = [
    '\0', 'E', '\0', 'A', ' ', 'S', 'I', 'U',
    '\0', 'D', 'R', 'J', 'N', 'F', 'C', 'K',
    'T',  'Z', 'L', 'W', 'H', 'Y', 'P', 'Q',
    'O',  'B', 'G', '\0', 'M', 'X', 'V', '\0',
];

#[rustfmt::skip]
const FIGS: [char; 32] = [
    '\0', '3', '\0', '-', ' ', '\'', '8', '7',
    // Index 3 is BELL in Baudot; it is dropped rather than emitted, since a
    // control character in the transcript corrupts the terminal grid.
    '\0', '$', '4', '\0', ',', '!', ':', '(',
    '5',  '"', ')', '2', '#', '6', '0', '1',
    '9',  '?', '&', '\0', '.', '/', ';', '\0',
];
