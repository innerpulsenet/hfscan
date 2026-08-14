//! Baudot RTTY decoder. Default 45.45 baud / 170 Hz shift, which covers almost
//! all amateur HF RTTY. Uses an FM discriminator plus start-bit clock recovery.
//!
//! The slicer threshold is not fixed at 0 Hz: envelope trackers follow the
//! mark and space tone extremes and slice midway between them, so the cursor
//! only has to be near the pair, not exactly centred. Bits are decided from
//! the discriminator averaged over the middle of each bit rather than one
//! instantaneous sample. If the text comes out as garbage try the
//! Shift polarity is detected, not assumed. Both slicings of every bit are
//! framed in parallel and the one that actually frames — start bit a space,
//! stop bit a mark — wins; `r` still forces it either way.

use super::Decoder;
use crate::dsp::OnePole;
use num_complex::Complex32;

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

pub struct RttyDecoder {
    fs: f32,
    baud: f32,
    shift: f32,
    prev: Complex32,
    disc: OnePole,
    samples_per_bit: f32,
    /// Normal and reversed slicings of the same discriminator output. Which
    /// one is right is decided by which one frames, not assumed.
    framers: [Framer; 2],
    /// The winning polarity once known, or forced by the user.
    polarity: Option<usize>,
    /// Whether the polarity is still being worked out, or the user has said.
    auto: bool,
    /// Discriminator extremes (fast attack, slow decay); the slicer sits
    /// midway, which is what makes an off-centre tuning still decode.
    fmax: f32,
    fmin: f32,
    thr: f32,
    track_fast: f32,
    track_slow: f32,
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
    /// Integrate-and-dump over the middle of the current bit.
    bit_acc: f32,
    bit_n: u32,
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
            bit_acc: 0.0,
            bit_n: 0,
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
                        self.bit_acc = 0.0;
                        self.bit_n = 0;
                    }
                } else {
                    self.counter = 0.0;
                }
            }
            State::Data => {
                self.counter += 1.0;
                // Average the discriminator across the middle of the bit
                // (15%..50% of the way in, given the mid-bit dump below):
                // one noisy sample can no longer flip the decision.
                if self.counter >= samples_per_bit * 0.65 {
                    self.bit_acc += f;
                    self.bit_n += 1;
                }
                if self.counter >= samples_per_bit {
                    self.counter -= samples_per_bit;
                    let avg = if self.bit_n > 0 {
                        self.bit_acc / self.bit_n as f32
                    } else {
                        f
                    };
                    self.bit_acc = 0.0;
                    self.bit_n = 0;
                    let m = if self.invert { avg < thr } else { avg > thr };
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
            0x04 => self.text.push(' '),
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
    pub fn new(fs: f64) -> Self {
        let fs = fs as f32;
        let baud = 45.45;
        Self {
            fs,
            baud,
            shift: 170.0,
            prev: Complex32::new(0.0, 0.0),
            // smooth the discriminator over about a third of a bit
            disc: OnePole::new(fs / baud / 3.0),
            samples_per_bit: fs / baud,
            framers: [Framer::new(false), Framer::new(true)],
            polarity: None,
            auto: true,
            fmax: 0.0,
            fmin: 0.0,
            thr: 0.0,
            track_fast: 1.0 - (-1.0f32 / (0.004 * fs)).exp(),
            track_slow: 1.0 - (-1.0f32 / (1.5 * fs)).exp(),
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
        self.disc = OnePole::new(self.samples_per_bit / 3.0);
    }

    pub fn set_shift(&mut self, shift: f32) {
        self.shift = shift;
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
            // Instantaneous frequency via the phase difference between samples.
            let d = s * self.prev.conj();
            self.prev = s;
            let f = if d.norm() > 1e-12 {
                d.arg() * self.fs / (2.0 * std::f32::consts::PI)
            } else {
                0.0
            };
            let f = self.disc.process(f);

            // Follow the tone extremes (fast out, slow back) and slice midway
            // between them. Only trust the midpoint once the spread looks
            // like a real FSK pair, so noise alone does not drag it around.
            if f > self.fmax {
                self.fmax += (f - self.fmax) * self.track_fast;
            } else {
                self.fmax += (f - self.fmax) * self.track_slow;
            }
            if f < self.fmin {
                self.fmin += (f - self.fmin) * self.track_fast;
            } else {
                self.fmin += (f - self.fmin) * self.track_slow;
            }
            if self.fmax - self.fmin > 0.35 * self.shift {
                self.thr = 0.5 * (self.fmax + self.fmin);
            }

            // Both polarities are framed; only one of them is the station.
            let (thr, spb) = (self.thr, self.samples_per_bit);
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
            return String::new();
        }
        std::mem::take(&mut self.framers[p].text)
    }

    fn status(&self) -> String {
        let fr = &self.framers[self.chosen().unwrap_or(0)];
        let total = fr.good + fr.err;
        let good = if total > 0 {
            100.0 * fr.good as f32 / total as f32
        } else {
            0.0
        };
        let off = if self.thr.abs() > 5.0 {
            format!(" {:+.0}Hz", self.thr)
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

    /// `r` forces the polarity and stops the detector second-guessing it.
    fn toggle(&mut self) {
        let cur = self.chosen().unwrap_or(0);
        let next = 1 - cur;
        self.auto = false;
        self.polarity = Some(next);
        for fr in self.framers.iter_mut() {
            fr.clear();
        }
    }

    fn reset(&mut self) {
        for fr in self.framers.iter_mut() {
            fr.clear();
        }
        // A reset means a new signal, so the polarity is an open question
        // again — unless the user answered it, in which case it stays theirs.
        if self.auto {
            self.polarity = None;
        }
        self.fmax = 0.0;
        self.fmin = 0.0;
        self.thr = 0.0;
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
