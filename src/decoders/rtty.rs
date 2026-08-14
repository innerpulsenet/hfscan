//! Baudot RTTY decoder. Default 45.45 baud / 170 Hz shift, which covers almost
//! all amateur HF RTTY. Uses an FM discriminator plus start-bit clock recovery.
//!
//! Tune so the cursor sits midway between the mark and space tones; if the text
//! comes out as garbage try the reverse-shift toggle (`r`).

use super::Decoder;
use crate::dsp::OnePole;
use num_complex::Complex32;

pub struct RttyDecoder {
    fs: f32,
    baud: f32,
    shift: f32,
    reverse: bool,
    prev: Complex32,
    disc: OnePole,
    samples_per_bit: f32,
    state: State,
    counter: f32,
    bits: u8,
    nbits: u32,
    figs: bool,
    text: String,
    locked_bits: u32,
    err_bits: u32,
}

#[derive(PartialEq)]
enum State {
    Idle,
    Data,
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
            reverse: false,
            prev: Complex32::new(0.0, 0.0),
            // smooth the discriminator over about a third of a bit
            disc: OnePole::new(fs / baud / 3.0),
            samples_per_bit: fs / baud,
            state: State::Idle,
            counter: 0.0,
            bits: 0,
            nbits: 0,
            figs: false,
            text: String::new(),
            locked_bits: 0,
            err_bits: 0,
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

    fn emit(&mut self, code: u8) {
        match code {
            0x1F => self.figs = false, // LTRS
            0x1B => self.figs = true,  // FIGS
            0x04 => self.text.push(' '),
            0x08 => self.text.push('\n'), // CR
            0x02 => {}                    // LF (CR already breaks the line)
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
            let mut f = if d.norm() > 1e-12 {
                d.arg() * self.fs / (2.0 * std::f32::consts::PI)
            } else {
                0.0
            };
            if self.reverse {
                f = -f;
            }
            let f = self.disc.process(f);
            // Mark is the high tone, space the low one.
            let mark = f > 0.0;

            match self.state {
                State::Idle => {
                    // A start bit is a space; sample it half a bit in to confirm.
                    if !mark {
                        self.counter += 1.0;
                        if self.counter >= self.samples_per_bit * 0.5 {
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
                    if self.counter >= self.samples_per_bit {
                        self.counter -= self.samples_per_bit;
                        if self.nbits < 5 {
                            // Baudot sends the least significant bit first.
                            if mark {
                                self.bits |= 1 << self.nbits;
                            }
                            self.nbits += 1;
                        } else {
                            // Stop bit must be mark; otherwise we lost framing.
                            if mark {
                                let code = self.bits;
                                self.emit(code);
                                self.locked_bits = self.locked_bits.saturating_add(1);
                            } else {
                                self.err_bits = self.err_bits.saturating_add(1);
                            }
                            self.state = State::Idle;
                            self.counter = 0.0;
                        }
                    }
                }
            }
        }
        std::mem::take(&mut self.text)
    }

    fn status(&self) -> String {
        let total = self.locked_bits + self.err_bits;
        let good = if total > 0 {
            100.0 * self.locked_bits as f32 / total as f32
        } else {
            0.0
        };
        format!(
            "{:.2} baud/{:.0}Hz {} {:.0}% framed",
            self.baud,
            self.shift,
            if self.reverse { "REV" } else { "NOR" },
            good
        )
    }

    fn toggle(&mut self) {
        self.reverse = !self.reverse;
    }

    fn reset(&mut self) {
        self.state = State::Idle;
        self.counter = 0.0;
        self.bits = 0;
        self.nbits = 0;
        self.figs = false;
        self.text.clear();
        self.locked_bits = 0;
        self.err_bits = 0;
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
    '\0', '$', '4', '\u{7}', ',', '!', ':', '(',
    '5',  '"', ')', '2', '#', '6', '0', '1',
    '9',  '?', '&', '\0', '.', '/', ';', '\0',
];
