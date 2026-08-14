//! BPSK31 decoder.
//!
//! PSK31 is differentially encoded (a phase reversal is a 0 bit, no reversal a
//! 1), so no carrier recovery loop is needed: comparing the phase of adjacent
//! symbols cancels any modest frequency offset. A slow AFC term mops up the
//! residual rotation. Symbol timing comes from the amplitude dip that the
//! raised-cosine pulse shaping puts at every symbol boundary.

use super::Decoder;
use num_complex::Complex32;

const BAUD: f32 = 31.25;

pub struct Psk31Decoder {
    sps: usize,
    idx: usize,
    energy: Vec<f32>,
    boundary: usize,
    since: usize,
    symbol_len: usize,
    acc: Complex32,
    prev: Complex32,
    afc: f32,
    code: String,
    pending_zero: bool,
    text: String,
    symbols: u32,
    quality: f32,
    have_prev: bool,
    #[cfg(test)]
    pub(crate) captured_bits: Vec<bool>,
}

impl Psk31Decoder {
    pub fn new(fs: f64) -> Self {
        let sps = (fs as f32 / BAUD).round().max(4.0) as usize;
        Self {
            sps,
            idx: 0,
            energy: vec![0.0; sps],
            boundary: 0,
            since: 0,
            symbol_len: sps,
            acc: Complex32::new(0.0, 0.0),
            prev: Complex32::new(0.0, 0.0),
            afc: 0.0,
            code: String::new(),
            pending_zero: false,
            text: String::new(),
            symbols: 0,
            quality: 0.0,
            have_prev: false,
            #[cfg(test)]
            captured_bits: Vec::new(),
        }
    }

    fn on_symbol(&mut self, sym: Complex32) {
        self.symbols = self.symbols.wrapping_add(1);
        if !self.have_prev {
            self.prev = sym;
            self.have_prev = true;
            return;
        }
        let d = sym * self.prev.conj();
        self.prev = sym;
        if d.norm() < 1e-12 {
            return;
        }
        // De-rotate by the tracked residual carrier offset.
        let derot = d * Complex32::from_polar(1.0, -self.afc);
        let bit = derot.re >= 0.0;

        // Confidence: how close the symbol sits to the real axis after removing
        // the modulation. 1.0 is a clean lock, 0.0 is noise.
        let q = derot.re.abs() / derot.norm().max(1e-12);
        self.quality = 0.98 * self.quality + 0.02 * q;

        // Strip the BPSK modulation, then nudge the AFC toward the leftover phase.
        let resid = if bit { derot } else { -derot };
        self.afc += 0.02 * resid.arg();
        self.afc = self.afc.clamp(-0.8, 0.8);

        self.push_bit(bit);
    }

    fn push_bit(&mut self, bit: bool) {
        #[cfg(test)]
        self.captured_bits.push(bit);
        if bit {
            if self.pending_zero {
                self.code.push('0');
                self.pending_zero = false;
            }
            self.code.push('1');
            if self.code.len() > 12 {
                self.code.clear(); // never a valid varicode; resync
            }
        } else if self.pending_zero {
            // "00" terminates a character.
            if !self.code.is_empty() {
                if let Some(c) = varicode_lookup(&self.code) {
                    self.text.push(c);
                }
                self.code.clear();
            }
            self.pending_zero = false;
        } else {
            self.pending_zero = true;
        }
    }
}

impl Decoder for Psk31Decoder {
    fn name(&self) -> &'static str {
        "PSK31"
    }

    fn bandwidth(&self) -> f32 {
        200.0
    }

    fn process(&mut self, samples: &[Complex32]) -> String {
        for &s in samples {
            // Free-running phase within the symbol. Each bin is touched once per
            // symbol, so this averages the envelope over roughly 20 symbols.
            self.idx = (self.idx + 1) % self.sps;
            self.energy[self.idx] = 0.95 * self.energy[self.idx] + 0.05 * s.norm();
            self.acc += s;
            self.since += 1;

            // Dump on a sample countdown rather than a phase match: the symbol
            // period stays ~sps no matter how the timing estimate moves, which
            // is what keeps the bit stream from slipping.
            if self.since >= self.symbol_len {
                let sym = self.acc;
                self.acc = Complex32::new(0.0, 0.0);
                self.since = 0;
                self.on_symbol(sym);

                // The envelope minimum marks the true symbol boundary; steer the
                // next symbol's length to walk the dump onto it.
                let mut best = 0usize;
                let mut best_v = f32::MAX;
                for (i, &e) in self.energy.iter().enumerate() {
                    if e < best_v {
                        best_v = e;
                        best = i;
                    }
                }
                self.boundary = best;
                self.symbol_len = if self.symbols < 3 {
                    self.sps
                } else {
                    let n = self.sps as isize;
                    let err = ((best as isize - self.idx as isize + n / 2).rem_euclid(n)) - n / 2;
                    // Pull hard while acquiring, then only trim, so an
                    // established lock stays steady.
                    let limit = if self.symbols < 60 { n / 8 } else { 2 };
                    (n + err.clamp(-limit, limit)).max(4) as usize
                };
            }
        }
        std::mem::take(&mut self.text)
    }

    fn status(&self) -> String {
        let hz = self.afc * BAUD / (2.0 * std::f32::consts::PI);
        format!("q={:.0}% afc={:+.1}Hz", self.quality * 100.0, hz)
    }

    fn reset(&mut self) {
        self.code.clear();
        self.text.clear();
        self.pending_zero = false;
        self.afc = 0.0;
        self.have_prev = false;
        self.acc = Complex32::new(0.0, 0.0);
        self.quality = 0.0;
        self.since = 0;
        self.symbol_len = self.sps;
        self.energy.iter_mut().for_each(|e| *e = 0.0);
    }
}

fn varicode_lookup(code: &str) -> Option<char> {
    VARICODE
        .iter()
        .position(|c| *c == code)
        .map(|i| i as u8 as char)
}

/// Standard PSK31 varicode, indexed by ASCII code point.
#[rustfmt::skip]
pub(crate) const VARICODE: [&str; 128] = [
    "1010101011", "1011011011", "1011101101", "1101110111", "1011101011", "1101011111",
    "1011101111", "1011111101", "1011111111", "11101111",   "11101",      "1101101111",
    "1011011101", "11111",      "1101110101", "1110101011", "1011110111", "1011110101",
    "1110101101", "1110101111", "1101011011", "1101101011", "1101101101", "1101010111",
    "1101111011", "1101111101", "1110110111", "1101010101", "1101011101", "1110111011",
    "1011111011", "1101111111",
    "1",          "111111111",  "101011111",  "111110101",  "111011011",  "1011010101",
    "1010111011", "101111111",  "11111011",   "11110111",   "101101111",  "111011111",
    "1110101",    "110101",     "1010111",    "110101111",
    "10110111",   "10111101",   "11101101",   "11111111",   "101110111",  "101011011",
    "101101011",  "110101101",  "110101011",  "110110111",
    "11110101",   "110111101",  "111101101",  "1010101",    "111010111",  "1010101111",
    "1010111101",
    "1111101",    "11101011",   "10101101",   "10110101",   "1110111",    "11011011",
    "11111101",   "101010101",  "1111111",    "111111101",  "101111101",  "11010111",
    "10111011",   "11011101",   "10101011",   "11010101",   "111011101",  "10101111",
    "1101111",    "1101101",    "101010111",  "110110101",  "101011101",  "101110101",
    "101111011",  "1010101101",
    "111110111",  "111101111",  "111111011",  "1010111111", "101101101",  "1011011111",
    "1011",       "1011111",    "101111",     "101101",     "11",         "111101",
    "1011011",    "101011",     "1101",       "111101011",  "10111111",   "11011",
    "111011",     "1111",       "111",        "111111",     "110111111",  "10101",
    "10111",      "101",        "110111",     "1111011",    "1101011",    "11011111",
    "1011101",    "111010101",
    "1010110111", "110111011",  "1010110101", "1011010111", "1110110101",
];
