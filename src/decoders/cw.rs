//! Morse (CW) decoder: envelope detection with an adaptive threshold and a
//! self-tuning dit length, so it tracks operator speed without configuration.

use super::Decoder;
use crate::dsp::OnePole;
use num_complex::Complex32;

const ENV_DECIM: usize = 8; // audio rate -> ~1 kHz envelope rate

pub struct CwDecoder {
    env_rate: f32,
    smooth: OnePole,
    peak: f32,
    floor: f32,
    peak_decay: f32,
    floor_attack: f32,
    floor_decay: f32,
    decim_ctr: usize,
    key_down: bool,
    run: f32, // samples in the current mark/space run
    dit: f32, // adaptive dit length, in envelope samples
    symbol: String,
    text: String,
    idle: f32,
    started: bool,
}

impl CwDecoder {
    pub fn new(fs: f64) -> Self {
        let env_rate = fs as f32 / ENV_DECIM as f32;
        Self {
            env_rate,
            // ~4 ms envelope smoothing still passes 40 WPM dits (30 ms)
            smooth: OnePole::new(0.004 * fs as f32),
            peak: 0.0,
            floor: 0.0,
            // peak holds briefly then decays; the floor rises slowly and falls fast
            peak_decay: 1.0 - (-1.0f32 / (0.6 * env_rate)).exp(),
            floor_attack: 1.0 - (-1.0f32 / (2.0 * env_rate)).exp(),
            floor_decay: 1.0 - (-1.0f32 / (0.05 * env_rate)).exp(),
            decim_ctr: 0,
            key_down: false,
            run: 0.0,
            dit: 0.06 * env_rate, // start at 20 WPM
            symbol: String::new(),
            text: String::new(),
            idle: 0.0,
            started: false,
        }
    }

    pub fn wpm(&self) -> f32 {
        let dit_ms = self.dit / self.env_rate * 1000.0;
        if dit_ms > 1.0 {
            1200.0 / dit_ms
        } else {
            0.0
        }
    }

    fn push_symbol(&mut self) {
        if self.symbol.is_empty() {
            return;
        }
        let c = morse_lookup(&self.symbol).unwrap_or('*');
        self.text.push(c);
        self.symbol.clear();
    }

    fn on_mark_end(&mut self, len: f32) {
        // Adapt the dit estimate from whichever element this looks like.
        if len < 2.0 * self.dit {
            self.dit = 0.85 * self.dit + 0.15 * len;
            self.symbol.push('.');
        } else {
            self.dit = 0.85 * self.dit + 0.15 * (len / 3.0);
            self.symbol.push('-');
        }
        self.dit = self.dit.clamp(0.015 * self.env_rate, 0.25 * self.env_rate);
    }

    fn on_space_end(&mut self, len: f32) {
        if len >= 2.0 * self.dit {
            self.push_symbol();
            if len >= 5.0 * self.dit && !self.text.ends_with(' ') && !self.text.is_empty() {
                self.text.push(' ');
            }
        }
    }
}

impl Decoder for CwDecoder {
    fn name(&self) -> &'static str {
        "CW"
    }

    fn bandwidth(&self) -> f32 {
        400.0
    }

    fn process(&mut self, samples: &[Complex32]) -> String {
        for s in samples {
            let env = self.smooth.process(s.norm());
            self.decim_ctr += 1;
            if self.decim_ctr < ENV_DECIM {
                continue;
            }
            self.decim_ctr = 0;

            // Peak follower: instant attack, slow decay.
            if env > self.peak {
                self.peak = env;
            } else {
                self.peak += (env - self.peak) * self.peak_decay;
            }
            // Noise floor: rises slowly, drops quickly toward quiet samples.
            if env < self.floor {
                self.floor += (env - self.floor) * self.floor_decay;
            } else {
                self.floor += (env - self.floor) * self.floor_attack;
            }

            let span = (self.peak - self.floor).max(1e-9);
            let on_thr = self.floor + 0.55 * span;
            let off_thr = self.floor + 0.35 * span;
            // Require the tone to stand clearly above the floor before keying.
            let snr_ok = self.peak > 2.5 * self.floor.max(1e-9);

            let next = if self.key_down {
                env > off_thr
            } else {
                env > on_thr && snr_ok
            };

            if next == self.key_down {
                self.run += 1.0;
                if !self.key_down {
                    self.idle += 1.0;
                    // Long silence: flush any pending character.
                    if self.idle > 8.0 * self.dit && !self.symbol.is_empty() {
                        self.push_symbol();
                    }
                }
                continue;
            }

            let len = self.run;
            self.run = 1.0;
            if self.key_down {
                // A mark just ended; ignore implausibly short blips.
                if len > 0.012 * self.env_rate {
                    self.on_mark_end(len);
                }
                self.idle = 0.0;
            } else {
                // A space just ended.
                if self.started {
                    self.on_space_end(len);
                }
                self.started = true;
                self.idle = 0.0;
            }
            self.key_down = next;
        }
        std::mem::take(&mut self.text)
    }

    fn status(&self) -> String {
        format!("{:.0} WPM", self.wpm())
    }

    fn reset(&mut self) {
        self.symbol.clear();
        self.text.clear();
        self.key_down = false;
        self.run = 0.0;
        self.started = false;
        self.peak = 0.0;
        self.floor = 0.0;
    }
}

fn morse_lookup(sym: &str) -> Option<char> {
    const TABLE: &[(&str, char)] = &[
        (".-", 'A'), ("-...", 'B'), ("-.-.", 'C'), ("-..", 'D'), (".", 'E'),
        ("..-.", 'F'), ("--.", 'G'), ("....", 'H'), ("..", 'I'), (".---", 'J'),
        ("-.-", 'K'), (".-..", 'L'), ("--", 'M'), ("-.", 'N'), ("---", 'O'),
        (".--.", 'P'), ("--.-", 'Q'), (".-.", 'R'), ("...", 'S'), ("-", 'T'),
        ("..-", 'U'), ("...-", 'V'), (".--", 'W'), ("-..-", 'X'), ("-.--", 'Y'),
        ("--..", 'Z'),
        ("-----", '0'), (".----", '1'), ("..---", '2'), ("...--", '3'),
        ("....-", '4'), (".....", '5'), ("-....", '6'), ("--...", '7'),
        ("---..", '8'), ("----.", '9'),
        (".-.-.-", '.'), ("--..--", ','), ("..--..", '?'), (".----.", '\''),
        ("-.-.--", '!'), ("-..-.", '/'), ("-.--.", '('), ("-.--.-", ')'),
        (".-...", '&'), ("---...", ':'), ("-.-.-.", ';'), ("-...-", '='),
        (".-.-.", '+'), ("-....-", '-'), ("..--.-", '_'), (".-..-.", '"'),
        ("...-..-", '$'), (".--.-.", '@'),
    ];
    TABLE.iter().find(|(s, _)| *s == sym).map(|(_, c)| *c)
}
