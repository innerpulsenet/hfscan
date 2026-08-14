//! Digital mode decoders. Each consumes complex baseband centred on the
//! cursor at `DecodeChain::fs_out()` and emits decoded text incrementally.

pub mod cw;
pub mod ft8;
pub mod psk31;
pub mod rtty;

#[cfg(test)]
mod tests;

use num_complex::Complex32;

/// One decoded FT8/FT4 transmission, kept structured so the UI can build
/// activity maps and station lists rather than just printing lines.
#[derive(Clone, Debug)]
pub struct FtMessage {
    /// UTC hhmmss of the slot the decode came from, matching WSJT-X's log.
    pub stamp: String,
    pub snr_db: f32,
    pub dt_sec: f32,
    /// Audio frequency above the dial, Hz.
    pub freq_hz: f32,
    /// Unpacked message text, e.g. "CQ K1ABC FN42".
    pub text: String,
}

impl FtMessage {
    /// The transcript line format the decode pane has always shown.
    pub fn format(&self) -> String {
        format!(
            "{}  {:>3.0} dB  {:+.1}s  {:>4.0} Hz  {}",
            self.stamp, self.snr_db, self.dt_sec, self.freq_hz, self.text
        )
    }
}

#[allow(dead_code)]
pub trait Decoder: Send {
    fn name(&self) -> &'static str;
    /// Suggested receive bandwidth in Hz for this mode.
    fn bandwidth(&self) -> f32;
    /// Consume samples, returning any newly decoded text.
    fn process(&mut self, samples: &[Complex32]) -> String;
    /// Short human-readable state (speed estimate, lock indicator, ...).
    fn status(&self) -> String;
    fn reset(&mut self);
    /// Mode-specific toggle (currently only RTTY normal/reverse shift).
    fn toggle(&mut self) {}
    /// Offset added to the cursor when centring the tuning chain. FT8/FT4 use
    /// this to place the dial at the bottom edge of an audio passband.
    fn offset_shift(&self) -> f64 {
        0.0
    }
    /// Whether the squelch may gate this decoder. Slot-based modes must keep
    /// receiving regardless of what is under the cursor.
    fn squelched(&self) -> bool {
        true
    }
    /// Drain structured messages decoded since the last call. Only FT8/FT4
    /// produce these; other modes return nothing.
    fn take_messages(&mut self) -> Vec<FtMessage> {
        Vec::new()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Off,
    Cw,
    Rtty,
    Psk31,
    Ft8,
    Ft4,
}

impl Mode {
    pub const ALL: [Mode; 6] = [
        Mode::Off,
        Mode::Cw,
        Mode::Rtty,
        Mode::Psk31,
        Mode::Ft8,
        Mode::Ft4,
    ];

    pub fn next(self) -> Mode {
        let i = Mode::ALL.iter().position(|m| *m == self).unwrap_or(0);
        Mode::ALL[(i + 1) % Mode::ALL.len()]
    }

    pub fn label(self) -> &'static str {
        match self {
            Mode::Off => "OFF",
            Mode::Cw => "CW",
            Mode::Rtty => "RTTY",
            Mode::Psk31 => "PSK31",
            Mode::Ft8 => "FT8",
            Mode::Ft4 => "FT4",
        }
    }

    /// Rate the tuning chain should deliver for this mode. FT8/FT4 require
    /// exactly 12 kHz because the decoder's timing is derived from it.
    pub fn audio_rate(self) -> f64 {
        match self {
            Mode::Ft8 | Mode::Ft4 => ft8::AUDIO_RATE,
            _ => 8000.0,
        }
    }

    pub fn make(self, fs: f64) -> Option<Box<dyn Decoder>> {
        match self {
            Mode::Off => None,
            Mode::Cw => Some(Box::new(cw::CwDecoder::new(fs))),
            Mode::Rtty => Some(Box::new(rtty::RttyDecoder::new(fs))),
            Mode::Psk31 => Some(Box::new(psk31::Psk31Decoder::new(fs))),
            Mode::Ft8 => Some(Box::new(ft8::FtDecoder::new(fs, false))),
            Mode::Ft4 => Some(Box::new(ft8::FtDecoder::new(fs, true))),
        }
    }
}
