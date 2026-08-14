//! Digital mode decoders. Each consumes complex baseband centred on the
//! cursor at `DecodeChain::fs_out()` and emits decoded text incrementally.

pub mod cw;
pub mod ft8;
pub mod psk31;
pub mod rtty;

#[cfg(test)]
pub(crate) mod tests;

use num_complex::Complex32;

/// Snapshot of the CW decoder for the dedicated scope pane.
#[derive(Clone, Debug)]
pub struct CwView {
    /// Normalised envelope 0..1, oldest first.
    pub env: Vec<f32>,
    /// Key-down flag for each envelope sample.
    pub keyed: Vec<bool>,
    /// Slice thresholds as 0..1 of the current peak–floor span.
    pub on_thr: f32,
    pub off_thr: f32,
    /// Lock offset from the cursor, Hz.
    pub lock_hz: f32,
    /// Residual tone after the lock mix, Hz. Zero means centred.
    pub tune_err_hz: f32,
    pub wpm: f32,
    pub quality: f32,
    pub key_down: bool,
    /// Morse being assembled (`.`, `-`).
    pub symbol: String,
    pub dit_ms: f32,
    pub locked: bool,
    pub hits: Vec<cw::CwHit>,
}

/// Snapshot of the PSK31 decoder for the scope / tuner pane.
#[derive(Clone, Debug)]
pub struct PskView {
    /// Recent normalised symbols (I/Q on the unit circle), oldest first.
    pub symbols: Vec<Complex32>,
    /// |baseband| 0..1 after the lock mix, oldest first.
    pub env: Vec<f32>,
    pub lock_hz: f32,
    /// Residual AFC after the lock mix, Hz.
    pub tune_err_hz: f32,
    pub quality: f32,
    pub reversals: f32,
    pub locked: bool,
    pub hits: Vec<psk31::PskHit>,
}

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
    /// Whether the soft AGC may ride this decoder's audio. Modes that
    /// normalise a whole capture themselves must say no: a running gain
    /// tracker moves the noise floor underneath their own soft metrics.
    fn wants_agc(&self) -> bool {
        true
    }
    /// Drain structured messages decoded since the last call. FT8/FT4 produce
    /// one per decode; PSK31 produces one when a CQ/DE callsign is recognised.
    fn take_messages(&mut self) -> Vec<FtMessage> {
        Vec::new()
    }
    /// Frequency the decoder has locked to, in Hz relative to the tuning-chain
    /// centre (the cursor). Zero when the mode does not track a carrier.
    fn lock_hz(&self) -> f32 {
        0.0
    }
    /// Whether the decoder has a confident lock on a signal.
    fn locked(&self) -> bool {
        false
    }
    /// Receiver identity, used for FT8 a-priori decoding and hash lookup.
    fn set_station(&mut self, _call: &str, _grid: &str) {}
    /// Drop the current lock (the cursor moved; baseband is a new place).
    fn hop(&mut self) {}
    /// Jump to the next / previous identified signal *inside* the current
    /// passband. Returns the new lock offset in Hz, or `None`.
    fn next_lock(&mut self, _forward: bool) -> Option<f32> {
        None
    }
    /// Offsets (Hz, relative to the cursor) of signals the decoder has
    /// identified in the current passband.
    fn candidate_hz(&self) -> Vec<f32> {
        Vec::new()
    }
    /// Live CW scope (envelope, lock, tune error). Other modes return none.
    fn cw_view(&self) -> Option<CwView> {
        None
    }
    fn psk_view(&self) -> Option<PskView> {
        None
    }
    /// Nudge the in-passband lock, Hz. Returns the new lock offset.
    fn nudge_lock(&mut self, _delta_hz: f32) -> Option<f32> {
        None
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
    /// Decode every digital signal in the span at once, each on its own
    /// tuning chain. Not a decoder itself — see `AutoSlot` in main.
    Auto,
}

impl Mode {
    pub const ALL: [Mode; 7] = [
        Mode::Off,
        Mode::Cw,
        Mode::Rtty,
        Mode::Psk31,
        Mode::Ft8,
        Mode::Ft4,
        Mode::Auto,
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
            Mode::Auto => "AUTO",
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
            // Auto builds its own decoders, one per signal found.
            Mode::Off | Mode::Auto => None,
            Mode::Cw => Some(Box::new(cw::CwDecoder::new(fs))),
            Mode::Rtty => Some(Box::new(rtty::RttyDecoder::new(fs))),
            Mode::Psk31 => Some(Box::new(psk31::Psk31Decoder::new(fs))),
            Mode::Ft8 => Some(Box::new(ft8::FtDecoder::new(fs, false))),
            Mode::Ft4 => Some(Box::new(ft8::FtDecoder::new(fs, true))),
        }
    }
}
