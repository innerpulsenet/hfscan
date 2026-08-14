//! FT8 and FT4 decoding.
//!
//! Unlike the other modes these are slot-based: transmissions occupy fixed
//! 15 s (FT8) or 7.5 s (FT4) windows aligned to UTC, and a whole slot must be
//! captured before anything can be decoded. So this decoder buffers audio,
//! and at each slot boundary hands the completed slot to a worker thread —
//! decoding takes long enough that doing it inline would stall the UI.
//!
//! The heavy lifting (Costas sync, LDPC, message unpacking) is `mfsk-core`.
//!
//! Audio convention follows WSJT-X: the cursor is the *dial* frequency and
//! signals sit at 200-2900 Hz above it. The tuning chain is centred half a
//! passband up (see `Decoder::offset_shift`), so an NCO here shifts that back
//! to put the dial at 0 Hz before taking the real part.

use super::{Decoder, FtMessage};
use crate::dsp::Nco;
use mfsk_core::msg::decode_request::DecodeRequest;
use mfsk_core::msg::wsjt77::unpack77;
use mfsk_core::{Ft4, Ft8};
use num_complex::Complex32;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::{SystemTime, UNIX_EPOCH};

pub const AUDIO_RATE: f64 = 12_000.0;
/// Audio passband centre; signals live at 200-2900 Hz above the dial.
pub const AUDIO_CENTRE: f64 = 1500.0;
pub const FREQ_MIN: f32 = 200.0;
pub const FREQ_MAX: f32 = 2900.0;
const MAX_CAND: usize = 60;

pub struct FtDecoder {
    ft4: bool,
    slot_secs: f64,
    nmax: usize,
    nco: Nco,
    buf: Vec<i16>,
    slot: u64,
    jobs: Sender<Vec<i16>>,
    results: Receiver<Vec<FtMessage>>,
    msgs: Vec<FtMessage>,
    decodes: u32,
    pending: bool,
    started: bool,
    level: f32,
    last_fill: f32,
}

impl FtDecoder {
    pub fn new(fs: f64, ft4: bool) -> Self {
        let slot_secs = if ft4 { 7.5 } else { 15.0 };
        let nmax = (slot_secs * AUDIO_RATE) as usize;
        let (jobs, job_rx) = channel::<Vec<i16>>();
        let (res_tx, results) = channel::<Vec<FtMessage>>();

        std::thread::spawn(move || {
            while let Ok(audio) = job_rx.recv() {
                // One batch per slot (possibly empty), so the UI can tell
                // "slot done, nothing heard".
                if res_tx.send(decode_slot(&audio, ft4)).is_err() {
                    return;
                }
            }
        });

        let mut nco = Nco::new();
        // Shift the passband back down so the dial sits at 0 Hz.
        nco.set_freq(-AUDIO_CENTRE, fs);

        Self {
            ft4,
            slot_secs,
            nmax,
            nco,
            buf: Vec::with_capacity(nmax + 4096),
            slot: current_slot(slot_secs),
            jobs,
            results,
            msgs: Vec::new(),
            decodes: 0,
            pending: false,
            started: false,
            level: 0.0,
            last_fill: 0.0,
        }
    }

    /// Decode one prepared slot directly, returning transcript lines. Used by
    /// the tests, which cannot wait on the wall clock.
    #[cfg(test)]
    pub fn decode_audio(audio: &[i16], ft4: bool) -> Vec<String> {
        decode_slot(audio, ft4).iter().map(|m| m.format()).collect()
    }

    #[cfg(test)]
    pub fn append_audio_for_test(&mut self, samples: &[Complex32]) {
        self.append_audio(samples);
    }

    #[cfg(test)]
    pub fn audio_buffer(&self) -> &[i16] {
        &self.buf
    }

    /// Convert complex baseband from the tuning chain into the i16 USB audio
    /// the FT decoder expects, appending to the slot buffer.
    ///
    /// The level is normalised rather than fixed-scaled: after narrowing a
    /// 192 kHz span down to 3 kHz the residual amplitude is tiny, and a fixed
    /// gain would quantise weak signals into nothing.
    fn append_audio(&mut self, samples: &[Complex32]) {
        let mut shifted = Vec::with_capacity(samples.len());
        self.nco.mix(samples, &mut shifted);

        let sum_sq: f32 = shifted.iter().map(|c| c.re * c.re).sum();
        if sum_sq > 0.0 {
            let rms = (sum_sq / shifted.len().max(1) as f32).sqrt();
            // Track the level slowly so the scale is stable across a slot.
            self.level = if self.level <= 0.0 {
                rms
            } else {
                0.9 * self.level + 0.1 * rms
            };
        }
        // Aim for a comfortable fraction of full scale.
        let gain = if self.level > 1e-12 {
            (4000.0 / self.level).min(1.0e7)
        } else {
            1.0
        };

        self.buf.extend(
            shifted
                .iter()
                .map(|c| (c.re * gain).clamp(-32_767.0, 32_767.0) as i16),
        );
    }

    fn flush_slot(&mut self) {
        // Only bother if we captured most of the slot; a partial buffer at
        // startup would just waste a decode.
        self.last_fill = self.buf.len() as f32 / self.nmax as f32 * 100.0;
        if self.buf.len() >= self.nmax * 3 / 4 {
            let mut audio = std::mem::take(&mut self.buf);
            audio.resize(self.nmax, 0);
            self.pending = true;
            let _ = self.jobs.send(audio);
        }
        self.buf.clear();
        self.buf.reserve(self.nmax + 4096);
    }
}

fn current_slot(slot_secs: f64) -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    (now / slot_secs) as u64
}

fn decode_slot(audio: &[i16], ft4: bool) -> Vec<FtMessage> {
    let stamp = slot_stamp();
    let mut out = Vec::new();
    if ft4 {
        let res = DecodeRequest::<Ft4>::new(audio, FREQ_MIN, FREQ_MAX, 1.0, MAX_CAND)
            .decode()
            .results;
        for r in &res {
            if let Some(text) = unpack77(r.message77()) {
                out.push(FtMessage {
                    stamp: stamp.clone(),
                    snr_db: r.snr_db,
                    dt_sec: r.dt_sec,
                    freq_hz: r.freq_hz,
                    text,
                });
            }
        }
    } else {
        let res = DecodeRequest::<Ft8>::new(audio, FREQ_MIN, FREQ_MAX, 1.0, MAX_CAND)
            .decode()
            .results;
        for r in &res {
            if let Some(text) = unpack77(r.message77()) {
                out.push(FtMessage {
                    stamp: stamp.clone(),
                    snr_db: r.snr_db,
                    dt_sec: r.dt_sec,
                    freq_hz: r.freq_hz,
                    text,
                });
            }
        }
    }
    out
}

/// UTC hhmmss of the slot that just ended, matching WSJT-X's log style.
fn slot_stamp() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let s = secs % 86400;
    format!("{:02}{:02}{:02}", s / 3600, (s % 3600) / 60, s % 60)
}

impl Decoder for FtDecoder {
    fn name(&self) -> &'static str {
        if self.ft4 {
            "FT4"
        } else {
            "FT8"
        }
    }

    fn bandwidth(&self) -> f32 {
        3000.0
    }

    fn offset_shift(&self) -> f64 {
        AUDIO_CENTRE
    }

    fn squelched(&self) -> bool {
        // Slot capture must run continuously; an SNR gate on the cursor is
        // meaningless when the passband holds many independent signals.
        false
    }

    fn process(&mut self, samples: &[Complex32]) -> String {
        let slot = current_slot(self.slot_secs);
        if slot != self.slot {
            if self.started {
                self.flush_slot();
            }
            self.slot = slot;
            self.started = true;
        }

        // Shift the dial back to 0 Hz and take the real part: this is exactly
        // the USB audio a conventional receiver would feed to WSJT-X.
        if self.buf.len() < self.nmax + 4096 {
            self.append_audio(samples);
        }

        let mut text = String::new();
        while let Ok(batch) = self.results.try_recv() {
            self.pending = false;
            for m in batch {
                self.decodes += 1;
                text.push_str(&m.format());
                text.push('\n');
                self.msgs.push(m);
            }
        }
        text
    }

    fn take_messages(&mut self) -> Vec<FtMessage> {
        std::mem::take(&mut self.msgs)
    }

    fn status(&self) -> String {
        let filled = (self.buf.len() as f32 / self.nmax as f32 * 100.0).min(100.0);
        format!(
            "slot {:>3.0}% (last {:.0}%){}  {} decoded",
            filled,
            self.last_fill,
            if self.pending { " decoding" } else { "" },
            self.decodes
        )
    }

    fn reset(&mut self) {
        self.buf.clear();
        self.msgs.clear();
        self.decodes = 0;
        self.started = false;
        self.level = 0.0;
        self.slot = current_slot(self.slot_secs);
    }
}
