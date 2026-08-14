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
use crate::report::is_callsign;
use mfsk_core::engine::equalize::EqMode;
use mfsk_core::msg::ap::ApHint;
use mfsk_core::msg::decode_request::DecodeRequest;
use mfsk_core::msg::hash_table::CallsignHashTable;
use mfsk_core::msg::wsjt77::{is_plausible_message, unpack77_with_hash};
use mfsk_core::{Ft4, Ft8};
use num_complex::Complex32;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::{SystemTime, UNIX_EPOCH};

pub const AUDIO_RATE: f64 = 12_000.0;
/// Audio passband centre; signals live at 200-3000 Hz above the dial.
pub const AUDIO_CENTRE: f64 = 1600.0;
pub const FREQ_MIN: f32 = 200.0;
/// Matches WSJT-X's default upper edge so stations at the top of the
/// waterfall are not silently dropped.
pub const FREQ_MAX: f32 = 3000.0;
const MAX_CAND: usize = 200;

pub struct FtDecoder {
    ft4: bool,
    slot_secs: f64,
    nmax: usize,
    nco: Nco,
    /// Raw real audio for the slot, quantised to i16 only at slot end so a
    /// single slot-wide gain can be used (see `append_audio`).
    buf: Vec<f32>,
    slot: u64,
    jobs: Sender<SlotJob>,
    results: Receiver<Vec<FtMessage>>,
    msgs: Vec<FtMessage>,
    decodes: u32,
    pending: bool,
    started: bool,
    last_fill: f32,
    my_call: String,
}

struct SlotJob {
    audio: Vec<i16>,
    my_call: String,
}

impl FtDecoder {
    pub fn new(fs: f64, ft4: bool) -> Self {
        let slot_secs = if ft4 { 7.5 } else { 15.0 };
        let nmax = (slot_secs * AUDIO_RATE) as usize;
        let (jobs, job_rx) = channel::<SlotJob>();
        let (res_tx, results) = channel::<Vec<FtMessage>>();

        std::thread::spawn(move || {
            let mut hash = CallsignHashTable::new();
            while let Ok(job) = job_rx.recv() {
                // If slots queued up faster than we can decode them (slow
                // CPU, busy band), keep only the freshest: stale decodes are
                // worth less than current ones, and the queue must stay
                // bounded.
                let mut latest = job;
                while let Ok(newer) = job_rx.try_recv() {
                    latest = newer;
                }
                if !latest.my_call.is_empty() {
                    hash.insert(&latest.my_call);
                }
                // One batch per slot (possibly empty), so the UI can tell
                // "slot done, nothing heard".
                if res_tx
                    .send(decode_slot(&latest.audio, ft4, &latest.my_call, &mut hash))
                    .is_err()
                {
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
            last_fill: 0.0,
            my_call: String::new(),
        }
    }

    /// Decode one prepared slot directly, returning transcript lines. Used by
    /// the tests, which cannot wait on the wall clock.
    #[cfg(test)]
    pub fn decode_audio(audio: &[i16], ft4: bool) -> Vec<String> {
        let mut hash = CallsignHashTable::new();
        decode_slot(audio, ft4, "", &mut hash)
            .iter()
            .map(|m| m.format())
            .collect()
    }

    #[cfg(test)]
    pub fn append_audio_for_test(&mut self, samples: &[Complex32]) {
        self.append_audio(samples);
    }

    /// The buffered slot, quantised the same way `flush_slot` does it.
    #[cfg(test)]
    pub fn audio_buffer(&self) -> Vec<i16> {
        quantize(&self.buf, self.nmax)
    }

    /// Convert complex baseband from the tuning chain into USB audio (real
    /// part, dial at 0 Hz) and append it to the slot buffer.
    ///
    /// Samples are kept as raw f32 until slot end: quantising there with one
    /// slot-wide gain beats tracking a running level, because a strong
    /// station appearing mid-slot would otherwise clip against the i16 rails
    /// until the tracker caught up, and the gain step it then applied would
    /// shift the noise floor under the LDPC soft metrics mid-message.
    fn append_audio(&mut self, samples: &[Complex32]) {
        let mut shifted = Vec::with_capacity(samples.len());
        self.nco.mix(samples, &mut shifted);
        self.buf.extend(shifted.iter().map(|c| c.re));
    }

    fn flush_slot(&mut self) {
        // Only bother if we captured most of the slot; a partial buffer at
        // startup would just waste a decode.
        self.last_fill = self.buf.len() as f32 / self.nmax as f32 * 100.0;
        if self.buf.len() >= self.nmax * 3 / 4 {
            let audio = quantize(&self.buf, self.nmax);
            self.pending = true;
            let _ = self.jobs.send(SlotJob {
                audio,
                my_call: self.my_call.clone(),
            });
        }
        self.buf.clear();
        self.buf.reserve(self.nmax + 4096);
    }
}

/// Normalise a whole slot to a comfortable fraction of full scale and
/// quantise to the i16 audio the FT decoder expects. One fixed gain for the
/// entire slot; after narrowing a 192 kHz span down to 3 kHz the residual
/// amplitude is tiny, so a fixed absolute scale would quantise weak signals
/// into nothing.
fn quantize(buf: &[f32], nmax: usize) -> Vec<i16> {
    let mut audio = buf.to_vec();
    audio.resize(nmax, 0.0);
    let sum_sq: f32 = audio.iter().map(|v| v * v).sum();
    let rms = (sum_sq / audio.len().max(1) as f32).sqrt();
    let gain = if rms > 1e-12 {
        (4000.0 / rms).min(1.0e7)
    } else {
        1.0
    };
    audio
        .iter()
        .map(|v| (v * gain).clamp(-32_767.0, 32_767.0) as i16)
        .collect()
}

fn current_slot(slot_secs: f64) -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    (now / slot_secs) as u64
}

fn decode_slot(
    audio: &[i16],
    ft4: bool,
    my_call: &str,
    hash: &mut CallsignHashTable,
) -> Vec<FtMessage> {
    let stamp = slot_stamp();
    let mut out = Vec::new();
    if ft4 {
        // Three SIC rounds is WSJT-X's nsp ceiling: decode, subtract,
        // decode the residual. Recovers weak stations sitting inside a
        // strong neighbour's occupied bandwidth — the situation a hot
        // front end (bias-T + LNA) creates. Local Costas EQ flattens
        // per-tone fading before the LLRs are built.
        let res = DecodeRequest::<Ft4>::new(audio, FREQ_MIN, FREQ_MAX, 0.9, MAX_CAND)
            .sic_rounds(3)
            .eq_mode(EqMode::Local)
            .osd(true)
            .decode()
            .results;
        collect_results(&res, &stamp, hash, &mut out);
    } else {
        // sic_early: WSJT-X's early-decode architecture — progressively
        // longer audio prefixes with subtraction between checkpoints. A
        // recall superset of plain multi-pass SIC.
        //
        // When we know our own callsign, lock those bits as an a-priori
        // hint (messages addressed to us). That drops the decode
        // threshold by a couple of dB on those candidates, which is the
        // difference between hearing a reply and missing it.
        let ap = if !my_call.is_empty() {
            Some(ApHint::new().with_call2(my_call))
        } else {
            None
        };
        let mut req = DecodeRequest::<Ft8>::new(audio, FREQ_MIN, FREQ_MAX, 0.9, MAX_CAND)
            .sic_early()
            .eq_mode(EqMode::Local)
            .osd(true);
        if let Some(ref hint) = ap {
            req = req.ap_hint(hint);
        }
        let res = req.decode().results;
        collect_results(&res, &stamp, hash, &mut out);
    }
    out
}

fn collect_results(
    res: &[mfsk_core::engine::pipeline::DecodeResult],
    stamp: &str,
    hash: &mut CallsignHashTable,
    out: &mut Vec<FtMessage>,
) {
    for r in res {
        let Some(text) = unpack77_with_hash(r.message77(), hash) else {
            continue;
        };
        if !is_plausible_message(&text) {
            continue;
        }
        for tok in text.split_whitespace() {
            let t = tok.trim_matches(|c| c == '<' || c == '>');
            if is_callsign(t) {
                hash.insert(t);
            }
        }
        out.push(FtMessage {
            stamp: stamp.to_string(),
            snr_db: r.snr_db,
            dt_sec: r.dt_sec,
            freq_hz: r.freq_hz,
            text,
        });
    }
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

    fn wants_agc(&self) -> bool {
        // `quantize` already scales each finished slot by a single gain, for
        // the reason spelled out there: a station appearing mid-slot must not
        // move the noise floor under the LDPC soft metrics part-way through a
        // message. A block-rate AGC ahead of the buffer reintroduces exactly
        // that, so the slot capture takes its audio unridden.
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
        self.slot = current_slot(self.slot_secs);
    }

    fn set_station(&mut self, call: &str, _grid: &str) {
        self.my_call = call.to_string();
    }
}
