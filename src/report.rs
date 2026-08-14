//! Reporting reception spots to pskreporter.info.
//!
//! Protocol: simplified IPFIX (RFC 5101) datagrams over UDP to
//! report.pskreporter.info:4739, as documented at
//! <https://www.pskreporter.info/pskdev.html>. Each datagram carries one
//! receiver record (us) plus one sender record per spotted station. The
//! housekeeping rules from that page are honoured here: templates ride along
//! with the first three datagrams and hourly after that, datagrams are sent
//! at most once per five minutes unless full, timed from program start with
//! jitter rather than synchronised to the wall clock, the sequence number
//! counts *records*, and a station is re-reported after an hour or when it
//! changes band.

use crate::decoders::FtMessage;
use std::collections::HashMap;
use std::net::{ToSocketAddrs, UdpSocket};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const DEST: &str = "report.pskreporter.info:4739";
/// Template IDs linking the record descriptors to the data blocks. Arbitrary
/// per the spec; chosen not to collide with the cookie-cutter examples.
const TPL_RX: u16 = 0x9A92;
const TPL_TX: u16 = 0x9A93;
/// PSKReporter's IANA enterprise number (30351).
const ENT: [u8; 4] = [0x00, 0x00, 0x76, 0x8F];
const SEND_INTERVAL: Duration = Duration::from_secs(300);
const MAX_RECORDS: usize = 60;
/// Re-report a station after this long, or sooner on a band change.
const REREPORT_SECS: u32 = 3600;
/// Shown as `decodingSoftware` on pskreporter.info.
///
/// The site treats the first ASCII digit in this field as the start of the
/// version. Putting a callsign such as `KQ2Y` *before* the version therefore
/// displays as `HFScan by KQ`. Version comes first so the full credit survives.
const SOFTWARE: &str = concat!("HFScan v", env!("CARGO_PKG_VERSION"), " by KQ2Y");

pub struct Spot {
    call: String,
    grid: Option<String>,
    freq_hz: u64,
    snr_db: i32,
    mode: String,
    time: u32,
}

/// Heuristic callsign test: digital-mode messages also contain grids (FN42),
/// signal reports (-12, R+05) and fixed words (CQ, 73, RR73), none of which
/// are callsigns.
pub fn is_callsign(tok: &str) -> bool {
    if !(3..=12).contains(&tok.len()) {
        return false;
    }
    if !tok.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'/') {
        return false;
    }
    if !tok.bytes().any(|b| b.is_ascii_digit()) || !tok.bytes().any(|b| b.is_ascii_alphabetic()) {
        return false;
    }
    let b = tok.as_bytes();
    // Grid locator: two letters A-R followed by two digits.
    if tok.len() == 4
        && b[0].is_ascii_alphabetic()
        && b[1].is_ascii_alphabetic()
        && b[0].to_ascii_uppercase() <= b'R'
        && b[1].to_ascii_uppercase() <= b'R'
        && b[2].is_ascii_digit()
        && b[3].is_ascii_digit()
    {
        return false;
    }
    // Signal report: optional leading R, then only digits and signs.
    let r = tok.strip_prefix('R').unwrap_or(tok);
    if r.bytes().all(|c| c == b'+' || c == b'-' || c.is_ascii_digit()) {
        return false;
    }
    tok != "RR73"
}

/// Maidenhead locator: two letters A-R, two digits, optionally two letters A-X.
pub fn is_grid(tok: &str) -> bool {
    let b = tok.as_bytes();
    let ok4 = b.len() >= 4
        && b[0].is_ascii_alphabetic()
        && b[1].is_ascii_alphabetic()
        && b[0].to_ascii_uppercase() <= b'R'
        && b[1].to_ascii_uppercase() <= b'R'
        && b[2].is_ascii_digit()
        && b[3].is_ascii_digit();
    match b.len() {
        4 => ok4,
        6 => {
            ok4 && b[4].is_ascii_alphabetic()
                && b[5].is_ascii_alphabetic()
                && b[4].to_ascii_uppercase() <= b'X'
                && b[5].to_ascii_uppercase() <= b'X'
        }
        _ => false,
    }
}

/// Extract the *transmitting* station from a decoded FT8/FT4 message.
///
/// FT8 messages are "<addressee> <source> <data>", so the sender is the first
/// callsign-looking token after the first word — that covers both
/// "K1ABC W9XYZ -12" (W9XYZ transmits) and "CQ [DX] K1ABC FN42" (K1ABC
/// transmits, and reveals their grid).
fn sender_of(text: &str) -> Option<(String, Option<String>)> {
    let toks: Vec<&str> = text.split_whitespace().collect();
    for (i, t) in toks.iter().enumerate().skip(1) {
        let t = t.trim_matches(|c| c == '<' || c == '>');
        if is_callsign(t) {
            let grid = toks
                .get(i + 1)
                .map(|g| g.trim_matches(|c| c == '<' || c == '>'))
                .filter(|g| is_grid(g))
                .map(str::to_string);
            return Some((t.to_string(), grid));
        }
    }
    None
}

/// Length-prefixed variable-length string field, per the spec.
fn str_field(out: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    let n = bytes.len().min(254);
    out.push(n as u8);
    out.extend(&bytes[..n]);
}

fn pad4(out: &mut Vec<u8>) {
    while out.len() % 4 != 0 {
        out.push(0);
    }
}

/// Build one datagram. Pure and allocation-only, so the tests can check the
/// byte layout against the worked examples in the spec.
fn build_packet(
    rx_call: &str,
    rx_grid: &str,
    software: &str,
    spots: &[Spot],
    seq: u32,
    session: u32,
    now: u32,
    templates: bool,
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend([0x00, 0x0A, 0x00, 0x00]); // version 10, length patched last
    out.extend(now.to_be_bytes());
    out.extend(seq.to_be_bytes());
    out.extend(session.to_be_bytes());

    if templates {
        // Receiver (options) template: receiverCallsign, receiverLocator,
        // decodingSoftware, antennaInformation.
        let mut t = vec![0x00, 0x03, 0x00, 0x2C];
        t.extend(TPL_RX.to_be_bytes());
        t.extend([0x00, 0x04, 0x00, 0x01]); // 4 fields, 1 scope field
        for f in [
            [0x80, 0x02, 0xFF, 0xFF],
            [0x80, 0x04, 0xFF, 0xFF],
            [0x80, 0x08, 0xFF, 0xFF],
            [0x80, 0x09, 0xFF, 0xFF],
        ] {
            t.extend(f);
            t.extend(ENT);
        }
        t.extend([0x00, 0x00]); // pad to a multiple of 4
        out.extend(t);

        // Sender template: senderCallsign, frequency(4), sNR(1), mode,
        // informationSource(1), senderLocator, flowStartSeconds(4).
        let mut t = vec![0x00, 0x02, 0x00, 0x3C];
        t.extend(TPL_TX.to_be_bytes());
        t.extend([0x00, 0x07]);
        for f in [
            [0x80, 0x01, 0xFF, 0xFF],
            [0x80, 0x05, 0x00, 0x04],
            [0x80, 0x06, 0x00, 0x01],
            [0x80, 0x0A, 0xFF, 0xFF],
            [0x80, 0x0B, 0x00, 0x01],
            [0x80, 0x03, 0xFF, 0xFF],
        ] {
            t.extend(f);
            t.extend(ENT);
        }
        t.extend([0x00, 0x96, 0x00, 0x04]); // flowStartSeconds, not enterprise
        out.extend(t);
    }

    // Receiver record: one per datagram.
    let mut rec = Vec::new();
    str_field(&mut rec, rx_call);
    str_field(&mut rec, rx_grid);
    str_field(&mut rec, software);
    str_field(&mut rec, ""); // antennaInformation: unknown
    pad4(&mut rec);
    out.extend(TPL_RX.to_be_bytes());
    out.extend((rec.len() as u16 + 4).to_be_bytes());
    out.extend(rec);

    // Sender records.
    let mut rec = Vec::new();
    for s in spots {
        str_field(&mut rec, &s.call);
        rec.extend((s.freq_hz as u32).to_be_bytes());
        rec.push(s.snr_db as i8 as u8);
        str_field(&mut rec, &s.mode);
        rec.push(1); // informationSource: automatically extracted
        str_field(&mut rec, s.grid.as_deref().unwrap_or(""));
        rec.extend(s.time.to_be_bytes());
    }
    pad4(&mut rec);
    out.extend(TPL_TX.to_be_bytes());
    out.extend((rec.len() as u16 + 4).to_be_bytes());
    out.extend(rec);

    let len = out.len() as u16;
    out[2..4].copy_from_slice(&len.to_be_bytes());
    out
}

/// Dedup and batching, split out from the thread so the re-report rule can
/// be exercised without a socket or the wall clock.
///
/// This is the part that decides a spot is *not* worth sending, which is
/// what makes a healthy reporter look stalled: once every station audible on
/// a band has been reported, nothing new goes out until either a new station
/// appears or the hour is up.
struct SpotQueue {
    pending: Vec<Spot>,
    /// (call, MHz band, mode) -> when we last reported it.
    seen: HashMap<(String, u64, String), u32>,
    /// Spots dropped as already-reported since the last successful send.
    suppressed: usize,
}

impl SpotQueue {
    fn new() -> Self {
        Self {
            pending: Vec::new(),
            seen: HashMap::new(),
            suppressed: 0,
        }
    }

    /// True when the spot was queued, false when the hourly rule suppressed it.
    fn accept(&mut self, spot: Spot) -> bool {
        let key = (spot.call.clone(), spot.freq_hz / 1_000_000, spot.mode.clone());
        let fresh = self
            .seen
            .get(&key)
            .map(|t| spot.time.saturating_sub(*t) >= REREPORT_SECS)
            .unwrap_or(true);
        if !fresh {
            self.suppressed += 1;
            return false;
        }
        if self.pending.len() >= 1000 {
            return false;
        }
        self.seen.insert(key, spot.time);
        self.pending.push(spot);
        true
    }

    fn ready(&self, due: bool) -> bool {
        !self.pending.is_empty() && (due || self.pending.len() >= MAX_RECORDS)
    }

    fn take_batch(&mut self) -> Vec<Spot> {
        let n = self.pending.len().min(MAX_RECORDS);
        self.pending.drain(..n).collect()
    }

    /// Put a failed batch back at the front, still bounded.
    fn requeue(&mut self, batch: Vec<Spot>) {
        let mut again = batch;
        again.append(&mut self.pending);
        again.truncate(1000);
        self.pending = again;
    }
}

/// UI-side handle: the reporter thread owns the socket, the dedup table and
/// the send timer so a DNS hiccup can never stall the display.
pub struct Reporter {
    tx: SyncSender<Spot>,
    my_call: String,
    sent: Arc<AtomicUsize>,
    queued: Arc<AtomicUsize>,
    /// Unix seconds of the last successful datagram; 0 until the first one.
    last_send: Arc<AtomicU64>,
    /// Decodes recognised but already reported this hour.
    suppressed: Arc<AtomicUsize>,
}

/// What the status line needs to tell "working, nothing new to say" apart
/// from "wedged", which a bare cumulative total cannot do.
pub struct ReportStats {
    pub sent: usize,
    pub queued: usize,
    pub suppressed: usize,
    /// Seconds since the last successful send, or `None` if none yet.
    pub since_send: Option<u64>,
}

impl Reporter {
    pub fn start(my_call: String, my_grid: String, log: SyncSender<String>) -> Reporter {
        let (tx, rx) = sync_channel::<Spot>(512);
        let counts = Counters {
            sent: Arc::new(AtomicUsize::new(0)),
            queued: Arc::new(AtomicUsize::new(0)),
            last_send: Arc::new(AtomicU64::new(0)),
            suppressed: Arc::new(AtomicUsize::new(0)),
        };
        let thread_counts = counts.clone();
        let call = my_call.clone();
        std::thread::spawn(move || run(rx, log, call, my_grid, thread_counts));
        Reporter {
            tx,
            my_call,
            sent: counts.sent,
            queued: counts.queued,
            last_send: counts.last_send,
            suppressed: counts.suppressed,
        }
    }

    pub fn stats(&self) -> ReportStats {
        let last = self.last_send.load(Ordering::Relaxed);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        ReportStats {
            sent: self.sent.load(Ordering::Relaxed),
            queued: self.queued.load(Ordering::Relaxed),
            suppressed: self.suppressed.load(Ordering::Relaxed),
            since_send: (last > 0).then(|| now.saturating_sub(last)),
        }
    }

    /// Queue a spot for the station that transmitted this decode. Our own
    /// callsign is never spotted.
    pub fn spot(&self, m: &FtMessage, dial_hz: f64, mode: &str) {
        let Some((call, grid)) = sender_of(&m.text) else {
            return;
        };
        if call == self.my_call {
            return;
        }
        let time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as u32)
            .unwrap_or(0);
        let _ = self.tx.try_send(Spot {
            call,
            grid,
            freq_hz: (dial_hz + m.freq_hz as f64).max(0.0) as u64,
            snr_db: m.snr_db as i32,
            mode: mode.to_string(),
            time,
        });
    }
}

#[derive(Clone)]
struct Counters {
    sent: Arc<AtomicUsize>,
    queued: Arc<AtomicUsize>,
    last_send: Arc<AtomicU64>,
    suppressed: Arc<AtomicUsize>,
}

/// Tiny xorshift for the send-time jitter; avoids pulling in a rand crate.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn run(
    rx: Receiver<Spot>,
    log: SyncSender<String>,
    my_call: String,
    my_grid: String,
    counts: Counters,
) {
    let sock = match UdpSocket::bind("0.0.0.0:0") {
        Ok(s) => s,
        Err(e) => {
            let _ = log.try_send(format!("pskreporter: socket failed: {e}"));
            return;
        }
    };
    let mut rng = Rng(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64 ^ d.as_secs())
            .unwrap_or(0x9e3779b9)
            | 1,
    );
    let session = rng.next() as u32;
    let mut seq: u32 = 0; // counts records, per the spec
    let mut packets: u32 = 0;
    let mut q = SpotQueue::new();
    let mut templates_at = Instant::now() - Duration::from_secs(3600);
    let mut next_send = Instant::now() + SEND_INTERVAL + Duration::from_secs(rng.next() % 60);

    loop {
        match rx.recv_timeout(Duration::from_secs(1)) {
            Ok(spot) => {
                q.accept(spot);
                counts.queued.store(q.pending.len(), Ordering::Relaxed);
                counts.suppressed.store(q.suppressed, Ordering::Relaxed);
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return,
        }

        if !q.ready(Instant::now() >= next_send) {
            continue;
        }
        let templates = packets < 3 || templates_at.elapsed() >= Duration::from_secs(3600);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as u32)
            .unwrap_or(0);
        let chunk = q.take_batch();
        let pkt = build_packet(&my_call, &my_grid, SOFTWARE, &chunk, seq, session, now, templates);
        let dest = DEST.to_socket_addrs().ok().and_then(|mut a| a.next());
        match dest {
            Some(addr) => match sock.send_to(&pkt, addr) {
                Ok(_) => {
                    seq += chunk.len() as u32;
                    packets += 1;
                    if templates {
                        templates_at = Instant::now();
                    }
                    counts.sent.fetch_add(chunk.len(), Ordering::Relaxed);
                    counts.last_send.store(now as u64, Ordering::Relaxed);
                    // Say what was held back too: a run of "0 new" is the
                    // difference between a quiet band and a broken reporter.
                    let dup = std::mem::take(&mut q.suppressed);
                    counts.suppressed.store(0, Ordering::Relaxed);
                    let held = if dup > 0 {
                        format!(", {dup} already reported this hour")
                    } else {
                        String::new()
                    };
                    let _ = log.try_send(format!(
                        "pskreporter: {} spot(s) sent{held}",
                        chunk.len()
                    ));
                }
                Err(e) => {
                    // Keep the spots for the next attempt, bounded.
                    let _ = log.try_send(format!("pskreporter: send failed: {e}"));
                    q.requeue(chunk);
                }
            },
            None => {
                let _ = log.try_send(format!("pskreporter: cannot resolve {DEST}"));
                q.requeue(chunk);
            }
        }
        counts.queued.store(q.pending.len(), Ordering::Relaxed);
        next_send = Instant::now() + SEND_INTERVAL + Duration::from_secs(rng.next() % 60);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spot(call: &str, freq: u64, snr: i32, time: u32) -> Spot {
        Spot {
            call: call.into(),
            grid: None,
            freq_hz: freq,
            snr_db: snr,
            mode: "FT8".into(),
            time,
        }
    }

    #[test]
    fn finds_the_transmitting_station() {
        let (call, grid) = sender_of("CQ K1ABC FN42").unwrap();
        assert_eq!((call.as_str(), grid.as_deref()), ("K1ABC", Some("FN42")));
        let (call, grid) = sender_of("W9XYZ K1ABC R-05").unwrap();
        assert_eq!((call.as_str(), grid), ("K1ABC", None));
        let (call, _) = sender_of("CQ DX JA1ABC PM95").unwrap();
        assert_eq!(call, "JA1ABC");
        let (call, _) = sender_of("<K1ABC> W9XYZ RR73").unwrap();
        assert_eq!(call, "W9XYZ");
        assert!(sender_of("CQ").is_none());
        assert!(sender_of("K1ABC 599").is_none());
    }

    #[test]
    fn grid_locators() {
        for g in ["FN42", "PM95", "IO91wm", "AA00", "RR99xx"] {
            assert!(is_grid(g), "{g} should be a grid");
        }
        for ng in ["K1ABC", "FN", "FN4", "SN42", "FN429", "1234", "FN42X"] {
            assert!(!is_grid(ng), "{ng} should not be a grid");
        }
    }

    /// Receiver record layout checked against the worked example on
    /// pskreporter.info/pskdev.html (N1DQ, FN42hn, "Homebrew v5.6").
    #[test]
    fn receiver_record_matches_the_spec() {
        let pkt = build_packet("N1DQ", "FN42hn", "Homebrew v5.6", &[], 1, 0, 1200960114, false);
        let mut want = vec![
            0x00, 0x0A, 0x00, 0x00, // version, length (patched below)
            0x47, 0x95, 0x32, 0x72, // time
            0x00, 0x00, 0x00, 0x01, // sequence
            0x00, 0x00, 0x00, 0x00, // session id
        ];
        want.extend([
            0x9A, 0x92, 0x00, 0x20, // receiver block, 32 bytes total
            0x04, b'N', b'1', b'D', b'Q', //
            0x06, b'F', b'N', b'4', b'2', b'h', b'n', //
            0x0D, b'H', b'o', b'm', b'e', b'b', b'r', b'e', b'w', b' ', b'v', b'5', b'.',
            b'6', //
            0x00, // empty antennaInformation
            0x00, // padding to 32
        ]);
        want.extend([0x9A, 0x93, 0x00, 0x04]); // empty sender block
        let len = (want.len() as u16).to_be_bytes();
        want[2] = len[0];
        want[3] = len[1];
        assert_eq!(pkt, want);
    }

    /// One sender record: callsign string, big-endian frequency, signed SNR
    /// byte, mode string, informationSource, empty locator, big-endian time.
    #[test]
    fn sender_record_layout() {
        let spots = [spot("N1DQ", 14070567, -12, 1200960084)];
        let pkt = build_packet("K1ABC", "FN42", "hfscan", &spots, 0, 0, 1200960114, true);
        // Length field must equal the packet size.
        assert_eq!(u16::from_be_bytes([pkt[2], pkt[3]]) as usize, pkt.len());
        // The sender record itself.
        let rec = [
            0x04, b'N', b'1', b'D', b'Q', // senderCallsign
            0x00, 0xD6, 0xB3, 0x27, // 14070567 Hz
            0xF4, // -12 dB
            0x03, b'F', b'T', b'8', // mode
            0x01, // informationSource: automatically extracted
            0x00, // no locator
            0x47, 0x95, 0x32, 0x54, // flowStartSeconds
        ];
        let pos = pkt
            .windows(rec.len())
            .position(|w| w == rec)
            .expect("sender record not found in packet");
        // The record sits right after the sender block header (20-byte
        // record + 4-byte header = 0x18).
        assert_eq!(&pkt[pos - 4..pos], &[0x9A, 0x93, 0x00, 0x18]);
    }

    #[test]
    fn software_id_does_not_put_callsign_digits_before_the_version() {
        // "HFScan by KQ2Y v0.1.0" is shown as "HFScan by KQ" because the
        // first digit (`2`) is parsed as the version. The version must be
        // the first digit run, and the author callsign must still be present.
        let first_digit = SOFTWARE
            .find(|c: char| c.is_ascii_digit())
            .expect("software id must include a version");
        assert!(
            SOFTWARE[..first_digit].ends_with('v'),
            "first digit must belong to the version, got {SOFTWARE:?}"
        );
        assert!(SOFTWARE.starts_with("HFScan v"), "got {SOFTWARE:?}");
        assert!(SOFTWARE.contains("KQ2Y"), "author credit missing from {SOFTWARE:?}");
    }

    /// The reason a working reporter looks stalled: everything audible has
    /// already been reported, and the rules say not to report it again yet.
    #[test]
    fn a_station_is_reported_once_an_hour_then_again() {
        let mut q = SpotQueue::new();
        assert!(q.accept(spot("K1ABC", 14_074_000, -5, 1_000_000)));
        // Same station, same band, same mode, minutes later: held back.
        assert!(!q.accept(spot("K1ABC", 14_074_500, -7, 1_000_600)));
        assert_eq!(q.suppressed, 1);
        assert_eq!(q.pending.len(), 1);
        // An hour on, it is worth reporting again.
        assert!(q.accept(spot("K1ABC", 14_074_000, -6, 1_003_600)));
        assert_eq!(q.pending.len(), 2);
        // A different band is a different report immediately.
        assert!(q.accept(spot("K1ABC", 7_074_000, -6, 1_000_700)));
    }

    /// A batch that could not go out has to come back, or the spots are lost
    /// for an hour: `seen` already counts them as reported.
    #[test]
    fn an_unsent_batch_is_requeued_not_dropped() {
        let mut q = SpotQueue::new();
        for i in 0..3 {
            assert!(q.accept(spot(&format!("K{i}AAA"), 14_074_000, -5, 1_000_000)));
        }
        let batch = q.take_batch();
        assert_eq!(batch.len(), 3);
        assert!(q.pending.is_empty());
        q.requeue(batch);
        assert_eq!(q.pending.len(), 3, "a failed send must not lose the spots");
        // And they are still suppressed as duplicates, so nothing double-sends.
        assert!(!q.accept(spot("K0AAA", 14_074_000, -5, 1_000_100)));
    }

    /// Bursts bigger than one datagram must drain in batches rather than
    /// waiting five minutes per 60 spots.
    #[test]
    fn a_large_burst_is_ready_immediately() {
        let mut q = SpotQueue::new();
        for i in 0..150 {
            q.accept(spot(&format!("K{i}XYZ"), 14_074_000, -5, 1_000_000));
        }
        assert!(q.ready(false), "a full queue should not wait for the timer");
        assert_eq!(q.take_batch().len(), MAX_RECORDS);
        assert!(q.ready(false));
        assert_eq!(q.take_batch().len(), MAX_RECORDS);
        // The tail waits for the timer.
        assert!(!q.ready(false));
        assert!(q.ready(true));
    }

    #[test]
    fn template_lengths_are_internally_consistent() {
        let spots = [spot("K1ABC", 14074000, 3, 1200960084)];
        let pkt = build_packet("K1ABC", "FN42", "hfscan", &spots, 0, 0, 1200960114, true);
        assert_eq!(u16::from_be_bytes([pkt[2], pkt[3]]) as usize, pkt.len());
        // Both template set headers and both data block headers are present.
        for magic in [[0x00, 0x03], [0x00, 0x02], [0x9A, 0x92], [0x9A, 0x93]] {
            assert!(
                pkt.windows(2).any(|w| w == magic),
                "missing {magic:02X?}"
            );
        }
    }
}
