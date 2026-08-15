//! hfscan - a terminal HF panadapter and digital-mode decoder for the
//! SDRplay RSP1A (or any SoapySDR device).

mod bands;
mod decoders;
mod dsp;
mod identify;
mod radio;
mod report;

use anyhow::Result;
use clap::Parser;
use decoders::cw::{self, CwHit};
use decoders::psk31::{self, PskHit};
use decoders::{CwView, Decoder, FtMessage, Mode, PskView};
use dsp::{smooth_bins, DecodeChain, SoftAgc, Spectrum};
use num_complex::Complex32;
use ratatui::crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers, MouseEventKind,
};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};
use report::{is_callsign, Reporter};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// A radio rate that divides evenly by both 8 kHz and 12 kHz, so every mode
/// (including FT8/FT4) gets its exact audio rate.
const FT_SAFE_RATE: f64 = 192_000.0;

const WF_INTERVALS_MS: [u64; 5] = [100, 250, 500, 1000, 2000];

/// Selectable FFT sizes. Larger means finer frequency resolution at the cost of
/// a slower spectrum update, since a full segment has to be collected first.
/// 65536 is 2.9 Hz per bin on a 192 kHz span and takes ~0.34 s to fill, which
/// is what a deeply zoomed view needs before the display, not the data, is
/// the thing limiting detail.
const FFT_SIZES: [usize; 7] = [1024, 2048, 4096, 8192, 16384, 32768, 65536];

/// Ceiling on waterfall history, in total floats. Rows are full-resolution
/// spectra, so the row count has to fall as the FFT grows or the history
/// alone would run to hundreds of megabytes. Every size up to 32768 still
/// gets the full 400 rows.
const WF_HISTORY_FLOATS: usize = 16 << 20;
const WF_MAX_ROWS: usize = 400;

/// How finely the waterfall subdivides a terminal cell.
///
/// A cell can carry exactly two colours, so the ceiling on detail is set by
/// how many subcells those two colours are made to cover. Half blocks split a
/// cell one way and paint each half exactly; quadrant glyphs split it both
/// ways and approximate the four values with the best two-colour fit, which
/// is the better trade for a heat map whose neighbours are usually similar.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WfRes {
    /// 2×2 subcells per character — twice the detail of either half-block mode.
    Quad,
    /// Two frequency bins per column, one time step per row.
    Freq,
    /// One frequency bin per column, two time steps per row.
    Time,
}

impl WfRes {
    const ALL: [WfRes; 3] = [WfRes::Quad, WfRes::Freq, WfRes::Time];

    fn next(self) -> Self {
        let i = Self::ALL.iter().position(|x| *x == self).unwrap_or(0);
        Self::ALL[(i + 1) % Self::ALL.len()]
    }

    fn label(self) -> &'static str {
        match self {
            WfRes::Quad => "2x frequency, 2x time (quadrant)",
            WfRes::Freq => "2x frequency, 1x time",
            WfRes::Time => "1x frequency, 2x time",
        }
    }

    /// Frequency subcells per column, and time steps per text row.
    fn cells(self) -> (usize, usize) {
        match self {
            WfRes::Quad => (2, 2),
            WfRes::Freq => (2, 1),
            WfRes::Time => (1, 2),
        }
    }
}

/// Most decoders one span can carry. Each is a tuning chain plus a decoder;
/// the chains alone measure well under 1% of a core each, but the transcript
/// stops being readable long before the CPU runs out.
const MAX_AUTO_SLOTS: usize = 10;
/// Drop a narrowband slot whose signal the classifier has not seen for this
/// long. FT8/FT4 slots are pinned to their calling frequencies instead.
const AUTO_IDLE: Duration = Duration::from_secs(25);
/// Flush a partial line from a character-at-a-time decoder after this many
/// seconds *of audio* without a new character, so a station that stops
/// sending still gets its last words shown. Measured in audio rather than
/// wall clock so the behaviour does not depend on how fast blocks arrive.
const AUTO_FLUSH_SECS: f64 = 2.0;
/// Emit a line at this length even mid-transmission, so a long-winded
/// station is not silent for the minute it takes to fill a screen line.
const AUTO_LINE: usize = 48;

/// Longest transcript the auto pane keeps. Rows are fixed-height, so this is
/// also how far back the pane can be scrolled.
const DECODE_LOG_MAX: usize = 800;

/// Characters of rolling copy a held row keeps. Only the tail that fits the
/// pane is ever drawn; the rest is what survives a narrow pane being widened.
const ROW_COPY_MAX: usize = 512;
/// A row with copy this recent counts as live, and sorts above the rest.
const ROW_ACTIVE: Duration = Duration::from_secs(20);
/// A row silent this long is forgotten entirely.
const ROW_RETIRE: Duration = Duration::from_secs(600);
/// Most held rows kept. The oldest silent row goes first.
const MAX_ROWS: usize = 60;

/// Which face the auto pane is showing.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AutoView {
    /// One held row per signal, copy accumulating in place.
    Rows,
    /// Every line of copy in the order it arrived.
    Log,
}

/// One signal the auto decoders are holding, and everything heard on it.
///
/// The chronological log emits a line only when a decoder finishes one, so a
/// slow CW station shows a few characters at a time and they scroll away.
/// A held row instead accumulates into a buffer that stays put on screen, so
/// the copy builds up in place and can be read as it arrives.
struct SignalRow {
    dial_hz: f64,
    kind: identify::Kind,
    mode: &'static str,
    /// Rolling copy, oldest trimmed off the front. Already sanitised.
    copy: String,
    /// When copy last arrived — decides live versus silent, and the order.
    last_copy: Instant,
}

impl SignalRow {
    fn live(&self, now: Instant) -> bool {
        now.duration_since(self.last_copy) < ROW_ACTIVE
    }

    /// Add what just came off the decoder, keeping the buffer bounded.
    fn push_copy(&mut self, text: &str) {
        self.copy.push_str(text);
        self.last_copy = Instant::now();
        // Trimmed on a character boundary, from the front, so the newest copy
        // — the part actually on screen — is never what gets dropped.
        let over = self.copy.chars().count().saturating_sub(ROW_COPY_MAX);
        if over > 0 {
            let cut = self
                .copy
                .char_indices()
                .nth(over)
                .map_or(self.copy.len(), |(i, _)| i);
            self.copy.drain(..cut);
        }
    }
}

/// One line of copy from one automatic decoder, kept in columns.
///
/// Keeping the pieces apart is what makes the pane both structured (the
/// frequency and mode columns line up, and are coloured per mode) and safe to
/// scroll: every entry renders to exactly one row, so the row count is known
/// without measuring text, and the viewport can never disagree with the
/// scroll offset about how tall the content is.
#[derive(Clone)]
struct DecodeEntry {
    /// UTC hh:mm:ss when the line was emitted.
    stamp: String,
    /// Where the copy was heard. Zero for the scanner's own remarks, which
    /// render without the frequency and mode columns.
    dial_hz: f64,
    kind: identify::Kind,
    mode: &'static str,
    /// Already sanitised: printable, no control characters, no newlines.
    text: String,
}

/// Strip anything that would corrupt the terminal grid out of decoder output.
///
/// Demodulators fed noise produce arbitrary bytes. Escape sequences, C0/C1
/// controls, zero-width and combining characters all render as fewer (or
/// more) cells than the layout assumed, which shifts every following cell and
/// leaves the frame's own border characters in the wrong columns — the damage
/// then persists frame to frame. Everything kept here occupies exactly one
/// cell. Newlines are dropped: callers split on them first.
fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            ' '..='~' => c,
            // Punctuation a decoder may legitimately produce, plus the few
            // marks the scanner writes into its own remarks. All single-cell.
            '°' | '±' | 'µ' | '→' | '←' | '—' | '–' | '…' | '·' => c,
            _ => ' ',
        })
        .collect()
}

/// `sanitize` for a whole transcript, keeping line structure intact.
fn sanitize_text(s: &str) -> String {
    s.split('\n')
        .map(sanitize)
        .collect::<Vec<_>>()
        .join("\n")
}

/// One automatically tuned decoder inside the current span.
///
/// `Mode::Auto` runs a fleet of these instead of the single cursor-following
/// chain. Each owns its tuning, so a CW station at one end of the span and an
/// FT8 pile-up at the other are decoded from the same IQ simultaneously.
struct AutoSlot {
    /// Absolute dial frequency this decoder is tuned to.
    dial_hz: f64,
    kind: identify::Kind,
    /// Set for FT8/FT4 slots, which are pinned to a calling frequency and
    /// must not be retired just because the band went quiet for a slot.
    pinned: bool,
    chain: DecodeChain,
    agc: SoftAgc,
    decoder: Box<dyn Decoder>,
    audio: Vec<Complex32>,
    /// Text from a character-at-a-time decoder not yet emitted as a line.
    partial: String,
    last_seen: Instant,
    /// Audio samples processed since the decoder last produced a character.
    quiet: usize,
}

impl AutoSlot {
    fn new(kind: identify::Kind, dial_hz: f64, center: f64, rate: f64, pinned: bool) -> Option<Self> {
        let mode = match kind {
            identify::Kind::Cw => Mode::Cw,
            identify::Kind::Rtty => Mode::Rtty,
            identify::Kind::Psk31 => Mode::Psk31,
            identify::Kind::Ft8 => Mode::Ft8,
            identify::Kind::Ft4 => Mode::Ft4,
            _ => return None,
        };
        let mut chain = DecodeChain::new(rate, 400.0, mode.audio_rate());
        let decoder = mode.make(chain.fs_out())?;
        chain.set_bandwidth(decoder.bandwidth());
        chain.set_offset(dial_hz - center + decoder.offset_shift());
        let now = Instant::now();
        Some(Self {
            dial_hz,
            kind,
            pinned,
            agc: SoftAgc::new(chain.fs_out()),
            chain,
            decoder,
            audio: Vec::new(),
            partial: String::new(),
            last_seen: now,
            quiet: 0,
        })
    }

    /// How close another detection has to be to count as the same signal.
    fn same_signal(&self, kind: identify::Kind, dial_hz: f64) -> bool {
        let slack = match kind {
            identify::Kind::Rtty => 300.0,
            identify::Kind::Ft8 | identify::Kind::Ft4 => 2000.0,
            _ => 120.0,
        };
        self.kind == kind && (self.dial_hz - dial_hz).abs() < slack
    }
}

/// The sixteen quadrant glyphs, indexed by a bitmask of which quarters take
/// the foreground colour: bit 0 upper-left, 1 upper-right, 2 lower-left,
/// 3 lower-right.
#[rustfmt::skip]
const QUADRANTS: [char; 16] = [
    ' ', '▘', '▝', '▀', '▖', '▌', '▞', '▛',
    '▗', '▚', '▐', '▜', '▄', '▙', '▟', '█',
];

/// Fit four subcell values into the two colours a cell can hold.
///
/// The split goes wherever the widest gap in the sorted values falls, so an
/// edge — a carrier standing out of the noise floor — lands on the colour
/// boundary and stays crisp. Returns the glyph plus the foreground and
/// background values to colour it with.
fn quad_cell(v: [f32; 4]) -> (char, f32, f32) {
    let mut idx = [0usize, 1, 2, 3];
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut cut = 0usize;
    let mut widest = -1.0f32;
    for k in 0..3 {
        let gap = v[idx[k + 1]] - v[idx[k]];
        if gap > widest {
            widest = gap;
            cut = k;
        }
    }
    // A cell with nothing to separate is a solid block. Splitting it anyway
    // would pick some arbitrary three-quarter glyph and lean on the terminal
    // to draw two identical colours seamlessly.
    if widest <= 1.0 / 255.0 {
        let mean = (v[0] + v[1] + v[2] + v[3]) * 0.25;
        return ('█', mean, mean);
    }
    let n_lo = cut + 1;
    let lo = idx[..n_lo].iter().map(|&i| v[i]).sum::<f32>() / n_lo as f32;
    let hi = idx[n_lo..].iter().map(|&i| v[i]).sum::<f32>() / (4 - n_lo) as f32;
    let mask = idx[n_lo..].iter().fold(0usize, |m, &i| m | 1 << i);
    (QUADRANTS[mask], hi, lo)
}

/// Temporal weight of each new spectrum estimate (the rest is history).
const SMOOTH_TIME: [f32; 3] = [0.28, 0.12, 0.06];
/// Frequency-domain binomial width: 1 = off, 3 = light, 5 = heavy.
const SMOOTH_BINS: [usize; 3] = [1, 3, 5];
const SMOOTH_LABELS: [&str; 3] = ["light", "medium", "heavy"];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AgcMode {
    /// Hang AGC on the decoder path + slow hardware trim.
    Soft,
    /// Device AGC. Fast, and it pumps the whole spectrum.
    Hardware,
    /// Manual IF/RF gain.
    Off,
}

impl AgcMode {
    fn next(self) -> Self {
        match self {
            AgcMode::Soft => AgcMode::Hardware,
            AgcMode::Hardware => AgcMode::Off,
            AgcMode::Off => AgcMode::Soft,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RxFilter {
    Auto,
    Hz80,
    Hz200,
    Hz500,
    Hz1500,
    Hz3000,
}

impl RxFilter {
    const ALL: [RxFilter; 6] = [
        RxFilter::Auto,
        RxFilter::Hz80,
        RxFilter::Hz200,
        RxFilter::Hz500,
        RxFilter::Hz1500,
        RxFilter::Hz3000,
    ];
    fn next(self) -> Self {
        let i = Self::ALL.iter().position(|x| *x == self).unwrap_or(0);
        Self::ALL[(i + 1) % Self::ALL.len()]
    }
    fn label(self) -> &'static str {
        match self {
            RxFilter::Auto => "auto",
            RxFilter::Hz80 => "80 Hz",
            RxFilter::Hz200 => "200 Hz",
            RxFilter::Hz500 => "500 Hz",
            RxFilter::Hz1500 => "1.5 kHz",
            RxFilter::Hz3000 => "3 kHz",
        }
    }
    fn hz(self, mode_default: f32) -> f32 {
        match self {
            RxFilter::Auto => mode_default,
            RxFilter::Hz80 => 80.0,
            RxFilter::Hz200 => 200.0,
            RxFilter::Hz500 => 500.0,
            RxFilter::Hz1500 => 1500.0,
            RxFilter::Hz3000 => 3000.0,
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "hfscan", about = "HF band scanner and digital decoder for the RSP1A")]
struct Args {
    /// SoapySDR device arguments
    #[arg(long, default_value = "driver=sdrplay")]
    device: String,
    /// Starting centre frequency in Hz (accepts e.g. 14070000).
    /// The default sits 10 kHz below the 20 m digital segment rather than on
    /// it — see `bands::Band::default` for why that matters.
    #[arg(short, long, default_value_t = 14_060_000.0)]
    freq: f64,
    /// Sample rate in Hz; this is also the width of the spectrum view
    #[arg(short, long, default_value_t = FT_SAFE_RATE)]
    rate: f64,
    /// FFT size (1024..32768); higher gives finer resolution
    #[arg(long, default_value_t = 8192)]
    fft: usize,
    /// Start with a decoder active: off, cw, rtty, psk31, ft8, ft4, auto
    /// ("auto" decodes every digital signal it finds across the span)
    #[arg(short, long, default_value = "off")]
    mode: String,
    /// Your amateur radio callsign — enables spot reporting to pskreporter.info
    #[arg(long)]
    call: Option<String>,
    /// Your Maidenhead grid locator (e.g. FN42), sent with reception reports
    #[arg(long)]
    grid: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ScanKind {
    Energy,
    Psk31,
    Cw,
}

#[derive(Clone)]
struct ScanHit {
    freq: f64,
    score: f32,
    label: &'static str,
}

struct ScanState {
    end: f64,
    step: f64,
    cur: f64,
    dwell_until: Instant,
    results: Vec<ScanHit>,
    kind: ScanKind,
}

struct App {
    center: f64,
    rate: f64,
    cursor: f64, // offset from centre, Hz
    zoom: f64,   // 1.0 = whole span; higher zooms in around the cursor
    gain: f64,
    agc: AgcMode,
    biast: bool,
    smooth_idx: usize,
    rx_filter: RxFilter,
    soft_agc: SoftAgc,
    hw_trim_at: Instant,
    hw_hot: u32,
    hw_quiet: u32,
    band_idx: usize,
    /// Last hardware gain used on each HF band, so a hot band cannot
    /// starve the next one after `b`.
    band_gains: Vec<f64>,

    spectrum: Vec<f32>,
    noise_tracker: dsp::NoiseFloor,
    noise_floor: Vec<f32>,
    spec_work: Vec<f32>,
    smoothed: Vec<f32>,
    waterfall: VecDeque<Vec<f32>>,
    wf_accum: Vec<f32>,
    wf_last: Instant,
    wf_idx: usize,
    fft_idx: usize,
    floor_db: f32,
    ceil_db: f32,

    mode: Mode,
    decoder: Option<Box<dyn Decoder>>,
    chain: DecodeChain,
    text: String,
    /// One row per line of copy in `Mode::Auto`, newest last. Auto mode has
    /// many decoders talking at once, so its output is kept as records with
    /// their own columns rather than folded into the flat transcript.
    decode_log: VecDeque<DecodeEntry>,
    /// Receiver front-end cleanup, and the block it works in.
    front: dsp::FrontEnd,
    iq_buf: Vec<Complex32>,
    /// One held row per signal heard in `Mode::Auto`. Live signals sort to
    /// the top, signals that have gone quiet sink to the bottom.
    rows: Vec<SignalRow>,
    auto_view: AutoView,
    /// Structured FT8/FT4 decodes, newest last; drives the FT panes.
    ft_msgs: VecDeque<FtMessage>,
    /// Stations heard, in first-heard order so the list updates in place.
    stations: Vec<(String, Station)>,
    /// Decode pane size: 0 = default, 1 = large, 2 = huge.
    decode_zoom: u8,
    /// Scroll offsets: transcript lines up from live, stations/slots skipped,
    /// waterfall entries back in time. Zero means pinned to live.
    msg_scroll: usize,
    /// Rows the decode pane last drew with. The scroll clamp needs the
    /// viewport height, which only the renderer knows; recording it here
    /// keeps the offset and what is on screen from disagreeing about how far
    /// back the copy goes.
    msg_rows: std::cell::Cell<usize>,
    st_scroll: usize,
    act_scroll: usize,
    wf_scroll: usize,
    wf_res: WfRes,

    /// Independent decoders running across the span in `Mode::Auto`.
    auto: Vec<AutoSlot>,

    /// Station identity for spot reporting; empty when unconfigured.
    my_call: String,
    my_grid: String,
    reporter: Option<Reporter>,
    /// Log messages from the reporter thread.
    rlog: Receiver<String>,
    rlog_tx: SyncSender<String>,
    /// Station settings dialog state (callsign + grid), when open.
    settings: Option<SettingsEdit>,

    log: VecDeque<String>,
    scan: Option<ScanState>,
    show_help: bool,
    squelch: bool,
    squelch_db: f32,
    cursor_snr: f32,

    /// Recent radio IQ used by the span scout and the signature classifier.
    scout_iq: Vec<Complex32>,
    /// Confirmed PSK31 signals in the current span (offsets from centre).
    psk_hits: Vec<PskHit>,
    /// Confirmed CW tones in the current span (offsets from centre).
    cw_hits: Vec<CwHit>,
    /// Occupied slices labelled CW / PSK / SSB / … from the last classify.
    idents: Vec<identify::Ident>,
    /// Held, fading spectrum labels. `idents` is the visible snapshot of these.
    tracks: Vec<LabelTrack>,
    /// Detections that linger after the live label fades, for the activity strip.
    heard: Vec<Heard>,
    /// Status / detection lines shown in the bottom activity window.
    notes: VecDeque<Note>,
    /// How many activity-log lines to step back from live.
    note_scroll: usize,
    scout_at: Instant,
    ident_at: Instant,
    /// Set by a finished PSK31 band scan so the UI loop can retune.
    pending_tune: Option<f64>,
    /// Slow hardware-gain nudge from the software AGC supervisor.
    pending_gain: Option<f64>,
}

/// How long a chip stays fully readable after the last classify hit.
const LABEL_HOLD: Duration = Duration::from_millis(5000);
/// Rise time of a new chip, seconds.
const LABEL_FADE_IN: f32 = 0.28;
/// Fall time after the hold expires, seconds.
const LABEL_FADE_OUT: f32 = 1.6;

/// One spectrum label with hold / fade so a single missed classify
/// cannot blink the chip off.
struct LabelTrack {
    offset_hz: f32,
    bw_hz: f32,
    snr_db: f32,
    kind: identify::Kind,
    score: f32,
    first: Instant,
    last_seen: Instant,
    pending_kind: identify::Kind,
    pending_hits: u8,
}

impl LabelTrack {
    fn from_ident(id: &identify::Ident, now: Instant) -> Self {
        Self {
            offset_hz: id.offset_hz,
            bw_hz: id.bw_hz,
            snr_db: id.snr_db,
            kind: id.kind,
            score: id.score,
            first: now,
            last_seen: now,
            pending_kind: id.kind,
            pending_hits: 0,
        }
    }

    fn ident(&self) -> identify::Ident {
        identify::Ident {
            offset_hz: self.offset_hz,
            bw_hz: self.bw_hz,
            snr_db: self.snr_db,
            kind: self.kind,
            score: self.score,
        }
    }

    fn alpha(&self, now: Instant) -> f32 {
        let age = now.saturating_duration_since(self.first).as_secs_f32();
        let since = now.saturating_duration_since(self.last_seen).as_secs_f32();
        let hold = LABEL_HOLD.as_secs_f32();
        // Appear nearly solid so a first-frame chip is readable; the
        // fade that matters is on the way out.
        let fade_in = 0.78 + 0.22 * (age / LABEL_FADE_IN).clamp(0.0, 1.0);
        let fade_out = if since <= hold {
            1.0
        } else {
            (1.0 - (since - hold) / LABEL_FADE_OUT).clamp(0.0, 1.0)
        };
        fade_in * fade_out
    }
}

/// A signal we have labelled in this span. Lives on after the live
/// spectrum chip drops, so the bottom strip can still name the frequency.
struct Heard {
    freq_hz: f64,
    freq_lo: f64,
    freq_hi: f64,
    kind: identify::Kind,
    snr_db: f32,
    count: u32,
    last: Instant,
}

struct Note {
    text: String,
    kind: Option<identify::Kind>,
}

/// Station settings dialog state.
struct SettingsEdit {
    call: String,
    grid: String,
    field: usize, // 0 = callsign, 1 = grid locator
}

impl App {
    fn new(center: f64, rate: f64, mode: Mode) -> Self {
        let (rlog_tx, rlog) = sync_channel(64);
        let mut app = Self {
            center,
            rate,
            cursor: 0.0,
            zoom: 1.0,
            gain: 36.0,
            agc: AgcMode::Soft,
            biast: false,
            smooth_idx: 1, // medium
            rx_filter: RxFilter::Auto,
            soft_agc: SoftAgc::new(mode.audio_rate()),
            hw_trim_at: Instant::now(),
            hw_hot: 0,
            hw_quiet: 0,
            band_idx: 0,
            band_gains: vec![36.0; bands::BANDS.len()],
            spectrum: Vec::new(),
            noise_tracker: dsp::NoiseFloor::new(),
            noise_floor: Vec::new(),
            spec_work: Vec::new(),
            smoothed: Vec::new(),
            waterfall: VecDeque::new(),
            wf_accum: Vec::new(),
            wf_last: Instant::now(),
            wf_idx: 2, // 500 ms per row
            fft_idx: 3, // 8192
            floor_db: -90.0,
            ceil_db: -20.0,
            mode: Mode::Off,
            decoder: None,
            chain: DecodeChain::new(rate, 400.0, Mode::Off.audio_rate()),
            text: String::new(),
            decode_log: VecDeque::new(),
            front: dsp::FrontEnd::new(rate),
            iq_buf: Vec::new(),
            rows: Vec::new(),
            auto_view: AutoView::Rows,
            ft_msgs: VecDeque::new(),
            stations: Vec::new(),
            decode_zoom: 0,
            msg_scroll: 0,
            msg_rows: std::cell::Cell::new(0),
            st_scroll: 0,
            act_scroll: 0,
            wf_scroll: 0,
            wf_res: WfRes::Quad,
            auto: Vec::new(),
            my_call: String::new(),
            my_grid: String::new(),
            reporter: None,
            rlog,
            rlog_tx,
            settings: None,
            log: VecDeque::new(),
            scan: None,
            show_help: false,
            squelch: true,
            squelch_db: 12.0,
            cursor_snr: 0.0,
            scout_iq: Vec::new(),
            psk_hits: Vec::new(),
            cw_hits: Vec::new(),
            idents: Vec::new(),
            tracks: Vec::new(),
            heard: Vec::new(),
            notes: VecDeque::new(),
            note_scroll: 0,
            scout_at: Instant::now(),
            ident_at: Instant::now(),
            pending_tune: None,
            pending_gain: None,
        };
        app.set_mode(mode);
        app
    }

    fn set_mode(&mut self, mode: Mode) {
        self.mode = mode;
        // Each mode wants its own audio rate, so the chain is rebuilt.
        self.chain = DecodeChain::new(self.rate, 3000.0, mode.audio_rate());
        self.decoder = mode.make(self.chain.fs_out());
        // Auto rebuilds its fleet from the next classify pass.
        self.auto.clear();
        self.soft_agc = SoftAgc::new(mode.audio_rate());
        self.apply_rx_filter();
        self.ft_msgs.clear();
        self.stations.clear();
        self.decode_log.clear();
        self.rows.clear();
        self.msg_scroll = 0;
        self.st_scroll = 0;
        self.act_scroll = 0;
        self.psk_hits.clear();
        self.cw_hits.clear();
        self.idents.clear();
        self.tracks.clear();
        self.scout_iq.clear();
        self.apply_station();
        self.log(format!("decoder: {}", mode.label()));
    }

    fn apply_station(&mut self) {
        if let Some(d) = &mut self.decoder {
            d.set_station(&self.my_call, &self.my_grid);
        }
    }

    fn mode_bandwidth(&self) -> f32 {
        self.decoder
            .as_ref()
            .map(|d| d.bandwidth())
            .unwrap_or(400.0)
    }

    fn rx_bandwidth(&self) -> f32 {
        self.rx_filter.hz(self.mode_bandwidth())
    }

    fn apply_rx_filter(&mut self) {
        self.chain.set_bandwidth(self.rx_bandwidth());
    }

    /// Keep the ADC in a comfortable range without touching the spectrum
    /// every block. Only runs in software-AGC mode, at most every 1.5 s.
    fn supervise_hw_gain(&mut self, block: &[Complex32]) {
        let mut peak = 0.0f32;
        for s in block {
            let m = s.re.abs().max(s.im.abs());
            if m > peak {
                peak = m;
            }
        }
        if peak > 0.85 {
            self.hw_hot = self.hw_hot.saturating_add(1);
            self.hw_quiet = 0;
        } else if peak < 0.04 {
            self.hw_quiet = self.hw_quiet.saturating_add(1);
            self.hw_hot = 0;
        } else {
            self.hw_hot = 0;
            self.hw_quiet = 0;
        }
        if self.hw_trim_at.elapsed() < Duration::from_millis(1500) {
            return;
        }
        if self.hw_hot >= 4 && self.gain > 2.0 {
            self.gain = (self.gain - 3.0).max(0.0);
            self.pending_gain = Some(self.gain);
            self.hw_trim_at = Instant::now();
            self.hw_hot = 0;
            self.log(format!("AGC: hardware gain {:+.0} dB (hot ADC)", self.gain));
        } else if self.hw_quiet >= 12 && self.gain < 46.0 {
            self.gain = (self.gain + 2.0).min(48.0);
            self.pending_gain = Some(self.gain);
            self.hw_trim_at = Instant::now();
            self.hw_quiet = 0;
            self.log(format!("AGC: hardware gain {:+.0} dB (quiet ADC)", self.gain));
        }
    }

    fn fft_size(&self) -> usize {
        FFT_SIZES[self.fft_idx]
    }

    /// Frequency resolution of one FFT bin.
    fn bin_hz(&self) -> f64 {
        self.rate / self.fft_size() as f64
    }

    fn wf_interval(&self) -> Duration {
        Duration::from_millis(WF_INTERVALS_MS[self.wf_idx])
    }

    /// How many lines of copy the decode pane could show, at most.
    ///
    /// Auto mode's log is one line per entry; the flat transcript is measured
    /// unwrapped, which under-counts on a narrow pane but is still a real
    /// bound — the render clamps the offset to the wrapped length anyway.
    fn transcript_len(&self) -> usize {
        match self.mode {
            Mode::Auto if self.auto_view == AutoView::Rows => self.rows.len(),
            Mode::Auto => self.decode_log.len(),
            // In FT modes the same offset scrolls the message table.
            Mode::Ft8 | Mode::Ft4 => self.ft_msgs.len(),
            _ => self.text.lines().count(),
        }
    }

    /// Move the decode pane's viewport `up` lines up the screen (negative to
    /// go down), clamped to the content outside it.
    ///
    /// The two auto faces anchor at opposite ends — the log at its newest
    /// line, the roster at its top row — so the stored offset counts
    /// backwards from live in one and forwards from the top in the other.
    /// Taking a screen direction rather than a raw offset delta keeps that a
    /// rendering detail: up is up in both, for the keys and the wheel alike.
    ///
    /// Clamping matters as much as direction: an offset allowed to run past
    /// the end costs a dozen dead keypresses to unwind and puts a nonsense
    /// count in the title.
    fn scroll_transcript(&mut self, up: isize) {
        let rows_view = self.mode == Mode::Auto && self.auto_view == AutoView::Rows;
        let delta = if rows_view { -up } else { up };
        let max = self.transcript_len().saturating_sub(self.msg_rows.get());
        let next = self.msg_scroll as isize + delta;
        self.msg_scroll = next.clamp(0, max as isize) as usize;
    }

    /// The held row for a signal, created on first copy.
    ///
    /// Matched the same way the auto slots themselves are matched, so a
    /// station drifting a little stays on its own row instead of sprouting a
    /// new one every time the classifier re-reports it a few hertz over.
    fn row_for(&mut self, dial_hz: f64, kind: identify::Kind, mode: &'static str) -> usize {
        let slack = match kind {
            identify::Kind::Rtty => 300.0,
            identify::Kind::Ft8 | identify::Kind::Ft4 => 2000.0,
            _ => 120.0,
        };
        if let Some(i) = self
            .rows
            .iter()
            .position(|r| r.kind == kind && (r.dial_hz - dial_hz).abs() < slack)
        {
            // Follow the signal as it drifts, so the frequency shown is
            // where it is now rather than where it was first heard.
            self.rows[i].dial_hz = dial_hz;
            return i;
        }
        self.rows.push(SignalRow {
            dial_hz,
            kind,
            mode,
            copy: String::new(),
            last_copy: Instant::now(),
        });
        self.rows.len() - 1
    }

    /// Forget rows that have been silent too long, and order the rest: live
    /// signals first by frequency, then the silent ones most-recent first, so
    /// a station that stops sending sinks down the pane rather than vanishing
    /// out from under whoever was reading it.
    fn sort_rows(&mut self) {
        let now = Instant::now();
        self.rows
            .retain(|r| now.duration_since(r.last_copy) < ROW_RETIRE);
        self.rows.sort_by(|a, b| {
            match b.live(now).cmp(&a.live(now)) {
                std::cmp::Ordering::Equal => {}
                other => return other,
            }
            if a.live(now) {
                a.dial_hz
                    .partial_cmp(&b.dial_hz)
                    .unwrap_or(std::cmp::Ordering::Equal)
            } else {
                b.last_copy.cmp(&a.last_copy)
            }
        });
        // Over the cap, the longest-silent rows are the ones to lose — they
        // are already at the bottom after the sort.
        self.rows.truncate(MAX_ROWS);
    }

    /// Put a remark of the scanner's own — a scan summary, a state change —
    /// into the auto transcript, so it reads in sequence with the copy around
    /// it rather than vanishing when auto mode stops showing `text`.
    fn push_decode_note(&mut self, text: String) {
        self.decode_log.push_back(DecodeEntry {
            stamp: utc_stamp(),
            dial_hz: 0.0,
            kind: identify::Kind::Unknown,
            mode: "",
            text: sanitize(&text),
        });
        while self.decode_log.len() > DECODE_LOG_MAX {
            self.decode_log.pop_front();
        }
    }

    fn log(&mut self, msg: String) {
        self.log.push_back(msg.clone());
        while self.log.len() > 6 {
            self.log.pop_front();
        }
        self.push_note(msg, None);
    }

    fn push_note(&mut self, text: String, kind: Option<identify::Kind>) {
        self.notes.push_back(Note { text, kind });
        while self.notes.len() > 80 {
            self.notes.pop_front();
        }
        self.note_scroll = 0;
    }

    fn tuned_freq(&self) -> f64 {
        self.center + self.cursor
    }

    /// Run every automatic decoder over one IQ block and fold what they say
    /// into the transcript, each line tagged with the frequency it came from.
    fn feed_auto(&mut self, block: &[Complex32]) {
        let soft = self.agc == AgcMode::Soft;
        let stamp = utc_stamp();
        let mut lines: Vec<DecodeEntry> = Vec::new();
        // (message, dial, mode) — spotting needs `&self.reporter`, so it
        // cannot happen while `self.auto` is mutably borrowed.
        let mut spots: Vec<(FtMessage, f64, &'static str)> = Vec::new();
        // (dial, kind, mode, copy) for the held rows, applied after the loop
        // for the same borrow reason.
        let mut copies: Vec<(f64, identify::Kind, &'static str, String)> = Vec::new();

        for slot in &mut self.auto {
            slot.chain.process(block, &mut slot.audio);
            if soft && slot.decoder.wants_agc() {
                slot.agc.process(&mut slot.audio);
            }
            let text = slot.decoder.process(&slot.audio);
            let (dial_hz, kind, mode) = (slot.dial_hz, slot.kind, slot.decoder.name());
            let row = |text: String| DecodeEntry {
                stamp: stamp.clone(),
                dial_hz,
                kind,
                mode,
                text: sanitize(&text),
            };

            // Slot-based modes hand back whole lines already; the
            // character-at-a-time modes need collecting into readable ones.
            if matches!(slot.kind, identify::Kind::Ft8 | identify::Kind::Ft4) {
                for m in slot.decoder.take_messages() {
                    // The held row reads as a stream of recent traffic on
                    // this calling frequency; the log keeps the full detail.
                    copies.push((dial_hz, kind, mode, format!("  {}  ", m.text)));
                    lines.push(row(m.format()));
                    spots.push((m, slot.dial_hz, slot.decoder.name()));
                }
            } else {
                for m in slot.decoder.take_messages() {
                    spots.push((m, slot.dial_hz, slot.decoder.name()));
                }
                if text.is_empty() {
                    slot.quiet += slot.audio.len();
                } else {
                    // Held rows take characters the moment they arrive, so a
                    // slow CW station builds up copy in place instead of
                    // waiting on a full line to be flushed to the log.
                    copies.push((dial_hz, kind, mode, text.clone()));
                    slot.partial.push_str(&text);
                    slot.quiet = 0;
                }
                // Emit on a line break, once a line's worth has built up, or
                // after a pause — otherwise a station that stops mid-word
                // never appears at all.
                while let Some(i) = slot.partial.find('\n') {
                    let line: String = slot.partial.drain(..=i).collect();
                    let line = line.trim_end().to_string();
                    if !line.is_empty() {
                        lines.push(row(line));
                    }
                }
                let idle = slot.quiet as f64 >= AUTO_FLUSH_SECS * slot.chain.fs_out();
                if slot.partial.len() >= AUTO_LINE
                    || (idle && !slot.partial.trim().is_empty())
                {
                    let take = slot
                        .partial
                        .char_indices()
                        .nth(AUTO_LINE * 2)
                        .map_or(slot.partial.len(), |(i, _)| i);
                    let line: String = slot.partial.drain(..take).collect();
                    lines.push(row(line.trim_end().to_string()));
                }
            }
        }

        for (dial_hz, kind, mode, text) in copies {
            let text = sanitize(&text);
            let i = self.row_for(dial_hz, kind, mode);
            self.rows[i].push_copy(&text);
        }
        self.sort_rows();

        // A row of nothing but spaces survives sanitising but says nothing.
        let mut added = 0usize;
        for line in lines {
            if line.text.trim().is_empty() {
                continue;
            }
            self.decode_log.push_back(line);
            added += 1;
        }
        while self.decode_log.len() > DECODE_LOG_MAX {
            self.decode_log.pop_front();
        }
        // Scrolled-back readers stay on the line they were reading as new
        // copy arrives underneath them, instead of being dragged along.
        if self.msg_scroll > 0 && added > 0 {
            self.scroll_transcript(added as isize);
        }
        if let Some(r) = &self.reporter {
            for (m, dial, mode) in &spots {
                r.spot(m, *dial, mode);
            }
        }
        for (mut m, _, _) in spots {
            // PSK31 spots reach this table too, and their text comes straight
            // off a demodulator that may have been listening to noise.
            m.text = sanitize(&m.text);
            self.update_stations(&m);
            self.ft_msgs.push_back(m);
        }
        while self.ft_msgs.len() > 600 {
            self.ft_msgs.pop_front();
        }
    }

    /// Point the automatic decoders at what is actually on the band.
    ///
    /// FT8 and FT4 are pinned to their calling frequencies rather than to
    /// anything the classifier found: they always live there, a pile-up is
    /// hard to localise from a periodogram, and a slot that stays put keeps
    /// decoding through the gaps between transmissions.
    fn reconcile_auto(&mut self) {
        if self.mode != Mode::Auto {
            if !self.auto.is_empty() {
                self.auto.clear();
            }
            return;
        }
        let now = Instant::now();
        let half = self.rate * 0.45;
        let mut wanted: Vec<(identify::Kind, f64, bool)> = Vec::new();

        for m in bands::MARKERS {
            let kind = match m.label {
                "FT8" => identify::Kind::Ft8,
                "FT4" => identify::Kind::Ft4,
                _ => continue,
            };
            // The decoder needs the whole 200–3000 Hz passband inside the span.
            if (m.freq - self.center).abs() < half - 3200.0 {
                wanted.push((kind, m.freq, true));
            }
        }

        // Narrowband modes go wherever the classifier saw them, strongest
        // first so a crowded span spends its slots on the best signals.
        let mut found: Vec<&identify::Ident> = self
            .idents
            .iter()
            .filter(|i| {
                matches!(
                    i.kind,
                    identify::Kind::Cw | identify::Kind::Rtty | identify::Kind::Psk31
                )
            })
            .collect();
        found.sort_by(|a, b| {
            b.snr_db
                .partial_cmp(&a.snr_db)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for id in found {
            wanted.push((id.kind, self.center + id.offset_hz as f64, false));
        }

        for (kind, dial, pinned) in wanted {
            if let Some(s) = self.auto.iter_mut().find(|s| s.same_signal(kind, dial)) {
                s.last_seen = now;
                continue;
            }
            if self.auto.len() >= MAX_AUTO_SLOTS {
                continue;
            }
            if let Some(slot) = AutoSlot::new(kind, dial, self.center, self.rate, pinned) {
                self.auto.push(slot);
            }
        }
        self.auto
            .retain(|s| s.pinned || s.last_seen.elapsed() < AUTO_IDLE);
    }

    /// Fold one decode into the station table. Stations keep the order they
    /// were first heard in — the list updates in place instead of reshuffling
    /// every slot, so a callsign stays put while you read it.
    fn update_stations(&mut self, m: &FtMessage) {
        let secs = stamp_secs(&m.stamp);
        let cq_msg = m.text.starts_with("CQ");
        let mut first_call = true;
        for tok in m.text.split_whitespace() {
            if !is_callsign(tok) {
                continue;
            }
            match self.stations.iter_mut().find(|(c, _)| c == tok) {
                Some((_, s)) => {
                    s.snr = s.snr.max(m.snr_db);
                    s.last_secs = secs;
                    s.freq = m.freq_hz;
                    s.count += 1;
                    if cq_msg && first_call {
                        s.cq = true;
                    }
                }
                None => self.stations.push((
                    tok.to_string(),
                    Station {
                        snr: m.snr_db,
                        last_secs: secs,
                        count: 1,
                        cq: cq_msg && first_call,
                        freq: m.freq_hz,
                    },
                )),
            }
            first_call = false;
        }
        // Bound the table: once it outgrows the pane, drop stations that have
        // been silent for over five minutes.
        if self.stations.len() > 40 {
            let now = utc_secs() as i64 % 86400;
            self.stations
                .retain(|(_, s)| (now - s.last_secs).rem_euclid(86400) <= 300);
        }
    }

    /// Visible slice of the span, as offsets from centre, clamped to the span.
    fn view_range(&self) -> (f64, f64) {
        let half_span = self.rate / 2.0;
        let half_view = (self.rate / self.zoom / 2.0).min(half_span);
        let mut lo = self.cursor - half_view;
        let mut hi = self.cursor + half_view;
        if lo < -half_span {
            hi += -half_span - lo;
            lo = -half_span;
        }
        if hi > half_span {
            lo -= hi - half_span;
            hi = half_span;
        }
        (lo.max(-half_span), hi.min(half_span))
    }

    /// Cursor step: 1/200th of what is on screen, so zooming in gives finer
    /// tuning and every keypress visibly moves the display.
    fn step_hz(&self) -> f64 {
        let (lo, hi) = self.view_range();
        ((hi - lo) / 200.0).max(1.0)
    }

    fn feed(&mut self, block: &[Complex32], spec: &mut Spectrum, out: &mut Vec<Complex32>) {
        // Front-end cleanup first, so nothing downstream — spectrum,
        // classifier, scouts, decoders — ever sees the receiver's own DC
        // offset or the image its IQ imbalance mirrors about the LO. Applied
        // here rather than in the radio thread so the tests go through it
        // too; the software AGC was invisible to every bench for exactly the
        // opposite reason.
        let mut buf = std::mem::take(&mut self.iq_buf);
        buf.clear();
        buf.extend_from_slice(block);
        self.front.process(&mut buf);
        let block: &[Complex32] = &buf;

        spec.power_db(block, &mut self.spectrum);
        self.noise_floor = self.noise_tracker.update(&self.spectrum).to_vec();
        smooth_bins(
            &self.spectrum,
            SMOOTH_BINS[self.smooth_idx],
            &mut self.spec_work,
        );
        let a = SMOOTH_TIME[self.smooth_idx];
        if self.smoothed.len() != self.spec_work.len() {
            self.smoothed = self.spec_work.clone();
        } else {
            for (s, v) in self.smoothed.iter_mut().zip(&self.spec_work) {
                *s = (1.0 - a) * *s + a * *v;
            }
        }

        // Accumulate the peak between waterfall rows instead of pushing a row
        // per block - otherwise the display scrolls far faster than it reads.
        if self.wf_accum.len() != self.smoothed.len() {
            self.wf_accum = self.smoothed.clone();
        } else {
            for (a, v) in self.wf_accum.iter_mut().zip(&self.smoothed) {
                *a = a.max(*v);
            }
        }
        if self.wf_last.elapsed() >= self.wf_interval() {
            self.waterfall.push_front(std::mem::take(&mut self.wf_accum));
            let cap = (WF_HISTORY_FLOATS / self.fft_size().max(1)).clamp(16, WF_MAX_ROWS);
            while self.waterfall.len() > cap {
                self.waterfall.pop_back();
            }
            self.wf_last = Instant::now();
        }

        // Auto-range the colour scale from the current noise floor.
        let mut sorted = self.smoothed.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if !sorted.is_empty() {
            let mut floors = self.noise_floor.clone();
            floors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let med = floors.get(floors.len() / 2).copied().unwrap_or(sorted[sorted.len() / 2]);
            let hi = sorted[sorted.len() * 999 / 1000];
            self.floor_db = 0.94 * self.floor_db + 0.06 * (med - 4.0);
            self.ceil_db = 0.90 * self.ceil_db + 0.10 * (hi + 8.0).max(med + 18.0);
        }

        // Always keep a rolling slice of radio IQ. The signature classifier
        // needs it in every mode; CW / PSK31 scouts reuse the same buffer.
        self.scout_iq.extend_from_slice(block);
        // PSK31 is 31.25 baud, so 0.6 s is nineteen symbols — too few to
        // confirm a weak signal is BPSK rather than noise, which is what kept
        // copyable 10 dB signals off the screen entirely. The scouts get a
        // longer look in the modes that run them.
        let secs = if self.mode == Mode::Cw
            || self.scan.as_ref().is_some_and(|s| s.kind == ScanKind::Cw)
        {
            0.85
        } else if matches!(self.mode, Mode::Psk31 | Mode::Auto) {
            1.60
        } else {
            0.60
        };
        let max = (self.rate * secs) as usize;
        if self.scout_iq.len() > max {
            let drain = self.scout_iq.len() - max;
            self.scout_iq.drain(..drain);
        }
        if self.scout_at.elapsed() >= Duration::from_millis(800) {
            match self.mode {
                Mode::Psk31 => refresh_psk_hits(self),
                Mode::Cw => refresh_cw_hits(self),
                // Auto needs both: the span classifier alone cannot see a
                // narrowband signal that is perfectly copyable (see
                // `scout_idents`), and auto mode is where that matters most.
                Mode::Auto => {
                    refresh_psk_hits(self);
                    refresh_cw_hits(self);
                }
                _ => {}
            }
            self.scout_at = Instant::now();
        }
        if self.ident_at.elapsed() >= Duration::from_millis(1200) {
            refresh_idents(self);
            self.ident_at = Instant::now();
        }

        // Signal-to-noise inside the decoder's passband, used for the squelch
        // and the status line. Once PSK31 is locked, measure a tight window
        // on the carrier rather than the whole search band.
        if let Some(d) = &self.decoder {
            let n = self.smoothed.len();
            if n > 0 {
                let bin_hz = self.rate / n as f64;
                let (centre_hz, half_hz) = if matches!(self.mode, Mode::Psk31 | Mode::Cw)
                    && d.locked()
                {
                    (self.cursor + d.lock_hz() as f64, 40.0)
                } else {
                    (self.cursor, self.rx_bandwidth() as f64 / 2.0)
                };
                let half = (half_hz / bin_hz).ceil().max(1.0) as isize;
                let centre = (n as f64 / 2.0 + centre_hz / bin_hz) as isize;
                let lo = (centre - half).clamp(0, n as isize - 1) as usize;
                let hi = (centre + half).clamp(0, n as isize - 1) as usize;
                let peak = self.smoothed[lo..=hi.max(lo)]
                    .iter()
                    .cloned()
                    .fold(f32::MIN, f32::max);
                let floor = self.noise_floor.get(lo..=hi.max(lo))
                    .and_then(|v| v.iter().copied().reduce(f32::max))
                    .unwrap_or(sorted[sorted.len() / 2]);
                self.cursor_snr = peak - floor;
            }
        }

        if self.agc == AgcMode::Soft {
            self.supervise_hw_gain(block);
        }

        if self.mode == Mode::Auto {
            self.feed_auto(block);
        }

        if self.decoder.is_some() {
            let (shift, gated, rides_agc) = self
                .decoder
                .as_ref()
                .map(|d| (d.offset_shift(), d.squelched(), d.wants_agc()))
                .unwrap_or((0.0, true, true));
            self.chain.set_offset(self.cursor + shift);
            self.chain.process(block, out);
            if self.agc == AgcMode::Soft && rides_agc {
                self.soft_agc.process(out);
            }
            // Feeding noise to a decoder just fills the pane with junk - but
            // slot-based modes must keep capturing regardless.
            let open = !self.squelch || !gated || self.cursor_snr >= self.squelch_db;
            if let Some(d) = &mut self.decoder {
                let new = if open { d.process(out) } else { String::new() };
                if !new.is_empty() {
                    // Never let raw demodulator bytes reach the terminal.
                    self.text.push_str(&sanitize_text(&new));
                    // Keep the transcript bounded, without splitting a
                    // character in half at the cut.
                    if self.text.len() > 8000 {
                        let cut = self.text.len() - 6000;
                        let cut = (cut..self.text.len())
                            .find(|i| self.text.is_char_boundary(*i))
                            .unwrap_or(self.text.len());
                        self.text = self.text[cut..].to_string();
                    }
                }
                let msgs = d.take_messages();
                if let Some(r) = &self.reporter {
                    let dial = self.tuned_freq();
                    for m in &msgs {
                        r.spot(m, dial, self.mode.label());
                    }
                }
                // Structured messages feed the FT traffic panes; PSK31
                // also emits them (for spotting) but they must not land
                // in the FT station/activity tables.
                if matches!(self.mode, Mode::Ft8 | Mode::Ft4) {
                    for m in msgs {
                        self.update_stations(&m);
                        self.ft_msgs.push_back(m);
                    }
                    while self.ft_msgs.len() > 600 {
                        self.ft_msgs.pop_front();
                    }
                }
            }
        }

        // Hand the scratch buffer back so the next block reuses it.
        self.iq_buf = buf;
    }
}

/// ~/.config/hfscan/config.toml (or $XDG_CONFIG_HOME/hfscan/config.toml).
fn config_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(base.join("hfscan").join("config.toml"))
}

/// The config file is two keys, `call` and `grid`; parse it by hand rather
/// than pulling in a TOML crate.
fn load_config() -> (Option<String>, Option<String>) {
    let mut call = None;
    let mut grid = None;
    if let Some(text) = config_path().and_then(|p| std::fs::read_to_string(p).ok()) {
        for line in text.lines() {
            let Some((k, v)) = line.split_once('=') else {
                continue;
            };
            let v = v.trim().trim_matches('"').to_string();
            match k.trim() {
                "call" if !v.is_empty() => call = Some(v),
                "grid" if !v.is_empty() => grid = Some(v),
                _ => {}
            }
        }
    }
    (call, grid)
}

fn save_config(call: &str, grid: &str) -> std::io::Result<()> {
    let Some(path) = config_path() else {
        return Err(std::io::Error::other("no config directory"));
    };
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&path, format!("call = \"{call}\"\ngrid = \"{grid}\"\n"))
}

fn parse_mode(s: &str) -> Mode {
    match s.to_ascii_lowercase().as_str() {
        "cw" => Mode::Cw,
        "rtty" => Mode::Rtty,
        "psk" | "psk31" => Mode::Psk31,
        "ft8" => Mode::Ft8,
        "ft4" => Mode::Ft4,
        "auto" | "all" => Mode::Auto,
        _ => Mode::Off,
    }
}

fn needs_exact_audio(mode: Mode) -> bool {
    matches!(mode, Mode::Ft8 | Mode::Ft4)
}

fn rate_ok_for_ft(rate: f64) -> bool {
    let d = rate / decoders::ft8::AUDIO_RATE;
    (d - d.round()).abs() < 1e-9 && d >= 1.0
}

fn main() -> Result<()> {
    let args = Args::parse();
    // Route SoapySDR's chatter into the `log` facade. With no logger installed
    // it is discarded, which keeps driver messages off the TUI.
    soapysdr::configure_logging();

    let mode = parse_mode(&args.mode);
    let mut rate = args.rate;
    if needs_exact_audio(mode) && !rate_ok_for_ft(rate) {
        rate = FT_SAFE_RATE;
    }

    let radio = radio::spawn(args.device.clone(), rate, args.freq)?;
    let mut app = App::new(args.freq, rate, mode);

    // Station identity: CLI flags win over the config file.
    let (file_call, file_grid) = load_config();
    let call = args.call.or(file_call).map(|c| c.to_uppercase());
    let grid = args.grid.or(file_grid).map(|g| g.to_uppercase());
    if let Some(c) = call {
        app.my_call = c.clone();
        app.my_grid = grid.clone().unwrap_or_default();
        app.reporter = Some(Reporter::start(
            c,
            grid.unwrap_or_default(),
            app.rlog_tx.clone(),
        ));
        app.apply_station();
        app.log(format!("de {} — spotting to pskreporter.info", app.my_call));
    } else {
        app.log("press o to set your callsign (enables pskreporter spotting)".into());
    }
    if rate != args.rate {
        app.log(format!("sample rate forced to {rate:.0} Hz for FT8/FT4"));
    }
    app.fft_idx = FFT_SIZES
        .iter()
        .position(|n| *n >= args.fft)
        .unwrap_or(FFT_SIZES.len() - 1);
    app.band_idx = bands::BANDS
        .iter()
        .position(|b| args.freq >= b.start && args.freq <= b.end)
        .unwrap_or(5);

    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout))?;

    let res = run_app(&mut terminal, &mut app, &radio, args.fft);

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    let _ = radio.cmd.send(radio::Cmd::Quit);
    res
}

fn run_app<B: Backend>(
    terminal: &mut Terminal<B>,
    app: &mut App,
    radio: &radio::Radio,
    fft_size: usize,
) -> Result<()> {
    let _ = fft_size;
    let mut spec = Spectrum::new(app.fft_size());
    let mut baseband: Vec<Complex32> = Vec::new();
    let mut last_draw = Instant::now();

    loop {
        if spec.size() != app.fft_size() {
            spec = Spectrum::new(app.fft_size());
            app.spectrum.clear();
            app.spec_work.clear();
            app.smoothed.clear();
            app.wf_accum.clear();
            app.waterfall.clear();
        }
        // Pull everything the radio has produced since the last pass.
        while let Ok(block) = radio.iq.try_recv() {
            app.feed(&block, &mut spec, &mut baseband);
        }
        while let Ok(msg) = radio.log.try_recv() {
            app.log(msg);
        }
        while let Ok(msg) = app.rlog.try_recv() {
            app.log(msg);
        }

        if let Some(done) = step_scan(app) {
            if done {
                app.scan = None;
            }
        }
        if let Some(f) = app.pending_tune.take() {
            retune(app, radio, f);
        }
        if let Some(g) = app.pending_gain.take() {
            let _ = radio.cmd.send(radio::Cmd::Gain(g));
        }

        if last_draw.elapsed() >= Duration::from_millis(50) {
            terminal.draw(|f| draw(f, app))?;
            last_draw = Instant::now();
        }

        if event::poll(Duration::from_millis(10))? {
            match event::read()? {
                Event::Key(k) => {
                if k.kind != KeyEventKind::Press {
                    continue;
                }
                // While the settings dialog is open it owns all keys.
                if app.settings.is_some() {
                    settings_key(app, k);
                    continue;
                }
                let shift = k.modifiers.contains(KeyModifiers::SHIFT);
                let step = app.step_hz();
                match k.code {
                    KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                    KeyCode::Char('?') => app.show_help = !app.show_help,
                    KeyCode::Char('o') => {
                        app.settings = Some(SettingsEdit {
                            call: app.my_call.clone(),
                            grid: app.my_grid.clone(),
                            field: 0,
                        });
                    }
                    KeyCode::Left => nudge_cursor(app, if shift { -step * 10.0 } else { -step }),
                    KeyCode::Right => nudge_cursor(app, if shift { step * 10.0 } else { step }),
                    KeyCode::Char('z') => app.zoom = (app.zoom * 2.0).min(512.0),
                    KeyCode::Char('Z') => app.zoom = (app.zoom / 2.0).max(1.0),
                    KeyCode::Char('n') => next_signal(app, true),
                    KeyCode::Char('N') => next_signal(app, false),
                    KeyCode::Char('p') => mode_scan_key(app, radio),
                    KeyCode::Char('f') => {
                        app.fft_idx = (app.fft_idx + 1) % FFT_SIZES.len();
                        let n = app.fft_size();
                        let hz = app.bin_hz();
                        app.log(format!("FFT {n} ({hz:.1} Hz/bin)"));
                    }
                    KeyCode::Char('F') => {
                        app.fft_idx = (app.fft_idx + FFT_SIZES.len() - 1) % FFT_SIZES.len();
                        let n = app.fft_size();
                        let hz = app.bin_hz();
                        app.log(format!("FFT {n} ({hz:.1} Hz/bin)"));
                    }
                    KeyCode::Char('w') => {
                        app.wf_idx = (app.wf_idx + 1) % WF_INTERVALS_MS.len();
                        let ms = WF_INTERVALS_MS[app.wf_idx];
                        app.log(format!("waterfall {ms} ms/row"));
                    }
                    KeyCode::Char('v') => {
                        app.decode_zoom = (app.decode_zoom + 1) % 3;
                        let size = ["normal", "large", "huge"][app.decode_zoom as usize];
                        app.log(format!("decode pane: {size}"));
                    }
                    KeyCode::Char('V') => {
                        // The two faces anchor at opposite ends, so an offset
                        // carried across would land somewhere arbitrary.
                        app.msg_scroll = 0;
                        app.auto_view = match app.auto_view {
                            AutoView::Rows => AutoView::Log,
                            AutoView::Log => AutoView::Rows,
                        };
                        let what = match app.auto_view {
                            AutoView::Rows => "held rows",
                            AutoView::Log => "chronological log",
                        };
                        app.log(format!("auto decode view: {what}"));
                    }
                    KeyCode::Char('[') => retune(app, radio, app.center - 10_000.0),
                    KeyCode::Char(']') => retune(app, radio, app.center + 10_000.0),
                    KeyCode::PageUp => retune(app, radio, app.center - app.rate / 2.0),
                    KeyCode::PageDown => retune(app, radio, app.center + app.rate / 2.0),
                    KeyCode::Char('c') => {
                        let f = app.tuned_freq();
                        app.cursor = 0.0;
                        retune(app, radio, f);
                    }
                    KeyCode::Char('b') => {
                        app.band_idx = (app.band_idx + 1) % bands::BANDS.len();
                        let f = bands::BANDS[app.band_idx].default;
                        app.cursor = 0.0;
                        retune(app, radio, f);
                    }
                    KeyCode::Char('B') => {
                        app.band_idx = (app.band_idx + bands::BANDS.len() - 1) % bands::BANDS.len();
                        let f = bands::BANDS[app.band_idx].default;
                        app.cursor = 0.0;
                        retune(app, radio, f);
                    }
                    KeyCode::Char('d') => {
                        let next = app.mode.next();
                        // FT8/FT4 need a radio rate that divides by 12 kHz.
                        if needs_exact_audio(next) && !rate_ok_for_ft(app.rate) {
                            app.rate = FT_SAFE_RATE;
                            let _ = radio.cmd.send(radio::Cmd::Rate(FT_SAFE_RATE));
                            // The front end's time constants are in samples,
                            // so its estimates belong to the old rate.
                            app.front = dsp::FrontEnd::new(FT_SAFE_RATE);
                            app.waterfall.clear();
                            app.log(format!("sample rate -> {FT_SAFE_RATE:.0} Hz for FT"));
                        }
                        app.set_mode(next);
                    }
                    KeyCode::Char('r') => {
                        if let Some(d) = &mut app.decoder {
                            d.toggle();
                        }
                    }
                    KeyCode::Char('u') if matches!(app.mode, Mode::Cw | Mode::Psk31) => {
                        lock_nudge(app, -2.0);
                    }
                    KeyCode::Char('i') if matches!(app.mode, Mode::Cw | Mode::Psk31) => {
                        lock_nudge(app, 2.0);
                    }
                    KeyCode::Char('g') if matches!(app.mode, Mode::Cw | Mode::Psk31) => {
                        centre_on_lock(app);
                    }
                    KeyCode::Char('x') => {
                        app.text.clear();
                        app.decode_log.clear();
                        app.rows.clear();
                        app.ft_msgs.clear();
                        app.stations.clear();
                        app.msg_scroll = 0;
                        app.st_scroll = 0;
                        app.act_scroll = 0;
                        if let Some(d) = &mut app.decoder {
                            d.reset();
                        }
                    }
                    KeyCode::Up => {
                        let n = if shift { 10 } else { 1 };
                        app.scroll_transcript(n as isize);
                    }
                    KeyCode::Down => {
                        let n = if shift { 10 } else { 1 };
                        app.scroll_transcript(-(n as isize));
                    }
                    KeyCode::Char('W') => {
                        app.wf_res = app.wf_res.next();
                        app.wf_scroll = 0;
                        let what = app.wf_res.label();
                        app.log(format!("waterfall: {what}"));
                    }
                    KeyCode::Char('a') => cycle_agc(app, radio),
                    KeyCode::Char('e') => {
                        app.smooth_idx = (app.smooth_idx + 1) % SMOOTH_LABELS.len();
                        app.log(format!(
                            "spectrum smooth: {}",
                            SMOOTH_LABELS[app.smooth_idx]
                        ));
                    }
                    KeyCode::Char('j') => {
                        let level = app.front.cycle_blanker();
                        app.log(format!("noise blanker: {level}"));
                    }
                    KeyCode::Char('l') => {
                        app.rx_filter = app.rx_filter.next();
                        app.apply_rx_filter();
                        app.log(format!(
                            "RX filter: {} ({:.0} Hz)",
                            app.rx_filter.label(),
                            app.rx_bandwidth()
                        ));
                    }
                    KeyCode::Char('t') => {
                        app.biast = !app.biast;
                        let _ = radio.cmd.send(radio::Cmd::BiasT(app.biast));
                    }
                    KeyCode::Char('+') | KeyCode::Char('=') => {
                        app.gain = (app.gain + 2.0).min(48.0);
                        apply_agc_mode(app, radio, AgcMode::Off);
                    }
                    KeyCode::Char('-') => {
                        app.gain = (app.gain - 2.0).max(0.0);
                        apply_agc_mode(app, radio, AgcMode::Off);
                    }
                    KeyCode::Char('k') => {
                        app.squelch = !app.squelch;
                        let msg = format!("squelch {}", if app.squelch { "on" } else { "off" });
                        app.log(msg);
                    }
                    KeyCode::Char(',') => app.squelch_db = (app.squelch_db - 1.0).max(0.0),
                    KeyCode::Char('.') => app.squelch_db = (app.squelch_db + 1.0).min(40.0),
                    KeyCode::Char('s') => {
                        if app.scan.is_some() {
                            app.scan = None;
                            app.log("scan cancelled".into());
                        } else {
                            start_scan(app, radio, ScanKind::Energy);
                        }
                    }
                    _ => {}
                }
                }
                Event::Mouse(m) => {
                    let delta: isize = match m.kind {
                        MouseEventKind::ScrollUp => -3,
                        MouseEventKind::ScrollDown => 3,
                        _ => continue,
                    };
                    let size = terminal.size()?;
                    let area = Rect::new(0, 0, size.width, size.height);
                    scroll_pane_at(app, area, m.column, m.row, delta);
                }
                _ => {}
            }
        }
    }
}

/// Key handling while the settings dialog is open.
fn settings_key(app: &mut App, k: KeyEvent) {
    let Some(ed) = &mut app.settings else {
        return;
    };
    match k.code {
        KeyCode::Esc => app.settings = None,
        KeyCode::Tab | KeyCode::Up | KeyCode::Down => ed.field = 1 - ed.field,
        KeyCode::Backspace => {
            let s = if ed.field == 0 { &mut ed.call } else { &mut ed.grid };
            s.pop();
        }
        KeyCode::Enter => {
            let ed = app.settings.take().unwrap();
            if !is_callsign(&ed.call) {
                app.log(format!("'{}' doesn't look like a callsign", ed.call));
                app.settings = Some(ed);
                return;
            }
            if !ed.grid.is_empty() && !report::is_grid(&ed.grid) {
                app.log(format!("'{}' doesn't look like a grid locator", ed.grid));
                app.settings = Some(ed);
                return;
            }
            match save_config(&ed.call, &ed.grid) {
                Ok(()) => app.log("settings saved to config file".into()),
                Err(e) => app.log(format!("could not save settings: {e}")),
            }
            app.my_call = ed.call.clone();
            app.my_grid = ed.grid.clone();
            app.apply_station();
            // Dropping the old handle disconnects the old reporter thread.
            app.reporter = Some(Reporter::start(ed.call, ed.grid, app.rlog_tx.clone()));
            app.log(format!("de {} — spotting to pskreporter.info", app.my_call));
        }
        KeyCode::Char(c) => {
            let c = c.to_ascii_uppercase();
            if ed.field == 0 {
                if (c.is_ascii_alphanumeric() || c == '/') && ed.call.len() < 12 {
                    ed.call.push(c);
                }
            } else if c.is_ascii_alphanumeric() && ed.grid.len() < 6 {
                ed.grid.push(c);
            }
        }
        _ => {}
    }
}

/// Wheel-scroll whichever pane is under the mouse: the FT sub-panes scroll
/// independently, the waterfall scrolls back through its history.
fn scroll_pane_at(app: &mut App, area: Rect, col: u16, row: u16, delta: isize) {
    let chunks = pane_rects(area, app);
    let inside = |r: Rect| {
        col >= r.x && col < r.x + r.width && row >= r.y && row < r.y + r.height
    };
    let adj = |v: &mut usize| *v = v.saturating_add_signed(delta);
    if inside(chunks[3]) {
        if matches!(app.mode, Mode::Ft8 | Mode::Ft4) {
            let cols = ft_cols(chunks[3]);
            if inside(cols[0]) {
                adj(&mut app.act_scroll);
            } else if inside(cols[1]) {
                // `delta` counts rows down the pane; the transcript takes a
                // direction, and wheel-up means older.
                app.scroll_transcript(-delta);
            } else {
                adj(&mut app.st_scroll);
            }
        } else if matches!(app.mode, Mode::Cw | Mode::Psk31) {
            let cols = cw_cols(chunks[3]);
            if inside(cols[1]) {
                // `delta` counts rows down the pane; the transcript takes a
                // direction, and wheel-up means older.
                app.scroll_transcript(-delta);
            } else if inside(cols[2]) {
                adj(&mut app.st_scroll);
            }
        } else {
            app.scroll_transcript(-delta);
        }
    } else if inside(chunks[2]) {
        adj(&mut app.wf_scroll);
    } else if inside(chunks[4]) {
        adj(&mut app.note_scroll);
        let max = app.notes.len().saturating_sub(1);
        if app.note_scroll > max {
            app.note_scroll = max;
        }
    }
}

fn nudge_cursor(app: &mut App, delta: f64) {
    let limit = app.rate * 0.45;
    app.cursor = (app.cursor + delta).clamp(-limit, limit);
}

/// Jump the cursor to the next detected signal, so a busy band can be walked
/// without hunting for peaks by eye. In PSK31 mode this hops confirmed
/// PSK31 carriers, not raw energy peaks.
fn next_signal(app: &mut App, forward: bool) {
    if app.mode == Mode::Psk31 {
        next_psk31(app, forward);
        return;
    }
    if app.mode == Mode::Cw {
        next_cw(app, forward);
        return;
    }
    let mut peaks = find_peaks(&app.smoothed, &app.noise_floor, 0.0, app.rate);
    if peaks.is_empty() {
        app.log("no signals above the noise floor".into());
        return;
    }
    peaks.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    // Ignore anything essentially under the cursor already.
    let guard = (app.rate / 2000.0).max(50.0);
    let target = if forward {
        peaks.iter().find(|(f, _)| *f > app.cursor + guard)
    } else {
        peaks.iter().rev().find(|(f, _)| *f < app.cursor - guard)
    };
    match target {
        Some((f, snr)) => {
            let (f, snr) = (*f, *snr);
            app.cursor = f.clamp(-app.rate * 0.45, app.rate * 0.45);
            let msg = format!(
                "-> {:.3} kHz ({:.0} dB)",
                (app.center + app.cursor) / 1000.0,
                snr
            );
            app.log(msg);
        }
        None => app.log("no further signals in this direction".into()),
    }
}

fn refresh_psk_hits(app: &mut App) {
    let mut peaks = scout_peaks(&app.smoothed, &app.noise_floor, 0.0, app.rate);
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    peaks.truncate(24);
    app.psk_hits = psk31::scan_span(&app.scout_iq, app.rate, &peaks);
}

fn refresh_cw_hits(app: &mut App) {
    let mut peaks = scout_peaks(&app.smoothed, &app.noise_floor, 0.0, app.rate);
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    peaks.truncate(24);
    app.cw_hits = cw::scan_span(&app.scout_iq, app.rate, &peaks);
}

/// Turn what the narrowband scouts found into idents the auto fleet can use.
///
/// The span classifier only ever looks at signals the occupancy detector
/// hands it, and that detector needs a peak 8 dB above the median *in a
/// coarse FFT bin*. A PSK31 signal is 31 Hz wide in a 47 Hz bin, so one that
/// is a perfectly copyable 10 dB in its own bandwidth reads as about 8 dB
/// there and is never even offered for classification — measured: the whole
/// auto path worked at 15 dB and saw nothing at 10, while the decoder itself
/// copies at 10 dB and `scan_span` confirms it with quality 0.89.
///
/// The mode-specific scouts do not have that problem, because they mix each
/// candidate down to baseband and match it there rather than reading a
/// bin. They were already running — just not in the mode that needed them.
fn scout_idents(app: &App) -> Vec<identify::Ident> {
    if app.mode != Mode::Auto || app.smoothed.is_empty() {
        return Vec::new();
    }
    let n = app.smoothed.len();
    let bin = app.rate as f32 / n as f32;
    let mut sorted = app.smoothed.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = sorted[n / 2];
    // The scouts report an offset, not a level; read the level back off the
    // spectrum so a crowded span still spends its slots strongest-first.
    let snr_at = |off_hz: f32| {
        let i = ((off_hz / bin) + n as f32 / 2.0).round();
        let i = (i.clamp(0.0, n as f32 - 1.0)) as usize;
        app.smoothed[i] - med
    };
    let mut out = Vec::new();
    for h in &app.psk_hits {
        if h.quality < 0.75 {
            continue;
        }
        out.push(identify::Ident {
            offset_hz: h.offset_hz,
            bw_hz: 62.0,
            snr_db: snr_at(h.offset_hz),
            kind: identify::Kind::Psk31,
            score: h.quality,
        });
    }
    for h in &app.cw_hits {
        if h.quality < 0.55 {
            continue;
        }
        out.push(identify::Ident {
            offset_hz: h.offset_hz,
            bw_hz: 150.0,
            snr_db: snr_at(h.offset_hz),
            kind: identify::Kind::Cw,
            score: h.quality,
        });
    }
    out
}

fn refresh_idents(app: &mut App) {
    if app.smoothed.is_empty() || app.scout_iq.len() < (app.rate * 0.25) as usize {
        return;
    }
    let mut raw = identify::classify_span(&app.scout_iq, app.rate, &app.smoothed, app.center);
    // The classifier wins where it has an opinion; the scouts only add what
    // it never saw, so this cannot override a considered classification.
    for s in scout_idents(app) {
        let known = raw
            .iter()
            .any(|r| (r.offset_hz - s.offset_hz).abs() < 120.0);
        if !known {
            raw.push(s);
        }
    }
    apply_idents(app, raw);
    merge_heard(app);
    let before = app.auto.len();
    app.reconcile_auto();
    if app.auto.len() != before {
        let n = app.auto.len();
        app.log(format!("auto: {n} decoder(s) running"));
    }
}

/// Merge a classify pass into held tracks: smooth position, keep the chip
/// up through a few misses, and only switch kind after a second vote.
fn apply_idents(app: &mut App, raw: Vec<identify::Ident>) {
    let now = Instant::now();
    for id in raw {
        if id.kind == identify::Kind::Unknown {
            continue;
        }
        if let Some(i) = match_track(&app.tracks, &id) {
            let t = &mut app.tracks[i];
            if matches!(t.kind, identify::Kind::Ft8 | identify::Kind::Ft4)
                && t.kind == id.kind
            {
                let lo = (t.offset_hz - t.bw_hz * 0.5).min(id.offset_hz - id.bw_hz * 0.5);
                let hi = (t.offset_hz + t.bw_hz * 0.5).max(id.offset_hz + id.bw_hz * 0.5);
                t.offset_hz = (lo + hi) * 0.5;
                t.bw_hz = (hi - lo).max(80.0);
            } else {
                t.offset_hz = 0.6 * t.offset_hz + 0.4 * id.offset_hz;
                t.bw_hz = 0.7 * t.bw_hz + 0.3 * id.bw_hz;
            }
            t.snr_db = 0.6 * t.snr_db + 0.4 * id.snr_db;
            t.score = t.score.max(id.score);
            t.last_seen = now;
            if id.kind == t.kind {
                t.pending_hits = 0;
                t.pending_kind = t.kind;
            } else if id.kind == t.pending_kind {
                t.pending_hits = t.pending_hits.saturating_add(1);
                if t.pending_hits >= 2 {
                    t.kind = id.kind;
                    t.pending_hits = 0;
                }
            } else {
                t.pending_kind = id.kind;
                t.pending_hits = 1;
            }
        } else {
            app.tracks.push(LabelTrack::from_ident(&id, now));
        }
    }
    let hold_out = LABEL_HOLD + Duration::from_secs_f32(LABEL_FADE_OUT);
    app.tracks.retain(|t| {
        let abs = app.center + t.offset_hz as f64;
        let ham = matches!(
            t.kind,
            identify::Kind::Cw
                | identify::Kind::Psk31
                | identify::Kind::Rtty
                | identify::Kind::Ft8
                | identify::Kind::Ft4
        );
        !(ham && !bands::in_amateur(abs)) && t.last_seen.elapsed() < hold_out
    });
    app.idents = app
        .tracks
        .iter()
        .filter(|t| t.alpha(now) >= 0.12)
        .map(|t| t.ident())
        .collect();
}

fn match_track(tracks: &[LabelTrack], id: &identify::Ident) -> Option<usize> {
    tracks
        .iter()
        .enumerate()
        .filter(|(_, t)| {
            let ft = matches!(
                (t.kind, id.kind),
                (identify::Kind::Ft8, identify::Kind::Ft8)
                    | (identify::Kind::Ft4, identify::Kind::Ft4)
            );
            let slack = if ft {
                t.bw_hz.max(id.bw_hz) * 0.5 + 2500.0
            } else {
                t.bw_hz.max(id.bw_hz).max(120.0) * 0.7
            };
            (t.offset_hz - id.offset_hz).abs() < slack
        })
        .min_by(|(_, a), (_, b)| {
            let da = (a.offset_hz - id.offset_hz).abs()
                + if a.kind == id.kind { 0.0 } else { 800.0 };
            let db = (b.offset_hz - id.offset_hz).abs()
                + if b.kind == id.kind { 0.0 } else { 800.0 };
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
}

/// Keep a per-span memory of labelled signals so the activity strip can
/// name a frequency after the live spectrum chip has gone.
fn merge_heard(app: &mut App) {
    let now = Instant::now();
    let mut fresh: Vec<(f64, identify::Kind, f32)> = Vec::new();
    for id in &app.idents {
        if id.kind == identify::Kind::Unknown {
            continue;
        }
        let freq = app.center + id.offset_hz as f64;
        let half = id.bw_hz as f64 * 0.5;
        let ft = matches!(id.kind, identify::Kind::Ft8 | identify::Kind::Ft4);
        let slack = if ft {
            3500.0
        } else {
            id.bw_hz.max(80.0) as f64 * 0.6
        };
        if let Some(h) = app
            .heard
            .iter_mut()
            .find(|h| h.kind == id.kind && (h.freq_hz - freq).abs() < slack)
        {
            h.freq_lo = h.freq_lo.min(freq - half);
            h.freq_hi = h.freq_hi.max(freq + half);
            h.freq_hz = 0.5 * (h.freq_lo + h.freq_hi);
            h.snr_db = h.snr_db.max(id.snr_db);
            h.count = h.count.saturating_add(1).max(1);
            h.last = now;
        } else {
            app.heard.push(Heard {
                freq_hz: freq,
                freq_lo: freq - half,
                freq_hi: freq + half,
                kind: id.kind,
                snr_db: id.snr_db,
                count: 1,
                last: now,
            });
            fresh.push((freq, id.kind, id.snr_db));
        }
    }
    for (freq, kind, snr) in fresh {
        let text = if matches!(kind, identify::Kind::Ft8 | identify::Kind::Ft4) {
            format!("{:.3} kHz  {}  pile-up", freq / 1000.0, kind.label())
        } else {
            format!("{:.3} kHz  {}  {:+.0} dB", freq / 1000.0, kind.label(), snr)
        };
        app.push_note(text, Some(kind));
    }
    app.heard.retain(|h| {
        let ham = matches!(
            h.kind,
            identify::Kind::Cw
                | identify::Kind::Psk31
                | identify::Kind::Rtty
                | identify::Kind::Ft8
                | identify::Kind::Ft4
        );
        !(ham && !bands::in_amateur(h.freq_hz)) && h.last.elapsed() < Duration::from_secs(180)
    });
    app.heard.sort_by(|a, b| {
        a.freq_hz
            .partial_cmp(&b.freq_hz)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

fn cycle_agc(app: &mut App, radio: &radio::Radio) {
    apply_agc_mode(app, radio, app.agc.next());
}

fn apply_agc_mode(app: &mut App, radio: &radio::Radio, mode: AgcMode) {
    app.agc = mode;
    match mode {
        AgcMode::Soft => {
            let _ = radio.cmd.send(radio::Cmd::Agc(false));
            let _ = radio.cmd.send(radio::Cmd::Gain(app.gain));
            app.soft_agc.reset();
            app.log(format!("AGC soft (hang)  hw {:+.0} dB", app.gain));
        }
        AgcMode::Hardware => {
            let _ = radio.cmd.send(radio::Cmd::Agc(true));
            app.log("AGC hardware (fast — can pump)".into());
        }
        AgcMode::Off => {
            let _ = radio.cmd.send(radio::Cmd::Agc(false));
            let _ = radio.cmd.send(radio::Cmd::Gain(app.gain));
            app.log(format!("AGC off  gain {:+.0} dB", app.gain));
        }
    }
}

fn hop_cursor_to(app: &mut App, offset_hz: f64) {
    if let Some(d) = &mut app.decoder {
        d.hop();
    }
    app.cursor = offset_hz.clamp(-app.rate * 0.45, app.rate * 0.45);
}

/// Next / previous confirmed PSK31. Prefers another signal already inside
/// the decoder's passband (no retune); otherwise scouts the visible span
/// and moves the cursor onto it.
fn next_psk31(app: &mut App, forward: bool) {
    if let Some(d) = &mut app.decoder
        && let Some(hz) = d.next_lock(forward)
    {
        app.log(format!("PSK31 lock {hz:+.1} Hz (in passband)"));
        return;
    }
    // Don't wipe a band-scan list; only scout the current span if we
    // don't already have confirmed hits to walk.
    if app.psk_hits.is_empty() {
        refresh_psk_hits(app);
    }
    if app.psk_hits.is_empty() {
        if app.scout_iq.len() < (app.rate * 0.25) as usize {
            app.log("PSK31 scout: collecting audio…".into());
        } else {
            app.log("no PSK31 in this span — press p to scan the band".into());
        }
        return;
    }
    let cur = app.cursor
        + app
            .decoder
            .as_ref()
            .map(|d| d.lock_hz() as f64)
            .unwrap_or(0.0);
    let guard = 40.0;
    let target = if forward {
        app.psk_hits
            .iter()
            .find(|h| h.offset_hz as f64 > cur + guard)
            .or_else(|| app.psk_hits.first())
    } else {
        app.psk_hits
            .iter()
            .rev()
            .find(|h| (h.offset_hz as f64) < cur - guard)
            .or_else(|| app.psk_hits.last())
    };
    match target {
        Some(h) => {
            let hz = h.offset_hz as f64;
            let abs = app.center + hz;
            if hz.abs() > app.rate * 0.40 {
                if let Some(d) = &mut app.decoder {
                    d.hop();
                }
                app.cursor = 0.0;
                app.pending_tune = Some(abs);
                // Re-base stored hits onto the new centre.
                for hit in &mut app.psk_hits {
                    hit.offset_hz = (app.center + hit.offset_hz as f64 - abs) as f32;
                }
            } else {
                hop_cursor_to(app, hz);
            }
            app.log(format!(
                "PSK31 -> {:.3} kHz  ({} found)",
                abs / 1000.0,
                app.psk_hits.len()
            ));
        }
        None => app.log("no further PSK31 in this span".into()),
    }
}

fn lock_nudge(app: &mut App, delta_hz: f32) {
    if let Some(d) = &mut app.decoder
        && let Some(hz) = d.nudge_lock(delta_hz)
    {
        app.log(format!("{} tune {hz:+.1} Hz", app.mode.label()));
    }
}

/// Move the cursor onto the locked tone so the filter is centred and the
/// residual reads near zero — the manual "zero-beat".
fn centre_on_lock(app: &mut App) {
    let Some(d) = app.decoder.as_ref() else {
        return;
    };
    let off = d.lock_hz() as f64;
    if off.abs() < 0.5 {
        app.log(format!("{} already centred", app.mode.label()));
        return;
    }
    hop_cursor_to(app, app.cursor + off);
    app.log(format!(
        "{} centred {:.3} kHz",
        app.mode.label(),
        app.tuned_freq() / 1000.0
    ));
}

/// Next / previous confirmed CW. Same hop rules as PSK31.
fn next_cw(app: &mut App, forward: bool) {
    if let Some(d) = &mut app.decoder
        && let Some(hz) = d.next_lock(forward)
    {
        app.log(format!("CW lock {hz:+.1} Hz (in passband)"));
        return;
    }
    if app.cw_hits.is_empty() {
        refresh_cw_hits(app);
    }
    if app.cw_hits.is_empty() {
        if app.scout_iq.len() < (app.rate * 0.35) as usize {
            app.log("CW scout: collecting audio…".into());
        } else {
            app.log("no CW in this span — press p to scan the band".into());
        }
        return;
    }
    let cur = app.cursor
        + app
            .decoder
            .as_ref()
            .map(|d| d.lock_hz() as f64)
            .unwrap_or(0.0);
    let guard = 40.0;
    let target = if forward {
        app.cw_hits
            .iter()
            .find(|h| h.offset_hz as f64 > cur + guard)
            .or_else(|| app.cw_hits.first())
    } else {
        app.cw_hits
            .iter()
            .rev()
            .find(|h| (h.offset_hz as f64) < cur - guard)
            .or_else(|| app.cw_hits.last())
    };
    match target {
        Some(h) => {
            let hz = h.offset_hz as f64;
            let abs = app.center + hz;
            if hz.abs() > app.rate * 0.40 {
                if let Some(d) = &mut app.decoder {
                    d.hop();
                }
                app.cursor = 0.0;
                app.pending_tune = Some(abs);
                for hit in &mut app.cw_hits {
                    hit.offset_hz = (app.center + hit.offset_hz as f64 - abs) as f32;
                }
            } else {
                hop_cursor_to(app, hz);
            }
            app.log(format!(
                "CW -> {:.3} kHz  ({} found)",
                abs / 1000.0,
                app.cw_hits.len()
            ));
        }
        None => app.log("no further CW in this span".into()),
    }
}

/// `p`: lock the next PSK31 or CW in the current span. If the scout sees
/// none, walk the band and lock the first one it confirms.
fn mode_scan_key(app: &mut App, radio: &radio::Radio) {
    let kind = match app.mode {
        Mode::Psk31 => ScanKind::Psk31,
        Mode::Cw => ScanKind::Cw,
        _ => {
            app.log("switch to CW or PSK31 first (d)".into());
            return;
        }
    };
    if app.scan.as_ref().is_some_and(|s| s.kind == kind) {
        app.scan = None;
        app.log(format!("{} scan cancelled", app.mode.label()));
        return;
    }
    match kind {
        ScanKind::Psk31 => {
            refresh_psk_hits(app);
            if !app.psk_hits.is_empty() {
                next_psk31(app, true);
                return;
            }
        }
        ScanKind::Cw => {
            refresh_cw_hits(app);
            if !app.cw_hits.is_empty() {
                next_cw(app, true);
                return;
            }
        }
        ScanKind::Energy => {}
    }
    start_scan(app, radio, kind);
}

fn retune(app: &mut App, radio: &radio::Radio, freq: f64) {
    let freq = freq.clamp(100_000.0, 30_000_000.0);
    if app.band_idx < app.band_gains.len() {
        app.band_gains[app.band_idx] = app.gain;
    }
    let old_band = app.band_idx;
    app.center = freq;
    // A new centre invalidates the old periodogram, scout list and AGC
    // hang. Leaving them in place is why coming back to a band looked
    // empty — the colour scale and hardware gain still belonged to the
    // previous band, which reads as a filter having switched on.
    app.waterfall.clear();
    app.spectrum.clear();
    app.smoothed.clear();
    app.spec_work.clear();
    app.wf_accum.clear();
    app.floor_db = -90.0;
    app.ceil_db = -20.0;
    app.psk_hits.clear();
    app.cw_hits.clear();
    app.idents.clear();
    app.tracks.clear();
    app.heard.clear();
    app.scout_iq.clear();
    // DC offset and IQ imbalance are both frequency-dependent, so what was
    // learned on the old centre is wrong for the new one.
    app.front.reset();
    app.ident_at = Instant::now();
    // Every automatic slot was tuned relative to the old centre.
    app.auto.clear();
    if let Some(d) = &mut app.decoder {
        d.hop();
    }
    app.soft_agc.reset();
    app.hw_hot = 0;
    app.hw_quiet = 0;
    app.hw_trim_at = Instant::now();
    let _ = radio.cmd.send(radio::Cmd::Tune(freq));
    if let Some(b) = bands::band_for(freq) {
        app.band_idx = bands::BANDS
            .iter()
            .position(|x| x.name == b.name)
            .unwrap_or(app.band_idx);
    }
    if app.band_idx != old_band && app.band_idx < app.band_gains.len() {
        app.gain = app.band_gains[app.band_idx];
        app.pending_gain = Some(app.gain);
    }
}

fn start_scan(app: &mut App, radio: &radio::Radio, kind: ScanKind) {
    let band = &bands::BANDS[app.band_idx];
    // Step by slightly less than the span so the edges overlap.
    let step = app.rate * 0.8;
    let start = band.start + app.rate / 2.0;
    // Mode scouts need a dwell long enough to score keying / BPSK.
    let dwell_ms = match kind {
        ScanKind::Cw => 900,
        ScanKind::Psk31 => 700,
        ScanKind::Energy => 550,
    };
    let state = ScanState {
        end: band.end,
        step,
        cur: start,
        dwell_until: Instant::now() + Duration::from_millis(dwell_ms),
        results: Vec::new(),
        kind,
    };
    let tag = match kind {
        ScanKind::Cw => "CW ",
        ScanKind::Psk31 => "PSK31 ",
        ScanKind::Energy => "",
    };
    app.log(format!(
        "{tag}scanning {} ({:.3}-{:.3} MHz)",
        band.name,
        band.start / 1e6,
        band.end / 1e6
    ));
    retune(app, radio, start);
    app.scan = Some(state);
}

/// Advance the scan; returns Some(true) when the sweep has finished.
fn step_scan(app: &mut App) -> Option<bool> {
    let (cur, end, step, ready) = {
        let s = app.scan.as_ref()?;
        (s.cur, s.end, s.step, Instant::now() >= s.dwell_until)
    };
    if !ready {
        return Some(false);
    }

    let kind = app.scan.as_ref()?.kind;
    match kind {
        ScanKind::Psk31 => {
            let peaks = find_peaks(&app.smoothed, &app.noise_floor, 0.0, app.rate);
            let hits = psk31::scan_span(&app.scout_iq, app.rate, &peaks);
            if let Some(s) = app.scan.as_mut() {
                for h in hits {
                    s.results.push(ScanHit {
                        freq: s.cur + h.offset_hz as f64,
                        score: h.quality * 100.0,
                        label: "PSK31",
                    });
                }
                s.cur += step;
                s.dwell_until = Instant::now() + Duration::from_millis(700);
            }
        }
        ScanKind::Cw => {
            let peaks = find_peaks(&app.smoothed, &app.noise_floor, 0.0, app.rate);
            let hits = cw::scan_span(&app.scout_iq, app.rate, &peaks);
            if let Some(s) = app.scan.as_mut() {
                for h in hits {
                    s.results.push(ScanHit {
                        freq: s.cur + h.offset_hz as f64,
                        score: h.quality * 100.0,
                        label: "CW",
                    });
                }
                s.cur += step;
                s.dwell_until = Instant::now() + Duration::from_millis(900);
            }
        }
        ScanKind::Energy => {
            let peaks = find_peaks(&app.smoothed, &app.noise_floor, cur, app.rate);
            let ids = identify::classify_span(&app.scout_iq, app.rate, &app.smoothed, app.center);
            if let Some(s) = app.scan.as_mut() {
                for (f, snr) in peaks {
                    let label = ids
                        .iter()
                        .min_by(|a, b| {
                            let da = (s.cur + a.offset_hz as f64 - f).abs();
                            let db = (s.cur + b.offset_hz as f64 - f).abs();
                            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .filter(|i| {
                            (s.cur + i.offset_hz as f64 - f).abs()
                                < i.bw_hz.max(400.0) as f64
                        })
                        .map(|i| i.kind.label())
                        .unwrap_or("");
                    s.results.push(ScanHit {
                        freq: f,
                        score: snr,
                        label,
                    });
                }
                s.cur += step;
                s.dwell_until = Instant::now() + Duration::from_millis(550);
            }
        }
    }

    if cur + step > end {
        // Sweep complete: summarise into the text pane.
        let kind = app.scan.as_ref()?.kind;
        let mut results = app.scan.as_mut()?.results.clone();
        if matches!(kind, ScanKind::Psk31 | ScanKind::Cw) {
            let label = match kind {
                ScanKind::Cw => "CW",
                _ => "PSK31",
            };
            results.sort_by(|a, b| {
                a.freq
                    .partial_cmp(&b.freq)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            results.dedup_by(|a, b| (a.freq - b.freq).abs() < 40.0);
            app.text
                .push_str(&format!("\n--- {label} scan (low → high) ---\n"));
            app.push_decode_note(format!("--- {label} scan (low → high) ---"));
            for h in results.iter() {
                let row = format!("{:>10.3} kHz  q={:.0}%", h.freq / 1000.0, h.score);
                app.text.push_str(&row);
                app.text.push('\n');
                app.push_decode_note(row);
            }
            app.text.push_str(&format!("--- end of {label} scan ---\n"));
            app.push_decode_note(format!("--- end of {label} scan ---"));
            app.log(format!("{label} scan: {} signal(s)", results.len()));
            if let Some(first) = results.first() {
                let f = first.freq;
                if let Some(d) = &mut app.decoder {
                    d.hop();
                }
                app.cursor = 0.0;
                app.pending_tune = Some(f);
                match kind {
                    ScanKind::Cw => {
                        app.cw_hits = results
                            .iter()
                            .map(|h| CwHit {
                                offset_hz: (h.freq - f) as f32,
                                score: h.score,
                                quality: h.score / 100.0,
                            })
                            .collect();
                    }
                    _ => {
                        app.psk_hits = results
                            .iter()
                            .map(|h| PskHit {
                                offset_hz: (h.freq - f) as f32,
                                score: h.score,
                                quality: h.score / 100.0,
                            })
                            .collect();
                    }
                }
            }
        } else {
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            results.dedup_by(|a, b| (a.freq - b.freq).abs() < 200.0);
            app.text.push_str("\n--- scan results (strongest first) ---\n");
            app.push_decode_note("--- scan results (strongest first) ---".into());
            for h in results.iter().take(25) {
                let marker = bands::MARKERS
                    .iter()
                    .find(|m| (m.freq - h.freq).abs() < 1500.0)
                    .map(|m| m.label)
                    .unwrap_or("");
                let kind = if h.label.is_empty() { marker } else { h.label };
                let row = format!(
                    "{:>10.3} kHz  {:5.1} dB  {}",
                    h.freq / 1000.0,
                    h.score,
                    kind
                );
                app.text.push_str(&row);
                app.text.push('\n');
                app.push_decode_note(row);
            }
            app.text.push_str("--- end of scan ---\n");
            app.push_decode_note("--- end of scan ---".into());
            let n = results.iter().filter(|h| !h.label.is_empty()).count();
            app.log(format!("scan complete — {n} labelled"));
        }
        return Some(true);
    }
    Some(false)
}

/// Minimum SNR over the median noise floor for something to count as a signal.
const SIGNAL_SNR_DB: f32 = 10.0;

/// Contiguous runs of spectrum standing above the noise floor, reported at
/// their strongest bin.
///
/// Segmenting on a threshold rather than picking local maxima matters: a
/// broadcast FM carrier or an SSB signal is a wide plateau, not a spike, and a
/// strict local-maximum test walks straight past it.
fn find_peaks(spectrum: &[f32], floor: &[f32], center: f64, rate: f64) -> Vec<(f64, f32)> {
    find_peaks_above(spectrum, floor, center, rate, SIGNAL_SNR_DB)
}

/// Candidate frequencies for the narrowband scouts.
///
/// `find_peaks` answers "where is the spectrum loud", by walking runs of bins
/// over a threshold. That is the right question for a wide signal and the
/// wrong one for a narrow one, twice over: a 31 Hz PSK31 signal spread across
/// a 47 Hz bin does not reach the 10 dB bar until it is well above where the
/// decoder copies it happily, and simply lowering the bar makes it worse —
/// the noise then forms long contiguous runs that each collapse to a single
/// peak, swallowing the real signal inside one.
///
/// So ask the other question: which bins are sharp local maxima standing out
/// of their *own* neighbourhood. That is what a narrowband carrier looks like
/// however weak it is, and the scout behind this mixes each candidate down
/// and matches it, which is a far stronger test than any bin level.
fn scout_peaks(spectrum: &[f32], floor: &[f32], center: f64, rate: f64) -> Vec<(f64, f32)> {
    const NEAR: usize = 3; // local-max half-width
    const CTX: usize = 40; // neighbourhood the prominence is measured against
    const PROMINENCE_DB: f32 = 2.75;
    let n = spectrum.len();
    if n < 4 * CTX {
        return Vec::new();
    }
    let dc = n / 2;
    let edge = (n / 50).max(CTX);
    let mut candidates: Vec<(f64, f32, f32)> = Vec::new();
    let mut ctx: Vec<f32> = Vec::with_capacity(2 * CTX + 1);
    for i in edge..n - edge {
        if i.abs_diff(dc) <= 2 {
            continue; // the LO spike is not a signal
        }
        let v = spectrum[i];
        if !(i - NEAR..=i + NEAR).all(|k| v >= spectrum[k]) {
            continue;
        }
        // Median of the surroundings, excluding the peak's own skirts.
        ctx.clear();
        ctx.extend(
            (i - CTX..=i + CTX)
                .filter(|k| k.abs_diff(i) > NEAR)
                .map(|k| spectrum[k]),
        );
        ctx.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let local = ctx[ctx.len() / 2];
        let reference = floor.get(i).copied().unwrap_or(local).max(local - 3.0);
        if v - reference < PROMINENCE_DB {
            continue;
        }
        let freq = center + (i as f64 - n as f64 / 2.0) * rate / n as f64;
        candidates.push((freq, v - reference, v - local));
    }
    // A lowered gate is useful when it adds a handful of historically-quiet
    // bins. If it opens across the whole span, the floor has not settled (or
    // the span is pure noise); retain the former 4 dB gate for that frame.
    let crowded = candidates.len() > 64;
    candidates.into_iter()
        .filter(|(_, _, local_prom)| !crowded || *local_prom >= 4.0)
        .map(|(freq, prom, _)| (freq, prom))
        .collect()
}

fn find_peaks_above(
    spectrum: &[f32],
    floor: &[f32],
    center: f64,
    rate: f64,
    snr_db: f32,
) -> Vec<(f64, f32)> {
    let n = spectrum.len();
    if n < 8 {
        return Vec::new();
    }
    let mut sorted = spectrum.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = sorted[n / 2];
    // The LO leaks a spike at the centre of the span; it is not a signal.
    let dc = n / 2;
    let edge = (n / 50).max(2);
    let usable = |i: usize| {
        let reference = floor.get(i).copied().unwrap_or(med).max(med - 3.0);
        i >= edge && i + edge < n && i.abs_diff(dc) > 2 && spectrum[i] >= reference + snr_db
    };

    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        if !usable(i) {
            i += 1;
            continue;
        }
        let mut best = i;
        while i < n && usable(i) {
            if spectrum[i] > spectrum[best] {
                best = i;
            }
            i += 1;
        }
        let freq = center + (best as f64 - n as f64 / 2.0) * rate / n as f64;
        out.push((freq, spectrum[best] - med));
    }
    out
}

#[cfg(test)]
mod noise_floor_detection_tests {
    use super::*;

    fn old_scout_count(spectrum: &[f32]) -> usize {
        let local = -80.0;
        spectrum.iter().enumerate().filter(|(i, v)| **v >= local + 4.0 && i.abs_diff(spectrum.len() / 2) > 2).count()
    }

    fn measured_thresholds() -> (f32, f32) {
        let n = 1024;
        let bin = 300;
        let mut floor = vec![-80.0; n];
        // This bin has historically been quiet; current local QRM has raised
        // the neighbourhood without erasing that information.
        floor[bin] = -90.0;
        let mut old_at = f32::INFINITY;
        let mut new_at = f32::INFINITY;
        for half_db in 0..=16 {
            let snr = half_db as f32 * 0.5;
            let mut spectrum = vec![-80.0; n];
            spectrum[bin] += snr;
            if old_scout_count(&spectrum) > 0 { old_at = old_at.min(snr); }
            if !scout_peaks(&spectrum, &floor, 0.0, 48_000.0).is_empty() { new_at = new_at.min(snr); }
        }
        (old_at, new_at)
    }

    #[test]
    fn tracked_floor_opens_the_candidate_gate_by_two_db_without_flat_noise_hits() {
        let (old_at, new_at) = measured_thresholds();
        assert!(old_at - new_at >= 2.0, "gate improved only {:.1} dB", old_at - new_at);
        let noise = vec![-80.0; 1024];
        assert_eq!(old_scout_count(&noise), 0);
        assert!(scout_peaks(&noise, &noise, 0.0, 48_000.0).is_empty());
    }

    #[test]
    #[ignore]
    fn bench_tracked_floor_candidate_gate() {
        let (old_at, new_at) = measured_thresholds();
        println!("scout candidate threshold: old {old_at:.1} dB, tracked floor {new_at:.1} dB ({:.1} dB improvement)", old_at - new_at);
    }
}

// ---------------------------------------------------------------- rendering

/// Status, spectrum, waterfall, decode, activity. Shared by the renderer
/// and mouse hit-testing so they can never disagree.
fn pane_rects(area: Rect, app: &App) -> [Rect; 5] {
    let wide = matches!(
        app.mode,
        Mode::Ft8 | Mode::Ft4 | Mode::Cw | Mode::Psk31 | Mode::Auto
    );
    // `v` enlarges the decode pane at the expense of the waterfall.
    let dec = match (wide, app.decode_zoom) {
        (false, 0) => Constraint::Length(9),
        (true, 0) => Constraint::Length(13),
        (_, 1) => Constraint::Percentage(45),
        (_, _) => Constraint::Percentage(65),
    };
    let chunks = Layout::vertical([
        // 2 content lines + 2 border rows
        Constraint::Length(4),
        Constraint::Length(10),
        Constraint::Min(3),
        dec,
        // 1 content line + 2 border rows
        Constraint::Length(3),
    ])
    .split(area);
    [chunks[0], chunks[1], chunks[2], chunks[3], chunks[4]]
}

/// The three FT sub-panes (activity, messages, stations).
fn ft_cols(area: Rect) -> [Rect; 3] {
    let cols = Layout::horizontal([
        Constraint::Percentage(32),
        Constraint::Percentage(40),
        Constraint::Min(28),
    ])
    .split(area);
    [cols[0], cols[1], cols[2]]
}

fn draw(f: &mut Frame, app: &App) {
    let chunks = pane_rects(f.area(), app);

    draw_status(f, chunks[0], app);
    draw_spectrum(f, chunks[1], app);
    draw_waterfall(f, chunks[2], app);
    match app.mode {
        Mode::Ft8 | Mode::Ft4 => draw_ft(f, chunks[3], app),
        Mode::Cw => draw_cw(f, chunks[3], app),
        Mode::Psk31 => draw_psk(f, chunks[3], app),
        _ => draw_decode(f, chunks[3], app),
    }
    draw_activity(f, chunks[4], app);

    if let Some(ed) = &app.settings {
        draw_settings(f, f.area(), ed);
    } else if app.show_help {
        draw_help(f, f.area());
    }
}

fn draw_settings(f: &mut Frame, area: Rect, ed: &SettingsEdit) {
    let cursor = "█";
    let (call, grid) = if ed.field == 0 {
        (format!("{}{cursor}", ed.call), ed.grid.clone())
    } else {
        (ed.call.clone(), format!("{}{cursor}", ed.grid))
    };
    let row = |label: &str, value: String, active: bool| {
        let style = if active {
            Style::default().fg(Color::Yellow)
        } else {
            Style::default()
        };
        Line::from(vec![
            Span::raw(format!("  {label}")),
            Span::styled(value, style),
        ])
    };
    let text = vec![
        row("callsign:  ", call, ed.field == 0),
        row("grid:      ", grid, ed.field == 1),
        Line::from(""),
        Line::from("  tab: switch field   enter: save   esc: cancel"),
        Line::from(""),
        Line::from(Span::styled(
            "  FT8, FT4 and PSK31 spots go to pskreporter.info",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(Span::styled(
            "  with this callsign and locator as the receiver",
            Style::default().fg(Color::DarkGray),
        )),
    ];
    let w = 56.min(area.width.saturating_sub(4));
    let h = (text.len() as u16 + 2).min(area.height);
    let rect = Rect {
        x: area.x + (area.width.saturating_sub(w)) / 2,
        y: area.y + (area.height.saturating_sub(h)) / 2,
        width: w,
        height: h,
    };
    f.render_widget(ratatui::widgets::Clear, rect);
    f.render_widget(
        Paragraph::new(text).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" station settings "),
        ),
        rect,
    );
}

fn draw_status(f: &mut Frame, area: Rect, app: &App) {
    let band = bands::band_for(app.center).map(|b| b.name).unwrap_or("--");
    let dec_status = app
        .decoder
        .as_ref()
        .map(|d| d.status())
        .unwrap_or_else(|| "-".into());
    let marker = bands::MARKERS
        .iter()
        .find(|m| (m.freq - app.tuned_freq()).abs() < 300.0)
        .map(|m| format!(" [{}]", m.label))
        .unwrap_or_default();
    let (lo, hi) = app.view_range();

    // UTC clock, and for the slot-based modes a countdown to the boundary so
    // you can see a slot is about to be decoded (and that the clock is sane).
    let now = utc_secs();
    let utc = format!(
        "{:02}:{:02}:{:02}",
        (now / 3600.0) as u64 % 24,
        (now / 60.0) as u64 % 60,
        now as u64 % 60
    );
    let slot = match app.mode {
        Mode::Ft8 => Some(format!("-{:2.0}s", 15.0 - now % 15.0)),
        Mode::Ft4 => Some(format!("-{:2.1}s", 7.5 - now % 7.5)),
        _ => None,
    };

    // Colour code every datum so the eye can jump straight to it: labels dim
    // grey, values coloured by what they mean.
    let dim = Style::default().fg(Color::DarkGray);
    let lbl = |s: &'static str| Span::styled(s, dim);
    let val = |s: String| Span::styled(s, Style::default().fg(Color::White));

    let mut spans1 = vec![
        Span::styled(
            format!("{:.4} kHz", app.tuned_freq() / 1000.0),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(marker, Style::default().fg(Color::Magenta)),
    ];
    if !app.my_call.is_empty() {
        spans1.push(Span::styled(
            format!("   de {}", app.my_call),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ));
    }
    spans1.extend([
        lbl("   centre "),
        val(format!("{:.3}", app.center / 1000.0)),
        lbl("  cursor "),
        Span::styled(
            format!("{:+.0} Hz", app.cursor),
            Style::default().fg(Color::LightCyan),
        ),
        Span::raw("  "),
        Span::styled(
            band,
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ),
        lbl("  view "),
        val(format!("{:.2} kHz", (hi - lo) / 1000.0)),
        Span::styled(format!(" (x{:.0})", app.zoom), dim),
        lbl("  step "),
        val(format!("{:.0} Hz", app.step_hz())),
        Span::styled(format!("  {:.1} Hz/bin", app.bin_hz()), dim),
    ]);
    let line1 = Line::from(spans1);

    // Mode gets its own hue so the active decoder is obvious at a glance.
    let mode_color = match app.mode {
        Mode::Off => Color::DarkGray,
        Mode::Cw => Color::Yellow,
        Mode::Rtty => Color::Magenta,
        Mode::Psk31 => Color::LightBlue,
        Mode::Ft8 => Color::Green,
        Mode::Ft4 => Color::LightGreen,
        Mode::Auto => Color::LightRed,
    };
    let snr_color = if app.cursor_snr >= 10.0 {
        Color::Green
    } else if app.cursor_snr >= 0.0 {
        Color::Yellow
    } else {
        Color::Red
    };
    let dec_style = if app.decoder.as_ref().is_some_and(|d| d.locked()) {
        Style::default()
            .fg(Color::LightGreen)
            .add_modifier(Modifier::BOLD)
    } else if app.mode == Mode::Psk31 {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::White)
    };
    let mut spans2 = vec![
        Span::styled(
            app.mode.label(),
            Style::default()
                .fg(mode_color)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(dec_status, dec_style),
        lbl("  UTC "),
        Span::styled(utc, Style::default().fg(Color::LightBlue)),
    ];
    if let Some(slot) = slot {
        spans2.push(lbl("  slot "));
        spans2.push(Span::styled(slot, Style::default().fg(Color::Magenta)));
    }
    // Front-end correction, shown only while it is doing something. A DC
    // offset or a poor image rejection is a property of the receiver, not of
    // the band, and is otherwise invisible — which is how it went unnoticed.
    let (dc, rej) = app.front.status();
    if rej < 45.0 {
        spans2.push(lbl("  img "));
        spans2.push(Span::styled(
            format!("{rej:.0} dB"),
            Style::default().fg(if rej < 30.0 {
                Color::Yellow
            } else {
                Color::Gray
            }),
        ));
    }
    if dc > 0.02 {
        spans2.push(lbl("  dc "));
        spans2.push(Span::styled(
            format!("{:.0}%", dc * 100.0),
            Style::default().fg(if dc > 0.15 { Color::Yellow } else { Color::Gray }),
        ));
    }
    let (nb_level, blanks) = app.front.blanker_status();
    if nb_level != "off" {
        spans2.push(lbl("  nb "));
        spans2.push(Span::styled(
            format!("{blanks}/s"),
            Style::default().fg(if blanks > 0 { Color::LightCyan } else { Color::Gray }),
        ));
    }
    spans2.extend([
        lbl("  snr "),
        Span::styled(
            format!("{:+.0} dB", app.cursor_snr),
            Style::default().fg(snr_color),
        ),
        lbl("  sq "),
        if app.squelch {
            Span::styled(
                format!("{:.0}", app.squelch_db),
                Style::default().fg(Color::Cyan),
            )
        } else {
            Span::styled("off", dim)
        },
        Span::raw("  "),
    ]);
    match app.agc {
        AgcMode::Soft => {
            spans2.push(Span::styled(
                "AGC",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ));
            spans2.push(lbl(" hang "));
            // The soft gain rails at 0.08 and 60. Sitting on a rail means the
            // level reaching the decoders is wrong — too little to slice, or
            // clipped — which is invisible if only the mode name is shown.
            let g = app.soft_agc.gain();
            let railed = !(0.09..=59.0).contains(&g);
            spans2.push(Span::styled(
                format!("{:+.0} dB", 20.0 * g.log10()),
                Style::default().fg(if railed { Color::Yellow } else { Color::Gray }),
            ));
        }
        AgcMode::Hardware => {
            spans2.push(Span::styled(
                "AGC",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ));
            spans2.push(lbl(" hw"));
        }
        AgcMode::Off => {
            spans2.push(lbl("gain "));
            spans2.push(Span::styled(
                format!("{:.0} dB", app.gain),
                Style::default().fg(Color::Yellow),
            ));
        }
    }
    spans2.push(lbl("  fil "));
    spans2.push(Span::styled(
        if app.rx_filter == RxFilter::Auto {
            format!("auto {:.0}", app.rx_bandwidth())
        } else {
            app.rx_filter.label().to_string()
        },
        Style::default().fg(Color::LightCyan),
    ));
    spans2.push(lbl("  bias-T "));
    spans2.push(if app.biast {
        Span::styled(
            "ON",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        )
    } else {
        Span::styled("off", dim)
    });
    spans2.push(lbl("  wf "));
    spans2.push(Span::styled(
        format!("{}ms", WF_INTERVALS_MS[app.wf_idx]),
        dim,
    ));
    if let Some(s) = &app.scan {
        let tag = match s.kind {
            ScanKind::Psk31 => "  PSK SCAN",
            ScanKind::Cw => "  CW SCAN",
            ScanKind::Energy => "  SCANNING",
        };
        spans2.push(Span::styled(
            tag,
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        ));
    }
    if app.mode == Mode::Psk31 && !app.psk_hits.is_empty() {
        spans2.push(lbl("  psk "));
        spans2.push(Span::styled(
            app.psk_hits.len().to_string(),
            Style::default().fg(Color::Cyan),
        ));
    }
    if app.mode == Mode::Cw && !app.cw_hits.is_empty() {
        spans2.push(lbl("  cw "));
        spans2.push(Span::styled(
            app.cw_hits.len().to_string(),
            Style::default().fg(Color::Yellow),
        ));
    }
    if !app.idents.is_empty() {
        spans2.push(lbl("  id "));
        spans2.push(Span::styled(
            identify::summary(&app.idents),
            Style::default().fg(Color::LightCyan),
        ));
    }
    if let Some(r) = &app.reporter {
        // A bare cumulative total cannot distinguish "nothing new to report"
        // from "wedged", and the hourly re-report rule means a healthy
        // reporter sits still for long stretches. Show the queue and the age
        // of the last datagram alongside it.
        let s = r.stats();
        spans2.push(lbl("  spots "));
        spans2.push(Span::styled(
            s.sent.to_string(),
            Style::default().fg(Color::Green),
        ));
        let mut detail = Vec::new();
        if s.queued > 0 {
            detail.push(format!("{} queued", s.queued));
        }
        if s.suppressed > 0 {
            detail.push(format!("{} dup", s.suppressed));
        }
        match s.since_send {
            Some(age) if age < 90 => detail.push(format!("sent {age}s ago")),
            Some(age) => detail.push(format!("sent {}m ago", age / 60)),
            None => detail.push("none sent yet".into()),
        }
        spans2.push(Span::styled(
            format!(" ({})", detail.join(", ")),
            Style::default().fg(Color::DarkGray),
        ));
    }
    let line2 = Line::from(spans2);

    let title = if app.my_call.is_empty() {
        " hfscan  —  press ? for help ".to_string()
    } else {
        format!(" hfscan  ·  {}  —  press ? for help ", app.my_call)
    };
    let p = Paragraph::new(vec![line1, line2]).block(
        Block::default().borders(Borders::ALL).title(title),
    );
    f.render_widget(p, area);
}

/// Thin full-width strip: remembered detections, then the latest status note.
fn draw_activity(f: &mut Frame, area: Rect, app: &App) {
    let live_n = app
        .idents
        .iter()
        .filter(|i| i.kind != identify::Kind::Unknown)
        .count();
    let title = if app.heard.is_empty() {
        " activity ".to_string()
    } else {
        format!(
            " activity  {} heard{} ",
            app.heard.len(),
            if live_n > 0 {
                format!(" · {live_n} live")
            } else {
                String::new()
            }
        )
    };
    let block = Block::default().borders(Borders::ALL).title(title);
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }
    let w = inner.width as usize;
    let mut spans: Vec<Span> = Vec::new();
    let mut used = 0usize;
    let mut shown = 0usize;
    for h in &app.heard {
        let live = app.idents.iter().any(|i| {
            i.kind == h.kind
                && ((app.center + i.offset_hz as f64) - h.freq_hz).abs()
                    < i.bw_hz.max(80.0) as f64
                        + if matches!(h.kind, identify::Kind::Ft8 | identify::Kind::Ft4) {
                            2500.0
                        } else {
                            0.0
                        }
        });
        let ft = matches!(h.kind, identify::Kind::Ft8 | identify::Kind::Ft4);
        let span = h.freq_hi - h.freq_lo;
        let chip = if ft && span > 400.0 {
            format!(
                "{:.1}–{:.1} {} ×{}",
                h.freq_lo / 1000.0,
                h.freq_hi / 1000.0,
                h.kind.label(),
                h.count.max(1)
            )
        } else {
            format!(
                "{:.3} {} {:+.0}",
                h.freq_hz / 1000.0,
                h.kind.label(),
                h.snr_db
            )
        };
        let extra = if shown == 0 { 0 } else { 3 };
        if used + extra + chip.len() + 4 > w {
            break;
        }
        if shown > 0 {
            spans.push(Span::styled(" · ", Style::default().fg(Color::DarkGray)));
            used += 3;
        }
        let style = if live {
            Style::default()
                .fg(Color::Black)
                .bg(ident_color(h.kind))
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(ident_color(h.kind))
        };
        spans.push(Span::styled(format!(" {chip} "), style));
        used += chip.len() + 2;
        shown += 1;
    }
    let hidden = app.heard.len().saturating_sub(shown);
    if hidden > 0 && used + 4 <= w {
        let more = format!(" +{hidden}");
        if used + more.len() <= w {
            spans.push(Span::styled(more, Style::default().fg(Color::DarkGray)));
            used += 4;
        }
    }
    if let Some(note) = app
        .notes
        .iter()
        .rev()
        .nth(app.note_scroll)
    {
        let room = w.saturating_sub(used);
        if room > 8 {
            let sep = if used == 0 { "" } else { " │ " };
            let budget = room.saturating_sub(sep.len());
            // Measured and cut in cells, not bytes: a byte-index truncate can
            // split a character and leaves the status line wider than the room
            // it was given, which pushes the frame's own border off the grid.
            let note_text = sanitize(&note.text);
            let mut text = note_text.clone();
            if note_text.chars().count() > budget {
                text = note_text.chars().take(budget.saturating_sub(1)).collect();
                text.push('…');
            }
            spans.push(Span::styled(sep, Style::default().fg(Color::DarkGray)));
            let style = match note.kind {
                Some(k) => Style::default().fg(ident_color(k)),
                None => Style::default().fg(Color::Gray).add_modifier(Modifier::ITALIC),
            };
            spans.push(Span::styled(text, style));
        }
    } else if spans.is_empty() {
        spans.push(Span::styled(
            "listening…",
            Style::default().fg(Color::DarkGray),
        ));
    }
    f.render_widget(Paragraph::new(Line::from(spans)), inner);
}

/// Map the visible frequency window onto bin indices.
fn view_bins(app: &App, n: usize) -> (usize, usize) {
    let (lo, hi) = app.view_range();
    let to_bin = |hz: f64| {
        (((hz + app.rate / 2.0) / app.rate) * n as f64).clamp(0.0, n as f64 - 1.0) as usize
    };
    let a = to_bin(lo);
    let b = to_bin(hi).max(a + 1).min(n);
    (a, b)
}

/// Reduce a bin range to one value per terminal column, keeping the peak so
/// narrow carriers stay visible.
fn resample(spectrum: &[f32], width: usize) -> Vec<f32> {
    if spectrum.is_empty() || width == 0 {
        return vec![-140.0; width];
    }
    let mut out = Vec::with_capacity(width);
    for x in 0..width {
        let a = x * spectrum.len() / width;
        let b = (((x + 1) * spectrum.len()) / width)
            .max(a + 1)
            .min(spectrum.len());
        let mut m = f32::MIN;
        for v in &spectrum[a..b] {
            if *v > m {
                m = *v;
            }
        }
        out.push(m);
    }
    out
}

fn cursor_col(app: &App, width: usize) -> usize {
    freq_col(app, width, app.cursor)
}

fn freq_col(app: &App, width: usize, offset_hz: f64) -> usize {
    let (lo, hi) = app.view_range();
    let frac = (offset_hz - lo) / (hi - lo).max(1.0);
    ((frac * width as f64) as isize).clamp(0, width as isize - 1) as usize
}

fn ident_rgb(kind: identify::Kind) -> (u8, u8, u8) {
    match kind {
        identify::Kind::Cw => (240, 196, 48),
        identify::Kind::Psk31 => (80, 176, 255),
        identify::Kind::Rtty => (224, 96, 196),
        identify::Kind::Ft8 => (48, 208, 88),
        identify::Kind::Ft4 => (132, 228, 116),
        identify::Kind::Ssb => (228, 228, 236),
        identify::Kind::Am => (255, 208, 80),
        identify::Kind::Carrier => (148, 148, 156),
        identify::Kind::Unknown => (160, 160, 160),
    }
}

fn ident_color(kind: identify::Kind) -> Color {
    let (r, g, b) = ident_rgb(kind);
    Color::Rgb(r, g, b)
}

fn mix_rgb(c: (u8, u8, u8), bg: (u8, u8, u8), a: f32) -> Color {
    let a = a.clamp(0.0, 1.0);
    Color::Rgb(
        (c.0 as f32 * a + bg.0 as f32 * (1.0 - a)).round() as u8,
        (c.1 as f32 * a + bg.1 as f32 * (1.0 - a)).round() as u8,
        (c.2 as f32 * a + bg.2 as f32 * (1.0 - a)).round() as u8,
    )
}

fn ident_fade_fg(kind: identify::Kind, a: f32) -> Color {
    mix_rgb(ident_rgb(kind), (18, 18, 24), a)
}

fn ident_fade_bg(kind: identify::Kind, a: f32) -> Color {
    let (r, g, b) = ident_rgb(kind);
    mix_rgb((r / 6, g / 6, b / 6), (18, 18, 24), a)
}

fn ident_chip(kind: identify::Kind) -> String {
    format!("▌{}▐", kind.label())
}

fn ident_col_span(app: &App, w: usize, offset_hz: f32, bw_hz: f32) -> (usize, usize) {
    let lo = freq_col(app, w, (offset_hz - bw_hz * 0.5) as f64);
    let hi = freq_col(app, w, (offset_hz + bw_hz * 0.5) as f64);
    let (a, b) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    (a, b.max(a))
}

fn draw_spectrum(f: &mut Frame, area: Rect, app: &App) {
    let title = if app.idents.is_empty() {
        " spectrum ".to_string()
    } else {
        format!(" spectrum  {} ", identify::summary(&app.idents))
    };
    let block = Block::default().borders(Borders::ALL).title(title);
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 || app.smoothed.is_empty() {
        return;
    }

    let w = inner.width as usize;
    let h = inner.height as usize;
    // The last row is the frequency axis; the bars get the rest.
    let bars_h = if h > 1 { h - 1 } else { h };
    let (a, b) = view_bins(app, app.smoothed.len());
    let cols = resample(&app.smoothed[a..b], w);
    let cur = cursor_col(app, w);
    let (lo, hi) = (app.floor_db, app.ceil_db.max(app.floor_db + 10.0));

    // Shade the decoder's listen window so you can see what it is
    // hearing: FT8/FT4 the 200-3000 Hz USB passband above the dial,
    // everything else the mode bandwidth around the cursor.
    let (band_lo, band_hi, band_bg) = match app.mode {
        Mode::Ft8 | Mode::Ft4 => (
            app.cursor + decoders::ft8::FREQ_MIN as f64,
            app.cursor + decoders::ft8::FREQ_MAX as f64,
            Color::Rgb(40, 40, 50),
        ),
        Mode::Psk31 => {
            let half = app.rx_bandwidth() as f64 / 2.0;
            (
                app.cursor - half,
                app.cursor + half,
                Color::Rgb(16, 36, 52),
            )
        }
        Mode::Cw | Mode::Rtty => {
            let half = app.rx_bandwidth() as f64 / 2.0;
            (app.cursor - half, app.cursor + half, Color::Rgb(28, 28, 40))
        }
        // Auto is not listening at the cursor at all — its decoders are
        // spread across the span and are drawn as their own markers.
        Mode::Off | Mode::Auto => (0.0, 0.0, Color::Reset),
    };
    let shade = !matches!(app.mode, Mode::Off | Mode::Auto);
    let lock_col = app.decoder.as_ref().and_then(|d| {
        if d.locked() {
            Some(freq_col(app, w, app.cursor + d.lock_hz() as f64))
        } else {
            None
        }
    });
    // Span-scout hits: offsets from the radio centre. In-passband hits
    // are relative to the cursor — mark both so `n`/`p` have a target.
    let mut hit_cols = Vec::new();
    for h in &app.psk_hits {
        hit_cols.push(freq_col(app, w, h.offset_hz as f64));
    }
    for h in &app.cw_hits {
        hit_cols.push(freq_col(app, w, h.offset_hz as f64));
    }
    let now = Instant::now();
    let ident_spans: Vec<(usize, usize, usize, identify::Kind, f32)> = app
        .tracks
        .iter()
        .filter_map(|t| {
            let a = t.alpha(now);
            if a < 0.12 {
                return None;
            }
            let (lo, hi) = ident_col_span(app, w, t.offset_hz, t.bw_hz);
            let mid = freq_col(app, w, t.offset_hz as f64);
            Some((lo, hi, mid, t.kind, a))
        })
        .collect();
    if let Some(d) = &app.decoder {
        for hz in d.candidate_hz() {
            hit_cols.push(freq_col(app, w, app.cursor + hz as f64));
        }
    }
    let (vlo, vhi) = app.view_range();
    let in_band = |x: usize| {
        if !shade {
            return false;
        }
        let off = vlo + (x as f64 + 0.5) * (vhi - vlo) / w as f64;
        off >= band_lo && off <= band_hi
    };

    const BARS: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let mut lines = Vec::with_capacity(h);
    for row in 0..bars_h {
        let from_bottom = bars_h - 1 - row;
        let mut spans = Vec::with_capacity(w);
        for (x, v) in cols.iter().enumerate() {
            let norm = ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
            let eighths = (norm * (bars_h * 8) as f32) as usize;
            let cell = eighths.saturating_sub(from_bottom * 8).min(8);
            let ch = BARS[cell];
            let on_peak = ident_spans.iter().find(|(_, _, mid, _, _)| *mid == x);
            let in_occ = ident_spans
                .iter()
                .find(|(lo, hi, _, _, _)| x >= *lo && x <= *hi);
            let style = if x == cur {
                Style::default().fg(Color::Magenta).bg(Color::Rgb(40, 0, 40))
            } else if lock_col == Some(x) {
                Style::default()
                    .fg(Color::LightCyan)
                    .bg(Color::Rgb(0, 40, 50))
            } else if hit_cols.contains(&x) {
                Style::default()
                    .fg(Color::Cyan)
                    .bg(Color::Rgb(0, 24, 32))
            } else if let Some((_, _, _, k, a)) = on_peak {
                Style::default()
                    .fg(ident_fade_fg(*k, *a))
                    .bg(ident_fade_bg(*k, *a))
            } else if let Some((_, _, _, k, a)) = in_occ {
                Style::default().fg(heat(norm)).bg(ident_fade_bg(*k, *a))
            } else {
                let mut s = Style::default().fg(heat(norm));
                if in_band(x) {
                    s = s.bg(band_bg);
                }
                s
            };
            spans.push(Span::styled(ch.to_string(), style));
        }
        lines.push(Line::from(spans));
    }
    paint_ident_labels(&mut lines, app, w);
    if h > 1 {
        lines.push(axis_row(app, w));
    }
    f.render_widget(Paragraph::new(lines), inner);
}

/// Overlay occupancy shelves and kind chips on the top spectrum row.
fn paint_ident_labels(lines: &mut [Line], app: &App, w: usize) {
    if lines.is_empty() || w == 0 || app.tracks.is_empty() {
        return;
    }
    let now = Instant::now();
    let mut order: Vec<&LabelTrack> = app
        .tracks
        .iter()
        .filter(|t| t.alpha(now) >= 0.12)
        .collect();
    order.sort_by(|a, b| {
        a.score
            .partial_cmp(&b.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                a.snr_db
                    .partial_cmp(&b.snr_db)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });
    let row = lines[0].spans.clone();
    if row.len() != w {
        return;
    }
    let mut cells: Vec<Span> = row;
    // Weakest first so a stronger neighbour overwrites the shelf.
    for t in &order {
        let a = t.alpha(now);
        let (lo, hi) = ident_col_span(app, w, t.offset_hz, t.bw_hz);
        let shelf = Style::default()
            .fg(ident_fade_fg(t.kind, a))
            .bg(ident_fade_bg(t.kind, a));
        for x in lo..=hi {
            cells[x] = Span::styled("▀", shelf);
        }
    }
    let mut used = vec![false; w];
    // Strongest chips on top. Dim ones lose the inverse fill so they
    // fade into the shelf instead of popping off as a solid block.
    order.reverse();
    for t in order {
        let a = t.alpha(now);
        if a < 0.28 {
            continue;
        }
        let chip: Vec<char> = ident_chip(t.kind).chars().collect();
        if chip.is_empty() {
            continue;
        }
        let col = freq_col(app, w, t.offset_hz as f64);
        let start = col
            .saturating_sub(chip.len() / 2)
            .min(w.saturating_sub(chip.len()));
        let end = start + chip.len();
        if end > w || used[start..end].iter().any(|&u| u) {
            continue;
        }
        let fg = ident_fade_fg(t.kind, a);
        let bg = ident_fade_bg(t.kind, a);
        let body = if a > 0.65 {
            Style::default()
                .fg(Color::Rgb(
                    (18.0 * (1.0 - a) + 0.0) as u8,
                    (18.0 * (1.0 - a)) as u8,
                    (24.0 * (1.0 - a)) as u8,
                ))
                .bg(fg)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(fg).bg(bg).add_modifier(Modifier::BOLD)
        };
        let cap = Style::default().fg(fg).bg(bg);
        for (i, ch) in chip.iter().enumerate() {
            let style = if *ch == '▌' || *ch == '▐' { cap } else { body };
            cells[start + i] = Span::styled(ch.to_string(), style);
            used[start + i] = true;
        }
        if end < w {
            used[end] = true;
        }
        if start > 0 {
            used[start - 1] = true;
        }
    }
    lines[0] = Line::from(cells);
}

/// Frequency axis for the spectrum: '┬' ticks at round steps, labelled in kHz.
fn axis_row(app: &App, w: usize) -> Line<'static> {
    let (lo, hi) = app.view_range();
    let span = hi - lo;
    let mut cells = vec![' '; w];
    if span > 0.0 && w > 4 {
        // Round step (1/2/5 * 10^n) keeping labels roughly ten columns apart.
        let min_step = span / (w as f64 / 10.0).max(1.0);
        let base = 10f64.powf(min_step.log10().floor());
        let mut step = base * 10.0;
        for m in [1.0, 2.0, 5.0, 10.0] {
            if base * m >= min_step {
                step = base * m;
                break;
            }
        }
        let decimals = if step < 100.0 {
            2
        } else if step < 1000.0 {
            1
        } else {
            0
        };
        let mut end = 0usize; // first column past the previous label
        let mut tick = (lo / step).ceil() * step;
        while tick <= hi {
            let col = ((tick - lo) / span * w as f64) as usize;
            if col >= end && col < w {
                cells[col] = '┬';
                let label = format!("{:.1$}", (app.center + tick) / 1000.0, decimals);
                if col + 1 + label.len() <= w {
                    for (i, ch) in label.chars().enumerate() {
                        cells[col + 1 + i] = ch;
                    }
                    end = col + 1 + label.len() + 2;
                } else {
                    end = w;
                }
            }
            tick += step;
        }
    }
    Line::from(Span::styled(
        cells.into_iter().collect::<String>(),
        Style::default().fg(Color::DarkGray),
    ))
}

fn draw_waterfall(f: &mut Frame, area: Rect, app: &App) {
    let (lo, hi) = app.view_range();
    let start = (app.center + lo) / 1000.0;
    let end = (app.center + hi) / 1000.0;
    let mut title = format!(" waterfall  {start:.3} .. {end:.3} kHz ");
    if app.wf_res != WfRes::Quad {
        title.push_str("(half-block) ");
    }
    if app.wf_scroll > 0 {
        title.push_str(&format!("({} back) ", app.wf_scroll));
    }
    let block = Block::default().borders(Borders::ALL).title(title);
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }

    let w = inner.width as usize;
    let cur = cursor_col(app, w);
    let (dlo, dhi) = (app.floor_db, app.ceil_db.max(app.floor_db + 10.0));
    let scroll = app.wf_scroll;
    let norm = |v: f32| ((v - dlo) / (dhi - dlo)).clamp(0.0, 1.0);

    let row_spec = |idx: usize| -> Option<&Vec<f32>> {
        let spec = app.waterfall.get(idx + scroll)?;
        if spec.is_empty() {
            None
        } else {
            Some(spec)
        }
    };

    // `fcells` frequency subcells across each column, `tcells` time steps
    // stacked into each text row.
    let (fcells, tcells) = app.wf_res.cells();
    let row_cols = |idx: usize| -> Option<Vec<f32>> {
        let spec = row_spec(idx)?;
        let (a, b) = view_bins(app, spec.len());
        Some(resample(&spec[a..b], w * fcells))
    };

    let mut lines = Vec::with_capacity(inner.height as usize);
    for row in 0..inner.height as usize {
        // The newest of the time steps this row covers. If even that is
        // missing the history simply does not reach this far back.
        let Some(upper) = row_cols(row * tcells) else {
            lines.push(Line::from(""));
            continue;
        };
        // A half-filled history repeats the row it does have rather than
        // punching a hole in the middle of the display.
        let lower = if tcells > 1 {
            row_cols(row * tcells + 1).unwrap_or_else(|| upper.clone())
        } else {
            upper.clone()
        };
        let at = |c: &[f32], i: usize| norm(c.get(i).copied().unwrap_or(f32::MIN));

        let mut spans = Vec::with_capacity(w);
        for x in 0..w {
            let (ch, fg, bg) = match app.wf_res {
                WfRes::Quad => quad_cell([
                    at(&upper, x * 2),
                    at(&upper, x * 2 + 1),
                    at(&lower, x * 2),
                    at(&lower, x * 2 + 1),
                ]),
                WfRes::Freq => ('▌', at(&upper, x * 2), at(&upper, x * 2 + 1)),
                WfRes::Time => ('▀', at(&upper, x), at(&lower, x)),
            };
            let style = Style::default().fg(heat(fg)).bg(heat(bg));
            if x == cur {
                // Keep the cursor readable against whatever is behind it.
                spans.push(Span::styled("│", style.fg(Color::Magenta)));
            } else {
                spans.push(Span::styled(ch.to_string(), style));
            }
        }
        lines.push(Line::from(spans));
    }
    f.render_widget(Paragraph::new(lines), inner);
}

/// Word-wrap the transcript and return the window that fits, newest last
/// unless `scroll` (lines up from live) says otherwise. In FT mode CQ calls
/// are highlighted.
fn transcript_lines(app: &App, width: usize, rows: usize, ft: bool) -> Vec<Line<'static>> {
    let width = width.max(1);
    app.msg_rows.set(rows);
    let mut wrapped: Vec<String> = Vec::new();
    for line in app.text.split('\n') {
        // Sanitising here as well as at ingestion keeps a decoder that is
        // added later from being able to corrupt the grid, and guarantees one
        // char is one cell so this hard wrap matches what is drawn.
        let line = sanitize(line);
        if line.is_empty() {
            wrapped.push(String::new());
            continue;
        }
        let chars: Vec<char> = line.chars().collect();
        for c in chars.chunks(width) {
            wrapped.push(c.iter().collect());
        }
    }
    let max = wrapped.len().saturating_sub(rows);
    let start = wrapped
        .len()
        .saturating_sub(rows + app.msg_scroll.min(max));
    let end = (start + rows).min(wrapped.len());
    wrapped[start..end]
        .iter()
        .map(|s| {
            let is_cq = ft
                && s.rsplit("Hz  ")
                    .next()
                    .map(|m| m.starts_with("CQ"))
                    .unwrap_or(false);
            if is_cq {
                Line::from(Span::styled(s.clone(), Style::default().fg(Color::Cyan)))
            } else {
                Line::from(s.clone())
            }
        })
        .collect()
}

fn draw_decode(f: &mut Frame, area: Rect, app: &App) {
    let extra = match (app.mode, app.decoder.as_ref()) {
        (Mode::Psk31, Some(d)) if d.locked() => format!("  lock {:+.1} Hz", d.lock_hz()),
        (Mode::Psk31, _) => "  searching".into(),
        (Mode::Auto, _) if app.auto.is_empty() => "  listening for signals".into(),
        (Mode::Auto, _) => {
            let mut tally: Vec<String> = Vec::new();
            for k in [
                identify::Kind::Ft8,
                identify::Kind::Ft4,
                identify::Kind::Cw,
                identify::Kind::Rtty,
                identify::Kind::Psk31,
            ] {
                let n = app.auto.iter().filter(|s| s.kind == k).count();
                if n > 0 {
                    tally.push(format!("{n} {}", k.label()));
                }
            }
            format!("  {}", tally.join("  "))
        }
        _ => String::new(),
    };
    let mut title = format!(" decode: {}{extra} ", app.mode.label());
    if app.mode == Mode::Auto {
        let now = Instant::now();
        let live = app.rows.iter().filter(|r| r.live(now)).count();
        title = match app.auto_view {
            // The roster's own counts say more than the slot tally does.
            AutoView::Rows if !app.rows.is_empty() => {
                format!(" decode: AUTO — {live} live / {} held ", app.rows.len())
            }
            AutoView::Rows => format!(" decode: AUTO{extra} "),
            AutoView::Log => format!(" decode: AUTO — log{extra} "),
        };
    }
    if app.msg_scroll > 0 {
        let what = match (app.mode, app.auto_view) {
            (Mode::Auto, AutoView::Rows) => "scrolled down",
            _ => "scrolled up",
        };
        title = format!("{}({what} {}) ", title, app.msg_scroll);
    }
    // A title wider than the box overruns the top border and takes the frame
    // with it, so it is cut to what the border can hold.
    let cap = area.width.saturating_sub(2) as usize;
    if title.chars().count() > cap {
        title = title.chars().take(cap).collect();
    }
    let block = Block::default().borders(Borders::ALL).title(title);
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }

    if app.mode == Mode::Auto {
        let body = match app.auto_view {
            AutoView::Rows => signal_rows_lines(app, inner),
            AutoView::Log => decode_log_lines(app, inner),
        };
        f.render_widget(Paragraph::new(body), inner);
        return;
    }
    let body = transcript_lines(app, inner.width as usize, inner.height as usize, false);
    f.render_widget(Paragraph::new(body), inner);
}

/// A silence long enough to matter, as a short human span: `12s`, `4m`, `1h`.
fn short_age(d: Duration) -> String {
    let s = d.as_secs();
    if s < 60 {
        format!("{s}s")
    } else if s < 3600 {
        format!("{}m", s / 60)
    } else {
        format!("{}h", s / 3600)
    }
}

/// The held rows: one line per signal, its copy accumulating in place.
///
/// The list is already ordered — live signals first, silent ones sinking to
/// the bottom — so this only windows it. `msg_scroll` counts rows skipped
/// from the top here, because the interesting end of a roster is the top,
/// unlike the log where it is the newest line at the bottom.
///
/// Each row draws its copy right-aligned to the newest character, so a slow
/// station's text fills the width and then scrolls, rather than appearing a
/// character or two at a time and being flushed away.
fn signal_rows_lines(app: &App, inner: Rect) -> Vec<Line<'static>> {
    let rows = inner.height as usize;
    let width = inner.width as usize;
    app.msg_rows.set(rows);
    let now = Instant::now();

    let dim = Style::default().fg(Color::DarkGray);
    if app.rows.is_empty() {
        return vec![Line::from(Span::styled(
            "  age    kHz  mode  copy — nothing heard yet",
            dim,
        ))];
    }

    // age (4) + gap, freq (9) + gap, mode (4) + two gaps before the copy.
    const HEAD: usize = 5 + 10 + 6;
    let copy_width = width.saturating_sub(HEAD).max(8);

    app.rows
        .iter()
        .skip(app.msg_scroll.min(app.rows.len().saturating_sub(1)))
        .take(rows)
        .map(|r| {
            let live = r.live(now);
            let (cr, cg, cb) = ident_rgb(r.kind);
            let mut spans: Vec<Span<'static>> = Vec::with_capacity(4);

            let age = if live {
                Span::styled(format!("{:<4} ", "live"), Style::default().fg(Color::Green))
            } else {
                let a = short_age(now.duration_since(r.last_copy));
                Span::styled(format!("{a:<4} "), dim)
            };
            spans.push(age);
            spans.push(Span::styled(
                format!("{:>9.1} ", r.dial_hz / 1000.0),
                Style::default().fg(if live { Color::Gray } else { Color::DarkGray }),
            ));
            spans.push(Span::styled(
                format!("{:<4}  ", r.mode),
                Style::default().fg(if live {
                    Color::Rgb(cr, cg, cb)
                } else {
                    Color::DarkGray
                }),
            ));

            // The tail, so the newest copy is always on screen. A leading
            // ellipsis marks that there is more behind it.
            let n = r.copy.chars().count();
            let copy: String = if n > copy_width {
                let skip = n - copy_width + 1;
                std::iter::once('…')
                    .chain(r.copy.chars().skip(skip))
                    .collect()
            } else {
                r.copy.clone()
            };
            let style = if !live {
                Style::default().fg(Color::Gray)
            } else if r.copy.contains("CQ") {
                Style::default().fg(Color::Cyan)
            } else {
                Style::default()
            };
            spans.push(Span::styled(copy, style));
            Line::from(spans)
        })
        .collect()
}

/// The automatic transcript as fixed columns: time, frequency, mode, copy.
///
/// Every entry is exactly one row, truncated rather than wrapped, so the
/// pane's height in rows is its length in entries. That is what makes the
/// scroll offset unambiguous, and it means no line can ever spill past the
/// right-hand border. Wide readers get the time column; narrow ones lose it
/// before the copy itself is squeezed.
fn decode_log_lines(app: &App, inner: Rect) -> Vec<Line<'static>> {
    let rows = inner.height as usize;
    let width = inner.width as usize;
    app.msg_rows.set(rows);
    let total = app.decode_log.len();
    let max = total.saturating_sub(rows);
    let start = total.saturating_sub(rows + app.msg_scroll.min(max));
    let end = (start + rows).min(total);

    let show_time = width >= 46;
    // time (8) + gap, freq (9) + gap, mode (4) + two gaps before the copy.
    let head = if show_time { 9 + 10 + 6 } else { 10 + 6 };
    let copy_width = width.saturating_sub(head).max(8);

    let dim = Style::default().fg(Color::DarkGray);
    if total == 0 {
        let hint = if show_time {
            "  time      kHz  mode  copy"
        } else {
            "       kHz  mode  copy"
        };
        return vec![Line::from(Span::styled(hint, dim))];
    }
    app.decode_log
        .iter()
        .skip(start)
        .take(end.saturating_sub(start))
        .map(|e| {
            let (r, g, b) = ident_rgb(e.kind);
            let mut spans: Vec<Span<'static>> = Vec::with_capacity(6);
            if show_time {
                spans.push(Span::styled(format!("{} ", e.stamp), dim));
            }
            if e.dial_hz > 0.0 {
                spans.push(Span::styled(
                    format!("{:>9.1} ", e.dial_hz / 1000.0),
                    Style::default().fg(Color::Gray),
                ));
                spans.push(Span::styled(
                    format!("{:<4}  ", e.mode),
                    Style::default().fg(Color::Rgb(r, g, b)),
                ));
            } else {
                // The scanner's own remarks keep the copy column aligned but
                // leave the frequency and mode columns empty.
                spans.push(Span::raw(" ".repeat(16)));
            }
            let mut copy: String = e.text.chars().take(copy_width).collect();
            if e.text.chars().count() > copy_width && copy_width > 1 {
                copy.pop();
                copy.push('…');
            }
            let style = if e.dial_hz <= 0.0 {
                dim
            } else if e.text.starts_with("CQ") || e.text.contains(" CQ ") {
                Style::default().fg(Color::Cyan)
            } else {
                Style::default()
            };
            spans.push(Span::styled(copy, style));
            Line::from(spans)
        })
        .collect()
}

fn cw_cols(area: Rect) -> [Rect; 3] {
    let cols = Layout::horizontal([
        Constraint::Percentage(34),
        Constraint::Percentage(38),
        Constraint::Min(26),
    ])
    .split(area);
    [cols[0], cols[1], cols[2]]
}

fn draw_cw(f: &mut Frame, area: Rect, app: &App) {
    let cols = cw_cols(area);
    let view = app.decoder.as_ref().and_then(|d| d.cw_view());
    draw_cw_envelope(f, cols[0], app, view.as_ref());
    draw_cw_text(f, cols[1], app);
    draw_cw_tuner(f, cols[2], app, view.as_ref());
}

/// Scrolling keying envelope. Newest on the right; green while the key
/// is down. The slice thresholds are a dim dotted rule.
fn draw_cw_envelope(f: &mut Frame, area: Rect, _app: &App, view: Option<&CwView>) {
    let title = match view {
        Some(v) if v.key_down => " envelope  KEY ",
        Some(_) => " envelope ",
        None => " envelope ",
    };
    let block = Block::default().borders(Borders::ALL).title(title);
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }
    let Some(v) = view else {
        f.render_widget(
            Paragraph::new("waiting for audio").style(Style::default().fg(Color::DarkGray)),
            inner,
        );
        return;
    };
    let w = inner.width as usize;
    let h = inner.height as usize;
    let n = v.env.len();
    let mut cols = vec![0.0f32; w];
    let mut keyed = vec![false; w];
    if n > 0 {
        for x in 0..w {
            let i = x * n / w;
            cols[x] = v.env[i.min(n - 1)];
            keyed[x] = v.keyed.get(i.min(n - 1)).copied().unwrap_or(false);
        }
    }
    const BARS: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let mut lines = Vec::with_capacity(h);
    for row in 0..h {
        let from_bottom = h - 1 - row;
        let mut spans = Vec::with_capacity(w);
        for x in 0..w {
            let eighths = (cols[x].clamp(0.0, 1.0) * (h * 8) as f32) as usize;
            let cell = eighths.saturating_sub(from_bottom * 8).min(8);
            let ch = BARS[cell];
            let on_row = ((1.0 - v.on_thr) * h as f32) as usize;
            let off_row = ((1.0 - v.off_thr) * h as f32) as usize;
            let style = if keyed[x] {
                Style::default().fg(Color::LightGreen)
            } else if row == on_row || row == off_row {
                Style::default().fg(Color::DarkGray)
            } else {
                Style::default().fg(Color::Gray)
            };
            spans.push(Span::styled(ch.to_string(), style));
        }
        lines.push(Line::from(spans));
    }
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_cw_text(f: &mut Frame, area: Rect, app: &App) {
    let mut title = " copy ".to_string();
    if app.msg_scroll > 0 {
        title = format!(" copy (scrolled up {}) ", app.msg_scroll);
    }
    let block = Block::default().borders(Borders::ALL).title(title);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let body = transcript_lines(app, inner.width.max(1) as usize, inner.height as usize, false);
    f.render_widget(Paragraph::new(body), inner);
}

fn draw_cw_tuner(f: &mut Frame, area: Rect, app: &App, view: Option<&CwView>) {
    let rf = app.tuned_freq();
    let lock = view.map(|v| v.lock_hz as f64).unwrap_or(0.0);
    let center = rf + lock;
    let title = format!(" tuner  {:.3} kHz ", center / 1000.0);
    let block = Block::default().borders(Borders::ALL).title(title);
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }
    let dim = Style::default().fg(Color::DarkGray);
    let mut lines: Vec<Line> = Vec::new();
    match view {
        None => lines.push(Line::from(Span::styled("no decoder", dim))),
        Some(v) => {
            let err = v.tune_err_hz;
            let err_col = if err.abs() < 2.0 {
                Color::LightGreen
            } else if err.abs() < 8.0 {
                Color::Yellow
            } else {
                Color::Red
            };
            lines.push(Line::from(vec![
                Span::styled("rf     ", dim),
                Span::styled(
                    format!("{:.3} kHz", center / 1000.0),
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
                ),
            ]));
            lines.push(Line::from(vec![
                Span::styled("lock   ", dim),
                Span::styled(
                    format!("{:+.1} Hz", v.lock_hz),
                    Style::default().fg(if v.locked {
                        Color::LightCyan
                    } else {
                        Color::DarkGray
                    }),
                ),
            ]));
            lines.push(Line::from(vec![
                Span::styled("tune   ", dim),
                Span::styled(format!("{err:+.1} Hz"), Style::default().fg(err_col)),
            ]));
            lines.push(tune_meter(inner.width as usize, err));
            lines.push(Line::from(vec![
                Span::styled("wpm    ", dim),
                Span::styled(format!("{:.0}", v.wpm), Style::default().fg(Color::White)),
                Span::styled(format!("   dit {:.0} ms", v.dit_ms), dim),
            ]));
            lines.push(Line::from(vec![
                Span::styled("q      ", dim),
                Span::styled(
                    format!("{:.0}%", v.quality * 100.0),
                    Style::default().fg(Color::White),
                ),
                Span::styled(
                    if v.key_down { "   KEY" } else { "   —" },
                    Style::default().fg(if v.key_down {
                        Color::LightGreen
                    } else {
                        Color::DarkGray
                    }),
                ),
            ]));
            let elem = if v.symbol.is_empty() {
                "·".into()
            } else {
                v.symbol.clone()
            };
            lines.push(Line::from(vec![
                Span::styled("elem   ", dim),
                Span::styled(elem, Style::default().fg(Color::Cyan)),
            ]));
            lines.push(Line::from(Span::styled(
                "u/i fine   g centre   n next",
                dim,
            )));
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled("tones", dim)));
            let passband = !v.hits.is_empty();
            let mut listed: Vec<(f32, f32, bool)> = if passband {
                v.hits
                    .iter()
                    .map(|h| {
                        (
                            h.offset_hz,
                            h.quality,
                            (h.offset_hz - v.lock_hz).abs() < 15.0,
                        )
                    })
                    .collect()
            } else {
                app.cw_hits
                    .iter()
                    .map(|h| (h.offset_hz, h.quality, false))
                    .collect()
            };
            listed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let skip = app.st_scroll.min(listed.len().saturating_sub(1));
            for (off, q, on) in listed.into_iter().skip(skip).take(8) {
                let abs = if passband {
                    (rf + off as f64) / 1000.0
                } else {
                    (app.center + off as f64) / 1000.0
                };
                let mark = if on { '>' } else { ' ' };
                let style = if on {
                    Style::default().fg(Color::LightCyan)
                } else {
                    Style::default()
                };
                lines.push(Line::from(Span::styled(
                    format!("{mark}{abs:8.3}  q={:.0}%", q * 100.0),
                    style,
                )));
            }
        }
    }
    f.render_widget(Paragraph::new(lines), inner);
}

fn tune_meter(width: usize, err_hz: f32) -> Line<'static> {
    let inner = width.saturating_sub(2).max(8);
    let mut cells = vec!['·'; inner];
    let mid = inner / 2;
    cells[mid] = '|';
    let span = 20.0f32;
    let x = (mid as f32 + (err_hz / span) * mid as f32)
        .round()
        .clamp(0.0, (inner - 1) as f32) as usize;
    cells[x] = if err_hz.abs() < 2.0 { '●' } else { '◆' };
    let bar: String = cells.into_iter().collect();
    let col = if err_hz.abs() < 2.0 {
        Color::LightGreen
    } else if err_hz.abs() < 8.0 {
        Color::Yellow
    } else {
        Color::Red
    };
    Line::from(Span::styled(format!("[{bar}]"), Style::default().fg(col)))
}

fn draw_psk(f: &mut Frame, area: Rect, app: &App) {
    let cols = cw_cols(area);
    let view = app.decoder.as_ref().and_then(|d| d.psk_view());
    draw_psk_scope(f, cols[0], view.as_ref());
    draw_cw_text(f, cols[1], app);
    draw_psk_tuner(f, cols[2], app, view.as_ref());
}

/// Constellation (I/Q of recent symbols) over a short envelope of the
/// locked baseband. BPSK sits on the real axis; a residual carrier
/// tilts the cloud — that is the fine-tune cue.
fn draw_psk_scope(f: &mut Frame, area: Rect, view: Option<&PskView>) {
    let title = match view {
        Some(v) if v.locked => " eye  LOCK ",
        _ => " eye ",
    };
    let block = Block::default().borders(Borders::ALL).title(title);
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }
    let Some(v) = view else {
        f.render_widget(
            Paragraph::new("waiting for audio").style(Style::default().fg(Color::DarkGray)),
            inner,
        );
        return;
    };
    let w = inner.width as usize;
    let h = inner.height as usize;
    let env_h = if h > 4 { 2 } else { 0 };
    let eye_h = h.saturating_sub(env_h).max(1);

    let mut grid = vec![vec![' '; w]; eye_h];
    let mid_x = w / 2;
    let mid_y = eye_h / 2;
    for x in 0..w {
        grid[mid_y][x] = '─';
    }
    for y in 0..eye_h {
        grid[y][mid_x] = '│';
    }
    grid[mid_y][mid_x] = '┼';
    for (i, s) in v.symbols.iter().enumerate() {
        let x = ((s.re + 1.25) / 2.5 * w as f32).round() as isize;
        let y = ((1.25 - s.im) / 2.5 * eye_h as f32).round() as isize;
        if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < eye_h {
            let ch = if i + 4 >= v.symbols.len() { '●' } else { '·' };
            grid[y as usize][x as usize] = ch;
        }
    }
    let mut lines: Vec<Line> = grid
        .into_iter()
        .map(|row| {
            Line::from(Span::styled(
                row.into_iter().collect::<String>(),
                Style::default().fg(Color::LightCyan),
            ))
        })
        .collect();

    if env_h > 0 && !v.env.is_empty() {
        const BARS: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        for row in 0..env_h {
            let from_bottom = env_h - 1 - row;
            let mut spans = Vec::with_capacity(w);
            for x in 0..w {
                let i = x * v.env.len() / w.max(1);
                let e = v.env.get(i).copied().unwrap_or(0.0);
                let eighths = (e.clamp(0.0, 1.0) * (env_h * 8) as f32) as usize;
                let cell = eighths.saturating_sub(from_bottom * 8).min(8);
                spans.push(Span::styled(
                    BARS[cell].to_string(),
                    Style::default().fg(Color::Magenta),
                ));
            }
            lines.push(Line::from(spans));
        }
    }
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_psk_tuner(f: &mut Frame, area: Rect, app: &App, view: Option<&PskView>) {
    let rf = app.tuned_freq();
    let lock = view.map(|v| v.lock_hz as f64).unwrap_or(0.0);
    let center = rf + lock;
    let title = format!(" tuner  {:.3} kHz ", center / 1000.0);
    let block = Block::default().borders(Borders::ALL).title(title);
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }
    let dim = Style::default().fg(Color::DarkGray);
    let mut lines: Vec<Line> = Vec::new();
    match view {
        None => lines.push(Line::from(Span::styled("no decoder", dim))),
        Some(v) => {
            let err = v.tune_err_hz;
            let err_col = if err.abs() < 1.0 {
                Color::LightGreen
            } else if err.abs() < 4.0 {
                Color::Yellow
            } else {
                Color::Red
            };
            lines.push(Line::from(vec![
                Span::styled("rf     ", dim),
                Span::styled(
                    format!("{:.3} kHz", center / 1000.0),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
            ]));
            lines.push(Line::from(vec![
                Span::styled("lock   ", dim),
                Span::styled(
                    format!("{:+.1} Hz", v.lock_hz),
                    Style::default().fg(if v.locked {
                        Color::LightCyan
                    } else {
                        Color::DarkGray
                    }),
                ),
            ]));
            lines.push(Line::from(vec![
                Span::styled("afc    ", dim),
                Span::styled(format!("{err:+.1} Hz"), Style::default().fg(err_col)),
            ]));
            lines.push(tune_meter(inner.width as usize, err));
            lines.push(Line::from(vec![
                Span::styled("q      ", dim),
                Span::styled(
                    format!("{:.0}%", v.quality * 100.0),
                    Style::default().fg(Color::White),
                ),
                Span::styled(
                    format!("   rev {:.0}%", v.reversals * 100.0),
                    dim,
                ),
            ]));
            lines.push(Line::from(Span::styled(
                "u/i fine   g centre   n next",
                dim,
            )));
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled("signals", dim)));
            let passband = !v.hits.is_empty();
            let mut listed: Vec<(f32, f32, bool)> = if passband {
                v.hits
                    .iter()
                    .map(|h| {
                        (
                            h.offset_hz,
                            h.quality,
                            (h.offset_hz - v.lock_hz).abs() < 8.0,
                        )
                    })
                    .collect()
            } else {
                app.psk_hits
                    .iter()
                    .map(|h| (h.offset_hz, h.quality, false))
                    .collect()
            };
            listed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let skip = app.st_scroll.min(listed.len().saturating_sub(1));
            for (off, q, on) in listed.into_iter().skip(skip).take(8) {
                let abs = if passband {
                    (rf + off as f64) / 1000.0
                } else {
                    (app.center + off as f64) / 1000.0
                };
                let mark = if on { '>' } else { ' ' };
                let style = if on {
                    Style::default().fg(Color::LightCyan)
                } else {
                    Style::default()
                };
                lines.push(Line::from(Span::styled(
                    format!("{mark}{abs:8.3}  q={:.0}%", q * 100.0),
                    style,
                )));
            }
        }
    }
    f.render_widget(Paragraph::new(lines), inner);
}

/// FT8/FT4 replace the plain decode pane with three views of the same decodes:
/// an activity map of which audio frequencies carried decodes in recent slots,
/// the raw message transcript, and the stations heard in the last 5 minutes.
fn draw_ft(f: &mut Frame, area: Rect, app: &App) {
    let cols = ft_cols(area);
    draw_ft_activity(f, cols[0], app);
    draw_ft_messages(f, cols[1], app);
    draw_ft_stations(f, cols[2], app);
}

/// Slot-by-slot map of decoded audio frequencies, newest slot first. A QSO
/// shows up as two frequencies lighting up on alternate rows, and the colour
/// of a cell carries the SNR. The axis along the bottom reads absolute RF
/// (dial + audio offset).
fn draw_ft_activity(f: &mut Frame, area: Rect, app: &App) {
    let dial = app.tuned_freq();
    let (f_lo, f_hi) = (decoders::ft8::FREQ_MIN, decoders::ft8::FREQ_MAX);
    let mut title = format!(
        " activity {}  {:.1}-{:.1} kHz ",
        app.mode.label(),
        (dial + f_lo as f64) / 1000.0,
        (dial + f_hi as f64) / 1000.0
    );
    if app.act_scroll > 0 {
        title = format!(" activity {} ({} slots back) ", app.mode.label(), app.act_scroll);
    }
    let block = Block::default().borders(Borders::ALL).title(title);
    let inner = block.inner(area);
    f.render_widget(block, area);
    const LABEL: usize = 8; // ">" marker + hhmmss + space
    if inner.width as usize <= LABEL || inner.height == 0 {
        return;
    }
    // The last row is the frequency axis; slots get the rest.
    let has_axis = inner.height > 1;
    let rows = if has_axis {
        inner.height as usize - 1
    } else {
        1
    };
    let w = inner.width as usize - LABEL;

    // Distinct slot stamps, newest first — one row per slot, skipping
    // `act_scroll` slots back in time when the pane is scrolled.
    let mut stamps: Vec<&str> = Vec::with_capacity(rows + app.act_scroll);
    for m in app.ft_msgs.iter().rev() {
        if stamps.last().copied() != Some(m.stamp.as_str()) {
            stamps.push(m.stamp.as_str());
            if stamps.len() == rows + app.act_scroll {
                break;
            }
        }
    }
    let stamps = &stamps[app.act_scroll.min(stamps.len())..];

    let mut lines = Vec::with_capacity(rows);
    for (r, stamp) in stamps.iter().enumerate() {
        let mut spans = Vec::with_capacity(LABEL + w);
        // '>' marks the live slot; hidden while scrolled back.
        let marker = if r == 0 && app.act_scroll == 0 { '>' } else { ' ' };
        spans.push(Span::styled(
            format!("{marker}{stamp} "),
            Style::default().fg(Color::DarkGray),
        ));
        for x in 0..w {
            let lo = f_lo + x as f32 * (f_hi - f_lo) / w as f32;
            let hi = f_lo + (x + 1) as f32 * (f_hi - f_lo) / w as f32;
            let best = app
                .ft_msgs
                .iter()
                .filter(|m| m.stamp == *stamp && m.freq_hz >= lo && m.freq_hz < hi)
                .map(|m| m.snr_db)
                .fold(f32::MIN, f32::max);
            if best > f32::MIN {
                // -24 dB .. +16 dB mapped onto the heat scale.
                let norm = ((best + 24.0) / 40.0).clamp(0.0, 1.0);
                spans.push(Span::styled("█", Style::default().fg(heat(norm))));
            } else {
                spans.push(Span::raw(" "));
            }
        }
        lines.push(Line::from(spans));
    }
    if has_axis {
        // Absolute RF ticks every 500 Hz of audio offset.
        let mut cells = vec![' '; LABEL + w];
        let mut end = 0usize;
        let mut tick = 500.0f32;
        while tick < f_hi {
            let col = LABEL + ((tick - f_lo) / (f_hi - f_lo) * w as f32) as usize;
            if col >= end && col < LABEL + w {
                cells[col] = '┬';
                let label = format!("{:.1}", (dial + tick as f64) / 1000.0);
                if col + 1 + label.len() <= LABEL + w {
                    for (i, ch) in label.chars().enumerate() {
                        cells[col + 1 + i] = ch;
                    }
                    end = col + 1 + label.len() + 1;
                } else {
                    end = LABEL + w;
                }
            }
            tick += 500.0;
        }
        lines.push(Line::from(Span::styled(
            cells.into_iter().collect::<String>(),
            Style::default().fg(Color::DarkGray),
        )));
    }
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_ft_messages(f: &mut Frame, area: Rect, app: &App) {
    let mut title = " messages ".to_string();
    if app.msg_scroll > 0 {
        title = format!(" messages (scrolled up {}) ", app.msg_scroll);
    }
    let block = Block::default().borders(Borders::ALL).title(title);
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.height < 2 || inner.width == 0 {
        return;
    }
    let rows = inner.height as usize - 1; // one row for the column header
    app.msg_rows.set(rows);
    let w = inner.width as usize;

    // One structured line per decode: clear UTC time, SNR, timing offset and
    // the absolute RF frequency (dial + audio), not just the audio offset.
    const PREFIX: usize = 8 + 1 + 4 + 1 + 4 + 1 + 9 + 1; // time snr dt rf
    let total = app.ft_msgs.len();
    let max = total.saturating_sub(rows);
    let start = total.saturating_sub(rows + app.msg_scroll.min(max));
    let dial = app.tuned_freq();
    let live_stamp = app.ft_msgs.back().map(|m| m.stamp.as_str()).unwrap_or("");

    let mut lines = Vec::with_capacity(rows);
    for m in app.ft_msgs.iter().skip(start).take(rows) {
        let t = &m.stamp;
        let time = format!(
            "{}:{}:{}",
            t.get(0..2).unwrap_or("??"),
            t.get(2..4).unwrap_or("??"),
            t.get(4..6).unwrap_or("??")
        );
        let rf = (dial + m.freq_hz as f64) / 1000.0;
        let avail = w.saturating_sub(PREFIX);
        let text: String = if m.text.chars().count() > avail && avail > 1 {
            format!("{}…", m.text.chars().take(avail - 1).collect::<String>())
        } else {
            m.text.clone()
        };
        let mut style = Style::default();
        if m.text.starts_with("CQ") {
            style = style.fg(Color::Cyan);
        }
        // The newest slot stands out while pinned to live.
        if app.msg_scroll == 0 && m.stamp == live_stamp {
            style = style.add_modifier(Modifier::BOLD);
        }
        lines.push(Line::from(Span::styled(
            format!("{time} {:>4.0} {:+4.1} {rf:>9.3} {text}", m.snr_db, m.dt_sec),
            style,
        )));
    }
    // Scan results land in the raw transcript rather than coming through the
    // decoder; keep them visible under the structured decodes.
    if app.msg_scroll == 0 {
        for line in app.text.lines().filter(|l| {
            !l.is_empty()
                && !l.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false)
        }) {
            lines.push(Line::from(Span::styled(
                line.to_string(),
                Style::default().fg(Color::DarkGray),
            )));
        }
    }
    // Bottom-align, chat-style: blank rows go at the top.
    let pad = rows.saturating_sub(lines.len());
    let mut body: Vec<Line> = vec![Line::from(Span::styled(
        "time      snr   dt       kHz message",
        Style::default().fg(Color::DarkGray),
    ))];
    body.extend((0..pad).map(|_| Line::from("")));
    body.extend(lines.into_iter().take(rows));
    f.render_widget(Paragraph::new(body), inner);
}

/// One entry per callsign heard. Rows live in `App::stations` in first-heard
/// order; `last_secs` (seconds of day) is converted to an age at draw time.
struct Station {
    snr: f32,       // best SNR seen
    last_secs: i64, // slot stamp of the most recent decode
    count: u32,     // decodes total
    cq: bool,       // was calling CQ
    freq: f32,      // audio Hz of the most recent decode
}

/// Seconds since the Unix epoch.
fn utc_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Current UTC time as hh:mm:ss, for stamping lines of copy.
fn utc_stamp() -> String {
    let s = utc_secs() as i64 % 86400;
    format!("{:02}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
}

/// hhmmss (as stamped on a slot) to seconds of day.
fn stamp_secs(stamp: &str) -> i64 {
    let field = |r: std::ops::Range<usize>| stamp.get(r).and_then(|s| s.parse::<i64>().ok()).unwrap_or(0);
    field(0..2) * 3600 + field(2..4) * 60 + field(4..6)
}

fn draw_ft_stations(f: &mut Frame, area: Rect, app: &App) {
    let now = utc_secs() as i64 % 86400;
    let mut title = format!(" stations ({}) ", app.stations.len());
    if app.st_scroll > 0 {
        title = format!(" stations ({}, skipped {}) ", app.stations.len(), app.st_scroll);
    }
    let block = Block::default().borders(Borders::ALL).title(title);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let rows = inner.height as usize;

    let mut lines = Vec::with_capacity(rows);
    lines.push(Line::from(Span::styled(
        format!("{:<10} {:>4} {:>4} {:>8}", "call", "snr", "age", "kHz"),
        Style::default().fg(Color::DarkGray),
    )));
    let max = app.stations.len().saturating_sub(rows.saturating_sub(1));
    let dial = app.tuned_freq();
    for (call, s) in app
        .stations
        .iter()
        .skip(app.st_scroll.min(max))
        .take(rows.saturating_sub(1))
    {
        let age = (now - s.last_secs).rem_euclid(86400);
        let age_str = if age < 120 {
            format!("{age}s")
        } else {
            format!("{}m{:02}", age / 60, age % 60)
        };
        let rf = (dial + s.freq as f64) / 1000.0;
        let call: String = call.chars().take(10).collect();
        // CQ callers stand out; stations silent for over five minutes fade.
        let style = if s.cq {
            Style::default().fg(Color::Cyan)
        } else if age > 300 {
            Style::default().fg(Color::DarkGray)
        } else {
            Style::default()
        };
        lines.push(Line::from(Span::styled(
            format!("{call:<10} {:>4.0} {:>4} {rf:>8.2}", s.snr, age_str),
            style,
        )));
    }
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_help(f: &mut Frame, area: Rect) {
    let text = vec![
        Line::from("  hfscan key bindings"),
        Line::from(""),
        Line::from("  ← →         tune by one step   (shift: 10 steps)"),
        Line::from("  ↑ / ↓       scroll the decode transcript (shift: 10 lines)"),
        Line::from("  V           auto decode view: held rows / chronological log"),
        Line::from("              rows hold one signal each, copy building in place;"),
        Line::from("              signals that go quiet sink to the bottom of the list"),
        Line::from("  wheel       scroll the pane under the mouse (waterfall too)"),
        Line::from("  z / Z       zoom in / out — also sets the tuning step"),
        Line::from("  n / N       next / previous signal (CW/PSK31: confirmed)"),
        Line::from("  p           CW/PSK31: lock next in span, or scan the band"),
        Line::from("  u / i       CW/PSK31: fine-tune lock −2 / +2 Hz"),
        Line::from("  g           CW/PSK31: centre the cursor on the lock"),
        Line::from("  [ ]         retune centre ±10 kHz"),
        Line::from("  PgUp/PgDn   retune centre ± half span"),
        Line::from("  c           centre the radio on the cursor"),
        Line::from("  b / B       next / previous band preset"),
        Line::from("  d           decoder: off → CW → RTTY → PSK31 → FT8 → FT4 → AUTO"),
        Line::from("              AUTO decodes every digital signal in the span at once —"),
        Line::from("              FT8/FT4 on their calling frequencies, CW/RTTY/PSK31"),
        Line::from("              wherever found. Each line is tagged with its frequency."),
        Line::from("  r           force RTTY shift polarity (it is detected"),
        Line::from("              automatically; this overrides that)"),
        Line::from("              PSK31 auto-locks to a nearby carrier (AFC)"),
        Line::from("  s           scan the current band; results are labelled"),
        Line::from("              spectrum chips + bottom activity strip (heard)"),
        Line::from("  v           enlarge the decode pane (cycles sizes)"),
        Line::from("  w / W       waterfall speed / subcell resolution"),
        Line::from("  f / F       FFT size (frequency resolution)"),
        Line::from("  a           AGC: soft hang → hardware → off   + / - gain"),
        Line::from("  e           spectrum smoothing (light / medium / heavy)"),
        Line::from("  j           impulse blanker (off / gentle / normal / aggressive)"),
        Line::from("  l           RX bandpass (auto / 80 / 200 / 500 / 1.5k / 3k)"),
        Line::from("  k           squelch on/off  , / .   squelch threshold"),
        Line::from("  t           toggle bias-T (external preamp power)"),
        Line::from("  o           station settings (callsign, grid, spotting)"),
        Line::from("  x           clear the decode pane"),
        Line::from("  ? / q       toggle help / quit"),
    ];
    let w = 68.min(area.width.saturating_sub(4));
    let h = (text.len() as u16 + 2).min(area.height);
    let rect = Rect {
        x: area.x + (area.width.saturating_sub(w)) / 2,
        y: area.y + (area.height.saturating_sub(h)) / 2,
        width: w,
        height: h,
    };
    f.render_widget(ratatui::widgets::Clear, rect);
    f.render_widget(
        Paragraph::new(text).block(Block::default().borders(Borders::ALL).title(" help ")),
        rect,
    );
}

/// Blue → cyan → green → yellow → red → white heat map.
fn heat(v: f32) -> Color {
    let v = v.clamp(0.0, 1.0);
    let (r, g, b) = if v < 0.2 {
        let t = v / 0.2;
        (0.0, 0.0, 0.3 + 0.7 * t)
    } else if v < 0.4 {
        let t = (v - 0.2) / 0.2;
        (0.0, t, 1.0)
    } else if v < 0.6 {
        let t = (v - 0.4) / 0.2;
        (0.0, 1.0, 1.0 - t)
    } else if v < 0.8 {
        let t = (v - 0.6) / 0.2;
        (t, 1.0, 0.0)
    } else {
        let t = (v - 0.8) / 0.2;
        (1.0, 1.0 - 0.6 * t, 0.4 * t)
    };
    Color::Rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}


#[cfg(test)]
mod tests {
    use super::report::is_callsign;
    use super::{App, FtMessage, Mode};

    fn ftm(stamp: &str, snr: f32, freq: f32, text: &str) -> FtMessage {
        FtMessage {
            stamp: stamp.into(),
            snr_db: snr,
            dt_sec: 0.1,
            freq_hz: freq,
            text: text.into(),
        }
    }

    use super::{identify, MAX_AUTO_SLOTS};
    use num_complex::Complex32;
    use std::time::{Duration, Instant};

    fn ident(kind: identify::Kind, offset_hz: f32, snr_db: f32) -> identify::Ident {
        identify::Ident {
            offset_hz,
            bw_hz: 100.0,
            snr_db,
            kind,
            score: 0.8,
        }
    }

    /// FT8/FT4 always live on their calling frequencies, so Auto pins a
    /// decoder there rather than waiting for the classifier to localise a
    /// pile-up — and only where the whole 200–3000 Hz passband fits.
    #[test]
    fn auto_pins_the_ft_calling_frequencies_in_the_span() {
        // 192 kHz centred on 14.10 MHz covers both 14074 (FT8) and 14080 (FT4).
        let mut app = App::new(14_100_000.0, 192_000.0, Mode::Auto);
        app.reconcile_auto();
        let dials: Vec<f64> = app.auto.iter().map(|s| s.dial_hz).collect();
        assert!(
            dials.iter().any(|d| (*d - 14_074_000.0).abs() < 1.0),
            "FT8 calling frequency not covered: {dials:?}"
        );
        assert!(
            dials.iter().any(|d| (*d - 14_080_000.0).abs() < 1.0),
            "FT4 calling frequency not covered: {dials:?}"
        );
        // 20m PSK at 14070 is not an FT mode and must not be pinned.
        assert!(app.auto.iter().all(|s| s.kind != identify::Kind::Psk31));
        // Pinned slots survive a quiet band; that is the point of pinning.
        app.reconcile_auto();
        assert!(app.auto.iter().all(|s| s.pinned));
    }

    /// A marker that sits too close to the edge of the span cannot have its
    /// whole passband received, so decoding it would just waste a slot.
    #[test]
    fn auto_skips_an_ft_frequency_hanging_off_the_edge() {
        // Centre so 14074 is only 1 kHz inside a 192 kHz span.
        let app_lo = 14_074_000.0 - 96_000.0 + 1_000.0;
        let mut app = App::new(app_lo, 192_000.0, Mode::Auto);
        app.reconcile_auto();
        assert!(
            app.auto.iter().all(|s| (s.dial_hz - 14_074_000.0).abs() > 1.0),
            "took an FT8 slot whose passband runs off the span"
        );
    }

    /// Narrowband signals come and go; their slots have to follow.
    #[test]
    fn auto_tracks_then_retires_a_narrowband_signal() {
        let mut app = App::new(7_030_000.0, 192_000.0, Mode::Auto);
        app.idents = vec![ident(identify::Kind::Cw, 3000.0, 20.0)];
        app.reconcile_auto();
        let cw: Vec<_> = app
            .auto
            .iter()
            .filter(|s| s.kind == identify::Kind::Cw)
            .collect();
        assert_eq!(cw.len(), 1, "expected one CW decoder");
        assert!((cw[0].dial_hz - 7_033_000.0).abs() < 1.0);
        assert!(!cw[0].pinned);

        // Seen again a moment later at a slightly different estimate: same
        // signal, same decoder, not a second one.
        app.idents = vec![ident(identify::Kind::Cw, 3040.0, 19.0)];
        app.reconcile_auto();
        assert_eq!(
            app.auto.iter().filter(|s| s.kind == identify::Kind::Cw).count(),
            1,
            "a wobbling estimate must not spawn duplicate decoders"
        );

        // Gone from the classifier, and aged past the idle window.
        app.idents.clear();
        for s in &mut app.auto {
            s.last_seen = Instant::now() - super::AUTO_IDLE - Duration::from_secs(1);
        }
        app.reconcile_auto();
        assert!(
            app.auto.iter().all(|s| s.kind != identify::Kind::Cw),
            "a signal that stopped should release its slot"
        );
    }

    /// A crowded band must not spawn decoders without limit, and the slots
    /// it does spend should go to the strongest signals.
    #[test]
    fn auto_caps_slots_and_prefers_strong_signals() {
        let mut app = App::new(7_100_000.0, 192_000.0, Mode::Auto);
        app.idents = (0..40)
            .map(|i| ident(identify::Kind::Cw, -80_000.0 + i as f32 * 4000.0, i as f32))
            .collect();
        app.reconcile_auto();
        assert!(
            app.auto.len() <= MAX_AUTO_SLOTS,
            "spawned {} decoders, cap is {MAX_AUTO_SLOTS}",
            app.auto.len()
        );
        // The strongest ident was the last one (snr 39), so it must have a slot.
        let strongest = 7_100_000.0 + (-80_000.0 + 39.0 * 4000.0);
        assert!(
            app.auto.iter().any(|s| (s.dial_hz - strongest).abs() < 200.0),
            "cap dropped the strongest signal"
        );
    }

    /// Every automatic slot is tuned relative to the radio centre, so a
    /// retune has to tear the fleet down rather than leave it pointing at
    /// frequencies that no longer exist.
    #[test]
    fn auto_slot_labels_carry_the_frequency_and_mode() {
        let mut app = App::new(7_030_000.0, 192_000.0, Mode::Auto);
        app.decode_log.push_back(super::DecodeEntry {
            stamp: "12:34:56".into(),
            dial_hz: 7_033_250.0,
            kind: identify::Kind::Cw,
            mode: "CW",
            text: "CQ DE W1AW".into(),
        });
        let rect = ratatui::layout::Rect::new(0, 0, 80, 4);
        let line = &super::decode_log_lines(&app, rect)[0];
        let rendered: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(
            rendered.contains("7033.2") || rendered.contains("7033.3"),
            "{rendered:?}"
        );
        assert!(rendered.contains("CW"), "{rendered:?}");
        assert!(rendered.contains("12:34:56"), "{rendered:?}");
        assert!(rendered.contains("CQ DE W1AW"), "{rendered:?}");
    }

    /// Keyed CW at `tone_hz` off the span centre, sending `text` at `wpm`.
    fn cw_iq(text: &str, wpm: f32, tone_hz: f64, fs: f64, secs: f64) -> Vec<Complex32> {
        fn code(c: char) -> &'static str {
            match c {
                'A' => ".-", 'B' => "-...", 'C' => "-.-.", 'D' => "-..", 'E' => ".",
                'G' => "--.", 'K' => "-.-", 'N' => "-.", 'O' => "---", 'Q' => "--.-",
                'R' => ".-.", 'S' => "...", 'T' => "-", 'W' => ".--",
                _ => "",
            }
        }
        let dit = (1.2 / wpm as f64 * fs) as usize;
        let mut key: Vec<bool> = Vec::new();
        while key.len() < (fs * secs) as usize {
            for ch in text.chars() {
                if ch == ' ' {
                    key.extend(std::iter::repeat_n(false, dit * 4));
                    continue;
                }
                for el in code(ch).chars() {
                    let n = if el == '-' { dit * 3 } else { dit };
                    key.extend(std::iter::repeat_n(true, n));
                    key.extend(std::iter::repeat_n(false, dit));
                }
                key.extend(std::iter::repeat_n(false, dit * 2));
            }
        }
        // Shape the key envelope over ~5 ms. A real transmitter does this;
        // hard on/off edges splatter across the band and the span classifier
        // would rightly report the clicks as more CW signals.
        //
        // The band noise matters just as much: with a noiseless synthetic
        // signal the spectrum median sits at the f32 leakage floor, and a
        // median-relative detector then finds "signals" across the whole
        // span. No receiver ever sees that.
        let rise = (0.005 * fs) as usize;
        let mut phase = 0.0f64;
        let step = 2.0 * std::f64::consts::PI * tone_hz / fs;
        let mut env = 0.0f32;
        let mut rng = 0x2545F491u32;
        key.iter()
            .map(|&on| {
                let target = if on { 1.0 } else { 0.0 };
                let k = 1.0 / rise.max(1) as f32;
                env += (target - env).clamp(-k, k);
                let mut noise = || {
                    rng ^= rng << 13;
                    rng ^= rng >> 17;
                    rng ^= rng << 5;
                    (rng as f32 / u32::MAX as f32 - 0.5) * 0.004
                };
                let s = Complex32::from_polar(0.05 * env, phase as f32)
                    + Complex32::new(noise(), noise());
                phase += step;
                s
            })
            .collect()
    }

    /// The whole automatic path on a real signal: the classifier points a
    /// slot at a frequency, the slot's own chain and decoder run over raw
    /// span IQ, and what comes out lands in the transcript tagged with where
    /// it was heard. This is what the single-decoder tests cannot cover.
    #[test]
    fn auto_decodes_a_cw_signal_and_tags_it_with_the_frequency() {
        let fs = 192_000.0;
        let mut app = App::new(7_030_000.0, fs, Mode::Auto);
        app.agc = super::AgcMode::Off;
        // The classifier reports CW 3 kHz up; the tone sits 60 Hz off that,
        // inside the decoder's own search window, as a real one would.
        app.idents = vec![ident(identify::Kind::Cw, 3000.0, 25.0)];
        app.reconcile_auto();
        // This span also covers the 40m FT4 and FT8 calling frequencies, so
        // those get pinned decoders too — they simply have nothing to hear.
        assert_eq!(
            app.auto.iter().filter(|s| s.kind == identify::Kind::Cw).count(),
            1,
            "expected exactly one CW decoder, got {:?}",
            app.auto.iter().map(|s| s.dial_hz).collect::<Vec<_>>()
        );

        let mut iq = cw_iq("CQ CQ DE W1AW ", 20.0, 3060.0, fs, 14.0);
        // Then the station stops. The trailing quiet is what flushes the
        // last partial line, exactly as a real band would.
        iq.extend(cw_iq(" ", 20.0, 3060.0, fs, 3.0));
        let mut out = Vec::new();
        let mut spec = super::Spectrum::new(4096);
        for block in iq.chunks(16_384) {
            app.feed(block, &mut spec, &mut out);
        }

        assert!(
            !app.decode_log.is_empty(),
            "auto mode decoded nothing from a clean CW signal"
        );
        // Every entry carries the frequency and mode it came from, and its
        // copy is printable — nothing that could corrupt the terminal grid.
        for e in &app.decode_log {
            assert!(e.dial_hz > 0.0 && !e.mode.is_empty(), "untagged entry");
            assert!(
                e.text.chars().all(|c| (' '..='~').contains(&c)),
                "unprintable copy reached the transcript: {:?}",
                e.text
            );
        }
        let ours: String = app
            .decode_log
            .iter()
            .filter(|e| (e.dial_hz - 7_033_000.0).abs() < 100.0)
            .map(|e| e.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(
            !ours.is_empty(),
            "nothing from the signal's own frequency; heard {:?}",
            app.decode_log.iter().map(|e| e.dial_hz).collect::<Vec<_>>()
        );
        assert!(
            ours.contains("CQ") || ours.contains("W1AW") || ours.contains("DE"),
            "expected recognisable morse from 7033.0, got {ours:?}"
        );
    }

    /// The quadrant renderer is only worth its extra detail if the glyph it
    /// picks actually matches which quarters are hot; get the mapping wrong
    /// and a carrier lands in the wrong corner of the cell.
    #[test]
    fn quadrant_glyph_matches_the_hot_quarters() {
        // [upper-left, upper-right, lower-left, lower-right]
        let cases = [
            ([1.0, 0.0, 0.0, 0.0], '▘'),
            ([0.0, 1.0, 0.0, 0.0], '▝'),
            ([0.0, 0.0, 1.0, 0.0], '▖'),
            ([0.0, 0.0, 0.0, 1.0], '▗'),
            ([1.0, 1.0, 0.0, 0.0], '▀'),
            ([0.0, 0.0, 1.0, 1.0], '▄'),
            ([1.0, 0.0, 1.0, 0.0], '▌'),
            ([0.0, 1.0, 0.0, 1.0], '▐'),
            ([1.0, 0.0, 0.0, 1.0], '▚'),
            ([0.0, 1.0, 1.0, 0.0], '▞'),
            ([1.0, 1.0, 1.0, 0.0], '▛'),
            ([1.0, 1.0, 0.0, 1.0], '▜'),
            ([1.0, 0.0, 1.0, 1.0], '▙'),
            ([0.0, 1.0, 1.0, 1.0], '▟'),
        ];
        for (v, want) in cases {
            let (ch, fg, bg) = super::quad_cell(v);
            assert_eq!(ch, want, "wrong glyph for {v:?}");
            assert!(fg > bg, "hot quarters must take the foreground for {v:?}");
        }
    }

    /// A cell holds two colours however finely it is carved up, so the split
    /// has to fall on the biggest step in the data — otherwise a weak signal
    /// beside a strong one gets averaged into its neighbour.
    #[test]
    fn quadrant_split_follows_the_largest_step() {
        let (_, fg, bg) = super::quad_cell([0.90, 0.95, 0.10, 0.12]);
        assert!((fg - 0.925).abs() < 0.01, "foreground averaged the hot pair");
        assert!((bg - 0.11).abs() < 0.01, "background averaged the cold pair");
    }

    #[test]
    fn callsign_heuristic() {
        for call in ["K1ABC", "W1AW", "JA1ABC", "G0ABC", "3D2AG", "K1ABC/P", "VE3XYZ"] {
            assert!(is_callsign(call), "{call} should be a callsign");
        }
        for not in [
            "CQ", "DE", "73", "RRR", "RR73", "FN42", "PM95", "IO91", "-12", "+03", "R-05",
            "R+10",
        ] {
            assert!(!is_callsign(not), "{not} should not be a callsign");
        }
    }

    #[test]
    fn labels_hold_across_a_missed_classify() {
        let mut app = App::new(14_070_000.0, 192_000.0, Mode::Off);
        let hit = super::identify::Ident {
            offset_hz: 4000.0,
            bw_hz: 80.0,
            snr_db: 14.0,
            kind: super::identify::Kind::Cw,
            score: 0.8,
        };
        super::apply_idents(&mut app, vec![hit]);
        assert_eq!(app.tracks.len(), 1);
        assert_eq!(app.idents.len(), 1);
        super::apply_idents(&mut app, Vec::new());
        assert_eq!(app.tracks.len(), 1, "a miss must not drop the track");
        assert_eq!(app.idents.len(), 1, "chip stays visible through the hold");
        let a = app.tracks[0].alpha(std::time::Instant::now());
        assert!(a > 0.7, "just-missed label should still be bright, got {a}");
    }

    #[test]
    fn heard_lingers_after_live_idents_drop() {
        let mut app = App::new(14_070_000.0, 192_000.0, Mode::Off);
        app.idents = vec![super::identify::Ident {
            offset_hz: 4000.0,
            bw_hz: 80.0,
            snr_db: 14.0,
            kind: super::identify::Kind::Cw,
            score: 0.8,
        }];
        super::merge_heard(&mut app);
        assert_eq!(app.heard.len(), 1);
        assert!(
            app.notes
                .iter()
                .any(|n| n.text.contains("CW") && n.text.contains("14074")),
            "first sighting should land in the activity log: {:?}",
            app.notes.iter().map(|n| n.text.as_str()).collect::<Vec<_>>()
        );
        let notes = app.notes.len();
        super::merge_heard(&mut app);
        assert_eq!(app.heard.len(), 1, "same signal must not duplicate");
        assert_eq!(app.notes.len(), notes, "re-detect must not spam the log");
        app.idents.clear();
        assert_eq!(app.heard.len(), 1, "memory stays after the live chip fades");
    }

    /// Stations must keep their first-heard order as decodes arrive, so the
    /// list updates in place instead of reshuffling.
    #[test]
    fn stations_update_in_place() {
        let mut app = App::new(14_074_000.0, 192_000.0, Mode::Ft8);
        app.update_stations(&ftm("120000", -8.0, 1500.0, "CQ K1ABC FN42"));
        app.update_stations(&ftm("120015", 3.0, 2100.0, "W9XYZ K1ABC R-05"));
        app.update_stations(&ftm("120015", -15.0, 800.0, "CQ JA1ABC PM95"));
        app.update_stations(&ftm("120030", -2.0, 1500.0, "K1ABC W9XYZ 73"));

        let calls: Vec<&str> = app.stations.iter().map(|(c, _)| c.as_str()).collect();
        assert_eq!(calls, ["K1ABC", "W9XYZ", "JA1ABC"]);
        let k1abc = &app.stations[0].1;
        assert_eq!(k1abc.count, 3);
        assert_eq!(k1abc.snr, 3.0); // best SNR wins
        assert_eq!(k1abc.freq, 1500.0); // most recent decode
        assert!(k1abc.cq);
    }

    /// The FT panes must not panic on any terminal size, with decodes present
    /// or not. Layout math lives on the render path, so exercise it directly.
    #[test]
    fn ft_panes_render_without_panicking() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        for mode in [Mode::Ft8, Mode::Ft4] {
            let mut app = App::new(14_074_000.0, 192_000.0, mode);
            for m in [
                ftm("123015", -2.0, 2100.0, "K1ABC W9XYZ -12"),
                ftm("123030", 3.0, 1500.0, "W9XYZ K1ABC R-05"),
                ftm("123045", -8.0, 1500.0, "CQ K1ABC FN42"),
                ftm("123045", -15.0, 2100.0, "CQ JA1ABC PM95"),
            ] {
                app.update_stations(&m);
                app.text.push_str(&m.format());
                app.text.push('\n');
                app.ft_msgs.push_back(m);
            }
            for (w, h) in [(80u16, 24u16), (40, 12), (150, 50)] {
                let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
                t.draw(|f| super::draw(f, &app)).unwrap();
            }
            // The enlarged decode pane layouts must render too.
            app.decode_zoom = 1;
            let mut t = Terminal::new(TestBackend::new(100, 30)).unwrap();
            t.draw(|f| super::draw(f, &app)).unwrap();
            app.decode_zoom = 2;
            t.draw(|f| super::draw(f, &app)).unwrap();
            // Scrolled panes must render too, in every waterfall resolution.
            app.decode_zoom = 0;
            app.msg_scroll = 3;
            app.st_scroll = 1;
            app.act_scroll = 1;
            app.wf_scroll = 5;
            for res in super::WfRes::ALL {
                app.wf_res = res;
                t.draw(|f| super::draw(f, &app)).unwrap();
            }
            // The settings dialog overlays everything.
            app.settings = Some(super::SettingsEdit {
                call: "K1ABC".into(),
                grid: "FN42".into(),
                field: 0,
            });
            t.draw(|f| super::draw(f, &app)).unwrap();
            app.settings = None;
        }
    }

    #[test]
    fn psk_panes_render_without_panicking() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let mut app = App::new(14_070_000.0, 192_000.0, Mode::Psk31);
        app.text = "CQ CQ DE TEST\n".into();
        for (w, h) in [(80u16, 24u16), (40, 12), (150, 50)] {
            let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
            t.draw(|f| super::draw(f, &app)).unwrap();
        }
        let mut t = Terminal::new(TestBackend::new(120, 30)).unwrap();
        t.draw(|f| super::draw(f, &app)).unwrap();
        let buf = t.backend().buffer();
        let mut text = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                text.push(buf[(x, y)].symbol().chars().next().unwrap_or(' '));
            }
            text.push('\n');
        }
        assert!(text.contains("eye"), "eye pane missing:\n{text}");
        assert!(text.contains("tuner"), "tuner pane missing:\n{text}");
        assert!(text.contains("copy"), "copy pane missing:\n{text}");
        assert!(text.contains("activity"), "activity strip missing:\n{text}");
    }

    #[test]
    fn cw_panes_render_without_panicking() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let mut app = App::new(14_026_000.0, 192_000.0, Mode::Cw);
        app.text = "CQ CQ DE W1AW K\n".into();
        for (w, h) in [(80u16, 24u16), (40, 12), (150, 50)] {
            let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
            t.draw(|f| super::draw(f, &app)).unwrap();
        }
        let mut t = Terminal::new(TestBackend::new(120, 30)).unwrap();
        t.draw(|f| super::draw(f, &app)).unwrap();
        let buf = t.backend().buffer();
        let mut text = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                text.push(buf[(x, y)].symbol().chars().next().unwrap_or(' '));
            }
            text.push('\n');
        }
        assert!(text.contains("envelope"), "envelope pane missing:\n{text}");
        assert!(text.contains("tuner"), "tuner pane missing:\n{text}");
        assert!(text.contains("copy"), "copy pane missing:\n{text}");
    }

    /// Scrolling the transcript shows older lines; zero stays pinned to live.
    /// A decoder fed noise emits arbitrary bytes. If an escape sequence or a
    /// zero-width character reaches the terminal the grid shifts, the frame's
    /// own borders land in the wrong columns, and the damage compounds frame
    /// after frame. Nothing that leaves the auto pane may be more or less
    /// than one cell wide.
    #[test]
    fn hostile_decoder_output_cannot_corrupt_the_grid() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let mut app = App::new(14_070_000.0, 192_000.0, Mode::Auto);
        let junk = "\u{1b}[2J\u{1b}[31mRED\u{7}\r\n\u{0}\u{200b}\u{feff}日本語\ttab";
        for i in 0..40 {
            app.decode_log.push_back(super::DecodeEntry {
                stamp: "12:34:56".into(),
                dial_hz: 14_070_000.0 + i as f64,
                kind: identify::Kind::Psk31,
                mode: "PSK",
                text: super::sanitize(junk),
            });
        }
        // And a line far longer than any pane is wide.
        app.decode_log.push_back(super::DecodeEntry {
            stamp: "12:34:57".into(),
            dial_hz: 14_074_000.0,
            kind: identify::Kind::Cw,
            mode: "CW",
            text: "X".repeat(500),
        });
        // The same, held in rows: what a decoder chewing on noise builds up.
        for i in 0..12 {
            let k = app.row_for(14_000_000.0 + i as f64 * 900.0, identify::Kind::Cw, "CW");
            app.rows[k].push_copy(&super::sanitize(&junk.repeat(9)));
        }
        app.rows[0].last_copy = super::Instant::now() - Duration::from_secs(300);

        for view in [super::AutoView::Rows, super::AutoView::Log] {
            app.auto_view = view;
            for (w, h) in [(80u16, 24u16), (40, 12), (200, 60)] {
                for scroll in [0usize, 7, 9999] {
                    app.msg_scroll = scroll;
                    let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
                    t.draw(|f| super::draw(f, &app)).unwrap();
                    let buf = t.backend().buffer();
                    for y in 0..h {
                        for x in 0..w {
                            let sym = buf[(x, y)].symbol();
                            assert_eq!(
                                sym.chars().count(),
                                1,
                                "cell ({x},{y}) is not one character: {sym:?}"
                            );
                            let c = sym.chars().next().unwrap();
                            assert!(
                                !c.is_control(),
                                "control character reached cell ({x},{y}): {c:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    fn row(app: &mut App, freq: f64, kind: identify::Kind, copy: &str, ago_secs: u64) {
        app.rows.push(super::SignalRow {
            dial_hz: freq,
            kind,
            mode: kind.label(),
            copy: copy.into(),
            last_copy: super::Instant::now() - Duration::from_secs(ago_secs),
        });
    }

    /// The point of a held row: copy from one signal accumulates in place
    /// instead of being flushed away a few characters at a time, and once it
    /// outgrows the pane the row shows the newest end of it.
    #[test]
    fn a_held_row_accumulates_copy_and_shows_the_newest_end() {
        let mut app = App::new(7_030_000.0, 192_000.0, Mode::Auto);
        let i = app.row_for(7_033_000.0, identify::Kind::Cw, "CW");
        for chunk in ["CQ ", "CQ ", "DE ", "W1AW ", "W1AW ", "K"] {
            app.rows[i].push_copy(chunk);
        }
        assert_eq!(app.rows[i].copy, "CQ CQ DE W1AW W1AW K");

        // The same signal drifting a little stays on its own row.
        let j = app.row_for(7_033_060.0, identify::Kind::Cw, "CW");
        assert_eq!(i, j, "a small drift started a second row");
        assert_eq!(app.rows.len(), 1);

        let text_of = |l: &super::Line| {
            l.spans.iter().map(|s| s.content.to_string()).collect::<String>()
        };
        // Copy wider than the pane shows its tail, marked as continuing.
        app.rows[0].push_copy(&"-".repeat(400));
        let rect = ratatui::layout::Rect::new(0, 0, 60, 4);
        let drawn = text_of(&super::signal_rows_lines(&app, rect)[0]);
        assert!(drawn.contains('…'), "no continuation mark: {drawn:?}");
        assert!(drawn.ends_with('-'), "not showing the newest copy: {drawn:?}");
        assert!(drawn.chars().count() <= 60, "row overran the pane: {drawn:?}");

        // The buffer stays bounded however long the station transmits.
        for _ in 0..50 {
            app.rows[0].push_copy(&"x".repeat(100));
        }
        assert!(app.rows[0].copy.chars().count() <= super::ROW_COPY_MAX);
    }

    /// A signal that stops sending must sink below the live ones rather than
    /// disappearing, and must eventually be forgotten.
    #[test]
    fn silent_rows_sink_to_the_bottom_then_retire() {
        let mut app = App::new(14_070_000.0, 192_000.0, Mode::Auto);
        row(&mut app, 14_083_000.0, identify::Kind::Rtty, "old", 120);
        row(&mut app, 14_074_000.0, identify::Kind::Ft8, "new", 0);
        row(&mut app, 14_030_000.0, identify::Kind::Cw, "recent", 30);
        row(&mut app, 14_040_000.0, identify::Kind::Cw, "live too", 5);
        row(&mut app, 14_025_000.0, identify::Kind::Cw, "ancient", 9_999);
        app.sort_rows();

        // The long-silent row is gone; the rest are live-first by frequency,
        // then silent by how recently they were last heard.
        let order: Vec<f64> = app.rows.iter().map(|r| r.dial_hz).collect();
        assert_eq!(
            order,
            vec![14_040_000.0, 14_074_000.0, 14_030_000.0, 14_083_000.0],
            "rows out of order"
        );

        // A row that goes quiet moves down; it is not dropped.
        app.rows[0].last_copy = super::Instant::now() - Duration::from_secs(60);
        app.sort_rows();
        assert!(
            app.rows.iter().any(|r| r.dial_hz == 14_040_000.0),
            "a row that went quiet was dropped instead of sinking"
        );
        assert_eq!(app.rows[0].dial_hz, 14_074_000.0, "live row not on top");
    }

    /// The roster anchors at its top row and the log at its newest line, so
    /// the stored offset counts in opposite directions. What the keys and the
    /// wheel ask for is a direction on screen, and up must mean up in both.
    #[test]
    fn both_auto_views_scroll_the_same_direction() {
        let mut app = App::new(14_070_000.0, 192_000.0, Mode::Auto);
        for i in 0..30 {
            row(&mut app, 14_000_000.0 + i as f64 * 1000.0, identify::Kind::Cw, "x", 0);
        }
        let rect = ratatui::layout::Rect::new(0, 0, 60, 5);
        let text_of = |l: &super::Line| {
            l.spans.iter().map(|s| s.content.to_string()).collect::<String>()
        };
        let _ = super::signal_rows_lines(&app, rect); // records the row count

        // The roster starts at its top row; scrolling down walks the list.
        assert!(text_of(&super::signal_rows_lines(&app, rect)[0]).contains("14000.0"));
        app.scroll_transcript(-3);
        assert!(text_of(&super::signal_rows_lines(&app, rect)[0]).contains("14003.0"));
        app.scroll_transcript(3);
        assert!(text_of(&super::signal_rows_lines(&app, rect)[0]).contains("14000.0"));
        // And it cannot be scrolled up past the top, or down past the end.
        app.scroll_transcript(50);
        assert_eq!(app.msg_scroll, 0);
        app.scroll_transcript(-500);
        assert_eq!(app.msg_scroll, app.rows.len() - 5);
        let tail = super::signal_rows_lines(&app, rect);
        assert_eq!(tail.len(), 5, "the last page must still be full");
        assert!(text_of(&tail[4]).contains("14029.0"));

        // The log counts the other way but answers the same directions: up
        // shows older lines, and the top clamps.
        app.auto_view = super::AutoView::Log;
        app.msg_scroll = 0;
        for i in 0..30 {
            app.push_decode_note(format!("note {i}"));
        }
        let _ = super::decode_log_lines(&app, rect);
        assert!(text_of(&super::decode_log_lines(&app, rect)[4]).contains("note 29"));
        app.scroll_transcript(3);
        assert!(text_of(&super::decode_log_lines(&app, rect)[4]).contains("note 26"));
        app.scroll_transcript(-3);
        assert!(text_of(&super::decode_log_lines(&app, rect)[4]).contains("note 29"));
        app.scroll_transcript(500);
        assert_eq!(app.msg_scroll, app.decode_log.len() - 5);
    }

    /// Scrolling the auto pane must never move content past its own border,
    /// and must come back to live in as many steps as it went up.
    #[test]
    fn auto_pane_scrolls_within_its_bounds() {
        let mut app = App::new(14_070_000.0, 192_000.0, Mode::Auto);
        app.auto_view = super::AutoView::Log;
        for i in 0..50 {
            app.decode_log.push_back(super::DecodeEntry {
                stamp: "00:00:00".into(),
                dial_hz: 14_070_000.0,
                kind: identify::Kind::Cw,
                mode: "CW",
                text: format!("line {i}"),
            });
        }
        let rect = ratatui::layout::Rect::new(0, 0, 60, 5);
        let text_of = |l: &super::Line| {
            l.spans.iter().map(|s| s.content.to_string()).collect::<String>()
        };

        let live = super::decode_log_lines(&app, rect);
        assert_eq!(live.len(), 5);
        assert!(text_of(live.last().unwrap()).ends_with("line 49"));
        // Nothing renders wider than the pane it is drawn into.
        for l in &live {
            assert!(text_of(l).chars().count() <= 60, "{:?}", text_of(l));
        }

        app.scroll_transcript(10);
        let up = super::decode_log_lines(&app, rect);
        assert!(text_of(up.last().unwrap()).ends_with("line 39"));

        // Scrolling past the top clamps, so one keypress-worth of down-scroll
        // moves rather than burning off a nonsense offset.
        for _ in 0..200 {
            app.scroll_transcript(10);
        }
        // The clamp stops exactly where the oldest line reaches the top row.
        assert_eq!(app.msg_scroll, app.decode_log.len() - 5);
        let top = super::decode_log_lines(&app, rect);
        assert!(text_of(top.first().unwrap()).ends_with("line 0"));
        app.scroll_transcript(-10);
        let back = super::decode_log_lines(&app, rect);
        assert!(
            text_of(back.first().unwrap()).ends_with("line 10"),
            "{:?}",
            text_of(back.first().unwrap())
        );
    }

    #[test]
    fn transcript_scrolls() {
        let mut app = App::new(14_074_000.0, 192_000.0, Mode::Off);
        app.text = (0..20)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let text_of = |l: &super::Line| {
            l.spans.iter().map(|s| s.content.to_string()).collect::<String>()
        };
        let live = super::transcript_lines(&app, 40, 5, false);
        assert_eq!(text_of(live.last().unwrap()), "line 19");
        let up = {
            app.msg_scroll = 5;
            super::transcript_lines(&app, 40, 5, false)
        };
        assert_eq!(text_of(up.last().unwrap()), "line 14");
        // Scrolling past the top clamps instead of panicking.
        app.msg_scroll = 9999;
        let top = super::transcript_lines(&app, 40, 5, false);
        assert_eq!(text_of(top.first().unwrap()), "line 0");
    }

    /// The FT messages pane shows clear UTC time and absolute RF frequency,
    /// not raw WSJT-X lines or bare audio offsets.
    #[test]
    fn ft_messages_show_time_and_rf() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let mut app = App::new(14_074_000.0, 192_000.0, Mode::Ft8);
        let m = ftm("123045", -8.0, 1500.0, "CQ K1ABC FN42");
        app.update_stations(&m);
        app.ft_msgs.push_back(m);
        let mut t = Terminal::new(TestBackend::new(120, 30)).unwrap();
        t.draw(|f| super::draw(f, &app)).unwrap();
        let buf = t.backend().buffer();
        let mut text = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                text.push(buf[(x, y)].symbol().chars().next().unwrap_or(' '));
            }
            text.push('\n');
        }
        // 14 074 000 Hz dial + 1 500 Hz audio = 14 075 500 Hz.
        assert!(text.contains("12:30:45"), "clear time missing:\n{text}");
        assert!(text.contains("14075.500"), "message RF missing:\n{text}");
        assert!(text.contains("14075.50"), "station RF missing:\n{text}");
        assert!(text.contains("14074.2"), "activity RF range missing:\n{text}");
    }
}



#[cfg(test)]
mod psk_auto_tests {
    use super::*;
    const PI_F: f32 = std::f32::consts::PI;

    /// Drive one span of IQ through auto mode, including the periodic passes
    /// that `feed` gates on wall-clock time — a test loop runs far faster
    /// than real time and would otherwise never trigger them.
    pub(super) fn run_auto(app: &mut App, iq: &[Complex32], fs: f64) {
        let mut spec = Spectrum::new(4096);
        let mut out = Vec::new();
        let per_pass = ((fs * 1.2) as usize / 16_384).max(1);
        for (i, block) in iq.chunks(16_384).enumerate() {
            app.feed(block, &mut spec, &mut out);
            if i % per_pass == per_pass - 1 {
                refresh_psk_hits(app);
                refresh_cw_hits(app);
                refresh_idents(app);
            }
        }
    }

    fn psk_span(offset: f64, db: f32, fs: f64, secs: f64) -> Vec<Complex32> {
        let n_total = (1.0f32 / 6.0).sqrt();
        let in_bw = n_total * (31.25 / fs as f32).sqrt();
        let scale = 10f32.powf(-db / 20.0) / in_bw;
        decoders::tests::gen_psk31_at("CQ CQ DE W1AW W1AW K ", fs, offset, scale, secs)
    }

    /// The whole reason a PSK31 signal on the band reaches the screen: the
    /// span has to be searched for it, an ident raised, a slot built, and the
    /// slot's own chain and decoder have to copy it. Each piece is tested
    /// alone elsewhere; only together do they answer the question.
    ///
    /// Before the narrowband scouts ran in auto mode this needed 15 dB —
    /// because the span classifier only sees what the occupancy detector
    /// hands it, and that needs 8 dB *in a 47 Hz bin*, which a 31 Hz signal
    /// cannot reach until well above where it decodes fine.
    #[test]
    fn psk31_reaches_the_screen_in_auto_mode() {
        let fs = 192_000.0f64;
        let iq = psk_span(1_500.0, 10.0, fs, 14.0);
        let mut app = App::new(14_070_000.0, fs, Mode::Auto);
        app.agc = AgcMode::Off;
        run_auto(&mut app, &iq, fs);

        assert!(
            app.auto.iter().any(|s| s.kind == identify::Kind::Psk31),
            "no PSK31 slot for a 10 dB signal; idents were {:?}",
            app.idents.iter().map(|i| i.kind.label()).collect::<Vec<_>>()
        );
        let copy: String = app
            .rows
            .iter()
            .filter(|r| r.kind == identify::Kind::Psk31)
            .map(|r| r.copy.clone())
            .collect();
        assert!(
            copy.contains("W1AW"),
            "a 10 dB PSK31 signal produced no readable copy: {copy:?}"
        );
    }

    /// Making weak signals visible must not make imaginary ones visible.
    #[test]
    fn auto_mode_invents_no_narrowband_signals_from_noise() {
        let fs = 192_000.0f64;
        let mut rng = 0x1234_5678u32;
        let mut noise = || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            (rng as f32 / u32::MAX as f32) - 0.5
        };
        let iq: Vec<Complex32> = (0..(fs * 14.0) as usize)
            .map(|_| Complex32::new(noise(), noise()) * 0.3)
            .collect();
        let mut app = App::new(14_070_000.0, fs, Mode::Auto);
        app.agc = AgcMode::Off;
        run_auto(&mut app, &iq, fs);

        let phantom: Vec<&str> = app
            .idents
            .iter()
            .filter(|i| matches!(i.kind, identify::Kind::Psk31 | identify::Kind::Cw))
            .map(|i| i.kind.label())
            .collect();
        assert!(phantom.is_empty(), "noise classified as {phantom:?}");
        let chars: usize = app
            .rows
            .iter()
            .map(|r| r.copy.chars().filter(|c| !c.is_whitespace()).count())
            .sum();
        assert!(chars <= 4, "noise produced {chars} characters of copy");
    }

    /// A receiver's own DC artefact: the constant bias every zero-IF front
    /// end leaves at the LO, plus the phase-noise skirt around it. Both are
    /// ordinary; neither is in a synthetic span unless it is put there, which
    /// is why every measurement so far has been flattering near the centre.
    fn add_dc_artifact(iq: &mut [Complex32], fs: f64, spike_amp: f32) {
        if spike_amp <= 0.0 {
            return;
        }
        let mut rng = 0x51ed_2700u32;
        let mut n = || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            (rng as f32 / u32::MAX as f32) - 0.5
        };
        // Phase noise: a slow random walk on a carrier sitting at the LO.
        let mut ph = 0.0f32;
        let mut drift = 0.0f32;
        for (i, s) in iq.iter_mut().enumerate() {
            drift = 0.9995 * drift + 0.0005 * n() * 40.0;
            ph += 2.0 * PI_F * drift / fs as f32;
            *s += Complex32::new(spike_amp, 0.0); // DC offset
            *s += Complex32::from_polar(spike_amp * 0.5, ph); // LO skirt
            let _ = i;
        }
    }

    #[test]
    #[ignore]
    fn bench_psk31_vs_dc_offset() {
        let fs = 192_000.0f64;
        let n_total = (1.0f32 / 6.0).sqrt();
        let in_bw = n_total * (31.25 / fs as f32).sqrt();
        let scale = 10f32.powf(-15.0 / 20.0) / in_bw;
        println!("\nPSK31 at 15 dB, by distance from the LO and size of the LO artefact");
        println!("(artefact amplitude is relative to the wanted signal's)");
        println!("{:>12}{:>10}{:>10}{:>10}", "offset", "clean", "1x", "5x");
        // Offsets chosen clear of the FT8/FT4 windows above 14.074, which
        // veto narrowband classification for their own good reasons.
        for off in [200.0f64, 400.0, 800.0, 1_500.0, 2_500.0, 3_500.0, 20_000.0] {
            let base = decoders::tests::gen_psk31_at(
                "CQ CQ DE W1AW W1AW K ", fs, off, scale, 12.0);
            let mut row = Vec::new();
            for spike in [0.0f32, 1.0, 5.0] {
                let mut iq = base.clone();
                add_dc_artifact(&mut iq, fs, spike);
                let mut app = App::new(14_070_000.0, fs, Mode::Auto);
                app.agc = AgcMode::Off;
                run_auto(&mut app, &iq, fs);
                let copy: String = app
                    .rows
                    .iter()
                    .filter(|r| r.kind == identify::Kind::Psk31)
                    .map(|r| r.copy.clone())
                    .collect();
                row.push(if copy.contains("W1AW") { "copy" } else { "--" });
            }
            println!(
                "{:>12}{:>10}{:>10}{:>10}",
                format!("{:+.0} Hz", off),
                row[0], row[1], row[2]
            );
        }
    }

    /// Every candidate picker blanks the bins either side of the LO, because
    /// a zero-IF front end leaves a spike there that is not a signal. That is
    /// right, but it means a real signal within about 94 Hz of the LO cannot
    /// be seen at all — and the band defaults used to park the LO exactly on
    /// the PSK31 dial frequency, which is the bottom of the sub-band people
    /// actually work.
    ///
    /// Moving the LO is the whole fix: the same signal, same strength, same
    /// everything, seen or not seen purely by where the receiver sits.
    #[test]
    fn a_signal_on_top_of_the_lo_is_invisible_until_the_lo_moves() {
        let fs = 192_000.0f64;
        let n_total = (1.0f32 / 6.0).sqrt();
        let in_bw = n_total * (31.25 / fs as f32).sqrt();
        let scale = 10f32.powf(-20.0 / 20.0) / in_bw;
        // A PSK31 signal 60 Hz above the dial: inside the blanked region if
        // the receiver is parked on the dial frequency.
        let iq = decoders::tests::gen_psk31_at(
            "CQ CQ DE W1AW W1AW K ", fs, 60.0, scale, 12.0);

        let mut on_top = App::new(14_070_000.0, fs, Mode::Auto);
        on_top.agc = AgcMode::Off;
        run_auto(&mut on_top, &iq, fs);
        let seen_on_top = on_top
            .auto
            .iter()
            .any(|s| s.kind == identify::Kind::Psk31);

        // The same IQ, with the receiver 10 kHz lower: the signal now sits
        // clear of the LO and everything about it is otherwise identical.
        let mut moved = App::new(14_060_000.0, fs, Mode::Auto);
        moved.agc = AgcMode::Off;
        let shifted: Vec<Complex32> = iq
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let ph = 2.0 * PI_F * 10_000.0 * i as f32 / fs as f32;
                *s * Complex32::from_polar(1.0, ph)
            })
            .collect();
        run_auto(&mut moved, &shifted, fs);
        let seen_moved = moved.auto.iter().any(|s| s.kind == identify::Kind::Psk31);

        assert!(
            !seen_on_top,
            "expected the LO blanking to hide a signal sitting on it"
        );
        assert!(
            seen_moved,
            "the same signal 10 kHz off the LO should be found"
        );
    }

    /// PSK31 through the whole automatic path: the span classifier has to
    /// flag it, `reconcile_auto` has to build a slot for it, and the slot's
    /// own chain and decoder have to copy it. Each of those is tested
    /// elsewhere in isolation; only together do they answer whether a PSK31
    /// signal on the band actually reaches the screen.
    /// A PSK31 waterfall is not one signal — operators pack in every 50-100
    /// Hz. This is the PSK31 analogue of the adjacent-station problem that
    /// was wrecking CW copy.
    #[test]
    #[ignore]
    fn bench_psk31_neighbours() {
        let fs = 192_000.0f64;
        let center = 14_070_000.0f64;
        let n_total = (1.0f32 / 6.0).sqrt();
        let in_bw = n_total * (31.25 / fs as f32).sqrt();
        let scale = 10f32.powf(-20.0 / 20.0) / in_bw;
        for sep in [200.0f64, 120.0, 80.0, 50.0] {
            let want = decoders::tests::gen_psk31_at(
                "CQ CQ DE W1AW W1AW K ", fs, 1_500.0, scale, 20.0);
            let other = decoders::tests::gen_psk31_at(
                "TEST DE G4XYZ G4XYZ K ", fs, 1_500.0 + sep, 0.0, 20.0);
            let mut iq = want;
            for (i, s) in iq.iter_mut().enumerate() {
                if let Some(o) = other.get(i) {
                    *s += o;
                }
            }
            let mut app = App::new(center, fs, Mode::Auto);
            app.agc = AgcMode::Off;
            let mut spec = Spectrum::new(4096);
            let mut out = Vec::new();
            let per_pass = (fs * 1.2) as usize / 16_384;
            for (i, block) in iq.chunks(16_384).enumerate() {
                app.feed(block, &mut spec, &mut out);
                if i % per_pass.max(1) == per_pass.max(1) - 1 {
                    refresh_psk_hits(&mut app);
                    refresh_idents(&mut app);
                }
            }
            let idents: Vec<String> = app
                .idents
                .iter()
                .map(|i| format!("{} @{:+.0}", i.kind.label(), i.offset_hz))
                .collect();
            let copy: Vec<String> = app
                .rows
                .iter()
                .filter(|r| r.kind == identify::Kind::Psk31)
                .map(|r| format!("{:.1}k: {:?}", r.dial_hz / 1000.0, r.copy))
                .collect();
            println!("--- {sep:.0} Hz apart ---");
            println!("  idents: {idents:?}");
            for c in &copy {
                println!("  {c}");
            }
        }
    }

    /// Everything above lowered a threshold, so the question is what an
    /// empty band now produces. Nothing may be invented out of noise.
    #[test]
    #[ignore]
    fn bench_psk31_false_positives() {
        let fs = 192_000.0f64;
        for seed in [0x1234_5678u32, 0xfeed_face, 0x0bad_c0de] {
            let mut rng = seed;
            let mut noise = || {
                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                (rng as f32 / u32::MAX as f32) - 0.5
            };
            let iq: Vec<Complex32> = (0..(fs * 20.0) as usize)
                .map(|_| Complex32::new(noise(), noise()) * 0.3)
                .collect();
            let mut app = App::new(14_070_000.0, fs, Mode::Auto);
            app.agc = AgcMode::Off;
            let mut spec = Spectrum::new(4096);
            let mut out = Vec::new();
            let per_pass = (fs * 1.2) as usize / 16_384;
            for (i, block) in iq.chunks(16_384).enumerate() {
                app.feed(block, &mut spec, &mut out);
                if i % per_pass.max(1) == per_pass.max(1) - 1 {
                    refresh_psk_hits(&mut app);
                    refresh_cw_hits(&mut app);
                    refresh_idents(&mut app);
                }
            }
            let narrow: Vec<String> = app
                .idents
                .iter()
                .filter(|i| {
                    matches!(i.kind, identify::Kind::Psk31 | identify::Kind::Cw)
                })
                .map(|i| format!("{} @{:+.0}", i.kind.label(), i.offset_hz))
                .collect();
            let copy: usize = app
                .rows
                .iter()
                .map(|r| r.copy.chars().filter(|c| !c.is_whitespace()).count())
                .sum();
            println!(
                "  seed {seed:#x}: {} phantom ident(s) {narrow:?}, {copy} chars of copy",
                narrow.len()
            );
        }
    }

    #[test]
    #[ignore]
    fn bench_psk31_end_to_end() {
        let fs = 192_000.0f64;
        let center = 14_070_000.0f64;
        let offset = 1_500.0f64; // 14.0715 MHz, a real PSK31 watering hole
        for db in [30, 20, 15, 10, 8, 6, 3] {
            let n_total = (1.0f32 / 6.0).sqrt();
            let in_bw = n_total * (31.25 / fs as f32).sqrt();
            let scale = 10f32.powf(-(db as f32) / 20.0) / in_bw;
            let iq = decoders::tests::gen_psk31_at(
                "CQ CQ DE W1AW W1AW K ",
                fs,
                offset,
                scale,
                20.0,
            );

            let mut app = App::new(center, fs, Mode::Auto);
            app.agc = AgcMode::Off;
            run_auto(&mut app, &iq, fs);
            let kinds: Vec<String> = app
                .idents
                .iter()
                .map(|i| format!("{} @{:+.0}Hz q={:.2}", i.kind.label(), i.offset_hz, i.score))
                .collect();
            let slots: Vec<String> = app
                .auto
                .iter()
                .map(|s| format!("{} @{:.1}k", s.kind.label(), s.dial_hz / 1000.0))
                .collect();
            let copy: String = app
                .rows
                .iter()
                .filter(|r| r.kind == identify::Kind::Psk31)
                .map(|r| r.copy.clone())
                .collect::<Vec<_>>()
                .join(" | ");
            println!("--- {db} dB ---");
            println!("  idents: {kinds:?}");
            println!("  slots:  {slots:?}");
            println!("  copy:   {copy:?}");
        }
    }
}

#[cfg(test)]
mod scout_cost {
    use super::*;

    /// Auto mode now runs both narrowband scouts every 800 ms over 1.6 s of
    /// span IQ. That is the price of seeing weak signals; it has to stay well
    /// under the budget or the waterfall stutters.
    #[test]
    #[ignore]
    fn bench_scout_cost() {
        let fs = 192_000.0f64;
        let mut rng = 0x1234_5678u32;
        let mut noise = || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            (rng as f32 / u32::MAX as f32) - 0.5
        };
        // A busy band: many narrow carriers for the scouts to chew on.
        let n = (fs * 1.6) as usize;
        let mut iq: Vec<Complex32> = (0..n)
            .map(|_| Complex32::new(noise(), noise()) * 0.05)
            .collect();
        for k in 0..40 {
            let off = -40_000.0 + k as f64 * 2_000.0;
            for (i, s) in iq.iter_mut().enumerate() {
                let ph = 2.0 * std::f64::consts::PI * off * i as f64 / fs;
                *s += Complex32::from_polar(0.5, ph as f32);
            }
        }
        let mut app = App::new(14_070_000.0, fs, Mode::Auto);
        let mut spec = Spectrum::new(4096);
        let mut out = Vec::new();
        for block in iq.chunks(16_384) {
            app.feed(block, &mut spec, &mut out);
        }
        for (name, f) in [
            ("psk scout", refresh_psk_hits as fn(&mut App)),
            ("cw scout", refresh_cw_hits as fn(&mut App)),
        ] {
            let t = Instant::now();
            let passes = 5;
            for _ in 0..passes {
                f(&mut app);
            }
            let per = t.elapsed().as_secs_f64() / passes as f64;
            println!(
                "  {name}: {:.1} ms per pass ({:.1}% of the 800 ms budget)",
                per * 1000.0,
                per / 0.8 * 100.0
            );
        }
        println!(
            "  candidates: {} peaks",
            scout_peaks(&app.smoothed, &app.noise_floor, 0.0, app.rate).len()
        );
    }
}

#[cfg(test)]
mod band_plan_tests {
    use super::*;

    /// The band plan overlaps: FT4 shares a dial frequency with 30 m PSK31
    /// and with 20 m RTTY. Those sub-bands used to be inside an FT window,
    /// which vetoed every other classification, so they were invisible.
    #[test]
    fn narrowband_subbands_inside_ft_windows_are_still_classified() {
        // 30 m: PSK31 and FT4 share 10.140.
        assert_eq!(bands::narrow_mode(10_140_800.0), Some("PSK"));
        assert_eq!(bands::ft_mode(10_140_800.0), Some("FT4"));
        // 20 m: RTTY and FT4 share 14.080.
        assert_eq!(bands::narrow_mode(14_080_800.0), Some("RTTY"));
        // 20 m PSK31 is not shared and must stay unshadowed.
        assert_eq!(bands::narrow_mode(14_070_800.0), Some("PSK"));
        assert_eq!(bands::ft_mode(14_070_800.0), None);
        // The FT8 calling frequency itself is not a narrowband sub-band.
        assert_eq!(bands::narrow_mode(14_075_500.0), None);
        assert_eq!(bands::ft_mode(14_075_500.0), Some("FT8"));
    }

    /// End to end on 30 m, where PSK31 sits on top of FT4's dial frequency.
    #[test]
    fn psk31_reaches_the_screen_on_30m() {
        let fs = 192_000.0f64;
        let center = 10_140_000.0f64;
        let n_total = (1.0f32 / 6.0).sqrt();
        let in_bw = n_total * (31.25 / fs as f32).sqrt();
        let scale = 10f32.powf(-15.0 / 20.0) / in_bw;
        // 800 Hz into the passband: inside FT4's window as well.
        let iq =
            decoders::tests::gen_psk31_at("CQ CQ DE W1AW W1AW K ", fs, 800.0, scale, 14.0);
        let mut app = App::new(center, fs, Mode::Auto);
        app.agc = AgcMode::Off;
        super::psk_auto_tests::run_auto(&mut app, &iq, fs);

        let copy: String = app
            .rows
            .iter()
            .filter(|r| r.kind == identify::Kind::Psk31)
            .map(|r| r.copy.clone())
            .collect();
        assert!(
            copy.contains("W1AW"),
            "30 m PSK31 produced no copy; idents were {:?}",
            app.idents.iter().map(|i| i.kind.label()).collect::<Vec<_>>()
        );
    }
}
