//! hfscan - a terminal HF panadapter and digital-mode decoder for the
//! SDRplay RSP1A (or any SoapySDR device).

mod bands;
mod decoders;
mod dsp;
mod radio;
mod report;

use anyhow::Result;
use clap::Parser;
use decoders::{Decoder, FtMessage, Mode};
use dsp::{DecodeChain, Spectrum};
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
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
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
const FFT_SIZES: [usize; 6] = [1024, 2048, 4096, 8192, 16384, 32768];

#[derive(Parser, Debug)]
#[command(name = "hfscan", about = "HF band scanner and digital decoder for the RSP1A")]
struct Args {
    /// SoapySDR device arguments
    #[arg(long, default_value = "driver=sdrplay")]
    device: String,
    /// Starting centre frequency in Hz (accepts e.g. 14070000)
    #[arg(short, long, default_value_t = 14_070_000.0)]
    freq: f64,
    /// Sample rate in Hz; this is also the width of the spectrum view
    #[arg(short, long, default_value_t = FT_SAFE_RATE)]
    rate: f64,
    /// FFT size (1024..32768); higher gives finer resolution
    #[arg(long, default_value_t = 8192)]
    fft: usize,
    /// Start with a decoder active: off, cw, rtty, psk31, ft8, ft4
    #[arg(short, long, default_value = "off")]
    mode: String,
    /// Your amateur radio callsign — enables spot reporting to pskreporter.info
    #[arg(long)]
    call: Option<String>,
    /// Your Maidenhead grid locator (e.g. FN42), sent with reception reports
    #[arg(long)]
    grid: Option<String>,
}

struct ScanState {
    end: f64,
    step: f64,
    cur: f64,
    dwell_until: Instant,
    results: Vec<(f64, f32)>,
}

struct App {
    center: f64,
    rate: f64,
    cursor: f64, // offset from centre, Hz
    zoom: f64,   // 1.0 = whole span; higher zooms in around the cursor
    gain: f64,
    agc: bool,
    biast: bool,
    band_idx: usize,

    spectrum: Vec<f32>,
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
    /// Structured FT8/FT4 decodes, newest last; drives the FT panes.
    ft_msgs: VecDeque<FtMessage>,
    /// Stations heard, in first-heard order so the list updates in place.
    stations: Vec<(String, Station)>,
    /// Decode pane size: 0 = default, 1 = large, 2 = huge.
    decode_zoom: u8,
    /// Scroll offsets: transcript lines up from live, stations/slots skipped,
    /// waterfall entries back in time. Zero means pinned to live.
    msg_scroll: usize,
    st_scroll: usize,
    act_scroll: usize,
    wf_scroll: usize,
    /// Waterfall hi-res mode: two frequency bins per column instead of two
    /// time steps per row.
    wf_wide: bool,

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
            gain: 40.0,
            agc: true,
            biast: false,
            band_idx: 0,
            spectrum: Vec::new(),
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
            ft_msgs: VecDeque::new(),
            stations: Vec::new(),
            decode_zoom: 0,
            msg_scroll: 0,
            st_scroll: 0,
            act_scroll: 0,
            wf_scroll: 0,
            wf_wide: false,
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
        };
        app.set_mode(mode);
        app
    }

    fn set_mode(&mut self, mode: Mode) {
        self.mode = mode;
        // Each mode wants its own audio rate, so the chain is rebuilt.
        self.chain = DecodeChain::new(self.rate, 3000.0, mode.audio_rate());
        self.decoder = mode.make(self.chain.fs_out());
        if let Some(d) = &self.decoder {
            self.chain.set_bandwidth(d.bandwidth());
        }
        self.ft_msgs.clear();
        self.stations.clear();
        self.st_scroll = 0;
        self.act_scroll = 0;
        self.log(format!("decoder: {}", mode.label()));
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

    fn log(&mut self, msg: String) {
        self.log.push_back(msg);
        while self.log.len() > 6 {
            self.log.pop_front();
        }
    }

    fn tuned_freq(&self) -> f64 {
        self.center + self.cursor
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
        spec.power_db(block, &mut self.spectrum);
        if self.smoothed.len() != self.spectrum.len() {
            self.smoothed = self.spectrum.clone();
        } else {
            for (s, v) in self.smoothed.iter_mut().zip(&self.spectrum) {
                *s = 0.6 * *s + 0.4 * *v;
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
            while self.waterfall.len() > 400 {
                self.waterfall.pop_back();
            }
            self.wf_last = Instant::now();
        }

        // Auto-range the colour scale from the current noise floor.
        let mut sorted = self.smoothed.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if !sorted.is_empty() {
            let med = sorted[sorted.len() / 2];
            let hi = sorted[sorted.len() * 999 / 1000];
            self.floor_db = 0.8 * self.floor_db + 0.2 * (med - 5.0);
            self.ceil_db = 0.8 * self.ceil_db + 0.2 * (hi + 10.0).max(med + 20.0);
        }

        // Signal-to-noise inside the decoder's passband, used for the squelch.
        if let Some(d) = &self.decoder {
            let n = self.smoothed.len();
            if n > 0 {
                let bin_hz = self.rate / n as f64;
                let half = ((d.bandwidth() as f64 / 2.0) / bin_hz).ceil().max(1.0) as isize;
                let centre = (n as f64 / 2.0 + self.cursor / bin_hz) as isize;
                let lo = (centre - half).clamp(0, n as isize - 1) as usize;
                let hi = (centre + half).clamp(0, n as isize - 1) as usize;
                let peak = self.smoothed[lo..=hi.max(lo)]
                    .iter()
                    .cloned()
                    .fold(f32::MIN, f32::max);
                self.cursor_snr = peak - sorted[sorted.len() / 2];
            }
        }

        if self.decoder.is_some() {
            let (shift, gated) = self
                .decoder
                .as_ref()
                .map(|d| (d.offset_shift(), d.squelched()))
                .unwrap_or((0.0, true));
            self.chain.set_offset(self.cursor + shift);
            self.chain.process(block, out);
            // Feeding noise to a decoder just fills the pane with junk - but
            // slot-based modes must keep capturing regardless.
            let open = !self.squelch || !gated || self.cursor_snr >= self.squelch_db;
            if let Some(d) = &mut self.decoder {
                let new = if open { d.process(out) } else { String::new() };
                if !new.is_empty() {
                    self.text.push_str(&new);
                    // Keep the transcript bounded.
                    if self.text.len() > 8000 {
                        let cut = self.text.len() - 6000;
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

fn parse_mode(s: &str) -> Mode {    match s.to_ascii_lowercase().as_str() {
        "cw" => Mode::Cw,
        "rtty" => Mode::Rtty,
        "psk" | "psk31" => Mode::Psk31,
        "ft8" => Mode::Ft8,
        "ft4" => Mode::Ft4,
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
                    KeyCode::Char('x') => {
                        app.text.clear();
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
                        app.msg_scroll = app.msg_scroll.saturating_add(n);
                    }
                    KeyCode::Down => {
                        let n = if shift { 10 } else { 1 };
                        app.msg_scroll = app.msg_scroll.saturating_sub(n);
                    }
                    KeyCode::Char('W') => {
                        app.wf_wide = !app.wf_wide;
                        app.wf_scroll = 0;
                        let what = if app.wf_wide {
                            "2x frequency, 1x time"
                        } else {
                            "1x frequency, 2x time"
                        };
                        app.log(format!("waterfall: {what}"));
                    }
                    KeyCode::Char('a') => {
                        app.agc = !app.agc;
                        let _ = radio.cmd.send(radio::Cmd::Agc(app.agc));
                        let msg = format!("AGC {}", if app.agc { "on" } else { "off" });
                        app.log(msg);
                    }
                    KeyCode::Char('t') => {
                        app.biast = !app.biast;
                        let _ = radio.cmd.send(radio::Cmd::BiasT(app.biast));
                    }
                    KeyCode::Char('+') | KeyCode::Char('=') => {
                        app.gain = (app.gain + 2.0).min(48.0);
                        app.agc = false;
                        let _ = radio.cmd.send(radio::Cmd::Gain(app.gain));
                    }
                    KeyCode::Char('-') => {
                        app.gain = (app.gain - 2.0).max(0.0);
                        app.agc = false;
                        let _ = radio.cmd.send(radio::Cmd::Gain(app.gain));
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
                            start_scan(app, radio);
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
fn scroll_pane_at(app: &mut App, area: Rect, col: u16, row: u16, delta: isize) {    let chunks = pane_rects(area, app);
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
                adj(&mut app.msg_scroll);
            } else {
                adj(&mut app.st_scroll);
            }
        } else {
            adj(&mut app.msg_scroll);
        }
    } else if inside(chunks[2]) {
        adj(&mut app.wf_scroll);
    }
}

fn nudge_cursor(app: &mut App, delta: f64) {
    let limit = app.rate * 0.45;
    app.cursor = (app.cursor + delta).clamp(-limit, limit);
}

/// Jump the cursor to the next detected signal, so a busy band can be walked
/// without hunting for peaks by eye.
fn next_signal(app: &mut App, forward: bool) {
    let mut peaks = find_peaks(&app.smoothed, 0.0, app.rate);
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

fn retune(app: &mut App, radio: &radio::Radio, freq: f64) {
    let freq = freq.clamp(100_000.0, 30_000_000.0);
    app.center = freq;
    app.waterfall.clear();
    let _ = radio.cmd.send(radio::Cmd::Tune(freq));
    if let Some(b) = bands::band_for(freq) {
        app.band_idx = bands::BANDS
            .iter()
            .position(|x| x.name == b.name)
            .unwrap_or(app.band_idx);
    }
}

fn start_scan(app: &mut App, radio: &radio::Radio) {
    let band = &bands::BANDS[app.band_idx];
    // Step by slightly less than the span so the edges overlap.
    let step = app.rate * 0.8;
    let start = band.start + app.rate / 2.0;
    let state = ScanState {
        end: band.end,
        step,
        cur: start,
        dwell_until: Instant::now() + Duration::from_millis(400),
        results: Vec::new(),
    };
    app.log(format!(
        "scanning {} ({:.3}-{:.3} MHz)",
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

    let peaks = find_peaks(&app.smoothed, cur, app.rate);
    if let Some(s) = app.scan.as_mut() {
        s.results.extend(peaks);
        s.cur += step;
    }

    if cur + step > end {
        // Sweep complete: summarise into the text pane.
        let mut results = app.scan.as_mut()?.results.clone();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.dedup_by(|a, b| (a.0 - b.0).abs() < 200.0);
        app.text.push_str("\n--- scan results (strongest first) ---\n");
        for (f, snr) in results.iter().take(25) {
            let marker = bands::MARKERS
                .iter()
                .find(|m| (m.freq - f).abs() < 1500.0)
                .map(|m| m.label)
                .unwrap_or("");
            app.text.push_str(&format!(
                "{:>10.3} kHz  {:5.1} dB  {}\n",
                f / 1000.0,
                snr,
                marker
            ));
        }
        app.text.push_str("--- end of scan ---\n");
        app.log("scan complete".into());
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
fn find_peaks(spectrum: &[f32], center: f64, rate: f64) -> Vec<(f64, f32)> {
    let n = spectrum.len();
    if n < 8 {
        return Vec::new();
    }
    let mut sorted = spectrum.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = sorted[n / 2];
    let thr = med + SIGNAL_SNR_DB;
    // The LO leaks a spike at the centre of the span; it is not a signal.
    let dc = n / 2;
    let usable = |i: usize| i.abs_diff(dc) > 2 && spectrum[i] >= thr;

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

// ---------------------------------------------------------------- rendering

/// The four vertical panes (status, spectrum, waterfall, decode). Shared by
/// the renderer and mouse hit-testing so they can never disagree.
fn pane_rects(area: Rect, app: &App) -> [Rect; 4] {
    let ft = matches!(app.mode, Mode::Ft8 | Mode::Ft4);
    // `v` enlarges the decode pane at the expense of the waterfall.
    let dec = match (ft, app.decode_zoom) {
        (false, 0) => Constraint::Length(9),
        (true, 0) => Constraint::Length(14),
        (_, 1) => Constraint::Percentage(45),
        (_, _) => Constraint::Percentage(65),
    };
    let chunks = Layout::vertical([
        // 2 content lines + 2 border rows
        Constraint::Length(4),
        Constraint::Length(10),
        Constraint::Min(3),
        dec,
    ])
    .split(area);
    [chunks[0], chunks[1], chunks[2], chunks[3]]
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
    let ft = matches!(app.mode, Mode::Ft8 | Mode::Ft4);
    let chunks = pane_rects(f.area(), app);

    draw_status(f, chunks[0], app);
    draw_spectrum(f, chunks[1], app);
    draw_waterfall(f, chunks[2], app);
    if ft {
        draw_ft(f, chunks[3], app);
    } else {
        draw_decode(f, chunks[3], app);
    }

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
            "  FT8/FT4 spots are reported to pskreporter.info",
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
        "UTC {:02}:{:02}:{:02}",
        (now / 3600.0) as u64 % 24,
        (now / 60.0) as u64 % 60,
        now as u64 % 60
    );
    let slot = match app.mode {
        Mode::Ft8 => format!("  slot -{:2.0}s", 15.0 - now % 15.0),
        Mode::Ft4 => format!("  slot -{:2.1}s", 7.5 - now % 7.5),
        _ => String::new(),
    };
    let spotting = if let Some(r) = &app.reporter {
        format!("  spots {}", r.sent_count())
    } else {
        String::new()
    };

    let mut spans1 = vec![
        Span::styled(
            format!("{:.4} kHz", app.tuned_freq() / 1000.0),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(marker),
    ];
    if !app.my_call.is_empty() {
        spans1.push(Span::styled(
            format!("   de {}", app.my_call),
            Style::default().fg(Color::Cyan),
        ));
    }
    spans1.push(Span::raw(format!(
        "   centre {:.3}  cursor {:+.0} Hz  {}  view {:.2} kHz (x{:.0})  step {:.0} Hz  {:.1} Hz/bin",
        app.center / 1000.0,
        app.cursor,
        band,
        (hi - lo) / 1000.0,
        app.zoom,
        app.step_hz(),
        app.bin_hz(),
    )));
    let line1 = Line::from(spans1);
    let line2 = Line::from(format!(
        "{}  {}  {}{}  snr {:+.0} dB  sq {}  {}  bias-T {}  wf {}ms  {}{}  {}",
        app.mode.label(),
        dec_status,
        utc,
        slot,
        app.cursor_snr,
        if app.squelch {
            format!("{:.0}", app.squelch_db)
        } else {
            "off".to_string()
        },
        if app.agc {
            "AGC".to_string()
        } else {
            format!("{:.0} dB", app.gain)
        },
        if app.biast { "ON" } else { "off" },
        WF_INTERVALS_MS[app.wf_idx],
        if app.scan.is_some() { "SCANNING" } else { "" },
        spotting,
        app.log.back().cloned().unwrap_or_default(),
    ));

    let p = Paragraph::new(vec![line1, line2]).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" hfscan — press ? for help "),
    );
    f.render_widget(p, area);
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
    let (lo, hi) = app.view_range();
    let frac = (app.cursor - lo) / (hi - lo).max(1.0);
    ((frac * width as f64) as isize).clamp(0, width as isize - 1) as usize
}

fn draw_spectrum(f: &mut Frame, area: Rect, app: &App) {
    let block = Block::default().borders(Borders::ALL).title(" spectrum ");
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

    // In the FT modes the decoder monitors a fixed 200-2900 Hz window above
    // the dial (the cursor); shade that band so it is visible at a glance.
    let ft = matches!(app.mode, Mode::Ft8 | Mode::Ft4);
    let band_lo = app.cursor + decoders::ft8::FREQ_MIN as f64;
    let band_hi = app.cursor + decoders::ft8::FREQ_MAX as f64;
    let (vlo, vhi) = app.view_range();
    let in_band = |x: usize| {
        let off = vlo + (x as f64 + 0.5) * (vhi - vlo) / w as f64;
        ft && off >= band_lo && off <= band_hi
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
            let style = if x == cur {
                Style::default().fg(Color::Magenta).bg(Color::Rgb(40, 0, 40))
            } else {
                let mut s = Style::default().fg(heat(norm));
                if in_band(x) {
                    s = s.bg(Color::Rgb(40, 40, 50));
                }
                s
            };
            spans.push(Span::styled(ch.to_string(), style));
        }
        lines.push(Line::from(spans));
    }
    if h > 1 {
        lines.push(axis_row(app, w));
    }
    f.render_widget(Paragraph::new(lines), inner);
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
    if app.wf_wide {
        title.push_str("(hi-res freq) ");
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

    let mut lines = Vec::with_capacity(inner.height as usize);
    if app.wf_wide {
        // Hi-res: two frequency bins per column ('▌' paints the left bin in
        // the foreground colour and the right in the background), one time
        // step per row — twice the frequency detail at half the time density.
        for row in 0..inner.height as usize {
            let Some(spec) = row_spec(row) else {
                lines.push(Line::from(""));
                continue;
            };
            let (a, b) = view_bins(app, spec.len());
            let cols = resample(&spec[a..b], w * 2);
            let mut spans = Vec::with_capacity(w);
            for x in 0..w {
                let l = norm(cols[x * 2]);
                let r = norm(cols[x * 2 + 1]);
                let style = Style::default().fg(heat(l)).bg(heat(r));
                if x == cur {
                    spans.push(Span::styled("│", style.fg(Color::Magenta)));
                } else {
                    spans.push(Span::styled("▌", style));
                }
            }
            lines.push(Line::from(spans));
        }
        f.render_widget(Paragraph::new(lines), inner);
        return;
    }

    // Each text row holds two time steps: '▀' paints the upper half in the
    // foreground colour and the lower half in the background, doubling the
    // vertical resolution the terminal can show.
    let row_cols = |idx: usize| -> Option<Vec<f32>> {
        let spec = row_spec(idx)?;
        let (a, b) = view_bins(app, spec.len());
        Some(resample(&spec[a..b], w))
    };

    for row in 0..inner.height as usize {
        let upper = row_cols(row * 2);
        let lower = row_cols(row * 2 + 1);
        let mut spans = Vec::with_capacity(w);
        if upper.is_none() && lower.is_none() {
            lines.push(Line::from(""));
            continue;
        }
        let normc = |cols: &Option<Vec<f32>>, x: usize| -> Option<f32> {
            cols.as_ref()
                .and_then(|c| c.get(x))
                .map(|v| norm(*v))
        };
        for x in 0..w {
            let up = normc(&upper, x);
            let dn = normc(&lower, x);
            let mut style = Style::default();
            if let Some(u) = up {
                style = style.fg(heat(u));
            }
            if let Some(d) = dn {
                style = style.bg(heat(d));
            }
            if x == cur {
                // Keep the cursor readable against whatever is behind it.
                spans.push(Span::styled("│", style.fg(Color::Magenta)));
            } else {
                spans.push(Span::styled("▀", style));
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
    let mut wrapped: Vec<String> = Vec::new();
    for line in app.text.split('\n') {
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
    let mut title = format!(" decode: {} ", app.mode.label());
    if app.msg_scroll > 0 {
        title = format!(" decode: {} (scrolled up {}) ", app.mode.label(), app.msg_scroll);
    }
    let block = Block::default().borders(Borders::ALL).title(title);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let body = transcript_lines(app, inner.width.max(1) as usize, inner.height as usize, false);
    f.render_widget(Paragraph::new(body).wrap(Wrap { trim: false }), inner);
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
        Line::from("  wheel       scroll the pane under the mouse (waterfall too)"),
        Line::from("  z / Z       zoom in / out — also sets the tuning step"),
        Line::from("  n / N       jump to next / previous signal"),
        Line::from("  [ ]         retune centre ±10 kHz"),
        Line::from("  PgUp/PgDn   retune centre ± half span"),
        Line::from("  c           centre the radio on the cursor"),
        Line::from("  b / B       next / previous band preset"),
        Line::from("  d           decoder: off → CW → RTTY → PSK31 → FT8 → FT4"),
        Line::from("  r           RTTY normal/reverse shift"),
        Line::from("  s           scan the current band for signals"),
        Line::from("  v           enlarge the decode pane (cycles sizes)"),
        Line::from("  w / W       waterfall speed / hi-res frequency mode"),
        Line::from("  f / F       FFT size (frequency resolution)"),
        Line::from("  a           toggle AGC      + / -   manual gain"),
        Line::from("  k           squelch on/off  , / .   squelch threshold"),
        Line::from("  t           toggle bias-T (external preamp power)"),
        Line::from("  o           station settings (callsign, grid, spotting)"),
        Line::from("  x           clear the decode pane"),
        Line::from("  ? / q       toggle help / quit"),
    ];
    let w = 62.min(area.width.saturating_sub(4));
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
            // Scrolled panes and hi-res waterfall must render too.
            app.decode_zoom = 0;
            app.msg_scroll = 3;
            app.st_scroll = 1;
            app.act_scroll = 1;
            app.wf_scroll = 5;
            app.wf_wide = true;
            t.draw(|f| super::draw(f, &app)).unwrap();
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

    /// Scrolling the transcript shows older lines; zero stays pinned to live.
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
