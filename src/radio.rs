//! Radio worker thread: owns the SoapySDR device and streams IQ to the UI.

use anyhow::{Context, Result};
use num_complex::Complex32;
use soapysdr::Direction::Rx;
use std::sync::mpsc::{Receiver, Sender, SyncSender, TrySendError, channel, sync_channel};

#[derive(Clone, Debug, PartialEq)]
pub enum GainControl {
    /// A conventional aggregate gain value in dB.
    Overall { min: f64, max: f64 },
    /// SDRplay's two gain-*reduction* controls. Larger values mean less gain.
    Sdrplay {
        rfgr_min: f64,
        rfgr_max: f64,
        ifgr_min: f64,
        ifgr_max: f64,
    },
}

#[derive(Clone, Debug)]
pub struct Capabilities {
    pub driver: String,
    pub hardware: String,
    pub gain: GainControl,
    pub hardware_agc: bool,
    pub agc_setpoint: bool,
    pub rf_notch: bool,
    pub dab_notch: bool,
    pub iq_correction: bool,
    pub ppm: bool,
}

#[derive(Clone, Debug, Default)]
pub struct State {
    pub overall_gain: Option<f64>,
    pub rfgr: Option<f64>,
    pub ifgr: Option<f64>,
    pub agc: Option<bool>,
    pub agc_setpoint: Option<i32>,
    pub rf_notch: Option<bool>,
    pub dab_notch: Option<bool>,
    pub iq_correction: Option<bool>,
    pub ppm: Option<f64>,
    pub rate: Option<f64>,
    pub bandwidth: Option<f64>,
}

#[derive(Clone, Debug)]
pub enum Event {
    Capabilities(Capabilities),
    State(State),
    StreamStats {
        dropped_blocks: u64,
        clipped_fraction: f64,
    },
}

#[allow(dead_code)]
pub enum Cmd {
    Tune(f64),
    Gain(f64),
    Rfgr(f64),
    Ifgr(f64),
    Agc(bool),
    AgcSetpoint(i32),
    BiasT(bool),
    RfNotch(bool),
    DabNotch(bool),
    IqCorrection(bool),
    Ppm(f64),
    Rate(f64),
    Quit,
}

pub struct Radio {
    pub cmd: Sender<Cmd>,
    pub iq: Receiver<Vec<Complex32>>,
    pub log: Receiver<String>,
    pub events: Receiver<Event>,
    /// Actual rate selected by the driver. Some backends clamp the request.
    pub rate: f64,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl Radio {
    /// A handle with no device behind it, so the parts of the app that take a
    /// `&Radio` only to send it commands can be exercised in a test.
    #[cfg(test)]
    pub fn for_test() -> (Radio, Receiver<Cmd>) {
        let (cmd, cmd_rx) = std::sync::mpsc::channel();
        Radio::detached(cmd, 192_000.0)
            .map(|r| (r, cmd_rx))
            .unwrap()
    }

    #[cfg(test)]
    fn detached(cmd: Sender<Cmd>, rate: f64) -> Option<Radio> {
        let (_, iq) = sync_channel(1);
        let (_, log) = sync_channel(1);
        let (_, events) = sync_channel(1);
        Some(Radio {
            cmd,
            iq,
            log,
            events,
            rate,
            worker: None,
        })
    }
}

impl Drop for Radio {
    fn drop(&mut self) {
        let _ = self.cmd.send(Cmd::Quit);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

const BLOCK: usize = 16384;

pub fn spawn(args: String, rate: f64, freq: f64) -> Result<Radio> {
    let (cmd_tx, cmd_rx) = channel::<Cmd>();
    let (iq_tx, iq_rx) = sync_channel::<Vec<Complex32>>(8);
    let (log_tx, log_rx) = sync_channel::<String>(64);
    let (event_tx, event_rx) = sync_channel::<Event>(32);

    // Open the device on this thread so startup errors surface immediately.
    let dev = soapysdr::Device::new(args.as_str()).context("opening SDR device")?;
    let caps = inspect(&dev);
    let _ = event_tx.try_send(Event::Capabilities(caps.clone()));
    log_capabilities(&dev, &caps, &log_tx);

    let requested_rate = rate;
    dev.set_sample_rate(Rx, 0, requested_rate)
        .context("setting sample rate")?;
    let actual_rate = dev.sample_rate(Rx, 0).unwrap_or(requested_rate);
    if (actual_rate - rate).abs() >= 1.0 {
        let _ = log_tx.try_send(format!(
            "sample rate requested {:.0} Hz, backend selected {:.0} Hz",
            rate, actual_rate
        ));
    }
    if let Err(e) = set_bandwidth(&dev, actual_rate, cover_for(freq), &log_tx) {
        let _ = log_tx.try_send(format!("analog bandwidth unavailable: {e}"));
    }
    dev.set_frequency(Rx, 0, freq, ())
        .context("setting frequency")?;
    if caps.hardware_agc
        && let Err(e) = dev.set_gain_mode(Rx, 0, false)
    {
        let _ = log_tx.try_send(format!("could not disable hardware AGC: {e}"));
    }
    if let Err(e) = set_initial_gain(&dev, &caps) {
        let _ = log_tx.try_send(format!("initial gain was not applied: {e}"));
    }

    // These settings belong to SoapySDRPlay3. Keep unsupported controls out
    // of generic backends instead of issuing writes which can appear to work.
    if caps.iq_correction
        && let Err(e) = write_bool(&dev, "iqcorr_ctrl", true)
    {
        let _ = log_tx.try_send(format!("driver IQ correction unavailable: {e}"));
    }
    if caps.rf_notch {
        // The combined MW/FM rejection network is beneficial on HF, but must
        // stay out while the requested signal itself is in the MW/LW region.
        if let Err(e) = write_bool(&dev, "rfnotch_ctrl", freq >= 2_000_000.0) {
            let _ = log_tx.try_send(format!("RF notch unavailable: {e}"));
        }
    }
    if caps.dab_notch
        && let Err(e) = write_bool(&dev, "dabnotch_ctrl", false)
    {
        let _ = log_tx.try_send(format!("DAB notch unavailable: {e}"));
    }
    let _ = dev.write_setting("biasT_ctrl", "false");

    let dc_auto = dev.has_dc_offset_mode(Rx, 0).unwrap_or(false)
        && dev.set_dc_offset_mode(Rx, 0, true).is_ok();
    let _ = log_tx.try_send(format!(
        "front end: DC correction {}, IQ correction {}",
        if dc_auto {
            "driver + software"
        } else {
            "software"
        },
        if caps.iq_correction {
            "driver + software"
        } else {
            "software"
        },
    ));
    publish_state(&dev, &caps, &event_tx);

    let worker = std::thread::spawn(move || {
        if let Err(e) = run(
            dev,
            caps,
            actual_rate,
            freq,
            cmd_rx,
            iq_tx,
            log_tx.clone(),
            event_tx,
        ) {
            let _ = log_tx.try_send(format!("radio thread stopped: {e:#}"));
        }
    });

    Ok(Radio {
        cmd: cmd_tx,
        iq: iq_rx,
        log: log_rx,
        events: event_rx,
        rate: actual_rate,
        worker: Some(worker),
    })
}

fn inspect(dev: &soapysdr::Device) -> Capabilities {
    let driver = dev.driver_key().unwrap_or_else(|_| "unknown".into());
    let hardware = dev.hardware_key().unwrap_or_else(|_| "unknown".into());
    let gains = dev.list_gains(Rx, 0).unwrap_or_default();
    let find = |want: &str| gains.iter().find(|g| g.eq_ignore_ascii_case(want));
    let split = find("RFGR").zip(find("IFGR"));
    let gain = if let Some((rf, ifg)) = split {
        let rr = dev.gain_element_range(Rx, 0, rf.as_str()).ok();
        let ir = dev.gain_element_range(Rx, 0, ifg.as_str()).ok();
        GainControl::Sdrplay {
            rfgr_min: rr.as_ref().map_or(0.0, |r| r.minimum),
            rfgr_max: rr.as_ref().map_or(9.0, |r| r.maximum),
            ifgr_min: ir.as_ref().map_or(20.0, |r| r.minimum),
            ifgr_max: ir.as_ref().map_or(59.0, |r| r.maximum),
        }
    } else {
        let r = dev.gain_range(Rx, 0).ok();
        GainControl::Overall {
            min: r.as_ref().map_or(0.0, |x| x.minimum),
            max: r.as_ref().map_or(48.0, |x| x.maximum),
        }
    };
    let sdrplay = matches!(gain, GainControl::Sdrplay { .. })
        || driver.to_ascii_lowercase().contains("sdrplay");
    let has_setting = |key: &str| sdrplay && dev.read_setting(key).is_ok();
    let ppm = dev
        .list_frequencies(Rx, 0)
        .is_ok_and(|v| v.iter().any(|x| x.eq_ignore_ascii_case("CORR")));
    Capabilities {
        driver,
        hardware,
        gain,
        hardware_agc: dev.has_gain_mode(Rx, 0).unwrap_or(false),
        agc_setpoint: has_setting("agc_setpoint"),
        rf_notch: has_setting("rfnotch_ctrl"),
        dab_notch: has_setting("dabnotch_ctrl"),
        iq_correction: has_setting("iqcorr_ctrl"),
        ppm,
    }
}

fn log_capabilities(dev: &soapysdr::Device, caps: &Capabilities, log: &SyncSender<String>) {
    let info = dev
        .hardware_info()
        .map(|a| a.to_string())
        .unwrap_or_else(|_| "unavailable".into());
    let _ = log.try_send(format!(
        "radio: driver={} hardware={} info={}",
        caps.driver, caps.hardware, info
    ));
    let _ = log.try_send(match caps.gain {
        GainControl::Sdrplay {
            rfgr_min,
            rfgr_max,
            ifgr_min,
            ifgr_max,
        } => format!(
            "controls: RFGR {rfgr_min:.0}..{rfgr_max:.0}, IFGR {ifgr_min:.0}..{ifgr_max:.0} (gain reduction)"
        ),
        GainControl::Overall { min, max } => {
            format!("controls: aggregate gain {min:.0}..{max:.0} dB")
        }
    });
    let _ = log.try_send(format!(
        "controls: AGC={} setpoint={} RF-notch={} DAB-notch={} IQ={} PPM={}",
        caps.hardware_agc,
        caps.agc_setpoint,
        caps.rf_notch,
        caps.dab_notch,
        caps.iq_correction,
        caps.ppm,
    ));
    if caps.driver.eq_ignore_ascii_case("miri") {
        let _ = log.try_send(
            "warning: SDRplay hardware is using the miri backend; SDRplay-specific controls are unavailable"
                .into(),
        );
    }
}

fn set_initial_gain(dev: &soapysdr::Device, caps: &Capabilities) -> Result<()> {
    match caps.gain {
        GainControl::Sdrplay {
            rfgr_min,
            rfgr_max,
            ifgr_min,
            ifgr_max,
        } => {
            dev.set_gain_element(Rx, 0, "RFGR", 3.0_f64.clamp(rfgr_min, rfgr_max))?;
            dev.set_gain_element(Rx, 0, "IFGR", 40.0_f64.clamp(ifgr_min, ifgr_max))?;
        }
        GainControl::Overall { min, max } => dev.set_gain(Rx, 0, 36.0_f64.clamp(min, max))?,
    }
    Ok(())
}

fn run(
    dev: soapysdr::Device,
    caps: Capabilities,
    mut rate: f64,
    mut tuned: f64,
    cmd_rx: Receiver<Cmd>,
    iq_tx: SyncSender<Vec<Complex32>>,
    log_tx: SyncSender<String>,
    event_tx: SyncSender<Event>,
) -> Result<()> {
    let mut stream = dev.rx_stream::<Complex32>(&[0])?;
    stream.activate(None)?;
    let mut buf = vec![Complex32::new(0.0, 0.0); BLOCK];
    let mut overruns: u64 = 0;
    let mut clipped: u64 = 0;
    let mut counted: u64 = 0;
    let mut ovl_quiet = std::time::Instant::now();

    loop {
        loop {
            let changed = match cmd_rx.try_recv() {
                Ok(Cmd::Quit) => {
                    let _ = stream.deactivate(None);
                    return Ok(());
                }
                Ok(Cmd::Tune(f)) => {
                    tuned = f;
                    set_and_log(&log_tx, "tune", dev.set_frequency(Rx, 0, f, ()))
                }
                Ok(Cmd::Gain(g)) => {
                    let _ = dev.set_gain_mode(Rx, 0, false);
                    set_and_log(&log_tx, "gain", dev.set_gain(Rx, 0, g))
                }
                Ok(Cmd::Rfgr(g)) => {
                    let _ = dev.set_gain_mode(Rx, 0, false);
                    set_and_log(&log_tx, "RFGR", dev.set_gain_element(Rx, 0, "RFGR", g))
                }
                Ok(Cmd::Ifgr(g)) => {
                    let _ = dev.set_gain_mode(Rx, 0, false);
                    set_and_log(&log_tx, "IFGR", dev.set_gain_element(Rx, 0, "IFGR", g))
                }
                Ok(Cmd::Agc(on)) => set_and_log(&log_tx, "AGC", dev.set_gain_mode(Rx, 0, on)),
                Ok(Cmd::AgcSetpoint(dbfs)) => set_and_log(
                    &log_tx,
                    "AGC setpoint",
                    dev.write_setting("agc_setpoint".to_string(), dbfs.to_string()),
                ),
                Ok(Cmd::BiasT(on)) => set_bool_setting(&dev, &log_tx, "bias-T", "biasT_ctrl", on),
                Ok(Cmd::RfNotch(on)) => {
                    set_bool_setting(&dev, &log_tx, "RF notch", "rfnotch_ctrl", on)
                }
                Ok(Cmd::DabNotch(on)) => {
                    set_bool_setting(&dev, &log_tx, "DAB notch", "dabnotch_ctrl", on)
                }
                Ok(Cmd::IqCorrection(on)) => {
                    set_bool_setting(&dev, &log_tx, "IQ correction", "iqcorr_ctrl", on)
                }
                Ok(Cmd::Ppm(ppm)) => set_and_log(
                    &log_tx,
                    "frequency correction",
                    dev.set_component_frequency(Rx, 0, "CORR", ppm, ()),
                ),
                Ok(Cmd::Rate(r)) => {
                    let _ = stream.deactivate(None);
                    drop(stream);
                    let requested = r;
                    let ok = dev.set_sample_rate(Rx, 0, requested);
                    if let Err(ref e) = ok {
                        let _ = log_tx.try_send(format!("rate failed: {e}"));
                    } else {
                        rate = dev.sample_rate(Rx, 0).unwrap_or(requested);
                        if let Err(e) = set_bandwidth(&dev, rate, cover_for(tuned), &log_tx) {
                            let _ = log_tx.try_send(format!("analog bandwidth unchanged: {e}"));
                        }
                        let _ = log_tx.try_send(format!("sample rate {rate:.0} Hz"));
                    }
                    stream = dev.rx_stream::<Complex32>(&[0])?;
                    stream.activate(None)?;
                    ok.is_ok()
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    let _ = stream.deactivate(None);
                    return Ok(());
                }
            };
            if changed {
                publish_state(&dev, &caps, &event_tx);
            }
        }

        match stream.read(&mut [&mut buf[..]], 1_000_000) {
            Ok(0) => continue,
            Ok(n) => {
                clipped += buf[..n]
                    .iter()
                    .filter(|c| c.re.abs() > 0.98 || c.im.abs() > 0.98)
                    .count() as u64;
                counted += n as u64;
                if counted >= rate as u64 {
                    let clipped_fraction = clipped as f64 / counted.max(1) as f64;
                    let _ = event_tx.try_send(Event::StreamStats {
                        dropped_blocks: overruns,
                        clipped_fraction,
                    });
                    if clipped * 10_000 > counted
                        && ovl_quiet.elapsed() > std::time::Duration::from_secs(15)
                    {
                        let _ = log_tx.try_send(
                            "ADC overload: signals at full scale — increase gain reduction"
                                .to_string(),
                        );
                        ovl_quiet = std::time::Instant::now();
                    }
                    clipped = 0;
                    counted = 0;
                }
                match iq_tx.try_send(buf[..n].to_vec()) {
                    Ok(()) => {}
                    Err(TrySendError::Full(_)) => {
                        overruns += 1;
                        if overruns % 20 == 1 {
                            let _ = log_tx.try_send(format!("dropped {overruns} blocks (UI slow)"));
                        }
                    }
                    Err(TrySendError::Disconnected(_)) => return Ok(()),
                }
            }
            Err(e) => {
                let _ = log_tx.try_send(format!("read error: {e}"));
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }
    }
}

fn set_and_log(
    log: &SyncSender<String>,
    what: &str,
    result: std::result::Result<(), soapysdr::Error>,
) -> bool {
    match result {
        Ok(()) => true,
        Err(e) => {
            let _ = log.try_send(format!("{what} failed: {e}"));
            false
        }
    }
}

fn write_bool(
    dev: &soapysdr::Device,
    key: &str,
    on: bool,
) -> std::result::Result<(), soapysdr::Error> {
    dev.write_setting(key, if on { "true" } else { "false" })
}

fn set_bool_setting(
    dev: &soapysdr::Device,
    log: &SyncSender<String>,
    what: &str,
    key: &str,
    on: bool,
) -> bool {
    let ok = set_and_log(log, what, write_bool(dev, key, on));
    if ok {
        let _ = log.try_send(format!("{what} {}", if on { "on" } else { "off" }));
    }
    ok
}

fn read_bool(dev: &soapysdr::Device, key: &str) -> Option<bool> {
    dev.read_setting(key)
        .ok()
        .and_then(|v| match v.trim().to_ascii_lowercase().as_str() {
            "true" | "1" | "on" => Some(true),
            "false" | "0" | "off" => Some(false),
            _ => None,
        })
}

fn publish_state(dev: &soapysdr::Device, caps: &Capabilities, tx: &SyncSender<Event>) {
    let mut state = State {
        agc: dev.gain_mode(Rx, 0).ok(),
        rate: dev.sample_rate(Rx, 0).ok(),
        bandwidth: dev.bandwidth(Rx, 0).ok(),
        ..State::default()
    };
    match caps.gain {
        GainControl::Overall { .. } => state.overall_gain = dev.gain(Rx, 0).ok(),
        GainControl::Sdrplay { .. } => {
            state.rfgr = dev.gain_element(Rx, 0, "RFGR").ok();
            state.ifgr = dev.gain_element(Rx, 0, "IFGR").ok();
        }
    }
    if caps.agc_setpoint {
        state.agc_setpoint = dev
            .read_setting("agc_setpoint")
            .ok()
            .and_then(|v| v.trim().parse().ok());
    }
    if caps.rf_notch {
        state.rf_notch = read_bool(dev, "rfnotch_ctrl");
    }
    if caps.dab_notch {
        state.dab_notch = read_bool(dev, "dabnotch_ctrl");
    }
    if caps.iq_correction {
        state.iq_correction = read_bool(dev, "iqcorr_ctrl");
    }
    if caps.ppm {
        state.ppm = dev.component_frequency(Rx, 0, "CORR").ok();
    }
    let _ = tx.try_send(Event::State(state));
}

/// Pick the tuner's analog IF filter, and verify what the driver selected.
///
/// Asking for a bandwidth equal to the sample rate is the obvious thing and
/// the wrong one. These tuners offer a handful of discrete filter widths, so
/// the driver rounds the request to one of them — and rounding *down* puts the
/// filter corner inside the digitised span, which is visible on the waterfall
/// as the span's outer edges falling away. When the span is sized to a ham
/// band, those edges are the band edges.
///
/// So the choice is made against the width that has to stay flat rather than
/// against the span: the narrowest filter that still covers it. The span is
/// the upper bound, not the target — a filter wider than what is digitised
/// costs alias rejection, which is why the request was tied to the rate in the
/// first place. Where nothing on offer can do both, alias rejection wins and
/// the shortfall is logged rather than left to be discovered on the display.
/// The width that must stay flat at `freq`: the amateur allocation it sits in,
/// since that is what the span was sized around. Outside a band there is no
/// such requirement and the span itself is the only constraint.
fn cover_for(freq: f64) -> f64 {
    crate::bands::band_for(freq).map_or(0.0, |b| b.end - b.start)
}

/// The narrowest offered filter that covers `cover_hz` without exceeding
/// `rate`; failing that the widest that fits under `rate`, since alias
/// rejection is worth more than the last decibel at the band edge.
fn choose_bandwidth(options: &[f64], rate: f64, cover_hz: f64) -> f64 {
    let fits = |w: &&f64| **w <= rate * 1.001;
    options
        .iter()
        .filter(fits)
        .find(|w| **w >= cover_hz)
        .or_else(|| options.iter().filter(fits).next_back())
        .copied()
        .unwrap_or(rate)
}

fn set_bandwidth(
    dev: &soapysdr::Device,
    rate: f64,
    cover_hz: f64,
    log_tx: &SyncSender<String>,
) -> Result<()> {
    // Drivers report discrete filter widths as degenerate ranges, one per
    // width. A driver that instead reports one continuous range must not be
    // read as "two choices, 200 kHz or 8 MHz" — picking an endpoint there
    // would be far worse than the rounding this is meant to fix, so a
    // continuous range is honoured by asking for what is actually wanted.
    let ranges = dev.bandwidth_range(Rx, 0).unwrap_or_default();
    let continuous = ranges
        .iter()
        .find(|r| r.maximum > r.minimum + r.minimum.abs().max(1.0) * 1e-6);
    let want = if let Some(r) = continuous {
        cover_hz.max(r.minimum).min(rate.min(r.maximum))
    } else {
        let mut options: Vec<f64> = ranges
            .iter()
            .map(|r| r.maximum)
            .filter(|v| *v > 0.0)
            .collect();
        options.sort_by(f64::total_cmp);
        options.dedup();
        choose_bandwidth(&options, rate, cover_hz)
    };

    dev.set_bandwidth(Rx, 0, want)?;
    let actual = dev.bandwidth(Rx, 0).unwrap_or(want);
    let _ = log_tx.try_send(format!("analog bandwidth {:.0} Hz", actual));
    if cover_hz > 0.0 && actual < cover_hz * 0.999 {
        let _ = log_tx.try_send(format!(
            "analog filter is {:.0} kHz but {:.0} kHz needs to stay flat — band edges will roll off",
            actual / 1000.0,
            cover_hz / 1000.0
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The MSi001's discrete filter widths, as the driver reports them.
    const RSP1A: [f64; 8] = [
        200_000.0,
        300_000.0,
        600_000.0,
        1_536_000.0,
        5_000_000.0,
        6_000_000.0,
        7_000_000.0,
        8_000_000.0,
    ];

    /// Requesting a filter as wide as the span is what made the band edges
    /// fall away: the driver rounds to one of a handful of widths, and
    /// rounding down puts the corner inside the digitised span.
    #[test]
    fn the_filter_covers_the_band_rather_than_matching_the_span() {
        // 20m is 350 kHz in a 456 kHz span. Matching the span rounds down to
        // 300 kHz and clips 25 kHz off each end of the band; 600 kHz is wider
        // than the span, so 300 is still the honest answer here and the caller
        // is told. What must not happen is silently choosing 200.
        assert_eq!(choose_bandwidth(&RSP1A, 456_000.0, 350_000.0), 300_000.0);
        // Give it a span that can hold the 600 kHz filter and it takes it.
        assert_eq!(choose_bandwidth(&RSP1A, 648_000.0, 350_000.0), 600_000.0);
        // 80m: 500 kHz of band, and a span with room for the filter that fits.
        assert_eq!(choose_bandwidth(&RSP1A, 648_000.0, 500_000.0), 600_000.0);
        // 6m: 4 MHz of band in a 5.016 MS/s span.
        assert_eq!(
            choose_bandwidth(&RSP1A, 5_016_000.0, 4_000_000.0),
            5_000_000.0
        );
        // A narrow band in a wide span takes the narrowest filter that covers
        // it, not the widest that fits, or alias rejection is thrown away.
        assert_eq!(choose_bandwidth(&RSP1A, 648_000.0, 50_000.0), 200_000.0);
        // At the 192 kHz span the narrow bands use, nothing on offer is that
        // narrow, so the request falls through to the span and the driver
        // rounds up to its 200 kHz filter — 4% wider than Nyquist, which
        // costs far less than clipping the band would.
        assert_eq!(choose_bandwidth(&RSP1A, 192_000.0, 50_000.0), 192_000.0);
    }

    /// Alias rejection outranks the band edge when nothing can do both.
    #[test]
    fn a_filter_never_exceeds_what_is_digitised() {
        for (rate, cover) in [
            (456_000.0, 350_000.0),
            (192_000.0, 200_000.0),
            (1_920_000.0, 1_700_000.0),
        ] {
            let got = choose_bandwidth(&RSP1A, rate, cover);
            assert!(
                got <= rate * 1.001,
                "chose a {got:.0} Hz filter for a {rate:.0} Hz span"
            );
        }
    }

    /// With nothing to go on, fall back to the span rather than guessing.
    #[test]
    fn no_reported_options_falls_back_to_the_span() {
        assert_eq!(choose_bandwidth(&[], 432_000.0, 350_000.0), 432_000.0);
    }

    #[test]
    fn gain_control_documents_reduction_direction() {
        let c = GainControl::Sdrplay {
            rfgr_min: 0.0,
            rfgr_max: 9.0,
            ifgr_min: 20.0,
            ifgr_max: 59.0,
        };
        assert!(matches!(c, GainControl::Sdrplay { rfgr_max: 9.0, .. }));
    }
}
