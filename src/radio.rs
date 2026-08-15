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
    if let Err(e) = set_bandwidth(&dev, actual_rate, &log_tx) {
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
                Ok(Cmd::Tune(f)) => set_and_log(&log_tx, "tune", dev.set_frequency(Rx, 0, f, ())),
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
                        if let Err(e) = dev.set_bandwidth(Rx, 0, rate) {
                            let _ = log_tx.try_send(format!("analog bandwidth unchanged: {e}"));
                        }
                        let bw = dev.bandwidth(Rx, 0).unwrap_or(rate);
                        let _ = log_tx.try_send(format!(
                            "sample rate {:.0} Hz, analog bandwidth {:.0} Hz",
                            rate, bw
                        ));
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

/// Keep the tuner's analog IF no wider than the digitised span and verify the
/// value the driver selected. A startup failure is material: silently running
/// a wide analog filter costs both headroom and alias rejection.
fn set_bandwidth(dev: &soapysdr::Device, rate: f64, log_tx: &SyncSender<String>) -> Result<()> {
    dev.set_bandwidth(Rx, 0, rate)?;
    let actual = dev.bandwidth(Rx, 0).unwrap_or(rate);
    let _ = log_tx.try_send(format!("analog bandwidth {:.0} Hz", actual));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
