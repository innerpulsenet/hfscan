//! Radio worker thread: owns the SoapySDR device and streams IQ to the UI.

use anyhow::{Context, Result};
use num_complex::Complex32;
use soapysdr::Direction::Rx;
use std::sync::mpsc::{sync_channel, Receiver, Sender, SyncSender, TrySendError};
use std::sync::mpsc::channel;

#[allow(dead_code)]
pub enum Cmd {
    Tune(f64),
    Gain(f64),
    Agc(bool),
    BiasT(bool),
    Rate(f64),
    Quit,
}

pub struct Radio {
    pub cmd: Sender<Cmd>,
    pub iq: Receiver<Vec<Complex32>>,
    pub log: Receiver<String>,
}

const BLOCK: usize = 16384;

pub fn spawn(args: String, rate: f64, freq: f64) -> Result<Radio> {
    let (cmd_tx, cmd_rx) = channel::<Cmd>();
    let (iq_tx, iq_rx) = sync_channel::<Vec<Complex32>>(8);
    let (log_tx, log_rx) = sync_channel::<String>(64);

    // Open the device on this thread so startup errors surface immediately.
    let dev = soapysdr::Device::new(args.as_str()).context("opening SDR device")?;
    dev.set_sample_rate(Rx, 0, rate).context("setting sample rate")?;
    let _ = set_bandwidth(&dev, rate, &log_tx);
    dev.set_frequency(Rx, 0, freq, ()).context("setting frequency")?;
    // Hardware AGC on the RSP1A pumps on static and FT8 bursts; start
    // in manual gain and let the UI's hang AGC own the level.
    let _ = dev.set_gain_mode(Rx, 0, false);
    let _ = dev.set_gain(Rx, 0, 36.0);
    let _ = dev.write_setting("biasT_ctrl", "false");
    // Ask the driver to null the DC offset itself. Where a device can do that
    // — in hardware, or knowing its own front end — it does it better than
    // anything downstream, and it is the artefact that forces every candidate
    // picker here to blank the bins around the LO. Support is patchy across
    // SoapySDR backends and there is no penalty for asking a device that
    // lacks it. SoapySDR exposes no equivalent automatic mode for IQ balance
    // (only a manual correction value), so that one is always ours.
    // `dsp::FrontEnd` cleans up whatever is left of both regardless.
    let dc_auto = dev.has_dc_offset_mode(Rx, 0).unwrap_or(false)
        && dev.set_dc_offset_mode(Rx, 0, true).is_ok();
    let _ = log_tx.try_send(format!(
        "front end: DC correction {}, IQ balance software",
        if dc_auto { "driver + software" } else { "software" },
    ));

    std::thread::spawn(move || {
        if let Err(e) = run(dev, rate, cmd_rx, iq_tx, log_tx.clone()) {
            let _ = log_tx.try_send(format!("radio thread stopped: {e}"));
        }
    });

    Ok(Radio {
        cmd: cmd_tx,
        iq: iq_rx,
        log: log_rx,
    })
}

fn run(
    dev: soapysdr::Device,
    mut rate: f64,
    cmd_rx: Receiver<Cmd>,
    iq_tx: SyncSender<Vec<Complex32>>,
    log_tx: SyncSender<String>,
) -> Result<()> {
    let mut stream = dev.rx_stream::<Complex32>(&[0])?;
    stream.activate(None)?;
    let mut buf = vec![Complex32::new(0.0, 0.0); BLOCK];
    let mut overruns: u64 = 0;
    // ADC overload watch: samples parked at the rails mean the front end is
    // clipping (classic symptom of bias-T feeding an LNA without backing the
    // gain off), which splatters intermod across the passband and costs weak
    // decodes. Count over ~1 s windows so a single hot block doesn't nag.
    let mut clipped: u64 = 0;
    let mut counted: u64 = 0;
    let mut ovl_quiet = std::time::Instant::now();

    loop {
        // Drain pending commands before the next read.
        loop {
            match cmd_rx.try_recv() {
                Ok(Cmd::Quit) => {
                    let _ = stream.deactivate(None);
                    return Ok(());
                }
                Ok(Cmd::Tune(f)) => {
                    if let Err(e) = dev.set_frequency(Rx, 0, f, ()) {
                        let _ = log_tx.try_send(format!("tune failed: {e}"));
                    }
                }
                Ok(Cmd::Gain(g)) => {
                    let _ = dev.set_gain_mode(Rx, 0, false);
                    if let Err(e) = dev.set_gain(Rx, 0, g) {
                        let _ = log_tx.try_send(format!("gain failed: {e}"));
                    }
                }
                Ok(Cmd::Agc(on)) => {
                    if let Err(e) = dev.set_gain_mode(Rx, 0, on) {
                        let _ = log_tx.try_send(format!("agc failed: {e}"));
                    }
                }
                Ok(Cmd::BiasT(on)) => {
                    let v = if on { "true" } else { "false" };
                    if let Err(e) = dev.write_setting("biasT_ctrl", v) {
                        let _ = log_tx.try_send(format!("bias-T failed: {e}"));
                    } else {
                        let _ = log_tx.try_send(format!("bias-T {v}"));
                    }
                }
                Ok(Cmd::Rate(r)) => {
                    // Sample rate changes need the stream torn down and rebuilt.
                    let _ = stream.deactivate(None);
                    drop(stream);
                    if let Err(e) = dev.set_sample_rate(Rx, 0, r) {
                        let _ = log_tx.try_send(format!("rate failed: {e}"));
                    } else {
                        rate = r;
                        let _ = log_tx.try_send(format!("sample rate {:.0} Hz", r));
                        let _ = set_bandwidth(&dev, r, &log_tx);
                    }
                    stream = dev.rx_stream::<Complex32>(&[0])?;
                    stream.activate(None)?;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    let _ = stream.deactivate(None);
                    return Ok(());
                }
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
                    // More than one sample in ten thousand at the rails, kept
                    // up for a whole second, is overload — not a hot signal.
                    if clipped * 10_000 > counted
                        && ovl_quiet.elapsed() > std::time::Duration::from_secs(15)
                    {
                        let _ = log_tx.try_send(
                            "ADC overload: signals at full scale — reduce gain (bias-T raises it)"
                                .to_string(),
                        );
                        ovl_quiet = std::time::Instant::now();
                    }
                    clipped = 0;
                    counted = 0;
                }
                // Drop blocks rather than block the radio if the UI falls behind.
                match iq_tx.try_send(buf[..n].to_vec()) {
                    Ok(()) => {}
                    Err(TrySendError::Full(_)) => {
                        overruns += 1;
                        if overruns % 200 == 1 {
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
        let _ = rate;
    }
}

/// Keep the tuner's analog IF no wider than the digitised span. Backends are
/// allowed not to support this setting, so failure is deliberately harmless.
fn set_bandwidth(dev: &soapysdr::Device, rate: f64, log_tx: &SyncSender<String>) -> Result<()> {
    dev.set_bandwidth(Rx, 0, rate)?;
    let actual = dev.bandwidth(Rx, 0).unwrap_or(rate);
    let _ = log_tx.try_send(format!("analog bandwidth {:.0} Hz", actual));
    Ok(())
}
