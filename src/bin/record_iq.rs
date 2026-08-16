//! Live IQ capture utility: streams raw complex IQ samples from an SDR into a file
//! with companion JSON metadata for benchmarking and offline analysis.

use anyhow::{Context, Result};
use clap::Parser;
use hfscan::dsp::Spectrum;
use num_complex::Complex32;
use soapysdr::Direction::Rx;
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Parser, Debug)]
#[command(
    name = "record_iq",
    about = "Record live RF IQ spectrum data to file for benchmarking"
)]
struct Args {
    /// SoapySDR device arguments
    #[arg(long, default_value = "driver=sdrplay")]
    device: String,

    /// Center frequency in Hz (defaults to 14.060 MHz for 20m)
    #[arg(short, long, default_value_t = 14_060_000.0)]
    freq: f64,

    /// Sample rate in Hz (defaults to 192 kS/s for 20m)
    #[arg(short, long, default_value_t = 192_000.0)]
    rate: f64,

    /// Duration of capture in seconds
    #[arg(short, long, default_value_t = 60.0)]
    duration: f64,

    /// SDRplay IF gain reduction in dB (larger means less gain; 48 dB is optimal for 20m)
    #[arg(long, default_value_t = 48.0)]
    ifgr: f64,

    /// SDRplay RF gain reduction in dB (0 means full RF front-end gain)
    #[arg(long, default_value_t = 0.0)]
    rfgr: f64,

    /// Enable SDR Bias-T for active antennas
    #[arg(long, default_value_t = true)]
    biast: bool,

    /// Analog filter bandwidth in Hz (0 = match sample rate)
    #[arg(long, default_value_t = 200_000.0)]
    bandwidth: f64,

    /// Output IQ file path
    #[arg(short, long, default_value = "captures/20m_14060khz_192ksps_60s.iq")]
    output: PathBuf,

    /// Output metadata JSON file path
    #[arg(short, long, default_value = "captures/20m_14060khz_192ksps_60s.json")]
    meta: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    soapysdr::configure_logging();

    println!("============================================================");
    println!("  Live RF IQ Recorder — 60s 20m Spectrum Benchmark Capture  ");
    println!("============================================================");
    println!("Device:      {}", args.device);
    println!("Frequency:   {:.3} MHz ({:.0} Hz)", args.freq / 1e6, args.freq);
    println!("Sample Rate: {:.0} S/s ({:.1} kHz span)", args.rate, args.rate / 1e3);
    println!("Duration:    {:.1} seconds", args.duration);
    println!("IF Gain Red: {:.1} dB", args.ifgr);
    println!("RF Gain Red: {:.1} dB", args.rfgr);
    println!("Bias-T:      {}", if args.biast { "enabled" } else { "disabled" });
    println!("Output:      {}", args.output.display());
    println!("------------------------------------------------------------");

    println!("Opening SoapySDR device...");
    let dev = soapysdr::Device::new(args.device.as_str())
        .with_context(|| format!("opening SDR device '{}'", args.device))?;

    let driver = dev.driver_key().unwrap_or_else(|_| "unknown".into());
    let hardware = dev.hardware_key().unwrap_or_else(|_| "unknown".into());
    println!("Connected to: {driver} ({hardware})");

    if args.biast {
        if let Err(e) = dev.write_setting("biasT_ctrl", "true") {
            eprintln!("Warning: could not set biasT_ctrl: {e}");
        } else {
            println!("Bias-T powered ON.");
        }
    }

    dev.set_sample_rate(Rx, 0, args.rate)
        .context("setting sample rate")?;
    let actual_rate = dev.sample_rate(Rx, 0).unwrap_or(args.rate);

    let bw = if args.bandwidth > 0.0 { args.bandwidth } else { actual_rate };
    let _ = dev.set_bandwidth(Rx, 0, bw);
    let actual_bw = dev.bandwidth(Rx, 0).unwrap_or(bw);

    dev.set_frequency(Rx, 0, args.freq, ())
        .context("setting center frequency")?;
    let actual_freq = dev.frequency(Rx, 0).unwrap_or(args.freq);

    // Disable hardware AGC if supported so manual gains take effect
    let _ = dev.set_gain_mode(Rx, 0, false);
    let _ = dev.set_gain_element(Rx, 0, "IFGR", args.ifgr);
    let _ = dev.set_gain_element(Rx, 0, "RFGR", args.rfgr);

    // Turn on RF notch if available on HF (> 2 MHz)
    let _ = dev.write_setting("rfnotch_ctrl", if args.freq >= 2_000_000.0 { "true" } else { "false" });

    println!("Receiver configured:");
    println!("  Actual Center Freq: {:.1} Hz", actual_freq);
    println!("  Actual Sample Rate: {:.1} S/s", actual_rate);
    println!("  Actual Bandwidth:   {:.1} Hz", actual_bw);

    // Pre-calculate sample target
    let target_samples = (args.duration * actual_rate).round() as usize;
    println!("Target samples: {} ({:.2} MB in RAM)", target_samples, (target_samples * 8) as f64 / (1024.0 * 1024.0));

    let mut stream = dev.rx_stream::<Complex32>(&[0])
        .context("activating RX stream")?;
    stream.activate(None)
        .context("activating stream")?;
    let mtu = stream.mtu().unwrap_or(16384);

    let mut samples: Vec<Complex32> = Vec::with_capacity(target_samples + mtu);
    let mut rx_buf = vec![Complex32::new(0.0, 0.0); mtu];

    let mut spectrum = Spectrum::new(8192);
    let mut spec = Vec::new();
    let mut spec_accum = vec![0.0f32; 8192];
    let mut spec_count = 0usize;

    println!("\n>>> Recording started (capturing {:.1}s)...", args.duration);
    let start_time = Instant::now();
    let utc_timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut last_progress = Instant::now();
    let mut total_read = 0usize;

    while total_read < target_samples {
        let to_read = (target_samples - total_read).min(mtu);
        match stream.read(&mut [&mut rx_buf[..to_read]], 1_000_000) {
            Ok(n) if n > 0 => {
                let chunk = &rx_buf[..n];
                samples.extend_from_slice(chunk);
                total_read += n;

                spectrum.power_db(chunk, &mut spec);
                if spec.len() == spec_accum.len() {
                    for (acc, &val) in spec_accum.iter_mut().zip(spec.iter()) {
                        *acc += val;
                    }
                    spec_count += 1;
                }

                if last_progress.elapsed() >= Duration::from_millis(1000) {
                    let elapsed = start_time.elapsed().as_secs_f64();
                    let pct = (total_read as f64 / target_samples as f64) * 100.0;
                    let current_rate = total_read as f64 / elapsed;
                    print!("\r  Progress: {:5.1}% | {:8} / {} samples | {:4.1}s / {:4.1}s | Rate: {:.1} kS/s  ",
                           pct, total_read, target_samples, elapsed, args.duration, current_rate / 1e3);
                    let _ = std::io::stdout().flush();
                    last_progress = Instant::now();
                }
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("\nWarning: stream read error: {e}");
            }
        }
    }

    let elapsed = start_time.elapsed();
    let _ = stream.deactivate(None);
    println!("\n>>> Recording complete! Collected {} samples in {:.2}s ({:.1} kS/s).",
             samples.len(), elapsed.as_secs_f64(), (samples.len() as f64 / elapsed.as_secs_f64()) / 1e3);

    // Compute average spectrum & noise floor
    if spec_count > 0 {
        for val in spec_accum.iter_mut() {
            *val /= spec_count as f32;
        }
    }

    let mut sorted_spec = spec_accum.clone();
    sorted_spec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_noise = if !sorted_spec.is_empty() {
        sorted_spec[sorted_spec.len() / 2]
    } else {
        -70.0
    };

    println!("\nSpectrum analysis:");
    println!("  Median Noise Floor: {:.1} dBFS", median_noise);

    // Find peak activity markers
    let bin_hz = actual_rate / spec_accum.len() as f64;
    let mut peaks: Vec<(f64, f32)> = Vec::new(); // (freq_hz, snr_db)
    for i in 1..(spec_accum.len() - 1) {
        let val = spec_accum[i];
        if val > spec_accum[i - 1] && val > spec_accum[i + 1] && val - median_noise >= 6.0 {
            let offset_hz = (i as f64 - spec_accum.len() as f64 / 2.0) * bin_hz;
            let freq_hz = actual_freq + offset_hz;
            let snr = val - median_noise;
            peaks.push((freq_hz, snr));
        }
    }
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("  Found {} prominent carriers (> 6 dB SNR) across 20m span:", peaks.len());
    for (f, snr) in peaks.iter().take(10) {
        println!("    {:.3} kHz ({:+.1} kHz from LO): SNR = {:+.1} dB", f / 1e3, (f - actual_freq) / 1e3, snr);
    }

    // Save binary IQ file
    if let Some(parent) = args.output.parent() {
        create_dir_all(parent)
            .with_context(|| format!("creating output directory '{}'", parent.display()))?;
    }

    println!("\nWriting raw IQ data to {}...", args.output.display());
    let file = File::create(&args.output)
        .with_context(|| format!("creating output file '{}'", args.output.display()))?;
    let mut writer = BufWriter::with_capacity(1024 * 1024, file);

    for s in &samples {
        writer.write_all(&s.re.to_le_bytes())?;
        writer.write_all(&s.im.to_le_bytes())?;
    }
    writer.flush()?;

    let bytes_written = std::fs::metadata(&args.output)?.len();
    println!("Saved {:.2} MB ({bytes_written} bytes) to {}", bytes_written as f64 / (1024.0 * 1024.0), args.output.display());

    // Save JSON metadata
    if let Some(parent) = args.meta.parent() {
        create_dir_all(parent)?;
    }

    let meta_json = format!(
        r#"{{
  "format": "Complex32_LE",
  "sample_format": "2 x float32 (I, Q)",
  "bytes_per_sample": 8,
  "center_frequency_hz": {:.1},
  "sample_rate_hz": {:.1},
  "analog_bandwidth_hz": {:.1},
  "duration_seconds": {:.3},
  "total_samples": {},
  "timestamp_utc": {},
  "receiver": {{
    "driver": {:?},
    "hardware": {:?},
    "ifgr_db": {:.1},
    "rfgr_db": {:.1},
    "biast": {}
  }},
  "spectrum_stats": {{
    "median_noise_floor_dbfs": {:.2},
    "prominent_carriers_count": {}
  }}
}}
"#,
        actual_freq,
        actual_rate,
        actual_bw,
        elapsed.as_secs_f64(),
        samples.len(),
        utc_timestamp,
        driver,
        hardware,
        args.ifgr,
        args.rfgr,
        args.biast,
        median_noise,
        peaks.len()
    );

    std::fs::write(&args.meta, meta_json)
        .with_context(|| format!("writing metadata JSON to '{}'", args.meta.display()))?;
    println!("Saved metadata to {}", args.meta.display());

    println!("\nCapture finished successfully! The file is ready for benchmarking.");
    Ok(())
}
