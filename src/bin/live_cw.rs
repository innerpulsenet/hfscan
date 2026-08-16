//! Live CW probe against the SDRplay hardware on 20m with bias-T enabled.

use anyhow::{Context, Result};
use hfscan::decoders::cw::CwDecoder;
use hfscan::decoders::Decoder;
use hfscan::dsp::{DecodeChain, Spectrum};
use num_complex::Complex32;
use soapysdr::Direction::Rx;
use std::time::Instant;

fn main() -> Result<()> {
    println!("Connecting to SDRplay device...");
    let dev = soapysdr::Device::new("driver=sdrplay").context("opening SDRplay device")?;
    println!("Device opened: {}", dev.hardware_key().unwrap_or_default());

    // 1. Enable Bias-T
    println!("Enabling Bias-T (biasT_ctrl = true)...");
    if let Err(e) = dev.write_setting("biasT_ctrl", "true") {
        eprintln!("Warning: biasT_ctrl failed: {e}");
    } else {
        println!("Bias-T successfully enabled.");
    }

    // 2. Tune to 20m (14.035 MHz, 192 kS/s covers 13.939 to 14.131 MHz)
    let center_freq = 14_035_000.0;
    let sample_rate = 192_000.0;
    dev.set_frequency(Rx, 0, center_freq, ())?;
    dev.set_sample_rate(Rx, 0, sample_rate)?;
    let _ = dev.set_bandwidth(Rx, 0, sample_rate);

    // Set gains: SDRplay reduction controls (rfgr = 0, ifgr = 48 for 20m)
    let _ = dev.set_gain_element(Rx, 0, "IFGR", 48.0);
    let _ = dev.set_gain_element(Rx, 0, "RFGR", 0.0);

    let actual_freq = dev.frequency(Rx, 0)?;
    let actual_rate = dev.sample_rate(Rx, 0)?;
    println!("Tuned to {actual_freq:.0} Hz at {actual_rate:.0} S/s");

    // 3. Setup Rx stream
    let mut stream = dev.rx_stream::<Complex32>(&[0])?;
    stream.activate(None)?;
    let mtu = stream.mtu()?;
    println!("Stream active, MTU = {mtu}");

    let mut spectrum = Spectrum::new(8192);
    let mut buffer = vec![Complex32::new(0.0, 0.0); mtu];
    let mut recorded_blocks: Vec<Vec<Complex32>> = Vec::new();
    let mut spec = Vec::new();

    println!("\nCapturing live RF for 15 seconds on 20m...");
    let start = Instant::now();
    let mut total_samples = 0usize;

    while start.elapsed().as_secs() < 15 {
        match stream.read(&mut [&mut buffer], 1_000_000) {
            Ok(n) if n > 0 => {
                let chunk = buffer[..n].to_vec();
                spectrum.power_db(&chunk, &mut spec);
                recorded_blocks.push(chunk);
                total_samples += n;
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("Stream read warning: {e}");
            }
        }
    }
    stream.deactivate(None)?;
    println!("Capture finished: {total_samples} samples collected ({:.1} seconds).", start.elapsed().as_secs_f32());

    // 4. Spectrum analysis & peak finding
    let mut sorted_spec = spec.clone();
    sorted_spec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_noise = sorted_spec[sorted_spec.len() / 2];
    println!("Median noise floor on 20m: {median_noise:.1} dBFS");

    let bin_hz = sample_rate / spec.len() as f64;
    let mut candidate_peaks: Vec<(f32, f32)> = Vec::new(); // (offset_hz, snr_db)

    // Search 14.000 to 14.070 MHz (offset: -35 kHz to +35 kHz relative to 14.035 MHz)
    let min_bin = ((spec.len() as f64 / 2.0) - (35_000.0 / bin_hz)).max(10.0) as usize;
    let max_bin = ((spec.len() as f64 / 2.0) + (35_000.0 / bin_hz)).min(spec.len() as f64 - 10.0) as usize;

    for i in min_bin..=max_bin {
        let val = spec[i];
        if val > spec[i - 1] && val > spec[i + 1] && val - median_noise >= 4.0 {
            let offset_hz = ((i as f64 - spec.len() as f64 / 2.0) * bin_hz) as f32;
            let snr = val - median_noise;
            candidate_peaks.push((offset_hz, snr));
        }
    }

    candidate_peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidate_peaks.truncate(20);

    println!("\n== Found {} CW candidate peaks on 20m CW sub-band (14.000 - 14.070 MHz) ==", candidate_peaks.len());
    for (off_hz, snr) in &candidate_peaks {
        let dial = center_freq + *off_hz as f64;
        println!("  Candidate at {dial:.1} Hz (offset {:+.1} Hz): SNR = {:+.1} dB", off_hz, snr);
    }

    // 5. Run CW decoder on each candidate peak
    println!("\n== Decoding CW candidates across the 15-second capture ==");
    let audio_fs = 8000.0;
    for (off_hz, snr) in candidate_peaks {
        let dial = center_freq + off_hz as f64;
        let mut decoder = CwDecoder::new(audio_fs);
        let mut chain = DecodeChain::new(sample_rate, decoder.bandwidth(), audio_fs);
        chain.set_offset(off_hz as f64 + decoder.offset_shift());

        let mut audio_buf = Vec::new();
        let mut text_output = String::new();
        let mut max_conf = 0.0f32;

        for block in &recorded_blocks {
            chain.process(block, &mut audio_buf);
            let t = decoder.process(&audio_buf);
            if !t.is_empty() {
                let conf = decoder.confidence().unwrap_or(0.0);
                max_conf = max_conf.max(conf);
                text_output.push_str(&t);
            }
        }

        let status = decoder.status();
        let wpm = decoder.wpm();

        println!("------------------------------------------------------------");
        println!("Freq: {dial:.1} Hz | SNR: {snr:.1} dB | Max Conf: {:.0}% | {status} | WPM: {wpm:.1}", max_conf * 100.0);
        println!("Decoded text: {:?}", text_output.trim());
    }

    Ok(())
}
