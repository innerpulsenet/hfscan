//! Replay & Benchmark Suite: Evaluates receiver front-end, DSP pipeline,
//! channelizer, and digital decoders (FT8, CW, PSK31, RTTY) against a real-world
//! 60-second 20m live RF recording, with automated baseline comparison & regression detection.

use anyhow::{Context, Result};
use clap::Parser;
use hfscan::bench::{
    BenchmarkMetrics, ChannelizerMetric, CwMetrics, CwSignalResult, DemodThroughputMetric,
    EndToEndMetric, FftMetric, FrontEndMetrics, Ft8Metrics, Ft8SlotResult, IqRecording, MetricStatus,
};
use hfscan::decoders::cw::CwDecoder;
use hfscan::decoders::ft8::{self, AUDIO_CENTRE, AUDIO_RATE, FtDecoder};
use hfscan::decoders::psk31::Psk31Decoder;
use hfscan::decoders::rtty::RttyDecoder;
use hfscan::decoders::Decoder;
use hfscan::dsp::{
    smooth_bins, ChannelTap, Channelizer, FrontEnd, Nco, NoiseFloor, Spectrum,
};
use num_complex::Complex32;
use std::path::PathBuf;
use std::time::Instant;

const DEFAULT_BASELINE_PATH: &str = "captures/20m_baseline_metrics.json";

#[derive(Parser, Debug)]
#[command(
    name = "bench_replay",
    about = "Benchmark receiver DSP and decoders against 60s 20m live RF capture with baseline tracking"
)]
struct Args {
    /// Path to the recorded 20m IQ file
    #[arg(short, long, default_value = "captures/20m_14060khz_192ksps_60s.iq")]
    input: PathBuf,

    /// Save the current run metrics as the reference baseline file
    #[arg(long)]
    save_baseline: Option<Option<PathBuf>>,

    /// Path to baseline metrics JSON to compare against (defaults to captures/20m_baseline_metrics.json if present)
    #[arg(long)]
    baseline: Option<PathBuf>,

    /// Fail with non-zero exit code if any metric regresses beyond tolerance
    #[arg(long)]
    fail_on_regression: bool,

    /// Regression tolerance percentage (default: 5.0%)
    #[arg(long, default_value_t = 5.0)]
    tolerance: f64,

    /// Skip full FT8 decode pass (useful for fast DSP iterations)
    #[arg(long)]
    skip_ft8: bool,

    /// Skip CW decode pass
    #[arg(long)]
    skip_cw: bool,
}

fn print_header(title: &str) {
    println!("\n{}", "=".repeat(78));
    println!("  {}", title);
    println!("{}", "=".repeat(78));
}

fn print_sub(title: &str) {
    println!("\n-- {} {}", title, "-".repeat(74usize.saturating_sub(title.len())));
}

fn main() -> Result<()> {
    let args = Args::parse();

    print_header("HFSCAN RECEIVER & DECODER BENCHMARK (LIVE 20M RF)");

    if !args.input.exists() {
        eprintln!(
            "Error: Capture file '{}' not found!\nRun 'cargo run --release --bin record_iq' first to capture 60s of live 20m RF.",
            args.input.display()
        );
        std::process::exit(1);
    }

    println!("Loading dataset: {} ...", args.input.display());
    let t0 = Instant::now();
    let rec = IqRecording::load_file(&args.input)
        .with_context(|| format!("loading '{}'", args.input.display()))?;
    let load_time = t0.elapsed();

    let total_samples = rec.total_samples();
    let duration = rec.duration_seconds();
    let fs = rec.sample_rate;
    let fc = rec.center_freq;
    let size_mb = (total_samples * 8) as f64 / (1024.0 * 1024.0);

    println!("  Dataset loaded in {:.3}s", load_time.as_secs_f64());
    println!("  Total Samples:    {} ({:.2} MB in RAM)", total_samples, size_mb);
    println!("  Duration:         {:.2} s (60s reference capture)", duration);
    println!("  Center Frequency: {:.3} MHz", fc / 1e6);
    println!("  Sample Rate:      {:.0} S/s ({:.1} kHz span)", fs, fs / 1e3);
    if rec.timestamp_utc > 0 {
        println!("  Capture UTC Time: {}", rec.timestamp_utc);
    }

    // =========================================================================
    // 1. FRONT-END DSP PIPELINE BENCHMARK
    // =========================================================================
    print_sub("1. Front-End DSP Pipeline Benchmark (DC + IQ Corr + Blanker)");
    let frontend_metrics: FrontEndMetrics;
    {
        let mut front = FrontEnd::new(fs);
        let mut test_buf = rec.samples.clone();
        let block_size = 16384;
        let num_blocks = test_buf.len() / block_size;

        let start = Instant::now();
        for chunk in test_buf.chunks_mut(block_size) {
            front.process(chunk);
        }
        let elapsed = start.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0;
        let msps = (total_samples as f64 / elapsed.as_secs_f64()) / 1e6;
        let speedup = duration / elapsed.as_secs_f64();
        let us_per_block = (elapsed.as_micros() as f64) / num_blocks as f64;

        println!("  Processed:   {} samples ({} blocks of {})", total_samples, num_blocks, block_size);
        println!("  Total Time:  {:.2} ms", ms);
        println!("  Throughput:  {:.2} MSamples/sec", msps);
        println!("  Speedup:     {:.1}x real-time ({:.2}% of 1 core at 192 kS/s)", speedup, (100.0 / speedup));
        println!("  Block Lat:   {:.1} µs per 16384-sample block (85.3 ms of RF)", us_per_block);

        frontend_metrics = FrontEndMetrics {
            total_samples,
            elapsed_ms: ms,
            throughput_msps: msps,
            speedup,
            block_latency_us: us_per_block,
        };
    }

    // =========================================================================
    // 2. SPECTRUM FFT & NOISE FLOOR TRACKING BENCHMARK
    // =========================================================================
    print_sub("2. Spectrum FFT & Noise Floor Tracking Benchmark");
    println!("   FFT Size | Transforms |  Total Time |   Throughput |    Frame Latency |  Speedup");
    println!("  ----------+------------+-------------+--------------+------------------+---------");

    let mut fft_metrics = Vec::new();
    for &fft_size in &[1024, 2048, 4096, 8192, 16384, 32768, 65536] {
        let mut spec_engine = Spectrum::new(fft_size);
        let mut spec_out = Vec::with_capacity(fft_size);
        let mut tracker = NoiseFloor::new();
        let mut smoothed = vec![0.0f32; fft_size];

        let num_frames = total_samples / fft_size;
        let dt = fft_size as f32 / fs as f32;
        let start = Instant::now();

        for i in 0..num_frames {
            let chunk = &rec.samples[i * fft_size..(i + 1) * fft_size];
            spec_engine.power_db(chunk, &mut spec_out);
            let _ = tracker.update(&spec_out, dt);
            smooth_bins(&spec_out, 3, &mut smoothed);
        }

        let elapsed = start.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0;
        let msps = ((num_frames * fft_size) as f64 / elapsed.as_secs_f64()) / 1e6;
        let speedup = (num_frames * fft_size) as f64 / (fs * elapsed.as_secs_f64());
        let us_per_frame = (elapsed.as_micros() as f64) / num_frames as f64;

        println!(
            "  {:>9} | {:>10} | {:>8.2} ms | {:>8.2} MS/s | {:>10.1} µs/fr | {:>6.1}x",
            fft_size, num_frames, ms, msps, us_per_frame, speedup
        );

        fft_metrics.push(FftMetric {
            fft_size,
            num_frames,
            elapsed_ms: ms,
            throughput_msps: msps,
            frame_latency_us: us_per_frame,
            speedup,
        });
    }

    // =========================================================================
    // 3. CHANNELIZER MULTI-SLOT SCALING BENCHMARK
    // =========================================================================
    print_sub("3. Channelizer Multi-Slot Scaling Benchmark (192 kS/s -> 8 kHz Channels)");
    println!("   Active Slots | Total Time |   Throughput | Speedup vs Real-Time | Cost per Slot");
    println!("  --------------+------------+--------------+----------------------+---------------");

    let block_size = 16384;
    let mut channelizer_metrics = Vec::new();
    for &num_slots in &[1, 4, 8, 16, 24, 32] {
        let mut channelizer = Channelizer::new(fs);
        let hop = channelizer.hop();
        let mut taps: Vec<ChannelTap> = (0..num_slots)
            .map(|i| {
                let offset_hz = -30_000.0 + (i as f32 * 2_000.0);
                let mut tap = ChannelTap::new(fs, 500.0, 8000.0, hop);
                tap.set_offset(offset_hz as f64);
                tap
            })
            .collect();

        let mut tap_audio: Vec<Vec<Complex32>> = vec![Vec::new(); num_slots];

        let start = Instant::now();
        for block in rec.samples.chunks(block_size) {
            let frames = channelizer.push(block);
            for f in 0..frames {
                let (spec, start_idx) = channelizer.frame(f);
                for (tap_idx, tap) in taps.iter_mut().enumerate() {
                    tap_audio[tap_idx].clear();
                    tap.process_frame(spec, start_idx, &mut tap_audio[tap_idx]);
                }
            }
        }
        let elapsed = start.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0;
        let speedup = duration / elapsed.as_secs_f64();
        let msps = (total_samples as f64 / elapsed.as_secs_f64()) / 1e6;
        let cpu_pct = 100.0 / speedup;
        let per_slot_us = (elapsed.as_micros() as f64) / (num_slots as f64 * (total_samples / block_size) as f64);

        println!(
            "  {:>13} | {:>7.2} ms | {:>8.2} MS/s | {:>16.1}x ({:4.2}% CPU) | {:>7.1} µs/blk",
            num_slots, ms, msps, speedup, cpu_pct, per_slot_us
        );

        channelizer_metrics.push(ChannelizerMetric {
            num_slots,
            elapsed_ms: ms,
            throughput_msps: msps,
            speedup,
            cpu_pct,
            per_slot_cost_us: per_slot_us,
        });
    }

    // =========================================================================
    // 4. FT8 DECODER BENCHMARK (14.074 MHz Calling Frequency)
    // =========================================================================
    let mut ft8_metrics: Option<Ft8Metrics> = None;
    if !args.skip_ft8 {
        print_sub("4. FT8 Decoder Benchmark (14.074 MHz — UTC-Aligned Slots)");
        let ft8_freq = 14_074_000.0;
        let ft8_iq = rec.extract_iq(ft8_freq + AUDIO_CENTRE, 3000.0, AUDIO_RATE);

        let mut nco = Nco::new();
        nco.set_freq(-AUDIO_CENTRE, AUDIO_RATE);
        let mut shifted = Vec::with_capacity(ft8_iq.len());
        nco.mix(&ft8_iq, &mut shifted);
        let ft8_audio: Vec<f32> = shifted.iter().map(|s| s.re).collect();

        let slot_samples = (15.0 * AUDIO_RATE) as usize;
        let utc_stamp = rec.timestamp_utc;
        let start_offset_secs = if utc_stamp > 0 {
            (15.0 - (utc_stamp as f64 % 15.0)) % 15.0
        } else {
            0.0
        };
        let start_sample = (start_offset_secs * AUDIO_RATE) as usize;

        let num_slots = if ft8_audio.len() > start_sample {
            (ft8_audio.len() - start_sample) / slot_samples
        } else {
            0
        };

        println!("  Audio length: {:.2}s | UTC Slot Offset: {:+.1}s | Aligned Slots: {}",
                 ft8_audio.len() as f64 / AUDIO_RATE, start_offset_secs, num_slots);
        println!("   Slot # | Window (UTC) | Decode Time | Speedup | Decodes | Decoded Messages");
        println!("  --------+--------------+-------------+---------+---------+-------------------------------------");

        let mut total_ft8_decodes = 0;
        let mut total_ft8_time = 0.0;
        let mut slot_results = Vec::new();

        for slot_idx in 0..num_slots {
            let slot_start = start_sample + (slot_idx * slot_samples);
            let slot_end = slot_start + slot_samples;
            let slot_raw = &ft8_audio[slot_start..slot_end];
            let slot_i16 = ft8::quantize(slot_raw, slot_samples);

            let t_start = Instant::now();
            let msgs = FtDecoder::decode_slot_messages(&slot_i16, false, true);
            let slot_elapsed = t_start.elapsed();
            let slot_secs = slot_elapsed.as_secs_f64();

            total_ft8_time += slot_secs;
            total_ft8_decodes += msgs.len();

            let speedup = 15.0 / slot_secs;
            let t_window = start_offset_secs + (slot_idx as f64 * 15.0);
            let window_str = format!("{:04.1}s-{:04.1}s", t_window, t_window + 15.0);

            let sample_calls: Vec<String> = msgs.iter().take(3).map(|m| {
                format!("{:+2.0}dB {}", m.snr_db, m.text.trim())
            }).collect();
            let summary = if sample_calls.is_empty() {
                "(no decodes in slot)".into()
            } else {
                sample_calls.join(" | ")
            };

            println!(
                "  {:>6} | {:>12} | {:>8.2} ms | {:>6.1}x | {:>7} | {}",
                slot_idx + 1, window_str, slot_secs * 1000.0, speedup, msgs.len(), summary
            );

            if msgs.len() > 3 {
                for m in msgs.iter().skip(3) {
                    println!("          |              |             |         |         |   {:+2.0}dB {}", m.snr_db, m.text.trim());
                }
            }

            slot_results.push(Ft8SlotResult {
                slot_index: slot_idx + 1,
                window_str,
                decode_time_ms: slot_secs * 1000.0,
                speedup,
                decodes_count: msgs.len(),
                sample_messages: sample_calls,
            });
        }

        let avg_time = total_ft8_time / num_slots.max(1) as f64;
        println!("  ---------------------------------------------------------------------------------");
        println!("  Total FT8 Messages Decoded: {} across 60s", total_ft8_decodes);
        println!("  Average Slot Decode Latency: {:.2} ms ({:.1}x real-time)", avg_time * 1000.0, 15.0 / avg_time);

        ft8_metrics = Some(Ft8Metrics {
            total_decodes: total_ft8_decodes,
            total_slots: num_slots,
            avg_slot_latency_ms: avg_time * 1000.0,
            avg_speedup: 15.0 / avg_time.max(1e-6),
            slots: slot_results,
        });
    }

    // =========================================================================
    // 5. CW DECODER BENCHMARK (14.000 - 14.070 MHz Sub-Band)
    // =========================================================================
    let mut cw_metrics: Option<CwMetrics> = None;
    if !args.skip_cw {
        print_sub("5. CW Decoder Benchmark (20m CW Sub-band)");

        let mut spec_engine = Spectrum::new(8192);
        let mut spec_accum = vec![0.0f32; 8192];
        let mut spec_work = Vec::new();
        let mut spec_frames = 0;

        for chunk in rec.samples.chunks(8192) {
            if chunk.len() == 8192 {
                spec_engine.power_db(chunk, &mut spec_work);
                for (acc, &v) in spec_accum.iter_mut().zip(spec_work.iter()) {
                    *acc += v;
                }
                spec_frames += 1;
            }
        }
        for v in spec_accum.iter_mut() {
            *v /= spec_frames.max(1) as f32;
        }

        let mut sorted = spec_accum.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_noise = sorted[sorted.len() / 2];

        let bin_hz = fs / spec_accum.len() as f64;
        let mut cw_peaks = Vec::new();

        let min_bin = ((spec_accum.len() as f64 / 2.0) - (60_000.0 / bin_hz)) as usize;
        let max_bin = ((spec_accum.len() as f64 / 2.0) + (10_000.0 / bin_hz)) as usize;

        for i in min_bin..max_bin {
            let val = spec_accum[i];
            if val > spec_accum[i - 1] && val > spec_accum[i + 1] && val - median_noise >= 4.0 {
                let off_hz = (i as f64 - spec_accum.len() as f64 / 2.0) * bin_hz;
                cw_peaks.push((fc + off_hz, val - median_noise));
            }
        }
        cw_peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        cw_peaks.truncate(8);

        println!("  Testing CW decoder against {} candidate frequencies:", cw_peaks.len());
        println!("   Dial Freq   | SNR (dB) | Demod Time | Audio MS/s | Speedup | Max Conf | WPM  | Decoded Text");
        println!("  -------------+----------+------------+------------+---------+----------+------+----------------------");

        let mut cw_results = Vec::new();
        let mut total_msps = 0.0;

        for (dial_hz, snr) in cw_peaks {
            let iq_audio = rec.extract_iq(dial_hz, 400.0, 8000.0);
            let mut decoder = CwDecoder::new(8000.0);

            let t_start = Instant::now();
            let mut decoded_text = String::new();
            let mut max_conf = 0.0f32;

            for block in iq_audio.chunks(512) {
                let t = decoder.process(block);
                if !t.is_empty() {
                    let conf = decoder.confidence().unwrap_or(0.0);
                    max_conf = max_conf.max(conf);
                    decoded_text.push_str(&t);
                }
            }

            let elapsed = t_start.elapsed();
            let audio_msps = (iq_audio.len() as f64 / elapsed.as_secs_f64()) / 1e6;
            let speedup = (iq_audio.len() as f64 / 8000.0) / elapsed.as_secs_f64();
            let text_snippet: String = decoded_text.chars().take(24).collect();
            total_msps += audio_msps;

            println!(
                "  {:>9.2} kHz | {:>6.1} dB | {:>7.2} ms | {:>8.2}   | {:>6.0}x | {:>7.0}% | {:>4.1} | {:?}",
                dial_hz / 1e3, snr, elapsed.as_secs_f64() * 1000.0, audio_msps, speedup, max_conf * 100.0, decoder.wpm(), text_snippet
            );

            cw_results.push(CwSignalResult {
                dial_khz: dial_hz / 1e3,
                snr_db: snr,
                demod_time_ms: elapsed.as_secs_f64() * 1000.0,
                audio_msps,
                speedup,
                max_confidence_pct: max_conf * 100.0,
                wpm: decoder.wpm(),
                text_snippet,
            });
        }

        let avg_msps = total_msps / cw_results.len().max(1) as f64;
        cw_metrics = Some(CwMetrics {
            candidate_count: cw_results.len(),
            avg_audio_msps: avg_msps,
            signals: cw_results,
        });
    }

    // =========================================================================
    // 6. PSK31 & RTTY DECODERS BENCHMARK
    // =========================================================================
    print_sub("6. PSK31 & RTTY Demodulator Throughput Benchmark");
    let mut demod_metrics = Vec::new();
    {
        let psk_iq = rec.extract_iq(14_070_000.0, 3000.0, 8000.0);
        let mut psk_dec = Psk31Decoder::new(8000.0);
        let t_start = Instant::now();
        for block in psk_iq.chunks(512) {
            let _ = psk_dec.process(block);
        }
        let psk_elapsed = t_start.elapsed();
        let psk_speedup = (psk_iq.len() as f64 / 8000.0) / psk_elapsed.as_secs_f64();
        let psk_msps = (psk_iq.len() as f64 / psk_elapsed.as_secs_f64()) / 1e6;

        println!("  PSK31 (60s audio @ 8 kHz): {:.2} ms | {:.2} MS/s audio | {:.0}x real-time",
                 psk_elapsed.as_secs_f64() * 1000.0, psk_msps, psk_speedup);

        demod_metrics.push(DemodThroughputMetric {
            mode: "PSK31".into(),
            elapsed_ms: psk_elapsed.as_secs_f64() * 1000.0,
            audio_msps: psk_msps,
            speedup: psk_speedup,
        });

        let rtty_iq = rec.extract_iq(14_085_000.0, 2000.0, 8000.0);
        let mut rtty_dec = RttyDecoder::new(8000.0);
        let t_start = Instant::now();
        for block in rtty_iq.chunks(512) {
            let _ = rtty_dec.process(block);
        }
        let rtty_elapsed = t_start.elapsed();
        let rtty_speedup = (rtty_iq.len() as f64 / 8000.0) / rtty_elapsed.as_secs_f64();
        let rtty_msps = (rtty_iq.len() as f64 / rtty_elapsed.as_secs_f64()) / 1e6;

        println!("  RTTY  (60s audio @ 8 kHz): {:.2} ms | {:.2} MS/s audio | {:.0}x real-time",
                 rtty_elapsed.as_secs_f64() * 1000.0, rtty_msps, rtty_speedup);

        demod_metrics.push(DemodThroughputMetric {
            mode: "RTTY".into(),
            elapsed_ms: rtty_elapsed.as_secs_f64() * 1000.0,
            audio_msps: rtty_msps,
            speedup: rtty_speedup,
        });
    }

    // =========================================================================
    // 7. END-TO-END PIPELINE REAL-TIME REPLAY BENCHMARK
    // =========================================================================
    print_sub("7. End-to-End Replay Simulation (FrontEnd + FFT + 16 Auto Channels)");
    let end_to_end_metric: EndToEndMetric;
    {
        let block_size = 16384;
        let mut front = FrontEnd::new(fs);
        let mut spec_engine = Spectrum::new(8192);
        let mut spec_out = Vec::new();
        let mut channelizer = Channelizer::new(fs);
        let hop = channelizer.hop();
        let mut taps: Vec<ChannelTap> = (0..16)
            .map(|i| {
                let offset_hz = -20_000.0 + (i as f32 * 2_500.0);
                let mut tap = ChannelTap::new(fs, 500.0, 8000.0, hop);
                tap.set_offset(offset_hz as f64);
                tap
            })
            .collect();
        let mut tap_audio: Vec<Vec<Complex32>> = vec![Vec::new(); 16];
        let mut cw_decoders: Vec<CwDecoder> = (0..16).map(|_| CwDecoder::new(8000.0)).collect();

        let t_start = Instant::now();
        let mut block_buf = vec![Complex32::new(0.0, 0.0); block_size];

        for chunk in rec.samples.chunks(block_size) {
            block_buf[..chunk.len()].copy_from_slice(chunk);
            let active = &mut block_buf[..chunk.len()];

            front.process(active);
            spec_engine.power_db(active, &mut spec_out);

            let frames = channelizer.push(active);
            for f in 0..frames {
                let (spec, start_idx) = channelizer.frame(f);
                for (tap_idx, tap) in taps.iter_mut().enumerate() {
                    tap_audio[tap_idx].clear();
                    tap.process_frame(spec, start_idx, &mut tap_audio[tap_idx]);
                    if !tap_audio[tap_idx].is_empty() {
                        let _ = cw_decoders[tap_idx].process(&tap_audio[tap_idx]);
                    }
                }
            }
        }

        let elapsed = t_start.elapsed();
        let sim_time = elapsed.as_secs_f64();
        let speedup = duration / sim_time;
        let cpu_occupancy = 100.0 / speedup;

        println!("  60.0 Seconds of Live 20m RF processed in: {:.3} seconds", sim_time);
        println!("  Overall Processing Speedup:               {:.1}x real-time", speedup);
        println!("  Estimated Single-Core CPU Occupancy:      {:.2}%", cpu_occupancy);

        end_to_end_metric = EndToEndMetric {
            duration_rf_secs: duration,
            process_time_secs: sim_time,
            speedup,
            cpu_occupancy_pct: cpu_occupancy,
        };
    }

    // Assemble full report metrics
    let current_metrics = BenchmarkMetrics {
        timestamp_utc: rec.timestamp_utc,
        dataset_file: args.input.display().to_string(),
        dataset_duration_secs: duration,
        dataset_samples: total_samples,
        frontend: frontend_metrics,
        fft: fft_metrics,
        channelizer: channelizer_metrics,
        ft8: ft8_metrics,
        cw: cw_metrics,
        demodulators: demod_metrics,
        end_to_end: end_to_end_metric,
    };

    // Save baseline if requested
    if let Some(opt_path) = &args.save_baseline {
        let path = opt_path
            .clone()
            .unwrap_or_else(|| PathBuf::from(DEFAULT_BASELINE_PATH));
        current_metrics.save_json(&path)?;
        println!("\n>>> Baseline metrics successfully saved to: {}", path.display());
    }

    // =========================================================================
    // 8. BASELINE COMPARISON & REGRESSION REPORT
    // =========================================================================
    let baseline_path = args.baseline
        .or_else(|| {
            let p = PathBuf::from(DEFAULT_BASELINE_PATH);
            if p.exists() { Some(p) } else { None }
        });

    let mut had_regressions = false;
    if let Some(path) = baseline_path {
        if path.exists() {
            print_header(&format!("BASELINE COMPARISON VS {}", path.display()));
            match BenchmarkMetrics::load_json(&path) {
                Ok(baseline) => {
                    let diffs = baseline.compare(&current_metrics, args.tolerance);

                    println!("   Category       | Metric                      | Baseline   | Current    | Delta (%) | Status");
                    println!("  ----------------+-----------------------------+------------+------------+-----------+-------------");

                    for d in diffs {
                        let status_str = match d.status {
                            MetricStatus::Improved => "\x1b[32mIMPROVED\x1b[0m",
                            MetricStatus::Ok => "PASS",
                            MetricStatus::Regressed => {
                                had_regressions = true;
                                "\x1b[31mREGRESSION\x1b[0m"
                            }
                        };
                        let delta_str = format!("{:>+6.1}%", d.delta_pct);
                        println!(
                            "  {:15} | {:27} | {:>10} | {:>10} | {:>9} | {}",
                            d.category, d.name, d.baseline_val, d.current_val, delta_str, status_str
                        );
                    }

                    if had_regressions {
                        eprintln!("\n\x1b[31m[!] WARNING: One or more performance/accuracy regressions detected vs baseline!\x1b[0m");
                        if args.fail_on_regression {
                            eprintln!("Failing build as requested by --fail-on-regression.");
                            std::process::exit(1);
                        }
                    } else {
                        println!("\n\x1b[32m[✓] All metrics meet or exceed the baseline standards (tolerance: {:.1}%).\x1b[0m", args.tolerance);
                    }
                }
                Err(e) => {
                    eprintln!("Warning: could not load baseline file '{}': {e}", path.display());
                }
            }
        }
    }

    print_header("BENCHMARK COMPLETED SUCCESSFULLY");
    Ok(())
}
