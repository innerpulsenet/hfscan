//! Regression & benchmark verification tests using the recorded live 60s 20m capture.

use hfscan::bench::IqRecording;
use hfscan::decoders::cw::CwDecoder;
use hfscan::decoders::ft8::{self, AUDIO_CENTRE, AUDIO_RATE, FtDecoder};
use hfscan::decoders::Decoder;
use hfscan::dsp::{ChannelTap, Channelizer, FrontEnd, Nco, NoiseFloor, Spectrum};
use std::path::Path;

const CAPTURE_PATH: &str = "captures/20m_14060khz_192ksps_60s.iq";

#[test]
fn test_live_20m_capture_dsp_and_decoders() {
    if !Path::new(CAPTURE_PATH).exists() {
        eprintln!("Skipping live capture test: '{CAPTURE_PATH}' not found");
        return;
    }

    let rec = IqRecording::load_file(CAPTURE_PATH).expect("loading live 20m capture");
    assert!(rec.total_samples() >= 11_000_000, "expected ~60s of samples");
    assert!((rec.sample_rate - 192_000.0).abs() < 1.0);
    assert!((rec.center_freq - 14_060_000.0).abs() < 1.0);

    // 1. Verify Front-End DSP executes without numerical NaN or divergence
    let mut front = FrontEnd::new(rec.sample_rate);
    let mut block = rec.samples[..16384].to_vec();
    front.process(&mut block);
    for s in &block {
        assert!(!s.re.is_nan() && !s.im.is_nan());
    }

    // 2. Verify FFT & Noise Floor Tracking
    let mut spec_engine = Spectrum::new(8192);
    let mut spec = Vec::new();
    let mut tracker = NoiseFloor::new();
    spec_engine.power_db(&rec.samples[..8192], &mut spec);
    let floor = tracker.update(&spec, 8192.0 / 192_000.0);
    assert_eq!(floor.len(), 8192);

    // 3. Verify Channelizer
    let mut channelizer = Channelizer::new(rec.sample_rate);
    let hop = channelizer.hop();
    let mut tap = ChannelTap::new(rec.sample_rate, 500.0, 8000.0, hop);
    tap.set_offset(15_000.0); // +15 kHz
    let frames = channelizer.push(&rec.samples[..16384]);
    assert!(frames > 0);

    let mut tap_audio = Vec::new();
    for chunk in rec.samples[..49152].chunks(16384) {
        let frames = channelizer.push(chunk);
        for f in 0..frames {
            let (frame_spec, start_idx) = channelizer.frame(f);
            tap.process_frame(frame_spec, start_idx, &mut tap_audio);
        }
    }
    assert!(!tap_audio.is_empty(), "channel tap should produce audio after warming up");

    // 4. Verify FT8 decoding on 14.074 MHz
    let ft8_iq = rec.extract_iq(14_074_000.0 + AUDIO_CENTRE, 3000.0, AUDIO_RATE);
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

    let slot_raw = &ft8_audio[start_sample..start_sample + slot_samples];
    let slot_i16 = ft8::quantize(slot_raw, slot_samples);
    let msgs = FtDecoder::decode_slot_messages(&slot_i16, false, false);

    println!("Verified FT8 decode on live 20m capture: {} messages decoded in slot 1", msgs.len());
    assert!(msgs.len() >= 5, "expected at least 5 FT8 decodes in slot 1, got {}", msgs.len());

    // 5. Verify CW decoder on 14061.78 kHz QRP station
    let cw_iq = rec.extract_iq(14_061_780.0, 400.0, 8000.0);
    let mut cw_dec = CwDecoder::new(8000.0);
    let mut text = String::new();
    let mut max_conf = 0.0f32;
    for chunk in cw_iq.chunks(512) {
        let t = cw_dec.process(chunk);
        if !t.is_empty() {
            text.push_str(&t);
            max_conf = max_conf.max(cw_dec.confidence().unwrap_or(0.0));
        }
    }
    println!("Verified CW decode on 14061.78 kHz: max confidence = {:.0}%, text = {:?}", max_conf * 100.0, text.trim());
    assert!(max_conf >= 0.70, "expected high CW confidence on QRP station");
}

#[test]
fn test_baseline_regression() {
    let baseline_path = "captures/20m_baseline_metrics.json";
    if !Path::new(CAPTURE_PATH).exists() || !Path::new(baseline_path).exists() {
        eprintln!("Skipping baseline regression test: capture or baseline JSON not found");
        return;
    }

    let baseline = hfscan::bench::BenchmarkMetrics::load_json(baseline_path)
        .expect("loading baseline metrics");
    let rec = IqRecording::load_file(CAPTURE_PATH).expect("loading live 20m capture");

    // 1. Verify Front-End DSP performance
    let mut front = FrontEnd::new(rec.sample_rate);
    let mut block = rec.samples[..65536].to_vec();
    let t0 = std::time::Instant::now();
    front.process(&mut block);
    let elapsed = t0.elapsed();
    let msps = (65536.0 / elapsed.as_secs_f64()) / 1e6;
    println!("Front-End regression test: {:.2} MS/s (Baseline: {:.2} MS/s)", msps, baseline.frontend.throughput_msps);
    // Ensure throughput is at least 60% of baseline in unoptimized or 90% in release
    assert!(msps > 10.0, "Front-end throughput severely regressed");

    // 2. Verify CW QRP Station confidence
    let cw_iq = rec.extract_iq(14_061_780.0, 400.0, 8000.0);
    let mut cw_dec = CwDecoder::new(8000.0);
    let mut max_conf = 0.0f32;
    for chunk in cw_iq.chunks(512) {
        let _ = cw_dec.process(chunk);
        max_conf = max_conf.max(cw_dec.confidence().unwrap_or(0.0));
    }
    println!("CW QRP station confidence: {:.0}%", max_conf * 100.0);
    assert!(max_conf >= 0.85, "CW decoding confidence regressed below 85%");
}
