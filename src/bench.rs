//! Receiver characterisation, run against the hardware instead of the TUI.
//!
//! Three questions, in the order they matter:
//!
//! 1. What does this receiver actually offer? Sample rates, filter widths,
//!    gain elements and their ranges — read from the driver rather than
//!    assumed. Asking for a sample rate the device cannot produce gets it
//!    silently clamped, and a band plan built on a rate that never happened
//!    puts the digital segment outside the view. That is not hypothetical:
//!    it is how FT8 decoding was lost here, and this report would have caught
//!    it in one run.
//!
//! 2. Does the band plan survive contact with the driver? Every preset is
//!    requested in turn and the achieved rate and filter width read back, then
//!    checked against what the plan assumed.
//!
//! 3. Where should the gain sit? Sweeping the gain and watching the noise
//!    floor finds the point where the receiver stops adding its own noise and
//!    starts amplifying the band's — above that, gain buys intermodulation
//!    and nothing else.
//!
//! Everything here is device-agnostic. The RSP1A's split RF/IF gain reduction
//! and an RTL-SDR's single tuner gain are both just gain elements to sweep,
//! and the analysis is the same either way.
//!
//! What this cannot do is measure a noise figure or an MDS in absolute terms.
//! Both need a calibrated source; without one the floor is only known relative
//! to full scale, which is enough to find the knee and not enough to put a
//! number in dBm on it.

use crate::bands;
use anyhow::{Context, Result};
use num_complex::Complex32;
use soapysdr::Direction::Rx;
use std::path::Path;

/// In-memory representation of an IQ recording for benchmarks and offline testing.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct IqRecording {
    pub samples: Vec<Complex32>,
    pub center_freq: f64,
    pub sample_rate: f64,
    pub duration_secs: f64,
    pub timestamp_utc: u64,
}

#[allow(dead_code)]
impl IqRecording {
    /// Create an IQ recording from raw samples in memory.
    pub fn from_samples(samples: Vec<Complex32>, center_freq: f64, sample_rate: f64) -> Self {
        let duration_secs = if sample_rate > 0.0 {
            samples.len() as f64 / sample_rate
        } else {
            0.0
        };
        Self {
            samples,
            center_freq,
            sample_rate,
            duration_secs,
            timestamp_utc: 0,
        }
    }

    /// Load raw little-endian Complex32 samples from an `.iq` file.
    /// Also checks for a companion `.json` metadata file in the same directory.
    pub fn load_file(iq_path: impl AsRef<Path>) -> Result<Self> {
        let path = iq_path.as_ref();
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading IQ file '{}'", path.display()))?;

        let sample_count = bytes.len() / 8;
        let mut samples = Vec::with_capacity(sample_count);

        for chunk in bytes.chunks_exact(8) {
            let re = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let im = f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
            samples.push(Complex32::new(re, im));
        }

        let mut center_freq = 14_060_000.0;
        let mut sample_rate = 192_000.0;
        let mut timestamp_utc = 0u64;

        let json_path = path.with_extension("json");
        if let Ok(meta_str) = std::fs::read_to_string(&json_path) {
            if let Some(pos) = meta_str.find("\"center_frequency_hz\":") {
                let rest = &meta_str[pos + 22..];
                if let Some(end) = rest.find(|c: char| c == ',' || c == '\n' || c == '}') {
                    if let Ok(f) = rest[..end].trim().parse::<f64>() {
                        center_freq = f;
                    }
                }
            }
            if let Some(pos) = meta_str.find("\"sample_rate_hz\":") {
                let rest = &meta_str[pos + 17..];
                if let Some(end) = rest.find(|c: char| c == ',' || c == '\n' || c == '}') {
                    if let Ok(r) = rest[..end].trim().parse::<f64>() {
                        sample_rate = r;
                    }
                }
            }
            if let Some(pos) = meta_str.find("\"timestamp_utc\":") {
                let rest = &meta_str[pos + 16..];
                if let Some(end) = rest.find(|c: char| c == ',' || c == '\n' || c == '}') {
                    if let Ok(t) = rest[..end].trim().parse::<u64>() {
                        timestamp_utc = t;
                    }
                }
            }
        }

        let duration_secs = if sample_rate > 0.0 {
            samples.len() as f64 / sample_rate
        } else {
            0.0
        };

        Ok(Self {
            samples,
            center_freq,
            sample_rate,
            duration_secs,
            timestamp_utc,
        })
    }

    pub fn total_samples(&self) -> usize {
        self.samples.len()
    }

    pub fn duration_seconds(&self) -> f64 {
        self.duration_secs
    }

    /// Iterator over slices of IQ samples sized to `block_size`.
    pub fn block_iter<'a>(&'a self, block_size: usize) -> impl Iterator<Item = &'a [Complex32]> {
        self.samples.chunks(block_size)
    }

    /// Extract demodulated audio (real part) for a specific dial frequency using the DSP `DecodeChain`.
    pub fn extract_audio(&self, dial_hz: f64, bandwidth_hz: f32, audio_rate: f64) -> Vec<f32> {
        let mut chain = crate::dsp::DecodeChain::new(self.sample_rate, bandwidth_hz, audio_rate);
        let offset = dial_hz - self.center_freq;
        chain.set_offset(offset);

        let mut out = Vec::with_capacity((self.duration_secs * audio_rate) as usize + 1024);
        let mut block_iq = Vec::new();
        for block in self.block_iter(16384) {
            chain.process(block, &mut block_iq);
            out.extend(block_iq.iter().map(|s| s.re));
        }
        out
    }

    /// Extract demodulated complex baseband IQ for a specific dial frequency using the DSP `DecodeChain`.
    pub fn extract_iq(&self, dial_hz: f64, bandwidth_hz: f32, audio_rate: f64) -> Vec<Complex32> {
        let mut chain = crate::dsp::DecodeChain::new(self.sample_rate, bandwidth_hz, audio_rate);
        let offset = dial_hz - self.center_freq;
        chain.set_offset(offset);

        let mut out = Vec::with_capacity((self.duration_secs * audio_rate) as usize + 1024);
        let mut block_iq = Vec::new();
        for block in self.block_iter(16384) {
            chain.process(block, &mut block_iq);
            out.extend_from_slice(&block_iq);
        }
        out
    }
}

use serde::{Deserialize, Serialize};

#[allow(dead_code)]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FrontEndMetrics {
    pub total_samples: usize,
    pub elapsed_ms: f64,
    pub throughput_msps: f64,
    pub speedup: f64,
    pub block_latency_us: f64,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FftMetric {
    pub fft_size: usize,
    pub num_frames: usize,
    pub elapsed_ms: f64,
    pub throughput_msps: f64,
    pub frame_latency_us: f64,
    pub speedup: f64,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChannelizerMetric {
    pub num_slots: usize,
    pub elapsed_ms: f64,
    pub throughput_msps: f64,
    pub speedup: f64,
    pub cpu_pct: f64,
    pub per_slot_cost_us: f64,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Ft8SlotResult {
    pub slot_index: usize,
    pub window_str: String,
    pub decode_time_ms: f64,
    pub speedup: f64,
    pub decodes_count: usize,
    pub sample_messages: Vec<String>,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Ft8Metrics {
    pub total_decodes: usize,
    pub total_slots: usize,
    pub avg_slot_latency_ms: f64,
    pub avg_speedup: f64,
    pub slots: Vec<Ft8SlotResult>,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CwSignalResult {
    pub dial_khz: f64,
    pub snr_db: f32,
    pub demod_time_ms: f64,
    pub audio_msps: f64,
    pub speedup: f64,
    pub max_confidence_pct: f32,
    pub wpm: f32,
    pub text_snippet: String,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CwMetrics {
    pub candidate_count: usize,
    pub avg_audio_msps: f64,
    pub signals: Vec<CwSignalResult>,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DemodThroughputMetric {
    pub mode: String,
    pub elapsed_ms: f64,
    pub audio_msps: f64,
    pub speedup: f64,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EndToEndMetric {
    pub duration_rf_secs: f64,
    pub process_time_secs: f64,
    pub speedup: f64,
    pub cpu_occupancy_pct: f64,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkMetrics {
    pub timestamp_utc: u64,
    pub dataset_file: String,
    pub dataset_duration_secs: f64,
    pub dataset_samples: usize,
    pub frontend: FrontEndMetrics,
    pub fft: Vec<FftMetric>,
    pub channelizer: Vec<ChannelizerMetric>,
    pub ft8: Option<Ft8Metrics>,
    pub cw: Option<CwMetrics>,
    pub demodulators: Vec<DemodThroughputMetric>,
    pub end_to_end: EndToEndMetric,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetricStatus {
    Improved,
    Ok,
    Regressed,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct MetricDiff {
    pub category: &'static str,
    pub name: String,
    pub baseline_val: String,
    pub current_val: String,
    pub delta_pct: f64,
    pub status: MetricStatus,
    pub note: &'static str,
}

#[allow(dead_code)]
impl BenchmarkMetrics {
    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)
            .context("serializing benchmark metrics to JSON")?;
        std::fs::write(path, json)
            .with_context(|| format!("writing benchmark metrics to '{}'", path.display()))?;
        Ok(())
    }

    pub fn load_json(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("reading benchmark metrics from '{}'", path.display()))?;
        let metrics = serde_json::from_str(&content)
            .with_context(|| format!("parsing JSON benchmark metrics from '{}'", path.display()))?;
        Ok(metrics)
    }

    /// Compare `current` metrics against `self` (baseline).
    /// `tolerance_pct` defines the regression margin (e.g. 5.0%).
    pub fn compare(&self, current: &BenchmarkMetrics, tolerance_pct: f64) -> Vec<MetricDiff> {
        let mut diffs = Vec::new();

        let add_higher_better = |diffs: &mut Vec<MetricDiff>, cat: &'static str, name: String, base: f64, cur: f64, unit: &str, tol: f64| {
            let delta = if base.abs() > 1e-9 { ((cur - base) / base) * 100.0 } else { 0.0 };
            let status = if delta >= tol {
                MetricStatus::Improved
            } else if delta <= -tol {
                MetricStatus::Regressed
            } else {
                MetricStatus::Ok
            };
            diffs.push(MetricDiff {
                category: cat,
                name,
                baseline_val: format!("{:.2}{}", base, unit),
                current_val: format!("{:.2}{}", cur, unit),
                delta_pct: delta,
                status,
                note: if status == MetricStatus::Regressed { "throughput dropped" } else { "" },
            });
        };

        let add_lower_better = |diffs: &mut Vec<MetricDiff>, cat: &'static str, name: String, base: f64, cur: f64, unit: &str, tol: f64| {
            let delta = if base.abs() > 1e-9 { ((cur - base) / base) * 100.0 } else { 0.0 };
            let status = if delta <= -tol {
                MetricStatus::Improved
            } else if delta >= tol {
                MetricStatus::Regressed
            } else {
                MetricStatus::Ok
            };
            diffs.push(MetricDiff {
                category: cat,
                name,
                baseline_val: format!("{:.2}{}", base, unit),
                current_val: format!("{:.2}{}", cur, unit),
                delta_pct: delta,
                status,
                note: if status == MetricStatus::Regressed { "latency increased" } else { "" },
            });
        };

        // 1. Front-End
        add_higher_better(&mut diffs, "DSP Front-End", "Throughput".into(), self.frontend.throughput_msps, current.frontend.throughput_msps, " MS/s", tolerance_pct);
        add_lower_better(&mut diffs, "DSP Front-End", "Block Latency".into(), self.frontend.block_latency_us, current.frontend.block_latency_us, " µs", tolerance_pct);

        // 2. Spectrum FFT (compare 1024, 8192, 65536)
        for size in [1024, 8192, 65536] {
            if let (Some(b), Some(c)) = (self.fft.iter().find(|m| m.fft_size == size), current.fft.iter().find(|m| m.fft_size == size)) {
                add_higher_better(&mut diffs, "Spectrum FFT", format!("{} FFT Throughput", size), b.throughput_msps, c.throughput_msps, " MS/s", tolerance_pct);
            }
        }

        // 3. Channelizer (compare 16, 32 slots)
        for slots in [16, 32] {
            if let (Some(b), Some(c)) = (self.channelizer.iter().find(|m| m.num_slots == slots), current.channelizer.iter().find(|m| m.num_slots == slots)) {
                add_higher_better(&mut diffs, "Channelizer", format!("{} Slots Throughput", slots), b.throughput_msps, c.throughput_msps, " MS/s", tolerance_pct);
                add_lower_better(&mut diffs, "Channelizer", format!("{} Slots CPU %", slots), b.cpu_pct, c.cpu_pct, "%", tolerance_pct);
            }
        }

        // 4. FT8 Decodes & Latency
        if let (Some(b), Some(c)) = (&self.ft8, &current.ft8) {
            let dec_delta = if b.total_decodes > 0 { ((c.total_decodes as f64 - b.total_decodes as f64) / b.total_decodes as f64) * 100.0 } else { 0.0 };
            let status = if c.total_decodes > b.total_decodes {
                MetricStatus::Improved
            } else if c.total_decodes < b.total_decodes {
                MetricStatus::Regressed
            } else {
                MetricStatus::Ok
            };
            diffs.push(MetricDiff {
                category: "FT8 Decoder",
                name: "Total Decoded Messages".into(),
                baseline_val: format!("{}", b.total_decodes),
                current_val: format!("{}", c.total_decodes),
                delta_pct: dec_delta,
                status,
                note: if status == MetricStatus::Regressed { "decoded fewer stations" } else if status == MetricStatus::Improved { "decoded more stations!" } else { "stable" },
            });

            add_lower_better(&mut diffs, "FT8 Decoder", "Avg Slot Latency".into(), b.avg_slot_latency_ms, c.avg_slot_latency_ms, " ms", tolerance_pct);
        }

        // 5. CW Decoder
        if let (Some(b), Some(c)) = (&self.cw, &current.cw) {
            add_higher_better(&mut diffs, "CW Decoder", "Demod Throughput".into(), b.avg_audio_msps, c.avg_audio_msps, " MS/s", tolerance_pct);

            // Compare 14061.78 kHz QRP station confidence
            let b_qrp = b.signals.iter().find(|s| (s.dial_khz - 14061.78).abs() < 0.5);
            let c_qrp = c.signals.iter().find(|s| (s.dial_khz - 14061.78).abs() < 0.5);
            if let (Some(bq), Some(cq)) = (b_qrp, c_qrp) {
                let conf_delta = (cq.max_confidence_pct - bq.max_confidence_pct) as f64;
                let status = if cq.max_confidence_pct > bq.max_confidence_pct + 1.0 {
                    MetricStatus::Improved
                } else if cq.max_confidence_pct < bq.max_confidence_pct - 3.0 {
                    MetricStatus::Regressed
                } else {
                    MetricStatus::Ok
                };
                diffs.push(MetricDiff {
                    category: "CW Decoder",
                    name: "14061.8 kHz QRP Confidence".into(),
                    baseline_val: format!("{:.0}%", bq.max_confidence_pct),
                    current_val: format!("{:.0}%", cq.max_confidence_pct),
                    delta_pct: conf_delta,
                    status,
                    note: if status == MetricStatus::Regressed { "lower copy confidence" } else { "" },
                });
            }
        }

        // 6. Demodulators
        for mode in ["PSK31", "RTTY"] {
            if let (Some(b), Some(c)) = (self.demodulators.iter().find(|m| m.mode == mode), current.demodulators.iter().find(|m| m.mode == mode)) {
                add_higher_better(&mut diffs, "Demodulators", format!("{} Audio MS/s", mode), b.audio_msps, c.audio_msps, " MS/s", tolerance_pct);
            }
        }

        // 7. End-to-End
        add_higher_better(&mut diffs, "End-to-End", "Overall Speedup".into(), self.end_to_end.speedup, current.end_to_end.speedup, "x", tolerance_pct);
        add_lower_better(&mut diffs, "End-to-End", "CPU Occupancy %".into(), self.end_to_end.cpu_occupancy_pct, current.end_to_end.cpu_occupancy_pct, "%", tolerance_pct);

        diffs
    }
}

/// One gain setting and the noise floor measured at it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GainPoint {
    pub gain_db: f64,
    /// Mean power in dBFS.
    pub floor_dbfs: f32,
}

/// Whether a gain control counts upward in gain or downward in *reduction*.
///
/// SDRplay expresses both its elements as gain reduction — a bigger number is
/// less gain — while an RTL-SDR's tuner gain counts the usual way. Rather than
/// keep a table of which is which, read it off the sweep: the noise floor can
/// only fall as a control is turned toward less gain.
pub fn control_is_reduction(points: &[GainPoint]) -> bool {
    let (mut rising, mut falling) = (0i32, 0i32);
    for w in points.windows(2) {
        if w[1].gain_db <= w[0].gain_db {
            continue;
        }
        if w[1].floor_dbfs > w[0].floor_dbfs {
            rising += 1;
        } else {
            falling += 1;
        }
    }
    falling > rising
}

/// Where added gain stops improving sensitivity.
///
/// Turn the gain up from the bottom and at first the floor barely moves: the
/// receiver's own noise dominates, and signal and noise rise together. Past
/// some point the floor starts following gain decibel for decibel, which means
/// the band's noise now dominates — from there on, more gain buys nothing but
/// intermodulation and lost headroom. The knee is the first setting where the
/// band has taken over, and it is the one worth running.
///
/// Returned in the device's own units, whichever direction they count in, so
/// the answer can be applied directly.
///
/// `None` when the floor never tracks: the receiver is still the dominant
/// noise source everywhere, which usually means no antenna.
pub fn find_gain_knee(points: &[GainPoint]) -> Option<GainPoint> {
    if points.len() < 3 {
        return None;
    }
    // Walk in order of increasing actual gain, whichever way the control runs.
    let mut ordered: Vec<GainPoint> = points.to_vec();
    ordered.sort_by(|a, b| a.gain_db.total_cmp(&b.gain_db));
    if control_is_reduction(points) {
        ordered.reverse();
    }
    // Tracking means the floor follows gain with at least this slope. Half a
    // dB per dB is well clear of the flat region and well below the 1.0 an
    // ideal external-noise-limited receiver would give.
    const TRACKING: f64 = 0.5;
    for w in ordered.windows(2) {
        let (a, b) = (w[0], w[1]);
        let step = (b.gain_db - a.gain_db).abs();
        if step < 1e-9 {
            continue;
        }
        if (b.floor_dbfs - a.floor_dbfs) as f64 / step >= TRACKING {
            return Some(b);
        }
    }
    None
}

/// Mean power of a block, in dBFS.
pub fn block_floor_dbfs(iq: &[Complex32]) -> f32 {
    if iq.is_empty() {
        return -140.0;
    }
    let mean = iq.iter().map(|s| s.norm_sqr()).sum::<f32>() / iq.len() as f32;
    10.0 * mean.max(1e-20).log10()
}

/// What a band preset assumed, against what the driver actually did.
#[derive(Clone, Debug, PartialEq)]
pub struct BandCheck {
    pub name: &'static str,
    pub asked_rate: f64,
    pub got_rate: f64,
    pub got_bandwidth: f64,
    /// Calling frequencies that fall outside the view the achieved rate gives.
    pub markers_lost: Vec<&'static str>,
    /// Whether the achieved rate still divides by the FT audio clock.
    pub ft_usable: bool,
}

impl BandCheck {
    pub fn ok(&self) -> bool {
        self.markers_lost.is_empty()
            && self.ft_usable
            && (self.got_rate - self.asked_rate).abs() < 1.0
    }
}

/// Check one band's plan against an achieved rate. Pure, so the rules are
/// testable without a receiver attached.
pub fn check_band(band: &'static bands::Band, got_rate: f64, got_bandwidth: f64) -> BandCheck {
    let half = got_rate / 2.0;
    let mut markers_lost = Vec::new();
    for m in bands::MARKERS {
        if m.label == "WWV" || m.freq < band.start || m.freq > band.end {
            continue;
        }
        let off = m.freq - band.default;
        // 3 kHz of USB passband sits above the dial, and the pickers discard
        // the outermost bins.
        if off <= -(half - 5_000.0) || off + 3_000.0 >= half - 5_000.0 {
            markers_lost.push(m.label);
        }
    }
    BandCheck {
        name: band.name,
        asked_rate: band.span,
        got_rate,
        got_bandwidth,
        markers_lost,
        ft_usable: {
            let d = got_rate / crate::decoders::ft8::AUDIO_RATE;
            (d - d.round()).abs() < 1e-9 && d >= 1.0
        },
    }
}

fn ranges(label: &str, rs: &[soapysdr::Range]) -> String {
    if rs.is_empty() {
        return format!("{label}: (none reported)");
    }
    // Degenerate ranges are discrete choices; a real span is continuous.
    let discrete = rs
        .iter()
        .all(|r| r.maximum <= r.minimum + r.minimum.abs().max(1.0) * 1e-6);
    if discrete {
        let list: Vec<String> = rs.iter().map(|r| format!("{:.0}", r.maximum)).collect();
        format!("{label}: {} discrete — {}", rs.len(), list.join(", "))
    } else {
        let list: Vec<String> = rs
            .iter()
            .map(|r| format!("{:.0}..{:.0}", r.minimum, r.maximum))
            .collect();
        format!("{label}: continuous — {}", list.join(", "))
    }
}

pub fn run(device_args: &str, sweep_freq: f64) -> Result<()> {
    let dev = soapysdr::Device::new(device_args).context("opening device")?;

    println!("== receiver ==");
    println!(
        "  driver: {}",
        dev.driver_key().unwrap_or_else(|_| "?".into())
    );
    println!(
        "  hardware: {}",
        dev.hardware_key().unwrap_or_else(|_| "?".into())
    );
    println!(
        "  {}",
        ranges(
            "sample rates",
            &dev.get_sample_rate_range(Rx, 0).unwrap_or_default()
        )
    );
    println!(
        "  {}",
        ranges(
            "bandwidths",
            &dev.bandwidth_range(Rx, 0).unwrap_or_default()
        )
    );
    let elements = dev.list_gains(Rx, 0).unwrap_or_default();
    for e in &elements {
        if let Ok(r) = dev.gain_element_range(Rx, 0, e.as_str()) {
            println!("  gain {e}: {:.0}..{:.0} dB", r.minimum, r.maximum);
        }
    }
    if elements.is_empty()
        && let Ok(r) = dev.gain_range(Rx, 0)
    {
        println!(
            "  gain: {:.0}..{:.0} dB (single element)",
            r.minimum, r.maximum
        );
    }

    println!("\n== band plan vs driver ==");
    println!(
        "  {:>5} {:>9} {:>9} {:>9}  result",
        "band", "asked", "got", "filter"
    );
    let mut failures = 0;
    for b in bands::BANDS {
        if b.name == "WWV" {
            continue;
        }
        dev.set_frequency(Rx, 0, b.default, ())?;
        let _ = dev.set_sample_rate(Rx, 0, b.span);
        let got = dev.sample_rate(Rx, 0).unwrap_or(0.0);
        let _ = dev.set_bandwidth(Rx, 0, got);
        let bw = dev.bandwidth(Rx, 0).unwrap_or(0.0);
        let c = check_band(b, got, bw);
        if c.ok() {
            println!(
                "  {:>5} {:>9.0} {:>9.0} {:>9.0}  ok",
                c.name, c.asked_rate, c.got_rate, c.got_bandwidth
            );
            continue;
        }
        failures += 1;
        let mut notes = Vec::new();
        if (c.got_rate - c.asked_rate).abs() >= 1.0 {
            notes.push("RATE CLAMPED".to_string());
        }
        if !c.ft_usable {
            notes.push("FT8/FT4 UNUSABLE".to_string());
        }
        if !c.markers_lost.is_empty() {
            notes.push(format!("MARKERS OFF-SCREEN: {}", c.markers_lost.join(" ")));
        }
        println!(
            "  {:>5} {:>9.0} {:>9.0} {:>9.0}  {}",
            c.name,
            c.asked_rate,
            c.got_rate,
            c.got_bandwidth,
            notes.join(", ")
        );
    }

    let _ = sweep_freq;
    println!("\n== gain knee per band ==");
    println!("  antenna connected; each band is swept at its own centre and span");
    println!(
        "  {:>5} {:>9} {:>9} {:>7} {:>7}  {}",
        "band", "knee", "floor", "RFGR", "IFGR", "note"
    );
    for b in bands::BANDS {
        if b.name == "WWV" {
            continue;
        }
        let points = match sweep_gain(&dev, b.default, b.span) {
            Ok(p) => p,
            Err(e) => {
                println!("  {:>5}  sweep failed: {e}", b.name);
                continue;
            }
        };
        if std::env::var("HFSCAN_BENCH_CURVES").is_ok() {
            let c: Vec<String> = points
                .iter()
                .map(|p| format!("{:.0}:{:.1}", p.gain_db, p.floor_dbfs))
                .collect();
            println!("  {:>5} curve {}", b.name, c.join("  "));
        }
        match find_gain_knee(&points) {
            Some(k) => {
                dev.set_gain(Rx, 0, k.gain_db)?;
                let rfgr = dev.gain_element(Rx, 0, "RFGR").unwrap_or(f64::NAN);
                let ifgr = dev.gain_element(Rx, 0, "IFGR").unwrap_or(f64::NAN);
                let span = points.iter().map(|p| p.floor_dbfs).fold(f32::MIN, f32::max)
                    - points.iter().map(|p| p.floor_dbfs).fold(f32::MAX, f32::min);
                println!(
                    "  {:>5} {:>8.1}dB {:>8.1}dB {:>7.0} {:>7.0}  {:.0} dB of travel",
                    b.name, k.gain_db, k.floor_dbfs, rfgr, ifgr, span
                );
            }
            None => println!(
                "  {:>5}       —         —       —       —  no knee: band quieter than the receiver",
                b.name
            ),
        }
    }

    if failures > 0 {
        println!("\n{failures} band(s) did not come up as planned.");
    }
    Ok(())
}

fn sweep_gain(dev: &soapysdr::Device, freq: f64, span: f64) -> Result<Vec<GainPoint>> {
    dev.set_frequency(Rx, 0, freq, ())?;
    dev.set_sample_rate(Rx, 0, span)?;
    let _ = dev.set_bandwidth(Rx, 0, span);
    let _ = dev.set_gain_mode(Rx, 0, false);

    let range = dev.gain_range(Rx, 0).context("no overall gain range")?;
    let mut stream = dev.rx_stream::<Complex32>(&[0])?;
    stream.activate(None)?;
    let mut buf = vec![Complex32::new(0.0, 0.0); 65_536];

    let mut out = Vec::new();
    let steps = 12;
    for i in 0..=steps {
        let g = range.minimum + (range.maximum - range.minimum) * i as f64 / steps as f64;
        dev.set_gain(Rx, 0, g)?;
        // Let the front end settle, and throw away what was in flight.
        std::thread::sleep(std::time::Duration::from_millis(120));
        for _ in 0..4 {
            let _ = stream.read(&mut [&mut buf], 200_000);
        }
        let n = stream.read(&mut [&mut buf], 200_000).unwrap_or(0);
        out.push(GainPoint {
            gain_db: g,
            floor_dbfs: block_floor_dbfs(&buf[..n]),
        });
    }
    let _ = stream.deactivate(None);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(gain_db: f64, floor_dbfs: f32) -> GainPoint {
        GainPoint {
            gain_db,
            floor_dbfs,
        }
    }

    /// The knee is where the floor starts following gain. Below it the
    /// receiver's own noise dominates and the floor barely moves.
    #[test]
    fn the_knee_is_where_the_floor_starts_tracking_gain() {
        let sweep = [
            pt(0.0, -70.0),
            pt(10.0, -69.6),
            pt(20.0, -69.1),
            pt(30.0, -58.0), // external noise takes over here
            pt(40.0, -48.0),
            pt(50.0, -38.0),
        ];
        assert!(!control_is_reduction(&sweep));
        assert_eq!(find_gain_knee(&sweep), Some(pt(30.0, -58.0)));
    }

    /// SDRplay counts both its gain elements as *reduction*, so the floor
    /// falls as the number rises; an RTL-SDR counts the usual way. The sweep
    /// itself says which, so no table of devices is needed.
    ///
    /// These are the real numbers from an RSP1A on 20m into an antenna.
    #[test]
    fn a_gain_reduction_control_is_read_the_right_way_round() {
        let sweep = [
            pt(0.0, -43.02),
            pt(4.0, -46.43),
            pt(8.0, -50.44),
            pt(12.0, -54.35),
            pt(16.0, -58.67),
            pt(20.0, -62.51),
            pt(24.0, -66.22),
            pt(28.0, -69.57),
            pt(32.0, -72.31),
            pt(36.0, -73.58),
            pt(40.0, -75.39),
            pt(44.0, -75.29),
            pt(48.0, -75.62),
        ];
        assert!(
            control_is_reduction(&sweep),
            "a floor that falls as the number rises is a reduction control"
        );
        // Read toward more gain — down the scale — the floor is flat from 48
        // to 40 (slopes 0.08, -0.03), then climbs through 0.45 and 0.32 before
        // clearly tracking at 0.69. The transition is gradual, so the answer
        // is the first step that is unambiguously tracking rather than the
        // first that twitches: erring toward less gain costs a little
        // sensitivity, erring the other way costs intermodulation.
        let knee = find_gain_knee(&sweep).expect("this sweep has a knee");
        assert_eq!(knee.gain_db, 28.0, "knee in the device's own units");
    }

    /// A disconnected antenna never tracks: the receiver is the only noise
    /// source at every setting, and reporting a knee would be a lie.
    #[test]
    fn a_flat_sweep_has_no_knee() {
        let sweep = [
            pt(0.0, -70.0),
            pt(10.0, -69.8),
            pt(20.0, -69.6),
            pt(30.0, -69.5),
            pt(40.0, -69.3),
        ];
        assert_eq!(find_gain_knee(&sweep), None);
        assert_eq!(find_gain_knee(&[pt(0.0, -70.0)]), None);
    }

    /// A sweep that tracks from the very first step means there was never a
    /// receiver-limited region: the band dominates even at minimum gain, and
    /// the knee is the bottom of the range.
    #[test]
    fn a_sweep_that_tracks_immediately_knees_at_the_bottom() {
        let sweep = [pt(0.0, -70.0), pt(10.0, -60.0), pt(20.0, -50.0)];
        assert_eq!(find_gain_knee(&sweep), Some(pt(10.0, -60.0)));
    }

    /// The band check has to fail exactly the case that shipped broken: a
    /// clamped rate under a centre chosen for the width that was asked for.
    #[test]
    fn a_clamped_rate_that_hides_the_digital_segment_is_caught() {
        // 10m spreads its calling frequencies over 110 kHz and plans for a
        // 384 kHz span to hold them.
        let ten = bands::BANDS.iter().find(|b| b.name == "10m").unwrap();
        let good = check_band(ten, ten.span, ten.span);
        assert!(good.ok(), "the shipped plan should pass: {good:?}");

        // Delivered a quarter of what it asked for: FT4 at 28.180 falls off
        // the top. This is the shape of the failure that shipped — a clamped
        // rate under a centre chosen for the width that was requested.
        let clamped = check_band(ten, 96_000.0, 96_000.0);
        assert!(!clamped.ok(), "a clamped rate must not pass");
        assert!(
            clamped.markers_lost.contains(&"FT4"),
            "should report FT4 off-screen, got {:?}",
            clamped.markers_lost
        );
    }

    /// A rate that no longer divides by the FT audio clock is a failure even
    /// when everything is still visible, because the decoders stop.
    #[test]
    fn a_rate_that_breaks_the_ft_clock_is_caught() {
        let thirty = bands::BANDS.iter().find(|b| b.name == "30m").unwrap();
        let c = check_band(thirty, 250_000.0, 250_000.0);
        assert!(!c.ft_usable, "250 kS/s does not divide by 12 kHz");
        assert!(!c.ok());
    }

    #[test]
    fn floor_of_silence_is_the_bottom_of_the_scale() {
        assert!(block_floor_dbfs(&[]) < -100.0);
        let quiet = vec![Complex32::new(0.001, 0.0); 128];
        let loud = vec![Complex32::new(0.5, 0.0); 128];
        assert!(block_floor_dbfs(&quiet) < block_floor_dbfs(&loud) - 40.0);
    }
}
