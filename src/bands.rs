//! HF band plan: presets to jump between, plus the digital-mode calling
//! frequencies worth marking on the spectrum.

pub struct Band {
    pub name: &'static str,
    pub start: f64,
    pub end: f64,
    /// Where to park the receiver when jumping to this band.
    pub default: f64,
}

pub const BANDS: &[Band] = &[
    Band { name: "160m", start: 1_800_000.0,  end: 2_000_000.0,  default: 1_838_000.0 },
    Band { name: "80m",  start: 3_500_000.0,  end: 4_000_000.0,  default: 3_580_000.0 },
    Band { name: "60m",  start: 5_330_000.0,  end: 5_405_000.0,  default: 5_357_000.0 },
    Band { name: "40m",  start: 7_000_000.0,  end: 7_300_000.0,  default: 7_040_000.0 },
    Band { name: "30m",  start: 10_100_000.0, end: 10_150_000.0, default: 10_140_000.0 },
    Band { name: "20m",  start: 14_000_000.0, end: 14_350_000.0, default: 14_070_000.0 },
    Band { name: "17m",  start: 18_068_000.0, end: 18_168_000.0, default: 18_100_000.0 },
    Band { name: "15m",  start: 21_000_000.0, end: 21_450_000.0, default: 21_080_000.0 },
    Band { name: "12m",  start: 24_890_000.0, end: 24_990_000.0, default: 24_920_000.0 },
    Band { name: "10m",  start: 28_000_000.0, end: 29_700_000.0, default: 28_120_000.0 },
    Band { name: "WWV",  start: 4_990_000.0,  end: 15_010_000.0, default: 10_000_000.0 },
];

pub struct Marker {
    pub freq: f64,
    pub label: &'static str,
}

pub const MARKERS: &[Marker] = &[
    Marker { freq: 1_838_000.0,  label: "PSK" },
    Marker { freq: 1_840_000.0,  label: "FT8" },
    Marker { freq: 3_573_000.0,  label: "FT8" },
    Marker { freq: 3_580_000.0,  label: "PSK" },
    Marker { freq: 3_590_000.0,  label: "RTTY" },
    Marker { freq: 5_357_000.0,  label: "FT8" },
    Marker { freq: 7_040_000.0,  label: "PSK" },
    Marker { freq: 7_047_000.0,  label: "RTTY" },
    Marker { freq: 7_047_500.0,  label: "FT4" },
    Marker { freq: 7_074_000.0,  label: "FT8" },
    Marker { freq: 10_136_000.0, label: "FT8" },
    Marker { freq: 10_140_000.0, label: "FT4" },
    Marker { freq: 10_140_000.0, label: "PSK" },
    Marker { freq: 10_142_000.0, label: "RTTY" },
    Marker { freq: 14_070_000.0, label: "PSK" },
    Marker { freq: 14_074_000.0, label: "FT8" },
    Marker { freq: 14_080_000.0, label: "RTTY" },
    Marker { freq: 14_080_000.0, label: "FT4" },
    Marker { freq: 18_100_000.0, label: "PSK" },
    Marker { freq: 18_100_000.0, label: "RTTY" },
    Marker { freq: 18_104_000.0, label: "FT8" },
    Marker { freq: 18_108_000.0, label: "FT4" },
    Marker { freq: 21_070_000.0, label: "PSK" },
    Marker { freq: 21_074_000.0, label: "FT8" },
    Marker { freq: 21_080_000.0, label: "RTTY" },
    Marker { freq: 21_140_000.0, label: "FT4" },
    Marker { freq: 24_915_000.0, label: "FT8" },
    Marker { freq: 24_919_000.0, label: "FT4" },
    Marker { freq: 28_070_000.0, label: "PSK" },
    Marker { freq: 28_074_000.0, label: "FT8" },
    Marker { freq: 28_080_000.0, label: "RTTY" },
    Marker { freq: 28_180_000.0, label: "FT4" },
    Marker { freq: 5_000_000.0,  label: "WWV" },
    Marker { freq: 10_000_000.0, label: "WWV" },
    Marker { freq: 15_000_000.0, label: "WWV" },
];

/// The narrowband sub-band `freq` falls in, on the same USB convention the
/// FT windows use: the marker is a dial frequency and the signals sit in the
/// audio passband above it.
///
/// PSK31 and RTTY are packed into a couple of kHz above their marker, so this
/// is deliberately narrower than the FT window. It exists because the two
/// overlap: FT4 shares a dial frequency with 30 m PSK31 and with 20 m RTTY,
/// and sits 500 Hz above 40 m RTTY. Without it those sub-bands are inside an
/// FT window and can never be classified as anything else at all.
pub fn narrow_mode(freq: f64) -> Option<&'static str> {
    MARKERS
        .iter()
        .filter(|m| matches!(m.label, "PSK" | "RTTY"))
        .find(|m| (100.0..2600.0).contains(&(freq - m.freq)))
        .map(|m| m.label)
}

pub fn band_for(freq: f64) -> Option<&'static Band> {
    BANDS
        .iter()
        .find(|b| freq >= b.start && freq <= b.end && b.name != "WWV")
}

/// True if `freq` sits inside a real amateur allocation (not the WWV preset).
pub fn in_amateur(freq: f64) -> bool {
    band_for(freq).is_some()
}

/// FT8 / FT4 live in the 200–3000 Hz USB passband above the dial, not
/// as one-off carriers scattered across the band.
pub fn ft_mode(freq: f64) -> Option<&'static str> {
    let mut best: Option<(&'static str, f64)> = None;
    for m in MARKERS {
        let label = match m.label {
            "FT8" | "FT4" => m.label,
            _ => continue,
        };
        let off = freq - m.freq;
        if (150.0..3200.0).contains(&off) {
            let d = (off - 1500.0).abs();
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some((label, d));
            }
        }
    }
    best.map(|(l, _)| l)
}
