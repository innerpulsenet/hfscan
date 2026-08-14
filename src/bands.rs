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
    Marker { freq: 7_074_000.0,  label: "FT8" },
    Marker { freq: 10_136_000.0, label: "FT8" },
    Marker { freq: 10_140_000.0, label: "PSK" },
    Marker { freq: 10_142_000.0, label: "RTTY" },
    Marker { freq: 14_070_000.0, label: "PSK" },
    Marker { freq: 14_074_000.0, label: "FT8" },
    Marker { freq: 14_080_000.0, label: "RTTY" },
    Marker { freq: 18_100_000.0, label: "PSK" },
    Marker { freq: 18_100_000.0, label: "RTTY" },
    Marker { freq: 18_104_000.0, label: "FT8" },
    Marker { freq: 21_070_000.0, label: "PSK" },
    Marker { freq: 21_074_000.0, label: "FT8" },
    Marker { freq: 21_080_000.0, label: "RTTY" },
    Marker { freq: 24_915_000.0, label: "FT8" },
    Marker { freq: 28_070_000.0, label: "PSK" },
    Marker { freq: 28_074_000.0, label: "FT8" },
    Marker { freq: 28_080_000.0, label: "RTTY" },
    Marker { freq: 5_000_000.0,  label: "WWV" },
    Marker { freq: 10_000_000.0, label: "WWV" },
    Marker { freq: 15_000_000.0, label: "WWV" },
];

pub fn band_for(freq: f64) -> Option<&'static Band> {
    BANDS
        .iter()
        .find(|b| freq >= b.start && freq <= b.end && b.name != "WWV")
}
