//! HF band plan: presets to jump between, plus the digital-mode calling
//! frequencies worth marking on the spectrum.

pub struct Band {
    pub name: &'static str,
    pub start: f64,
    pub end: f64,
    /// Where to park the receiver when jumping to this band.
    ///
    /// Deliberately *not* on a digital calling frequency, though that is the
    /// obvious choice. A zero-IF front end leaves a spike at the local
    /// oscillator, so every candidate picker here blanks the bins either side
    /// of it — which means a real signal within roughly 50 Hz of the LO is
    /// invisible by construction, and the LO's phase noise and 1/f skirt sit
    /// over the first few hundred Hz beyond that. Parking on the dial
    /// frequency put exactly that region on the bottom of the sub-band people
    /// work. These sit ~10 kHz below the segment instead, so the whole thing
    /// lands at a clean positive offset with every marker still well inside
    /// a 192 kHz span.
    pub default: f64,
}

pub const BANDS: &[Band] = &[
    Band { name: "160m", start: 1_800_000.0,  end: 2_000_000.0,  default: 1_828_000.0 },
    Band { name: "80m",  start: 3_500_000.0,  end: 4_000_000.0,  default: 3_563_000.0 },
    Band { name: "60m",  start: 5_330_000.0,  end: 5_405_000.0,  default: 5_347_000.0 },
    Band { name: "40m",  start: 7_000_000.0,  end: 7_300_000.0,  default: 7_030_000.0 },
    Band { name: "30m",  start: 10_100_000.0, end: 10_150_000.0, default: 10_126_000.0 },
    Band { name: "20m",  start: 14_000_000.0, end: 14_350_000.0, default: 14_060_000.0 },
    Band { name: "17m",  start: 18_068_000.0, end: 18_168_000.0, default: 18_090_000.0 },
    Band { name: "15m",  start: 21_000_000.0, end: 21_450_000.0, default: 21_060_000.0 },
    Band { name: "12m",  start: 24_890_000.0, end: 24_990_000.0, default: 24_905_000.0 },
    Band { name: "10m",  start: 28_000_000.0, end: 29_700_000.0, default: 28_120_000.0 },
    Band { name: "6m",   start: 50_000_000.0, end: 54_000_000.0, default: 50_303_000.0 },
    Band { name: "2m",   start: 144_000_000.0, end: 148_000_000.0, default: 144_164_000.0 },
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
    Marker { freq: 50_313_000.0, label: "FT8" },
    Marker { freq: 50_318_000.0, label: "FT4" },
    Marker { freq: 144_170_000.0, label: "FT4" },
    Marker { freq: 144_174_000.0, label: "FT8" },
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A band default must not park the receiver on a calling frequency.
    ///
    /// The obvious default is the calling frequency itself, and it is the
    /// wrong one: the LO blanking hides anything within tens of Hz of it, and
    /// the phase-noise skirt covers the first part of the sub-band beyond
    /// that. Every default used to do exactly that.
    #[test]
    fn defaults_keep_the_lo_off_the_calling_frequencies() {
        for b in BANDS {
            if b.name == "WWV" {
                continue;
            }
            for m in MARKERS {
                if m.label == "WWV" || m.freq < b.start || m.freq > b.end {
                    continue;
                }
                let off = (m.freq - b.default).abs();
                assert!(
                    off >= 5_000.0,
                    "{} parks the LO {off:.0} Hz from the {} marker at {:.3} MHz",
                    b.name,
                    m.label,
                    m.freq / 1e6
                );
            }
        }
    }

    /// ...while still keeping the markers where they can be seen: inside a
    /// 192 kHz span, and clear of the edge bins the pickers discard.
    #[test]
    fn defaults_keep_the_calling_frequencies_in_view() {
        const HALF: f64 = 96_000.0;
        const EDGE: f64 = 5_000.0; // discarded edge plus the USB passband
        for b in BANDS {
            if b.name == "WWV" {
                continue;
            }
            let mut seen = false;
            for m in MARKERS {
                if m.label == "WWV" || m.freq < b.start || m.freq > b.end {
                    continue;
                }
                seen = true;
                // 10 m spreads its digital segments over 110 kHz, so FT4 at
                // 28.180 cannot share a span with PSK31 at 28.070 whatever
                // the default is; the main cluster is what is kept in view.
                if b.name == "10m" && m.freq > 28_150_000.0 {
                    continue;
                }
                let off = m.freq - b.default;
                assert!(
                    off > -(HALF - EDGE) && off + 3_000.0 < HALF - EDGE,
                    "{} puts the {} marker at {off:+.0} Hz, outside a usable span",
                    b.name,
                    m.label
                );
            }
            assert!(seen || b.name == "60m", "{} has no markers", b.name);
        }
    }
}
