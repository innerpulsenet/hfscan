//! HF band plan: presets to jump between, plus the digital-mode calling
//! frequencies worth marking on the spectrum.

pub struct Band {
    pub name: &'static str,
    pub start: f64,
    pub end: f64,
    /// Where to park the receiver when jumping to this band.
    ///
    /// Whatever else it is, it must not be a digital calling frequency. A
    /// zero-IF front end leaves a spike at the local oscillator, so every
    /// candidate picker here blanks the bins either side of it — a real signal
    /// within roughly 50 Hz of the LO is invisible by construction, and the
    /// LO's phase noise and 1/f skirt sit over the first few hundred Hz beyond
    /// that. Parking on the dial frequency put exactly that region on the
    /// bottom of the sub-band people work.
    ///
    /// Bands whose `span` covers the whole allocation are centred on it, which
    /// puts the LO in the middle of the phone segment and well clear of every
    /// marker. The four narrow ones already fit inside the 192 kHz default, so
    /// they sit ~10 kHz below their digital segment instead: the whole segment
    /// then lands at a clean positive offset.
    pub default: f64,
    /// Sample rate to run at on this band, which is also the width of the
    /// spectrum view — chosen so the whole allocation fits in one span.
    ///
    /// Every value is a multiple of 24 kHz, the lowest common multiple of the
    /// 8 kHz and 12 kHz audio clocks, so FT8 and FT4 keep an exact divisor and
    /// stay usable at full-band width instead of forcing a retune down to
    /// `FT_SAFE_RATE`. Each carries at least 8% margin over the allocation so
    /// the band edges are not sitting in the anti-alias skirt or in the bins
    /// the candidate pickers discard.
    ///
    /// Capped at 6 MS/s: the RSP1A's converter runs at 14 bits up to about
    /// there and drops to 12, 10 and 8 as the rate climbs to 10.66, so a wider
    /// view past this point is bought with dynamic range — the wrong trade for
    /// a receiver whose job is weak signals. Nothing needs it: the widest
    /// allocation here is 4 MHz.
    pub span: f64,
}

pub const BANDS: &[Band] = &[
    Band { name: "160m", start: 1_800_000.0,   end: 2_000_000.0,   default: 1_900_000.0,   span: 240_000.0 },
    Band { name: "80m",  start: 3_500_000.0,   end: 4_000_000.0,   default: 3_750_000.0,   span: 600_000.0 },
    Band { name: "60m",  start: 5_330_000.0,   end: 5_405_000.0,   default: 5_347_000.0,   span: 192_000.0 },
    Band { name: "40m",  start: 7_000_000.0,   end: 7_300_000.0,   default: 7_150_000.0,   span: 360_000.0 },
    Band { name: "30m",  start: 10_100_000.0,  end: 10_150_000.0,  default: 10_126_000.0,  span: 192_000.0 },
    Band { name: "20m",  start: 14_000_000.0,  end: 14_350_000.0,  default: 14_175_000.0,  span: 432_000.0 },
    Band { name: "17m",  start: 18_068_000.0,  end: 18_168_000.0,  default: 18_090_000.0,  span: 192_000.0 },
    Band { name: "15m",  start: 21_000_000.0,  end: 21_450_000.0,  default: 21_225_000.0,  span: 528_000.0 },
    Band { name: "12m",  start: 24_890_000.0,  end: 24_990_000.0,  default: 24_905_000.0,  span: 192_000.0 },
    Band { name: "10m",  start: 28_000_000.0,  end: 29_700_000.0,  default: 28_850_000.0,  span: 1_920_000.0 },
    Band { name: "6m",   start: 50_000_000.0,  end: 54_000_000.0,  default: 52_000_000.0,  span: 4_320_000.0 },
    Band { name: "2m",   start: 144_000_000.0, end: 148_000_000.0, default: 146_000_000.0, span: 4_320_000.0 },
    Band { name: "WWV",  start: 4_990_000.0,   end: 15_010_000.0,  default: 10_000_000.0,  span: 192_000.0 },
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

    /// ...while still keeping every marker where it can be seen: inside the
    /// band's own span, and clear of the edge bins the pickers discard.
    ///
    /// The 10 m exception is gone. Its digital segments are spread over
    /// 110 kHz, which no 192 kHz span could hold along with the LO offset;
    /// at 1.92 MS/s the whole band is in view and so is every marker on it.
    #[test]
    fn defaults_keep_the_calling_frequencies_in_view() {
        const EDGE: f64 = 5_000.0; // discarded edge plus the USB passband
        for b in BANDS {
            if b.name == "WWV" {
                continue;
            }
            let half = b.span / 2.0;
            let mut seen = false;
            for m in MARKERS {
                if m.label == "WWV" || m.freq < b.start || m.freq > b.end {
                    continue;
                }
                seen = true;
                let off = m.freq - b.default;
                assert!(
                    off > -(half - EDGE) && off + 3_000.0 < half - EDGE,
                    "{} puts the {} marker at {off:+.0} Hz, outside its {:.0} kHz span",
                    b.name,
                    m.label,
                    b.span / 1000.0
                );
            }
            assert!(seen, "{} has no markers to check", b.name);
        }
    }

    /// Every band is viewable whole. This is the point of the per-band span:
    /// jumping to a band should show the band, not a slice of it.
    #[test]
    fn every_band_fits_inside_its_own_span() {
        for b in BANDS {
            if b.name == "WWV" {
                continue;
            }
            let half = b.span / 2.0;
            let (lo, hi) = (b.default - half, b.default + half);
            assert!(
                lo <= b.start && hi >= b.end,
                "{} spans {:.3}-{:.3} MHz but the band is {:.3}-{:.3}",
                b.name,
                lo / 1e6,
                hi / 1e6,
                b.start / 1e6,
                b.end / 1e6
            );
        }
    }

    /// The converter holds 14 bits to about 6 MS/s and sheds them above that,
    /// so a wider view past this point is bought with dynamic range.
    #[test]
    fn no_span_exceeds_the_converters_full_resolution_rate() {
        for b in BANDS {
            assert!(b.span <= 6_000_000.0, "{} asks for {:.1} MS/s", b.name, b.span / 1e6);
        }
    }

    /// Spans are multiples of 24 kHz — the lowest common multiple of the 8 kHz
    /// and 12 kHz audio clocks — so FT8 and FT4 keep an exact divisor and stay
    /// usable at full-band width instead of forcing a retune.
    #[test]
    fn spans_keep_both_audio_clocks_exact() {
        for b in BANDS {
            let d = b.span / 24_000.0;
            assert!(
                (d - d.round()).abs() < 1e-9,
                "{} span {:.0} Hz is not a multiple of 24 kHz",
                b.name,
                b.span
            );
        }
    }

}
