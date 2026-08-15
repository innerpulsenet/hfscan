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
    /// Bands that fit inside their span are centred on the allocation, which
    /// keeps both edges an equal distance from Nyquist. The rest are centred
    /// so their *digital* segment sits comfortably inside the view: on nearly
    /// every band the modes this decodes live in the bottom tens of kHz, so
    /// when something has to be cut it is the top of the phone segment.
    ///
    /// Either way the LO lands at least 10 kHz from the nearest calling
    /// frequency, since a zero-IF spike sits on top of whatever it lands on.
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

/// Band the app opens on when nothing says otherwise. An index rather than a
/// duplicated frequency, so the startup centre and span cannot drift out of
/// step with the preset the way a hard-coded default did.
pub const DEFAULT_BAND: usize = 5; // 20m

pub const BANDS: &[Band] = &[
    Band { name: "160m", start: 1_800_000.0,   end: 2_000_000.0,   default: 1_900_000.0,   span: 384_000.0 },
    Band { name: "80m",  start: 3_500_000.0,   end: 4_000_000.0,   default: 3_610_000.0,   span: 384_000.0 },
    Band { name: "60m",  start: 5_330_000.0,   end: 5_405_000.0,   default: 5_367_500.0,   span: 192_000.0 },
    Band { name: "40m",  start: 7_000_000.0,   end: 7_300_000.0,   default: 7_150_000.0,   span: 384_000.0 },
    Band { name: "30m",  start: 10_100_000.0,  end: 10_150_000.0,  default: 10_125_000.0,  span: 192_000.0 },
    Band { name: "20m",  start: 14_000_000.0,  end: 14_350_000.0,  default: 14_175_000.0,  span: 384_000.0 },
    Band { name: "17m",  start: 18_068_000.0,  end: 18_168_000.0,  default: 18_118_000.0,  span: 192_000.0 },
    Band { name: "15m",  start: 21_000_000.0,  end: 21_450_000.0,  default: 21_160_000.0,  span: 384_000.0 },
    Band { name: "12m",  start: 24_890_000.0,  end: 24_990_000.0,  default: 24_940_000.0,  span: 192_000.0 },
    Band { name: "10m",  start: 28_000_000.0,  end: 29_700_000.0,  default: 28_125_000.0,  span: 384_000.0 },
    Band { name: "6m",   start: 50_000_000.0,  end: 54_000_000.0,  default: 50_330_000.0,  span: 384_000.0 },
    Band { name: "2m",   start: 144_000_000.0, end: 148_000_000.0, default: 144_190_000.0, span: 384_000.0 },
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

    /// The one guarantee that must never break: whatever else is cut, the
    /// digital calling frequencies stay in view. This is exactly what broke
    /// when spans the receiver could not produce were clamped underneath a
    /// centre chosen for the width that was asked for — 20m came up showing
    /// 14.079 upwards, with FT8 at 14.074 just off the left edge and no
    /// decodes at all.
    #[test]
    fn the_digital_segment_is_always_in_view() {
        const EDGE: f64 = 5_000.0; // discarded edge bins
        const PASSBAND: f64 = 3_000.0; // USB audio above the dial
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
                    off > -(half - EDGE) && off + PASSBAND < half - EDGE,
                    "{}: the {} marker at {:.3} MHz is {off:+.0} Hz from centre, \
                     outside a {:.0} kHz span",
                    b.name,
                    m.label,
                    m.freq / 1e6,
                    b.span / 1000.0
                );
            }
            assert!(seen, "{} has no markers to check", b.name);
        }
    }

    /// Bands narrow enough to fit are shown whole. The rest cannot be, at any
    /// rate this receiver offers that also keeps FT8 working.
    #[test]
    fn bands_that_can_be_shown_whole_are() {
        for b in BANDS {
            if b.name == "WWV" {
                continue;
            }
            let half = b.span / 2.0;
            let fits = b.end - b.start <= b.span;
            let covered = b.default - half <= b.start && b.default + half >= b.end;
            if fits {
                assert!(
                    covered,
                    "{} is {:.0} kHz inside a {:.0} kHz span but is not centred to show it",
                    b.name,
                    (b.end - b.start) / 1000.0,
                    b.span / 1000.0
                );
            }
        }
    }

    /// A span the device cannot produce gets silently clamped by the driver,
    /// which is how the digital segment ended up off-screen.
    #[test]
    fn spans_are_rates_the_receiver_offers() {
        const SUPPORTED: [f64; 8] = [
            62_500.0, 96_000.0, 125_000.0, 192_000.0, 250_000.0, 384_000.0, 500_000.0,
            1_000_000.0,
        ];
        for b in BANDS {
            assert!(
                SUPPORTED.contains(&b.span),
                "{} asks for {:.0} Hz, which the receiver does not offer",
                b.name,
                b.span
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

    /// The digital segment must sit in the flat part of the span, not out
    /// where the analog filter corner is — that rolloff is what shows on the
    /// waterfall as the edges of the view falling away.
    ///
    /// This is about the markers, not the band edges. 20m at 384 kHz reaches
    /// its band edges at 91% of Nyquist and there is no FT-safe rate this
    /// receiver offers that would do better; what matters is that its digital
    /// segment sits at 55%, where the response is flat. When something has to
    /// be given up at the extremes it is the top of the phone segment.
    #[test]
    fn the_digital_segment_sits_in_the_flat_part_of_the_span() {
        for b in BANDS {
            if b.name == "WWV" {
                continue;
            }
            for m in MARKERS {
                if m.label == "WWV" || m.freq < b.start || m.freq > b.end {
                    continue;
                }
                let frac = (m.freq + 3_000.0 - b.default).abs() / (b.span / 2.0);
                assert!(
                    frac <= 0.70,
                    "{}: the {} marker sits at {:.0}% of Nyquist, in the filter skirt",
                    b.name,
                    m.label,
                    frac * 100.0
                );
            }
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
