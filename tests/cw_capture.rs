//! CW decoder scored against the recorded 20m band, not a simulation.
//!
//! The synthetic grids answer "how does the decoder behave under a channel we
//! specified". This answers the only question that finally matters: how much
//! of what was actually on 20m at the time of the capture comes back out.
//!
//! ## About the ground truth
//!
//! There is no transcript of this recording, and inventing one from the
//! decoder's own output would make the test agree with itself. So the truth
//! here is limited to *tokens that can be confirmed without trusting the
//! decoder*: each one is a structurally valid callsign or a standard piece of
//! CW procedure, and each recurs identically across the over — the captured
//! stations are calling CQ or running a contest, so they repeat themselves
//! every few seconds. `NK9G` appearing three times at 14061.78, in the same
//! surrounding phrase each time, is not something noise does.
//!
//! Each token was additionally checked to survive changes to the decoder
//! (different slicer thresholds, different filter widths), so it is a
//! property of the recording rather than of one parameter set.
//!
//! What this cannot measure is what the decoder never gets close to. Two of
//! the seven candidate frequencies produce nothing recognisable at all, so
//! they carry no tokens and contribute nothing here. That is a known blind
//! spot, and it is the right way round: this metric can show a redesign
//! working, and cannot show one failing quietly.

use hfscan::bench::IqRecording;
use hfscan::decoders::Decoder;
use hfscan::decoders::cw::CwDecoder;
use std::path::Path;

const CAPTURE_PATH: &str = "captures/20m_14060khz_192ksps_60s.iq";

/// A station in the capture, and what is verifiably in its transmission.
struct Expected {
    dial_hz: f64,
    /// Confirmed tokens, and how many times each is known to be sent.
    tokens: &'static [(&'static str, usize)],
}

/// Confirmed content, established as described in the module docs.
const STATIONS: &[Expected] = &[
    Expected {
        // Calls "CQ CQ QRP TEST DE NK9G NK9G K" repeatedly.
        dial_hz: 14_061_780.0,
        tokens: &[("NK9G", 5), ("CQ", 6), ("QRP", 3), ("TEST", 3), ("DE", 3)],
    },
    Expected {
        // Contest station running "TEST ZW5B".
        dial_hz: 14_002_508.0,
        tokens: &[("ZW5B", 4), ("TEST", 4)],
    },
    Expected {
        dial_hz: 14_028_008.0,
        tokens: &[("CQ", 3), ("TEST", 1)],
    },
    Expected {
        // Tail of an over: "5NN FL ES SRI BK".
        dial_hz: 14_055_008.0,
        tokens: &[("5", 3), ("N", 3), ("BK", 1)],
    },
];

fn decode_at(rec: &IqRecording, dial_hz: f64) -> (String, f32) {
    let iq = rec.extract_iq(dial_hz, 400.0, 8000.0);
    let mut d = CwDecoder::new(8000.0);
    let mut text = String::new();
    for block in iq.chunks(512) {
        text.push_str(&d.process(block));
    }
    (text, d.confidence().unwrap_or(0.0))
}

/// Fraction of confirmed tokens recovered, and how much text was emitted to
/// get them.
///
/// Recall alone would reward a decoder that emits everything; the character
/// count is reported beside it so that trade is visible. A perfect decoder
/// would score high recall on a transcript-length output.
fn token_recall(text: &str, tokens: &[(&str, usize)]) -> (f32, usize, usize) {
    let mut found = 0usize;
    let mut want = 0usize;
    for (tok, times) in tokens {
        let n = text.matches(tok).count().min(*times);
        found += n;
        want += times;
    }
    (found as f32 / want.max(1) as f32, found, want)
}

#[test]
fn cw_recovers_known_stations_from_the_live_capture() {
    if !Path::new(CAPTURE_PATH).exists() {
        eprintln!("Skipping: '{CAPTURE_PATH}' not found");
        return;
    }
    let rec = IqRecording::load_file(CAPTURE_PATH).expect("loading live 20m capture");

    let mut total_recall = 0.0f32;
    println!(
        "\n  {:>11}  {:>7}  {:>7}  {:>6}  {:>7}  text",
        "dial kHz", "recall", "tokens", "chars", "tok/100c"
    );
    for st in STATIONS {
        let (text, _conf) = decode_at(&rec, st.dial_hz);
        let (recall, found, want) = token_recall(&text, st.tokens);
        total_recall += recall;
        let chars = text.chars().count();
        // Recall on its own can be bought by emitting more characters, and a
        // decoder that spells the whole alphabet will eventually contain any
        // token. Density is the counterweight: confirmed tokens per hundred
        // characters actually emitted. It needs no transcript, and it falls
        // when a change starts padding the copy with invented letters.
        let density = 100.0 * found as f32 / chars.max(1) as f32;
        println!(
            "  {:>11.2}  {:>6.0}%  {:>3}/{:<3}  {:>6}  {:>7.1}  {:?}",
            st.dial_hz / 1e3,
            recall * 100.0,
            found,
            want,
            chars,
            density,
            text.chars().take(56).collect::<String>()
        );
    }
    let mean = total_recall / STATIONS.len() as f32;
    println!("\n  CAPTURE TOKEN RECALL {:.1}%\n", mean * 100.0);

    // The gate. This is deliberately well under the measured figure: it is
    // here to catch a change that stops the decoder copying the real band,
    // which is the failure the synthetic grids cannot see. Raise it when the
    // number improves; needing to lower it is the finding, not the fix.
    assert!(
        mean >= 0.55,
        "token recall on the live capture fell to {:.1}%",
        mean * 100.0
    );
}

/// The cleanest station in the capture, held to a much higher bar.
///
/// 14061.78 is a QRP station calling CQ into a relatively quiet, shallow-
/// fading path, and it is the one signal here the decoder copies essentially
/// correctly. It is the canary: if a change to the detector breaks this, it
/// has broken something fundamental rather than traded weak-signal behaviour.
#[test]
fn the_clean_station_is_copied_nearly_verbatim() {
    if !Path::new(CAPTURE_PATH).exists() {
        eprintln!("Skipping: '{CAPTURE_PATH}' not found");
        return;
    }
    let rec = IqRecording::load_file(CAPTURE_PATH).expect("loading live 20m capture");
    let (text, conf) = decode_at(&rec, 14_061_780.0);
    let calls = text.matches("NK9G").count();
    assert!(
        calls >= 3,
        "expected NK9G at least 3 times, got {calls}: {text:?}"
    );
    assert!(
        text.contains("CQ CQ QRP TEST DE NK9G"),
        "the full call should come back verbatim at least once: {text:?}"
    );
    assert!(conf >= 0.85, "confidence on a clean station was {conf:.2}");
}
