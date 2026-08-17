//! Round-trip tests: synthesise each mode, push it through the decoder and
//! check the text comes back out.

use super::psk31::VARICODE;
use super::*;
use num_complex::Complex32;
use std::f32::consts::PI;

const FS: f64 = 8000.0;

fn noise(rng: &mut u32) -> f32 {
    // xorshift, so the tests stay deterministic without a dependency
    *rng ^= *rng << 13;
    *rng ^= *rng >> 17;
    *rng ^= *rng << 5;
    (*rng as f32 / u32::MAX as f32) - 0.5
}

const CQ: &str = "CQ CQ DE W1AW W1AW K";
const RY_CQ: &str = "RYRY CQ CQ DE W1AW W1AW K ";

// ------------------------------------------------------------------- CW

fn morse_for(c: char) -> &'static str {
    match c {
        'A' => ".-",
        'B' => "-...",
        'C' => "-.-.",
        'D' => "-..",
        'E' => ".",
        'F' => "..-.",
        'G' => "--.",
        'H' => "....",
        'I' => "..",
        'J' => ".---",
        'K' => "-.-",
        'L' => ".-..",
        'M' => "--",
        'N' => "-.",
        'O' => "---",
        'P' => ".--.",
        'Q' => "--.-",
        'R' => ".-.",
        'S' => "...",
        'T' => "-",
        'U' => "..-",
        'V' => "...-",
        'W' => ".--",
        'X' => "-..-",
        'Y' => "-.--",
        'Z' => "--..",
        '0' => "-----",
        '1' => ".----",
        '2' => "..---",
        '3' => "...--",
        '4' => "....-",
        '5' => ".....",
        '6' => "-....",
        '7' => "--...",
        '8' => "---..",
        '9' => "----.",
        '/' => "-..-.",
        '?' => "..--..",
        _ => "",
    }
}

/// Residual tuning error the decoder actually sees.
///
/// The tuning chain mixes the cursor (or the auto slot's dial) to DC, and
/// `CwDecoder::offset_shift` is zero, so a CW carrier arrives near baseband —
/// never at an audio pitch. The decoder's own +/-180 Hz search exists to mop
/// up what the classifier got wrong, so a small offset is the realistic case.
/// Feeding it a 600 Hz tone tests a receiver the app never builds.
const CW_TONE: f32 = 50.0;

fn gen_cw(text: &str, wpm: f32, snr_scale: f32) -> Vec<Complex32> {
    gen_cw_at(text, wpm, snr_scale, CW_TONE)
}

fn gen_cw_at(text: &str, wpm: f32, snr_scale: f32, tone: f32) -> Vec<Complex32> {
    key_to_iq(&gen_cw_key(text, wpm, 3.0, &[1.0]), snr_scale, tone)
}

/// Key line for `text`: dahs are `dah_units` dits long and every element's
/// length is multiplied by the next entry of `jitter`, cycling — a crude
/// model of a human fist rather than a keyer.
fn gen_cw_key(text: &str, wpm: f32, dah_units: f32, jitter: &[f32]) -> Vec<f32> {
    let dit = (1.2 / wpm * FS as f32) as usize;
    let mut ji = 0usize;
    let mut key: Vec<f32> = Vec::new();
    // lead-in silence lets the threshold tracker settle
    key.extend(std::iter::repeat(0.0).take(dit * 8));
    for ch in text.chars() {
        if ch == ' ' {
            key.extend(std::iter::repeat(0.0).take(dit * 4));
            continue;
        }
        for el in morse_for(ch).chars() {
            let units = if el == '.' { 1.0 } else { dah_units };
            let n = ((dit as f32) * units * jitter[ji % jitter.len()]) as usize;
            ji += 1;
            key.extend(std::iter::repeat(1.0).take(n.max(1)));
            key.extend(std::iter::repeat(0.0).take(dit)); // inter-element
        }
        key.extend(std::iter::repeat(0.0).take(dit * 2)); // char gap (total 3)
    }
    // Trailing silence. The decoder flushes the last character once six dits
    // of idle have passed, so the tail has to outlast that — and the tuning
    // chain in front of it is overlap-save, which holds back a block of about
    // 1.2 k samples that never reach the decoder at all. Eight dits alone is
    // enough at 12 WPM and not at 18, which capped every accuracy figure in
    // this file at 90 % for the missing final character regardless of what
    // the decoder did. The fixed pad covers the chain's held block at any
    // speed; a real receiver keeps running after the station stops, so this
    // is the honest case rather than a favour to the decoder.
    key.extend(std::iter::repeat(0.0).take(dit * 8 + 2048));
    key
}

/// AM the key line onto `tone` with noise, as `gen_cw_at` always did.
fn key_to_iq(key: &[f32], snr_scale: f32, tone: f32) -> Vec<Complex32> {
    key_to_iq_seed(key, snr_scale, tone, 0x1234_5678)
}

/// As `key_to_iq`, with the noise realisation chosen by the caller.
///
/// One seed per cell makes a grid of single trials, and near the copy
/// threshold a single trial is close to a coin flip: whether one particular
/// noise burst lands inside one particular dah decides the character, and the
/// cell reads 100 % or 0 % on that. Tuning against that measures the seed.
fn key_to_iq_seed(key: &[f32], snr_scale: f32, tone: f32, seed: u32) -> Vec<Complex32> {
    // Shape the key envelope so it has realistic rise/fall instead of clicks.
    let rise = (0.005 * FS as f32) as usize;
    let mut env = key.to_vec();
    let mut acc = 0.0f32;
    let a = 1.0 - (-1.0 / rise as f32).exp();
    for v in env.iter_mut() {
        acc += (*v - acc) * a;
        *v = acc;
    }

    let mut rng = seed;
    env.iter()
        .enumerate()
        .map(|(i, &e)| {
            let ph = 2.0 * PI * tone * i as f32 / FS as f32;
            let s = Complex32::from_polar(e, ph);
            s + Complex32::new(noise(&mut rng), noise(&mut rng)) * snr_scale
        })
        .collect()
}

/// `gen_cw_at` with the noise realisation chosen by the caller.
fn gen_cw_seed(text: &str, wpm: f32, snr_scale: f32, tone: f32, seed: u32) -> Vec<Complex32> {
    key_to_iq_seed(&gen_cw_key(text, wpm, 3.0, &[1.0]), snr_scale, tone, seed)
}

#[test]
fn cw_decodes_clean_signal() {
    let msg = "CQ CQ DE W1AW K";
    let sig = gen_cw(msg, 20.0, 0.02);
    let mut d = cw::CwDecoder::new(FS);
    let mut out = String::new();
    for chunk in sig.chunks(4096) {
        out.push_str(&d.process(chunk));
    }
    let got = out.trim().to_string();
    assert!(
        got.contains("CQ") && got.contains("W1AW"),
        "expected to see the callsign, got {got:?} ({})",
        d.status()
    );
}

#[test]
fn cw_tracks_speed() {
    for wpm in [15.0f32, 25.0, 35.0] {
        let sig = gen_cw("PARIS PARIS PARIS", wpm, 0.01);
        let mut d = cw::CwDecoder::new(FS);
        for chunk in sig.chunks(4096) {
            let _ = d.process(chunk);
        }
        let est = d.wpm();
        assert!(
            (est - wpm).abs() < wpm * 0.25,
            "speed estimate {est:.1} too far from {wpm:.1}"
        );
    }
}

/// A mid-stream speed change must be followed, not locked to the old dit.
#[test]
fn cw_follows_a_speed_change() {
    let slow = gen_cw("PARIS PARIS ", 15.0, 0.01);
    let fast = gen_cw("PARIS PARIS", 32.0, 0.01);
    let mut d = cw::CwDecoder::new(FS);
    for chunk in slow.chunks(4096) {
        let _ = d.process(chunk);
    }
    let mid = d.wpm();
    assert!(
        (mid - 15.0).abs() < 6.0,
        "should have locked ~15 WPM first, got {mid:.1}"
    );
    let mut out = String::new();
    for chunk in fast.chunks(4096) {
        out.push_str(&d.process(chunk));
    }
    let est = d.wpm();
    assert!(
        (est - 32.0).abs() < 10.0,
        "failed to follow 15→32 WPM, estimate {est:.1}"
    );
    assert!(
        out.contains("PARIS") || out.contains("ARIS"),
        "speed-changed text lost: {out:?} ({})",
        d.status()
    );
}

/// Calls scraped out of a decoder's own copy, the way the reporter takes them.
fn spotted(d: &mut dyn Decoder, sig: &[Complex32]) -> Vec<String> {
    let mut calls = Vec::new();
    for chunk in sig.chunks(4096) {
        d.process(chunk);
        calls.extend(d.take_messages().into_iter().map(|m| m.text));
    }
    calls
}

/// The whole point of the CW spotter: a station calling CQ ends up on the map.
#[test]
fn cw_spots_a_calling_station() {
    let sig = gen_cw("CQ CQ DE W1AW W1AW K", 20.0, 0.02);
    let mut d = cw::CwDecoder::new(FS);
    let calls = spotted(&mut d, &sig);
    assert!(
        calls.iter().any(|c| c == "CQ W1AW"),
        "no spot from a clean CQ, got {calls:?} ({})",
        d.status()
    );
}

/// ...and so does a station working someone, which on CW is the commoner case
/// by far. "to, from" ordering means the second call of the pair is sending.
#[test]
fn cw_spots_the_sender_of_an_exchange() {
    let sig = gen_cw("W1AW K1ABC 5NN 001 ", 20.0, 0.02);
    let mut d = cw::CwDecoder::new(FS);
    let calls = spotted(&mut d, &sig);
    assert!(
        calls.iter().any(|c| c == "CQ K1ABC"),
        "expected the sending station, got {calls:?} ({})",
        d.status()
    );
    assert!(
        !calls.iter().any(|c| c == "CQ W1AW"),
        "spotted the station being worked: {calls:?}"
    );
}

/// A signal report is not a callsign, however much `5NN` looks like one.
#[test]
fn cw_does_not_spot_a_contest_exchange_as_a_station() {
    let sig = gen_cw("W1AW 5NN 5NN TU ", 20.0, 0.02);
    let mut d = cw::CwDecoder::new(FS);
    let calls = spotted(&mut d, &sig);
    assert!(calls.is_empty(), "spotted a signal report: {calls:?}");
}

/// An empty frequency spells letters through the slicer. None of them may
/// reach pskreporter as a callsign — the report is wrong on someone else's map
/// and there is no way to take it back.
#[test]
fn cw_spots_nothing_from_noise() {
    let mut rng = 0x1234_5678u32;
    let sig: Vec<Complex32> = (0..(FS as usize * 20))
        .map(|_| Complex32::new(noise(&mut rng), noise(&mut rng)))
        .collect();
    let mut d = cw::CwDecoder::new(FS);
    let calls = spotted(&mut d, &sig);
    assert!(calls.is_empty(), "noise produced spots: {calls:?}");
}

#[test]
fn rtty_spots_a_calling_station() {
    let sig = gen_rtty("RYRY CQ CQ DE W1AW W1AW K ", 45.45, 170.0);
    let mut d = rtty::RttyDecoder::new(FS);
    let calls = spotted(&mut d, &sig);
    assert!(
        calls.iter().any(|c| c == "CQ W1AW"),
        "no spot from a clean RTTY CQ, got {calls:?} ({})",
        d.status()
    );
}

#[test]
fn rtty_spots_nothing_from_noise() {
    let mut rng = 0x9E37_79B9u32;
    let sig: Vec<Complex32> = (0..(FS as usize * 20))
        .map(|_| Complex32::new(noise(&mut rng), noise(&mut rng)))
        .collect();
    let mut d = rtty::RttyDecoder::new(FS);
    let calls = spotted(&mut d, &sig);
    assert!(calls.is_empty(), "noise produced spots: {calls:?}");
}

/// A human fist: light dahs (2.2 dits) with ±15% element jitter, so the
/// shortest dahs land at 1.87 dits. A fixed dit/dah boundary at 2.0
/// misreads those; the adaptive boundary must settle between the
/// operator's own clusters.
#[test]
fn cw_decodes_a_sloppy_fist() {
    let key = gen_cw_key(
        "CQ CQ DE W1AW W1AW K",
        20.0,
        2.2,
        &[0.85, 1.1, 1.0, 0.9, 1.15],
    );
    let sig = key_to_iq(&key, 0.02, CW_TONE);
    let mut d = cw::CwDecoder::new(FS);
    let mut out = String::new();
    for chunk in sig.chunks(4096) {
        out.push_str(&d.process(chunk));
    }
    assert!(
        out.contains("W1AW"),
        "sloppy weighting lost the callsign: {out:?} ({})",
        d.status()
    );
}

/// Farnsworth timing: characters sent at high speed (24 WPM) with stretched
/// spacing (equivalent to 16 WPM spacing).
#[test]
fn cw_decodes_farnsworth_timing() {
    let text = "CQ CQ DE W1AW K";
    let dit_elem = (1.2 / 24.0 * FS as f32) as usize;
    let dit_space = (1.2 / 16.0 * FS as f32) as usize;
    let mut key: Vec<f32> = Vec::new();
    key.extend(std::iter::repeat(0.0).take(dit_elem * 8));
    for ch in text.chars() {
        if ch == ' ' {
            key.extend(std::iter::repeat(0.0).take(dit_space * 5));
            continue;
        }
        for el in morse_for(ch).chars() {
            let units = if el == '.' { 1.0 } else { 3.0 };
            let n = ((dit_elem as f32) * units) as usize;
            key.extend(std::iter::repeat(1.0).take(n));
            key.extend(std::iter::repeat(0.0).take(dit_elem));
        }
        key.extend(std::iter::repeat(0.0).take(dit_space * 2));
    }
    key.extend(std::iter::repeat(0.0).take(dit_elem * 8));
    let sig = key_to_iq(&key, 0.02, CW_TONE);
    let mut d = cw::CwDecoder::new(FS);
    let mut out = String::new();
    for chunk in sig.chunks(4096) {
        out.push_str(&d.process(chunk));
    }
    assert!(
        out.contains("W1AW") && out.contains("CQ"),
        "Farnsworth timing lost copy: {out:?} ({})",
        d.status()
    );
}

/// Semi-automatic bug keyer: long dahs (3.8x) with light jitter.
#[test]
fn cw_decodes_bug_keyer_weighting() {
    let key = gen_cw_key("CQ CQ DE W1AW W1AW K", 22.0, 3.8, &[0.90, 1.15, 1.0, 0.95]);
    let sig = key_to_iq(&key, 0.02, CW_TONE);
    let mut d = cw::CwDecoder::new(FS);
    let mut out = String::new();
    for chunk in sig.chunks(4096) {
        out.push_str(&d.process(chunk));
    }
    assert!(
        out.contains("W1AW"),
        "bug keyer weighting lost the callsign: {out:?} ({})",
        d.status()
    );
}

/// All-dit sequences (e.g. 5, H, S, E, I) must not trigger deaf warm-up state resets.
#[test]
fn cw_decodes_all_dit_sequences() {
    let sig = gen_cw("555 5NN H H S S E E 73", 20.0, 0.02);
    let mut d = cw::CwDecoder::new(FS);
    let mut out = String::new();
    for chunk in sig.chunks(4096) {
        out.push_str(&d.process(chunk));
    }
    assert!(
        out.contains("555") || out.contains("5NN") || out.contains("73"),
        "all-dit sequence caused decoder failure: {out:?} ({})",
        d.status()
    );
}

/// QSB dropouts and static crashes: brief envelope glitches must be
/// debounced, not decoded as extra elements or split gaps.
#[test]
fn cw_survives_dropouts_and_spikes() {
    let mut sig = gen_cw("CQ CQ DE W1AW K", 20.0, 0.02);
    let glitch = (0.007 * FS as f32) as usize;
    // 7 ms fades every 150 ms...
    let mut i = (0.15 * FS as f32) as usize;
    while i + glitch < sig.len() {
        for s in &mut sig[i..i + glitch] {
            *s *= 0.02;
        }
        i += (0.15 * FS as f32) as usize;
    }
    // ...and 7 ms carrier bursts every 190 ms (in phase with the keyed
    // tone, so inside a mark they add rather than cancel).
    let mut i = (0.095 * FS as f32) as usize;
    while i + glitch < sig.len() {
        for (k, s) in sig[i..i + glitch].iter_mut().enumerate() {
            let ph = 2.0 * PI * CW_TONE * (i + k) as f32 / FS as f32;
            *s += Complex32::from_polar(0.9, ph);
        }
        i += (0.19 * FS as f32) as usize;
    }
    let mut d = cw::CwDecoder::new(FS);
    let mut out = String::new();
    for chunk in sig.chunks(4096) {
        out.push_str(&d.process(chunk));
    }
    assert!(
        out.contains("W1AW"),
        "glitched signal lost the callsign: {out:?} ({})",
        d.status()
    );
}

#[test]
fn cw_view_exposes_envelope_and_center() {
    let sig = gen_cw("CQ DE W1AW", 20.0, 0.02);
    let mut d = cw::CwDecoder::new(FS);
    for chunk in sig.chunks(4096) {
        let _ = d.process(chunk);
    }
    let v = d.cw_view().expect("CW view");
    assert!(
        v.env.len() > 50,
        "envelope history too short: {}",
        v.env.len()
    );
    assert_eq!(v.env.len(), v.keyed.len());
    assert!(v.wpm > 8.0 && v.wpm < 35.0, "wpm {}", v.wpm);
    let before = v.lock_hz;
    let after = d.nudge_lock(5.0).unwrap();
    assert!((after - before - 5.0).abs() < 0.1);
}

/// The span scout must confirm keyed Morse and ignore a dead carrier.
#[test]
fn cw_span_scout_finds_cw_and_skips_a_carrier() {
    let cw_sig = gen_cw_at("CQ CQ DE W1AW", 20.0, 0.02, 400.0);
    let n = cw_sig.len();
    let mut iq = cw_sig;
    for i in 0..n {
        let ph = 2.0 * PI * (-250.0) * i as f32 / FS as f32;
        iq[i] += Complex32::from_polar(0.9, ph);
    }
    let hits = cw::scan_span(&iq, FS, &[(400.0, 20.0), (-250.0, 20.0)]);
    assert!(
        hits.iter().any(|h| (h.offset_hz - 400.0).abs() < 40.0),
        "scout missed the CW at +400 Hz: {hits:?}"
    );
    assert!(
        !hits.iter().any(|h| (h.offset_hz + 250.0).abs() < 40.0),
        "scout locked a plain carrier: {hits:?}"
    );
}

/// Two CW tones in the passband: lock one, then next_lock hops to the other.
#[test]
fn cw_next_lock_hops_to_the_other_signal() {
    let a = gen_cw_at("CQ CQ CQ DE TEST", 20.0, 0.015, 80.0);
    let b = gen_cw_at("CQ CQ CQ DE TEST", 20.0, 0.015, -120.0);
    let n = a.len().min(b.len());
    let sig: Vec<Complex32> = (0..n).map(|i| a[i] + b[i]).collect();
    let mut d = cw::CwDecoder::new(FS);
    for chunk in sig.chunks(4096) {
        let _ = d.process(chunk);
    }
    assert!(
        d.locked() || !d.candidate_hz().is_empty(),
        "should find at least one CW tone ({})",
        d.status()
    );
    if d.candidate_hz().len() >= 2 {
        let first = d.lock_hz();
        let hopped = d.next_lock(true);
        assert!(hopped.is_some(), "next_lock should hop ({})", d.status());
        assert!(
            (d.lock_hz() - first).abs() > 40.0,
            "next_lock stayed at {first:.1} ({})",
            d.status()
        );
    }
}

/// All-dah mark sequences (M, O, 0, 8, 9) must not confuse the clusterer into
/// cutting WPM in half.
#[test]
fn cw_decodes_all_dah_sequences_without_wpm_swings() {
    let msg = "MMMM OOOO 000 888 999 599 001";
    let sig = gen_cw(msg, 20.0, 0.01);
    let mut d = cw::CwDecoder::new(FS);
    let mut out = String::new();
    for chunk in sig.chunks(4096) {
        out.push_str(&d.process(chunk));
    }
    let wpm = d.wpm();
    assert!(
        (wpm - 20.0).abs() < 5.0,
        "WPM swung on all-dah sequence, got {wpm:.1} WPM ({})",
        d.status()
    );
    assert!(
        out.contains("MM") || out.contains("OO") || out.contains("599"),
        "all-dah sequence copy lost: {out:?}"
    );
}

/// Multi-over QSO with 2.5-3.0 second pauses between overs must decode every over cleanly.
#[test]
fn cw_decodes_multi_over_qso_with_pauses() {
    let mut d = cw::CwDecoder::new(FS);
    let over1 = gen_cw("CQ CQ DE W1AW K", 20.0, 0.01);
    let pause1 = vec![Complex32::new(0.0, 0.0); (2.5 * FS) as usize];
    let over2 = gen_cw("W1AW DE G4XYZ K", 22.0, 0.01);
    let pause2 = vec![Complex32::new(0.0, 0.0); (3.0 * FS) as usize];
    let over3 = gen_cw("G4XYZ DE W1AW 5NN TU 73 SK", 20.0, 0.01);

    let mut out1 = String::new();
    for chunk in over1.chunks(4096) {
        out1.push_str(&d.process(chunk));
    }
    for chunk in pause1.chunks(4096) {
        let _ = d.process(chunk);
    }

    let mut out2 = String::new();
    for chunk in over2.chunks(4096) {
        out2.push_str(&d.process(chunk));
    }
    for chunk in pause2.chunks(4096) {
        let _ = d.process(chunk);
    }

    let mut out3 = String::new();
    for chunk in over3.chunks(4096) {
        out3.push_str(&d.process(chunk));
    }

    assert!(
        out1.contains("W1AW"),
        "Over 1 lost: {out1:?} ({})",
        d.status()
    );
    assert!(
        out2.contains("G4XYZ") || out2.contains("4XYZ"),
        "Over 2 lost after pause: {out2:?} ({})",
        d.status()
    );
    assert!(
        out3.contains("5NN") || out3.contains("73"),
        "Over 3 lost after pause: {out3:?} ({})",
        d.status()
    );
}

/// Short transmissions ("K", "TU", "5NN", "73", "BK") must decode and achieve confidence >= 0.40.
#[test]
fn cw_decodes_short_overs_with_high_confidence() {
    for word in ["5NN", "TU", "73", "BK", "K"] {
        let sig = gen_cw(word, 20.0, 0.01);
        let mut d = cw::CwDecoder::new(FS);
        let mut out = String::new();
        for chunk in sig.chunks(4096) {
            out.push_str(&d.process(chunk));
        }
        let conf = d.confidence().unwrap_or(0.0);
        assert!(
            conf >= 0.40,
            "Short over {word:?} produced low confidence {conf:.2} (needs >= 0.40 to pass copy_floor)"
        );
        assert!(
            out.contains(word) || out.contains(&word[..word.len().min(2)]),
            "Short over {word:?} lost in copy: {out:?}"
        );
    }
}

// ----------------------------------------------------------------- RTTY

fn ita2_code(c: char) -> Option<(u8, bool)> {
    const LTRS: &str = "\0E\nA SIU\rDRJNFCKTZLWHYPQOBG\0MXV\0";
    // Look the character up in the letters table first, then figures.
    if let Some(i) = LTRS.chars().position(|x| x == c) {
        return Some((i as u8, false));
    }
    let figs = [
        ('3', 1),
        ('-', 3),
        ('\'', 5),
        ('8', 6),
        ('7', 7),
        ('$', 9),
        ('4', 10),
        (',', 12),
        ('!', 13),
        (':', 14),
        ('(', 15),
        ('5', 16),
        ('"', 17),
        (')', 18),
        ('2', 19),
        ('#', 20),
        ('6', 21),
        ('0', 22),
        ('1', 23),
        ('9', 24),
        ('?', 25),
        ('&', 26),
        ('.', 28),
        ('/', 29),
        (';', 30),
    ];
    figs.iter()
        .find(|(x, _)| *x == c)
        .map(|(_, i)| (*i as u8, true))
}

fn gen_rtty(text: &str, baud: f32, shift: f32) -> Vec<Complex32> {
    gen_rtty_snr(text, baud, shift, 0.03)
}

fn gen_rtty_snr(text: &str, baud: f32, shift: f32, snr_scale: f32) -> Vec<Complex32> {
    gen_rtty_faded(text, baud, shift, snr_scale, 1.0, 1.0)
}

/// The ITA2 bit stream for `text`, framed as the decoder expects it.
fn rtty_bits(text: &str, baud: f32) -> Vec<bool> {
    let mut bits: Vec<bool> = Vec::new();
    // idle mark so the decoder starts in a known state
    bits.extend(std::iter::repeat(true).take((baud as usize).max(20)));
    let mut figs_state = false;
    for c in text.chars() {
        let Some((code, figs)) = ita2_code(c) else {
            continue;
        };
        if figs != figs_state && c != ' ' {
            let shift_code = if figs { 0x1B } else { 0x1F };
            bits.push(false); // start
            for b in 0..5 {
                bits.push(shift_code & (1 << b) != 0);
            }
            bits.push(true); // stop
            bits.push(true); // 1.5 stop bits, rounded up
            figs_state = figs;
        }
        bits.push(false); // start
        for b in 0..5 {
            bits.push(code & (1 << b) != 0);
        }
        bits.push(true);
        bits.push(true);
    }
    bits.extend(std::iter::repeat(true).take(20));
    bits
}

/// Noiseless, unmodulated-amplitude RTTY: the input a channel model takes.
///
/// `gen_rtty_faded` bakes its noise in, which is fine for a fixed tone
/// imbalance but wrong once a channel is in circuit — the fading has to be
/// applied to the signal alone, before the noise floor is added, or the
/// "SNR" being swept is not one.
fn gen_rtty_clean(text: &str, baud: f32, shift: f32) -> Vec<Complex32> {
    modulate_rtty(&rtty_bits(text, baud), baud, shift)
}

/// FSK-modulate an arbitrary bit stream at unit amplitude, continuous phase.
fn modulate_rtty(bits: &[bool], baud: f32, shift: f32) -> Vec<Complex32> {
    let sps = FS as f32 / baud;
    let mut out = Vec::new();
    let mut phase = 0.0f32;
    for &mark in bits {
        let f = if mark { shift / 2.0 } else { -shift / 2.0 };
        for _ in 0..sps as usize {
            phase += 2.0 * PI * f / FS as f32;
            out.push(Complex32::from_polar(1.0, phase));
        }
    }
    out
}

fn gen_rtty_faded(
    text: &str,
    baud: f32,
    shift: f32,
    snr_scale: f32,
    mark_amp: f32,
    space_amp: f32,
) -> Vec<Complex32> {
    let sps = FS as f32 / baud;
    let bits = rtty_bits(text, baud);

    let mut rng = 0x9e37_79b9u32;
    let mut out = Vec::new();
    let mut phase = 0.0f32;
    for (i, _) in bits.iter().enumerate() {
        for n in 0..sps as usize {
            let idx = i;
            let mark = bits[idx];
            let f = if mark { shift / 2.0 } else { -shift / 2.0 };
            phase += 2.0 * PI * f / FS as f32;
            let _ = n;
            let amp = if mark { mark_amp } else { space_amp };
            out.push(
                Complex32::from_polar(amp, phase)
                    + Complex32::new(noise(&mut rng), noise(&mut rng)) * snr_scale,
            );
        }
    }
    out
}

#[test]
fn rtty_matched_filters_survive_selective_fading() {
    let sig = gen_rtty_faded("RYRY RYRY CQ DE TEST TEST", 45.45, 170.0, 0.025, 1.0, 0.1);
    let mut d = rtty::RttyDecoder::new(FS);
    let mut out = String::new();
    for chunk in sig.chunks(4096) {
        out.push_str(&d.process(chunk));
    }
    assert!(
        out.contains("TEST"),
        "-20 dB space fade was not copyable: {out:?} ({})",
        d.status()
    );
}

/// `snr_scale` for a wanted SNR in dB measured in the bit-rate bandwidth —
/// 45.45 Hz, which is the noise bandwidth a matched filter for one bit has,
/// and therefore the figure the demodulator's own performance is set by.
fn rtty_scale_for_snr(db: f32) -> f32 {
    let n_total = (1.0f32 / 6.0).sqrt();
    let in_bw = n_total * (45.45 / FS as f32).sqrt();
    let wanted = 10f32.powf(-db / 20.0);
    wanted / in_bw
}

/// The tone filters have to be matched to a bit.
///
/// What stood here was a one-pole leaky integrator with a time constant of a
/// third of a bit — several times the noise bandwidth the signal occupies,
/// and most of the coherent gain an FSK tone offers thrown away. Replacing it
/// with an integrate-and-dump across exactly one bit, dumped on the framer's
/// own boundary, took 12 dB copy from 73% to 95% and 8 dB from 32% to 50%.
#[test]
fn rtty_copies_at_twelve_db() {
    let msg = "RYRY CQ DE W1AW W1AW K";
    let sig = gen_rtty_snr(msg, 45.45, 170.0, rtty_scale_for_snr(12.0));
    let got = decode_rtty(&sig);
    let acc = accuracy(msg, &got);
    assert!(
        acc >= 0.85,
        "12 dB RTTY copied {:.0}%, was 73% with the leaky integrator: {got:?}",
        acc * 100.0
    );
}

#[test]
#[ignore]
fn bench_rtty_snr() {
    let msg = "RYRY CQ DE W1AW W1AW K";
    println!("\nsent: {msg:?}");
    println!("\n== character accuracy by SNR (in 45.45 Hz) ==");
    for db in [20, 15, 12, 10, 8, 6, 4, 2] {
        let sig = gen_rtty_snr(msg, 45.45, 170.0, rtty_scale_for_snr(db as f32));
        let got = decode_rtty(&sig);
        println!(
            "  {db:>2} dB  {:>5}  {got:?}",
            format!("{:.0}%", accuracy(msg, &got) * 100.0)
        );
    }
}

#[test]
#[ignore]
fn bench_rtty_matched_filter_fade() {
    for fade_db in [0.0f32, -10.0, -20.0] {
        let amp = 10f32.powf(fade_db / 20.0);
        let sig = gen_rtty_faded("RYRY RYRY CQ DE TEST TEST", 45.45, 170.0, 0.025, 1.0, amp);
        let mut d = rtty::RttyDecoder::new(FS);
        let mut out = String::new();
        for chunk in sig.chunks(4096) {
            out.push_str(&d.process(chunk));
        }
        println!(
            "RTTY matched filters, space {fade_db:+.0} dB: {} chars, TEST={}",
            out.len(),
            out.contains("TEST")
        );
    }
}

// --------------------------------------------- RTTY through a real channel

/// RTTY through a Watterson channel: fade the signal, *then* add the noise.
///
/// `snr_db` is the mean SNR in the 45.45 Hz bit bandwidth, mean being the
/// operative word — the instantaneous value is what the channel decides.
fn gen_rtty_channel(
    text: &str,
    baud: f32,
    shift: f32,
    cond: Condition,
    snr_db: f32,
    seed: u32,
) -> Vec<Complex32> {
    let mut sig = channel::watterson(
        &gen_rtty_clean(text, baud, shift),
        FS as f32,
        cond,
        seed,
    );
    let mut rng = seed ^ 0xa5a5_a5a5;
    let n_scale = rtty_scale_for_snr(snr_db);
    for s in sig.iter_mut() {
        *s += Complex32::new(noise(&mut rng), noise(&mut rng)) * n_scale;
    }
    sig
}

const RTTY_FADE_SEEDS: [u32; 8] = [
    0x1234_5678,
    0x9e37_79b9,
    0x0bad_f00d,
    0x5eed_1234,
    0xdead_beef,
    0x0f0f_1e1e,
    0xc0ff_ee11,
    0x3141_5926,
];
const RTTY_FADE_MSG: &str = "RYRY RYRY CQ CQ DE W1AW W1AW K UR RST 599 QTH NEWINGTON";

/// Mean copy across `RTTY_FADE_SEEDS` for one channel and SNR.
fn rtty_fade_cell(cond: Condition, snr_db: f32) -> f32 {
    let n = RTTY_FADE_SEEDS.len() as f32;
    RTTY_FADE_SEEDS
        .iter()
        .map(|&seed| {
            let sig = gen_rtty_channel(RTTY_FADE_MSG, 45.45, 170.0, cond, snr_db, seed);
            accuracy(RTTY_FADE_MSG, &decode_rtty(&sig))
        })
        .sum::<f32>()
        / n
}

/// An operator pausing mid-over must not cost the rest of the transmission.
///
/// This is the hazard a *short* per-tone hold buys: RTTY idles on mark, so
/// through a long pause the space tone carries nothing and `space_peak`
/// decays toward the noise. Come the next character, `es/space_peak` is
/// inflated by however far it fell, and the discriminator reads spaces that
/// are not there. The hold has to outlast a pause a real operator takes.
#[test]
fn rtty_recovers_after_a_long_idle_mark() {
    const TAIL: &str = "DE W1AW W1AW K";
    for idle_s in [2.0f32, 5.0, 10.0] {
        let mut bits = rtty_bits("CQ CQ", 45.45);
        bits.extend(std::iter::repeat(true).take((45.45 * idle_s) as usize));
        bits.extend(rtty_bits(TAIL, 45.45));
        let sig = modulate_rtty(&bits, 45.45, 170.0);
        let mut rng = 0x1234_5678u32;
        let n_scale = rtty_scale_for_snr(15.0);
        let sig: Vec<Complex32> = sig
            .iter()
            .map(|s| s + Complex32::new(noise(&mut rng), noise(&mut rng)) * n_scale)
            .collect();
        let got = decode_rtty(&sig);
        assert!(
            accuracy(TAIL, got.split("CQ").last().unwrap_or("")) >= 0.6,
            "{idle_s}s idle mark then {TAIL:?}: got {got:?}"
        );
    }
}

/// Copy against path and SNR, the instrument RTTY did not have.
///
/// A 170 Hz shift straddles the coherence bandwidth of every CCIR path here —
/// `1/(2*pi*delay)` is 318 Hz at `CCIR_GOOD` and 80 Hz at `CCIR_POOR` — so
/// the mark and space tones fade independently, which is the case the
/// discriminator's per-tone max-hold has to survive and the case
/// `gen_rtty_faded`'s static amplitudes cannot produce.
#[test]
#[ignore]
fn bench_rtty_fading() {
    println!("\n  RTTY copy vs path and SNR (in 45.45 Hz), mean of 4 seeds\n");
    print!("{:>16}", "path");
    for db in [20, 15, 12, 10, 8] {
        print!("{:>8}", format!("{db}dB"));
    }
    println!();
    for (name, cond) in [
        ("flat", channel::FLAT),
        ("good", channel::CCIR_GOOD),
        ("moderate", channel::CCIR_MODERATE),
        ("poor", channel::CCIR_POOR),
        ("flutter", channel::CCIR_FLUTTER),
    ] {
        print!("{name:>16}");
        for db in [20.0f32, 15.0, 12.0, 10.0, 8.0] {
            print!("{:>8}", format!("{:.0}%", rtty_fade_cell(cond, db) * 100.0));
        }
        println!();
    }
}

/// Fading, not noise, is what limits RTTY copy — so it needs a gate.
///
/// At 20 dB in the bit bandwidth the flat case is a solved problem: 100 %.
/// What these two cells measure is the cost of the ionosphere alone at a
/// level where noise costs nothing, which is where `PEAK_HOLD_S` was found
/// and the only place a regression in the tone references will show.
#[test]
fn rtty_fading_copy_does_not_regress() {
    for (name, cond, floor) in [
        ("moderate", channel::CCIR_MODERATE, 0.72f32),
        ("poor", channel::CCIR_POOR, 0.62),
    ] {
        let got = rtty_fade_cell(cond, 20.0);
        assert!(
            got >= floor,
            "RTTY {name} at 20 dB copied {:.0}%, below the {:.0}% gate \
             (was 77%/58% with the 1.5 s hold, 83%/74% with 0.3 s)",
            got * 100.0,
            floor * 100.0
        );
    }
}

#[test]
fn rtty_decodes_baudot() {
    let msg = "RYRY CQ DE TEST";
    let sig = gen_rtty(msg, 45.45, 170.0);
    let mut d = rtty::RttyDecoder::new(FS);
    let mut out = String::new();
    for chunk in sig.chunks(4096) {
        out.push_str(&d.process(chunk));
    }
    assert!(
        out.contains("RYRY") && out.contains("TEST"),
        "expected RYRY/TEST, got {out:?} ({})",
        d.status()
    );
}

/// The whole FSK pair sitting 100 Hz off the cursor — beyond half the
/// shift, so a fixed 0 Hz slicer never even sees a start bit. The
/// adaptive threshold has to find the midpoint on its own.
#[test]
fn rtty_tolerates_a_tuning_offset() {
    let sig = gen_rtty("RYRY RYRY CQ DE TEST TEST", 45.45, 170.0);
    let off: Vec<Complex32> = sig
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let ph = 2.0 * PI * 100.0 * i as f32 / FS as f32;
            s * Complex32::from_polar(1.0, ph)
        })
        .collect();
    let mut d = rtty::RttyDecoder::new(FS);
    let mut out = String::new();
    for chunk in off.chunks(4096) {
        out.push_str(&d.process(chunk));
    }
    assert!(
        out.contains("TEST"),
        "offset RTTY failed: {out:?} ({})",
        d.status()
    );
}

#[test]
fn rtty_reverse_shift_is_garbage_then_recovers() {
    let sig = gen_rtty("TEST TEST", 45.45, 170.0);
    let mut d = rtty::RttyDecoder::new(FS);
    d.toggle(); // reversed: should not produce the message
    let mut out = String::new();
    for chunk in sig.chunks(4096) {
        out.push_str(&d.process(chunk));
    }
    assert!(
        !out.contains("TEST"),
        "reversed shift should not decode: {out:?}"
    );
}

// ---------------------------------------------------------------- PSK31

fn gen_psk31(text: &str, freq_offset: f32) -> Vec<Complex32> {
    gen_psk31_snr(text, freq_offset, 0.02)
}

fn gen_psk31_snr(text: &str, freq_offset: f32, snr_scale: f32) -> Vec<Complex32> {
    let clean = gen_psk31_clean(text, freq_offset);
    let mut rng = 0xdead_beefu32;
    clean
        .iter()
        .map(|&s| s + Complex32::new(noise(&mut rng), noise(&mut rng)) * snr_scale)
        .collect()
}

/// Noiseless PSK31: the input a channel model takes.
fn gen_psk31_clean(text: &str, freq_offset: f32) -> Vec<Complex32> {
    let sps = (FS as f32 / 31.25) as usize;
    let mut bits: Vec<bool> = Vec::new();
    // Idle: a run of 0 bits (continuous reversals) for the receiver to lock to.
    bits.extend(std::iter::repeat(false).take(64));
    for c in text.chars() {
        let code = VARICODE[c as usize];
        for ch in code.chars() {
            bits.push(ch == '1');
        }
        bits.push(false);
        bits.push(false); // inter-character "00"
    }
    bits.extend(std::iter::repeat(false).take(64));

    // Differential encoding: 0 flips the symbol, 1 keeps it.
    let mut syms: Vec<f32> = Vec::with_capacity(bits.len() + 1);
    let mut cur = 1.0f32;
    syms.push(cur);
    for b in &bits {
        if !*b {
            cur = -cur;
        }
        syms.push(cur);
    }

    // Raised-cosine pulses of width 2T, summed - this is what puts the
    // amplitude notch at a reversal and keeps full amplitude otherwise.
    let total = (syms.len() + 2) * sps;
    let mut base = vec![0.0f32; total];
    for (k, &a) in syms.iter().enumerate() {
        let centre = (k + 1) * sps;
        for n in 0..2 * sps {
            let idx = centre + n - sps;
            if idx >= total {
                continue;
            }
            let x = (n as f32 - sps as f32) / sps as f32; // -1..1
            let p = 0.5 * (1.0 + (PI * x).cos());
            base[idx] += a * p;
        }
    }

    base.iter()
        .enumerate()
        .map(|(i, &v)| {
            let ph = 2.0 * PI * freq_offset * i as f32 / FS as f32;
            Complex32::from_polar(v.abs(), ph + if v < 0.0 { PI } else { 0.0 })
        })
        .collect()
}

#[test]
#[ignore]
fn psk31_debug_bitstream() {
    let msg = "CQ DE TEST";
    // Rebuild the transmitted bit sequence for comparison.
    let mut tx: Vec<bool> = std::iter::repeat(false).take(64).collect();
    for c in msg.chars() {
        for ch in VARICODE[c as usize].chars() {
            tx.push(ch == '1');
        }
        tx.push(false);
        tx.push(false);
    }
    let sig = gen_psk31(msg, 0.0);
    let mut d = psk31::Psk31Decoder::new(FS);
    for chunk in sig.chunks(4096) {
        let _ = d.process(chunk);
    }
    let rx = &d.captured_bits;
    let s = |b: &[bool]| {
        b.iter()
            .map(|x| if *x { '1' } else { '0' })
            .collect::<String>()
    };
    println!("tx ({:3}): {}", tx.len(), s(&tx));
    println!("rx ({:3}): {}", rx.len(), s(rx));
    // Find the alignment that matches best.
    let mut best = (0usize, 0usize);
    for off in 0..rx.len().saturating_sub(tx.len()).min(200) {
        let n = tx.iter().zip(&rx[off..]).filter(|(a, b)| a == b).count();
        if n > best.1 {
            best = (off, n);
        }
    }
    println!("best offset {} matching {}/{}", best.0, best.1, tx.len());
}

#[test]
fn psk31_view_exposes_symbols_and_nudge() {
    let sig = gen_psk31("CQ DE TEST", 0.0);
    let mut d = psk31::Psk31Decoder::new(FS);
    for chunk in sig.chunks(4096) {
        let _ = d.process(chunk);
    }
    let v = d.psk_view().expect("PSK view");
    assert!(!v.symbols.is_empty() || !v.env.is_empty());
    let before = v.lock_hz;
    let after = d.nudge_lock(3.0).unwrap();
    assert!((after - before - 3.0).abs() < 1.0, "{before} -> {after}");
}

#[test]
fn psk31_decodes_text() {
    let msg = "CQ DE TEST";
    let sig = gen_psk31(msg, 0.0);
    let mut d = psk31::Psk31Decoder::new(FS);
    let mut out = String::new();
    for chunk in sig.chunks(4096) {
        out.push_str(&d.process(chunk));
    }
    assert!(
        out.contains("CQ DE TEST"),
        "expected {msg:?}, got {out:?} ({})",
        d.status()
    );
}

#[test]
fn psk31_tolerates_frequency_offset() {
    let sig = gen_psk31("HELLO WORLD", 3.0);
    let mut d = psk31::Psk31Decoder::new(FS);
    let mut out = String::new();
    for chunk in sig.chunks(4096) {
        out.push_str(&d.process(chunk));
    }
    assert!(
        out.contains("HELLO WORLD"),
        "offset signal failed: {out:?} ({})",
        d.status()
    );
}

/// An unmodulated tone squares to a loud peak too — it must not be
/// mistaken for PSK31 (no phase reversals).
#[test]
fn psk31_ignores_a_plain_carrier() {
    let n = (FS * 3.0) as usize;
    let sig: Vec<Complex32> = (0..n)
        .map(|i| {
            let ph = 2.0 * PI * 15.0 * i as f32 / FS as f32;
            Complex32::from_polar(1.0, ph)
        })
        .collect();
    let mut d = psk31::Psk31Decoder::new(FS);
    let mut out = String::new();
    for chunk in sig.chunks(4096) {
        out.push_str(&d.process(chunk));
    }
    assert!(
        !d.locked(),
        "carrier must not look like PSK31 ({})",
        d.status()
    );
    assert!(
        out.chars().filter(|c| c.is_ascii_alphanumeric()).count() < 3,
        "carrier leaked text: {out:?}"
    );
}

/// Offsets well outside the old ±4 Hz AFC pull-in: the searcher has to
/// identify the carrier and mix it onto DC itself.
#[test]
fn psk31_identifies_and_calibrates_to_a_nearby_signal() {
    for offset in [18.0f32, -22.0, 35.0] {
        let sig = gen_psk31("HELLO WORLD", offset);
        let mut d = psk31::Psk31Decoder::new(FS);
        let mut out = String::new();
        for chunk in sig.chunks(4096) {
            out.push_str(&d.process(chunk));
        }
        assert!(
            out.contains("HELLO WORLD"),
            "failed to lock {offset:+.0} Hz signal: {out:?} ({})",
            d.status()
        );
        assert!(
            d.locked(),
            "decoder should report lock at {offset:+.0} Hz ({})",
            d.status()
        );
        assert!(
            (d.lock_hz() - offset).abs() < 4.0,
            "calibrated {:+.1} Hz, wanted {offset:+.1} Hz ({})",
            d.lock_hz(),
            d.status()
        );
    }
}

/// Two carriers in the same passband: lock one, then `next_lock` must
/// hop to the other.
#[test]
fn psk31_next_lock_hops_to_the_other_signal() {
    let a = gen_psk31("ALPHA", 20.0);
    let b = gen_psk31("BRAVO", -50.0);
    let n = a.len().min(b.len());
    let sig: Vec<Complex32> = (0..n).map(|i| a[i] + b[i]).collect();
    let mut d = psk31::Psk31Decoder::new(FS);
    for chunk in sig.chunks(4096) {
        let _ = d.process(chunk);
    }
    assert!(d.locked(), "should lock one of the two ({})", d.status());
    let first = d.lock_hz();
    let hopped = d.next_lock(true);
    assert!(
        hopped.is_some(),
        "next_lock should find the other signal ({})",
        d.status()
    );
    let second = d.lock_hz();
    assert!(
        (first - second).abs() > 20.0,
        "next_lock stayed at {first:.1} Hz ({})",
        d.status()
    );
}

/// The span scout must confirm a real PSK31 peak and ignore a dead carrier.
#[test]
fn psk31_span_scout_finds_psk_and_skips_a_carrier() {
    let psk = gen_psk31("HELLO", 80.0);
    let n = psk.len();
    let mut iq = psk;
    // Unmodulated tone well away from the PSK31 signal.
    for i in 0..n {
        let ph = 2.0 * PI * (-120.0) * i as f32 / FS as f32;
        iq[i] += Complex32::from_polar(1.0, ph);
    }
    let hits = psk31::scan_span(&iq, FS, &[(80.0, 20.0), (-120.0, 20.0)]);
    assert!(
        hits.iter().any(|h| (h.offset_hz - 80.0).abs() < 8.0),
        "scout missed the PSK31 at +80 Hz: {hits:?}"
    );
    assert!(
        !hits.iter().any(|h| (h.offset_hz + 120.0).abs() < 8.0),
        "scout locked a plain carrier: {hits:?}"
    );
}

// ------------------------------------------------------------- FT8 / FT4

/// Synthesise a slot containing one message and decode it back. This exercises
/// the same audio conventions the live path uses: 12 kHz, signal placed at an
/// audio offset above the dial, starting 0.5 s into the slot.
#[cfg(test)]
fn ft_roundtrip(ft4: bool, text_call: &str, grid: &str, audio_hz: f32) -> Vec<String> {
    ft8::FtDecoder::decode_audio(&ft_audio(ft4, text_call, grid, audio_hz), ft4)
}

fn ft_audio(ft4: bool, text_call: &str, grid: &str, audio_hz: f32) -> Vec<i16> {
    use mfsk_core::msg::wsjt77::pack77;

    let slot_secs = if ft4 { 7.5 } else { 15.0 };
    let nmax = (slot_secs * 12_000.0) as usize;
    let msg77 = pack77("CQ", text_call, grid).expect("pack77");

    let frame: Vec<i16> = if ft4 {
        use mfsk_core::ft4::encode::{message_to_tones, tones_to_i16};
        let tones = message_to_tones(&msg77);
        tones_to_i16(&tones, audio_hz, 20_000)
    } else {
        use mfsk_core::ft8::wave_gen::{message_to_tones, tones_to_i16};
        let tones = message_to_tones(&msg77);
        tones_to_i16(&tones, audio_hz, 20_000)
    };

    let mut audio = vec![0i16; nmax];
    let start = (0.5 * 12_000.0) as usize;
    for (i, &s) in frame.iter().enumerate() {
        if start + i < audio.len() {
            audio[start + i] = s;
        }
    }
    audio
}

#[test]
fn ft8_decodes_a_slot() {
    let out = ft_roundtrip(false, "JA1ABC", "PM95", 1500.0);
    assert!(
        out.iter().any(|l| l.contains("CQ JA1ABC PM95")),
        "expected the message back, got {out:?}"
    );
}

#[test]
#[ignore]
fn bench_ft8_decode_depth() {
    let audio = ft_audio(false, "JA1ABC", "PM95", 1500.0);
    for deep in [false, true] {
        let at = std::time::Instant::now();
        let out = ft8::FtDecoder::decode_audio_depth(&audio, false, deep);
        println!(
            "FT8 {} depth: {} decode(s), {:.2}s",
            if deep { "deep" } else { "conservative" },
            out.len(),
            at.elapsed().as_secs_f64()
        );
    }
}

/// A weak signal parked next to a much stronger one must still decode: the
/// SIC pass has to subtract the strong signal and find it. This is the
/// live-band situation a hot front end (bias-T feeding an LNA) creates.
#[test]
fn ft8_decodes_weak_beside_strong() {
    use mfsk_core::ft8::wave_gen::{message_to_tones, tones_to_i16};
    use mfsk_core::msg::wsjt77::pack77;

    let nmax = 15 * 12_000;
    // Accumulate in i32 so the mix can't wrap before the final clamp.
    let mut audio = vec![0i32; nmax];
    let start = (0.5 * 12_000.0) as usize;
    let strong = pack77("CQ", "W1AW", "FN31").expect("pack77");
    let weak = pack77("CQ", "JA1ABC", "PM95").expect("pack77");
    for (msg77, hz, amp) in [(&strong, 1500.0f32, 20_000), (&weak, 1515.0, 3_500)] {
        let frame = tones_to_i16(&message_to_tones(msg77), hz, amp);
        for (i, &s) in frame.iter().enumerate() {
            if start + i < audio.len() {
                audio[start + i] += s as i32;
            }
        }
    }
    let audio: Vec<i16> = audio
        .iter()
        .map(|v| (*v).clamp(-32_767, 32_767) as i16)
        .collect();

    let out = ft8::FtDecoder::decode_audio(&audio, false);
    assert!(
        out.iter().any(|l| l.contains("CQ W1AW FN31")),
        "strong signal missing: {out:?}"
    );
    assert!(
        out.iter().any(|l| l.contains("CQ JA1ABC PM95")),
        "weak signal masked by its neighbour: {out:?}"
    );
}

#[test]
fn ft4_decodes_a_slot() {
    let out = ft_roundtrip(true, "JA1ABC", "PM95", 1500.0);
    assert!(
        out.iter().any(|l| l.contains("CQ JA1ABC PM95")),
        "expected the message back, got {out:?}"
    );
}

/// The decoder must find signals across the passband, not just at centre.
#[test]
fn ft8_decodes_off_centre_signal() {
    let out = ft_roundtrip(false, "W1AW", "FN31", 700.0);
    assert!(
        out.iter().any(|l| l.contains("CQ W1AW FN31")),
        "expected an off-centre decode, got {out:?}"
    );
}

/// End-to-end: an FT8 signal placed on RF, taken through the real tuning chain
/// (NCO, decimating FIR) and the audio conversion, then decoded.
///
/// The earlier FT tests hand prepared audio straight to the decoder, so they
/// cannot catch a wrong mixing direction, a passband that excludes the signal,
/// or a level that quantises the signal away. This one can.
#[cfg(test)]
fn ft8_through_chain(radio_rate: f64, audio_hz: f64) -> Vec<String> {
    use crate::dsp::DecodeChain;
    use mfsk_core::ft8::wave_gen::{message_to_tones, tones_to_i16};
    use mfsk_core::msg::wsjt77::pack77;

    // Build one slot of 12 kHz USB audio containing the signal.
    let msg77 = pack77("CQ", "G0ABC", "IO91").expect("pack77");
    let tones = message_to_tones(&msg77);
    let frame = tones_to_i16(&tones, audio_hz as f32, 20_000);
    let nmax = 15 * 12_000;
    let mut audio = vec![0i16; nmax];
    let start = (0.5 * 12_000.0) as usize;
    for (i, &s) in frame.iter().enumerate() {
        if start + i < audio.len() {
            audio[start + i] = s;
        }
    }

    // Upsample to the radio rate as a real baseband signal: this is what the
    // receiver would see with the dial at 0 Hz offset in the span.
    let up = (radio_rate / 12_000.0).round() as usize;
    let mut iq = Vec::with_capacity(audio.len() * up);
    for w in audio.windows(2) {
        let (a, b) = (w[0] as f32, w[1] as f32);
        for k in 0..up {
            let t = k as f32 / up as f32;
            let v = (a + (b - a) * t) / 32768.0 * 0.05; // small, like real IQ
            iq.push(Complex32::new(v, 0.0));
        }
    }

    // Same chain the app builds for FT8, tuned with the dial under the cursor.
    let mut chain = DecodeChain::new(radio_rate, 3000.0, 12_000.0);
    let mut dec = ft8::FtDecoder::new(chain.fs_out(), false);
    chain.set_offset(dec.offset_shift());

    let mut bb = Vec::new();
    for block in iq.chunks(16384) {
        chain.process(block, &mut bb);
        dec.append_audio_for_test(&bb);
    }
    let buf = dec.audio_buffer().to_vec();
    assert!(
        buf.len() > nmax * 3 / 4,
        "chain produced {} samples, expected ~{nmax}",
        buf.len()
    );
    let peak = buf.iter().map(|v| v.abs() as i32).max().unwrap_or(0);
    assert!(
        peak > 500,
        "audio level far too low for decoding: peak {peak}"
    );
    ft8::FtDecoder::decode_audio(&buf, false)
}

#[test]
fn ft8_survives_the_tuning_chain() {
    let out = ft8_through_chain(192_000.0, 1200.0);
    assert!(
        out.iter().any(|l| l.contains("CQ G0ABC IO91")),
        "signal did not survive the chain, got {out:?}"
    );
}


// -------------------------------------------- simulated band (Stage 0)

use super::channel::{self, Condition};

/// One station on the simulated band.
#[derive(Clone, Copy)]
struct BandStation {
    text: &'static str,
    wpm: f32,
    /// Offset from the decoder's centre, Hz.
    offset_hz: f32,
    /// Level relative to the wanted station, dB.
    level_db: f32,
}

/// Build a stretch of band: a wanted station, its neighbours, an ionosphere,
/// a noise floor and some weather.
///
/// Each station gets its own channel realisation, because each is a different
/// path from a different place — they do not fade together, and a decoder
/// choosing between them meets that every time.
///
/// `snr_db` is the wanted station's *mean* SNR in the decoder's 400 Hz
/// bandwidth. Mean is the operative word once the channel is in circuit.
fn gen_band(
    wanted: &BandStation,
    others: &[BandStation],
    cond: Condition,
    snr_db: f32,
    crashes_per_sec: f32,
    seed: u32,
) -> Vec<Complex32> {
    let mut sig = channel::watterson(
        &gen_cw_at(wanted.text, wanted.wpm, 0.0, CW_TONE + wanted.offset_hz),
        FS as f32,
        cond,
        seed,
    );
    for (k, o) in others.iter().enumerate() {
        let other = channel::watterson(
            &gen_cw_at(o.text, o.wpm, 0.0, CW_TONE + o.offset_hz),
            FS as f32,
            cond,
            seed ^ (0x9e37_79b9u32.wrapping_mul(k as u32 + 1)),
        );
        let g = 10f32.powf(o.level_db / 20.0);
        for (i, s) in sig.iter_mut().enumerate() {
            if let Some(v) = other.get(i) {
                *s += v * g;
            }
        }
    }
    let mut rng = seed ^ 0xa5a5_a5a5;
    let n_scale = scale_for_snr(snr_db);
    for s in sig.iter_mut() {
        *s += Complex32::new(noise(&mut rng), noise(&mut rng)) * n_scale;
    }
    channel::static_crashes(
        &mut sig,
        FS as f32,
        crashes_per_sec,
        12.0,
        seed ^ 0x5eed_5eed,
    );
    sig
}

/// The band as it actually is, scored the way `cw_score_grid` scores the
/// laboratory.
///
/// This exists because the flat grid cannot see the decoder's largest real
/// failure. Across the live 20m capture a single station's mark level moves
/// by 12 to 31 dB inside one over, every passband holds two to four stations
/// rather than one, and which signals get copied tracks fade depth far more
/// closely than it tracks SNR. `bench_cw_fading` shows the mechanism on a
/// bare sinusoidal fade; this grid puts a proper Watterson channel, real
/// neighbours and static crashes behind it, and is the yardstick any redesign
/// of the detector should be judged on.
fn cw_band_score_grid() -> (Vec<(String, f32)>, f32) {
    const SEEDS: [u32; 4] = [0x1234_5678, 0x9e37_79b9, 0x0bad_f00d, 0x5eed_1234];
    const MSG: &str = "CQ CQ DE W1AW W1AW K UR RST 599 599 QTH NEWINGTON CT \
                       NAME JOE JOE HW CPY? BK TNX FER THE QSO 73 ES GL DE W1AW K";
    let wanted = BandStation {
        text: MSG,
        wpm: 20.0,
        offset_hz: 0.0,
        level_db: 0.0,
    };
    // Neighbours at the separations and speeds a crowded CW segment actually
    // has: inside the 400 Hz passband, keyed by other operators at other
    // speeds, one of them stronger than the wanted station.
    let qrm = [
        BandStation {
            text: "TEST DE G4XYZ G4XYZ 599 001 001 TU QRZ TEST G4XYZ K",
            wpm: 27.0,
            offset_hz: 160.0,
            level_db: 0.0,
        },
        BandStation {
            text: "CQ CQ DE VK2ABC VK2ABC PSE K",
            wpm: 16.0,
            offset_hz: -230.0,
            level_db: 3.0,
        },
        BandStation {
            text: "DE JA1XYZ UR 599 599 BK",
            wpm: 32.0,
            offset_hz: 90.0,
            level_db: -4.0,
        },
    ];

    // name, neighbours, channel, mean SNR, crashes/s
    let cases: Vec<(String, &[BandStation], Condition, f32, f32)> = vec![
        // 1. The ionosphere alone, at a level the flat grid copies perfectly.
        ("chan flat 12dB".into(), &[][..], channel::FLAT, 12.0, 0.0),
        ("chan good 12dB".into(), &[][..], channel::CCIR_GOOD, 12.0, 0.0),
        ("chan moderate 12dB".into(), &[][..], channel::CCIR_MODERATE, 12.0, 0.0),
        ("chan poor 12dB".into(), &[][..], channel::CCIR_POOR, 12.0, 0.0),
        ("chan flutter 12dB".into(), &[][..], channel::CCIR_FLUTTER, 12.0, 0.0),
        // 2. Level, through a middling path.
        ("moderate 20dB".into(), &[][..], channel::CCIR_MODERATE, 20.0, 0.0),
        ("moderate 6dB".into(), &[][..], channel::CCIR_MODERATE, 6.0, 0.0),
        // 3. Neighbours, fading independently of the wanted station.
        ("qrm x1 moderate".into(), &qrm[..1], channel::CCIR_MODERATE, 12.0, 0.0),
        ("qrm x2 moderate".into(), &qrm[..2], channel::CCIR_MODERATE, 12.0, 0.0),
        ("qrm x3 moderate".into(), &qrm[..], channel::CCIR_MODERATE, 12.0, 0.0),
        ("qrm x3 poor".into(), &qrm[..], channel::CCIR_POOR, 12.0, 0.0),
        // 4. Weather.
        ("crashes 2/s moderate".into(), &[][..], channel::CCIR_MODERATE, 12.0, 2.0),
        ("crashes 6/s moderate".into(), &[][..], channel::CCIR_MODERATE, 12.0, 6.0),
        // 5. Everything at once — a Saturday afternoon on 20m.
        ("the band, 10dB".into(), &qrm[..], channel::CCIR_POOR, 10.0, 2.0),
        ("the band, 4dB".into(), &qrm[..], channel::CCIR_POOR, 4.0, 2.0),
    ];

    let mut cells: Vec<(String, f32)> = Vec::new();
    for (name, others, cond, snr, crashes) in cases {
        let acc = SEEDS
            .iter()
            .map(|&seed| {
                let sig = gen_band(&wanted, others, cond, snr, crashes, seed);
                accuracy(MSG, &decode_cw(&sig, 0.0))
            })
            .sum::<f32>()
            / SEEDS.len() as f32;
        cells.push((name, acc));
    }

    // An occupied but wanted-station-free stretch: neighbours fading in and
    // out, nothing on frequency. Anything decoded here is invented, and a
    // detector rebuilt for fading has every opportunity to invent more.
    let empty = SEEDS
        .iter()
        .map(|&seed| {
            let quiet = BandStation {
                text: "",
                wpm: 20.0,
                offset_hz: 0.0,
                level_db: 0.0,
            };
            let sig = gen_band(&quiet, &qrm[..], channel::CCIR_POOR, 12.0, 2.0, seed);
            let letters = decode_cw(&sig, 0.0)
                .chars()
                .filter(|c| !c.is_whitespace())
                .count();
            1.0 - (letters as f32 / 10.0).min(1.0)
        })
        .sum::<f32>()
        / SEEDS.len() as f32;
    cells.push(("empty, qrm only".into(), empty));

    let mean = cells.iter().map(|(_, a)| *a).sum::<f32>() / cells.len() as f32;
    (cells, mean)
}

/// The band grid as a gate, like `cw_score_does_not_regress` but for the
/// conditions that actually break this decoder.
///
/// The bar is low because the decoder is currently bad at this — 43 % against
/// the flat grid's 90 — and that gap is the point of Stage 0 rather than
/// something to hide. The gate exists so the number cannot quietly get worse
/// while the flat grid is being tuned, which is exactly what has been
/// happening: the flat grid moved three points this session for changes that
/// this grid barely notices.
///
/// Raise it as the detector is rebuilt. The interesting milestones are
/// `chan moderate 12dB` reaching the high nineties, and `moderate 20dB`
/// pulling clear of `moderate 6dB` — today they are 49 % and 40 %, which says
/// fourteen decibels of extra signal buys almost nothing and the decoder is
/// limited by fading rather than by noise.
#[test]
fn cw_band_score_does_not_regress() {
    let (cells, mean) = cw_band_score_grid();
    const FLOOR: f32 = 0.38;
    if mean < FLOOR {
        let mut worst: Vec<&(String, f32)> = cells.iter().collect();
        worst.sort_by(|a, b| a.1.total_cmp(&b.1));
        let detail: Vec<String> = worst
            .iter()
            .take(6)
            .map(|(n, a)| format!("{n} {:.0}%", a * 100.0))
            .collect();
        panic!(
            "CW band score {:.2}% is below the {:.0}% gate; worst cells: {}",
            mean * 100.0,
            FLOOR * 100.0,
            detail.join(", ")
        );
    }
}

#[test]
#[ignore]
fn bench_cw_band() {
    let (cells, mean) = cw_band_score_grid();
    for (name, acc) in &cells {
        println!("  {name:<24} {:>5.1}%", acc * 100.0);
    }
    println!("\n  CW BAND SCORE {:.2}%  ({} cells)", mean * 100.0, cells.len());
}

// ------------------------------------------------- CW accuracy bench

/// Levenshtein distance, for scoring decoded text against what was sent.
fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let sub = prev[j - 1] + usize::from(a[i - 1] != b[j - 1]);
            cur[j] = sub.min(prev[j] + 1).min(cur[j - 1] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Fraction of characters recovered, 0..1.
fn accuracy(sent: &str, got: &str) -> f32 {
    if sent.is_empty() {
        return 1.0;
    }
    let d = edit_distance(sent, got);
    1.0 - (d as f32 / sent.chars().count() as f32).min(1.0)
}

/// `snr_scale` for a wanted SNR in dB, measured in the decoder's 400 Hz
/// bandwidth against a full-scale mark.
fn scale_for_snr(db: f32) -> f32 {
    // noise() is uniform(-0.5,0.5): variance 1/12 per component, so a
    // complex sample has variance 1/6 before scaling, spread over FS.
    let n_total = (1.0f32 / 6.0).sqrt();
    let in_bw = n_total * (400.0 / FS as f32).sqrt();
    let wanted = 10f32.powf(-db / 20.0);
    wanted / in_bw
}

/// Decode as the app does: through the same tuning chain, at the same
/// bandwidth the decoder asks for. Handing the decoder raw wideband audio
/// instead measures a receiver nobody has.
fn decode_cw(sig: &[Complex32], tone: f32) -> String {
    let mut d = cw::CwDecoder::new(FS);
    let mut chain = crate::dsp::DecodeChain::new(FS, d.bandwidth(), FS);
    chain.set_offset(tone as f64 + d.offset_shift());
    let mut audio = Vec::new();
    let mut out = String::new();
    for chunk in sig.chunks(4096) {
        chain.process(chunk, &mut audio);
        out.push_str(&d.process(&audio));
    }
    out.trim().to_string()
}

/// Same as `decode_cw` but reports what the decoder thought it was doing.
fn decode_cw_dbg(sig: &[Complex32], tone: f32) -> (String, f32, f32) {
    let mut d = cw::CwDecoder::new(FS);
    let mut chain = crate::dsp::DecodeChain::new(FS, d.bandwidth(), FS);
    chain.set_offset(tone as f64 + d.offset_shift());
    let mut audio = Vec::new();
    let mut out = String::new();
    for chunk in sig.chunks(4096) {
        chain.process(chunk, &mut audio);
        out.push_str(&d.process(&audio));
    }
    (out.trim().to_string(), d.wpm(), d.lock_hz())
}

#[test]
#[ignore]
fn bench_cw_detail() {
    let msg = "CQ CQ DE W1AW W1AW K";
    println!("\nsent: {msg:?}\n");
    for wpm in [12.0f32, 18.0, 25.0, 35.0] {
        for db in [40, 20, 10] {
            let sig = gen_cw(msg, wpm, scale_for_snr(db as f32));
            let (got, est, lock) = decode_cw_dbg(&sig, 0.0);
            println!(
                "{wpm:>3.0} WPM {db:>3} dB  acc {:>4}  est {est:>4.1} WPM  lock {lock:>+6.1} Hz  {got:?}",
                format!("{:.0}%", accuracy(msg, &got) * 100.0)
            );
        }
        println!();
    }
}

#[test]
#[ignore]
fn bench_cw_long() {
    // A realistic over: long enough that acquisition is a small fraction.
    let msg = "CQ CQ DE W1AW W1AW K UR RST 599 599 QTH NEWINGTON CT \
               NAME JOE JOE HW CPY? BK TNX FER THE QSO 73 ES GL DE W1AW K";
    for wpm in [15.0f32, 20.0, 28.0] {
        for db in [30, 15, 8] {
            let sig = gen_cw(msg, wpm, scale_for_snr(db as f32));
            let (got, est, _) = decode_cw_dbg(&sig, 0.0);
            // Accuracy over the whole over, and over everything after the
            // first 20 characters — acquisition versus steady state.
            let tail_sent: String = msg.chars().skip(20).collect();
            let tail_got: String = got.chars().skip(20).collect();
            println!(
                "{wpm:>3.0} WPM {db:>3} dB  all {:>4}  tail {:>4}  est {est:.1} WPM",
                format!("{:.0}%", accuracy(msg, &got) * 100.0),
                format!("{:.0}%", accuracy(&tail_sent, &tail_got) * 100.0),
            );
            if db == 15 {
                println!("        {got:?}");
            }
        }
    }
}

#[test]
#[ignore]
fn bench_cw_tuning_error() {
    let msg = "CQ CQ DE W1AW W1AW K UR RST 599 QTH NEWINGTON DE W1AW K";
    // The chain is tuned where the classifier said; the tone is off by
    // `err`, which is what auto mode actually hands the decoder.
    println!("chain tuned to 700 Hz, tone offset by err, 20 WPM");
    print!("{:>8}", "err Hz");
    for db in [30, 15, 8] {
        print!("{:>9}", format!("{db}dB"));
    }
    println!();
    for err in [0.0f32, 25.0, 50.0, 100.0, 150.0, 200.0, -100.0, -200.0] {
        print!("{err:>8.0}");
        for db in [30, 15, 8] {
            let sig = gen_cw_at(msg, 20.0, scale_for_snr(db as f32), 700.0 + err);
            let got = decode_cw(&sig, 700.0);
            print!("{:>9}", format!("{:.0}%", accuracy(msg, &got) * 100.0));
        }
        println!();
    }
    println!();
    for err in [50.0f32, 150.0, 200.0] {
        let sig = gen_cw_at(msg, 20.0, scale_for_snr(15.0), 700.0 + err);
        let (got, _, lock) = decode_cw_dbg(&sig, 700.0);
        println!("  err {err:>4.0} Hz  lock {lock:>+6.1} Hz  {got:?}");
    }
}

#[test]
#[ignore]
fn bench_cw_neighbour() {
    let msg = "CQ CQ DE W1AW W1AW K UR RST 599 QTH NEWINGTON DE W1AW K";
    let qrm = "TEST DE G4XYZ G4XYZ 599 001 001 TU QRZ TEST G4XYZ K";
    println!("wanted at 700 Hz, 20 WPM, 15 dB; a second CW station nearby");
    println!("{:>10}{:>8}{:>8}{:>8}", "sep Hz", "-6dB", "same", "+6dB");
    for sep in [500.0f32, 400.0, 300.0, 250.0, 200.0, 150.0, 100.0] {
        print!("{sep:>10.0}");
        for rel in [0.5f32, 1.0, 2.0] {
            let want = gen_cw_at(msg, 20.0, scale_for_snr(15.0), 700.0);
            // A second station, keyed differently, at 25 WPM.
            let other = gen_cw_at(qrm, 25.0, 0.0, 700.0 + sep);
            let mut sig = want.clone();
            for (i, s) in sig.iter_mut().enumerate() {
                if let Some(o) = other.get(i) {
                    *s += o * rel;
                }
            }
            let got = decode_cw(&sig, 700.0);
            print!("{:>8}", format!("{:.0}%", accuracy(msg, &got) * 100.0));
        }
        println!();
    }
    println!("\nwhich station did it pick? (scored against both)");
    for sep in [200.0f32, 150.0, 100.0] {
        for rel in [1.0f32, 2.0] {
            let want = gen_cw_at(msg, 20.0, scale_for_snr(15.0), 700.0);
            let other = gen_cw_at(qrm, 25.0, 0.0, 700.0 + sep);
            let mut sig = want.clone();
            for (i, s) in sig.iter_mut().enumerate() {
                if let Some(o) = other.get(i) {
                    *s += o * rel;
                }
            }
            let (got, _, lock) = decode_cw_dbg(&sig, 700.0);
            println!(
                "  sep {sep:>4.0} Hz  qrm {:>4}  lock {lock:>+6.1} Hz  wanted {:>4}  qrm {:>4}  {got:?}",
                if rel > 1.5 { "+6dB" } else { "same" },
                format!("{:.0}%", accuracy(msg, &got) * 100.0),
                format!("{:.0}%", accuracy(qrm, &got) * 100.0),
            );
        }
    }
}

/// The post-mix filter has to be sized for the fist, not for the fastest
/// fist imaginable.
///
/// A four-pole 150 Hz filter passes ~147 Hz of noise to the envelope
/// detector while a 20 WPM signal's keying occupies about 33 Hz. That is
/// most of a 6 dB gap, and it showed up exactly where it should: copy held
/// at 90% down to 3 dB and then collapsed to 15-20% at 0 dB, which is a
/// signal-to-noise wall rather than anything about Morse.
///
/// Narrowing onto the tracked clock (floored at `POST_MIX_MIN_HZ`, for the
/// spiral described there) is worth about 3 dB of copy threshold.
#[test]
fn cw_copies_at_zero_db() {
    let msg = "CQ CQ DE W1AW W1AW K";
    for wpm in [18.0f32, 25.0] {
        let sig = gen_cw(msg, wpm, scale_for_snr(0.0));
        let acc = accuracy(msg, &decode_cw(&sig, 0.0));
        assert!(
            acc >= 0.60,
            "{wpm:.0} WPM at 0 dB copied {:.0}%, was 15-20% before the filter matched the clock \
             and 65-80% before the slicer thresholds were corrected for the max-hold peak",
            acc * 100.0
        );
    }
    // ...without giving anything back where it already worked.
    for wpm in [12.0f32, 18.0, 25.0, 35.0] {
        for db in [15.0f32, 6.0] {
            let sig = gen_cw(msg, wpm, scale_for_snr(db));
            let acc = accuracy(msg, &decode_cw(&sig, 0.0));
            assert!(
                acc >= 0.85,
                "{wpm:.0} WPM at {db:.0} dB regressed to {:.0}%",
                acc * 100.0
            );
        }
    }
}

/// One number for the whole CW decoder, so a change can be called an
/// improvement or a regression rather than argued about cell by cell.
///
/// The grid is deliberately weighted toward where the decoder actually
/// lives: a long over rather than a five-word call, several speeds, and
/// SNRs from comfortable down past the copy threshold. Insertions count
/// against it as heavily as misses, because the failure that makes a pane
/// of copy worthless on a real band is noise spelling extra letters between
/// the real ones — see `cw_stays_quiet_on_noise`.
///
/// Prints per-cell accuracy and a mean. Compare the mean across a change;
/// the cells say where it came from.
fn cw_score_grid() -> (Vec<(String, f32)>, f32) {
    let short = "CQ CQ DE W1AW W1AW K";
    let long = "CQ CQ DE W1AW W1AW K UR RST 599 599 QTH NEWINGTON CT \
                NAME JOE JOE HW CPY? BK TNX FER THE QSO 73 ES GL DE W1AW K";
    let qrm = "TEST DE G4XYZ G4XYZ 599 001 001 TU QRZ TEST G4XYZ K";
    // Every cell is the mean over these noise realisations. Near the copy
    // threshold one realisation decides a whole character, so a single-trial
    // grid moves by tens of points between neighbouring settings and rewards
    // whichever constant happened to suit one burst of noise.
    const SEEDS: [u32; 4] = [0x1234_5678, 0x9e37_79b9, 0x0bad_f00d, 0x5eed_1234];
    let mut cells: Vec<(String, f32)> = Vec::new();
    fn over_seeds(f: impl Fn(u32) -> f32) -> f32 {
        SEEDS.iter().map(|&s| f(s)).sum::<f32>() / SEEDS.len() as f32
    }

    for &wpm in &[12.0f32, 18.0, 25.0, 35.0] {
        for &db in &[15.0f32, 6.0, 3.0, 0.0, -3.0] {
            let acc = over_seeds(|seed| {
                let sig = gen_cw_seed(short, wpm, scale_for_snr(db), CW_TONE, seed);
                accuracy(short, &decode_cw(&sig, 0.0))
            });
            cells.push((format!("short {wpm:.0}wpm {db:+.0}dB"), acc));
        }
    }
    for &wpm in &[15.0f32, 22.0, 28.0] {
        for &db in &[12.0f32, 6.0, 3.0] {
            let acc = over_seeds(|seed| {
                let sig = gen_cw_seed(long, wpm, scale_for_snr(db), CW_TONE, seed);
                accuracy(long, &decode_cw(&sig, 0.0))
            });
            cells.push((format!("long {wpm:.0}wpm {db:+.0}dB"), acc));
        }
    }
    // Residual tuning error the classifier leaves behind.
    for &err in &[-150.0f32, -50.0, 50.0, 150.0] {
        let acc = over_seeds(|seed| {
            let sig = gen_cw_seed(long, 20.0, scale_for_snr(10.0), 700.0 + err, seed);
            accuracy(long, &decode_cw(&sig, 700.0))
        });
        cells.push((format!("tune {err:+.0}Hz"), acc));
    }
    // A second station inside the 400 Hz passband, keyed at another speed.
    for &sep in &[300.0f32, 200.0, -200.0] {
        let acc = over_seeds(|seed| {
            let mut sig = gen_cw_seed(long, 20.0, scale_for_snr(10.0), CW_TONE, seed);
            let other = gen_cw_at(qrm, 27.0, 0.0, CW_TONE + sep);
            for (i, s) in sig.iter_mut().enumerate() {
                if let Some(o) = other.get(i) {
                    *s += o;
                }
            }
            accuracy(long, &decode_cw(&sig, 0.0))
        });
        cells.push((format!("qrm {sep:+.0}Hz"), acc));
    }
    // A station that changes speed mid-over, which is where the post-mix
    // filter can trap itself: too narrow for the fist actually being sent
    // smears the keying, which reads as a *slower* clock, which asks for a
    // narrower filter still. Nothing else in this grid can see that — every
    // other cell is a single speed held throughout — so without this cell the
    // filter's safety floor looks like free score.
    for &(from, to) in &[(15.0f32, 32.0f32), (30.0, 14.0)] {
        const FIRST: &str = "CQ CQ DE W1AW K ";
        const SECOND: &str = "UR RST 599 QTH NEWINGTON DE W1AW K";
        let acc = over_seeds(|seed| {
            let mut sig = gen_cw_seed(FIRST, from, scale_for_snr(12.0), CW_TONE, seed);
            sig.extend(gen_cw_seed(
                SECOND,
                to,
                scale_for_snr(12.0),
                CW_TONE,
                seed ^ 0xffff,
            ));
            // Scored over the whole over: if the clock traps itself on the
            // change, everything after it is wreckage and the score says so.
            accuracy(&format!("{FIRST}{SECOND}"), &decode_cw(&sig, 0.0))
        });
        cells.push((format!("speed {from:.0}->{to:.0}wpm"), acc));
    }
    // Empty frequency: every character emitted is an error, and the score has
    // to feel that or a change can buy copy with insertions and look good.
    for &seed in &[0x1234_5678u32, 0xdead_beef, 0x0bad_f00d] {
        let mut rng = seed;
        let n = (12.0 * FS) as usize;
        let sig: Vec<Complex32> = (0..n)
            .map(|_| Complex32::new(noise(&mut rng), noise(&mut rng)) * 0.3)
            .collect();
        let letters = decode_cw(&sig, 0.0)
            .chars()
            .filter(|c| !c.is_whitespace())
            .count();
        // Ten spurious characters in twelve seconds scores zero.
        cells.push((
            format!("noise {seed:08x}"),
            1.0 - (letters as f32 / 10.0).min(1.0),
        ));
    }


    let mean = cells.iter().map(|(_, a)| *a).sum::<f32>() / cells.len() as f32;
    (cells, mean)
}

#[test]
#[ignore]
fn bench_cw_score() {
    let (cells, mean) = cw_score_grid();
    for (name, acc) in &cells {
        println!("  {name:<22} {:>5.1}%", acc * 100.0);
    }
    println!("\n  CW SCORE {:.2}%  ({} cells)", mean * 100.0, cells.len());
}

/// The score is a regression gate, not just something to read.
///
/// Every other CW test here asserts on one scenario, so a change that trades
/// weak-signal copy for clean-signal copy — or buys either by emitting more
/// characters — passes all of them while making the decoder worse. This is
/// the one that notices, because it is the only test that weighs the whole
/// grid, insertions on an empty frequency included.
///
/// The bar sits below the measured figure by about the amount the grid moves
/// between neighbouring settings of a single constant. It is meant to catch a
/// change that costs a point or more, not to pin the exact number: raise it
/// when the score improves, and treat needing to lower it as the finding.
#[test]
fn cw_score_does_not_regress() {
    let (cells, mean) = cw_score_grid();
    const FLOOR: f32 = 0.88;
    if mean < FLOOR {
        let mut worst: Vec<&(String, f32)> = cells.iter().collect();
        worst.sort_by(|a, b| a.1.total_cmp(&b.1));
        let detail: Vec<String> = worst
            .iter()
            .take(6)
            .map(|(n, a)| format!("{n} {:.0}%", a * 100.0))
            .collect();
        panic!(
            "CW score {:.2}% is below the {:.0}% gate; worst cells: {}",
            mean * 100.0,
            FLOOR * 100.0,
            detail.join(", ")
        );
    }
}

#[test]
#[ignore]
fn bench_cw_accuracy() {
    let msg = "CQ CQ DE W1AW W1AW K";
    println!("\nsent: {msg:?}\n");

    println!("== on frequency, by speed and SNR ==");
    print!("{:>6}", "WPM");
    for db in [20, 15, 10, 6, 3, 0, -3] {
        print!("{:>8}", format!("{db}dB"));
    }
    println!();
    for wpm in [12.0f32, 18.0, 25.0, 35.0] {
        print!("{wpm:>6.0}");
        for db in [20, 15, 10, 6, 3, 0, -3] {
            let sig = gen_cw(msg, wpm, scale_for_snr(db as f32));
            let got = decode_cw(&sig, 0.0);
            print!("{:>8}", format!("{:.0}%", accuracy(msg, &got) * 100.0));
        }
        println!();
    }

    println!("\n== 20 WPM, 15 dB, by residual tuning error ==");
    for tone in [-200.0f32, -100.0, -50.0, 0.0, 50.0, 100.0, 200.0, 300.0] {
        let sig = gen_cw_at(msg, 20.0, scale_for_snr(15.0), tone);
        let got = decode_cw(&sig, 0.0);
        println!(
            "  {tone:>6.0} Hz  {:>4}  {got:?}",
            format!("{:.0}%", accuracy(msg, &got) * 100.0)
        );
    }

    println!("\n== what it actually produces, 20 WPM on frequency ==");
    for db in [40, 20, 15, 10, 6, 3] {
        let sig = gen_cw(msg, 20.0, scale_for_snr(db as f32));
        println!("  {db:>2} dB  {:?}", decode_cw(&sig, 0.0));
    }
}

// --------------------------------------------- CW regression tests

/// The one that matters most on a real band.
///
/// The tuning chain hands the decoder a 400 Hz passband, and CW operators
/// routinely sit 100-300 Hz apart inside it. Without a post-mix channel
/// filter the envelope detector sums both stations and the slicer is keyed
/// by whichever one happens to be transmitting, so the copy is noise — which
/// is what a listener sees on any busy band, and what no clean single-signal
/// test can catch.
#[test]
fn cw_rejects_an_adjacent_station() {
    let msg = "CQ CQ DE W1AW W1AW K UR RST 599 QTH NEWINGTON DE W1AW K";
    let qrm = "TEST DE G4XYZ G4XYZ 599 001 001 TU QRZ TEST G4XYZ K";
    for sep in [250.0f32, 200.0, -200.0] {
        // Equal strength, keyed at a different speed so the interference is
        // uncorrelated with the wanted station's timing.
        let want = gen_cw_at(msg, 20.0, scale_for_snr(15.0), CW_TONE);
        let other = gen_cw_at(qrm, 27.0, 0.0, CW_TONE + sep);
        let mut sig = want;
        for (i, s) in sig.iter_mut().enumerate() {
            if let Some(o) = other.get(i) {
                *s += o;
            }
        }
        let got = decode_cw(&sig, 0.0);
        let acc = accuracy(msg, &got);
        assert!(
            acc > 0.9,
            "a station {sep:.0} Hz away wrecked the copy ({:.0}%): {got:?}",
            acc * 100.0
        );
    }
}

/// Copy must start with the first character actually sent.
///
/// Two things used to corrupt it: the filter chain fills from zero, and
/// slicing that ramp keys a phantom mark before the station has said
/// anything; and the dit estimate starts at 20 WPM, so the first characters
/// of an operator sending anything else are read against the wrong ruler.
#[test]
fn cw_starts_clean_at_any_speed() {
    for wpm in [12.0f32, 18.0, 25.0, 35.0] {
        let sig = gen_cw("CQ CQ DE W1AW W1AW K", wpm, scale_for_snr(20.0));
        let got = decode_cw(&sig, 0.0);
        assert!(
            got.starts_with("CQ"),
            "at {wpm:.0} WPM the copy did not start with what was sent: {got:?}"
        );
    }
}

fn cw_envelope_filter_gain_db() -> f32 {
    let mut rng = 0x71ac_2049u32;
    let dit = 60usize;
    let n = 6000;
    let raw: Vec<f32> = (0..n)
        .map(|i| {
            let mark = (i / dit) % 4 < 2;
            (if mark { 0.18 } else { 0.0 }) + noise(&mut rng) * 0.45
        })
        .collect();
    let window = (0.35 * dit as f32) as usize;
    let mut sum = 0.0;
    let mut filtered = Vec::with_capacity(n);
    for i in 0..n {
        sum += raw[i];
        if i >= window {
            sum -= raw[i - window];
        }
        filtered.push(sum / (i + 1).min(window) as f32);
    }
    let snr = |x: &[f32]| {
        let mut on = Vec::new();
        let mut off = Vec::new();
        for i in window..n {
            let pos = i % dit;
            if pos < window || pos + window >= dit {
                continue;
            }
            if (i / dit) % 4 < 2 {
                on.push(x[i]);
            } else {
                off.push(x[i]);
            }
        }
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let (mo, mf) = (mean(&on), mean(&off));
        let var = on
            .iter()
            .map(|v| (v - mo).powi(2))
            .chain(off.iter().map(|v| (v - mf).powi(2)))
            .sum::<f32>()
            / (on.len() + off.len()) as f32;
        (mo - mf).abs() / var.sqrt()
    };
    20.0 * (snr(&filtered) / snr(&raw)).log10()
}

#[test]
fn cw_matched_envelope_gains_two_db() {
    let gain = cw_envelope_filter_gain_db();
    assert!(gain >= 2.0, "matched envelope gained only {gain:.1} dB");
}

#[test]
#[ignore]
fn bench_cw_matched_envelope() {
    println!(
        "CW clock-matched envelope separation gain: {:.1} dB",
        cw_envelope_filter_gain_db()
    );
}

/// Steady-state copy on a signal weak enough to be worth decoding by
/// machine. 8 dB in 400 Hz is a perfectly ordinary HF signal.
#[test]
fn cw_copies_a_long_over_at_low_snr() {
    let msg = "CQ CQ DE W1AW W1AW K UR RST 599 599 QTH NEWINGTON CT \
               NAME JOE JOE HW CPY? BK TNX FER THE QSO 73 ES GL DE W1AW K";
    for wpm in [15.0f32, 20.0, 28.0] {
        let sig = gen_cw(msg, wpm, scale_for_snr(8.0));
        let got = decode_cw(&sig, 0.0);
        let acc = accuracy(msg, &got);
        assert!(
            acc > 0.9,
            "{wpm:.0} WPM at 8 dB copied {:.0}%: {got:?}",
            acc * 100.0
        );
    }
}

/// Multi-minute continuous stream: decoding accuracy must remain rock-solid
/// over extended text without threshold collapse or drifting deaf.
#[test]
fn cw_decodes_continuous_long_stream_without_degrading() {
    let passage = "CQ CQ DE W1AW W1AW K UR RST 599 599 QTH NEWINGTON CT \
                   NAME JOE JOE HW CPY? BK TNX FER THE QSO 73 ES GL DE W1AW K ";
    let full_text = passage.repeat(4);
    let sig = gen_cw(&full_text, 22.0, scale_for_snr(12.0));
    let got = decode_cw(&sig, 0.0);
    let acc = accuracy(&full_text, &got);
    assert!(
        acc > 0.92,
        "Continuous long stream accuracy degraded to {:.1}%: {got:?}",
        acc * 100.0
    );
}

/// The classifier places an auto slot from a spectrum peak, so the carrier
/// arrives with some tuning error. The decoder's own search must pull it in
/// across the range that error can plausibly cover.
#[test]
fn cw_pulls_in_a_mistuned_carrier() {
    let msg = "CQ CQ DE W1AW W1AW K UR RST 599 QTH NEWINGTON DE W1AW K";
    for err in [-100.0f32, -50.0, 0.0, 50.0, 100.0] {
        let sig = gen_cw_at(msg, 20.0, scale_for_snr(15.0), err);
        let got = decode_cw(&sig, 0.0);
        let acc = accuracy(msg, &got);
        assert!(
            acc > 0.9,
            "{err:+.0} Hz of tuning error copied {:.0}%: {got:?}",
            acc * 100.0
        );
    }
}

/// An empty frequency must stay empty.
///
/// Auto mode points a decoder at whatever the classifier flagged, and the
/// squelch deliberately does not gate CW (the passband scout has to keep
/// listening). So the decoder sees noise routinely, and noise sliced by a
/// threshold that has settled into it spells plausible-looking letters —
/// the fastest way to make a whole pane of copy worthless.
#[test]
fn cw_stays_quiet_on_noise() {
    for seed in [0x1234_5678u32, 0xdead_beef, 0x0bad_f00d] {
        let mut rng = seed;
        let n = (12.0 * FS) as usize;
        let sig: Vec<Complex32> = (0..n)
            .map(|_| Complex32::new(noise(&mut rng), noise(&mut rng)) * 0.3)
            .collect();
        let got = decode_cw(&sig, 0.0);
        let letters = got.chars().filter(|c| !c.is_whitespace()).count();
        assert!(
            letters <= 2,
            "12 s of noise produced {letters} characters: {got:?}"
        );
    }
}

/// A carrier — someone tuning up, or a birdie — is not Morse either.
#[test]
fn cw_stays_quiet_on_a_steady_carrier() {
    let n = (10.0 * FS) as usize;
    let mut rng = 0x5eed_1234u32;
    let sig: Vec<Complex32> = (0..n)
        .map(|i| {
            let ph = 2.0 * PI * CW_TONE * i as f32 / FS as f32;
            Complex32::from_polar(1.0, ph) + Complex32::new(noise(&mut rng), noise(&mut rng)) * 0.02
        })
        .collect();
    let got = decode_cw(&sig, 0.0);
    let letters = got.chars().filter(|c| !c.is_whitespace()).count();
    assert!(letters <= 2, "a steady carrier decoded as {got:?}");
}

// ---------------------------------------------- PSK31 accuracy bench

/// `snr_scale` for a wanted SNR in dB, measured in PSK31's own ~31 Hz
/// bandwidth against a full-scale symbol — the figure a waterfall shows.
fn psk_scale_for_snr(db: f32) -> f32 {
    let n_total = (1.0f32 / 6.0).sqrt();
    let in_bw = n_total * (31.25 / FS as f32).sqrt();
    let wanted = 10f32.powf(-db / 20.0);
    wanted / in_bw
}

/// Decode as the app does: through the tuning chain, at the bandwidth the
/// decoder asks for, with the carrier arriving near baseband.
fn decode_psk(sig: &[Complex32], tune: f32) -> String {
    let mut d = psk31::Psk31Decoder::new(FS);
    let mut chain = crate::dsp::DecodeChain::new(FS, d.bandwidth(), FS);
    chain.set_offset(tune as f64 + d.offset_shift());
    let mut audio = Vec::new();
    let mut out = String::new();
    for chunk in sig.chunks(4096) {
        chain.process(chunk, &mut audio);
        out.push_str(&d.process(&audio));
    }
    out.trim().to_string()
}

// -------------------------------------------- PSK31 through a real channel

/// PSK31 through a Watterson channel: fade the signal, *then* add the noise.
fn gen_psk31_channel(text: &str, cond: Condition, snr_db: f32, seed: u32) -> Vec<Complex32> {
    let mut sig = channel::watterson(&gen_psk31_clean(text, 0.0), FS as f32, cond, seed);
    let mut rng = seed ^ 0xa5a5_a5a5;
    let n_scale = psk_scale_for_snr(snr_db);
    for s in sig.iter_mut() {
        *s += Complex32::new(noise(&mut rng), noise(&mut rng)) * n_scale;
    }
    sig
}

const PSK_FADE_MSG: &str = "CQ CQ DE W1AW W1AW PSE K UR RST 599 QTH NEWINGTON";

/// Mean copy across `RTTY_FADE_SEEDS` for one channel and SNR.
fn psk31_fade_cell(cond: Condition, snr_db: f32) -> f32 {
    let n = RTTY_FADE_SEEDS.len() as f32;
    RTTY_FADE_SEEDS
        .iter()
        .map(|&seed| {
            let sig = gen_psk31_channel(PSK_FADE_MSG, cond, snr_db, seed);
            accuracy(PSK_FADE_MSG, &decode_psk(&sig, 0.0))
        })
        .sum::<f32>()
        / n
}

/// Copy against path and SNR — the question amplitude-immunity does not answer.
///
/// Normalising each symbol dump to unit magnitude (`psk31.rs`) genuinely does
/// make the demodulator indifferent to how deep a fade goes. It says nothing
/// about how fast the fade moves: PSK31 carries its data in the phase, and a
/// Doppler spread rotates exactly that. `CCIR_FLUTTER`'s 5 Hz against a 31.25
/// baud symbol rate is the case worth watching.
#[test]
#[ignore]
fn bench_psk31_fading() {
    println!("\n  PSK31 copy vs path and SNR (in 31.25 Hz), mean of 8 seeds\n");
    print!("{:>16}", "path");
    for db in [20, 15, 10, 8, 6] {
        print!("{:>8}", format!("{db}dB"));
    }
    println!();
    for (name, cond) in [
        ("flat", channel::FLAT),
        ("good", channel::CCIR_GOOD),
        ("moderate", channel::CCIR_MODERATE),
        ("poor", channel::CCIR_POOR),
        ("flutter", channel::CCIR_FLUTTER),
    ] {
        print!("{name:>16}");
        for db in [20.0f32, 15.0, 10.0, 8.0, 6.0] {
            print!("{:>8}", format!("{:.0}%", psk31_fade_cell(cond, db) * 100.0));
        }
        println!();
    }
}

/// The paths PSK31 can work, gated at a level where noise is not the limit.
///
/// `CCIR_FLUTTER` is deliberately not here. It copies 10 % at 20 dB and 10 %
/// at 15 dB — a floor that does not move with SNR, which normally means a
/// candidate is being vetoed rather than demodulated badly. It is not: the
/// decoder locks, and on two seeds in three it locks within a hertz and still
/// returns garbage. That is the mode meeting its own limit rather than a
/// defect. A 5 Hz Doppler spread decorrelates the phase in roughly
/// `1/(2*pi*5)` = 32 ms, which is one 31.25 baud symbol, so the channel is
/// incoherent across the very interval the demodulator has to integrate over.
/// Nothing in a tracking loop recovers that, and PSK31's reputation on
/// auroral paths says the same thing. Do not spend effort on it — the one
/// genuine wart is the AFC walking 20 Hz off on one seed in three while
/// chasing the Doppler, and fixing that would not buy a character.
#[test]
fn psk31_fading_copy_does_not_regress() {
    for (name, cond, floor) in [
        ("good", channel::CCIR_GOOD, 0.88f32),
        ("moderate", channel::CCIR_MODERATE, 0.78),
        ("poor", channel::CCIR_POOR, 0.68),
    ] {
        let got = psk31_fade_cell(cond, 20.0);
        assert!(
            got >= floor,
            "PSK31 {name} at 20 dB copied {:.0}%, below the {:.0}% gate",
            got * 100.0,
            floor * 100.0
        );
    }
}

fn decode_psk_dbg(sig: &[Complex32], tune: f32) -> (String, f32, bool) {
    let mut d = psk31::Psk31Decoder::new(FS);
    let mut chain = crate::dsp::DecodeChain::new(FS, d.bandwidth(), FS);
    chain.set_offset(tune as f64 + d.offset_shift());
    let mut audio = Vec::new();
    let mut out = String::new();
    for chunk in sig.chunks(4096) {
        chain.process(chunk, &mut audio);
        out.push_str(&d.process(&audio));
    }
    (out.trim().to_string(), d.lock_hz(), d.locked())
}

/// PSK31 has to degrade, not stop.
///
/// Copy used to fall from 88% at 8 dB to *nothing* at 6 dB, and the reason
/// was neither the demodulator nor the search: both were working. The
/// candidate was being vetoed by `keys_at_other_baud`, which asks whether
/// some other keying rate dominates the envelope spectrum — and on a signal
/// too weak to raise its own 31.25 Hz line, the "other rate" was whichever
/// noise bin happened to come top of 111 samples. The confirmation's own
/// noise rejection then had to be carried by `DC_FOCUS_MIN` instead, which
/// had been sitting close enough to noise to be no barrier at all.
///
/// 6 dB in 31 Hz is an ordinary weak PSK31 signal, well inside what the mode
/// exists to work.
#[test]
fn psk31_still_copies_at_six_db() {
    let msg = "CQ CQ DE W1AW W1AW PSE K";
    let sig = gen_psk31_snr(msg, 0.0, psk_scale_for_snr(6.0));
    let got = decode_psk(&sig, 0.0);
    let acc = accuracy(msg, &got);
    assert!(
        acc >= 0.30,
        "6 dB PSK31 copied {:.0}% of the message: {got:?}",
        acc * 100.0
    );
    // ...and the strong cases must not have been traded away for it.
    for db in [10.0f32, 15.0] {
        let sig = gen_psk31_snr(msg, 0.0, psk_scale_for_snr(db));
        let acc = accuracy(msg, &decode_psk(&sig, 0.0));
        assert!(
            acc >= 0.85,
            "{db:.0} dB PSK31 regressed to {:.0}%",
            acc * 100.0
        );
    }
}

#[test]
#[ignore]
fn bench_psk31_accuracy() {
    let msg = "CQ CQ DE W1AW W1AW PSE K";
    println!("\nsent: {msg:?}");

    println!("\n== on frequency, by SNR (in 31 Hz) ==");
    for db in [40, 30, 20, 15, 12, 10, 8, 6, 4, 2, 0] {
        let sig = gen_psk31_snr(msg, 0.0, psk_scale_for_snr(db as f32));
        let (got, lock, locked) = decode_psk_dbg(&sig, 0.0);
        println!(
            "  {db:>2} dB  {:>4}  lock {lock:>+6.1} Hz {}  {got:?}",
            format!("{:.0}%", accuracy(msg, &got) * 100.0),
            if locked { "yes" } else { "NO " },
        );
    }

    println!("\n== 15 dB, by residual tuning error ==");
    for err in [
        -150.0f32, -100.0, -50.0, -20.0, 0.0, 20.0, 50.0, 100.0, 150.0,
    ] {
        let sig = gen_psk31_snr(msg, err, psk_scale_for_snr(15.0));
        let (got, lock, locked) = decode_psk_dbg(&sig, 0.0);
        println!(
            "  {err:>+7.0} Hz  {:>4}  lock {lock:>+6.1} Hz {}  {got:?}",
            format!("{:.0}%", accuracy(msg, &got) * 100.0),
            if locked { "yes" } else { "NO " },
        );
    }
}

#[test]
#[ignore]
fn debug_psk31_gates() {
    let msg = "CQ CQ DE W1AW W1AW PSE K";
    for db in [20, 12, 10, 8, 6, 4, 2, 0] {
        eprintln!("===== {db} dB =====");
        let sig = gen_psk31_snr(msg, 0.0, psk_scale_for_snr(db as f32));
        let mut d = psk31::Psk31Decoder::new(FS);
        let mut chain = crate::dsp::DecodeChain::new(FS, d.bandwidth(), FS);
        chain.set_offset(0.0 + d.offset_shift());
        let mut audio = Vec::new();
        let mut out = String::new();
        for chunk in sig.chunks(4096) {
            chain.process(chunk, &mut audio);
            out.push_str(&d.process(&audio));
        }
        eprintln!(
            "{db:>3} dB  conf {:.3}  locked {}  hits {:?}  => {:?}",
            d.confidence().unwrap_or(-1.0),
            d.locked(),
            d.candidate_hz(),
            out.trim()
        );
    }
}

/// PSK31 at an arbitrary sample rate, for span-level tests.
pub(crate) fn gen_psk31_at(
    text: &str,
    fs: f64,
    offset_hz: f64,
    snr_scale: f32,
    secs: f64,
) -> Vec<Complex32> {
    let sps = (fs as f32 / BAUD_TEST) as usize;
    let mut bits: Vec<bool> = vec![false; 64];
    loop {
        for c in text.chars() {
            for ch in VARICODE[c as usize].chars() {
                bits.push(ch == '1');
            }
            bits.push(false);
            bits.push(false);
        }
        if bits.len() * sps > (fs * secs) as usize {
            break;
        }
    }
    let mut syms: Vec<f32> = vec![1.0];
    let mut cur = 1.0f32;
    for b in &bits {
        if !*b {
            cur = -cur;
        }
        syms.push(cur);
    }
    let total = (syms.len() + 2) * sps;
    let mut base = vec![0.0f32; total];
    for (k, &a) in syms.iter().enumerate() {
        let centre = (k + 1) * sps;
        for n in 0..2 * sps {
            let idx = centre + n - sps;
            if idx >= total {
                continue;
            }
            let x = (n as f32 - sps as f32) / sps as f32;
            base[idx] += a * 0.5 * (1.0 + (PI * x).cos());
        }
    }
    let mut rng = 0xfeed_1234u32;
    base.iter()
        .enumerate()
        .map(|(i, &v)| {
            let ph = 2.0 * PI * (offset_hz / fs) as f32 * i as f32;
            Complex32::from_polar(v.abs(), ph + if v < 0.0 { PI } else { 0.0 })
                + Complex32::new(noise(&mut rng), noise(&mut rng)) * snr_scale
        })
        .collect()
}

pub(crate) const BAUD_TEST: f32 = 31.25;

#[test]
#[ignore]
fn bench_psk31_span_classify() {
    // The span rate the app runs at, and the SNR figures a waterfall shows.
    let fs = 192_000.0f64;
    println!("\ndoes the span classifier flag PSK31 at all?");
    for db in [30, 20, 15, 10, 6] {
        // scan_span decimates to 8 kHz, so noise must be scaled for the
        // 31 Hz signal bandwidth against the full span.
        let n_total = (1.0f32 / 6.0).sqrt();
        let in_bw = n_total * (31.25 / fs as f32).sqrt();
        let scale = 10f32.powf(-(db as f32) / 20.0) / in_bw;
        let iq = gen_psk31_at("CQ CQ DE W1AW K ", fs, 12_000.0, scale, 1.5);
        let hits = psk31::scan_span(&iq, fs, &[(12_000.0, 20.0)]);
        println!(
            "  {db:>2} dB  scan_span -> {} hit(s) {:?}",
            hits.len(),
            hits.iter()
                .map(|h| format!("{:+.1} Hz q={:.2}", h.offset_hz - 12_000.0, h.quality))
                .collect::<Vec<_>>()
        );
    }
}

// --------------------------------------------- PSK31 regression tests

/// PSK31 exists to work weak signals — around 10 dB in its own 31 Hz is an
/// ordinary one, not an edge case. `dc_focus` used to reject those: it hard
/// -limited unfiltered wideband audio and compared the result against a fixed
/// fraction, so it measured signal-to-noise rather than the concentration it
/// was written to measure, and a perfectly good carrier scored 0.52 against a
/// 0.70 bar.
#[test]
fn psk31_copies_a_weak_signal() {
    let msg = "CQ CQ DE W1AW W1AW PSE K";
    for db in [20.0f32, 15.0, 10.0] {
        let sig = gen_psk31_snr(msg, 0.0, psk_scale_for_snr(db));
        let got = decode_psk(&sig, 0.0);
        let acc = accuracy(msg, &got);
        assert!(acc > 0.85, "{db:.0} dB copied {:.0}%: {got:?}", acc * 100.0);
    }
}

/// The residual tuning error an auto slot arrives with must not matter.
#[test]
fn psk31_pulls_in_a_mistuned_carrier() {
    let msg = "CQ CQ DE W1AW W1AW PSE K";
    for err in [-150.0f32, -50.0, 0.0, 50.0, 150.0] {
        let sig = gen_psk31_snr(msg, err, psk_scale_for_snr(15.0));
        let got = decode_psk(&sig, 0.0);
        let acc = accuracy(msg, &got);
        assert!(
            acc > 0.85,
            "{err:+.0} Hz of tuning error copied {:.0}%: {got:?}",
            acc * 100.0
        );
    }
}

/// Everything that made weak PSK31 visible lowered a threshold, so the
/// confirmation still has to reject a carrier that is not PSK31 at all.
#[test]
fn psk31_still_rejects_a_plain_carrier() {
    let n = (12.0 * FS) as usize;
    let mut rng = 0x33aa_55ffu32;
    let sig: Vec<Complex32> = (0..n)
        .map(|i| {
            let ph = 2.0 * PI * 40.0 * i as f32 / FS as f32;
            Complex32::from_polar(1.0, ph) + Complex32::new(noise(&mut rng), noise(&mut rng)) * 0.02
        })
        .collect();
    let hits = psk31::scan_span(&sig, FS, &[(0.0, 20.0)]);
    assert!(
        hits.is_empty(),
        "a steady carrier confirmed as PSK31: {hits:?}"
    );
}

// ------------------------------------------------------- AGC benches

/// Decode with the app's software AGC in the path, as auto mode runs it:
/// one scalar gain per audio block, applied after the tuning chain.
fn decode_with_agc(sig: &[Complex32], tune: f32, mut d: Box<dyn Decoder>, agc: bool) -> String {
    let bw = d.bandwidth();
    let shift = d.offset_shift();
    let mut chain = crate::dsp::DecodeChain::new(FS, bw, FS);
    chain.set_offset(tune as f64 + shift);
    let mut soft = crate::dsp::SoftAgc::new(chain.fs_out());
    let mut audio = Vec::new();
    let mut out = String::new();
    // ~85 ms of audio per block, which is what the radio delivers.
    let block = (FS * 0.085) as usize;
    for chunk in sig.chunks(block) {
        chain.process(chunk, &mut audio);
        if agc && d.wants_agc() {
            soft.process(&mut audio);
        }
        out.push_str(&d.process(&audio));
    }
    out.trim().to_string()
}

#[test]
#[ignore]
fn bench_agc_cost() {
    let cw_msg = "CQ CQ DE W1AW W1AW K UR RST 599 QTH NEWINGTON DE W1AW K";
    println!("\n== CW: accuracy with the software AGC on vs off ==");
    println!("{:>6}{:>8}{:>8}{:>8}", "WPM", "SNR", "AGC on", "AGC off");
    for wpm in [15.0f32, 20.0, 28.0] {
        for db in [30.0f32, 15.0, 8.0] {
            let sig = gen_cw(cw_msg, wpm, scale_for_snr(db));
            let on = decode_with_agc(&sig, 0.0, Box::new(cw::CwDecoder::new(FS)), true);
            let off = decode_with_agc(&sig, 0.0, Box::new(cw::CwDecoder::new(FS)), false);
            println!(
                "{wpm:>6.0}{:>8}{:>8}{:>8}",
                format!("{db:.0}dB"),
                format!("{:.0}%", accuracy(cw_msg, &on) * 100.0),
                format!("{:.0}%", accuracy(cw_msg, &off) * 100.0),
            );
        }
    }

    let rtty_msg = "RYRY CQ DE W1AW W1AW K";
    println!("\n== RTTY: accuracy with the software AGC on vs off ==");
    println!("{:>8}{:>8}{:>8}", "noise", "AGC on", "AGC off");
    for sc in [0.03f32, 0.15, 0.30] {
        let sig = gen_rtty_snr(rtty_msg, 45.45, 170.0, sc);
        let on = decode_with_agc(&sig, 0.0, Box::new(rtty::RttyDecoder::new(FS)), true);
        let off = decode_with_agc(&sig, 0.0, Box::new(rtty::RttyDecoder::new(FS)), false);
        println!(
            "{:>8}{:>8}{:>8}",
            format!("{sc:.2}"),
            format!("{:.0}%", accuracy(rtty_msg, &on) * 100.0),
            format!("{:.0}%", accuracy(rtty_msg, &off) * 100.0),
        );
    }

    let psk_msg = "CQ CQ DE W1AW W1AW PSE K";
    println!("\n== PSK31: accuracy with the software AGC on vs off ==");
    println!("{:>8}{:>8}{:>8}", "SNR", "AGC on", "AGC off");
    for db in [30.0f32, 20.0, 15.0, 10.0] {
        let sig = gen_psk31_snr(psk_msg, 0.0, psk_scale_for_snr(db));
        let on = decode_with_agc(&sig, 0.0, Box::new(psk31::Psk31Decoder::new(FS)), true);
        let off = decode_with_agc(&sig, 0.0, Box::new(psk31::Psk31Decoder::new(FS)), false);
        println!(
            "{:>8}{:>8}{:>8}",
            format!("{db:.0}dB"),
            format!("{:.0}%", accuracy(psk_msg, &on) * 100.0),
            format!("{:.0}%", accuracy(psk_msg, &off) * 100.0),
        );
    }
}

// ---------------------------------------------- RTTY polarity tests

fn decode_rtty(sig: &[Complex32]) -> String {
    let mut d = rtty::RttyDecoder::new(FS);
    let mut out = String::new();
    for chunk in sig.chunks(4096) {
        out.push_str(&d.process(chunk));
    }
    out.trim().to_string()
}

/// Which tone is the mark is a matter of which sideband the signal arrived
/// on, and the receiver is not told. Both polarities are ordinary on the air,
/// so requiring the operator to notice garbage and press a key means half of
/// all RTTY reads as garbage until they do.
#[test]
fn rtty_decodes_either_shift_polarity_unaided() {
    let msg = "RYRY CQ DE W1AW W1AW K";
    for shift in [170.0f32, -170.0] {
        let sig = gen_rtty(msg, 45.45, shift);
        let got = decode_rtty(&sig);
        let acc = accuracy(msg, &got);
        assert!(
            acc > 0.9,
            "{} shift copied {:.0}%: {got:?}",
            if shift > 0.0 { "normal" } else { "reversed" },
            acc * 100.0
        );
    }
}

/// The wrong polarity must never be the one that speaks: it decodes the same
/// bits inverted, which is plausible-looking Baudot, so emitting before the
/// vote settles would put a line of nonsense ahead of every transmission.
#[test]
fn rtty_emits_nothing_from_the_wrong_polarity() {
    let msg = "RYRY CQ DE W1AW W1AW K";
    for shift in [170.0f32, -170.0] {
        let sig = gen_rtty(msg, 45.45, shift);
        let got = decode_rtty(&sig);
        // Whatever comes out is the message, not a mirrored prefix of it.
        assert!(
            got.starts_with("RYRY") || got.starts_with("YRY") || got.starts_with("RY"),
            "copy did not start with the transmission: {got:?}"
        );
    }
}

/// Noise must not make up its mind either way.
#[test]
fn rtty_stays_quiet_on_noise() {
    for seed in [0x1234_5678u32, 0xdead_beef] {
        let mut rng = seed;
        let sig: Vec<Complex32> = (0..(12.0 * FS) as usize)
            .map(|_| Complex32::new(noise(&mut rng), noise(&mut rng)) * 0.3)
            .collect();
        let got = decode_rtty(&sig);
        let letters = got.chars().filter(|c| !c.is_whitespace()).count();
        assert!(
            letters <= 6,
            "12 s of noise produced {letters} chars: {got:?}"
        );
    }
}

#[test]
#[ignore]
fn bench_rtty_polarity() {
    let msg = "RYRY CQ DE W1AW W1AW K UR RST 599 QTH NEWINGTON DE W1AW K";
    println!("\n== RTTY: copy by shift polarity and noise, unaided ==");
    println!("{:>10}{:>10}{:>10}", "noise", "normal", "reversed");
    for sc in [0.03f32, 0.10, 0.20, 0.30, 0.45] {
        let n = gen_rtty_snr(msg, 45.45, 170.0, sc);
        let r = gen_rtty_snr(msg, 45.45, -170.0, sc);
        println!(
            "{:>10}{:>10}{:>10}",
            format!("{sc:.2}"),
            format!("{:.0}%", accuracy(msg, &decode_rtty(&n)) * 100.0),
            format!("{:.0}%", accuracy(msg, &decode_rtty(&r)) * 100.0),
        );
    }
}

// ------------------------------------------------- sideband / inversion

/// Receiving a signal on the opposite sideband conjugates its baseband —
/// the spectrum comes out mirrored about the tuned frequency. This is what
/// that does to each mode.
fn invert(sig: &[Complex32]) -> Vec<Complex32> {
    sig.iter().map(|s| s.conj()).collect()
}

/// Sideband cannot matter to CW or PSK31, and it is worth having that on
/// record rather than reasoned about.
///
/// CW is detected by envelope, and |s| is unchanged by conjugation. PSK31 is
/// differentially encoded, and conjugation negates every phase difference —
/// which maps 0 to 0 and pi to -pi, the same symbol. So both decode the same
/// either way up, and neither needs to know which sideband it arrived on.
/// (Data modes are USB by convention on every band regardless, including the
/// bands where voice is LSB, so this should never arise — but it costs
/// nothing to be immune to it.)
#[test]
fn cw_and_psk31_are_indifferent_to_sideband() {
    let cw_msg = "CQ CQ DE W1AW W1AW K UR RST 599 QTH NEWINGTON DE W1AW K";
    let cw = gen_cw(cw_msg, 20.0, scale_for_snr(15.0));
    let up = decode_cw(&cw, 0.0);
    let down = decode_cw(&invert(&cw), 0.0);
    assert!(
        accuracy(cw_msg, &up) > 0.9 && accuracy(cw_msg, &down) > 0.9,
        "CW differed by sideband: {up:?} vs {down:?}"
    );

    let psk_msg = "CQ CQ DE W1AW W1AW PSE K";
    let psk = gen_psk31_snr(psk_msg, 0.0, psk_scale_for_snr(15.0));
    let up = decode_psk(&psk, 0.0);
    let down = decode_psk(&invert(&psk), 0.0);
    assert!(
        accuracy(psk_msg, &up) > 0.85 && accuracy(psk_msg, &down) > 0.85,
        "PSK31 differed by sideband: {up:?} vs {down:?}"
    );
}

/// RTTY is the one narrowband mode sideband does change, because inverting
/// the spectrum swaps mark and space. That is the same thing as a reversed
/// shift, and is now detected rather than assumed.
#[test]
fn rtty_is_indifferent_to_sideband() {
    let msg = "RYRY CQ DE W1AW W1AW K";
    let sig = gen_rtty(msg, 45.45, 170.0);
    let up = decode_rtty(&sig);
    let down = decode_rtty(&invert(&sig));
    assert!(
        accuracy(msg, &up) > 0.9 && accuracy(msg, &down) > 0.9,
        "RTTY differed by sideband: {up:?} vs {down:?}"
    );
}

// ------------------------------- narrowband modes inside the FT windows

/// A real FT8 message as complex baseband: the encoder's own tone sequence,
/// rendered as continuous-phase 8-FSK at 6.25 baud and 6.25 Hz spacing.
///
/// Rendered complex rather than as real audio on purpose. A real-valued
/// signal carries its own mirror image, and squaring the pair produces a
/// strong DC term that fakes a BPSK carrier — an artefact of the synthesis,
/// not of FT8, and one that would make this test lie in the flattering
/// direction. An SDR delivers complex baseband with no such image.
fn ft8_as_iq(audio_hz: f32) -> Vec<Complex32> {
    use mfsk_core::ft8::wave_gen::message_to_tones;
    use mfsk_core::msg::wsjt77::pack77;
    let msg77 = pack77("CQ", "JA1ABC", "PM95").expect("pack77");
    let tones = message_to_tones(&msg77);
    let sps = (FS as f32 / 6.25) as usize; // 0.16 s per symbol
    let mut out = Vec::with_capacity(tones.len() * sps);
    let mut rng = 0x77aa_3311u32;
    let mut phase = 0.0f32;
    for &t in tones.iter() {
        let hz = audio_hz + t as f32 * 6.25;
        let step = 2.0 * PI * hz / FS as f32;
        for _ in 0..sps {
            phase += step;
            if phase > PI {
                phase -= 2.0 * PI;
            }
            out.push(
                Complex32::from_polar(1.0, phase)
                    + Complex32::new(noise(&mut rng), noise(&mut rng)) * 0.01,
            );
        }
    }
    out
}

/// The FT windows shadow whole narrowband sub-bands — 30 m PSK31 sits on top
/// of FT4, and 20 m and 30 m RTTY on top of FT4 as well. Letting the
/// narrowband modes be considered there is only safe if their confirmations
/// reject FT traffic, so that is checked directly rather than assumed.
#[test]
fn ft8_traffic_does_not_confirm_as_psk31() {
    let iq = ft8_as_iq(1500.0);
    // Look where the FT8 signal actually is, which is the hostile case.
    let hits = psk31::scan_span(&iq, FS, &[(1500.0, 20.0), (0.0, 20.0)]);
    assert!(hits.is_empty(), "FT8 traffic confirmed as PSK31: {hits:?}");
}

/// And the RTTY framer must not spell Baudot out of 8-FSK either.
#[test]
fn ft8_traffic_does_not_frame_as_rtty() {
    let iq = ft8_as_iq(1500.0);
    let mut d = rtty::RttyDecoder::new(FS);
    let mut chain = crate::dsp::DecodeChain::new(FS, d.bandwidth(), FS);
    chain.set_offset(1500.0);
    let mut audio = Vec::new();
    let mut out = String::new();
    for chunk in iq.chunks(4096) {
        chain.process(chunk, &mut audio);
        out.push_str(&d.process(&audio));
    }
    let letters = out.chars().filter(|c| !c.is_whitespace()).count();
    assert!(
        letters <= 4,
        "FT8 traffic framed as {letters} chars of RTTY: {out:?}"
    );
}

/// What the copy floor is actually made of.
///
/// The floor is only worth having if the number behind it separates the two
/// cases: a signal being copied, and a demodulator chewing on band noise.
/// This prints both so the default floor can be set from measurements
/// instead of taste.
#[test]
#[ignore]
fn bench_psk31_confidence() {
    let msg = "CQ CQ DE W1AW W1AW PSE K";
    println!("\n== confidence on signal, by SNR (in 31 Hz) ==");
    for db in [40, 30, 20, 15, 10, 6, 3, 0] {
        let sig = gen_psk31_snr(msg, 0.0, psk_scale_for_snr(db as f32));
        let mut d = psk31::Psk31Decoder::new(FS);
        let mut chain = crate::dsp::DecodeChain::new(FS, d.bandwidth(), FS);
        chain.set_offset(d.offset_shift());
        let mut audio = Vec::new();
        let mut out = String::new();
        let mut worst = 1.0f32;
        for chunk in sig.chunks(4096) {
            chain.process(chunk, &mut audio);
            out.push_str(&d.process(&audio));
            if d.locked() {
                worst = worst.min(d.confidence().unwrap_or(0.0));
            }
        }
        println!(
            "  {db:>2} dB  conf {:>4}  (min while locked {:>4})  copy {:>4}",
            format!("{:.0}%", d.confidence().unwrap_or(0.0) * 100.0),
            format!("{:.0}%", worst * 100.0),
            format!("{:.0}%", accuracy(msg, out.trim()) * 100.0),
        );
    }

    println!("\n== confidence on noise alone ==");
    for seed in [0x1234_5678u32, 0xdead_beef, 0x0bad_f00d] {
        let mut rng = seed;
        let sig: Vec<Complex32> = (0..(FS as usize * 20))
            .map(|_| Complex32::new(noise(&mut rng), noise(&mut rng)))
            .collect();
        let mut d = psk31::Psk31Decoder::new(FS);
        let mut chain = crate::dsp::DecodeChain::new(FS, d.bandwidth(), FS);
        chain.set_offset(d.offset_shift());
        let mut audio = Vec::new();
        let mut out = String::new();
        let mut peak = 0.0f32;
        for chunk in sig.chunks(4096) {
            chain.process(chunk, &mut audio);
            out.push_str(&d.process(&audio));
            peak = peak.max(d.confidence().unwrap_or(0.0));
        }
        println!(
            "  seed {seed:#010x}  peak conf {:>4}  printed {} chars: {:?}",
            format!("{:.0}%", peak * 100.0),
            out.len(),
            out.chars().take(40).collect::<String>(),
        );
    }
}

/// The copy floor rests on one claim: confidence separates a signal being
/// copied from a demodulator chewing on band noise. `bench_psk31_confidence`
/// prints the whole curve; this pins the two ends of it, either side of the
/// scanner's default 40% floor.
#[test]
fn psk31_confidence_separates_copy_from_noise() {
    let msg = "CQ CQ DE W1AW W1AW PSE K";
    let sig = gen_psk31_snr(msg, 0.0, psk_scale_for_snr(10.0));
    let mut d = psk31::Psk31Decoder::new(FS);
    let mut chain = crate::dsp::DecodeChain::new(FS, d.bandwidth(), FS);
    chain.set_offset(d.offset_shift());
    let mut audio = Vec::new();
    let mut out = String::new();
    // The floor is applied from the first character, so what matters is the
    // worst reading while the decoder is copying — not where it ends up.
    let mut worst = 1.0f32;
    for chunk in sig.chunks(4096) {
        chain.process(chunk, &mut audio);
        out.push_str(&d.process(&audio));
        if !out.trim().is_empty() {
            worst = worst.min(d.confidence().unwrap_or(0.0));
        }
    }
    assert!(
        worst > 0.45,
        "a 10 dB signal copied at {:.0}% accuracy dropped to {:.0}% confidence \
         while it was copying",
        accuracy(msg, out.trim()) * 100.0,
        worst * 100.0
    );

    // Noise must not merely score lower — it must stay under the floor for
    // the whole run, because the floor is applied moment by moment.
    for seed in [0x1234_5678u32, 0xdead_beef, 0x0bad_f00d] {
        let mut rng = seed;
        let sig: Vec<Complex32> = (0..(FS as usize * 20))
            .map(|_| Complex32::new(noise(&mut rng), noise(&mut rng)))
            .collect();
        let mut d = psk31::Psk31Decoder::new(FS);
        let mut chain = crate::dsp::DecodeChain::new(FS, d.bandwidth(), FS);
        chain.set_offset(d.offset_shift());
        let mut audio = Vec::new();
        let mut peak = 0.0f32;
        for chunk in sig.chunks(4096) {
            chain.process(chunk, &mut audio);
            let _ = d.process(&audio);
            peak = peak.max(d.confidence().unwrap_or(0.0));
        }
        assert!(
            peak < 0.40,
            "noise (seed {seed:#010x}) reached {:.0}% confidence — the default floor would let it print",
            peak * 100.0
        );
    }
}

/// The copy floor is one threshold across every mode, so CW and RTTY have to
/// read on the same scale PSK31 was calibrated to: signal well above the
/// floor, noise well below it, and — because the floor is applied from the
/// first character — a metric that gets there without a long warm-up.
#[test]
#[ignore]
fn bench_cw_rtty_confidence() {
    let msg = "CQ CQ DE W1AW K";
    println!("\n== CW confidence ==");
    for scale in [0.01f32, 0.05, 0.15, 0.3, 0.6] {
        let sig = gen_cw(msg, 20.0, scale);
        let mut d = cw::CwDecoder::new(FS);
        let mut out = String::new();
        let mut first = None;
        for chunk in sig.chunks(4096) {
            out.push_str(&d.process(chunk));
            if first.is_none() && !out.trim().is_empty() {
                first = d.confidence();
            }
        }
        println!(
            "  noise x{scale:<5} conf {:>4}  at first copy {:>4}  {:?}",
            format!("{:.0}%", d.confidence().unwrap_or(0.0) * 100.0),
            format!("{:.0}%", first.unwrap_or(0.0) * 100.0),
            out.trim().chars().take(28).collect::<String>(),
        );
    }
    for seed in [0x1234_5678u32, 0xdead_beef] {
        let mut rng = seed;
        let sig: Vec<Complex32> = (0..(FS as usize * 20))
            .map(|_| Complex32::new(noise(&mut rng), noise(&mut rng)))
            .collect();
        let mut d = cw::CwDecoder::new(FS);
        let mut peak = 0.0f32;
        let mut out = String::new();
        for chunk in sig.chunks(4096) {
            out.push_str(&d.process(chunk));
            peak = peak.max(d.confidence().unwrap_or(0.0));
        }
        println!(
            "  noise only     peak {:>4}  printed {} chars",
            format!("{:.0}%", peak * 100.0),
            out.trim().len()
        );
    }

    println!("\n== RTTY confidence ==");
    for scale in [0.03f32, 0.1, 0.3, 0.6] {
        let sig = gen_rtty_snr("CQ CQ DE W1AW K ", 45.45, 170.0, scale);
        let mut d = rtty::RttyDecoder::new(FS);
        let mut out = String::new();
        let mut first = None;
        for chunk in sig.chunks(4096) {
            out.push_str(&d.process(chunk));
            if first.is_none() && !out.trim().is_empty() {
                first = d.confidence();
            }
        }
        println!(
            "  noise x{scale:<5} conf {:>4}  at first copy {:>4}  {:?}",
            format!("{:.0}%", d.confidence().unwrap_or(0.0) * 100.0),
            format!("{:.0}%", first.unwrap_or(0.0) * 100.0),
            out.trim().chars().take(28).collect::<String>(),
        );
    }
    for seed in [0x1234_5678u32, 0xdead_beef] {
        let mut rng = seed;
        let sig: Vec<Complex32> = (0..(FS as usize * 20))
            .map(|_| Complex32::new(noise(&mut rng), noise(&mut rng)))
            .collect();
        let mut d = rtty::RttyDecoder::new(FS);
        let mut peak = 0.0f32;
        let mut out = String::new();
        for chunk in sig.chunks(4096) {
            out.push_str(&d.process(chunk));
            peak = peak.max(d.confidence().unwrap_or(0.0));
        }
        println!(
            "  noise only     peak {:>4}  printed {} chars",
            format!("{:.0}%", peak * 100.0),
            out.trim().len()
        );
    }
}

/// The floor is applied from the first character, so every mode has to be
/// confident *by* the first character — a metric that only settles after a
/// few seconds would silently eat the start of every transmission, which is
/// where the callsigns are. `bench_cw_rtty_confidence` prints the levels.
#[test]
fn confidence_is_ready_by_the_first_character() {
    /// The scanner's default copy floor (`main::COPY_FLOOR`).
    const FLOOR: f32 = 0.40;

    let msg = "CQ CQ DE W1AW K";
    let cases: Vec<(&str, Box<dyn Decoder>, Vec<Complex32>)> = vec![
        (
            "CW",
            Box::new(cw::CwDecoder::new(FS)),
            gen_cw(msg, 20.0, 0.15),
        ),
        (
            "RTTY",
            Box::new(rtty::RttyDecoder::new(FS)),
            gen_rtty_snr("CQ CQ DE W1AW K ", 45.45, 170.0, 0.1),
        ),
    ];
    for (name, mut d, sig) in cases {
        let mut out = String::new();
        let mut at_first = None;
        for chunk in sig.chunks(4096) {
            out.push_str(&d.process(chunk));
            if at_first.is_none() && !out.trim().is_empty() {
                // None is "no opinion", which the scanner passes through.
                at_first = Some(d.confidence().unwrap_or(1.0));
            }
        }
        assert!(
            out.contains("W1AW"),
            "{name}: expected copy to test against, got {out:?}"
        );
        let at_first = at_first.unwrap_or(0.0);
        assert!(
            at_first >= FLOOR,
            "{name} reported {:.0}% when its first characters arrived — the \
             default {:.0}% floor would have dropped them",
            at_first * 100.0,
            FLOOR * 100.0
        );
    }
}

/// RTTY at an arbitrary sample rate, for span-level tests.
///
/// Realistic in the way that matters for identification: the station idles on
/// mark, so the space tone is present only inside characters and is far weaker
/// than mark in an averaged spectrum.
pub(crate) fn gen_rtty_at(
    text: &str,
    fs: f64,
    offset_hz: f64,
    shift: f32,
    snr_scale: f32,
    secs: f64,
) -> Vec<Complex32> {
    let sps = (fs as f32 / 45.45) as usize;
    let mut bits: Vec<bool> = vec![true; 60];
    let mut figs_state = false;
    for c in text.chars() {
        let Some((code, figs)) = ita2_code(c) else {
            continue;
        };
        if figs != figs_state && c != ' ' {
            let sc = if figs { 0x1Bu8 } else { 0x1F };
            bits.push(false);
            for b in 0..5 {
                bits.push(sc & (1 << b) != 0);
            }
            bits.push(true);
            bits.push(true);
            figs_state = figs;
        }
        bits.push(false);
        for b in 0..5 {
            bits.push(code & (1 << b) != 0);
        }
        bits.push(true);
        bits.push(true);
    }
    bits.extend(std::iter::repeat_n(true, 60));

    let mut rng = 0x5eed_9001u32;
    let want = (fs * secs) as usize;
    let mut out = Vec::with_capacity(want);
    let mut phase = 0.0f32;
    let mut i = 0usize;
    while out.len() < want {
        let mark = bits[i % bits.len()];
        let f = offset_hz as f32 + if mark { shift / 2.0 } else { -shift / 2.0 };
        for _ in 0..sps {
            phase += 2.0 * PI * f / fs as f32;
            out.push(
                Complex32::from_polar(1.0, phase)
                    + Complex32::new(noise(&mut rng), noise(&mut rng)) * snr_scale,
            );
        }
        i += 1;
    }
    out.truncate(want);
    out
}

/// Does the keying rate actually separate the modes? Everything else the
/// PSK31 confirmation measures, a keyed carrier can fake.
#[test]
#[ignore]
fn bench_baud_line() {
    let show = |name: &str, sig: &[Complex32], hz: f32| {
        let mut out = format!("  {name:<34}");
        for secs in [1.0f32, 1.6, 2.0] {
            let n = (FS as f32 * secs) as usize;
            let cut = &sig[sig.len().saturating_sub(n)..];
            let (f, prom, at) = psk31::baud_line(cut, FS as f32, hz);
            out.push_str(&format!(
                "  {secs:.1}s: {f:>5.1}Hz x{prom:>5.1} (31.25: x{at:>5.1})"
            ));
        }
        println!("{out}");
    };
    println!("\n== PSK31 (expect ~31.25 Hz) ==");
    for db in [30.0f32, 20.0, 15.0, 10.0, 6.0] {
        let sig = gen_psk31_snr("CQ CQ DE W1AW W1AW PSE K", 0.0, psk_scale_for_snr(db));
        show(&format!("psk31 {db:.0} dB"), &sig, 0.0);
    }
    let idle = gen_psk31_snr("", 0.0, psk_scale_for_snr(20.0));
    if idle.len() > 8000 {
        show("psk31 idle", &idle, 0.0);
    }

    println!("\n== RTTY 45.45 baud (expect ~45.5 or ~22.7 Hz) ==");
    let rtty = gen_rtty_at("CQ CQ DE W1AW W1AW K ", FS, 0.0, 170.0, 0.03, 4.0);
    show("rtty, mixed to the mark tone", &rtty, 85.0);
    show("rtty, mixed to mid-shift", &rtty, 0.0);
    show("rtty, mixed to the space tone", &rtty, -85.0);
    let ry = gen_rtty_at("RYRYRYRYRYRYRYRY ", FS, 0.0, 170.0, 0.03, 4.0);
    show("rtty RY test pattern, mark", &ry, 85.0);

    println!("\n== things with no data on them ==");
    let mut phase = 0.0f32;
    let carrier: Vec<Complex32> = (0..(FS as usize * 4))
        .map(|_| {
            phase += 2.0 * PI * 20.0 / FS as f32;
            Complex32::from_polar(1.0, phase)
        })
        .collect();
    show("unkeyed carrier", &carrier, 20.0);
    let mut rng = 0x1234_5678u32;
    let n: Vec<Complex32> = (0..(FS as usize * 4))
        .map(|_| Complex32::new(noise(&mut rng), noise(&mut rng)))
        .collect();
    show("noise", &n, 0.0);
    let cw = gen_cw("CQ CQ DE W1AW K", 20.0, 0.03);
    show("cw 20 wpm", &cw, CW_TONE);
}

/// A spot's SNR has to be a measurement, not a placeholder: it is the number
/// that tells someone reading the map whether the path was marginal or solid.
/// Both decoders are checked against signals of known SNR, so what is asserted
/// here is that the reported figure tracks the band rather than sitting at a
/// clamp — the failure both estimators started out with.
#[test]
fn spot_snr_follows_the_band() {
    for (label, weak, strong) in [
        ("CW", gen_cw(CQ, 20.0, 5.0), gen_cw(CQ, 20.0, 0.6)),
        (
            "RTTY",
            gen_rtty_snr(RY_CQ, 45.45, 170.0, 5.0),
            gen_rtty_snr(RY_CQ, 45.45, 170.0, 0.6),
        ),
    ] {
        let mut snrs = Vec::new();
        for sig in [&weak, &strong] {
            let mut d: Box<dyn Decoder> = if label == "CW" {
                Box::new(cw::CwDecoder::new(FS))
            } else {
                Box::new(rtty::RttyDecoder::new(FS))
            };
            let mut best = f32::MIN;
            for chunk in sig.chunks(4096) {
                d.process(chunk);
                for m in d.take_messages() {
                    best = best.max(m.snr_db);
                }
            }
            snrs.push(best);
        }
        let (weak_db, strong_db) = (snrs[0], snrs[1]);
        assert!(
            weak_db > f32::MIN && strong_db > f32::MIN,
            "{label}: no spot to take an SNR from"
        );
        // The two differ by 18 dB of added noise. Anything that reports them
        // within a couple of dB of each other is measuring its own arithmetic.
        assert!(
            strong_db - weak_db > 8.0,
            "{label}: {weak_db:.1} dB weak vs {strong_db:.1} dB strong — not tracking"
        );
        assert!(
            (-24.0..=20.0).contains(&weak_db) && (-24.0..=20.0).contains(&strong_db),
            "{label}: SNR outside the reportable range: {weak_db} / {strong_db}"
        );
    }
}


/// Fading, at signal levels where the flat benchmark reads 100 %.
///
/// Every other CW figure in this file is measured on a constant-amplitude
/// carrier. Nothing on HF is constant-amplitude. Measured across the live 20m
/// capture, the mark level of a single station moves by 12 to 31 dB inside one
/// 60-second over, and which stations the decoder copies tracks that number
/// far more closely than it tracks their SNR: the one it copies cleanly fades
/// by 15 dB, the two it turns into nonsense fade by 30.
///
/// This is why: at 15 dB SNR — comfortable, 100 % copy when flat — 20 dB of
/// QSB takes it to about 40 %. The threshold tracker is the reason. `peak`
/// decays over 2.5 s and `floor` attacks over 2, deliberately slow so keying
/// cannot drag them, but QSB moves on the same timescale as the keying they
/// are trying to ignore. Through a fade-down the arming threshold is stranded
/// above the signal and whole characters vanish; on the way back up it sits far
/// below and noise walks in.
///
/// Fixing that needs a detector whose decision does not depend on an absolute
/// level — the numbers here are the yardstick for whether one works.
#[test]
#[ignore]
fn bench_cw_fading() {
    let msg = "CQ CQ DE W1AW W1AW K UR RST 599 QTH NEWINGTON DE W1AW K";
    println!("\n  copy vs fade depth, 20 WPM (0 dB row is the flat case)\n");
    println!("{:>8} {:>8} {:>8} {:>8}", "QSB dB", "15dB", "10dB", "6dB");
    for depth_db in [0.0f32, 6.0, 12.0, 20.0, 30.0] {
        print!("{depth_db:>8.0}");
        for snr in [15.0f32, 10.0, 6.0] {
            let key = gen_cw_key(msg, 20.0, 3.0, &[1.0]);
            // A slow fade, 0.15 Hz, applied in dB so the depth means what it says.
            let faded: Vec<f32> = key
                .iter()
                .enumerate()
                .map(|(i, &k)| {
                    let t = i as f32 / FS as f32;
                    let db = -0.5 * depth_db * (1.0 + (2.0 * PI * 0.15 * t).sin());
                    k * 10f32.powf(db / 20.0)
                })
                .collect();
            let sig = key_to_iq(&faded, scale_for_snr(snr), CW_TONE);
            let acc = accuracy(msg, &decode_cw(&sig, 0.0));
            print!("{:>8}", format!("{:.0}%", acc * 100.0));
        }
        println!();
    }
}


