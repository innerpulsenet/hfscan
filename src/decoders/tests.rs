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

// ------------------------------------------------------------------- CW

fn morse_for(c: char) -> &'static str {
    match c {
        'A' => ".-", 'B' => "-...", 'C' => "-.-.", 'D' => "-..", 'E' => ".",
        'F' => "..-.", 'G' => "--.", 'H' => "....", 'I' => "..", 'J' => ".---",
        'K' => "-.-", 'L' => ".-..", 'M' => "--", 'N' => "-.", 'O' => "---",
        'P' => ".--.", 'Q' => "--.-", 'R' => ".-.", 'S' => "...", 'T' => "-",
        'U' => "..-", 'V' => "...-", 'W' => ".--", 'X' => "-..-", 'Y' => "-.--",
        'Z' => "--..",
        '0' => "-----", '1' => ".----", '2' => "..---", '3' => "...--",
        '4' => "....-", '5' => ".....", '6' => "-....", '7' => "--...",
        '8' => "---..", '9' => "----.",
        '/' => "-..-.", '?' => "..--..",
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
    key.extend(std::iter::repeat(0.0).take(dit * 8));
    key
}

/// AM the key line onto `tone` with noise, as `gen_cw_at` always did.
fn key_to_iq(key: &[f32], snr_scale: f32, tone: f32) -> Vec<Complex32> {
    // Shape the key envelope so it has realistic rise/fall instead of clicks.
    let rise = (0.005 * FS as f32) as usize;
    let mut env = key.to_vec();
    let mut acc = 0.0f32;
    let a = 1.0 - (-1.0 / rise as f32).exp();
    for v in env.iter_mut() {
        acc += (*v - acc) * a;
        *v = acc;
    }

    let mut rng = 0x1234_5678u32;
    env.iter()
        .enumerate()
        .map(|(i, &e)| {
            let ph = 2.0 * PI * tone * i as f32 / FS as f32;
            let s = Complex32::from_polar(e, ph);
            s + Complex32::new(noise(&mut rng), noise(&mut rng)) * snr_scale
        })
        .collect()
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

/// A human fist: light dahs (2.2 dits) with ±15% element jitter, so the
/// shortest dahs land at 1.87 dits. A fixed dit/dah boundary at 2.0
/// misreads those; the adaptive boundary must settle between the
/// operator's own clusters.
#[test]
fn cw_decodes_a_sloppy_fist() {
    let key = gen_cw_key("CQ CQ DE W1AW W1AW K", 20.0, 2.2, &[0.85, 1.1, 1.0, 0.9, 1.15]);
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

// ----------------------------------------------------------------- RTTY

fn ita2_code(c: char) -> Option<(u8, bool)> {
    const LTRS: &str = "\0E\nA SIU\rDRJNFCKTZLWHYPQOBG\0MXV\0";
    // Look the character up in the letters table first, then figures.
    if let Some(i) = LTRS.chars().position(|x| x == c) {
        return Some((i as u8, false));
    }
    let figs = [
        ('3', 1), ('-', 3), ('\'', 5), ('8', 6), ('7', 7), ('$', 9), ('4', 10),
        (',', 12), ('!', 13), (':', 14), ('(', 15), ('5', 16), ('"', 17),
        (')', 18), ('2', 19), ('#', 20), ('6', 21), ('0', 22), ('1', 23),
        ('9', 24), ('?', 25), ('&', 26), ('.', 28), ('/', 29), (';', 30),
    ];
    figs.iter().find(|(x, _)| *x == c).map(|(_, i)| (*i as u8, true))
}

fn gen_rtty(text: &str, baud: f32, shift: f32) -> Vec<Complex32> {
    gen_rtty_snr(text, baud, shift, 0.03)
}

fn gen_rtty_snr(text: &str, baud: f32, shift: f32, snr_scale: f32) -> Vec<Complex32> {
    gen_rtty_faded(text, baud, shift, snr_scale, 1.0, 1.0)
}

fn gen_rtty_faded(text: &str, baud: f32, shift: f32, snr_scale: f32, mark_amp: f32, space_amp: f32) -> Vec<Complex32> {
    let sps = FS as f32 / baud;
    let mut bits: Vec<bool> = Vec::new();
    // idle mark so the decoder starts in a known state
    bits.extend(std::iter::repeat(true).take((baud as usize).max(20)));
    let mut figs_state = false;
    for c in text.chars() {
        let Some((code, figs)) = ita2_code(c) else { continue };
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
    for chunk in sig.chunks(4096) { out.push_str(&d.process(chunk)); }
    assert!(out.contains("TEST"), "-20 dB space fade was not copyable: {out:?} ({})", d.status());
}

#[test]
#[ignore]
fn bench_rtty_matched_filter_fade() {
    for fade_db in [0.0f32, -10.0, -20.0] {
        let amp = 10f32.powf(fade_db / 20.0);
        let sig = gen_rtty_faded("RYRY RYRY CQ DE TEST TEST", 45.45, 170.0, 0.025, 1.0, amp);
        let mut d = rtty::RttyDecoder::new(FS);
        let mut out = String::new();
        for chunk in sig.chunks(4096) { out.push_str(&d.process(chunk)); }
        println!("RTTY matched filters, space {fade_db:+.0} dB: {} chars, TEST={}", out.len(), out.contains("TEST"));
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
    assert!(!out.contains("TEST"), "reversed shift should not decode: {out:?}");
}

// ---------------------------------------------------------------- PSK31

fn gen_psk31(text: &str, freq_offset: f32) -> Vec<Complex32> {
    gen_psk31_snr(text, freq_offset, 0.02)
}

fn gen_psk31_snr(text: &str, freq_offset: f32, snr_scale: f32) -> Vec<Complex32> {
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

    let mut rng = 0xdead_beefu32;
    base.iter()
        .enumerate()
        .map(|(i, &v)| {
            let ph = 2.0 * PI * freq_offset * i as f32 / FS as f32;
            Complex32::from_polar(v.abs(), ph + if v < 0.0 { PI } else { 0.0 })
                + Complex32::new(noise(&mut rng), noise(&mut rng)) * snr_scale
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
    let s = |b: &[bool]| b.iter().map(|x| if *x { '1' } else { '0' }).collect::<String>();
    println!("tx ({:3}): {}", tx.len(), s(&tx));
    println!("rx ({:3}): {}", rx.len(), s(rx));
    // Find the alignment that matches best.
    let mut best = (0usize, 0usize);
    for off in 0..rx.len().saturating_sub(tx.len()).min(200) {
        let n = tx
            .iter()
            .zip(&rx[off..])
            .filter(|(a, b)| a == b)
            .count();
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
    ft8::FtDecoder::decode_audio(&audio, ft4)
}

#[test]
fn ft8_decodes_a_slot() {
    let out = ft_roundtrip(false, "JA1ABC", "PM95", 1500.0);
    assert!(
        out.iter().any(|l| l.contains("CQ JA1ABC PM95")),
        "expected the message back, got {out:?}"
    );
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
    assert!(peak > 500, "audio level far too low for decoding: peak {peak}");
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

#[test]
#[ignore]
fn bench_cw_accuracy() {
    let msg = "CQ CQ DE W1AW W1AW K";
    println!("\nsent: {msg:?}\n");

    println!("== on frequency, by speed and SNR ==");
    print!("{:>6}", "WPM");
    for db in [40, 20, 15, 10, 6, 3, 0] {
        print!("{:>8}", format!("{db}dB"));
    }
    println!();
    for wpm in [12.0f32, 18.0, 25.0, 35.0] {
        print!("{wpm:>6.0}");
        for db in [40, 20, 15, 10, 6, 3, 0] {
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
            Complex32::from_polar(1.0, ph)
                + Complex32::new(noise(&mut rng), noise(&mut rng)) * 0.02
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

#[test]
#[ignore]
fn bench_psk31_accuracy() {
    let msg = "CQ CQ DE W1AW W1AW PSE K";
    println!("\nsent: {msg:?}");

    println!("\n== on frequency, by SNR (in 31 Hz) ==");
    for db in [40, 30, 20, 15, 10, 6, 3, 0] {
        let sig = gen_psk31_snr(msg, 0.0, psk_scale_for_snr(db as f32));
        let (got, lock, locked) = decode_psk_dbg(&sig, 0.0);
        println!(
            "  {db:>2} dB  {:>4}  lock {lock:>+6.1} Hz {}  {got:?}",
            format!("{:.0}%", accuracy(msg, &got) * 100.0),
            if locked { "yes" } else { "NO " },
        );
    }

    println!("\n== 15 dB, by residual tuning error ==");
    for err in [-150.0f32, -100.0, -50.0, -20.0, 0.0, 20.0, 50.0, 100.0, 150.0] {
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
    for db in [20, 10, 6, 3] {
        eprintln!("--- {db} dB ---");
        let sig = gen_psk31_snr(msg, 0.0, psk_scale_for_snr(db as f32));
        let got = decode_psk(&sig, 0.0);
        eprintln!("  => {got:?}");
    }
}

/// PSK31 at an arbitrary sample rate, for span-level tests.
pub(crate) fn gen_psk31_at(text: &str, fs: f64, offset_hz: f64, snr_scale: f32, secs: f64) -> Vec<Complex32> {
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
        assert!(
            acc > 0.85,
            "{db:.0} dB copied {:.0}%: {got:?}",
            acc * 100.0
        );
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
            Complex32::from_polar(1.0, ph)
                + Complex32::new(noise(&mut rng), noise(&mut rng)) * 0.02
        })
        .collect();
    let hits = psk31::scan_span(&sig, FS, &[(0.0, 20.0)]);
    assert!(hits.is_empty(), "a steady carrier confirmed as PSK31: {hits:?}");
}

// ------------------------------------------------------- AGC benches

/// Decode with the app's software AGC in the path, as auto mode runs it:
/// one scalar gain per audio block, applied after the tuning chain.
fn decode_with_agc(
    sig: &[Complex32],
    tune: f32,
    mut d: Box<dyn Decoder>,
    agc: bool,
) -> String {
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
        assert!(letters <= 6, "12 s of noise produced {letters} chars: {got:?}");
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
    use mfsk_core::msg::wsjt77::pack77;
    use mfsk_core::ft8::wave_gen::message_to_tones;
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
    assert!(
        hits.is_empty(),
        "FT8 traffic confirmed as PSK31: {hits:?}"
    );
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
    assert!(letters <= 4, "FT8 traffic framed as {letters} chars of RTTY: {out:?}");
}
