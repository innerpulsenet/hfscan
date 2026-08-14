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

fn gen_cw(text: &str, wpm: f32, snr_scale: f32) -> Vec<Complex32> {
    let dit = (1.2 / wpm * FS as f32) as usize;
    let tone = 600.0f32; // audio offset inside the CW passband
    let mut key: Vec<f32> = Vec::new();
    // lead-in silence lets the threshold tracker settle
    key.extend(std::iter::repeat(0.0).take(dit * 8));
    for ch in text.chars() {
        if ch == ' ' {
            key.extend(std::iter::repeat(0.0).take(dit * 4));
            continue;
        }
        for el in morse_for(ch).chars() {
            let n = if el == '.' { dit } else { dit * 3 };
            key.extend(std::iter::repeat(1.0).take(n));
            key.extend(std::iter::repeat(0.0).take(dit)); // inter-element
        }
        key.extend(std::iter::repeat(0.0).take(dit * 2)); // char gap (total 3)
    }
    key.extend(std::iter::repeat(0.0).take(dit * 8));

    // Shape the key envelope so it has realistic rise/fall instead of clicks.
    let rise = (0.005 * FS as f32) as usize;
    let mut env = key.clone();
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
            out.push(
                Complex32::from_polar(1.0, phase)
                    + Complex32::new(noise(&mut rng), noise(&mut rng)) * 0.03,
            );
        }
    }
    out
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
                + Complex32::new(noise(&mut rng), noise(&mut rng)) * 0.02
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
