//! Stage 3 — rescoring decoded CW against the language ham CW actually uses.
//!
//! The detector is fade-limited and, at the SNRs this scanner spends its time
//! at, hands back words with an element wrong: `CQ` arrives as `CG` or `CO`,
//! `DE` as `LE`, `TEST` as `NEST`. Those are not equally likely mistakes with
//! every other word — they are one dropped or added element away from a token
//! that ham CW sends thousands of times an hour, and nothing else.
//!
//! So the distance that matters is not between characters. `C` and `G` are
//! unrelated letters and adjacent Morse patterns (`-.-.` against `--.`), and a
//! character-level edit distance cannot see that. Everything here works on the
//! element string, which is the space the errors are actually made in.
//!
//! **This corrects; it does not invent.** Three rules keep it honest:
//!
//! - A word that already looks like a callsign is never rewritten. Spots are
//!   the product; turning a real call into a procedural word to make the
//!   transcript read better would be exactly the wrong trade.
//! - A correction has to be both close in absolute terms and clearly closer
//!   than the runner-up. Two candidates at similar distance means the evidence
//!   does not name one, and the decode is left alone.
//! - Two lists, not one. `LEXICON` is broad and only ever answers "this is
//!   already a real word, leave it alone"; `CORRECT_TO` is small and is the
//!   only thing a decode may be rewritten *into*. Correcting toward the full
//!   list rewrote `NEWINGTON CT` into `NEWINGTON BT`, and `THE QSO` into
//!   `TEST QSO`, both from clean copy.
//! - The lexicon holds only vocabulary any CW operator would recognise —
//!   procedural signals, Q-codes and standard abbreviations. No callsign, no
//!   place name, and nothing lifted from the test corpus. A lexicon fitted to
//!   `bench_cw_score`'s messages would score well and mean nothing, which is
//!   why `cw_capture` — real off-air signals with ground truth established
//!   independently of this decoder — is the metric that counts here.

use super::cw::morse_elements;
use crate::report::is_callsign;

/// Ham CW's working vocabulary: procedural signals, Q-codes, and the
/// abbreviations that appear on any CW crib sheet.
///
/// Deliberately excludes callsigns, names and place names. Those are the
/// things worth spotting, and a lexicon that "corrects" toward them would
/// manufacture the very tokens the capture metric counts.
const LEXICON: &[&str] = &[
    // Procedural signals and the shape of a call.
    "CQ", "DE", "KN", "SK", "AR", "BK", "BT", "AS", "NW", "BTU", "BOTH",
    // Q-codes in common CW use.
    "QTH", "QSL", "QSO", "QRM", "QRN", "QRO", "QRP", "QRQ", "QRS", "QRT",
    "QRV", "QRX", "QRZ", "QSB", "QSY", "QRL",
    // Signal reports. 5NN is the contest cut-number form of 599.
    "RST", "599", "5NN", "579", "559", "589",
    // Standard abbreviations.
    "TU", "TNX", "TKS", "PSE", "UR", "URS", "ES", "HW", "CPY", "CPI", "FB",
    "OM", "YL", "XYL", "DX", "WX", "RIG", "ANT", "PWR", "WPM", "HR", "HRE",
    "AGN", "ABT", "GUD", "RPT", "SRI", "MNI", "WID", "CUL", "GL", "GB", "DR",
    "VY", "OP", "NAME", "TEST", "CONTEST", "POTA", "SOTA", "IOTA", "GM", "GA",
    "GE", "GN", "TEMP", "FER", "WKD", "WRK", "TTS", "PWR", "SIG", "SIGS",
    // Closings.
    "73", "72", "88", "EE",
];

/// The words a mis-copy may be corrected *to*.
///
/// Deliberately much smaller than `LEXICON`, and the distinction is the whole
/// safety argument. `LEXICON` answers "is this already a real word, leave it
/// alone"; this answers "is this worth rewriting a decode into", and the
/// second question has a far higher bar because a wrong answer corrupts copy
/// that was right.
///
/// Two things earn a place here: the token has to carry meaning — `CQ` and
/// `DE` gate `CallScanner`'s state machine and therefore the spots, which is
/// the product — and it has to be unlikely to sit one element from an
/// ordinary word. That second rule is why the filler signals are absent.
/// `BT` is one element from `CT`, and `NEWINGTON CT` is an address, not a
/// procedural separator; correcting toward `BT` cost real copy on the
/// laboratory grid and bought nothing.
const CORRECT_TO: &[&str] = &[
    "CQ", "DE", "TEST", "CONTEST", "POTA", "SOTA", "QRZ", "QTH", "QSL", "QSO",
    "QRP", "QRM", "QRN", "QSB", "RST", "599", "5NN", "TNX", "PSE", "AGN",
    "NAME", "QSY", "QRT",
];

/// Longest word worth rescoring. Beyond this a mis-copy is not a near miss.
const MAX_LEN: usize = 10;

/// A decoded character together with the elements it was decoded from.
///
/// The elements are what the rescorer needs; `ch` is `'*'` when the pattern
/// matched nothing in the table, and in that case the pattern is still the
/// best evidence available about what was sent.
#[derive(Clone, Debug)]
pub struct Sym {
    pub ch: char,
    pub elems: String,
}

/// Levenshtein distance between two element strings.
///
/// `'/'` separates characters and is edited like any other symbol, so a
/// character gap heard as an element gap — two letters run into one — costs a
/// single deletion, which is what it physically is.
fn elem_distance(a: &str, b: &str) -> usize {
    let a = a.as_bytes();
    let b = b.as_bytes();
    if a.is_empty() {
        return b.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ca != cb);
            cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// The element string for a word, `'/'` between characters.
fn elems_of_syms(w: &[Sym]) -> String {
    let mut out = String::new();
    for (i, s) in w.iter().enumerate() {
        if i > 0 {
            out.push('/');
        }
        out.push_str(&s.elems);
    }
    out
}

/// The element string for a known word.
fn elems_of_str(w: &str) -> Option<String> {
    let mut out = String::new();
    for (i, c) in w.chars().enumerate() {
        if i > 0 {
            out.push('/');
        }
        out.push_str(morse_elements(c)?);
    }
    Some(out)
}

/// How much element error to forgive on a candidate of this many elements.
///
/// One element for anything short. Two elements is enough to turn `THE` into
/// `TEST`, which is a correctly copied English word being replaced by a
/// procedural one — measured, and it cost more than it returned. Only a long
/// candidate, carrying enough evidence that two slips still leave the match
/// unambiguous, gets any more latitude.
fn budget(cand_elems: usize) -> usize {
    if cand_elems <= 13 { 1 } else { 2 }
}

/// The plain text of a decoded word.
fn text_of(w: &[Sym]) -> String {
    w.iter().map(|s| s.ch).collect()
}

/// Rescore one word against the lexicon.
///
/// Returns the replacement, or `None` to keep what was decoded.
pub fn rescore_word(w: &[Sym]) -> Option<&'static str> {
    if w.len() < 2 || w.len() > MAX_LEN {
        return None;
    }
    let decoded = text_of(w);
    // Never rewrite something already shaped like a callsign: those are the
    // product, and a wrong "correction" here is a wrong spot.
    if is_callsign(&decoded) {
        return None;
    }
    if LEXICON.contains(&decoded.as_str()) {
        return None;
    }

    let got = elems_of_syms(w);
    let mut best: Option<(usize, &'static str)> = None;
    let mut runner: Option<usize> = None;
    for cand in CORRECT_TO {
        if cand.len() < 2 {
            continue;
        }
        let Some(ce) = elems_of_str(cand) else {
            continue;
        };
        // A candidate of wildly different length is not a mis-copy of this.
        if ce.len().abs_diff(got.len()) > 6 {
            continue;
        }
        let d = elem_distance(&got, &ce);
        if d > budget(ce.len()) {
            continue;
        }
        match best {
            Some((bd, _)) if d >= bd => {
                if runner.is_none_or(|r| d < r) {
                    runner = Some(d);
                }
            }
            _ => {
                if let Some((bd, _)) = best {
                    runner = Some(runner.map_or(bd, |r| r.min(bd)));
                }
                best = Some((d, cand));
            }
        }
    }

    let (bd, bw) = best?;
    // The runner-up has to be clearly worse. Two candidates the same distance
    // away means the evidence does not pick one, so neither is used.
    if runner.is_some_and(|r| r <= bd) {
        return None;
    }
    Some(bw)
}

/// Split a procedural word that has run into the word after it.
///
/// `DE` and `CQ` are followed by a callsign, and a character gap heard as an
/// element gap glues them together — `DENK9G` for `DE NK9G`. The split is only
/// taken when the tail is callsign-shaped, so it cannot manufacture a spot out
/// of a long mis-copy.
pub fn split_prefix(word: &str) -> Option<(&'static str, &str)> {
    const PREFIXES: [&str; 2] = ["DE", "CQ"];
    for p in PREFIXES {
        if let Some(rest) = word.strip_prefix(p) {
            if rest.len() >= 3 && is_callsign(rest) {
                return Some((p, rest));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn syms(word: &str) -> Vec<Sym> {
        word.chars()
            .map(|c| Sym {
                ch: c,
                elems: morse_elements(c).unwrap_or_default().to_string(),
            })
            .collect()
    }

    #[test]
    fn one_element_slips_are_corrected() {
        // G is --. against Q's --.-, so CG is one element short of CQ.
        assert_eq!(rescore_word(&syms("CG")), Some("CQ"));
        assert_eq!(rescore_word(&syms("CO")), Some("CQ"));
        // L is .-.. against D's -.., one element out either way.
        assert_eq!(rescore_word(&syms("LE")), Some("DE"));
        assert_eq!(rescore_word(&syms("NEST")), Some("TEST"));
    }

    #[test]
    fn equidistant_candidates_are_refused() {
        // CX sits one element from both CQ and TNX. The evidence names
        // neither, so the decode stands rather than a coin being tossed.
        assert_eq!(rescore_word(&syms("CX")), None);
    }

    #[test]
    fn real_words_are_left_alone() {
        assert_eq!(rescore_word(&syms("CQ")), None);
        assert_eq!(rescore_word(&syms("TEST")), None);
        assert_eq!(rescore_word(&syms("QTH")), None);
    }

    #[test]
    fn callsigns_are_never_rewritten() {
        for call in ["W1AW", "NK9G", "ZW5B", "PT6T", "K1ABC", "2E0ABC"] {
            assert_eq!(rescore_word(&syms(call)), None, "rewrote {call}");
        }
    }

    /// The regression that shaped `CORRECT_TO`.
    ///
    /// Both of these are *correctly decoded* words that an earlier, broader
    /// correction target rewrote: `CT` in `NEWINGTON CT` sits one element from
    /// the procedural `BT`, and `THE` is two from `TEST`. Together they cost
    /// seven cells of `bench_cw_score` a clean 100 %.
    #[test]
    fn correct_copy_is_not_rewritten() {
        assert_eq!(rescore_word(&syms("CT")), None, "CT rewritten");
        assert_eq!(rescore_word(&syms("THE")), None, "THE rewritten");
        for w in ["JOE", "AND", "FOR", "WAS", "HIS", "CAR", "DOG"] {
            assert_eq!(rescore_word(&syms(w)), None, "rewrote {w}");
        }
    }

    #[test]
    fn distant_words_are_left_alone() {
        // Nothing in the lexicon is one or two elements from these.
        for junk in ["XZQJ", "MMMM", "EEEEE"] {
            assert_eq!(rescore_word(&syms(junk)), None, "invented from {junk}");
        }
    }

    #[test]
    fn ambiguous_matches_are_refused() {
        // GM and GN are one element apart (--. -- against --. -.), so a word
        // equidistant from both must not be resolved to either.
        let d_gm = elem_distance(&elems_of_str("GM").unwrap(), &elems_of_str("GN").unwrap());
        assert_eq!(d_gm, 1, "GM and GN should be one element apart");
    }

    #[test]
    fn a_glued_call_is_split() {
        assert_eq!(split_prefix("DENK9G"), Some(("DE", "NK9G")));
        assert_eq!(split_prefix("CQW1AW"), Some(("CQ", "W1AW")));
        // Not callsign-shaped: leave it.
        assert_eq!(split_prefix("DEEEEE"), None);
        assert_eq!(split_prefix("TEST"), None);
    }

    #[test]
    fn element_distance_counts_gap_errors() {
        // Two letters run together is one deletion: the separator.
        let two = elems_of_syms(&syms("EE"));
        let one = "..".to_string();
        assert_eq!(elem_distance(&two, &one), 1);
    }
}
