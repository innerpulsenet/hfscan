//! Recognising the transmitting station in a stream of decoded characters, so
//! the character-at-a-time modes can be reported to pskreporter.info.
//!
//! FT8 and FT4 hand back structured messages with the sender already packed
//! into a known field; CW, RTTY and PSK31 hand back prose. What they do have is
//! convention, and the same one whether the keying is Morse, Baudot or BPSK, so
//! one scanner serves all three. Three forms name the transmitting station:
//!
//! - `CQ <call>`, including the contest and activity variants.
//! - `DE <call> <call>`, the classic idle.
//! - `<addressee> <call>`, the exchange — two callsigns running together, where
//!   the second is the sender. This is the same "to, from" ordering FT8 packs
//!   into its message fields and that `report::sender_of` unpacks, and on CW
//!   and RTTY it is the commonest form of all: every contest exchange and every
//!   turn of an ordinary QSO is one.
//!
//! The exchange form is the one that can name the wrong station, so it is the
//! one hedged. An addressee repeated before the sender — `W1AW W1AW K1ABC` —
//! must not be mistaken for the pair, and `is_plausible_call` is stricter than
//! the general heuristic because there is no `CQ` or `DE` here to establish
//! that a callsign is what was meant: without it, the `5NN` of a CW signal
//! report parses as one.

use crate::report::is_callsign;
use std::time::{SystemTime, UNIX_EPOCH};

/// Longest word worth carrying. Nothing in a callsign announcement runs this
/// long, so anything that does is mis-copy — and carrying it forward only
/// mis-frames the words after it.
const MAX_WORD: usize = 16;
/// Recognised calls held between drains. The decoders drain every block, so
/// this only ever bounds a pathological stream.
const MAX_PENDING: usize = 32;
/// Words that may stand between a `CQ` and the callsign it belongs to.
///
/// A contest or activity name goes there — "CQ TEST", "CQ CONTEST", "CQ POTA",
/// "CQ FIELD DAY" — so giving up on the first word that is not a callsign
/// would miss most of a contest weekend. But the wait has to end somewhere: a
/// `CQ` heard minutes ago must not adopt whatever callsign turns up next.
const CQ_SKIPS: u8 = 4;

/// Whether a word is shaped like an amateur callsign, rather than merely
/// containing letters and digits the way `report::is_callsign` asks.
///
/// Two rules do the work, and both come from how callsigns are built: a
/// separating digit follows the prefix, so it is never the first character;
/// and the suffix is letters, so a call never ends on a digit. That is enough
/// to reject what a contest exchange is otherwise full of — `5NN` for a 599
/// report, `TT1` and `001` for serial numbers — while `2E0ABC`, `4X4AB`,
/// `9A1CD`, `GB100MCM` and `K1ABC/P` all still pass.
///
/// Only the exchange form needs this. `CQ` and `DE` say a callsign is coming,
/// so there the looser test is the right one: it accepts calls this would turn
/// away, and the keyword has already ruled out the numbers.
fn is_plausible_call(w: &str) -> bool {
    if !is_callsign(w) {
        return false;
    }
    let b = w.as_bytes();
    let stem = w.split('/').next().unwrap_or(w).as_bytes();
    b.last().is_some_and(|c| c.is_ascii_alphabetic())
        && stem.iter().skip(1).any(|c| c.is_ascii_digit())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Hear {
    Idle,
    AfterDe,
    AfterDeCall,
    /// Waiting for the caller's own callsign, having skipped `skips` words.
    AfterCq(u8),
    /// One callsign heard with no keyword in front of it. If another follows,
    /// the first was the addressee and the second is the sender.
    AfterCall,
}

pub struct CallScanner {
    word: String,
    hear: Hear,
    /// Call from the most recent `DE`, kept so an operator who sends it only
    /// once is still spotted.
    last: String,
    /// The station being called, while `hear` is `AfterCall`. Held so that an
    /// operator who repeats it — "W1AW W1AW K1ABC" — is not read as the pair.
    addressee: String,
    calls: Vec<String>,
}

impl CallScanner {
    pub fn new() -> Self {
        Self {
            word: String::new(),
            hear: Hear::Idle,
            last: String::new(),
            addressee: String::new(),
            calls: Vec::new(),
        }
    }

    /// Feed one decoded character. Any whitespace ends the current word.
    pub fn push(&mut self, c: char) {
        if c.is_ascii_whitespace() {
            self.flush_word();
            return;
        }
        if c.is_ascii_graphic() {
            self.word.push(c);
            if self.word.len() > MAX_WORD {
                self.word.clear();
                self.hear = Hear::Idle;
            }
        }
    }

    /// Drain the calls recognised since the last call. The caller turns them
    /// into spots, since only it knows the frequency and signal report.
    pub fn take_calls(&mut self) -> Vec<String> {
        std::mem::take(&mut self.calls)
    }

    /// Forget the partial word and where we were in a transmission. Used when
    /// the decoder loses its signal — what follows is a different station, and
    /// half of one announcement joined to half of another is a phantom call.
    pub fn reset(&mut self) {
        self.word.clear();
        self.hear = Hear::Idle;
        self.last.clear();
        self.addressee.clear();
        self.calls.clear();
    }

    fn flush_word(&mut self) {
        if self.word.is_empty() {
            return;
        }
        let w = self.word.to_ascii_uppercase();
        self.word.clear();
        match self.hear {
            Hear::Idle => {
                if w == "DE" {
                    self.hear = Hear::AfterDe;
                } else if w == "CQ" {
                    self.hear = Hear::AfterCq(0);
                } else if is_plausible_call(&w) {
                    self.addressee = w;
                    self.hear = Hear::AfterCall;
                }
            }
            Hear::AfterCall => {
                if w == "DE" {
                    // "W1AW DE K1ABC" — the keyword is the better evidence.
                    self.hear = Hear::AfterDe;
                } else if w == "CQ" {
                    self.hear = Hear::AfterCq(0);
                } else if w == self.addressee {
                    // The station being called, sent twice. Still waiting.
                } else if is_plausible_call(&w) {
                    self.emit(w);
                    self.hear = Hear::Idle;
                } else {
                    self.hear = Hear::Idle;
                }
            }
            Hear::AfterDe => {
                if is_callsign(&w) {
                    self.last = w;
                    self.hear = Hear::AfterDeCall;
                } else if w != "DE" {
                    self.hear = Hear::Idle;
                }
            }
            Hear::AfterDeCall => {
                // Classic idle: "DE CALL CALL". A single "DE CALL" is accepted
                // as well — plenty of operators send it only once.
                let call = if is_callsign(&w) {
                    w
                } else {
                    self.last.clone()
                };
                self.emit(call);
                self.hear = Hear::Idle;
            }
            Hear::AfterCq(skips) => {
                if is_callsign(&w) {
                    self.emit(w);
                    self.hear = Hear::Idle;
                } else if w == "CQ" || w == "DE" {
                    // The idiom repeats itself — "CQ CQ CQ DE K1ABC" — and a
                    // station that calls for a while is not drifting off
                    // topic, so its own words cost nothing.
                } else if w.chars().all(|c| c.is_ascii_alphabetic()) && skips < CQ_SKIPS {
                    self.hear = Hear::AfterCq(skips + 1);
                } else {
                    // A number or a symbol means this is an exchange, not a
                    // call: whoever is named here is being worked, not calling.
                    self.hear = Hear::Idle;
                }
            }
        }
    }

    fn emit(&mut self, call: String) {
        if call.is_empty() || self.calls.contains(&call) {
            return;
        }
        self.last = call.clone();
        if self.calls.len() < MAX_PENDING {
            self.calls.push(call);
        }
    }
}

pub fn utc_hhmmss() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let s = secs % 86400;
    format!("{:02}{:02}{:02}", s / 3600, (s % 3600) / 60, s % 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(text: &str) -> Vec<String> {
        let mut s = CallScanner::new();
        for c in text.chars() {
            s.push(c);
        }
        s.take_calls()
    }

    #[test]
    fn spots_the_calling_station() {
        // The two idioms, as sent on CW and RTTY.
        assert_eq!(scan("CQ CQ CQ DE K1ABC K1ABC K "), ["K1ABC"]);
        assert_eq!(scan("CQ DX JA1ABC JA1ABC "), ["JA1ABC"]);
        // Contest and activity CQs, which is most of what a busy band carries.
        assert_eq!(scan("CQ TEST W9XYZ W9XYZ "), ["W9XYZ"]);
        assert_eq!(scan("CQ CONTEST DE VE3ABC VE3ABC "), ["VE3ABC"]);
        assert_eq!(scan("CQ POTA DE N0CAL N0CAL "), ["N0CAL"]);
        assert_eq!(scan("CQ FIELD DAY DE W1AW W1AW "), ["W1AW"]);
        // A long string of CQs is a patient operator, not a lost thread.
        assert_eq!(scan("CQ CQ CQ CQ CQ CQ CQ DE G4XYZ G4XYZ "), ["G4XYZ"]);
        // Answering a call: the sender is the station after DE, not before it.
        assert_eq!(scan("K1ABC DE W9XYZ W9XYZ KN "), ["W9XYZ"]);
        // ...and with no DE at all, the second of the pair is still the sender:
        // "to, from", the same ordering FT8 packs into its message fields.
        assert_eq!(scan("W9XYZ K1ABC 599 001 "), ["K1ABC"]);
        assert_eq!(scan("K1ABC 5NN 001 W9XYZ VE3ABC TU "), ["VE3ABC"]);
        // An addressee sent twice before the sender is not the pair.
        assert_eq!(scan("W1AW W1AW K1ABC K1ABC "), ["K1ABC"]);
        // DE sent once, followed by anything.
        assert_eq!(scan("DE G4XYZ PSE K "), ["G4XYZ"]);
        // Portable calls survive the slash.
        assert_eq!(scan("CQ CQ DE K1ABC/P K1ABC/P "), ["K1ABC/P"]);
    }

    /// The exchange form has no keyword in front of it, so the only thing
    /// keeping a signal report or a serial number out of the callsign slot is
    /// the shape of the word itself.
    #[test]
    fn contest_shorthand_is_not_a_callsign() {
        // 5NN is 599 sent the way CW contesters send it, and it satisfies the
        // general heuristic: letters, a digit, three characters long.
        assert!(is_callsign("5NN"), "premise of this test has changed");
        assert!(!is_plausible_call("5NN"));
        // Serial numbers, cut or otherwise. A callsign never ends on a digit.
        for w in ["TT1", "001", "5NN001", "AB7"] {
            assert!(!is_plausible_call(w), "{w} should not read as a callsign");
        }
        // Real calls, including the awkward shapes, all still pass.
        for w in [
            "K1ABC",
            "W1AW",
            "G4XYZ",
            "JA1ABC",
            "VE3ABC",
            "N0CAL",
            "2E0ABC",
            "4X4AB",
            "9A1CD",
            "GB100MCM",
            "K1ABC/P",
            "VP2E/W1ABC",
        ] {
            assert!(is_plausible_call(w), "{w} should read as a callsign");
        }
        // And the exchange rule uses it: no spot when the second word is a
        // report rather than a station.
        assert!(scan("W1AW 5NN 5NN TU ").is_empty());
        assert!(scan("K1ABC TT1 TT1 ").is_empty());
    }

    #[test]
    fn stays_quiet_when_nobody_identifies() {
        assert!(scan("TU 73 GL OM ").is_empty());
        assert!(scan("CQ CQ CQ ").is_empty());
        // Mis-copy: CW spells an unresolvable symbol '*', which is not a call.
        assert!(scan("CQ DE K1A*C K1A*C ").is_empty());
        // A run-on word is mis-copy too, and must not frame the next one.
        assert!(scan("DE ABCDEFGHIJKLMNOPQRST K1ABC ").is_empty());
        // A CQ does not stay open forever waiting for a call to wander past.
        assert!(scan("CQ AND NOW FOR SOMETHING ELSE ENTIRELY K1ABC ").is_empty());
        // A number after the CQ means we are already into an exchange.
        assert!(scan("CQ 599 K1ABC ").is_empty());
    }

    #[test]
    fn each_call_is_reported_once_per_drain() {
        // The idiom repeats the call for a reason; the spot must not.
        assert_eq!(scan("CQ DE K1ABC K1ABC CQ DE K1ABC K1ABC "), ["K1ABC"]);
        let mut s = CallScanner::new();
        for c in "CQ DE K1ABC K1ABC DE W9XYZ W9XYZ ".chars() {
            s.push(c);
        }
        assert_eq!(s.take_calls(), ["K1ABC", "W9XYZ"]);
        // Drained, so the same station counts as new again — the reporter's
        // own hourly rule is what stops it going out twice.
        for c in "CQ DE K1ABC K1ABC ".chars() {
            s.push(c);
        }
        assert_eq!(s.take_calls(), ["K1ABC"]);
    }

    #[test]
    fn losing_the_signal_drops_a_half_heard_announcement() {
        let mut s = CallScanner::new();
        for c in "CQ CQ DE ".chars() {
            s.push(c);
        }
        s.reset();
        // "K1ABC" now belongs to whatever came next, not to that DE.
        for c in "K1ABC ".chars() {
            s.push(c);
        }
        assert!(s.take_calls().is_empty());
    }
}
