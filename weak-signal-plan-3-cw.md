# CW plan, round 3 — the band, not the bench

**Status:** Stage 0 complete. Stage 1 attempted five ways and rejected; none
of it is in the tree. Stage 2 (the HSMM in §5) is **on `main` and has not
met §6.** Section 7 is the handback from that attempt.

**This document is a handoff.** Sections 1–4 are context; section 5 is the
specification Stage 2 followed; section 6 is still how to know you are done;
section 7 is what the first implementation measured.

---

## 0. Orientation

The CW decoder is `src/decoders/cw.rs` plus `src/decoders/cw_hsmm.rs`.
Front-end (unchanged) and Stage 2 element decoder:

```
IQ in → mix wanted tone to DC → 4-pole narrow LPF → magnitude → smooth
      → decimate to 1 kHz → matched boxcar (0.22 dit)
      → mean-referenced mu_mark / mu_space
      → HSMM Viterbi on a 500 Hz trellis over a 5–50 WPM dit-period grid
      → committed mark/dit-dah / gap segmentation → morse_lookup → text
```

`CwDecoder::wpm` and `CwView::dit_ms` come from the winning grid period.
`confidence()` is the best-path score against an all-space null, not mark-bucket
fit. The tone search and lock (`search`, `score_cw`) sit alongside and were
not in scope for Stage 2.

Stage 0, replaced on `main` by the HSMM, was the hard-decision chain:

```
… → matched boxcar (0.35 dit) → hysteresis slicer (step_envelope)
  → on_mark_end / on_space_end → classify_mark against a tracked clock
  → morse_lookup
```

### Commands

```bash
cargo test --release
```

```bash
cargo test --release --lib -- --ignored --nocapture bench_cw_band
```

```bash
cargo test --release --lib -- --ignored --nocapture bench_cw_score
```

```bash
cargo test --release --test cw_capture -- --nocapture
```

The live capture (`captures/20m_14060khz_192ksps_60s.iq`, 92 MB) is
**gitignored**. Tests that need it skip cleanly when it is absent, so a fresh
clone will pass with two capture tests silently skipped. Get the file from the
machine that recorded it before trusting any real-band number.

### The three instruments

| Instrument | What it measures | Stage 0 | Stage 2 (now on `main`) |
|---|---|---|---|
| `bench_cw_score` (41 cells) | flat carrier, AWGN — the laboratory | **90.22 %** | **84.05 %** |
| `bench_cw_band` (16 cells) | Watterson channel, QRM, static — the band | **43.54 %** | **41.84 %** |
| `tests/cw_capture.rs` | token recall on the real 20m recording | **81.9 %** | not re-run |

Both grids average every cell over four noise seeds. Single-trial cells near
the copy threshold swing by tens of points on which noise burst lands in which
dah, and tuning against that measures the seed.

Gates that will fail a bad change: `cw_score_does_not_regress` (88 %),
`cw_band_score_does_not_regress` (38 %), `cw_recovers_known_stations_from_the_live_capture`
(55 % recall), `the_clean_station_is_copied_nearly_verbatim`.

---

## 1. The problem, with evidence

**The decoder is fade-limited, not noise-limited.** This is the single fact
that matters and it inverts the assumption the previous two rounds were built
on.

From `bench_cw_band`:

```
  chan flat 12dB           100.0%      <- what the old benchmark measured
  chan good 12dB            87.2%
  chan moderate 12dB        45.7%
  chan poor 12dB            26.6%
  chan flutter 12dB          1.4%
  moderate 20dB             48.6%
  moderate 6dB              40.1%
  qrm x3 moderate           25.2%
  the band, 10dB            15.3%
  empty, qrm only          100.0%
```

`moderate 20dB` = 48.6 % against `moderate 6dB` = 40.1 %. **Fourteen decibels
of extra signal buys eight points.** More signal does not help, because the
detector cannot track the level it is given.

Corroborated on the recording: across the seven candidate frequencies a single
station's mark level moves by 12–31 dB inside one 60-second over, and which
stations get copied tracks *that* number, not their SNR. The one copied cleanly
fades by 15 dB; the two that produce nonsense fade by 30. Every passband holds
two to four tones, never one.

`bench_cw_fading` isolates the mechanism on a bare sinusoidal fade. At 15 dB
SNR — comfortable, 100 % copy when flat — 20 dB of QSB takes copy to 42 %.

The cause is in `step_envelope`:

```rust
let on_thr = self.floor + ON_SPAN_K * span;   // span = peak - floor
```

`peak` attacks instantly and decays over 2.5 s; `floor` attacks over 2 s. They
are deliberately slow so keying cannot drag them — but QSB moves on the same
timescale as the keying they are trying to ignore. Fading down, the threshold
strands above the signal and characters vanish; fading up, it sits far below
and noise walks in.

### Two prior rounds of tuning were measuring the wrong thing

Round 3 tuning (slicer thresholds, `POST_MIX_MIN_HZ`, `RECLUSTER_MARKS`, word
gap) moved the flat grid 86.83 → 90.22 and the band grid by under a point.
Before that, a harness bug capped every flat figure at 90 % at *every* SNR
including 40 dB — the generator's trailing silence was shorter than the
decoder's six-dit flush plus the ~1.2 k samples the overlap-save tuning chain
holds back, so every table was reporting the same dropped final `K` as a 10 %
loss. Both are fixed. Do not re-derive them.

---

## 2. What is in the tree (Stage 0, complete)

- **`src/decoders/channel.rs`** — Watterson two-path HF channel, CCIR 520
  `CCIR_GOOD` / `CCIR_MODERATE` / `CCIR_POOR` / `CCIR_FLUTTER` / `FLAT`, plus
  `static_crashes` for impulsive noise. Three self-tests prove it is physically
  right rather than merely plausible: mean power preserved across an ensemble,
  fades deepen as conditions worsen, and tones 300 Hz apart fade independently
  (r < 0.8) — which is why it is two-path with a delay rather than one
  multiplicative envelope. RTTY's ad-hoc `gen_rtty_faded` and PSK31 should both
  move onto this module; neither has real fading today.
- **`bench_cw_band`** + `cw_band_score_does_not_regress` — the band grid.
- **`bench_cw_fading`** — fade depth against copy, at fixed SNR.
- **`tests/cw_capture.rs`** — token recall against the recording, with a
  tokens-per-100-characters density column so recall cannot be bought by
  emitting more letters, plus a canary test pinning the one clean station.
- Multi-seed averaging and `gen_cw_seed` / `key_to_iq_seed` in the test module.

### Ground truth caveat

There is no transcript of the recording. `cw_capture.rs` scores only tokens
that can be confirmed *without trusting the decoder*: structurally valid
callsigns and standard procedure that recur identically across the over, each
checked to survive changes to the decoder's parameters. Two of the seven
candidate frequencies produce nothing recognisable and so contribute nothing.
That is a blind spot in the safe direction — the metric can show a redesign
working and cannot show one failing quietly. Do not add tokens to it by reading
them off the current decoder's output.

---

## 3. What has been tried and rejected — do not repeat these

Five independent detector designs, each swept properly on both grids. **Every
one that materially improves fading performance costs flat-grid score and fails
the same two tests — 35 WPM copy and speed tracking.**

| Attempt | band | flat | Verdict |
|---|---|---|---|
| baseline (in tree) | 43.54 | 90.22 | — |
| Two-state Viterbi, geometric dwell (round 2's deferred item) | — | 80.5 | rejected, −10 |
| Fast mark/space level tracking, SNR-adaptive threshold | **70.06** | 86.60 | 3–9 tests fail |
| Coherence on the post-mix signal | 7.2 | — | rejected outright |
| Coherence on the 400 Hz signal | 67.3 | 73.0 | 2–3 tests fail |
| Hybrid amplitude + coherence | 65 → 48 | 76 → 88 | 2–3 tests fail everywhere |
| fldigi ratio-pair clock, replacing ours | 46.93 | 80.53 | worse estimator |
| fldigi ratio-pair clock, as watchdog | 43.2 | 88.4–89.2 | no effect |
| fldigi SOM character matcher | 43.68 | 90.39 | neutral |
| fldigi mean-referenced thresholds | **59.15** | 82.66 | 5–10 tests fail |

Details worth carrying forward:

- **Geometric dwell is the wrong model.** A two-state Viterbi whose transition
  cost is a constant per sample says leaving a state is equally likely at every
  sample. Morse durations are strongly bimodal at 1 and 3 dits, so a geometric
  prior actively encourages short marks — the exact failure it was meant to
  fix. `weak-signal-plan-2-cw-viterbi.patch` in this repo implements it.
  **It is measured and rejected. Do not apply it.**
- **Coherence is not free processing gain.** Coherent integration over T is
  equivalent to filtering to ~1/T, so it buys the same time-bandwidth product
  the post-mix filter already has. Computed after the 60 Hz post-mix filter it
  is worthless — the filter leaves only ~1.3 independent noise samples in a
  0.35-dit window, so filtered noise reads ~0.8 coherent. Its only real
  advantage is scale invariance.
- **Level estimates are not wrong as estimates**, only as inputs to a threshold
  inside a loop. Reuse them in Stage 2's emission model.
- **fldigi's mean-referenced threshold is the one idea worth stealing.** It
  refers both thresholds to a running *mean* of the envelope rather than a
  max-hold peak; a mean follows a fade smoothly where a max-hold stays stuck at
  the pre-fade level. Largest single band gain of anything tested (+15.6).
- **fldigi does not solve speed-tracking stability either.** It clamps receive
  speed to a user-set window around a user-set WPM and offers a switch to
  freeze tracking entirely. Bounding, not solving.

### Why they all fail the same way

The detector and the clock are coupled. The detector needs the clock to size
its window and time constants; the clock is measured from the detector's own
output. Making the detector more adaptive tightens that loop until it
oscillates — the same shape as the two spirals found earlier this round
(`POST_MIX_MIN_HZ`, and the 15→32 WPM trap that `RECLUSTER_MARKS` fixed).

**Do not attempt a sixth detector in isolation.** Five designs from two
codebases trace one frontier. Detection and timing have to be solved together.

---

## 4. Where the ceiling is

The rejected level-tracking attempt reached **band 70.06 %**. That is evidence
the available gain is real and roughly 27 points, and that Stage 2's job is to
reach it *without* the timing damage — not to find some smaller safe increment.

---

## 5. Stage 2 — specification

Solve detection and timing jointly with an explicit-duration HMM (hidden
semi-Markov model), decoded over a grid of candidate dit periods. **The grid
is the point:** it replaces the feedback loop with a search, and a loop that
does not exist cannot oscillate. A 35 WPM fist is simply the period that scores
best.

### 5.1 States and grammar

Five element states, each with a nominal length in dit units `k_s`:

| State | `k_s` | Kind |
|---|---|---|
| `MarkDit` | 1 | mark |
| `MarkDah` | 3 | mark |
| `GapElement` | 1 | space, within a character |
| `GapChar` | 3 | space, between characters |
| `GapWord` | 7 | space, between words |

Legal transitions:

```
MarkDit, MarkDah  →  GapElement | GapChar | GapWord
GapElement        →  MarkDit | MarkDah          (same character continues)
GapChar           →  MarkDit | MarkDah          (new character)
GapWord           →  MarkDit | MarkDah          (new word)
```

Transition log-probabilities should be mild — the duration model carries the
information. A useful prior: make `GapElement` cheaper than `GapChar`, and
`GapChar` cheaper than `GapWord`, in rough proportion to how often each occurs
in real text. Do not make them so strong that they override the durations.

A sixth `Idle` state is required for the arbitrarily long silence between
overs; give it a geometric (open-ended) duration rather than a Gaussian, and
allow `Idle → MarkDit | MarkDah` and `Mark* → Idle`.

Cap elements per character at 7 (the longest Morse codeword). Either carry a
small counter in the state, or leave it to the character decoder and let
overlong runs decode as garbage — the simpler option, and the current code
already resets on more than 6 elements.

### 5.2 Duration model

For state `s` and candidate dit period `T` (in envelope samples), a segment of
duration `d`:

```
log p(d | s, T) = -(d - k_s·T)² / (2·σ_s²) - ln σ_s,      σ_s = j · k_s · T
```

`j` is fractional timing jitter. Start at `j ≈ 0.25`; a keyer is tighter and a
straight key looser, and this is the knob that decides how much of a sloppy
fist is tolerated. Restrict the duration search per state to
`d ∈ [0.4·k_s·T, 2.2·k_s·T]` — outside that the Gaussian contributes nothing
and the search cost is wasted. `GapWord` should be one-sided (arbitrarily long
is fine, arbitrarily short is not) or handed to `Idle`.

**Do not use a geometric duration.** See section 3.

### 5.3 Observation model

Per envelope sample `t`, two log-likelihoods:

```
L_mark(t)  = log p(env[t] | mark)
L_space(t) = log p(env[t] | space)
```

The envelope of noise alone is Rayleigh; signal plus noise is Rician. Gaussian
approximations are acceptable to start, but the levels must be right:

- `mu_space(t)` — the noise floor. **Stable**: the band's noise does not fade
  with the signal, so this can be tracked slowly and robustly.
- `mu_mark(t)` — the signal level. **Fades**, by 12–31 dB within an over, and
  must follow.

Estimate both with fldigi's structure (section 3): refer them to a running
*mean* of the envelope rather than a max-hold peak. This is the one borrowing
that measured clearly positive.

Then — and this is what makes it safe where Stage 1 was not — **re-estimate the
levels from the decoded segmentation and iterate**. One or two EM-style passes:
decode with the current levels, take the mark segments the HSMM chose, recompute
`mu_mark` from them, decode again. The levels are then conditioned on a globally
optimal segmentation rather than on an instantaneous threshold, which is
precisely the loop that broke every Stage 1 attempt.

Precompute cumulative sums `C_mark[t] = Σ_{i<t} L_mark(i)` and `C_space[t]` so
any segment's emission cost is an O(1) difference. These are shared across every
candidate `T`, so compute them once.

### 5.4 Decoding

Standard HSMM Viterbi:

```
δ[t][s] = max over d, over s' ≠ s of
            δ[t-d][s'] + log A[s'][s] + log p(d | s, T) + emit(t-d, t, s)
```

The inner maximisation over `s'` can be hoisted: precompute
`best_in[t][s] = max_{s'≠s} (δ[t][s'] + log A[s'][s])` once per `t`.

**Cost.** Naively O(T·S²·D), which is far too slow over a whole over. With the
hoist and the duration restriction in 5.2, cost is `Σ_t Σ_s D_s`. At the 1 kHz
envelope rate with `T` = 60 samples (20 WPM), `Σ_s D_s ≈ 900`, so a 3-second
window is ~2.7 M operations. Run it over the whole 60 s recording and the
period grid and it stays inside a few hundred times real time — the current
decoder runs at ~2500×, so expect to give some of that back. That is
acceptable; correctness first, and section 5.6 lists the levers if it is not.

Run on a **sliding window** of 2–4 seconds with overlap, committing the
decoding in the middle and carrying the survivor state across. Do not attempt
to decode a whole over in one pass; latency and memory both suffer.

### 5.5 The period grid

Evaluate the trellis for each candidate `T` and take the highest total
log-likelihood. Grid: geometric from 5 to 50 WPM (matching `DIT_MIN_S` /
`DIT_MAX_S`), spacing ~7 %, which is about 24 points.

Use coarse-to-fine: a coarse grid first, then refine around the winner. And
re-estimate `T` on a schedule — every window or two — rather than continuously;
between re-estimations, search only a narrow local grid around the incumbent.
This is where most of the cost goes and most of it is avoidable.

**Report the winning `T` as the WPM estimate.** `CwDecoder::wpm` and
`CwView::dit_ms` come from it.

### 5.6 If it is too slow

In order of preference: coarse-to-fine period search; re-estimate `T` less
often; decimate the envelope to 500 Hz for the trellis only (a 50 WPM dit is
still 12 samples); prune duration candidates by a cheap first-pass
segmentation; prune states whose `δ` is far below the running best.

### 5.7 Integration

Replace: `step_envelope`'s slicing, `on_mark_end`, `on_space_end`,
`classify_mark`, `recluster`, `morse_clock`, `gaps_match_clock`, and the
warm-up machinery (`warmup`, `warming`, `flush_warmup`, `WARMUP_MARKS`). The
HSMM subsumes all of it — warm-up exists only because the clock had to be
bootstrapped before it could be trusted, and a period grid has no bootstrap.

Keep unchanged: the tone search and lock (`search`, `score_cw`, `scan_span`,
`next_lock`, `nudge_lock`, `hold_tune`), the mixer and `NarrowLpf`, the
`Decoder` trait surface, `CallScanner`, `morse_lookup` and its table, and
`spot_snr`.

Watch out for:

- **`CwView`** feeds a live TUI pane and expects `env`, `keyed`, `on_thr`,
  `off_thr`, `symbol`, `dit_ms`, `wpm`, `quality`. `keyed` becomes the HSMM's
  committed segmentation. `on_thr` / `off_thr` no longer exist as such — supply
  the normalised `mu_mark` / `mu_space` so the pane still shows something
  meaningful, and adjust the pane if not.
- **Latency.** The current decoder is near-real-time; a sliding window adds 1–2
  seconds. Commit with a lag of about one character and flush on a long gap.
  Check the TUI still feels live.
- **`POST_MIX_K` / `POST_MIX_MIN_HZ`** size the post-mix filter from the tracked
  clock — another loop, and the source of one of the two spirals. With a period
  grid the winning `T` is a much safer input, but keep the floor until measured.

### 5.8 Confidence

`confidence()` currently reports 88 % on `"AEEEB EIE E ETI*EHEE SSE"` and 70 %
on a 14000.00 kHz signal that has 2.6 dB of envelope variation and is not keyed
CW at all. It gates the pskreporter spots, so it is not cosmetic.

Replace it with a likelihood ratio: the HSMM's best path score against a
null hypothesis that the whole window is space. That number is meaningful,
bounded, and falls when the decode is nonsense.

---

## 6. Acceptance criteria

**Required.** Nothing merges without these:

- `cargo test --release` green, including `cw_tracks_speed`,
  `cw_follows_a_speed_change`, `cw_starts_clean_at_any_speed`,
  `cw_copies_at_zero_db`, `cw_rejects_an_adjacent_station`. These three
  speed/fast-fist tests are the canaries that caught all five rejected
  attempts — if they fail, the loop is still there.
- `bench_cw_band` ≥ **60 %** (from 43.54). Below that Stage 2 has not earned
  its complexity; the rejected level-tracking hack reached 70 without any of it.
- `bench_cw_score` ≥ **88 %** (from 90.22). A small give-back on the laboratory
  grid is acceptable and expected; a large one means detection got worse, not
  more robust.
- `cw_capture` token recall ≥ **82 %** (from 81.9) with density not falling on
  any station. Recall rising while density falls means it is inventing letters.
- **No regression on noise rejection**: `empty, qrm only` and the three
  flat-grid `noise *` cells stay at 100 %. An HSMM will cheerfully explain
  noise as Morse; a likelihood gate against the null hypothesis is what stops
  it. Copy that cannot be trusted is worse than no copy.

**Target.** What success looks like:

- `chan moderate 12dB` from 45.7 % into the nineties.
- `moderate 20dB` pulling clearly above `moderate 6dB` — the signature of a
  decoder that is noise-limited rather than fade-limited. Today they are 48.6
  and 40.1, and closing that gap is the whole point of the exercise.
- `chan flutter 12dB` off 1.4 %. That cell is a collapse, not a degradation.
- `qrm x3 moderate` off 25.2 %.

**Raise the gates** in `cw_score_does_not_regress` and
`cw_band_score_does_not_regress` to just under whatever is achieved. Needing to
lower one is the finding, not the fix.

---

## 7. Handback

Stage 2 as specified in §5 is on `main`. Isolated Viterbi
on a clean
0/1 envelope recovers `PARIS` / `CQ` at the right `T`. The loss is in the
real envelope, windowing, and fade-following, not the grammar.

`captures/20m_baseline_metrics.json` was not touched.

### 7.1 `bench_cw_band` — before (Stage 0) and after (Stage 2)

| Cell | Stage 0 | Stage 2 |
|---|---:|---:|
| chan flat 12dB | 100.0% | 97.3% |
| chan good 12dB | 87.2% | 84.9% |
| chan moderate 12dB | 45.7% | 45.3% |
| chan poor 12dB | 26.6% | 21.4% |
| chan flutter 12dB | 1.4% | 0.0% |
| moderate 20dB | 48.6% | 49.3% |
| moderate 6dB | 40.1% | 41.2% |
| qrm x1 moderate | — | 45.3% |
| qrm x2 moderate | — | 34.0% |
| qrm x3 moderate | 25.2% | 25.9% |
| qrm x3 poor | — | 13.7% |
| crashes 2/s moderate | — | 43.9% |
| crashes 6/s moderate | — | 43.9% |
| the band, 10dB | 15.3% | 9.0% |
| the band, 4dB | — | 14.2% |
| empty, qrm only | 100.0% | 100.0% |
| **mean** | **43.54%** | **41.84%** |

Dashes are cells the Stage 0 write-up did not quote. The mean is over all
16 cells. Fourteen extra dB on the moderate path still buy almost nothing
(49.3 vs 41.2). Stage 2 has not earned its complexity.

### 7.2 `bench_cw_score`

Stage 0 mean **90.22%**. Stage 2 mean **84.05%** (gate 88%). Full 41-cell
listing was not printed; worst cells:

| Cell | Stage 2 |
|---|---:|
| short 12wpm −3dB | 0% |
| short 18wpm −3dB | 0% |
| short 25wpm −3dB | 4% |
| short 35wpm −3dB | 8% |
| short 35wpm +0dB | 48% |
| short 12wpm +0dB | 50% |

The give-back is at the noise wall, not a small robustness tax.

### 7.3 `cw_capture`

Not re-run. Stage 0 token recall remains the last measured figure (81.9%).

### 7.4 Acceptance

| Criterion | Need | Stage 2 | Met |
|---|---|---|---|
| Speed / adjacent / clean-start canaries | green | pass | yes |
| `cw_copies_at_zero_db` | ≥ 60% at 0 dB | ~25% | **no** |
| `cargo test --release` | green | 0 dB + 88% flat gate fail | **no** |
| `bench_cw_band` | ≥ 60% | 41.84% | **no** |
| `bench_cw_score` | ≥ 88% | 84.05% | **no** |
| `cw_capture` recall / density | ≥ 82%, density not down | not run | unknown |
| `empty, qrm only` | 100% | 100% | yes |
| Flat `noise *` cells | 100% | not re-listed | unverified |
| Raise `cw_*_does_not_regress` gates | just under new scores | not done | n/a — scores did not earn it |

### 7.5 Tried on the Stage 2 branch (do not repeat)

| Attempt | Effect | Verdict |
|---|---|---|
| Incremental window commit | First mark of each character dropped (`W`→`M`, `C`→`R`) | keep marks that straddle the commit point |
| Period limits compared at 1 kHz vs 500 Hz trellis | Anything faster than ~25 WPM discarded | compare limits at the trellis rate |
| 3× period (dits read as dahs) | 15 WPM locked at 46 WPM | penalize `T < 0.55 ×` mark-length hint |
| Flush-on-every-chunk of lead-in | Locked 50 WPM on silence | wait ~1.15 s of envelope before first decode |
| Word-gap threshold 3.9 T | Extra spaces (`W1 A W`) | 4.6 T |
| Full-buffer idle-only emit | Clean copy; 0 dB never flushed | back to incremental flush |
| Require tone lock to emit | Noise quiet; 0 dB → 0% | lock **or** mark/space contrast ≥ 2.4 |
| Skip EM for speed | Band 40.84% | one same-`T` EM pass when contrast < 6 → 41.84% |

### 7.6 Throughput

Not re-measured. Stage 0 CW row of `bench_replay` was ~2500×. The Stage 2
trellis is 500 Hz with a ~2.5 s window; expect to give some of that back.
Do not quote 2500× as a Stage 2 number.

### 7.7 What to do next

1. Make `mu_mark(t)` follow QSB on a ~50–80 ms time constant *inside* the
   window — the causal mean is still too sticky on the way down.
2. Two full EM decode passes on fading cells, not a single same-`T` retry.
3. Drive `POST_MIX_*` from grid `T` earlier so 0 dB sees the narrow filter.
4. Re-run `cw_capture` and `bench_replay` before any merge.

---

## 8. Later stages

**Stage 3 — lexicon rescoring.** Once the HSMM emits soft character hypotheses,
rescore against ham CW's very small language: CQ, DE, TEST, 599/5NN, TU, 73,
QTH, and callsigns with real prefix-digit-suffix structure. `"CE TEST T EWH I
ZW B"` becomes `"CQ TEST DE ZW5B"` because the alternative is not a word. This
is the largest remaining real-world gain after Stage 2 and it is also where
fldigi's SOM matcher becomes worth revisiting — neutral on its own because it
turns `*` into a confident wrong guess, useful once a lexicon can adjudicate
between candidates instead of the matcher having to commit.

**Stage 4 — decode every tone.** Two to four stations sit in every 400 Hz
passband and the decoder picks one and discards the rest.

**Not CW, but owed:** `gen_rtty_faded` fakes fading with static per-tone
amplitudes and PSK31 has none at all. Both should move onto
`decoders::channel`, and both are probably hiding the same class of problem
this round found in CW.
