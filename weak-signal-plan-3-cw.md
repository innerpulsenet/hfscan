# CW plan, round 3 — the band, not the bench

**Status:** Stage 0 plus Stage 3 lexicon rescoring (§9) is on `main` and is the
shipping decoder. Stage 1 was
attempted five ways and rejected. **Stage 2 (the HSMM in §5) was built, measured
against every instrument, and reverted on 2026-08-16.** It never met §6, and the
throughput cost was not survivable: 73.5× realtime and 1.36 % CPU became 3.0×
and 33 %, with a latent lockup on top (§7.11). It bought band 43.54 → 49.27 and
capture recall 81.9 → 91.2, and gave back flat 90.22 → 84.72, a green test suite,
and 24× of throughput. That is not a trade worth making for a live 16-channel
scanner.

The HSMM is preserved on branch `cw-stage2-em-salvage` — it is not in this tree.

**This document is a handoff.** Sections 1–4 are context and still stand;
section 5 is the specification Stage 2 followed; section 6 is the acceptance bar
and **§6 is now known to be partly wrong** (see §7.8 — band ≥ 60 is probably
unreachable without Stage 4, and §6 never set a CPU budget, which is the
omission that allowed all of this). Section 7 is everything that was measured,
including three rounds of work that were reverted. **Stage 4 has since
shipped (§8) with a CPU budget attached**; it recovers the multi-station cells
on a throughput metric and leaves the lock-only mean where it was, so §7.8's
arithmetic about reaching 60 still stands.

### If you pick Stage 2 up again

Read §7.10 and §7.11 before writing any code. The two findings that matter:
the period estimator collapses to under half the true dit at −3 dB, and
`GapChar` / `GapWord` have never once fired because the geometric `Idle` state
absorbs every gap. Both are structural and neither is fixed by tuning. Add a
CPU budget to the acceptance criteria before starting.

---

## 0. Orientation

The CW decoder is `src/decoders/cw.rs`. This is the Stage 0 hard-decision
chain, which is what ships:

```
IQ in → mix wanted tone to DC → 4-pole narrow LPF → magnitude → smooth
      → decimate to 1 kHz → matched boxcar (0.35 dit)
      → hysteresis slicer (step_envelope)
      → on_mark_end / on_space_end → classify_mark against a tracked clock
      → morse_lookup → text
```

Stage 2 replaced everything after the boxcar with an HSMM Viterbi on a 500 Hz
trellis over a 5–50 WPM dit-period grid, taking `wpm` from the winning period
and `confidence()` from a likelihood ratio against an all-space null. It was
reverted; see the status block above and §7.

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

Stage 0 plus Stage 3 is what is in the tree. The Stage 2 columns are
historical, kept so a future attempt can tell immediately whether it is beating
the HSMM or repeating it.

| Instrument | What it measures | Stage 0 | Stage 2 | Stage 2 + EM | **In tree (0+3)** |
|---|---|---|---|---|---|
| `bench_cw_score` (41 cells) | flat carrier, AWGN — the laboratory | 90.22 % | 84.58 % | 84.72 % | **90.57 %** |
| `bench_cw_band` (16 cells) | Watterson channel, QRM, static — the band | 43.54 % | 41.84 % | 49.27 % | **44.03 %** |
| `bench_cw_band`, throughput | as above, but copied in *any* stream (Stage 4) | — | — | — | **46.90 %** |
| `tests/cw_capture.rs` | token recall on the real 20m recording | 81.9 % | 77.5 % | 91.2 % | **88.1 %** |
| `bench_replay` end-to-end | realtime multiple, 16 channels | 71.67× | 2.56× | 3.04× | **73.8×** |

Every figure in that table is a re-measurement, taken in one sitting on one
machine with the capture present. Where it disagrees with a number quoted
elsewhere in this document, the table is right.

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
  multiplicative envelope. RTTY and PSK31 are on it too now
  (`bench_rtty_fading`, `bench_psk31_fading`) — see §8 for what that found.
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

Stage 2 as specified in §5 is in the tree, plus one EM refinement (§7.7).
Isolated Viterbi on a clean 0/1 envelope recovers `PARIS` / `CQ` at the right
`T`. The residual loss is in the real envelope, windowing, and fade-following,
not the grammar.

`captures/20m_baseline_metrics.json` was not touched, so the replay baseline is
still genuine Stage 0.

Two rounds of work are recorded here. The first built the HSMM and handed back
honestly, though it left several criteria unmeasured. The second declared the
work complete without running the instruments; it was measured and reverted
(§7.9). Everything below is re-measured, not carried over.

### 7.1 `bench_cw_band`

| Cell | Stage 0 | Stage 2 | Stage 2 + EM |
|---|---:|---:|---:|
| chan flat 12dB | 100.0% | 97.3% | 97.3% |
| chan good 12dB | 87.2% | 84.9% | 87.4% |
| chan moderate 12dB | 45.7% | 45.3% | 56.1% |
| chan poor 12dB | 26.6% | 21.4% | 39.4% |
| chan flutter 12dB | 1.4% | 0.0% | 6.3% |
| moderate 20dB | 48.6% | 49.3% | 59.7% |
| moderate 6dB | 40.1% | 41.2% | 46.2% |
| qrm x1 moderate | — | 45.3% | 56.1% |
| qrm x2 moderate | — | 34.0% | 42.6% |
| qrm x3 moderate | 25.2% | 25.9% | 30.4% |
| qrm x3 poor | — | 13.7% | 27.9% |
| crashes 2/s moderate | — | 43.9% | 55.0% |
| crashes 6/s moderate | — | 43.9% | 54.3% |
| the band, 10dB | 15.3% | 9.0% | 15.1% |
| the band, 4dB | — | 14.2% | 14.4% |
| empty, qrm only | 100.0% | 100.0% | 100.0% |
| **mean** | **43.54%** | **41.84%** | **49.27%** |

Dashes are cells the Stage 0 write-up did not quote. Every cell improved or
held. `chan flutter` is off the floor but still a collapse. The 20 dB / 6 dB
spread widened from 8.1 points to 13.5, which is the §6 target signature —
extra signal is starting to buy copy again — but the decoder is still much
closer to fade-limited than to noise-limited.

### 7.2 `bench_cw_score`

Stage 0 **90.22%**, Stage 2 **84.58%**, Stage 2 + EM **84.52%** (gate 88%).
The EM pass is neutral here by design; the give-back is Stage 2's and it is
concentrated at the noise wall:

| Cell | Stage 0 | Stage 2 + EM |
|---|---:|---:|
| short 12wpm −3dB | — | 0% |
| short 18wpm −3dB | — | 0% |
| short 25wpm −3dB | — | 0% |
| short 35wpm −3dB | — | 6% |
| short 12wpm +0dB | — | 58% |
| short 35wpm +0dB | — | 59% |

All three `noise *` cells are at 100%. Do not merge anything that moves them.

### 7.3 `cw_capture`

Token recall **91.2%**, against 81.9% at Stage 0 and 77.5% at Stage 2, with
per-station density up on all four frequencies (20.5 / 5.4 / 2.4 / 4.9 against
Stage 2's 18.3 / 5.2 / 0.8 / 4.9). Recall and density rising together is the
one combination that is not an artefact of emitting more letters.

`the_clean_station_is_copied_nearly_verbatim` still fails, and has failed since
the HSMM landed. `NK9G` now recurs often enough to clear the first assertion;
what fails is `text.contains("CQ CQ QRP TEST DE NK9G")` — the decoder emits
`DENK9G`, dropping the word gap before the callsign. That is a gap-classifica-
tion bug, not a detection one, and it is the most concrete lead in this file.

### 7.4 Acceptance

| Criterion | Need | Stage 2 + EM | Met |
|---|---|---|---|
| Speed / adjacent / clean-start canaries | green | pass | yes |
| `empty, qrm only` | 100% | 100% | yes |
| Flat `noise *` cells | 100% | 100% | yes |
| `cw_capture` recall / density | ≥ 82%, density not down | 91.2%, density up | yes |
| `bench_cw_band` | ≥ 60% | 49.27% | **no** |
| `bench_cw_score` | ≥ 88% | 84.72% | **no** |
| `cw_copies_at_zero_db` | ≥ 60% at 0 dB | 35% | **no** |
| `the_clean_station_is_copied_nearly_verbatim` | pass | fails on the `DE` word gap | **no** |
| `spot_snr_follows_the_band` | pass | −2.2 dB weak vs 4.6 dB strong | **no** |
| `cargo test --release` | green | 132 pass, 3 fail | **no** |
| Raise `cw_*_does_not_regress` gates | just under new scores | not done | n/a — flat did not earn it |

The three `cargo test` failures are `cw_copies_at_zero_db`,
`cw_score_does_not_regress` and `spot_snr_follows_the_band`. All three predate
the EM pass and fail identically without it. `spot_snr_follows_the_band` went
unmentioned in the first handback; it is a real, open failure.

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
| Skip EM for speed | Band 40.84% | one same-`T` EM pass when contrast < 6 |

### 7.6 Throughput

**This is now the largest unaddressed problem.** Measured with `bench_replay`
against the Stage 0 baseline in `20m_baseline_metrics.json`:

| | CW demod | End-to-end | CPU |
|---|---:|---:|---:|
| Stage 0 baseline | 20.28 MS/s | 71.67× | 1.4% |
| Stage 2 | 0.11 MS/s | 2.56× | 39.1% |
| Stage 2 + EM | 0.08 MS/s | 1.94× | 58.6% |
| **+ §5.6 levers** | **0.13 MS/s** | **3.04×** | **32.9%** |

Two of §5.6's levers are now applied, and both were free — every instrument
came out level or better:

- **Re-estimate `T` on a schedule.** The full coarse grid ran every sixth
  window; it now runs every twenty-fourth (~14 s), with a narrow local grid in
  between, which is what §5.5 asked for in the first place. A station's WPM
  does not drift on a two-second timescale.
- **Prune duration candidates.** The duration search walked up to fifty-six
  probes per state per sample; twenty-eight locate the same Gaussian peak.
  Below about twenty it starts missing the peak on the long states and costs
  real copy (flat 84.4 %, band 48.7 % at eighteen).

That is 1.94× → 3.04× end-to-end and 58.6 % → 32.9 % CPU, with flat and band
both fractionally up and capture recall up 1.2 points. Still 24× short of the
Stage 0 baseline, and the remaining levers (beam pruning on `δ`, a coarser
trellis) have not been tried. The scanner is usable again but not comfortable.

### 7.7 The EM pass that was kept

One change on top of Stage 2, worth +7.4 band and +12.5 capture recall for
−0.06 flat:

`decode_window` now runs an unconditional EM pass before the existing
conditional one. It re-estimates levels from the first decode's segmentation,
refills the likelihoods, and lets the period re-settle one grid step either way
against the improved levels (`refine_steps(t_ds, rate, 1)`), keeping the result
when it scores within 5 of the incumbent. The second pass, previously gated at
contrast < 6, is now gated at < 8.

Why it works where the Stage 1 attempts did not: the levels are re-estimated
from a globally optimal segmentation rather than from an instantaneous
threshold, which is exactly the distinction §5.3 drew. Why the narrow refine:
five candidates cost 1.71× end-to-end against 1.94× for three, and scored
slightly worse on the band.

### 7.8 Where the remaining gap is

- **Band 49.27 → 60.** `chan flutter` (6.3%) and `the band` (15.1 / 14.4) are
  the three cells holding the mean down. Flutter is a fast-fading collapse; the
  band cells are the multi-station case Stage 4 is meant to address, and no
  amount of single-tone work will fix them. *(Stage 4 shipped — see §8. It
  recovers the QRM cells on the throughput metric but leaves the lock-only
  mean where it was, so the arithmetic below is unchanged.)* Note the arithmetic: lifting all
  three to 60 % only reaches a mean of 59. **Band ≥ 60 is probably not
  reachable without Stage 4**, and §6 was written before that was known.
- **Flat 84.72 → 88.** Almost entirely the −3 dB row, which is 61 % of the
  total deficit across all 41 cells and sits at 0–11 %. §7.10 diagnoses it: the
  period estimator collapses to under half the true dit. Fix that and the gate
  falls out; nothing else on the flat grid is worth more than a point.
- **The `DE` word gap** (§7.3, §7.10). Small, concrete, present even at 15 dB,
  and the one test whose failure says something is structurally wrong rather
  than merely weak.

### 7.9 Rejected: the second Stage 2 round

A round of work that reported itself complete was measured and reverted whole.
It is recorded because the ideas look plausible and should not be retried.

| Attempt | Effect | Verdict |
|---|---|---|
| `mu_mark` from a ±50 ms **running maximum**, bidirectionally smoothed, sold as fade tracking | Flat 84.58 → 48.34 in isolation | rejected. This is a max-hold peak, the structure §3 and §5.3 exist to eliminate. Refer levels to a **mean**. |
| `mu_space` pinned to a constant 12.5th percentile of the window | part of the above | rejected — throws away adaptive space tracking |
| `reest_levels` rebuilt on interpolated per-segment mark averages | Band 48.97 → 44.99, flat 84.77 → 83.78 | rejected; the simple mean-of-mark-samples estimator it replaced is better |
| SNR-adaptive emission variance `sigma = clamp(0.40·mu_s/span, 0.22, 0.65)`, to "remove the 0 dB cliff" | 0 dB cells 42.5/52.5/66.2/45.0 → 31.2/45.0/48.8/36.2 | rejected — it makes the exact cells it targets worse |
| Loosening ~13 detector thresholds at once (contrast, `inst_q`, `mark_frac`, `t_jump`, clock-fit tolerances, gap boundaries) | Band 45.41 → measured worse at every setting tried | rejected |
| Emission gates in `cw.rs` cut to `mark_env < 1.08·space_env && quality < 0.12`, replacing the `2.4×` contrast guard | **All three flat `noise *` cells 100% → 0%** | rejected outright. The decoder invented Morse from pure noise, which §6 forbids. |
| `update_post_mix` with the `have_period()` guard removed | reintroduces the §5.7 feedback loop; orphans `have_period` | rejected |

Combined, that round measured flat **39.97%**, band **32.93%** (below the 38%
gate), `noise *` at **0%**, capture recall **55.8%** with density collapsing
from 18.3 to 3.9 tok/100c, and 14 failing tests including two of the three
§6 canaries. The §7 it left behind reported none of this.

**The lesson worth keeping:** every one of those changes is a threshold or an
estimator tuned by inspection rather than measured. The one change that
survived (§7.7) is structural — it adds an inference pass, not a constant. On
this decoder, tuning constants by hand has now failed across three rounds and
roughly twenty attempts. Measure first, and run `bench_cw_score`'s `noise *`
cells and `cw_capture`'s density column before believing any improvement.

### 7.10 Rejected: one round of trying to close §6

An attempt to reach band 60 % and flat 88 % from 49 / 85. Nothing in it
survived. It is recorded in full because most of it looks obviously right, and
because the diagnostics are worth more than the attempts.

**What the −3 dB row actually is.** Not the gates, not the emission width — the
period estimate collapses. Disabling both amplitude gates in `cw.rs` does not
move the −3 dB cells at all (it only takes the `noise *` cells to zero), so
they are not what is binding. Instrumented at 18 WPM (true dit 67 envelope
samples), the search reports `t_best` of 62–70 at 3 dB, and at −3 dB it locks
at **30** — under half — then wanders 24–38 for fourteen windows before finding
72 near the end. `inst_q` reads 0.98 on that garbage, so the likelihood ratio
against null is saturated and cannot be used to detect the condition. The
mechanism is overfitting: a shorter period explains a noisy envelope with more,
smaller elements, and the per-segment charge in `best_period` is a flat 3.5
against an emission term that grows without bound as SNR drops.

**`GapChar` and `GapWord` are dead states.** Instrumenting the committed
decoder, *every* gap comes back labelled `Idle` — 1-dit element gaps, 3-dit
character gaps and 7-dit word gaps alike. `Idle` is a per-sample geometric
self-loop at ~0.002 nats/sample, so a 3-dit gap costs it ~0.4 against
`GapChar`'s one-off ~3.1, and it wins everywhere. The explicit-duration grammar
§5.1 specifies has never actually run. Charging a proper `Idle` entry cost
(−12 nats) does fix the labelling — `GapChar` lands at 3.1–3.3 dits, `GapWord`
at 7.1–7.4, and `t_dit` gets visibly more accurate — and it measures **worse**:
flat 84.2 %, band 47.6 %. The tight duration Gaussians are less robust to real
timing jitter than the geometric state that was accidentally doing the job.
Do not "fix" this without measuring it.

**The word-gap threshold is boxed in.** `WORD_GAP_DITS` is 4.6, the geometric
mean of 3 and 7. Measured gaps run ~15 % short, because the matched boxcar and
the post-mix filter smear mark energy into the silence on both sides, so a true
7-dit word gap arrives as 4.0–4.5 and reads as a character gap. That is the
`DE W1AW` / `DE NK9G` defect, and it is present at 15 dB, not just at the noise
wall. Lowering the threshold to 4.0 fixes those and is worth **+1.4 flat**
(86.10 %) — and it breaks `cw_decodes_farnsworth_timing` and
`cw_decodes_bug_keyer_weighting`, because Farnsworth stretches character gaps
on purpose and no fixed multiple of the dit separates a stretched character gap
from a word gap. §7.5 hit the same wall at 3.9. **No constant works here.**

| Attempt | Effect | Verdict |
|---|---|---|
| Emission sigma measured from the envelope (residual against the tracked level) instead of the fixed `VAR = 0.12` | flat 84.5 → 83.7, band 49.3 → 48.1 | rejected; the assumed constant beats the measured value |
| Per-segment charge scaled by measured dispersion, so fragmentation costs more when the evidence is poor | flat 85.1 / band 49.9 at slope 110, but **breaks `cw_decodes_bug_keyer_weighting`** above slope ≈ 10; at slope 10 it is flat 84.4 / band 48.3 | rejected. The slope that helps the noise wall is the slope that eats a bug keyer's light dits. |
| Flat per-segment charge raised (8 / 15 / 25 / 40 / 80) | −3 dB probe 0.00 → 0.27, grid mean 83.7 → 81.1 → 75.6 | rejected; helps the weak cells, costs more on the strong ones |
| Median-of-7-windows period vote instead of the EMA | `speed 30->14wpm` 98 % → 84 % | rejected. A median resists a genuine speed change, and the low-SNR period error is a bias, not noise — 14 of 20 windows were short, so voting cannot fix it. |
| `Idle` entry cost of −12 nats, restoring the §5.1 grammar | flat 84.2 %, band 47.6 % | rejected, see above |
| Word-gap transition prior flattened 0.06 → 0.25 so durations decide | no effect on the failing gap | rejected |
| `WORD_GAP_DITS` 4.6 → 4.0 | flat +1.4, two timing canaries fail, capture recall 91.2 → 83.8 | rejected, see above |

**The pattern.** Every change that moved a grid broke a canary, and every
change that kept the canaries green failed to move a grid. That is the same
frontier §3 describes from five Stage 1 detectors, met again from a different
direction. The canaries are load-bearing and not arbitrary:
`cw_decodes_bug_keyer_weighting` is the one that catches anything which
suppresses short elements, and it caught two separate attempts here.

**Where the next real gain is.** Not in constants. Two structural leads, both
untried:

1. **An adaptive gap model.** Cluster the observed gap durations into
   element / character / word rather than thresholding on `t_dit`, the way
   `recluster` did for marks in Stage 0. That fixes the `DE W1AW` class of
   defect without the Farnsworth trade, and §7.3 shows the word gap is the most
   concrete failure left.
2. **A period estimator that is not the decode objective.** The period is
   currently chosen by the same score it overfits, which is why it collapses at
   −3 dB. Anything independent — envelope autocorrelation, or a histogram of
   level-crossing intervals accumulated over the whole over — used as a hard
   prior rather than a hint would break that. `hint_dit` gestures at this but
   is a hard slicer and is worthless at the SNR where it matters.

### 7.11 The lockup, and the design trap behind it

Reported from real use: the app freezes for long periods and CPU goes to
100 %. This was not the constant-factor slowdown in §7.6. It was unbounded
growth, and it is the single most important thing on this page for anyone
rebuilding Stage 2.

`commit_path` only advances `committed` through a character gap, a word gap or
a long idle — **an element gap does not count**. So any signal the HSMM
segments as unbroken keying, marks separated only by element gaps, never
commits anything. `trim` then kept every uncommitted sample:

```rust
let keep_from = self.committed.min(now.saturating_sub(win));
```

With `committed` frozen, the envelope buffer grew for as long as the signal
lasted. Each window costs O(buffer), so cost per second of audio climbed
linearly and total work grew quadratically. On a 20 WPM square wave — a stuck
key, a station tuning up, an unattended keyer, all ordinary things to find on
14060 — one channel measured:

```
second  1:   0.3 ms of CPU per second of audio
second 10:  33.8 ms
second 20:  66.0 ms
second 30:  99.2 ms   and still climbing
```

One channel saturates a core in minutes; sixteen do it far sooner. Bounding
the buffer with a hard `MAX_WIN_S` floor regardless of commit progress fixes
it flat at ~21 ms/s, costs nothing on any instrument, and is on the
`cw-stage2-em-salvage` branch along with a regression test
(`cw_cost_does_not_grow_on_unbroken_keying`) that measures growth rather than
absolute speed.

**The trap, stated generally.** A sliding-window decoder that retains data
until it commits, and whose commit condition depends on the decode succeeding,
has an unbounded failure mode built into it — precisely on the inputs where the
decode does not succeed. Stage 0 has no such coupling: it slices, emits, and
keeps nothing. Any Stage 2 rebuild needs a hard bound on retained state that
does not depend on the decoder agreeing to make progress, and a test like the
one above from the first commit.

## 8. Later stages

**Stage 3 — lexicon rescoring. Done — see §9.**

**Stage 4 — decode every tone. Done.** `CwDecoder` now owns a `Vec<Tone>`,
one per station the search finds, capped at `MAX_TONES = 4`. The search and
its FFT stay shared; each tone gets its own mixer, post-mix filter, envelope,
slicer, clock and `CallScanner`.

The lock path is unchanged and measures unchanged — flat 90.57 %, band
44.03 %, both identical to the commit before. That is the point: the operator
hears the same station they always did. What is new is everything behind it.
`bench_cw_band` now reports a second column, **throughput of copy** — was the
wanted station copied in *any* stream, rather than in the one the lock landed
on — and that is where the multi-station cells move:

| cell | lock | any |
|---|---|---|
| `qrm x2 moderate` | 34.9 % | 45.7 % |
| `qrm x3 moderate` | 26.1 % | 45.5 % |
| `the band, 10dB` | 16.0 % | 25.0 % |
| grid mean | 44.03 % | **46.90 %** |

`qrm x3 moderate` recovering to within a point of the QRM-free `chan
moderate` cell (46.2 %) is the result worth reading: three neighbours cost
20 points of copy purely by capturing the lock, not by damaging the signal.
`qrm x3 poor` and `the band, 4dB` barely move, because those are
fading-limited rather than selection-limited — §7.8's arithmetic still holds
and Stage 4 is not what gets the band mean to 60.

**CPU, the budget §6 never set.** 1.0 ms per second of audio for four tones
against 0.4 ms for one — 0.1 % of a core. `bench_cw_cpu` prints it and
`cw_cost_does_not_grow_on_a_busy_band` measures *growth* rather than absolute
speed, which is §7.11 as a regression test: tone retirement is driven by the
search failing to find a station, never by that station's decode failing to
progress, so the coupling that caused the lockup cannot form here.

**Two things worth knowing if you touch this.** Acquisition is deliberately
single-tone: the primary is chosen before `sync_tones` runs, because letting
a tone spawned in the same search be adopted as the primary hands the user a
cold slicer that has missed the start of the transmission — measured, it
turned `CQ CQ DE ...` into `NQ CQ DE ...`. And `TONE_SEP_HZ` is 60, not the
search's 40: the post-mix filter is 60–150 Hz wide, so two tones closer than
that transcribe the same station twice, once well and once badly.

**On screen.** The copy column splits: the lock's transcript on top, an
`also copying` section under it with one line per background station, placed
at its own frequency and carrying the tail of what it has sent. The section
only exists when something is behind the lock, so the ordinary one-station
case looks exactly as it did. The tuner's tone list marks each hit `>` for
the lock, `·` for copied in the background, blank for found but not decoded —
which is the first time that list has distinguished *seen* from *copied*.

The copy floor moved with it. It used to gate every spot on `confidence()`,
which describes the lock; with several stations running that is the wrong
question, so `set_copy_floor` pushes the bar into the decoder and each tone
is held to it by its own quality. Held-back copy is discarded as it goes
rather than queued, so dropping the floor does not release a backlog.

**Not CW, but owed. Done.** Both are on `decoders::channel` now
(`bench_rtty_fading`, `bench_psk31_fading`), and the guess above was half
right.

RTTY was hiding the same class of problem. A 170 Hz shift straddles the
coherence bandwidth of every CCIR path, so the two tones fade independently,
and the discriminator's per-tone max-hold held its reference for 1.5 s —
long past the point where the tone that is down inverts against the tone that
is up. `gen_rtty_faded` could not see it, because a static per-tone amplitude
is exactly what a per-tone AGC is built to absorb. `PEAK_HOLD_S` is 0.3 s on a
0.15–0.5 s plateau; `CCIR_POOR` at 20 dB went from 58 % to 74 %.

PSK31 was not. Per-symbol normalisation makes it genuinely fade-depth-immune,
and it holds up on good, moderate and poor paths. It does collapse to 10 % on
`CCIR_FLUTTER` at every SNR, but that is the mode's limit, not a defect: 5 Hz
of Doppler decorrelates the phase in about one 31.25 baud symbol, and the
decoder was measured locking within a hertz and still returning garbage.
**Do not spend a round on it.** The one genuine wart is the AFC walking 20 Hz
off on one seed in three while chasing the Doppler, and fixing that would not
buy a character.

**Also found, unrelated to fading:** `spot_snr` referred its measurement to
2500 Hz with a hardcoded 12.3 dB — the bandwidth of the *fixed* post-mix
filter that `POST_MIX_K` had since made adaptive. Every CW spot was going out
about 4 dB optimistic. Derived from `post_hz` now, with
`cw_spot_snr_is_calibrated` asking the absolute question that
`spot_snr_follows_the_band` does not.

---

## 9. Stage 3 — lexicon rescoring (done)

`src/decoders/cwlex.rs`, on top of the Stage 0 slicer. The plan assumed this
would sit on the HSMM's soft character hypotheses; the HSMM is gone, and it
turns out not to matter — the elements each character was decoded from are
enough, and Stage 0 has them.

| Instrument | Stage 0 | Stage 3 |
|---|---:|---:|
| `bench_cw_score` | 90.22% | **90.57%** |
| `bench_cw_band` | 43.54% | **44.03%** |
| `cw_capture` recall | 81.9% | **88.1%** |
| `cargo test --release` | 134 / 0 | **143 / 0** |
| `bench_replay` end-to-end | 73.5× | 73.8×, every row PASS |

Everything up, nothing down, and free: it is string work on already-decoded
characters, so the replay figures are unchanged inside noise.

**Be honest about the capture number.** The +6.2 is one token. Only 14028.01
moved, 2/4 → 3/4, when `CG CVA PT6T` became `CQ CVA PT6T`. With four stations
and 39 tokens the metric is coarse, and a single recovery is worth six points.
The flat and band gains are small but broad, and those are the trustworthy
half of the result.

### 9.1 Distance is measured in elements, not characters

`C` and `G` are unrelated letters and neighbouring Morse patterns — `-.-.`
against `--.` — so a character-level edit distance cannot see that `CG` is a
near miss for `CQ` while `CX` is not. Everything works on the element string
with `/` between characters, and `/` is edited like any other symbol, so two
letters run together by a mis-heard gap cost exactly one deletion, which is
what they physically are.

### 9.2 Two lists, and why

The design mistake worth not repeating: a single lexicon, used both to
recognise real words and as the set of corrections to make. Measured, that
rewrote `NEWINGTON CT` into `NEWINGTON BT` — `CT` is one element from the
procedural `BT` — and `THE QSO` into `TEST QSO`. Seven cells of
`bench_cw_score` fell from a clean 100 % to 96.4 %, on *correctly decoded*
copy. The whole flat and band gain above is the difference between one list
and two.

So `LEXICON` is broad and only answers "already a real word, leave it alone",
while `CORRECT_TO` is small and is the only thing a decode may be rewritten
into. A token earns a place in the second list by carrying meaning — `CQ` and
`DE` gate `CallScanner` and therefore the spots — and by not sitting one
element from an ordinary word. The filler signals (`BT`, `AS`, `NW`) fail the
second test and are recognised but never corrected toward.

Three further rules keep it from inventing: a word already shaped like a
callsign is never rewritten, since spots are the product; the runner-up
candidate must be strictly worse, so `CX` at distance 1 from both `CQ` and
`TNX` is left alone; and the element budget is one for anything short, because
two is enough to turn `THE` into `TEST`.

### 9.3 What it does not do

No context. `CQ` is corrected the same way wherever it appears, and the plan's
worked example — `"CE TEST T EWH I ZW B"` → `"CQ TEST DE ZW5B"` — is only
half reachable, because repairing `ZW B` into `ZW5B` means inventing a
character inside a callsign. That is the line this module does not cross.

Positional context was the obvious next increment — `DE` and `CQ` are followed
by a callsign within a word or two, which `callscan.rs` already models. It was
built and measured and it does not pay; §9.4 records all three variants and
what they cost.

### 9.4 Rejected: positional context

§9.3 proposed it and it was built three ways. All three measure neutral, so
none of them is in the tree.

The mechanism: hold one finished word back so a correction can see the word
after it, then let `CQ` and `DE` be argued for by a following callsign, and
activity names by a preceding `CQ` — the grammar `callscan.rs` already models.

| Variant | flat | band | capture |
|---|---:|---:|---:|
| No context (in tree) | **90.57 %** | **44.03 %** | **88.1 %** |
| Context widens the element budget by one | 90.73 % | 43.86 % | 88.1 % |
| Context breaks ties, never widens the budget | 90.42 % | 44.12 % | 88.1 % |

Capture recall is identical to the token in all three. Flat and band move by
less than the seed noise, in opposite directions, depending on the variant.

**Why it does not pay.** The blind distance-1 rule already takes the cases that
exist — `CG` → `CQ` is distance 1 and needs no help. Candidates at distance 2
that context could rescue turn out to be rare in real copy. And the errors
actually left on the capture are not context problems: `ZWIIB` for `ZW5B` needs
a character invented *inside* a callsign, and `NK9GFK9G` is a callsign sent
twice with one letter wrong. Neither is reachable from the neighbours.

**What it cost, which is the part worth remembering.** About 196 lines, and
five new pieces of decoder state: a one-word hold, a separator flag, an
end-of-over drain threshold, and `prev_word` / `over_start` tracking. It also
introduced a bug that only a test caught. Moving the word separator to the
front of the following word — needed because `text` is drained every block and
cannot be asked what it ends with — delayed every word reaching `CallScanner`,
which completes a word only on whitespace. Every spot fired a word late and
`spot_snr_follows_the_band` failed at 7.0 dB of separation against a bar of 8.
The fix is to separate the two streams: `text` takes its separator in front,
`scan` takes one behind. If this is ever rebuilt, that asymmetry is the trap.

**The budget-widening variant was also qualitatively worse**, which the
aggregate numbers hide: on 14002.51 it turned `S EEE EIATWC5EE` into
`S DE EIATWC5EE`, manufacturing a procedural word out of what is almost
certainly noise. A second element of latitude is too much however good the
context looks — the same finding as §9.2, from a different direction.

**If it is picked up again**, do it for a correction that context genuinely
decides rather than one it merely agrees with, and have the case in hand before
building the plumbing. The word-hold machinery is the prerequisite for word
merging (`C T` → `CQ`, a spurious word gap, which is a real observed error on
14002.51) and that is a better reason to build it than context was.

