# Weak-signal and optimisation plan, round 2

> **STATUS: FINISHED — 2026-08-16.** All eight items implemented and measured.
> Final suite: **174 passed, 0 failed**, plus 38 benches passing.
>
> | | before | after |
> | --- | --- | --- |
> | PSK31 copy at 6 dB (31 Hz) | 0% | 46% |
> | CW copy at 0 dB, 18/25 WPM | 15% / 20% | 45% / 70% |
> | RTTY copy at 12 dB (45 Hz) | 73% | 95% |
> | RTTY at 8 dB | 32% | 50% |
> | Image rejection through static crashes | 14.0 dB | 41.1 dB |
> | Impulse blanker error-power | 52.0 dB | 54.1 dB |
> | Noise floor: bias vs true level | −7 to −15 dB | +0.0 to +0.8 dB |
> | Fleet cost, 24 slots at 192 kS/s | 7.0% | **1.8%** |
> | Fleet cost at 384 / 768 / 5000 kS/s | 12.6% / — / 840% | 2.5% / 3.8% / 231% |
> | Cost per slot | 2.9 ms/slot/s | 0.8 ms/slot/s |
>
> **Three of this plan's diagnoses were wrong, and the measurements said so.**
> They are corrected in the item text below; the short version:
>
> 1. **PSK31's matched filter was not the problem, and "fixing" it made things
>    worse.** PSK31 shapes its symbols with a *full* raised cosine, not a root
>    raised cosine, so a receive filter matched to the transmitted pulse
>    produces 33% adjacent-symbol ISI — worth −3.5 dB on reversal-heavy
>    traffic against the +1.2 dB the matched filter buys. Implemented and
>    measured: confidence at 12 dB fell from 0.759 to 0.538. The existing
>    one-symbol window is the better filter and was kept. The real cause of
>    the cliff was a *noise fluctuation* tripping the `keys_at_other_baud`
>    veto, with `DC_FOCUS_MIN` sitting too close to noise to carry the
>    rejection on its own.
> 2. **The CW Viterbi did not pay.** It was implemented in full — two-state
>    trellis, emissions from the tracked mark/space levels, transitions from
>    the dit clock, fixed-lag traceback over `u64` survivor paths — and it was
>    *worse* than the slicer at every SNR that mattered (35 WPM at 3 dB: 70%
>    → 30%). Measuring σ rather than deriving it recovered most of that but
>    never beat the baseline. Per this project's own rule it was not merged;
>    the patch is kept at `weak-signal-plan-2-cw-viterbi.patch`. The actual CW
>    win was somewhere the plan never looked: the post-mix filter passed
>    147 Hz of noise to a signal whose keying occupies ~33 Hz.
> 3. **The noise-floor bias correction cannot be used for detection.** With
>    the floor biased 11–19 dB low, `max(floor, local_median − 3)` always took
>    the clamp — so the candidate gate was effectively at *zero* prominence,
>    and the previous plan's "4.0 dB → 0.0 dB gate improvement" was a disabled
>    threshold rather than 4 dB of sensitivity. Correcting the bias raises the
>    bar and loses weak signals. Detection keeps the raw estimate; the
>    corrected level is exposed separately for the cursor SNR and colour
>    scale, which are the callers that wanted a level all along.
>
> Item 4 landed in two stages, both measured: sharing the forward transform
> took 7.0% → 5.3%, and decimating by folding the spectrum took it to 1.8%.
> Spans whose decimation does not divide the frame (reachable only by
> `--rate`) fall back to a whole-frame inverse transform, and
> `SLOT_RATE_BUDGET` is sized for that fallback rather than the fast path.
>
> Not done, and deliberately: `PROMINENCE_DB` was not lowered (see 3 above —
> the gate is already at zero prominence, so there is nothing to lower), and
> the FT8 AP hints beyond `with_call2` were left alone as the plan's own
> "do it last or not at all".


Successor to `weak-signal-plan.md` (items 1–8 finished 2026-08-14, plus the
receiver-control follow-up). This one starts from what that plan actually
left on the table, measured on the current tree rather than assumed.

## Measured baseline (2026-08-16, this tree, release build)

Suite: **167 passed, 0 failed, 35 ignored.**

| bench | result |
| --- | --- |
| `bench_frontend_cost` | 0.27 % of real time at 192 kHz |
| `bench_frontend_weak_signal_cost` | +0.00 dB on clean, +0.02 dB at 5 % imbalance |
| `bench_blanker_on_quiet_band` | +0.00 dB, 0 blanks/s |
| `bench_noise_blanker` | 52.0 dB error-power improvement |
| `bench_frequency_dependent_iq` | 22.1 → 51.9 dB worst band |
| `bench_tracked_floor_candidate_gate` | 4.0 dB → 0.0 dB |
| `bench_feed_cost_per_band` | **7.0 %** of real time, 24 slots at 192 kS/s; 12.6 % at 384 |
| `bench_slot_cost` | 2.9–3.2 ms/slot/s ≈ 0.29 % of real time per slot |
| `bench_cw_accuracy` | 90 % to 6 dB; 70–85 % at 3 dB; **15–20 % at 0 dB** |
| `bench_psk31_accuracy` | 100 % to 15 dB; 92 % at 10 dB; **0 % at 6 dB and below** |
| `bench_rtty_matched_filter_fade` | 25 chars copied with the space tone 20 dB down |

Two things fall straight out of that table.

**The front end is finished.** It costs a quarter of one percent of real
time and provably costs a weak signal nothing. There is no optimisation
worth doing there; the remaining front-end items below are correctness
issues, not performance ones.

**The decode fleet is the whole CPU budget, and the decoders are where the
dB are.** 24 slots is 7 % of real time against the front end's 0.27 %. And
the two narrowband decoders both fall off a cliff in the last few dB —
PSK31 catastrophically.

The README is stale on this point: it still quotes 14 % / 27 % / 70 % /
840 % for the four span widths. Measured now: 7.0 % at 192 kS/s and 12.6 %
at 384. Whatever landed since halved it. Fix the README as part of item 1.

Same rule as the previous plan: **build the bench first, make the change,
keep the bench, turn the measured result into an assertion test.** An
unmeasured weak-signal claim does not merge.

---

## 1. PSK31 below 10 dB — the largest single gap (start here)

`bench_psk31_accuracy` goes 92 % at 10 dB → **0 % at 6 dB**. That is not a
graceful degradation, and the interesting part is that the *search* is
fine through it: the bench prints `lock -0.4 Hz` at 6 dB and `lock +2.0 Hz`
at 3 dB. The decoder knows exactly where the signal is and produces
nothing. So this is a demodulator problem, not a detection problem, and
there is no reason a mode whose whole purpose is weak-signal work should
stop 4 dB above where its own search still tracks the carrier.

Three defects in `decoders/psk31.rs`, in order of expected gain:

**The pre-detection filter is a single one-pole at 60 Hz** (`lpf_a`,
`Psk31Decoder::new`). One pole at 60 Hz has an equivalent noise bandwidth
near 94 Hz against a signal occupying 31 Hz, and — worse — it is already
3.6 dB down *at 31 Hz*, inside the signal, with the matching phase lag.
That is ISI applied to the pulse before the matched filter ever sees it.
Replace with a linear-phase FIR sized to the signal (the project already
has `lowpass_taps` and `DecimFir`, and at 8 kHz audio the tap count is
free). Keep the corner where it is for the *search* path if the separation
measurements in `baud_line` depend on it — but the demod path should not
share it.

**The receive window does not match the transmit pulse.** `process` weights
the dump with `0.5·(1+cos(π(x−0.5)·2))` over *one* symbol. The transmitted
PSK31 pulse is a raised cosine spanning *two* symbol periods with adjacent
pulses overlapping (see `gen_psk31_at` in `decoders/tests.rs`, which
generates exactly that). Correlating a 1-symbol window against a 2-symbol
pulse throws away half the pulse energy and adds ISI from the neighbour.
Implement the actual matched filter: correlate against the true 2-symbol
raised cosine, overlapping, dumping once per symbol.

**Timing recovery jitters at low SNR.** `energy[idx] = 0.95·e + 0.05·|s|`
and the dump walks toward the minimum-energy bin. Envelope minima are
exactly what disappears first in noise. Either lengthen the averaging as
SNR falls, or move to a Gardner/Mueller-Müller error computed from the
matched-filter outputs (which are already normalised) rather than from the
raw envelope.

Bench: extend `bench_psk31_accuracy` to 8, 6, 4, 2, 0 dB and print
character accuracy. Assert copy ≥ 50 % at 6 dB (currently 0 %) with no
regression at 10 dB and above, and keep
`auto_mode_invents_no_narrowband_signals_from_noise` passing — that test is
the false-alarm guard for the whole change.

## 2. CW: the two-state smoother that never landed

Old plan item 7 had two halves. The matched envelope landed and measured
13.4 dB (`bench_cw_matched_envelope`, at the regression-safe 0.35-dit
boxcar). **The HMM/Viterbi half was never implemented** —
`CwDecoder::step_envelope` is still hard slicing with hysteresis plus a
debounce timer.

That is visible in `bench_cw_accuracy`: 90 % holds down to 6 dB, 70–85 % at
3 dB, then **15–20 % at 0 dB**. Instantaneous slicing is what shreds there:
a noise spike inside a space invents a dit, a QSB notch inside a dah splits
it, and the debounce is a fixed-length guess about which.

Implement the deferred half over each buffered stretch of envelope:

- Two states (mark / space), emission likelihoods from the on/off envelope
  distributions the decoder already tracks — `mark_env` and `space_env` are
  precisely the honest signal and noise levels for this (see their doc
  comment), not `peak`/`floor`, which are deliberately pulled together.
- Transition probabilities from the tracked dit clock: the probability of
  leaving a mark grows as the run approaches a dit or a dah, which is the
  information a fixed debounce cannot express.
- A short forward-backward or Viterbi pass over the buffered envelope,
  replacing the slicer output that feeds `on_mark_end`/`on_space_end`.
- Keep the existing clock recovery, `morse_clock` structure check, and the
  warm-up replay untouched — they are what stops noise being decoded, and
  a smoother that makes noise *look* more like Morse must not bypass them.

Bench: extend `bench_cw_accuracy` with a 0 and −3 dB row. Assert copy at
≥ 2 dB lower SNR than the slicer, no regression on the 15–35 WPM clean rows
in `decoders/tests.rs`, and no new copy at all on a pure-noise input.

## 3. RTTY: a real integrate-and-dump

Item 6 landed and works — 25 characters through a 20 dB selective fade is
the hard case and it passes. But what `RttyDecoder::process` calls a matched
filter is a one-pole leaky integrator, `a = min(3/samples_per_bit, 1)`,
i.e. a time constant of a third of a bit. The optimal detector for a
constant-envelope tone over a bit is an integrate-and-dump over exactly one
bit, synchronised to the bit clock — which the framer already knows, since
it is counting samples against `samples_per_bit`.

Replace the two leaky integrators with two boxcar correlators dumped on the
framer's own bit boundaries, keeping the per-tone ATC normalisation
(`mark_peak` / `space_peak`), the dual-polarity framing, and the AFC. Expect
1–2 dB. The `bit_acc` mid-bit averaging in `Framer::feed` becomes redundant
and should go with it.

Bench: extend `bench_rtty_matched_filter_fade` into a graded-SNR character
error rate table. Assert CER at a fixed low SNR improves, and the existing
fade case still copies.

## 4. Shared channeliser for the decode fleet

Every `AutoSlot` owns a full `DecodeChain` and mixes and filters the
*entire* wideband block independently: NCO over 192 k samples/s, then a
2047-tap overlap-save FIR whose forward FFT is recomputed per slot. Twenty
four slots do the same 4096-point forward FFT twenty four times on the same
input.

Do it once. This is a standard weighted-overlap-add channeliser and the
code is already 80 % of the way there, because `DecimFir` is already
overlap-save:

- Forward-FFT the wideband block once per block, shared by all slots.
- Per slot, the NCO becomes a bin rotation in the frequency domain (exact
  for an integer bin offset; a fractional-bin phase ramp covers the rest).
- Multiply by the filter response over the passband bins only — not the
  whole span.
- Inverse-transform at `fft_len / decim` rather than `fft_len`, which
  produces the decimated output directly instead of computing 24 samples
  to throw away 23.

Expected: per-slot cost roughly an order of magnitude down, fleet from 7 %
toward 1.5 % at 192 kS/s.

The point is not the CPU. It is that `MAX_AUTO_SLOTS = 24` and
`SLOT_RATE_BUDGET` exist only because of this cost, and the README is
explicit that a wide span shrinks the fleet — "a signal that never gets a
slot is a station nobody hears about". Cheaper slots mean the 384 kS/s
bands (10 m, 2 m) and any `--rate` override carry a full fleet, and the
per-slot filters can afford to get sharper at the same time.

Bench: `bench_slot_cost` and `bench_feed_cost_per_band` already exist and
measure exactly this — quote both before and after. The four `DecodeChain`
response tests and `overlap_save_matches_direct_decimation` are the
correctness guard, and `decimation_does_not_alias_into_the_channel` must
still hold at −80 dB.

## 5. Per-sample `sin_cos` in every decoder's mixer

`dsp::Rotator` exists specifically to kill this, and its doc comment
records that per-sample trigonometry was costing a quarter of a second per
scout pass. Four hot paths never got converted:

- `RttyDecoder::process` — **two** `sin_cos` calls per audio sample
  (`center_phase` and `tone_phase`).
- `Psk31Decoder::mix` — one per sample.
- `CwDecoder::mix` — one per sample.
- `decoders::cw::score_cw` — one per sample, on the scout path, which is
  the one the Rotator comment was written about.

Mechanical change, no signal-path risk beyond the renormalisation Rotator
already handles. Fold it into whichever of items 1–3 touches that file.

## 6. Front-end correctness (not performance)

Three real defects, none of which show up in the current benches:

**Blanking order.** `FrontEnd::process` runs DC → `correct_images` →
`blanker`. The image estimator therefore accumulates its cross-correlation
statistics over blocks that still contain impulses, which are broadband and
by definition the loudest thing in the block. Blank first, then estimate.

**Blanking method.** `NoiseBlanker` zeroes samples. The original plan said
"zero (or better, linearly interpolate across)" — a zeroed run is a
rectangular hole, and a narrow channel filter downstream rings on it. Linear
interpolation across the blanked run is a few lines and strictly better.

**Uncorrected tail.** `correct_images` iterates `chunks_exact_mut(4096)`, so
any block whose length is not a multiple of 4096 has its remainder passed
through with no image correction at all. `radio::BLOCK` is 16384 so the app
is safe today, but this is silent, and it makes the correction depend on a
constant three files away.

Bench: `bench_noise_blanker` and `bench_frequency_dependent_iq` cover the
first two; add an impulse-plus-imbalance case that exercises the ordering,
and a non-multiple-of-4096 block length to `clean_iq_is_left_alone`.

## 7. Noise-floor tracker: rate-independent time constants

`NoiseFloor::update` blends with fixed per-call coefficients (0.25 down,
0.002 up) and `App::feed` calls it once per IQ block. So the tracker's time
constant is set by the block rate, which is set by the sample rate: ~43 s
upward at 192 kS/s, ~21 s at 384. A detection parameter should not change
because the user pressed `b`.

Convert both coefficients to per-second and derive the per-call alpha from
the elapsed time. Two follow-ons become available once it is stable:

- `feed` calls `update` on every block even when `Spectrum::power_db` left
  `out` untouched (no whole segment available yet — routine at 32768-point
  FFT sizes, where a segment needs 170 ms). The same spectrum is then folded
  in repeatedly. Skip the update when the spectrum did not change.
- A minimum-statistics estimator sits *below* the true mean noise power by a
  known bias that depends on the number of averaged segments. Correcting it
  would let `PROMINENCE_DB` (currently 2.75) drop further, which is where
  the next weak CW and PSK31 candidates come from.

Bench: `bench_tracked_floor_candidate_gate` already measures the gate.
Add a rate-sweep assertion — the settled floor must be within a fraction of
a dB of the same value at 192 and 384 kS/s given the same input SNR.

## 8. Cheap leftovers

- `find_peaks_above` sorts a full copy of the spectrum on every call purely
  to take a median, even though the tracked floor is passed in and is the
  better reference. Use the floor and keep the median only as the cold-start
  fallback it already is in `usable()`.
- FT8/FT4 (`decoders/ft8.rs`) is done: `MAX_CAND` 600, sync 0.75, five SIC
  rounds, OSD, local EQ, own-call AP, and an adaptive depth guard against
  the slot budget. The one item from the old plan's list still unspent is AP
  hints beyond `with_call2` — standard CQ patterns. Low marginal value
  against items 1–3; do it last or not at all.
- README CPU figures (14 / 27 / 70 / 840 %) are stale by 2×. Update from
  `bench_feed_cost_per_band` with item 4, since that item moves them again.

## Sequencing

1. **Item 1 (PSK31).** Biggest measured gap, isolated to one file, and the
   bench already exists to show it.
2. **Item 2 (CW Viterbi).** Explicitly deferred work with a known 2–3 dB
   target.
3. **Items 6 and 7** — front-end and detector correctness. Cheap, and item 7
   makes every later detection measurement trustworthy.
4. **Item 3 (RTTY).** Independent of everything above.
5. **Item 4 (channeliser).** Largest change; do it once the decoders are
   settled, since it moves the code they run inside.
6. **Items 5 and 8** folded into whichever commit touches the file.

Items 1–3 are dB. Item 4 is stations that currently never get a slot. Items
6–8 are the measurements the rest depend on being honest.
