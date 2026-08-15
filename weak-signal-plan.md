# Weak-signal receiver improvement plan for hfscan

> **STATUS: FINISHED — 2026-08-14.** Items 1–8 were implemented, measured,
> regression-tested, and committed. The optional/later ideas in item 9 remain
> explicitly deferred; they are not required for this plan's completion.
>
> Measured outcomes: impulse blanking 14.2 dB; tracked-floor candidate gate
> 4.0 dB; frequency-dependent IQ correction 29.8 dB better than scalar in the
> worst band; 1101-tap FFT convolution 35.1× faster; RTTY copies through a
> −20 dB selective fade; clock-matched CW envelope separation +13.4 dB; deep
> FT8 found 2 results versus 1 at conservative depth while remaining inside
> the automatic slot-budget guard. Final suite: 101 passed, 0 failed (before
> the bounded RTTY AFC follow-up; all 10 RTTY assertions also pass afterward).
>
> Implementation notes for future agents: the full one-dit CW boxcar merged
> short gaps, so the regression-safe measured setting is 0.35 dit. The scout
> gate restores the former 4 dB rule on crowded/unsettled frames. Audio-domain
> operator NR and 75% Welch overlap remain optional.
>
> **RECEIVER-CONTROL FOLLOW-UP: FINISHED — 2026-08-14.** The gain-policy
> probe from item 9 is now implemented, together with Soapy capability/readback
> reporting, correct SDRplay RFGR/IFGR gain-reduction control, per-band split-gain
> memory, percentile headroom supervision, hardware AGC setpoint control,
> automatic/override MW/FM rejection, DAB and IQ-correction controls, PPM
> correction, a selectable 250 kS/s low-IF acquisition path, and visible stream
> drop/clipping counters. Unsupported controls are reported instead of silently
> accepted, and an RSP opened through `miri` produces a backend warning. The
> low-IF path intentionally returns to 192 kS/s for FT8/FT4/AUTO because those
> modes require an exact 12 kHz audio clock. Verification for this follow-up:
> 106 passed, 0 failed, 26 ignored.

Goal: significantly improve weak-signal decode performance. Extra CPU is
acceptable everywhere; the 192 kHz front-end path currently costs a few
percent of one core, so there is a large budget available.

Current architecture (for orientation):

- `src/radio.rs` — SoapySDR worker thread. Capability discovery/readback,
  aggregate gain for generic receivers or split RFGR/IFGR gain reduction for
  SDRplay, notch/IQ/PPM settings, stream-drop and ADC-clip reporting. A slow
  percentile/floor-probing supervisor lives in `main.rs::supervise_hw_gain`.
- `src/dsp.rs::FrontEnd` — software DC removal (~2 Hz one-pole) and a
  **scalar** IQ imbalance correction (one gain + one cross term for the whole
  span), applied to every raw block in `main.rs::feed` before anything else.
- `src/dsp.rs::Spectrum` — Welch periodogram, Blackman-Harris, 50 % overlap.
- `src/dsp.rs::DecodeChain` — NCO → 511-tap decimating FIR (anti-alias) →
  variable-length audio FIR (channel filter). Time-domain convolution.
- `src/dsp.rs::SoftAgc` — hang AGC on the decoder audio path.
- Detection: `main.rs::scout_peaks` (local-prominence peaks over a ±40-bin
  median), mode scouts, `identify.rs` classifier.
- Decoders: CW (envelope + adaptive threshold), RTTY (FM discriminator),
  PSK31 (coherent, RC matched filter, AFC), FT8/FT4 (`mfsk-core`, already
  SIC + OSD + AP + local EQ).

The project has a strong bench-test culture (`#[ignore]` benches that print
comparisons, plus assertion tests that encode the measured result). **Every
item below must follow that pattern: build a synthetic-signal bench first,
make the change, keep the bench, and turn the measured improvement into an
assertion test.** Weak-signal claims that aren't measured this way should not
be merged.

Items are ordered by expected dB-per-effort. 1–4 are the core of the plan.

---

## 1. Set the tuner IF/analog bandwidth (trivial, do first)

`radio.rs` never calls `dev.set_bandwidth(Rx, 0, ...)`. The RSP1A's analog
IF filter therefore sits at whatever the driver defaults to, which can be
wider than the sample rate: strong out-of-span broadcasters then alias into
the span and raise the effective floor, and they eat ADC headroom that the
gain supervisor responds to by lowering gain — directly costing weak-signal
sensitivity.

- In `spawn()` and in the `Cmd::Rate` handler, after `set_sample_rate`, call
  `dev.set_bandwidth(Rx, 0, rate)` (the driver will pick the nearest legal
  IF bandwidth ≤ rate; log what was actually set via `dev.bandwidth()`).
- Ignore errors (some backends don't support it), as with the other settings.

## 2. Wideband impulse-noise blanker (biggest single win on HF)

Lightning static crashes and local impulse noise (mains, fences, VDSL) are
the dominant weak-signal killer on HF below ~20 MHz, and the project has
nothing for them. An impulse is broadband and brief: at 192 kHz it is a
handful of samples that momentarily dominate the whole span. Removing it
before the channel filters smear it into a many-millisecond "thump" is worth
several dB of effective SNR for every decoder at once, and it is *only*
possible at the wideband rate — after decimation the energy is already
smeared. This is why it must live in the front end, not in the decoders.

Implementation, in `dsp.rs` as a new `NoiseBlanker` struct, called from
`FrontEnd::process` (or immediately after it in `main.rs::feed`) so that the
spectrum, scouts, classifier and all decoders benefit:

- Track a robust background magnitude: a slow one-pole (τ ≈ 50–100 ms) of
  `|x|`, updated with the sample magnitude *clamped* to a small multiple of
  the current estimate so impulses cannot inflate their own reference
  (this is the cheap stand-in for a running median and works well).
- Detect: `|x| > k · background`, k ≈ 4–6 (make it a tunable with 2–3 UI
  steps plus off; default on, middle setting).
- Blank with context: when a sample trips, zero (or better, linearly
  interpolate across) a window extending ~2 samples before and after the
  run of hot samples, with a 2–3 sample raised-cosine edge so the blanking
  itself doesn't splatter.
- Safety valve: if the duty cycle of blanked samples exceeds ~2 % over a
  second, the "impulses" are actually a strong signal or overload — freeze
  blanking and (once) log it. This prevents the blanker chewing up strong
  CW on quiet spans.
- Report blanks/second in the status line next to the FrontEnd status.

Bench (follow `frontend_bench` style): synthetic band noise + a weak FT8-like
tone + Poisson impulses at realistic static-crash rates/amplitudes; measure
tone SNR in a narrow window before/after. Assert ≥ 10 dB improvement at
heavy-impulse settings and < 0.2 dB harm on impulse-free clean signal.

## 3. Frequency-dependent IQ image correction

`FrontEnd` corrects imbalance with a single gain/cross pair. Real tuner
front ends have imbalance that varies across a 192 kHz–1 MHz span (analog
filters differ between I and Q paths), so a scalar correction bottoms out
around 30–40 dB of image rejection. Weak-signal detection thresholds are
low enough that residual images of strong stations still appear as
plausible weak signals.

Replace the scalar solve with a per-sub-band solve, keeping the existing
time-domain DC removal:

- Split the span into K = 8–16 sub-bands. Once per block, FFT a 4096-sample
  stretch (reuse `rustfft`), and for each sub-band accumulate the same
  three statistics the scalar version uses (`ii`, `qq`, `iq`) — computed
  from the correlation between bin k and the conjugate of bin −k, which is
  the frequency-domain signature of imbalance.
- Solve per band with the same clamps and settle logic as now; smooth the
  per-band corrections over time (τ ≈ seconds) *and* across neighbouring
  bands.
- Apply as a short (15–31 tap) FIR on the Q channel synthesized from the
  per-band gain/cross values (frequency-sampling design), or apply the
  correction in an overlap-save FFT pass if item 5 is done first — then it
  is nearly free.
- Keep `FrontEnd::status()` reporting the *worst* band's rejection.

Bench: extend `bench_frontend` with a frequency-*dependent* synthetic
imbalance (gain/phase error ramping across the span); assert the new
correction beats the scalar one by ≥ 15 dB on the worst-band image while
leaving clean IQ untouched (existing `clean_iq_is_left_alone` must pass).

## 4. Per-bin noise-floor tracking for detection

`scout_peaks` measures prominence against a ±40-bin median of the *current*
smoothed spectrum, and `find_peaks_above` uses the whole-span median. Both
conflate "noise floor" with "whatever is there right now", so thresholds
must stay conservative (4 dB prominence / 10 dB SNR) to avoid false alarms,
and genuinely weak signals sit below them.

Add a `NoiseFloor` tracker in `dsp.rs`:

- Minimum-statistics style: per bin, track a slow floor estimate that
  follows downward quickly and upward only slowly (e.g. `f = min(x, f) `
  blended with `f += α·(x − f)` where α_up ≪ α_down; or a rolling minimum
  over ~10 s windows with bias correction). This gives a floor that signals
  cannot pull up.
- Feed it from the *unsmoothed* periodogram each `feed()`.
- Use it in `scout_peaks` / `find_peaks_above` as the reference instead of
  the local/global median (keep the local-median term as a QRM guard: use
  `max(tracked_floor, local_median − 3 dB)`).
- Then lower `PROMINENCE_DB` from 4.0 toward 2.5–3.0 and re-tune with the
  existing scout benches: the scouts' mix-down-and-match stage is the real
  false-alarm filter, so the candidate gate can afford to open up. This is
  where weak CW/PSK31 signals that currently never become candidates start
  being found.
- Also drive the waterfall auto-range (`floor_db`) from it, replacing the
  per-call sort.

Bench: synthetic span with weak carriers at graded SNRs; measure the lowest
SNR at which each becomes a scout candidate, before/after. Assert ≥ 2 dB
improvement with false-candidate rate (pure-noise span) no worse than today.

## 5. Fast-convolution channel filters (CPU headroom → sharper filters)

Convert `DecimFir`'s hot path to overlap-save FFT convolution when the tap
count is large (keep direct form below ~64 taps). 511 taps at 192 kHz and
1023 taps at audio rate in direct form is the current CPU ceiling; fast
convolution makes 4–8× longer filters cheaper than today's cost.

Then spend the headroom:

- Raise `RADIO_TAPS` so the anti-alias stage's stopband goes from the
  Blackman window's floor to ≥ 90 dB with a tighter transition — less
  wideband noise and fewer strong neighbours folding into the channel.
- Raise `AUDIO_TAPS_MAX` and tighten `want_tr` in `set_bandwidth` so the
  narrow CW/PSK filters (80/200 Hz) get genuinely steep skirts; adjacent
  QRM a few tens of Hz away is the normal weak-CW situation.
- Keep all four existing `DecodeChain` response tests passing; tighten the
  `decimation_does_not_alias_into_the_channel` bound from −60 to −80 dB.

Latency note: block-based convolution adds one block of delay; keep blocks
≤ 4096 audio samples so CW timing recovery is unaffected.

## 6. RTTY: matched-filter detector instead of FM discriminator

An FM discriminator is 3–6 dB worse than optimal at low SNR and fails
completely under selective fading (one tone faded out — common on HF, and
exactly when signals are weak). Replace the detection core in
`decoders/rtty.rs`:

- Two complex matched filters at the mark and space frequencies (boxcar or
  RC-weighted integrate over one bit at 45.45 baud, i.e. correlate with each
  tone), producing per-bit energies `Em`, `Es`.
- Decision variable `Em − Es` with per-tone envelope normalisation (ATC,
  automatic threshold correction: track each tone's own recent peak
  envelope and normalise by it) so a faded tone still slices correctly.
- Keep the existing start-bit framer, mid-bit averaging, and the
  normal/reversed dual-slicing logic — only the discriminator is replaced.
  Keep the shift/polarity detection.
- The mark/space frequencies come from the existing tuning; add a slow AFC
  (centroid of the two tone energies) within ±10 Hz.

Bench: synthetic 170 Hz-shift RTTY at graded SNR in band-limited noise, plus
a selective-fade case (space tone −20 dB); assert character error rate at a
given low SNR improves vs. the discriminator, and the fade case goes from
uncopyable to copyable.

## 7. CW: matched-filter envelope + two-state smoothing

`decoders/cw.rs` slices an envelope with an adaptive threshold — the right
structure, but at low SNR instantaneous slicing wastes several dB. Two
additive steps, keeping the existing clock recovery and element classifier:

- Matched filtering: once `morse_clock` has an estimated dit length, filter
  the envelope with a moving average of one dit (the matched filter for the
  shortest element) before slicing. Track WPM changes by re-deriving the
  window from the current clock estimate.
- Replace hard slicing with a two-state (mark/space) HMM smoothed by a
  short Viterbi (or forward-backward) pass over each buffered stretch:
  emission likelihoods from the observed on/off envelope distributions the
  decoder already estimates (`peak`, `floor`), transition probabilities
  from the dit clock. This suppresses both noise-spike marks and mid-dash
  dropouts, which is precisely how weak CW currently shreds.

Bench: extend the existing CW decode tests with graded-SNR synthetic keying;
assert copy at ≥ 2 dB lower SNR than the current slicer, and no regression
on clean 15–35 wpm signals in `decoders/tests.rs`.

## 8. FT8/FT4 decode depth (cheap knobs, pure CPU trade)

`decoders/ft8.rs` already uses mfsk-core well (sic_early, OSD, local EQ, AP
on own call). Knobs to turn, each a measured A/B on recorded or synthetic
slots:

- Raise `MAX_CAND` from 200 to 400–600 and lower the sync threshold
  (the `0.9` in `DecodeRequest::new`) a step — candidates are cheap, LDPC
  attempts are what cost, and OSD is already on.
- FT4: `sic_rounds(3)` → 4–5 if the mfsk-core API allows; measure.
- Add AP hints beyond `with_call2(my_call)`: standard CQ patterns, and
  hashed callsign table hints if mfsk-core supports them (it already
  maintains `CallsignHashTable`).
- Budget guard: decode runs in a worker slot; log decode wall time and back
  off (restore defaults) if a slot ever exceeds ~80 % of the slot period.

## 9. Optional / later

- **Audio-domain spectral NR** (Ephraim-Malah / decision-directed Wiener on
  the decoder audio path) as a UI toggle for CW/RTTY by ear — helps the
  operator, rarely helps the decoders; low priority.
- **Gain policy refinement — DONE in receiver-control follow-up**:
  `supervise_hw_gain` reacts to clipping and
  quiet, but doesn't verify the external-noise-dominates condition. Add a
  check: on gain changes, compare the tracked noise floor (item 4) step to
  the gain step; if the floor doesn't follow the gain up (ADC-noise
  limited), prefer more gain; if it follows 1:1 (externally limited), stop
  — extra gain only costs headroom. Keep the per-band gain memory.
- **75 % Welch overlap** in `Spectrum` for the scout path (more segments
  averaged per second → calmer floor → item 4 thresholds can drop further).
  Pure CPU trade; bench with `bench_window_tradeoff`. A 2026-08-14 trial
  regressed two PSK31 auto-detection tests (weak copy and LO blanking), so it
  remains deferred until its estimator weighting/timing is reworked.

## Suggested sequencing for implementation

1. Item 1 (minutes) and item 2 (the blanker) — largest real-world gain.
2. Item 4 (noise-floor tracker) and re-tune scout thresholds.
3. Item 3 (frequency-dependent IQ), item 5 (fast convolution) — item 5
   first if convenient, since it makes item 3's apply stage nearly free.
4. Items 6 and 7 (RTTY/CW detectors) — independent of everything above.
5. Item 8 (FT8 knobs) any time; it's isolated to `ft8.rs`.

Each item lands as its own commit with its bench output quoted in the commit
message, matching the existing history's style.
