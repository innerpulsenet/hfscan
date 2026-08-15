# hfscan

A terminal HF panadapter and digital-mode decoder for the SDRplay RSP1A, in Rust.
Zoomable spectrum + waterfall, a tunable cursor, a band scanner, signal hopping,
and CW, RTTY, PSK31, FT8 and FT4 decoding. Works with any SoapySDR device.

## Build and run

Needs Rust 1.88+ (the FT8 decoder library uses let-chains). If the system rustc
is older, `rustup` provides a current toolchain.

```bash
cargo build --release
```

```bash
./target/release/hfscan --freq 14074000 --mode ft8
```

Options:

| flag | default | meaning |
| --- | --- | --- |
| `--freq` | 14070000 | starting centre frequency, Hz |
| `--rate` | per band | sample rate, and therefore the width of the spectrum view; band presets set their own |
| `--low-if` | off | request the SDRplay 250 kS/s low-IF acquisition path (not used for FT modes) |
| `--ppm` | 0 | receiver frequency correction in parts per million |
| `--mode` | off | start with a decoder running: `off`, `cw`, `rtty`, `psk31`, `ft8`, `ft4` |
| `--fft` | 8192 | FFT size; higher gives finer frequency resolution |
| `--device` | `driver=sdrplay` | SoapySDR device arguments |
| `--call` | — | your callsign, for pskreporter spotting |
| `--grid` | — | your Maidenhead grid locator, for pskreporter spotting |

### Band spans

Each band preset carries its own sample rate and centre, applied by `b` / `B`.
The span covers the **CW and digital stretch at the bottom of the band**, not
the whole allocation — operators respect those boundaries, so the SSB portion
above holds nothing any decoder here can read.

| band | decoded | span | | band | decoded | span |
| --- | --- | --- | --- | --- | --- | --- |
| 160m | 1.800–1.850 | 192 kHz | | 17m | 18.068–18.115 | 192 kHz |
| 80m | 3.500–3.600 | 192 kHz | | 15m | 21.000–21.150 | 192 kHz |
| 60m | 5.330–5.405 | 192 kHz | | 12m | 24.890–24.930 | 192 kHz |
| 40m | 7.000–7.100 | 192 kHz | | 10m | 28.000–28.200 | 384 kHz |
| 30m | 10.100–10.150 | 192 kHz | | 6m | 50.240–50.360 | 192 kHz |
| 20m | 14.000–14.150 | 192 kHz | | 2m | 144.000–144.300 | 384 kHz |

Four constraints pin those numbers.

**Rates the receiver actually has.** The RSP1A offers 62.5, 96, 125, 192, 250,
384, 500, 768 and 1000 kS/s as fixed steps, then anything from 2 to
10.66 MS/s. Ask for something else and the driver clamps silently — and a band
centred for a width it never got can put its calling frequencies outside the
view, which means no decodes on that band at all.

**An exact audio clock.** Every span divides by 24 kHz, the LCM of the 8 kHz
and 12 kHz audio rates, so FT8 and FT4 keep an exact divisor.

**Room for the filter.** The tuner's analog filters are 200, 300, 600, 1536 and
5000 kHz and the driver will not pick one wider than the span, so a span sized
tightly to what it carries leaves the filter corner inside the view — which is
what edge rolloff on the waterfall is. Every decoded segment sits inside the
filter its span lands on.

**Time.** Every decoder slot mixes and decimates from the full input rate, so
the decode fleet costs CPU in proportion to it: a full fleet is 14% of real
time on a 192 kS/s span, 27% at 384, 70% at 768 and 840% at 5 MS/s
(`bench_feed_cost_per_band`). The UI is single-threaded and the radio drops
blocks rather than blocking, so the last two stutter or cannot run. Sizing the
span to the decoded segment rather than the allocation is what keeps almost
every band at 192 kS/s. A hard budget on slots × sample rate backs this up, so
a `--rate` override shrinks the fleet instead of overrunning the stream.

### Measuring the receiver

`--bench` characterises the hardware instead of starting the UI:

```bash
./target/release/hfscan --bench
```

It reports what the driver actually offers (sample rates, filter widths, gain
elements and their ranges), requests every band preset in turn and checks the
achieved rate and filter against what the plan assumed, then sweeps the gain
and finds the knee — the setting where the noise floor stops being the
receiver's own and starts following the band's. Past that point more gain buys
intermodulation and lost headroom, not sensitivity.

Run it after any antenna change: the knee follows band noise, and on this
receiver the quiet bands wanted 16 dB more gain than the loud ones.

| band | IFGR | | band | IFGR | | band | IFGR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 160m | 56 | | 40m | 56 | | 15m | 48 |
| 80m | 52 | | 30m | 56 | | 12m | 48 |
| 60m | 56 | | 20m | 48 | | 10m | 40 |
| 17m | 48 | | 6m | 40 | | 2m | 56 |

RFGR stays at 0 on every band. The sweeps came nowhere near overload, and RF
gain reduction is the one that costs noise figure, so there is nothing to buy
by raising it; the IF figure carries the whole adjustment. These load as each
band's starting gain.

It is device-agnostic. An RSP1A expresses both its gain elements as *reduction*
(a bigger number is less gain) while an RTL-SDR's tuner gain counts the usual
way; the sweep reads the direction off the data rather than keeping a table of
which is which.

What it cannot do is give a noise figure or an MDS in absolute terms — both
need a calibrated source. The floor is known relative to full scale, which is
enough to find the knee and not enough to put a number in dBm on it.

## Keys

| key | action |
| --- | --- |
| `←` `→` | tune by one step (with shift, 10 steps) |
| `↑` `↓` | scroll the decode transcript (with shift, 10 lines) |
| wheel | scroll the pane under the mouse — messages, stations, activity, and the waterfall's history |
| `z` / `Z` | zoom in / out — this also sets the tuning step |
| `n` / `N` | jump to the next / previous signal (in CW/PSK31: next confirmed tone) |
| `p` | CW/PSK31: lock the next signal in the span, or walk the band for more |
| `u` / `i` | CW/PSK31: fine-tune the lock −2 / +2 Hz |
| `g` | CW/PSK31: centre the cursor on the locked tone |
| `[` `]` | retune the centre ±10 kHz |
| `PgUp` `PgDn` | retune the centre by half a span |
| `c` | recentre the radio on the cursor |
| `b` / `B` | next / previous band preset |
| `d` | cycle decoder: off → CW → RTTY → PSK31 → FT8 → FT4 |
| `r` | RTTY normal/reverse shift |
| `s` | scan the current band and list what it finds (with signature labels) |
| `v` | enlarge the decode pane (cycles normal / large / huge) |
| `w` / `W` | waterfall speed (100 ms … 2 s per row) / hi-res frequency mode |
| `f` / `F` | FFT size — frequency resolution (1024 … 32768) |
| `a` | cycle AGC: soft hang → hardware → off; `+` / `-` request more/less manual gain |
| `;` | cycle hardware AGC setpoint: −40 / −30 / −20 dBFS |
| `m` | cycle SDRplay MW/FM notch: auto / forced on / forced off |
| `D` / `I` | toggle SDRplay DAB notch / driver IQ correction |
| `y` / `Y` | frequency correction −/+ 0.1 ppm |
| `h` | switch 192 kS/s zero-IF / 250 kS/s low-IF acquisition |
| `e` | spectrum smoothing: light / medium / heavy |
| `l` | RX bandpass: auto (mode default) / 80 / 200 / 500 / 1500 / 3000 Hz |
| `k` | squelch on/off; `,` / `.` adjust the threshold |
| `<` / `>` | copy floor: hide decodes the decoder itself is not confident in |
| `t` | toggle the bias-T (external preamp power) |
| `o` | station settings: your callsign and grid locator |
| `x` | clear the decode pane |
| `?` / `q` | help / quit — the help lists every key with its **current setting** in yellow |

### Zoom is the tuning control

The cursor step is 1/200th of the visible width, so zoom and tuning precision
move together. At full 192 kHz span a step is ~960 Hz — coarse, but each press
visibly moves the display. Press `z` a few times and the view narrows, the step
shrinks to tens of Hz, and the same arrow keys become a fine tuning knob. The
view stays centred on the cursor, so the spectrum scrolls under a fixed marker.

### Resolution

Two independent controls. `f` sets the FFT size, and the status bar shows the
result in Hz/bin: at a 192 kHz span, 1024 points gives 188 Hz/bin while 32768
gives 5.9 Hz/bin — fine enough to separate individual FT8 signals, which sit
6.25 Hz apart. Large FFTs are built by accumulating IQ across blocks, so the
spectrum updates less often as resolution rises. `z` then decides how much of
that detail is on screen.

The waterfall packs two time steps into each terminal row using half-block
glyphs (upper half in the foreground colour, lower half in the background), so
a given pane height shows twice the history. Pressing `W` trades that for
twice the frequency resolution instead: one time step per row, two frequency
bins per column via left-half glyphs. The mouse wheel scrolls the waterfall
back through its history, and scrolls whichever decode pane is under the
cursor.

## How it works

`radio.rs` owns the SoapySDR device on its own thread and ships IQ blocks over a
bounded channel, dropping blocks rather than stalling the radio if the UI falls
behind. `dsp.rs` computes the averaged periodogram for the display, and runs the
decode chain: an NCO shifts the cursor frequency to 0 Hz, then a decimating
windowed-sinc FIR drops the rate to the audio rate the mode wants, with a
second FIR at the audio rate to sharpen the skirt. `l` selects a software
bandpass (auto uses the mode default; 80 Hz through 3 kHz override it).

The spectrum is a Welch periodogram (50 % hop) with a binomial frequency
smooth that keeps carriers sharp, then a slow time average. `e` cycles
light / medium / heavy. The colour scale tracks the noise floor slowly so
it does not flicker.

`e` is a display control only. The classifier, both narrowband scouts, the band
scanner and the cursor SNR read a separate copy of the spectrum at fixed
smoothing — otherwise a cosmetic preference becomes a detection parameter,
since "heavy" broadens every peak, merges CW signals sitting close together,
and slows the response to a station coming up.

AGC defaults to a *hang* loop on the decoder path: it ducks quickly when
the audio is hot, holds for ~0.8 s, then creeps gain back only if the
signal stays quiet. A slow supervisor uses the 99.9th-percentile converter
level (not merely rail hits) and stops adding gain once a measured gain probe
shows that external noise already dominates. `a` cycles hang → device AGC →
manual (`+` / `-`); `;` selects the hardware AGC target where supported.

At startup hfscan inventories the actual Soapy backend, gain elements and
readable controls, then reads settings back after changes. With SoapySDRPlay3,
manual gain is deliberately split into `RFGR` (coarse RF/LNA gain reduction)
and `IFGR` (fine IF gain reduction): larger numbers mean *less* gain. Other
receivers retain conventional aggregate gain in dB. If an SDRplay device is
opened through the `miri` backend, hfscan warns that these SDRplay-specific
controls are unavailable rather than presenting controls that do nothing.

The MW/FM rejection network defaults to automatic: on above 2 MHz and off
while receiving MW/LW. `m` can override it. The DAB notch is left off on HF.
Driver IQ correction is enabled when available, with the software front end
continuing to remove residual imbalance. The status line shows low-IF mode,
active notches, PPM correction, clipping, and dropped IQ blocks.

Changing band (`b` / `B`, or a retune that leaves the band) resets the
spectrum scale, waterfall, scout list and hang AGC, and restores the
hardware gain last used on that band (both RFGR and IFGR on SDRplay). Otherwise a hot band can leave the
colour scale and front-end gain wound up so the original band looks
empty — as if a filter had switched on.

The waterfall accumulates the peak between rows rather than pushing one row per
IQ block, so its scroll rate is set by `w` and not by the sample rate.

### Signature matching

The radio already delivers the whole span (typically ~192 kHz) in one IQ
stream. About once a second, occupied slices of that span are mixed to
baseband and scored the way a person reads a waterfall: occupied bandwidth,
tonality, tone spacing, envelope keying, a residual carrier, and (for
CW / PSK31) the same probes the hop-scouts use. The result is a label —
**CW**, **PSK**, **RTTY**, **FT8**, **FT4**, **SSB**, **AM**, **CAR** — not a
decode. Live signals get a coloured chip on the spectrum sitting on a
shelf as wide as the occupancy. Chips hold for a few seconds and
fade rather than blinking off when a classify pass misses. A thin **activity** strip at the bottom
keeps every detection in this span (frequency, mode, SNR) after the
chip fades, most recently heard first so the row leads with what is on the band
now rather than with whatever sits lowest in it. Beneath it, a second row
carries what the receiver has to say for itself — mode changes, retunes, spot uploads, warnings. The two are kept
strictly apart: detections are chips and only chips, because in auto mode on a
busy band they arrive fast enough to bury every message worth reading. Messages
are short and arrive in bursts, so as many as fit are shown, newest first and
leftmost, the older ones dimmed; only the newest is ever truncated. Wheel over
the strip to page back through earlier messages; an `↑n` marks how far back you
are. When spotting is on, the strip's title also breaks the spots down by mode
— `spots FT8 132 CW 41 PSK31 7` — which the running total on the status line
cannot tell you: with four decoders sharing a span, it is worth knowing that
three of them have never produced anything. Band scan (`s`) prints
the same tags next to each hit. Park the cursor on a chip and press `d`
to copy it.

In auto mode every classified signal gets its own decoder, up to 24 narrowband
ones at a time, spent strongest-first when the band offers more than that. FT8
and FT4 do not count against the limit — their slots are pinned to the calling
frequencies whether or not anyone is on them. A slot costs about 0.65% of a
core (`bench_slot_cost`), so the limit is set by how much of a busy CW segment
is worth carrying rather than by CPU; a signal that never gets a slot is a
station nobody hears about.

The order the tests run in matters as much as the tests themselves, because
the first one to confirm wins. FSK goes first: a signal alternating between
two tones cannot be anything else, whereas "looks like BPSK" and "looks like
keyed CW" are both things a *single* RTTY tone satisfies. RTTY idles on mark
and returns to mark between characters, so its mark tone is a carrier keying
on and off — and a BPSK detector reads keying on and off as symbols. With the
PSK31 probe running first, ordinary RTTY was labelled PSK31 and handed a
decoder, which is where the nonsense in the pane came from.

Two tones are detected from the *instantaneous frequency*, not from two peaks
in an averaged spectrum. The peak-pair test needs the weaker tone within about
6 dB of the stronger, which mostly-mark traffic never delivers; instantaneous
frequency is bimodal whatever the duty cycle between the tones. That also
measures the shift, so the 425 and 850 Hz shifts in amateur use are decoded at
the shift they are actually sent on instead of framed as if they were 170 Hz.

Backing that up, the PSK31 confirmation now checks *the rate the signal keys
at*. Energy at DC, symbols on the real axis and a plausible reversal rate are
all things a keyed carrier satisfies; keying 31.25 times a second is what
PSK31 **is**. The envelope's clock line sits at 31.25 Hz for PSK31, 45.45 (or
22.7) for RTTY, 15–25 for hand-sent CW, and nowhere for an unkeyed carrier —
so a candidate whose line is clearly somewhere else is rejected however BPSK
it looks. It is framed as a veto rather than a requirement on purpose: the
line's absolute level depends on how the audio arrived (the span scout's
decimation costs about 4 dB against the decoder's own filtered baseband), and
a level threshold tuned on one path quietly stops identifying real signals on
the other. The comparison survives that difference; the level does not.

### The decoders

- **CW** — envelope detection with hysteresis and a peak/noise-floor tracker.
  Dit length is estimated from a short/long cluster of recent marks, so a
  station that speeds up or slows down is followed instead of being decoded
  as garbage. After a pause the decoder re-acquires quickly for the next
  over. A passband scout finds keyed tones near the cursor (cyan ticks;
  `n` / `N` hop). `p` locks the next CW in the span or walks the band.
  Status shows estimated WPM and the lock offset. In CW mode the decode
  pane becomes three views, like FT8: a live **envelope** of the keying
  (green while the key is down), the **copy** transcript, and a **tuner**
  with absolute RF, lock offset, a ±20 Hz centre-frequency meter, WPM,
  and the tones in the passband. `u` / `i` trim the lock 2 Hz; `g`
  centres the cursor on it.
- **RTTY** — FM discriminator, start-bit clock recovery, 45.45 baud, ITA2 with
  LTRS/FIGS shifts. Reports the percentage of correctly framed characters,
  which is a good tuning indicator. The shift defaults to 170 Hz; in auto mode
  the classifier measures it and the slot is built with it, snapped to the
  nearest standard (170 / 425 / 850) so a noisy measurement cannot detune the
  matched filters.
- **PSK31** — differential BPSK, so no carrier recovery loop is needed. The
  decoder identifies a nearby PSK31 signal (within ~180 Hz of the cursor) by
  squaring the baseband — that wipes the modulation and leaves a tone at
  twice the offset — then confirms the candidate is actually PSK31 (BPSK
  symbols on the real axis, a reversal rate that is neither a dead carrier
  nor noise) and mixes it onto DC. A raised-cosine matched filter, envelope
  timing, symbol AGC and a slow AFC keep the lock calibrated as the signal
  drifts. The squelch is bypassed so a nearby signal can be found even when
  the cursor is sitting on noise.

  A span scout also watches the whole radio view: energy peaks are mixed
  down and scored the same way. Confirmed hits are marked in cyan on the
  spectrum. `n` / `N` hop to the next / previous PSK31 (first another
  signal already in the passband, then the next one in the span). `p`
  locks the next hit, or if the span is empty walks the current band,
  lists what it finds, and parks on the lowest one. In PSK31 mode the
  decode pane becomes three views: an **eye** (I/Q constellation of
  recent symbols plus the baseband envelope), the **copy** transcript,
  and a **tuner** with absolute RF, lock offset, AFC residual, quality
  and the signals in the passband. `u` / `i` trim the lock 2 Hz; `g`
  centres the cursor on it. Idle `DE CALL CALL` /
  `CQ CALL` decodes are spotted to pskreporter when a callsign is
  configured.
- **FT8 / FT4** — slot-based, so these work differently: the cursor is the
  *dial* frequency and the decoder listens to the whole 200–3000 Hz passband
  above it, exactly as WSJT-X does. Audio is buffered for the full 15 s (FT8) or
  7.5 s (FT4) UTC slot, then handed to a worker thread; the squelch is bypassed
  and the status line shows slot fill and decode count. Sync, LDPC and message
  unpacking come from [`mfsk-core`](https://crates.io/crates/mfsk-core).
  When a callsign is configured, messages addressed to you are a-priori
  locked (a couple of dB extra sensitivity on those candidates) and a
  running hash table resolves `<...>` placeholders.

  Slot audio is buffered as raw f32 and normalised with a single slot-wide
  gain at slot end, so a strong station starting mid-slot can neither clip
  against the i16 rails nor pump the gain under a decode in progress. Decoding
  runs multi-pass with successive interference cancellation (FT8: WSJT-X's
  early-decode checkpoints; FT4: three SIC rounds), local Costas
  equalisation and OSD fallback: each decoded signal is subtracted from the
  buffer and the residual decoded again, which recovers weak stations sitting
  inside a strong neighbour's occupied bandwidth — the exact situation a hot
  front end (bias-T feeding an LNA) creates. If slots ever queue up faster
  than they decode, only the freshest is kept.

  If the SDR's ADC itself clips (the classic bias-T/LNA symptom: signals hit
  full scale and splatter intermod across the passband, costing weak decodes),
  the status line warns `ADC overload: reduce gain`. Software can flag it, but
  only backing the gain off — or dropping the bias-T — fixes it.

  These modes also replace the plain decode pane with a three-part view of the
  whole passband's traffic:

  - **activity** — one row per decoded slot (newest first, marked `>`), one
    column per slice of the audio passband; each cell's colour is the SNR of
    the decode there. A QSO appears as two frequencies lighting up on
    alternate rows as the stations take turns. The title and the tick axis
    along the bottom read absolute RF (dial + audio offset).
  - **messages** — one line per decode, newest at the bottom and bold: UTC
    time as `hh:mm:ss`, SNR, timing offset, absolute RF frequency and the
    message text. CQ calls are cyan; long lines truncate with `…` (press `v`
    for a wider pane).
  - **stations** — every callsign heard, in the order first heard, so entries
    update in place instead of reshuffling each slot: best SNR, age and
    absolute frequency. CQ callers are cyan; stations silent for over five
    minutes go grey and are dropped only when the table outgrows the pane.

  The spectrum shades the monitored 200–3000 Hz window above the dial while
  either mode is active, and a frequency axis runs along the bottom of the
  spectrum. Press `v` to give the traffic panes more of the screen. PSK31
  shades its search window and marks a locked carrier in cyan.

### Reporting spots to pskreporter.info

hfscan can report the stations it decodes to
[pskreporter.info](https://www.pskreporter.info), using the IPFIX feed format
described in [pskdev.html](https://www.pskreporter.info/pskdev.html). Set your
identity with `o` in the app (a small dialog with callsign and Maidenhead grid
fields), or on the command line:

    hfscan --call M0ABC --grid IO91vl

Settings are saved to `~/.config/hfscan/config.toml`; command-line flags
override the file for that run. Once a callsign is configured, every FT8/FT4
decode whose *sender* callsign is parsed out of the message (e.g. `CQ G4XYZ
IO91` or `M0ABC G4XYZ -12`) becomes a spot carrying the sender call, absolute
RF frequency, SNR, mode, and the sender's grid locator when the message
contains one. Your own callsign and grid identify the receiver.

CW, RTTY and PSK31 are spotted too, from three forms:

- `CQ <call>`, including the contest and activity variants — `CQ TEST <call>`,
  `CQ POTA DE <call>`, `CQ FIELD DAY DE <call>`.
- `DE <call> <call>`, the classic idle. A single `DE <call>` counts as well.
- `<addressee> <call>`, the exchange: two callsigns running together, where the
  second is the station transmitting. This is the same "to, from" ordering FT8
  packs into its message fields, and on CW and RTTY it is the commonest form of
  all — every contest exchange and every turn of an ordinary QSO is one. An
  addressee repeated before the sender, `W1AW W1AW K1ABC`, is read correctly.

The exchange form has no keyword in front of it to establish that a callsign is
what was meant, so it applies a stricter test: a callsign's separating digit
follows the prefix and its suffix is letters, which is what keeps the `5NN` of
a CW signal report and the `001` of a serial number out of the callsign slot.

These modes carry no grid locator, and their copy has to clear the confidence
floor (`[`/`]`) before it is reported at all — a callsign scraped out of noise
is worse on pskreporter than it is on screen. The reported SNR is referred to a
2500 Hz bandwidth, measured from the mark/space envelope levels for CW and from
the passband noise for RTTY.

The software id sent with each report is `HFScan vX.Y.Z`. Nothing containing a
digit may precede the `v`: pskreporter.info treats the first digit in the field
as the start of the version and truncates the displayed name there.

Reports are batched and sent to `report.pskreporter.info:4739` at most once
every five minutes (plus a random delay, as the site asks); the first batch
goes out about five minutes after startup. A station is re-reported at most
once per hour per band and mode, unless it moves band. The status line shows
`spots N`, the running count of queued/sent spot records. Spotting only kicks
in once a callsign is set — without one, nothing is transmitted.

A squelch gates the streaming decoders on the SNR measured in the cursor's
passband — without it, decoders cheerfully turn noise into text.

### The copy floor

Every continuous-mode decoder reports how much of what it is emitting is real
copy rather than noise read as characters, on a scale where **0 is band noise
and 1 is every symbol resolved cleanly**. It is the `sig` column in the decode
pane, next to the sending speed (`31bd`, `18wpm`), and it is what `<` / `>`
threshold: copy from a decoder below the floor is dropped instead of printed,
and is not spotted to pskreporter either. The pane border shows the current
floor and how many decoders it is holding back, so a quiet pane is never a
mystery. FT8 and FT4 decide per transmission and report their own SNR in that
column instead.

The scale is calibrated per mode rather than merely trending the right way,
because a single threshold has to mean the same thing everywhere. PSK31 is the
instructive case: its raw measure is the mean of |cos θ| over differential
symbols, and for noise θ is uniform, so it settles at 2/π = 0.64 — *not* zero.
Every threshold below that was one noise could never fail, which is why a bad
lock used to fill the pane with plausible-looking varicode. Subtracting the
noise floor and rescaling puts band noise at 0.18–0.25 and a signal 10 dB out
of the noise — copied at 92% accuracy — at 0.53, so the default floor of 40%
sits in the gap. `decoders::tests::bench_psk31_confidence` prints the curve.

### Tuning tips

- **CW**: put the cursor near the tone (within ~180 Hz) or press `p` / `n`
  and let the scout find one. Speed is tracked automatically; a pause
  re-acquires if the next station is a different WPM.
- **RTTY**: centre the cursor *between* the mark and space tones. If the text is
  garbage, press `r` — the sideband convention flips the shift.
- **PSK31**: put the cursor *near* the carrier (within ~180 Hz), or press `p`
  / `n` and let the scout find one. Confirmed signals are cyan ticks on the
  spectrum; the lock itself is brighter cyan. `n` / `N` walk them. If the
  span is empty, `p` walks the current band.
- **FT8/FT4**: put the cursor on the dial frequency (14.074, 7.074, 14.080 for
  FT4 …), not on an individual signal. Press `b` to reach the band, then tune to
  the marked frequency.

  **The system clock must be right.** FT8 slots are UTC-aligned and the decoder
  tolerates only ~2 s of error, so an unsynchronised clock produces exactly zero
  decodes no matter how strong the signals are. Check it with:

  ```bash
  timedatectl show -p NTPSynchronized
  ```

  If that reports `no`, `sudo apt install systemd-timesyncd` and enable it. The
  status line shows a UTC clock and a countdown to the next slot boundary —
  check it against a known-good clock. The other health check is `last NN%`:
  it is the fraction of the previous slot actually captured, and anything well
  below 100% means IQ blocks are being dropped.

## Tests

```bash
cargo test --release
```

Every decoder is tested by synthesising a signal for that mode — shaped CW
keying, Baudot FSK, raised-cosine BPSK31 with a frequency offset, and encoded
FT8/FT4 slots including an off-centre one — pushing it through the decoder and
checking the text comes back. This is what actually validates the DSP; on-air
behaviour additionally depends on your antenna and conditions.

`cargo test --release -- --ignored --nocapture` also runs a PSK31 diagnostic
that prints the transmitted and recovered bit streams side by side, which is the
fastest way to spot a timing-recovery regression.

## Licence

`mfsk-core` is GPL-3.0-or-later, so this binary as a whole is GPL-3.0-or-later.
Dropping the FT8/FT4 modes would remove that constraint.
