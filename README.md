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
| `--rate` | 192000 | sample rate, and therefore the width of the spectrum view |
| `--mode` | off | start with a decoder running: `off`, `cw`, `rtty`, `psk31`, `ft8`, `ft4` |
| `--fft` | 8192 | FFT size; higher gives finer frequency resolution |
| `--device` | `driver=sdrplay` | SoapySDR device arguments |
| `--call` | — | your callsign, for pskreporter spotting |
| `--grid` | — | your Maidenhead grid locator, for pskreporter spotting |

The RSP1A also supports 62500, 96000, 125000, 250000, 384000, 500000 and
1000000. FT8/FT4 need a rate that divides evenly by 12 kHz, so selecting them
switches the radio to 192 kHz automatically if the current rate will not do.

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
| `a` | cycle AGC: soft hang → hardware → off; `+` / `-` set manual gain |
| `e` | spectrum smoothing: light / medium / heavy |
| `l` | RX bandpass: auto (mode default) / 80 / 200 / 500 / 1500 / 3000 Hz |
| `k` | squelch on/off; `,` / `.` adjust the threshold |
| `t` | toggle the bias-T (external preamp power) |
| `o` | station settings: your callsign and grid locator |
| `x` | clear the decode pane |
| `?` / `q` | help / quit |

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

AGC defaults to a *hang* loop on the decoder path: it ducks quickly when
the audio is hot, holds for ~0.8 s, then creeps gain back only if the
signal stays quiet. A slow supervisor trims the hardware gain if the ADC
is clipping or starved, so the spectrum does not pump. `a` cycles hang →
device AGC (fast, can pump) → manual (`+` / `-`).

Changing band (`b` / `B`, or a retune that leaves the band) resets the
spectrum scale, waterfall, scout list and hang AGC, and restores the
hardware gain last used on that band. Otherwise a hot band can leave the
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
decode. Labels sit on the spectrum and a tally (`3 CW  2 SSB`) is in the
status line and the spectrum title. Band scan (`s`) prints the same tags
next to each hit. Park the cursor on a label and press `d` to copy it.

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
- **RTTY** — FM discriminator, start-bit clock recovery, 45.45 baud / 170 Hz
  shift, ITA2 with LTRS/FIGS shifts. Reports the percentage of correctly framed
  characters, which is a good tuning indicator.
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
contains one. PSK31 spots are taken from `DE CALL CALL` and `CQ CALL`
decodes. Your own callsign and grid identify the receiver.

The software id sent with each report is `HFScan vX.Y.Z by KQ2Y`. The version
is placed before the callsign so pskreporter.info does not treat the `2` in
`KQ2Y` as the start of the version and display the name as `HFScan by KQ`.

Reports are batched and sent to `report.pskreporter.info:4739` at most once
every five minutes (plus a random delay, as the site asks); the first batch
goes out about five minutes after startup. A station is re-reported at most
once per hour per band and mode, unless it moves band. The status line shows
`spots N`, the running count of queued/sent spot records. Spotting only kicks
in once a callsign is set — without one, nothing is transmitted.

A squelch gates the streaming decoders on the SNR measured in the cursor's
passband — without it, decoders cheerfully turn noise into text.

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
