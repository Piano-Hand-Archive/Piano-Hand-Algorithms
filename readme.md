# Robotic Piano Hand Algorithm

Generates physically-valid, hardware-ready servo commands for a two-handed robotic piano player from a MusicXML score. The system finds optimal thumb positions and finger assignments for both hands at every timestep, respecting real physical constraints: hand span, movement speed, black-key reach, finger collision avoidance, and sustained-note locking.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [The Full Pipeline — Input to Output](#the-full-pipeline--input-to-output)
  - [Stage 1 — MusicXML Parsing](#stage-1--musicxml-parsing)
  - [Stage 2 — Timed Steps CSV](#stage-2--timed-steps-csv)
  - [Stage 3 — Find the Split Point](#stage-3--find-the-split-point)
  - [Stage 4 — Hand Assignment](#stage-4--hand-assignment)
  - [Stage 5 — Path Optimization (Dynamic Programming)](#stage-5--path-optimization-dynamic-programming)
  - [Stage 6 — Safety Validation](#stage-6--safety-validation)
  - [Stage 7 — Servo Command Generation](#stage-7--servo-command-generation)
  - [Stage 8 — Save All Outputs](#stage-8--save-all-outputs)
- [Output Files Reference](#output-files-reference)
  - [Servo Command Format](#servo-command-format)
  - [Finger Command Notation](#finger-command-notation)
- [findOptimalHandPos.py — Complete Argument Reference](#findoptimalhandpospy--complete-argument-reference)
- [verify_fingering.py — Complete Argument Reference](#verify_fingeringpy--complete-argument-reference)
- [Example Workflows](#example-workflows)
- [Tuning Guide](#tuning-guide)
- [Troubleshooting](#troubleshooting)

---

## Project Structure

```
Piano-Hand-Algorithms-1/
├── inputs/                         # Place all MusicXML input files here
│   ├── twinkletwinkle.musicxml
│   ├── maryhadlamb.musicxml
│   ├── hbd.musicxml
│   ├── hotcrossbuns.musicxml
│   ├── starspanbanner.musicxml
│   └── fuyunohanashi1.musicxml
├── outputs/                        # All generated files land here (auto-created)
│   ├── left_hand_commands.txt      # Hardware commands for left hand
│   ├── right_hand_commands.txt     # Hardware commands for right hand
│   ├── fingering_plan.csv          # Human-readable plan with finger assignments
│   ├── fingering_summary.csv       # Statistics and configuration used
│   ├── timed_steps.csv             # Intermediate: parsed notes with timing
│   └── conflict_resolutions.txt   # Only written when adjacent-key conflicts occur
├── findOptimalHandPos.py           # Main optimizer
└── verify_fingering.py             # Output verifier / safety checker
```

---

## Setup

**Requirements:** Python 3.9+, music21

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows

# Install the only dependency
pip install music21
```

---

## Quick Start

```bash
# 1. Activate the environment
source .venv/bin/activate

# 2. Run the optimizer — interactive file picker appears automatically
python findOptimalHandPos.py

# 3. Verify the output is physically valid
python verify_fingering.py
```

When no filename is given, the optimizer lists everything in `inputs/` and prompts you:

```
Available input files:
  1. fuyunohanashi1.musicxml
  2. hbd.musicxml
  3. hotcrossbuns.musicxml
  4. maryhadlamb.musicxml
  5. starspanbanner.musicxml
  6. twinkletwinkle.musicxml

Select a file [1-6]: 4
```

You can also pass a filename directly — the `inputs/` folder is resolved automatically:

```bash
python findOptimalHandPos.py maryhadlamb.musicxml
# same as: python findOptimalHandPos.py inputs/maryhadlamb.musicxml
```

---

## The Full Pipeline — Input to Output

The diagram below shows every stage the data passes through from a `.musicxml` file to hardware-ready commands:

```
inputs/song.musicxml
        │
        ▼
 ┌─────────────────┐
 │  Stage 1        │  parse_musicxml()
 │  Parse MusicXML │  Extracts notes, timing, black/white key flags
 └────────┬────────┘
          │ note_info[]
          ▼
 ┌─────────────────┐
 │  Stage 2        │  convert_to_timed_steps()  +  save_timed_steps_csv()
 │  Timed Steps    │  Groups notes by onset time → outputs/timed_steps.csv
 └────────┬────────┘
          │ note_groups[]
          ▼
 ┌─────────────────┐
 │  Stage 3        │  find_optimal_split_point()  OR  find_dynamic_split_points()
 │  Split Point    │  Divides the keyboard between left and right hands
 └────────┬────────┘
          │ split_point / split_sequence
          ▼
 ┌─────────────────┐
 │  Stage 4        │  assign_hands_to_notes()
 │  Hand Assign    │  Routes each note to left or right hand
 └────────┬────────┘
          │ l_groups[], r_groups[]
          ▼
 ┌─────────────────┐
 │  Stage 5        │  optimize_with_boundaries()  (Viterbi DP, per hand)
 │  DP Optimize    │  Finds lowest-cost thumb path through all timesteps
 └────────┬────────┘
          │ l_path[], r_path[]
          ▼
 ┌─────────────────┐
 │  Stage 6        │  validate_output()
 │  Safety Check   │  Velocity, hand gap, reachability warnings
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │  Stage 7        │  generate_servo_commands()
 │  Servo Commands │  Converts paths → timestamped step/servo command strings
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │  Stage 8        │  save_outputs()
 │  Write Files    │  Writes all output files to outputs/
 └─────────────────┘
          │
          ▼
 outputs/left_hand_commands.txt
 outputs/right_hand_commands.txt
 outputs/fingering_plan.csv
 outputs/fingering_summary.csv
 outputs/timed_steps.csv
```

---

### Stage 1 — MusicXML Parsing

**Function:** `parse_musicxml(file, auto_transpose)`

The score is loaded with [music21](https://web.mit.edu/music21/). Every note in every part is extracted with:

- **MIDI number** — standard 0–127 pitch encoding
- **Start time** — onset position in seconds
- **Duration** — how long the note is held (used for sustained-note locking)
- **White key index** — unified position on the 88-key keyboard (0 = A0, 52 = C8). White keys increment by 1; black keys share the index of the adjacent lower white key
- **is_black** — True if the note requires a black key

**`--transpose` mode (optional):** If enabled, music21 detects the key signature and transposes the entire score to C major (for major keys) or A minor (for minor keys). This removes all black keys at the cost of changing the actual pitches. Only use this if your hardware cannot reach black keys.

Chords are unpacked into individual notes. Rests are skipped.

---

### Stage 2 — Timed Steps CSV

**Function:** `convert_to_timed_steps()` → `save_timed_steps_csv()`

Notes that share the same onset time are grouped into a single timestep. The result is written to `outputs/timed_steps.csv` immediately. This file serves two purposes:

1. **Input to the optimizer** — `load_notes_grouped_by_time()` reads it back in Stage 5
2. **Ground truth for the verifier** — `verify_fingering.py` compares the optimizer's output against this file to confirm correctness

CSV columns: `start_time`, `midi`, `duration`, `white_key_index`, `is_black`

---

### Stage 3 — Find the Split Point

The **split point** is a white key index that partitions the keyboard. Notes with index ≤ split go to the left hand; notes with index > split go to the right hand (subject to the `--gap` constraint).

**Static split (default):**
`find_optimal_split_point()` evaluates every possible split position across the whole piece. The score for each candidate counts how often notes cross the boundary plus how much total hand movement each hand would require. The lowest-cost split is chosen once and used for the entire piece.

**Dynamic split (`--dynamic-split`):**
`find_dynamic_split_points()` divides the piece into fixed-size segments (`--segment-size` timesteps each) and independently optimises the split for each segment. Between adjacent segments the split can shift by at most `--split-max-change` keys; shifting incurs a `--split-change-penalty` cost. This helps when the note density or register shifts dramatically mid-piece.

---

### Stage 4 — Hand Assignment

**Function:** `assign_hands_to_notes()`

Each timestep's notes are assigned to left or right hand based on the split point. A note is "left hand" if its white key index is at or below the split; "right hand" if above.

**Adjacent key conflict resolution:** When two notes at the same timestep land on the boundary (one white key apart and straddling the split), both hands could claim them. The resolver tries all valid assignments and picks the one with the lowest combined movement cost. Resolved conflicts are logged to `outputs/conflict_resolutions.txt`.

---

### Stage 5 — Path Optimization (Dynamic Programming)

**Function:** `optimize_with_boundaries()`

This is the core of the system. Each hand is optimised independently using a **Viterbi-style DP search**.

**State:** The thumb position (white key index) at a single timestep.

**At each timestep** every candidate thumb position is scored. The total cost from the start to that state is:

```
cost(state) = cost(best_predecessor) + transition_cost + emission_cost
```

**Transition cost** (moving from one timestep to the next):
- `MOVE_PENALTY × |new_thumb − old_thumb|` — distance penalty
- `VELOCITY_PENALTY` (large) if the move exceeds `--speed` keys/second
- `LOOKAHEAD_VELOCITY_PENALTY` if any of the next `--lookahead` moves would also exceed the speed limit from this position

**Emission cost** (playing the notes assigned at this timestep from this thumb position):
- **0** — all notes land on fingers 1–5 in the natural hand position
- `--black-inner-penalty` per note where finger 2, 3, or 4 plays a black key
- `--black-outer-penalty` per note where the thumb (1) or pinky (5) plays a black key
- `--splay-penalty` per white key the thumb or pinky must stretch beyond natural reach
- `LOOKAHEAD_UNREACHABLE_PENALTY` if a note within the next `--lookahead` timesteps becomes completely unreachable from this position
- `--lookahead-penalty × difficulty` where difficulty counts how many future notes are hard (but not impossible) to reach

**Backtracking:** After processing all timesteps, the algorithm traces back through the predecessor pointers from the lowest-cost final state, yielding the globally optimal thumb path.

**Boundaries:** The left hand thumb can never exceed the split point; the right hand thumb can never go below `split_point + MIN_HAND_GAP`. This enforces the minimum hand gap at the thumb level.

---

### Stage 6 — Safety Validation

**Function:** `validate_output()`

Before generating commands the completed paths are checked for:

- **Velocity violations** — any move between adjacent timesteps that exceeds `--speed` keys/second
- **Hand gap violations** — right thumb less than `--gap` keys above left thumb at any timestep
- **Unreachable notes** — any note that falls outside the maximum reach (5 keys + `--max-splay`) from its hand's thumb position
- **Unresolved adjacent-key conflicts** — notes that still sit on the split boundary

Warnings are printed to the console but execution continues. Address them by adjusting parameters and re-running before deploying to hardware.

---

### Stage 7 — Servo Command Generation

**Function:** `generate_servo_commands()`

Each hand's thumb path is converted into a flat list of timestamped commands.

**Command structure:** Every note event at time `T` produces exactly two commands at timestamp `T`:

1. `T:step:OldNote-NewNote` — move the hand's thumb from the previous position to the new one. If the thumb doesn't move, `OldNote` and `NewNote` are the same (e.g. `C4-C4`). This still emits a step so the hardware always sees a step before every servo.
2. `T:servo:finger_cmd[,finger_cmd,...]` — actuate the assigned finger(s).

**Initial positioning:** Before the piece begins, an initial step command moves each hand from its **park position** (left: G1, right: F7) to the first playing position. This is always timestamped `0.000`.

**Preparation gap:** If the first note occurs before `t = 1.0s`, the entire timeline is shifted forward so the robot has at least 1 second to move into position before playing begins. The shift amount is printed to the console.

---

### Stage 8 — Save All Outputs

**Function:** `save_outputs()`

Writes all files to the `--output` directory (default: `outputs/`). The directory is created with `os.makedirs(..., exist_ok=True)` if it does not already exist.

---

## Output Files Reference

### `outputs/left_hand_commands.txt` / `outputs/right_hand_commands.txt`

One command per line: `<timestamp_seconds>:<type>:<payload>`

```
0.000:step:G1-C4
1.000:step:C4-C4
1.000:servo:1,3,5
2.500:step:C4-G3
2.500:servo:3b+
3.200:step:G3-A3
3.200:servo:1s+2
```

Commands are always in strict `step → servo` pairs at each timestamp. The lone `step` at `t=0.000` is the initial positioning move (no servo follows it).

---

### `outputs/fingering_plan.csv`

Full detail at every timestep:

| Column | Description |
|---|---|
| `Time` | Note onset in seconds |
| `L_Notes` | Left-hand note names (semicolon-separated if chord) |
| `L_Thumb` | Left thumb white key index |
| `L_Fingers` | Finger numbers assigned to each left-hand note |
| `L_Techniques` | Technique label per finger (see below) |
| `L_Commands` | Servo command string for the left hand |
| `R_Notes` | Right-hand note names |
| `R_Thumb` | Right thumb white key index |
| `R_Fingers` | Finger numbers for right-hand notes |
| `R_Techniques` | Technique labels for right hand |
| `R_Commands` | Servo command string for the right hand |

**Technique labels:**
- `normal` — standard white key press, finger directly over the key
- `black_key_inner` — inner finger (2, 3, or 4) on a black key
- `black_key_outer` — thumb (1) or pinky (5) on a black key
- `splay_thumb` — thumb stretching beyond the natural 5-key span
- `splay_pinky` — pinky stretching beyond the natural 5-key span

---

### `outputs/fingering_summary.csv`

Metrics and the configuration parameters used, with left-hand, right-hand, and combined columns. Includes: total hand position changes, black key usage counts, splay counts, split point used, speed limit, hand gap, and timeline shift applied.

---

### `outputs/timed_steps.csv`

Intermediate file. Columns: `start_time`, `midi`, `duration`, `white_key_index`, `is_black`. One row per note (not per chord). Used as ground truth by the verifier.

---

### `outputs/conflict_resolutions.txt` *(created only when needed)*

Lists every adjacent-key conflict the resolver encountered, which hand was assigned each note, and why.

---

### Servo Command Format

```
<time>:step:<from_note>-<to_note>
<time>:servo:<finger_cmd>[,<finger_cmd>,...]
```

| Part | Example | Meaning |
|---|---|---|
| `<time>` | `1.500` | Seconds since start (3 decimal places) |
| `step` | `step:C4-E4` | Move thumb from C4 to E4 |
| `step` (no move) | `step:C4-C4` | Stay at C4, still emits a step |
| `servo` | `servo:1,3` | Press finger 1 and finger 3 simultaneously |

---

### Finger Command Notation

Each comma-separated entry in a `servo` payload follows this grammar:

```
<finger>[type][direction][distance]
```

| Field | Values | Meaning |
|---|---|---|
| `finger` | `1`–`5` | Which finger (1 = thumb, 5 = pinky) |
| `type` | *(none)* | Normal white key — no special technique |
| `type` | `b` | Black key reach |
| `type` | `s` | Extended splay beyond 5-key span |
| `direction` | `-` | Toward lower keys (left on keyboard) |
| `direction` | `+` | Toward higher keys (right on keyboard) |
| `distance` | `1`, `2`, … | Keys splayed past natural reach (splay only) |
| *(any)* | `X` | Note unreachable — hardware should not fire |

**Complete examples:**

| Command | Interpretation |
|---|---|
| `1` | Thumb on a white key, natural position |
| `3` | Middle finger on a white key |
| `2b-` | Finger 2 reaching left to a black key |
| `3b+` | Finger 3 reaching right to a black key |
| `1s+2` | Thumb splayed 2 white keys to the right of its natural position |
| `5s-1` | Pinky splayed 1 white key to the left of its natural position |
| `1,3b+,5` | Chord: thumb (white), middle (black, right lean), pinky (white) |
| `X` | This note is unreachable — check safety warnings |

---

## findOptimalHandPos.py — Complete Argument Reference

```bash
python findOptimalHandPos.py [file] [options]
```

### Positional Argument

| Argument | Description |
|---|---|
| `file` | MusicXML filename or path. **Optional** — if omitted, an interactive list of files from `inputs/` is shown. Bare filenames (e.g. `hbd.musicxml`) are resolved to `inputs/hbd.musicxml` automatically. Full paths are also accepted. |

---

### Core Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--output <dir>` | string | `outputs` | Directory to write all output files. Created automatically. |
| `--speed <n>` | float | `10.0` | Maximum hand movement speed in **keys per second**. A hand moving 5 white keys must have at least 0.5 seconds between those timesteps or the move is penalised by `VELOCITY_PENALTY`. Increase for fast tempos; decrease for a slow or imprecise robot. |
| `--penalty <n>` | int | `4` | Cost multiplier per key of thumb movement. Higher values favour keeping the hand still and stretching fingers; lower values allow more hand repositioning. |
| `--gap <n>` | int | `6` | Minimum white keys that must always separate the left thumb from the right thumb. Prevents the two hands from physically colliding. Decrease only if your robot's hands are narrow. |
| `--transpose` | flag | off | Transpose the score to C major (major key) or A minor (minor key), eliminating all black key notes. **Legacy mode** — changes the musical pitch. Only use if your hardware cannot reach black keys. |

---

### Black Key & Splay Options

These penalties shape how the optimizer weights different finger techniques. Higher = avoid more.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--black-inner-penalty <n>` | int | `2` | Cost when finger 2, 3, or 4 plays a black key. Low default because inner fingers naturally reach black keys well. |
| `--black-outer-penalty <n>` | int | `10` | Cost when the thumb (1) or pinky (5) plays a black key. Higher because outer fingers must angle further to reach. |
| `--splay-penalty <n>` | int | `50` | Cost per key that the thumb or pinky splays beyond the natural 5-key span. Very high default because splay is mechanically demanding and should be a last resort. |
| `--max-splay <n>` | int | `2` | Maximum white keys the thumb or pinky can splay outward from the natural position. Set to `0` to disable splay entirely (all notes must land on fingers 1–5 with no stretch). |

---

### Look-Ahead Options

Look-ahead prevents "painted into a corner" situations where the locally cheapest position now makes an upcoming note impossible to reach.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--lookahead <n>` | int | `3` | Number of future timesteps to examine when scoring each candidate thumb position. `0` disables look-ahead entirely (faster but may create unreachable notes later). `5`–`6` provides strong forward planning at higher compute cost. |
| `--lookahead-penalty <n>` | int | `50` | Cost added per "difficulty point" when a current position makes a future note hard (but not impossible) to reach. Increase this to make the optimizer more conservative about future positions. |

---

### Dynamic Split Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--dynamic-split` | flag | off | Enable per-segment split point optimisation. The split between left and right hands can shift over time rather than being fixed for the whole piece. Recommended for pieces where the register shifts dramatically between sections. |
| `--segment-size <n>` | int | `8` | Number of timesteps per segment when using dynamic split. Smaller values (4–6) allow finer adaptation; larger values (12–16) produce a more stable split at the cost of less flexibility. |
| `--split-change-penalty <n>` | int | `100` | Cost incurred each time the split point shifts between two adjacent segments. Higher values keep the split stable. Lower values allow it to move more freely. |
| `--split-max-change <n>` | int | `3` | Hard cap on how many white keys the split can move between adjacent segments, regardless of cost. Prevents the split from jumping wildly. |

---

## verify_fingering.py — Complete Argument Reference

Run after the optimizer to confirm the output is physically valid and plays the correct notes before deploying to hardware.

```bash
python verify_fingering.py [options]
```

### Verification Checks Performed

| # | Check | What it catches |
|---|---|---|
| 1 | **Note Accuracy** | Every note in `timed_steps.csv` is actually played at the right time by one of the hands |
| 2 | **Finger Collision** | Crossed fingers within a hand (right hand: finger numbers must increase with note pitch; left hand: must decrease) |
| 3 | **Finger Locking** | A finger holding a sustained note must not be reassigned to a different note before the sustain ends |
| 4 | **Velocity** | No hand thumb moves faster than the configured keys-per-second limit |
| 5 | **Hand Gap** | Left and right thumbs never come closer than the minimum gap |
| 6 | **Physical Reachability** | All notes land within finger reach from the assigned thumb position |
| 7 | **Command Format** | Every `servo` is preceded by a `step` at the same timestamp; step position chains are continuous |

### Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--dir <path>` | string | `outputs` | Directory containing the files to verify (`fingering_plan.csv`, `timed_steps.csv`, `fingering_summary.csv`, `left_hand_commands.txt`, `right_hand_commands.txt`). |
| `--speed <n>` | float | `10.0` | Maximum keys-per-second to check against. **Use the same value you passed to the optimizer** so the verifier's constraints match. |
| `--gap <n>` | int | `6` | Minimum hand gap to enforce. Again, match the optimizer value. |
| `--tolerance <n>` | float | `0.005` | Time-matching window in seconds (5 ms). Notes whose timestamps differ by less than this are treated as simultaneous. Increase only if you have unusual floating-point precision issues in your MusicXML. |
| `-v` / `--verbose` | flag | off | Show every error instead of capping at the first 10 per category. Essential when debugging a piece with many violations. |
| `--no-format-check` | flag | off | Skip the step/servo command format and continuity checks. Useful if you have modified the command files manually. |
| `--json-output <file>` | string | — | Write the full verification result (summary + all errors + warnings) to a JSON file. Useful for CI pipelines or downstream tooling. |
| `--config <file>` | string | — | Load all constraint values (`max_keys_per_second`, `min_hand_gap`, `time_tolerance`, etc.) from a JSON file instead of CLI flags. The JSON keys match the `VerificationConfig` dataclass field names. |
| `--self-test` | flag | — | Run the verifier's built-in unit tests (MIDI conversion, finger assignment, command parsing) and exit. Does not require any output files to be present. |

---

## Example Workflows

### The standard workflow

```bash
source .venv/bin/activate
python findOptimalHandPos.py           # interactive picker
python verify_fingering.py             # reads outputs/ by default
```

### Pass a song directly

```bash
python findOptimalHandPos.py hbd.musicxml
python verify_fingering.py
```

### Conservative robot (slow, precise movements)

```bash
python findOptimalHandPos.py fuyunohanashi1.musicxml \
    --speed 6 \
    --penalty 8 \
    --gap 8 \
    --splay-penalty 100 \
    --max-splay 1
python verify_fingering.py --speed 6 --gap 8
```

### Fast robot (high-tempo pieces)

```bash
python findOptimalHandPos.py starspanbanner.musicxml \
    --speed 20 \
    --penalty 2 \
    --lookahead 5
python verify_fingering.py --speed 20
```

### White-keys-only hardware (transpose mode)

```bash
python findOptimalHandPos.py fuyunohanashi1.musicxml --transpose
python verify_fingering.py
```

### Complex piece with shifting hand regions

```bash
python findOptimalHandPos.py fuyunohanashi1.musicxml \
    --dynamic-split \
    --segment-size 6 \
    --split-change-penalty 50 \
    --split-max-change 5 \
    --lookahead 4
python verify_fingering.py --verbose
```

### Export verification report as JSON

```bash
python findOptimalHandPos.py starspanbanner.musicxml --speed 12 --gap 7
python verify_fingering.py --speed 12 --gap 7 --json-output outputs/report.json
```

### Batch process all songs in inputs/

```bash
for f in inputs/*.musicxml; do
    name=$(basename "$f" .musicxml)
    echo "=== $name ==="
    python findOptimalHandPos.py "$f" --output "outputs/$name"
    python verify_fingering.py --dir "outputs/$name"
done
```

### Run verifier self-tests (no song needed)

```bash
python verify_fingering.py --self-test
```

---

## Tuning Guide

| Situation | Recommended adjustments |
|---|---|
| Hands move too much, jumpy output | Increase `--penalty` (try 6–10) |
| Velocity violations in verifier | Increase `--speed` in both scripts, or use a slower-tempo MusicXML |
| Black key notes are being avoided too aggressively | Decrease `--black-outer-penalty` (try 5) and `--black-inner-penalty` (try 1) |
| Splay never used even when it would help | Decrease `--splay-penalty` (try 20–30) |
| Optimizer picks positions that strand later notes | Increase `--lookahead` (try 5) and `--lookahead-penalty` (try 80) |
| Split stays fixed when hands need to cross | Enable `--dynamic-split` with a small `--segment-size` (4–6) |
| Hands colliding in verification | Increase `--gap` (try 8–10) |
| "No valid split point found" error | Decrease `--gap` and/or increase `--max-splay` |
| Computation too slow | Set `--lookahead 0` or `--lookahead 1`; disable `--dynamic-split` |
| Output too conservative, fingers never splay | Decrease `--splay-penalty` and increase `--max-splay` |

---

## Troubleshooting

**`❌ No playable notes found`**
The MusicXML parser could not extract any notes. Check that the file is a valid MusicXML (not a MIDI or PDF), that it is placed in `inputs/`, and that it contains at least one pitched note (not just rests or percussion).

**`❌ FATAL: Could not find any valid split point`**
No single key divides the keyboard such that both hands can play within their spans. Usually caused by a very wide-range piece or an overly large `--gap`. Try:
- Reducing `--gap` (e.g. `--gap 4`)
- Increasing `--max-splay`
- Enabling `--dynamic-split`

**Verifier reports velocity violations**
The optimizer allowed moves that exceed the speed limit because the penalty wasn't high enough to prevent them. Increase `--speed` when running the optimizer (or reduce the tempo of the score), then re-run.

**Verifier reports hand gap violations**
The optimizer's boundary enforcement only applies to thumb positions, but the verifier also checks mid-chord finger positions. Increase `--gap` when running the optimizer.

**Black key notes missing from the output**
Make sure `--transpose` is **not** active. Also check that `--black-outer-penalty` isn't so high that the optimizer prefers to skip notes (which would show as note accuracy errors in the verifier).

**Too many conflict resolution warnings**
The piece has many notes clustered at the split boundary. Try shifting the split by adjusting `--gap`, or enable `--dynamic-split` so the boundary can move away from congested regions.
