# Piano Hand Algorithms

Software that reads sheet music and figures out exactly how a two-handed robotic piano player should move — which hand plays which note, where to position each hand, and which finger to use — then outputs the precise commands the robot needs to perform the piece.

---

## What Problem Does This Solve?

A robotic piano has real physical limits: each hand can only reach 5 keys at a time, both hands need to stay out of each other's way, and the motors can only move so fast. Deciding how to play a full piece of music while respecting all of those constraints — for every single note — is an incredibly complex puzzle.

This software solves that puzzle automatically. You give it a piece of sheet music, and it finds the best possible plan for the robot, then outputs ready-to-use motor commands.

---

## How It Works

1. **Read the sheet music** — The program reads a standard sheet music file (MusicXML format).
2. **Map out every note** — It records each note's pitch, when it starts, and how long it lasts.
3. **Divide the keyboard** — It decides where the left hand's territory ends and the right hand's begins.
4. **Find the best path** — Using an optimization algorithm, it calculates the smoothest, most efficient way for each hand to move through the entire piece without getting stuck or colliding.
5. **Assign fingers** — It picks which specific finger (thumb through pinky) plays each note.
6. **Write the commands** — It saves timestamped motor commands that tell the robot exactly what to do and when.

---

## Getting Started

### 1. Install Python

You need Python 3.9 or newer. Download it from [python.org](https://www.python.org/downloads/) if you don't have it.

### 2. Set Up the Project

Open a terminal, navigate to the project folder, and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install music21
```

> On Windows, replace `source .venv/bin/activate` with `.venv\Scripts\activate`

### 3. Run It

```bash
python findOptimalHandPos.py
```

A menu will appear listing the available songs. Type the number of the song you want and press Enter.

---

## Adding Your Own Songs

1. Export your sheet music as a **MusicXML file** from any notation software (MuseScore, Finale, Sibelius, etc.).
2. Drop the `.musicxml` file into the `inputs/` folder.
3. Run the program — your song will appear in the menu automatically.

### Songs Already Included

| File | Song |
|------|------|
| `hbd.musicxml` | Happy Birthday |
| `twinkletwinkle.musicxml` | Twinkle Twinkle Little Star |
| `maryhadlamb.musicxml` | Mary Had a Little Lamb |
| `hotcrossbuns.musicxml` | Hot Cross Buns |
| `starspanbanner.musicxml` | The Star-Spangled Banner |
| `fuyunohanashi1.musicxml` | Fuyu no Hanashi |

---

## What Gets Generated

After running the program, everything is saved to the `outputs/` folder:

| File | What It Contains |
|------|-----------------|
| `left_hand_commands.txt` | Timestamped motor commands for the left hand |
| `right_hand_commands.txt` | Timestamped motor commands for the right hand |
| `fingering_plan.csv` | The full note-by-note plan: which hand, which finger, where to position |
| `fingering_summary.csv` | Stats for the run — how many position changes, black keys used, etc. |
| `timed_steps.csv` | A list of every parsed note with timing and pitch info |

---

## Checking the Output

After generating commands, you can run a safety check to make sure the plan is physically valid:

```bash
python verify_fingering.py
```

This confirms that:
- Every note in the music is accounted for
- No fingers collide with each other
- The hands never move faster than the motors allow
- The two hands always stay far enough apart to avoid colliding
- Every note is actually reachable from the assigned hand position

---

## Physical Limits the Software Respects

| Constraint | Default | What It Means |
|------------|---------|---------------|
| Hand span | 5 white keys | Each hand can only reach 5 keys at once |
| Hand gap | 6 keys minimum | The two hands must stay at least 6 keys apart |
| Motor speed | 10 keys/second | How fast a hand can slide across the keyboard |

---

## Advanced: Tweaking the Settings

You can adjust how the algorithm behaves by adding options when you run it. This is useful for tuning the output to match your specific robot's capabilities.

```bash
python findOptimalHandPos.py hbd.musicxml --speed 8 --gap 8
```

### Common Options

| Option | What It Does | Default |
|--------|-------------|---------|
| `--speed <n>` | Max keys per second the hand can move | 10 |
| `--gap <n>` | Minimum distance (in keys) the two hands must maintain | 6 |
| `--penalty <n>` | How much to discourage unnecessary hand repositioning — higher means the hands move less | 4 |
| `--splay-penalty <n>` | How much to discourage stretching a finger beyond normal reach | 50 |
| `--lookahead <n>` | How many notes ahead the algorithm plans — higher means fewer "stuck" situations | 3 |
| `--dynamic-split` | Let the left/right boundary shift throughout the piece (helpful for complex songs) | off |
| `--transpose` | Shift the music to use only white keys (removes all black key notes) | off |

### Tuning Quick Reference

| Problem | Fix |
|---------|-----|
| Hands are moving around too much | Increase `--penalty` (try 6–10) |
| Safety check reports speed violations | Increase `--speed`, or use a slower MusicXML file |
| Hands keep getting too close | Increase `--gap` (try 8–10) |
| The optimizer seems to get "stuck" on hard passages | Increase `--lookahead` (try 5) |
| Song has wide register shifts between sections | Add `--dynamic-split` |

---

## Project Structure

```
Piano-Hand-Algorithms-1/
├── inputs/                   ← Put sheet music files here
├── outputs/                  ← Generated plans and motor commands appear here
├── findOptimalHandPos.py     ← Main program (run this)
├── verify_fingering.py       ← Safety checker for generated plans
└── run_all.py                ← Runs all songs in inputs/ at once
```

---

## Running All Songs at Once

To process every song in the `inputs/` folder in one go:

```bash
python run_all.py
```

Each song's output will be saved to its own folder inside `outputs/`.
