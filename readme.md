# Robotic Piano Fingering Optimizer

An advanced system for optimizing and verifying robotic piano fingering using dynamic programming and constraint satisfaction. This project converts MusicXML files into optimized servo and stepper motor commands for a robotic piano player.

## Overview

This project solves the complex optimization problem of determining optimal hand positions and finger assignments for a robotic piano system. It considers physical constraints, movement efficiency, and musical playability to generate hardware control commands.

### Key Features

- **Full Black Key Support**: Handles both white and black keys with sophisticated splay techniques
- **Dynamic Programming Optimization**: Finds globally optimal finger assignments
- **Look-Ahead Planning**: Prevents "painted into a corner" situations by considering future notes
- **Velocity Constraints**: Enforces maximum hand movement speeds
- **Collision Detection**: Prevents finger crossings and hand collisions
- **Sustained Note Handling**: Correctly manages finger locking for held notes
- **Comprehensive Verification**: Multi-stage validation of generated commands

## Project Structure

```
.
├── findOptimalHandPos.py          # Main optimization engine
├── verify_fingering.py             # Comprehensive verification suite
├── song.musicxml                   # Input MusicXML file
└── output/                         # Generated output files
    ├── fingering_plan.csv          # Human-readable fingering plan
    ├── fingering_summary.csv       # Optimization metrics and statistics
    ├── timed_steps.csv             # Parsed note timing data
    ├── left_hand_commands.txt      # Left hand motor commands
    └── right_hand_commands.txt     # Right hand motor commands
```

## Quick Start

### Prerequisites

```bash
pip install music21
```

### Basic Usage

1. **Generate Fingering Plan**:
```bash
python findOptimalHandPos.py song.musicxml
```

2. **Verify Output**:
```bash
python verify_fingering.py --dir output
```

### Advanced Options

```bash
# Adjust movement penalty
python findOptimalHandPos.py song.musicxml --move-penalty 5

# Change velocity limit
python findOptimalHandPos.py song.musicxml --max-velocity 15

# Enable dynamic split point optimization
python findOptimalHandPos.py song.musicxml --dynamic-split

# Verbose verification with custom constraints
python verify_fingering.py --dir output --verbose --speed 12 --gap 8
```

## Output Files

### `fingering_plan.csv`
Human-readable fingering plan with columns:
- **Time**: Timestamp in seconds
- **L_Notes/R_Notes**: Notes to play (left/right hand)
- **L_Thumb/R_Thumb**: Thumb position (white key index)
- **L_Fingers/R_Fingers**: Finger number (1=thumb, 2-5=fingers)
- **L_Techniques/R_Techniques**: Fingering technique used
- **L_Commands/R_Commands**: Servo command for finger

### `fingering_summary.csv`
Optimization statistics including:
- Split point (dividing line between hands)
- Position change counts
- Black key usage statistics
- Hardware constraints
- Look-ahead configuration
- Timeline adjustments

### `timed_steps.csv`
Parsed note data:
- `start_time`: Note start time (seconds)
- `midi`: MIDI note number
- `duration`: Note duration (seconds)
- `white_key_index`: Physical key position
- `is_black`: Black key flag (0=white, 1=black)

### `left_hand_commands.txt` / `right_hand_commands.txt`
Hardware control commands in format:
```
<time>:step:<from_note>-<to_note>    # Move hand position
<time>:servo:<finger_number>         # Actuate finger
```

Example:
```
0.0:step:G1-C3      # Move hand to C3 position
1.000:step:C3-C3    # Hold position
1.000:servo:1       # Press finger 1 (thumb)
```

## Configuration

### Global Parameters (in `findOptimalHandPos.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MOVE_PENALTY` | 4 | Cost for moving hand position |
| `MAX_KEYS_PER_SECOND` | 10.0 | Maximum hand velocity |
| `VELOCITY_PENALTY` | 100 | Penalty for exceeding velocity |
| `MIN_HAND_GAP` | 6 | Minimum keys between hands |
| `LOOKAHEAD_STEPS` | 3 | Future steps to consider |
| `DYNAMIC_SPLIT_ENABLED` | False | Enable dynamic split optimization |

### Black Key Handling

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INNER_FINGER_BLACK_KEY_PENALTY` | 2 | Penalty for fingers 2-4 on black keys |
| `OUTER_FINGER_BLACK_KEY_PENALTY` | 10 | Penalty for thumb/pinky on black keys |
| `OUTER_FINGER_SPLAY_PENALTY` | 50 | Penalty for extreme splay |
| `MAX_OUTER_SPLAY` | 2 | Maximum white keys for splay |

### Command-Line Arguments

**findOptimalHandPos.py**:
```bash
--move-penalty FLOAT          # Hand movement penalty (default: 4)
--max-velocity FLOAT          # Max keys/second (default: 10.0)
--min-gap INT                 # Minimum hand gap (default: 6)
--lookahead INT               # Look-ahead steps (default: 3)
--dynamic-split               # Enable dynamic split point
--auto-transpose              # Transpose to C Major/A Minor
```

**verify_fingering.py**:
```bash
--dir PATH                    # Output directory (default: output)
--speed FLOAT                 # Max velocity to verify (default: 10.0)
--gap INT                     # Min hand gap to verify (default: 6)
--verbose                     # Detailed output
--self-test                   # Run unit tests
```

## Verification System

The verification script performs 7 comprehensive checks:

1. **Note Accuracy**: Verifies all notes are played at correct times
2. **Finger Collision**: Detects crossed fingers within each hand
3. **Finger Locking**: Ensures held notes don't reuse fingers
4. **Velocity Constraints**: Validates hand movement speeds
5. **Hand Gap**: Checks minimum separation between hands
6. **Physical Reachability**: Confirms notes are within hand span
7. **Command Format**: Validates step-servo command patterns

### Example Verification Output

```
VERIFICATION PASSED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Summary:
  Total timesteps: 27
  Notes verified: 42
  Left hand notes: 14
  Right hand notes: 28
  
  [PASS] Note accuracy: All notes correct
  [PASS] Finger collisions: None detected
  [PASS] Finger locking: No violations
  [PASS] Velocity constraints: Within limits (max: 0.00 keys/sec)
  [PASS] Hand gap: Maintained (min: 7 keys)
  [PASS] Physical reach: All notes reachable
  [PASS] Command format: Valid
```

## How It Works

### 1. MusicXML Parsing
- Extracts notes, timing, and durations
- Converts to unified key position system
- Handles both white and black keys
- Optional auto-transpose to C Major/A Minor

### 2. Dynamic Programming Optimization
The optimizer uses a state-space search where each state represents:
- Left hand thumb position
- Right hand thumb position
- Finger assignments for active notes
- Cost accumulated from start

**Cost Function**:
```
Total Cost = Movement Penalty + Black Key Penalties + 
             Velocity Violations + Look-Ahead Penalties
```

### 3. Look-Ahead Planning
Evaluates future notes to ensure:
- Future notes remain reachable
- Movement requirements stay within velocity limits
- Hand positions don't create future conflicts

### 4. Hardware Command Generation
Converts optimal plan to timestamped commands:
- **Step commands**: Move hand to new position
- **Servo commands**: Actuate specific finger

Includes 1-second preparation time before first note.

## Technical Details

### Key Position System
- White keys indexed 0-51 (A0 to C8)
- Black keys positioned between white keys
- Splay techniques for reaching adjacent black keys

### Hand Span Model
- 5 fingers per hand (1=thumb to 5=pinky)
- Base span: 4 white keys
- Splay: ±2 keys for outer fingers
- Inner fingers preferred for black keys

### Optimization Complexity
- State space: O(K² × T) where K=keys, T=timesteps
- Pruning: Top 1000 states per timestep
- Look-ahead: Adds O(L) factor where L=look-ahead steps

## Testing

Run self-tests:
```bash
python verify_fingering.py --self-test
```

This validates:
- File parsing
- Collision detection
- Finger locking logic
- Velocity calculations
- Hand gap checks
- Command format validation

## Example Workflow

```bash
# 1. Generate fingering plan
python findOptimalHandPos.py beethoven_ode.musicxml --move-penalty 5

# 2. Verify output
python verify_fingering.py --dir output --verbose

# 3. (If needed) Adjust parameters and regenerate
python findOptimalHandPos.py beethoven_ode.musicxml --move-penalty 3 --max-velocity 12

# 4. Export configuration
python verify_fingering.py --dir output --export-config config.json
```

## Hardware Integration

The generated command files are ready for robotic piano systems with:
- **Stepper motors**: For hand positioning (step commands)
- **Servo motors**: For finger actuation (servo commands)
- **Timing precision**: Sub-millisecond accuracy required

### Command Timing
- All times in seconds with millisecond precision
- Step commands move hand to specified key range
- Servo commands actuate specific finger (1-5)
- Commands are pre-sorted by timestamp

## Performance Optimization Tips

1. **Reduce state space**: Lower `MAX_KEYS_PER_SECOND` if possible
2. **Adjust penalties**: Balance `MOVE_PENALTY` vs `VELOCITY_PENALTY`
3. **Limit look-ahead**: Reduce `LOOKAHEAD_STEPS` for faster computation
4. **Enable dynamic split**: Can improve results for complex pieces
5. **Use auto-transpose**: Simplifies optimization for pieces with many sharps/flats

## Troubleshooting

**Issue**: "No valid fingering found"
- Reduce `MIN_HAND_GAP` or increase `MAX_KEYS_PER_SECOND`
- Enable `--dynamic-split` for more flexibility
- Check if piece requires more than 5-finger span

**Issue**: Verification fails with velocity violations
- Increase `MAX_KEYS_PER_SECOND` in findOptimalHandPos.py
- Adjust `VELOCITY_PENALTY` to prioritize speed compliance

**Issue**: Black key notes skipped
- Ensure `auto_transpose` is disabled
- Check `OUTER_FINGER_BLACK_KEY_PENALTY` isn't too high

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions welcome! Areas for improvement:
- Support for pedaling
- Chord optimization
- Multi-staff support
- Real-time visualization
- Additional hardware backends

## References

- Music21 library: https://web.mit.edu/music21/
- MusicXML format: https://www.w3.org/2021/06/musicxml40/
- Dynamic programming for sequence optimization
- Constraint satisfaction for robotic control

---

**Created for robotic piano research and automation**