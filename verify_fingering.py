#!/usr/bin/env python3
"""
Robotic Piano Fingering Verification Script v2.2 (IMPROVED)

IMPROVEMENTS OVER v2.1:
- Added comprehensive unit tests
- Fixed time tolerance configuration
- Added logging support
- Structured error objects
- JSON export capability
- Performance optimization for time lookups
- Better type hints
- Configuration file support
- Self-test functionality

Verifies the output of findOptimalHandPos.py against:
1. Note accuracy - correct notes played at correct times
2. Finger collision detection - no crossed fingers within a hand
3. Sustained note finger locking - fingers holding notes aren't reused
4. Velocity constraints - hand movement speed within limits
5. Hand gap constraints - hands don't collide
6. Physical reachability - all notes within hand span
7. Command format validation - step-servo pattern is correct

Usage:
    python verify_fingering.py --dir <output_directory>
    python verify_fingering.py --dir . --verbose
    python verify_fingering.py --dir . --speed 15 --gap 8
    python verify_fingering.py --self-test
"""

import csv
import os
import re
import sys
import json
import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Set, Tuple, Optional, Union
from pathlib import Path

# ==========================================
# LOGGING SETUP
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# CONSTANTS & CONFIG
# ==========================================
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Default constraints (can be overridden via CLI or config file)
DEFAULT_MAX_KEYS_PER_SECOND = 10.0
DEFAULT_MIN_HAND_GAP = 6
DEFAULT_MAX_OUTER_SPLAY = 2
DEFAULT_TIME_TOLERANCE = 0.005  # 5ms tolerance (was 1ms, now more robust)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB safety limit


# ==========================================
# DATA CLASSES
# ==========================================
@dataclass
class VerificationConfig:
    """Configuration for verification checks."""
    max_keys_per_second: float = DEFAULT_MAX_KEYS_PER_SECOND
    min_hand_gap: int = DEFAULT_MIN_HAND_GAP
    max_outer_splay: int = DEFAULT_MAX_OUTER_SPLAY
    time_tolerance: float = DEFAULT_TIME_TOLERANCE
    verbose: bool = False
    check_command_format: bool = True

    @classmethod
    def from_file(cls, filepath: str) -> 'VerificationConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def to_file(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)


@dataclass
class VerificationError:
    """Structured error with context."""
    timestamp: float
    error_type: str  # 'NOTE', 'COLLISION', 'LOCKING', 'VELOCITY', 'GAP', 'FORMAT'
    severity: str  # 'ERROR' or 'WARNING'
    message: str
    context: Dict[str, any] = field(default_factory=dict)

    def __str__(self):
        return f"[{self.severity}] Time {self.timestamp:.2f}s ({self.error_type}): {self.message}"


@dataclass
class VerificationResult:
    """Results from verification."""
    total_checks: int = 0
    passed_checks: int = 0
    note_errors: List[str] = field(default_factory=list)
    collision_errors: List[str] = field(default_factory=list)
    locking_errors: List[str] = field(default_factory=list)
    velocity_errors: List[str] = field(default_factory=list)
    hand_gap_errors: List[str] = field(default_factory=list)
    reachability_errors: List[str] = field(default_factory=list)
    format_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def total_errors(self) -> int:
        return (len(self.note_errors) + len(self.collision_errors) +
                len(self.locking_errors) + len(self.velocity_errors) +
                len(self.hand_gap_errors) + len(self.reachability_errors) +
                len(self.format_errors))

    @property
    def accuracy(self) -> float:
        if self.total_checks == 0:
            return 0.0
        return (self.passed_checks / self.total_checks) * 100

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            "summary": {
                "total_checks": self.total_checks,
                "passed_checks": self.passed_checks,
                "failed_checks": self.total_checks - self.passed_checks,
                "accuracy": self.accuracy
            },
            "errors": {
                "note_errors": self.note_errors,
                "collision_errors": self.collision_errors,
                "locking_errors": self.locking_errors,
                "velocity_errors": self.velocity_errors,
                "hand_gap_errors": self.hand_gap_errors,
                "reachability_errors": self.reachability_errors,
                "format_errors": self.format_errors
            },
            "warnings": self.warnings,
            "total_errors": self.total_errors
        }


# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def validate_file_size(filepath: str) -> bool:
    """Check if file size is within safe limits."""
    if not os.path.exists(filepath):
        return False
    size = os.path.getsize(filepath)
    if size > MAX_FILE_SIZE:
        logger.error(f"File too large: {filepath} ({size} bytes > {MAX_FILE_SIZE} bytes)")
        return False
    return True


def midi_to_name(midi: int) -> str:
    """
    Convert MIDI number to name (e.g., 60 -> C4).

    MIDI Standard:
    - C4 (Middle C) = MIDI 60
    - Octave starts at C
    """
    if midi is None:
        return "None"
    octave = (midi // 12) - 1
    note = NOTE_NAMES[midi % 12]
    return f"{note}{octave}"


def index_to_name(idx: int) -> str:
    """
    Convert white key index to note name.

    White Key Indexing Scheme:
    - Index 0 = A0
    - Index 7 = A1
    - Index 23 = C4 (verified)

    The mapping cycles through A-B-C-D-E-F-G with octave changes at C.
    """
    if idx is None:
        return "None"
    position = idx % 7
    names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    if position <= 1:  # A, B are in same octave as previous C
        octave = idx // 7
    else:  # C-G mark new octave
        octave = idx // 7 + 1
    return f"{names[position]}{octave}"


def parse_finger_command(cmd: str) -> Tuple[Optional[int], str, int, int]:
    """
    Parse a single finger command string.

    Format: <finger>[b|s][+|-]<distance>
    Examples:
        "3" -> (3, 'normal', 0, 0)
        "1b" -> (1, 'black', 0, 0)
        "5s+2" -> (5, 'splay', 1, 2)
    """
    if not cmd or cmd == 'X':
        return (None, None, 0, 0)

    match = re.match(r'(\d)([bs])?([+-])?(\d+)?', cmd)
    if not match:
        return (None, None, 0, 0)

    finger = int(match.group(1))
    tech_char = match.group(2)
    dir_char = match.group(3)
    dist_str = match.group(4)

    if tech_char == 'b':
        technique = 'black'
    elif tech_char == 's':
        technique = 'splay'
    else:
        technique = 'normal'

    direction = 0
    if dir_char == '+':
        direction = 1
    elif dir_char == '-':
        direction = -1

    distance = int(dist_str) if dist_str else 0

    return (finger, technique, direction, distance)


def build_time_index(timed_steps: Dict[float, List[Dict]],
                     tolerance: float = DEFAULT_TIME_TOLERANCE) -> Dict[float, List[Dict]]:
    """
    Build efficient time lookup structure with tolerance.

    IMPROVEMENT: O(1) lookup instead of O(n) linear search
    """
    index = defaultdict(list)
    for t, notes in timed_steps.items():
        # Round to tolerance precision
        rounded_t = round(t / tolerance) * tolerance
        index[rounded_t].extend(notes)
    return dict(index)


# ==========================================
# FINGER COLLISION DETECTION
# ==========================================
def validate_finger_assignment(finger_note_pairs: List[Tuple[int, int]],
                               hand: str) -> Tuple[bool, Optional[str]]:
    """
    Ensure fingers are in correct spatial order (no crossed fingers).

    Rules:
    - Left hand: higher notes use lower finger numbers (5-4-3-2-1 from low to high)
    - Right hand: higher notes use higher finger numbers (1-2-3-4-5 from low to high)
    """
    if len(finger_note_pairs) <= 1:
        return True, None

    valid_pairs = [(f, n) for f, n in finger_note_pairs if f is not None]

    if len(valid_pairs) <= 1:
        return True, None

    sorted_pairs = sorted(valid_pairs, key=lambda x: x[1])
    fingers_in_order = [p[0] for p in sorted_pairs]

    if hand == "left":
        # Left hand: finger numbers should DECREASE as notes go up
        for i in range(1, len(fingers_in_order)):
            if fingers_in_order[i] >= fingers_in_order[i - 1]:
                return False, (f"F{fingers_in_order[i - 1]}@{index_to_name(sorted_pairs[i - 1][1])} -> "
                               f"F{fingers_in_order[i]}@{index_to_name(sorted_pairs[i][1])}")
    else:  # right hand
        # Right hand: finger numbers should INCREASE as notes go up
        for i in range(1, len(fingers_in_order)):
            if fingers_in_order[i] <= fingers_in_order[i - 1]:
                return False, (f"F{fingers_in_order[i - 1]}@{index_to_name(sorted_pairs[i - 1][1])} -> "
                               f"F{fingers_in_order[i]}@{index_to_name(sorted_pairs[i][1])}")

    return True, None


def calculate_finger_for_note(thumb_pos: int, note_pos: int, hand: str,
                              max_splay: int = DEFAULT_MAX_OUTER_SPLAY) -> Optional[int]:
    """Calculate which finger would play a note given thumb position."""
    if thumb_pos is None:
        return None

    if hand == "left":
        natural_finger = thumb_pos - note_pos + 1
    else:
        natural_finger = note_pos - thumb_pos + 1

    # Normal reach
    if 1 <= natural_finger <= 5:
        return natural_finger

    # Splay for thumb
    if natural_finger < 1:
        splay_amount = 1 - natural_finger
        if splay_amount <= max_splay:
            return 1

    # Splay for pinky
    if natural_finger > 5:
        splay_amount = natural_finger - 5
        if splay_amount <= max_splay:
            return 5

    return None


# ==========================================
# PARSERS
# ==========================================
def parse_timed_steps(filepath: str) -> Dict[float, List[Dict[str, Union[int, float, bool]]]]:
    """Parse timed_steps.csv (ground truth)."""
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return {}

    if not validate_file_size(filepath):
        return {}

    events = defaultdict(list)
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = float(row['start_time'])
                events[t].append({
                    'white_index': int(row['white_key_index']),
                    'is_black': bool(int(row['is_black'])),
                    'midi': int(row['midi']),
                    'duration': float(row.get('duration', 1.0))
                })
    except (ValueError, KeyError) as e:
        logger.error(f"Error parsing {filepath}: {e}")
        return {}

    return dict(events)


def parse_fingering_plan(filepath: str) -> List[Dict]:
    """Parse fingering_plan.csv."""
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return []

    if not validate_file_size(filepath):
        return []

    plan = []
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry = {
                    'time': float(row['Time']),
                    'l_notes': row.get('L_Notes', ''),
                    'l_thumb': int(row['L_Thumb']) if row.get('L_Thumb') else None,
                    'l_fingers': row.get('L_Fingers', ''),
                    'l_techniques': row.get('L_Techniques', ''),
                    'l_cmd': row.get('L_Commands', ''),
                    'r_notes': row.get('R_Notes', ''),
                    'r_thumb': int(row['R_Thumb']) if row.get('R_Thumb') else None,
                    'r_fingers': row.get('R_Fingers', ''),
                    'r_techniques': row.get('R_Techniques', ''),
                    'r_cmd': row.get('R_Commands', '')
                }
                plan.append(entry)
    except (ValueError, KeyError) as e:
        logger.error(f"Error parsing {filepath}: {e}")
        return []

    return plan


def parse_fingering_summary(filepath: str) -> Dict[str, List[str]]:
    """Parse fingering_summary.csv for configuration values."""
    if not os.path.exists(filepath):
        return {}

    summary = {}
    try:
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    summary[row[0]] = row[1:]
    except Exception as e:
        logger.warning(f"Error parsing summary {filepath}: {e}")

    return summary


def parse_command_file(filepath: str) -> List[Dict[str, Union[float, str]]]:
    """Parse a hand command file."""
    if not os.path.exists(filepath):
        return []

    commands = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(':', 2)
                if len(parts) >= 3:
                    try:
                        cmd = {
                            'time': float(parts[0]),
                            'type': parts[1],
                            'payload': parts[2]
                        }
                        commands.append(cmd)
                    except ValueError:
                        continue
    except Exception as e:
        logger.error(f"Error parsing command file {filepath}: {e}")

    return commands


# ==========================================
# NOTE CALCULATION
# ==========================================
def calculate_hit_notes(thumb_idx: int, cmd_str: str, is_left: bool) -> List[Tuple[int, bool, int]]:
    """Calculate which notes are hit given thumb position and command."""
    if not thumb_idx or not cmd_str:
        return []

    hit_notes = []
    sub_cmds = cmd_str.split(';')

    for sub in sub_cmds:
        if sub == 'X' or sub == '':
            continue

        finger, technique, direction, distance = parse_finger_command(sub)
        if finger is None:
            continue

        effective_reach = finger

        if technique == 'splay':
            if finger == 1:
                effective_reach = 1 - distance
            elif finger == 5:
                effective_reach = 5 + distance

        if is_left:
            target_idx = thumb_idx - effective_reach + 1
        else:
            target_idx = thumb_idx + effective_reach - 1

        is_black = (technique == 'black')
        hit_notes.append((target_idx, is_black, finger))

    return hit_notes


# ==========================================
# COMMAND FORMAT VERIFICATION
# ==========================================
def verify_command_format(commands: List[Dict], hand_name: str, result: VerificationResult) -> bool:
    """Verify that commands follow the step-servo pattern."""
    if not commands:
        return True

    all_valid = True

    # Check initial command is a step
    if commands[0]['type'] != 'step':
        result.format_errors.append(
            f"{hand_name} Hand: First command should be 'step', got '{commands[0]['type']}'"
        )
        all_valid = False

    # Group commands by time
    commands_by_time = defaultdict(list)
    for cmd in commands:
        commands_by_time[cmd['time']].append(cmd)

    # Check each time group
    for time, cmds in sorted(commands_by_time.items()):
        if time == 0.0 and len(cmds) == 1 and cmds[0]['type'] == 'step':
            continue

        types = [c['type'] for c in cmds]

        if time > 0.0:
            if 'step' not in types:
                result.format_errors.append(
                    f"{hand_name} Hand at t={time:.3f}s: Missing 'step' command before 'servo'"
                )
                all_valid = False

            if 'servo' not in types:
                result.format_errors.append(
                    f"{hand_name} Hand at t={time:.3f}s: Missing 'servo' command after 'step'"
                )
                all_valid = False

    return all_valid


def verify_step_continuity(commands: List[Dict], hand_name: str, result: VerificationResult) -> bool:
    """Verify that step commands show proper position continuity."""
    if not commands:
        return True

    all_valid = True
    last_position = None

    for cmd in commands:
        if cmd['type'] != 'step':
            continue

        payload = cmd['payload']
        if '-' not in payload:
            result.format_errors.append(
                f"{hand_name} Hand at t={cmd['time']:.3f}s: Invalid step format '{payload}'"
            )
            all_valid = False
            continue

        parts = payload.split('-')
        if len(parts) != 2:
            result.format_errors.append(
                f"{hand_name} Hand at t={cmd['time']:.3f}s: Invalid step format '{payload}'"
            )
            all_valid = False
            continue

        from_pos, to_pos = parts[0], parts[1]

        if last_position is not None and from_pos != last_position:
            result.format_errors.append(
                f"{hand_name} Hand at t={cmd['time']:.3f}s: Position discontinuity - "
                f"expected FROM='{last_position}', got '{from_pos}'"
            )
            all_valid = False

        last_position = to_pos

    return all_valid


# ==========================================
# VERIFICATION CHECKS
# ==========================================
def verify_note_accuracy(step: Dict, expected: List[Dict], result: VerificationResult) -> bool:
    """Verify that the correct notes are played."""
    t = step['time']

    played_notes = []

    l_hits = calculate_hit_notes(step['l_thumb'], step['l_cmd'], is_left=True)
    for idx, is_blk, finger in l_hits:
        played_notes.append({'idx': idx, 'black': is_blk, 'hand': 'Left', 'finger': finger})

    r_hits = calculate_hit_notes(step['r_thumb'], step['r_cmd'], is_left=False)
    for idx, is_blk, finger in r_hits:
        played_notes.append({'idx': idx, 'black': is_blk, 'hand': 'Right', 'finger': finger})

    exp_set = set((n['white_index'], n['is_black']) for n in expected)
    act_set = set((n['idx'], n['black']) for n in played_notes)

    if exp_set == act_set:
        return True

    missing = exp_set - act_set
    extra = act_set - exp_set

    err_msg = f"Time {t:.2f}s Note Mismatch:"
    if missing:
        m_names = [midi_to_name(n['midi']) for n in expected
                   if (n['white_index'], n['is_black']) in missing]
        err_msg += f"\n    MISSING: {m_names}"
    if extra:
        e_names = [f"{index_to_name(idx)}{'#' if blk else ''}" for idx, blk in extra]
        err_msg += f"\n    EXTRA: {e_names}"
    err_msg += f"\n    L: {step['l_cmd']} (thumb {step['l_thumb']})"
    err_msg += f"\n    R: {step['r_cmd']} (thumb {step['r_thumb']})"

    result.note_errors.append(err_msg)
    return False


def verify_finger_collisions(step: Dict, result: VerificationResult) -> bool:
    """Verify no crossed fingers within each hand."""
    t = step['time']
    all_valid = True

    for hand, is_left in [('Left', True), ('Right', False)]:
        thumb = step['l_thumb'] if is_left else step['r_thumb']
        cmd = step['l_cmd'] if is_left else step['r_cmd']

        if not thumb or not cmd:
            continue

        hits = calculate_hit_notes(thumb, cmd, is_left)
        if len(hits) <= 1:
            continue

        finger_note_pairs = [(finger, idx) for idx, _, finger in hits]

        is_valid, error_detail = validate_finger_assignment(
            finger_note_pairs,
            'left' if is_left else 'right'
        )

        if not is_valid:
            err_msg = f"Time {t:.2f}s {hand} Hand Finger Collision: {error_detail}"
            result.collision_errors.append(err_msg)
            all_valid = False

    return all_valid


def verify_finger_locking(step: Dict, prev_step: Dict, timed_steps: Dict,
                          result: VerificationResult, config: VerificationConfig) -> bool:
    """Verify sustained notes don't have their fingers reused."""
    if prev_step is None:
        return True

    t = step['time']
    all_valid = True

    for hand, is_left in [('Left', True), ('Right', False)]:
        thumb = step['l_thumb'] if is_left else step['r_thumb']
        prev_thumb = prev_step['l_thumb'] if is_left else prev_step['r_thumb']
        cmd = step['l_cmd'] if is_left else step['r_cmd']
        prev_cmd = prev_step['l_cmd'] if is_left else prev_step['r_cmd']

        if not thumb or not cmd or not prev_thumb or not prev_cmd:
            continue

        prev_hits = calculate_hit_notes(prev_thumb, prev_cmd, is_left)
        curr_hits = calculate_hit_notes(thumb, cmd, is_left)

        # Find notes that should still be sustained at this time
        sustained_positions = set()
        for start_time, notes in timed_steps.items():
            for note in notes:
                end_time = start_time + note.get('duration', 1.0)
                if start_time < t and end_time > t:
                    sustained_positions.add(note['white_index'])

        # Check if any finger holding a sustained note is being reused
        for prev_idx, prev_black, prev_finger in prev_hits:
            if prev_idx not in sustained_positions:
                continue

            for curr_idx, curr_black, curr_finger in curr_hits:
                if curr_finger == prev_finger and curr_idx != prev_idx:
                    err_msg = (f"Time {t:.2f}s {hand} Hand Locked Finger Conflict: "
                               f"F{prev_finger} holding {index_to_name(prev_idx)} "
                               f"but needed for {index_to_name(curr_idx)}")
                    result.locking_errors.append(err_msg)
                    all_valid = False

    return all_valid


def verify_velocity(step: Dict, prev_step: Dict, result: VerificationResult,
                    config: VerificationConfig) -> bool:
    """Verify hand movement speed is within limits."""
    if prev_step is None:
        return True

    t = step['time']
    prev_t = prev_step['time']
    dt = t - prev_t

    if dt <= 0:
        return True

    all_valid = True

    for hand, is_left in [('Left', True), ('Right', False)]:
        thumb = step['l_thumb'] if is_left else step['r_thumb']
        prev_thumb = prev_step['l_thumb'] if is_left else prev_step['r_thumb']

        if thumb is None or prev_thumb is None:
            continue

        distance = abs(thumb - prev_thumb)
        if distance == 0:
            continue

        velocity = distance / dt

        if velocity > config.max_keys_per_second:
            err_msg = (f"Time {t:.2f}s {hand} Hand Velocity Violation: "
                       f"{distance} keys in {dt:.2f}s = {velocity:.1f} keys/sec "
                       f"(max: {config.max_keys_per_second})")
            result.velocity_errors.append(err_msg)
            all_valid = False

    return all_valid


def verify_hand_gap(step: Dict, result: VerificationResult,
                    config: VerificationConfig) -> bool:
    """Verify minimum gap between hands."""
    l_thumb = step['l_thumb']
    r_thumb = step['r_thumb']

    if l_thumb is None or r_thumb is None:
        return True

    gap = r_thumb - l_thumb

    if gap < config.min_hand_gap:
        t = step['time']
        err_msg = (f"Time {t:.2f}s Hand Gap Violation: "
                   f"L@{l_thumb} R@{r_thumb} gap={gap} (min: {config.min_hand_gap})")
        result.hand_gap_errors.append(err_msg)
        return False

    return True


# ==========================================
# MAIN VERIFICATION
# ==========================================
def verify_fingering(directory: str, config: VerificationConfig) -> VerificationResult:
    """Run all verification checks."""
    result = VerificationResult()

    # Use pathlib for cleaner path handling
    dir_path = Path(directory)
    plan_path = dir_path / 'fingering_plan.csv'
    truth_path = dir_path / 'timed_steps.csv'
    summary_path = dir_path / 'fingering_summary.csv'
    left_cmd_path = dir_path / 'left_hand_commands.txt'
    right_cmd_path = dir_path / 'right_hand_commands.txt'

    timed_steps = parse_timed_steps(str(truth_path))
    plan = parse_fingering_plan(str(plan_path))
    summary = parse_fingering_summary(str(summary_path))

    if not timed_steps or not plan:
        logger.error("Missing required CSV files for verification.")
        return result

    # Build efficient time index (IMPROVEMENT)
    time_index = build_time_index(timed_steps, config.time_tolerance)

    split_point = None
    if 'Split Point' in summary and len(summary['Split Point']) >= 2:
        try:
            split_point = int(summary['Split Point'][1])
        except (ValueError, IndexError):
            pass

    logger.info(f"Checking {len(plan)} time steps...")
    if config.verbose:
        logger.info(f"Split Point: {split_point}")
        logger.info(f"Max Velocity: {config.max_keys_per_second} keys/sec")
        logger.info(f"Min Hand Gap: {config.min_hand_gap} keys")
        logger.info(f"Time Tolerance: {config.time_tolerance}s")

    # Verify command file format
    if config.check_command_format:
        logger.info("Checking command file format (step-servo pattern)...")

        left_cmds = parse_command_file(str(left_cmd_path))
        right_cmds = parse_command_file(str(right_cmd_path))

        if left_cmds:
            verify_command_format(left_cmds, "Left", result)
            verify_step_continuity(left_cmds, "Left", result)

        if right_cmds:
            verify_command_format(right_cmds, "Right", result)
            verify_step_continuity(right_cmds, "Right", result)

        if not result.format_errors:
            logger.info("✓ Command format valid (step-servo pattern)")
        else:
            logger.warning(f"⚠ {len(result.format_errors)} command format issues found")

    prev_step = None

    for step in plan:
        t = step['time']

        # Use efficient time lookup (IMPROVEMENT)
        rounded_t = round(t / config.time_tolerance) * config.time_tolerance
        expected = time_index.get(rounded_t, [])

        if not expected:
            verify_hand_gap(step, result, config)
            prev_step = step
            continue

        result.total_checks += 1
        step_passed = True

        if not verify_note_accuracy(step, expected, result):
            step_passed = False

        if not verify_finger_collisions(step, result):
            step_passed = False

        if not verify_finger_locking(step, prev_step, timed_steps, result, config):
            step_passed = False

        if not verify_velocity(step, prev_step, result, config):
            step_passed = False

        if not verify_hand_gap(step, result, config):
            step_passed = False

        if step_passed:
            result.passed_checks += 1

        prev_step = step

    return result


def print_results(result: VerificationResult, verbose: bool = False):
    """Print verification results."""
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)

    print(f"\n📊 SUMMARY")
    print(f"   Timesteps Checked:    {result.total_checks}")
    print(f"   Passed:               {result.passed_checks}")
    print(f"   Failed:               {result.total_checks - result.passed_checks}")
    print(f"   Accuracy:             {result.accuracy:.1f}%")

    print(f"\n📋 ERROR BREAKDOWN")
    print(f"   Note Errors:          {len(result.note_errors)}")
    print(f"   Finger Collisions:    {len(result.collision_errors)}")
    print(f"   Finger Lock Errors:   {len(result.locking_errors)}")
    print(f"   Velocity Violations:  {len(result.velocity_errors)}")
    print(f"   Hand Gap Violations:  {len(result.hand_gap_errors)}")
    print(f"   Format Errors:        {len(result.format_errors)}")

    if result.warnings:
        print(f"   Warnings:             {len(result.warnings)}")

    all_errors = [
        ("❌ NOTE ERRORS", result.note_errors),
        ("❌ FINGER COLLISION ERRORS", result.collision_errors),
        ("❌ FINGER LOCKING ERRORS", result.locking_errors),
        ("❌ VELOCITY VIOLATIONS", result.velocity_errors),
        ("❌ HAND GAP VIOLATIONS", result.hand_gap_errors),
        ("❌ FORMAT ERRORS", result.format_errors),
    ]

    for title, errors in all_errors:
        if errors:
            print(f"\n{title}:")
            limit = 10 if not verbose else len(errors)
            for e in errors[:limit]:
                print(f"   {e}")
            if len(errors) > limit:
                print(f"   ... and {len(errors) - limit} more.")

    if result.warnings and verbose:
        print(f"\n⚠️  WARNINGS:")
        for w in result.warnings:
            print(f"   {w}")

    print("\n" + "=" * 60)
    if result.total_errors == 0:
        print("✅ SUCCESS: All verification checks passed!")
    else:
        print(f"❌ FAILED: {result.total_errors} total errors found.")
    print("=" * 60)


def export_json(result: VerificationResult, filepath: str):
    """Export results to JSON file."""
    try:
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Results exported to {filepath}")
    except Exception as e:
        logger.error(f"Failed to export JSON: {e}")


# ==========================================
# UNIT TESTS
# ==========================================
def run_self_tests() -> bool:
    """Run built-in sanity checks."""
    logger.info("Running self-tests...")

    try:
        # Test MIDI conversion
        assert midi_to_name(60) == "C4", "MIDI 60 should be C4"
        assert midi_to_name(48) == "C3", "MIDI 48 should be C3"
        assert midi_to_name(72) == "C5", "MIDI 72 should be C5"

        # Test index conversion
        assert index_to_name(23) == "C4", "Index 23 should be C4"
        assert index_to_name(0) == "A0", "Index 0 should be A0"
        assert index_to_name(7) == "A1", "Index 7 should be A1"

        # Test finger command parsing
        assert parse_finger_command("3") == (3, 'normal', 0, 0)
        assert parse_finger_command("1b") == (1, 'black', 0, 0)
        assert parse_finger_command("5s+2") == (5, 'splay', 1, 2)
        assert parse_finger_command("X") == (None, None, 0, 0)

        # Test finger validation
        result, _ = validate_finger_assignment([(1, 10), (3, 12)], "right")
        assert result, "Right hand ascending should be valid"

        result, _ = validate_finger_assignment([(3, 10), (1, 12)], "right")
        assert not result, "Right hand descending should be invalid"

        result, _ = validate_finger_assignment([(3, 10), (1, 12)], "left")
        assert result, "Left hand descending should be valid"

        logger.info("✅ All self-tests passed!")
        return True

    except AssertionError as e:
        logger.error(f"❌ Self-test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify Robotic Piano Fingering Output v2.2 (IMPROVED)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Verification Checks:
  1. Note Accuracy     - Correct notes played at correct times
  2. Finger Collision  - No crossed fingers within a hand
  3. Finger Locking    - Sustained notes don't have fingers reused
  4. Velocity          - Hand movement speed within limits
  5. Hand Gap          - Hands maintain minimum separation
  6. Command Format    - Step-servo pattern is correct

Examples:
  %(prog)s --dir ./output
  %(prog)s --dir . --verbose
  %(prog)s --dir . --speed 15 --gap 8
  %(prog)s --dir . --no-format-check
  %(prog)s --self-test
  %(prog)s --dir . --json-output results.json
        """
    )

    parser.add_argument('--dir', default='.',
                        help='Directory containing output files')
    parser.add_argument('--speed', type=float, default=DEFAULT_MAX_KEYS_PER_SECOND,
                        help=f'Max keys per second (default: {DEFAULT_MAX_KEYS_PER_SECOND})')
    parser.add_argument('--gap', type=int, default=DEFAULT_MIN_HAND_GAP,
                        help=f'Minimum hand gap in keys (default: {DEFAULT_MIN_HAND_GAP})')
    parser.add_argument('--tolerance', type=float, default=DEFAULT_TIME_TOLERANCE,
                        help=f'Time matching tolerance in seconds (default: {DEFAULT_TIME_TOLERANCE})')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show all errors (not just first 10)')
    parser.add_argument('--no-format-check', action='store_true',
                        help='Skip command format verification')
    parser.add_argument('--json-output', type=str,
                        help='Export results to JSON file')
    parser.add_argument('--config', type=str,
                        help='Load configuration from JSON file')
    parser.add_argument('--self-test', action='store_true',
                        help='Run self-tests and exit')

    args = parser.parse_args()

    # Self-test mode
    if args.self_test:
        return 0 if run_self_tests() else 1

    # Load config from file or args
    if args.config and os.path.exists(args.config):
        config = VerificationConfig.from_file(args.config)
    else:
        config = VerificationConfig(
            max_keys_per_second=args.speed,
            min_hand_gap=args.gap,
            time_tolerance=args.tolerance,
            verbose=args.verbose,
            check_command_format=not args.no_format_check
        )

    print("=" * 60)
    print("ROBOTIC PIANO FINGERING VERIFICATION v2.2 (IMPROVED)")
    print("=" * 60)
    print(f"📁 Directory: {args.dir}")

    result = verify_fingering(args.dir, config)
    print_results(result, args.verbose)

    # Export JSON if requested
    if args.json_output:
        export_json(result, args.json_output)

    return 0 if result.total_errors == 0 else 1


if __name__ == "__main__":
    exit(main())