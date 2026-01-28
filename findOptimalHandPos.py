import csv
import os
import sys
import argparse
from music21 import *

# ==========================================
# GLOBAL CONFIGURATION (Defaults)
# These are updated by CLI arguments at runtime
# ==========================================
MOVE_PENALTY = 4
AUTO_TRANSPOSE = True
MAX_KEYS_PER_SECOND = 10.0
VELOCITY_PENALTY = 100
MIN_HAND_GAP = 6  # Minimum keys between Left Hand Max and Right Hand Min


# ==========================================
# PART 1: MUSICXML PARSER
# ==========================================

def midi_to_white_key_index(midi):
    """Convert MIDI note number to white key index (0-based, 0 = A0)."""
    offset = midi - 21  # A0 = MIDI 21 (standard piano range starts here)
    octave = offset // 12
    note_in_octave = offset % 12
    # Correct map for A-based indexing: A=0, B=2, C=3, D=5, E=7, F=8, G=10
    white_key_map = {0: 0, 2: 1, 3: 2, 5: 3, 7: 4, 8: 5, 10: 6}
    if note_in_octave not in white_key_map:
        return None
    return octave * 7 + white_key_map[note_in_octave]


def index_to_note_name(white_key_index):
    """Convert white key index back to note name (0 = A0, 1 = B0, 2 = C1, etc.)."""
    if white_key_index is None:
        return "None"
    # A0 and B0 are in octave 0, C1-G1 in octave 1, etc.
    position = white_key_index % 7
    names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

    # Calculate octave number: A and B stay in their group's octave,
    # but C-G are labeled with the next octave number
    if position <= 1:  # A or B
        octave = white_key_index // 7
    else:  # C, D, E, F, or G
        octave = white_key_index // 7 + 1

    return f"{names[position]}{octave}"


def parse_musicxml(file, auto_transpose=True):
    """
    Parse MusicXML and optionally transpose to white-keys-only (C Major or A Minor).
    Reports any black keys that remain after transposition.

    Args:
        file: Path to MusicXML file
        auto_transpose: If True, transposes to C Major/A Minor and filters black keys.
                       If False, includes all notes (for black key capable hardware).
    """
    try:
        score = converter.parse(file)
    except Exception as e:
        print(f"❌ Error parsing file: {e}")
        sys.exit(1)

    transposed = False
    target_key = None

    if auto_transpose:
        try:
            original_key = score.analyze('key')
            print(f"  Detected original key: {original_key.name}")

            if original_key.mode == 'major':
                target_key = key.Key('C')
            else:
                target_key = key.Key('a')

            original_tonic_midi = original_key.tonic.midi
            target_tonic_midi = target_key.tonic.midi
            semitones = (target_tonic_midi - original_tonic_midi) % 12
            if semitones > 6:
                semitones -= 12

            if semitones != 0:
                score = score.transpose(semitones)
                transposed = True
                print(f"  Transposed to {target_key.name} ({semitones:+d} semitones)")
            else:
                print(f"  Already in {target_key.name}")

        except Exception as e:
            print(f"  Warning: Auto-transpose failed ({e}). Using original key.")

    black_keys_found = []
    note_info = []

    for part in score.parts:
        for music_element in part.flatten().notesAndRests:
            if isinstance(music_element, note.Rest):
                continue

            if isinstance(music_element, note.Note):
                if music_element.pitch.alter != 0:
                    if auto_transpose:
                        black_keys_found.append({
                            'time': float(music_element.offset),
                            'note': music_element.pitch.nameWithOctave,
                            'type': 'single note'
                        })
                        continue

                info = {
                    'type': 'note',
                    'pitch': (music_element.pitch.step,
                              music_element.pitch.octave,
                              music_element.pitch.alter,
                              music_element.pitch.midi),
                    'duration': music_element.quarterLength,
                    'white_key_index': midi_to_white_key_index(music_element.pitch.midi),
                    'offset': music_element.offset
                }
                note_info.append(info)

            elif isinstance(music_element, chord.Chord):
                black_notes_in_chord = [n for n in music_element.notes if n.pitch.alter != 0]
                white_notes = [n for n in music_element.notes if n.pitch.alter == 0]

                if auto_transpose and black_notes_in_chord:
                    black_keys_found.append({
                        'time': float(music_element.offset),
                        'note': ', '.join([n.pitch.nameWithOctave for n in black_notes_in_chord]),
                        'type': 'chord'
                    })

                notes_to_include = white_notes if auto_transpose else music_element.notes

                if notes_to_include:
                    info = {
                        'type': 'chord',
                        'pitches': [(n.pitch.step, n.pitch.octave, n.pitch.alter, n.pitch.midi)
                                    for n in notes_to_include],
                        'duration': music_element.quarterLength,
                        'white_key_indices': [midi_to_white_key_index(n.pitch.midi)
                                              for n in notes_to_include],
                        'offset': music_element.offset
                    }
                    note_info.append(info)

    if auto_transpose and black_keys_found:
        save_black_key_report(file, black_keys_found, transposed, target_key)
    elif auto_transpose:
        print("\n✓ SUCCESS: All notes converted to white keys only!\n")
    else:
        print(f"\n✓ Loaded {len(note_info)} note events (including black keys)\n")

    note_info.sort(key=lambda x: x['offset'])
    return note_info


def save_black_key_report(file, black_keys_found, transposed, target_key):
    """Save detailed report of black keys that couldn't be converted."""
    output_dir = os.path.dirname(os.path.abspath(file)) if os.path.dirname(file) else "."
    csv_path = os.path.join(output_dir, "black_keys_report.csv")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Status', 'UNABLE TO CONVERT FULL SONG TO WHITE KEYS ONLY'])
        writer.writerow(['Transposed', 'Yes' if transposed else 'No'])
        if transposed and target_key:
            writer.writerow(['Target Key', target_key.name])
        writer.writerow([])
        writer.writerow(['Time (beats)', 'Note(s)', 'Context'])

        for item in black_keys_found:
            writer.writerow([f"{item['time']:.2f}", item['note'], item['type']])

        writer.writerow([])
        writer.writerow(['Total Black Key Occurrences', len(black_keys_found)])
        writer.writerow(['Warning', 'These notes will be SKIPPED during playback'])

    print(f"\n⚠️  WARNING: {len(black_keys_found)} black key events will be skipped.")
    print(f"   Details saved to: {csv_path}\n")


def convert_to_timed_steps(note_info):
    """Convert parsed note info to (time, [(midi, duration, white_key_index)]) format."""
    timed_steps = []
    for n in note_info:
        time_step = []
        if n['type'] == 'chord':
            for i, pitch in enumerate(n['pitches']):
                if n['white_key_indices'][i] is not None:
                    time_step.append((pitch[3], n['duration'], n['white_key_indices'][i]))
        else:
            if n['white_key_index'] is not None:
                time_step.append((n['pitch'][3], n['duration'], n['white_key_index']))

        if time_step:
            timed_steps.append((n['offset'], time_step))

    return timed_steps


def save_timed_steps_csv(timed_steps, output_dir="."):
    """Save intermediate timed steps to CSV for debugging and optimizer input."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "timed_steps.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_time", "midi", "duration", "white_key_index"])
        for start_time, step in timed_steps:
            for midi, duration, white_key_index in step:
                writer.writerow([start_time, midi, duration, white_key_index])


# ==========================================
# PART 2: FINGERING OPTIMIZER
# ==========================================

def load_notes_grouped_by_time(filename):
    """Load notes from CSV and group by timestamp."""
    notes_by_time = []
    current_time_step = None

    if not os.path.exists(filename):
        return []

    with open(filename, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            try:
                start_time = float(row['start_time'])
                white_key_index = int(row['white_key_index'])
                duration = float(row['duration'])
                key_name = index_to_note_name(white_key_index)

                # Group notes that occur at the same time (within 1ms tolerance)
                if current_time_step is None or abs(start_time - current_time_step['time']) > 0.001:
                    current_time_step = {
                        'time': start_time,
                        'notes': [white_key_index],
                        'durations': [duration],
                        'keys': [key_name]
                    }
                    notes_by_time.append(current_time_step)
                else:
                    current_time_step['notes'].append(white_key_index)
                    current_time_step['durations'].append(duration)
                    current_time_step['keys'].append(key_name)
            except (ValueError, KeyError):
                pass

    return notes_by_time


def get_active_notes_at_time(note_groups, current_index):
    """
    Get notes that are still being held (sustained) at the current time.
    This ensures the optimizer doesn't move the hand away from sustained notes.

    FIXED: Uses TIME-based lookback instead of INDEX-based to handle slow music correctly.
    """
    if current_index >= len(note_groups):
        return set()

    if note_groups[current_index] is None:
        return set()

    current_time = note_groups[current_index]['time']
    active_notes = set()

    # FIXED: Look back by TIME (10 seconds), not by INDEX
    # This handles slow music (whole notes, fermatas, etc.) correctly
    MAX_LOOKBACK_TIME = 10.0  # seconds - longer than any reasonable piano sustain

    for i in range(current_index - 1, -1, -1):
        group = note_groups[i]

        if group is None:
            continue

        # Stop if we've gone back more than 10 seconds
        if current_time - group['time'] > MAX_LOOKBACK_TIME:
            break

        group_start = group['time']

        for note_idx, duration in zip(group['notes'], group['durations']):
            # Check if note is still being held (with 50ms buffer for rounding)
            note_end_time = group_start + duration
            if note_end_time > current_time + 0.05:
                active_notes.add(note_idx)

    return active_notes


def get_possible_states(note_groups, current_index, max_position=None, min_position=None):
    """
    Get possible thumb positions with sustain checking.
    A position is only valid if ALL currently-held notes can still be reached.
    """
    note_group = note_groups[current_index]
    if note_group is None:
        return []

    # Get notes that need to start at this time
    new_notes = set(note_group['notes'])

    # Get notes that are still being held from previous time steps
    active_notes = get_active_notes_at_time(note_groups, current_index)

    # Combine new and sustained notes
    all_required_notes = new_notes | active_notes

    if not all_required_notes:
        return []

    min_note = min(all_required_notes)
    max_note = max(all_required_notes)
    span = max_note - min_note

    # Enhanced Error Reporting with anti-spam protection
    if span > 4:
        # Only print error once per unique timestamp to avoid console spam
        if not hasattr(get_possible_states, "last_error_time") or \
                get_possible_states.last_error_time != note_group['time']:
            print(f"\n❌ IMPOSSIBLE REACH at Time {note_group['time']:.2f}s:")
            print(f"   Required Notes (indices): {sorted(list(all_required_notes))}")
            print(f"   Span: {span + 1} keys (Max allowed: 5)")
            print(f"   Context: New notes {sorted(list(new_notes))} + Sustained {sorted(list(active_notes))}")
            get_possible_states.last_error_time = note_group['time']
        return []

    # Calculate range of valid thumb positions
    # Thumb can be at most 4 keys below the highest note
    start_range = max(0, max_note - 4)
    end_range = min_note

    # Apply boundary constraints if specified
    if max_position is not None:
        end_range = min(end_range, max_position)
    if min_position is not None:
        start_range = max(start_range, min_position)

    possible = list(range(start_range, end_range + 1))
    return possible if possible else []


def calculate_transition_cost(prev_state, curr_state, time_delta):
    """
    Calculate the cost of transitioning from one thumb position to another.

    Args:
        prev_state: Previous thumb position (white key index)
        curr_state: Current thumb position (white key index)
        time_delta: Time difference between positions (in beats/seconds)

    Returns:
        Total transition cost (distance + penalties)
    """
    # Calculate distance traveled
    distance = abs(curr_state - prev_state)

    # Base cost is just the distance
    cost = distance

    # Add movement penalty if hand moved
    if distance > 0:
        cost += MOVE_PENALTY

        # Calculate velocity penalty if movement is too fast
        if time_delta > 0:
            required_velocity = distance / time_delta
            if required_velocity > MAX_KEYS_PER_SECOND:
                velocity_excess = required_velocity - MAX_KEYS_PER_SECOND
                cost += VELOCITY_PENALTY * velocity_excess

    return cost


def optimize_with_boundaries(note_groups, hand_name, max_boundary=None, min_boundary=None):
    """
    Viterbi algorithm to find optimal thumb position path.

    Args:
        note_groups: List of note groups at each time step
        hand_name: "Left" or "Right" for debugging
        max_boundary: Maximum thumb position allowed (for left hand)
        min_boundary: Minimum thumb position allowed (for right hand)

    Returns:
        List of optimal thumb positions for each time step
    """
    # Filter out None groups and track their original indices
    valid_groups = [(i, g) for i, g in enumerate(note_groups) if g is not None]
    if not valid_groups:
        return [None] * len(note_groups)

    n = len(valid_groups)
    dp = [{} for _ in range(n)]
    backpointer = [{} for _ in range(n)]

    # Initial State
    first_real_idx = valid_groups[0][0]
    first_states = get_possible_states(note_groups, first_real_idx, max_boundary, min_boundary)

    # Fallback if boundaries are too tight
    if not first_states:
        first_states = get_possible_states(note_groups, first_real_idx, None, None)

    if not first_states:
        print(f"❌ {hand_name} hand: No valid starting position found.")
        return []

    for state in first_states:
        dp[0][state] = 0

    # Viterbi Forward Pass
    for i in range(1, n):
        curr_real_idx = valid_groups[i][0]
        prev_real_idx = valid_groups[i - 1][0]

        possible_states = get_possible_states(note_groups, curr_real_idx, max_boundary, min_boundary)
        if not possible_states:
            possible_states = get_possible_states(note_groups, curr_real_idx, None, None)
        if not possible_states:
            print(f"❌ {hand_name} hand: No valid position at time {note_groups[curr_real_idx]['time']:.2f}s")
            return []

        time_delta = note_groups[curr_real_idx]['time'] - note_groups[prev_real_idx]['time']

        for curr_state in possible_states:
            min_cost = float('inf')
            best_prev = None

            for prev_state, prev_cost in dp[i - 1].items():
                cost = prev_cost + calculate_transition_cost(prev_state, curr_state, time_delta)

                # Boundary Soft Penalties (keeps hands apart)
                if max_boundary and curr_state > max_boundary:
                    cost += 1000
                if min_boundary and curr_state < min_boundary:
                    cost += 1000

                if cost < min_cost:
                    min_cost = cost
                    best_prev = prev_state

            if best_prev is not None:
                dp[i][curr_state] = min_cost
                backpointer[i][curr_state] = best_prev

    # Backtrack to find optimal path
    if not dp[-1]:
        print(f"❌ {hand_name} hand: Optimization failed - no valid path found.")
        return []

    curr_state = min(dp[-1], key=dp[-1].get)
    path_segment = [curr_state]

    for i in range(n - 1, 0, -1):
        prev_state = backpointer[i][curr_state]
        path_segment.insert(0, prev_state)
        curr_state = prev_state

    # Map back to full timeline
    full_path = [None] * len(note_groups)
    for k, (real_idx, _) in enumerate(valid_groups):
        full_path[real_idx] = path_segment[k]

    return full_path


def assign_hands_to_notes(note_groups, split_point):
    """
    Assign notes to left or right hand based on split point.
    Notes below split go to left hand, notes above split+gap go to right hand.
    """
    l_groups, r_groups = [], []

    for group in note_groups:
        if not group:
            l_groups.append(None)
            r_groups.append(None)
            continue

        l_notes, r_notes = [], []
        l_keys, r_keys = [], []
        l_durs, r_durs = [], []

        for note, key, dur in zip(group['notes'], group['keys'], group['durations']):
            # Determine hand assignment
            is_right = False
            if note >= split_point + MIN_HAND_GAP:
                is_right = True
            elif note > split_point and note < split_point + MIN_HAND_GAP:
                # In the gap: assign based on proximity to typical hand centers
                if note > split_point + (MIN_HAND_GAP / 2):
                    is_right = True

            if is_right:
                r_notes.append(note)
                r_keys.append(key)
                r_durs.append(dur)
            else:
                l_notes.append(note)
                l_keys.append(key)
                l_durs.append(dur)

        l_groups.append({
                            'time': group['time'],
                            'notes': l_notes,
                            'durations': l_durs,
                            'keys': l_keys
                        } if l_notes else None)

        r_groups.append({
                            'time': group['time'],
                            'notes': r_notes,
                            'durations': r_durs,
                            'keys': r_keys
                        } if r_notes else None)

    return l_groups, r_groups


def calculate_path_cost(path, note_groups):
    """Calculate total movement cost of a path."""
    if not path:
        return 0

    valid_indices = [i for i, p in enumerate(path) if p is not None]
    if len(valid_indices) < 2:
        return 0

    cost = 0
    for k in range(1, len(valid_indices)):
        curr_i, prev_i = valid_indices[k], valid_indices[k - 1]
        dt = note_groups[curr_i]['time'] - note_groups[prev_i]['time']
        cost += calculate_transition_cost(path[prev_i], path[curr_i], dt)

    return cost


def find_optimal_split_point(note_groups):
    """
    Find optimal split point using 'Coarse-to-Fine' search.
    Phase 1: Check every 3rd key
    Phase 2: Refine around best candidate
    """
    all_notes = []
    for g in note_groups:
        all_notes.extend(g['notes'])

    if not all_notes:
        return None

    min_n, max_n = min(all_notes), max(all_notes)

    # Phase 1: Coarse Search (Every 3 keys)
    candidates = []
    step = 3
    print(
        f"  Phase 1: Scanning split points from {index_to_note_name(min_n)} to {index_to_note_name(max_n)} (step={step})...")

    search_range = list(range(min_n, max_n + 1, step))
    total_checks = len(search_range)
    checked = 0

    for split in search_range:
        checked += 1
        l_groups, r_groups = assign_hands_to_notes(note_groups, split)

        # Quick feasibility check before full Viterbi
        l_path = optimize_with_boundaries(l_groups, "Left", max_boundary=split)
        if not l_path:
            continue

        r_path = optimize_with_boundaries(r_groups, "Right", min_boundary=split + MIN_HAND_GAP)
        if not r_path:
            continue

        cost = calculate_path_cost(l_path, l_groups) + calculate_path_cost(r_path, r_groups)
        candidates.append((split, cost))
        print(f"\r    Progress: {checked}/{total_checks} - Testing {index_to_note_name(split)}: Cost {cost:.0f}  ",
              end="", flush=True)

    print()  # New line after progress

    if not candidates:
        print("  ❌ No valid split points found in coarse search.")
        return None

    best_candidate = min(candidates, key=lambda x: x[1])
    best_split, best_cost = best_candidate

    # Phase 2: Refine neighborhood (+/- 2 keys)
    print(f"  Phase 2: Refining around {index_to_note_name(best_split)}...")
    final_best_split = best_split
    final_best_cost = best_cost

    for split in range(best_split - 2, best_split + 3):
        if split == best_split or split < min_n or split > max_n:
            continue

        l_groups, r_groups = assign_hands_to_notes(note_groups, split)
        l_path = optimize_with_boundaries(l_groups, "Left", max_boundary=split)
        r_path = optimize_with_boundaries(r_groups, "Right", min_boundary=split + MIN_HAND_GAP)

        if l_path and r_path:
            cost = calculate_path_cost(l_path, l_groups) + calculate_path_cost(r_path, r_groups)
            if cost < final_best_cost:
                final_best_cost = cost
                final_best_split = split
                print(f"    Found better split: {index_to_note_name(split)} (Cost: {cost:.0f})")

    print(f"  ✓ Optimal Split: {index_to_note_name(final_best_split)} (Total Cost: {final_best_cost:.0f})")
    return final_best_split


def generate_servo_commands(hand_path, hand_groups, hand_name, start_position):
    """
    Generate servo commands from optimized path.

    FIXED: Properly handles preparation time by shifting timeline forward if needed.
    No negative timestamps are ever generated.

    Args:
        hand_path: List of thumb positions at each time step
        hand_groups: Note groups assigned to this hand
        hand_name: "Left" or "Right"
        start_position: Parked position (e.g., "G1" for left, "F7" for right)

    Returns:
        Tuple of (commands, time_shift):
            - commands: List of command strings in format "time:step:from-to" and "time:servo:1,3,5"
            - time_shift: Amount of time (seconds) the entire timeline was shifted forward
    """
    commands = []

    # Find first active note to move from park position
    first_idx = next((i for i, p in enumerate(hand_path) if p is not None), None)
    if first_idx is None:
        return [], 0.0

    curr_thumb = hand_path[first_idx]
    first_note_time = hand_groups[first_idx]['time']

    # Calculate time shift needed to ensure preparation time
    PREPARATION_TIME = 1.0  # seconds needed to move from park to position
    time_shift = 0.0

    if first_note_time < PREPARATION_TIME:
        # Shift entire timeline forward to make room for preparation
        time_shift = PREPARATION_TIME - first_note_time
        print(f"  ℹ️  {hand_name} hand: Timeline shifted forward by {time_shift:.2f}s for preparation")

    # Initial positioning command (always at t=0.0 after any shift)
    commands.append(f"0.0:step:{start_position}-{index_to_note_name(curr_thumb)}")
    prev_thumb = curr_thumb

    # Generate commands for all notes with time shift applied
    for i in range(first_idx, len(hand_groups)):
        if hand_path[i] is None:
            continue

        curr_thumb = hand_path[i]
        # Apply time shift to all timestamps
        curr_time = hand_groups[i]['time'] + time_shift

        # Step command (hand movement) - only if hand actually moved
        if curr_thumb != prev_thumb:
            commands.append(f"{curr_time:.3f}:step:{index_to_note_name(prev_thumb)}-{index_to_note_name(curr_thumb)}")

        # Servo command (finger activation)
        fingers = [str(n - curr_thumb + 1) for n in hand_groups[i]['notes']]
        commands.append(f"{curr_time:.3f}:servo:{','.join(fingers)}")

        prev_thumb = curr_thumb

    return commands, time_shift


def validate_output(l_path, r_path, note_groups):
    """
    Validate generated paths for physical feasibility.
    Checks for velocity violations and collision risks.

    Returns:
        List of warning/error messages
    """
    issues = []

    # 1. Velocity Checks
    for path, name in [(l_path, "Left"), (r_path, "Right")]:
        valid_idxs = [i for i, p in enumerate(path) if p is not None]
        for k in range(1, len(valid_idxs)):
            curr, prev = valid_idxs[k], valid_idxs[k - 1]
            dist = abs(path[curr] - path[prev])
            dt = note_groups[curr]['time'] - note_groups[prev]['time']
            if dt > 0 and (dist / dt) > MAX_KEYS_PER_SECOND:
                issues.append(
                    f"{name} Velocity Violation: {dist} keys in {dt:.3f}s "
                    f"({dist / dt:.1f} keys/sec) at Time {note_groups[curr]['time']:.2f}s"
                )

    # 2. Collision Checks
    for i in range(len(note_groups)):
        if l_path[i] is not None and r_path[i] is not None:
            # Check if hands are crossing
            if r_path[i] <= l_path[i]:
                issues.append(
                    f"Collision Risk: Hands crossed at Time {note_groups[i]['time']:.2f}s "
                    f"(Left thumb: {l_path[i]}, Right thumb: {r_path[i]})"
                )

    return issues


def save_outputs(l_cmd, r_cmd, l_path, r_path, l_groups, r_groups, split, note_groups, output_dir, time_shift=0.0):
    """Save all output files."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Servo command files
    with open(os.path.join(output_dir, "left_hand_commands.txt"), 'w') as f:
        f.write('\n'.join(l_cmd))

    with open(os.path.join(output_dir, "right_hand_commands.txt"), 'w') as f:
        f.write('\n'.join(r_cmd))

    # 2. Fingering Plan CSV
    with open(os.path.join(output_dir, "fingering_plan.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time', 'L_Notes', 'L_Thumb', 'L_Fingers', 'R_Notes', 'R_Thumb', 'R_Fingers'])

        for i in range(len(note_groups)):
            time = note_groups[i]['time'] if note_groups[i] else 0

            l_n = ';'.join(l_groups[i]['keys']) if l_groups[i] else ""
            l_t = str(l_path[i]) if l_path[i] is not None else ""
            l_f = ';'.join([str(n - l_path[i] + 1) for n in l_groups[i]['notes']]) if l_groups[i] and l_path[
                i] is not None else ""

            r_n = ';'.join(r_groups[i]['keys']) if r_groups[i] else ""
            r_t = str(r_path[i]) if r_path[i] is not None else ""
            r_f = ';'.join([str(n - r_path[i] + 1) for n in r_groups[i]['notes']]) if r_groups[i] and r_path[
                i] is not None else ""

            writer.writerow([time, l_n, l_t, l_f, r_n, r_t, r_f])

    # 3. Summary CSV
    with open(os.path.join(output_dir, "fingering_summary.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Left Hand', 'Right Hand', 'Combined'])

        l_moves = sum(1 for i in range(1, len(l_path))
                      if l_path[i] is not None and l_path[i - 1] is not None and l_path[i] != l_path[i - 1])
        r_moves = sum(1 for i in range(1, len(r_path))
                      if r_path[i] is not None and r_path[i - 1] is not None and r_path[i] != r_path[i - 1])

        writer.writerow(['Split Point', index_to_note_name(split), split, ''])
        writer.writerow(['Position Changes', l_moves, r_moves, l_moves + r_moves])
        writer.writerow(['Hardware Limit', f'{MAX_KEYS_PER_SECOND} keys/sec',
                         f'{MAX_KEYS_PER_SECOND} keys/sec', 'Enforced'])
        writer.writerow(['Movement Penalty', MOVE_PENALTY, MOVE_PENALTY, ''])
        writer.writerow(['Hand Gap', '', '', f'{MIN_HAND_GAP} keys'])

        # Add time shift information
        if time_shift > 0:
            writer.writerow(['Timeline Shift', f'{time_shift:.2f}s', f'{time_shift:.2f}s', 'Applied for preparation'])
        else:
            writer.writerow(['Timeline Shift', 'None', 'None', 'No shift needed'])


# ==========================================
# MAIN EXECUTION
# ==========================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Robotic Piano Fingering Optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s song.musicxml
  %(prog)s song.musicxml --speed 15 --gap 8
  %(prog)s song.musicxml --penalty 10 --no-transpose
  %(prog)s song.musicxml --output ./outputs
        """
    )

    parser.add_argument('file', nargs='?',
                        help='Input MusicXML file')
    parser.add_argument('--speed', type=float, default=10.0,
                        help='Max keys per second (default: 10)')
    parser.add_argument('--penalty', type=int, default=4,
                        help='Movement penalty cost (default: 4)')
    parser.add_argument('--gap', type=int, default=6,
                        help='Min keys between hands (default: 6)')
    parser.add_argument('--no-transpose', action='store_true',
                        help='Disable auto-transposition to white keys')
    parser.add_argument('--output', default='.',
                        help='Output directory (default: current directory)')

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    # Validate input file
    if not args.file:
        print("❌ Error: Please provide a MusicXML file")
        print("\nUsage: python findOptimalHandPos.py <file.musicxml> [options]")
        print("Use --help for more information")
        sys.exit(1)

    if not os.path.exists(args.file):
        print(f"❌ Error: File '{args.file}' not found")
        sys.exit(1)

    # Update Globals from CLI
    global MAX_KEYS_PER_SECOND, MOVE_PENALTY, MIN_HAND_GAP, AUTO_TRANSPOSE
    MAX_KEYS_PER_SECOND = args.speed
    MOVE_PENALTY = args.penalty
    MIN_HAND_GAP = args.gap
    AUTO_TRANSPOSE = not args.no_transpose

    print("=" * 50)
    print("ROBOTIC PIANO FINGERING OPTIMIZER")
    print("=" * 50)
    print(f"Input File:      {args.file}")
    print(f"Speed Limit:     {MAX_KEYS_PER_SECOND} keys/sec")
    print(f"Movement Penalty: {MOVE_PENALTY}")
    print(f"Hand Gap:        {MIN_HAND_GAP} keys")
    print(f"Auto-Transpose:  {AUTO_TRANSPOSE}")
    print(f"Output Dir:      {args.output}")
    print("=" * 50)
    print()

    # Step 1: Parse MusicXML
    print("STEP 1: Parsing MusicXML...")
    note_info = parse_musicxml(args.file, AUTO_TRANSPOSE)

    if not note_info:
        print("❌ No playable notes found.")
        sys.exit(1)

    timed_steps = convert_to_timed_steps(note_info)
    save_timed_steps_csv(timed_steps, args.output)
    print("  ✓ timed_steps.csv generated")

    # Step 2: Load grouped notes
    print("\nSTEP 2: Loading notes for optimization...")
    note_groups = load_notes_grouped_by_time(os.path.join(args.output, "timed_steps.csv"))
    print(f"  ✓ Loaded {len(note_groups)} time steps")

    # Step 3: Find optimal split point
    print("\nSTEP 3: Finding optimal split point...")
    split_point = find_optimal_split_point(note_groups)

    if split_point is None:
        print("❌ FATAL: Could not find any valid split point.")
        print("   The song may exceed the 5-key hand span or have other constraints.")
        sys.exit(1)

    # Step 4: Run global optimization
    print("\nSTEP 4: Running global path optimization...")
    l_groups, r_groups = assign_hands_to_notes(note_groups, split_point)

    print("  Optimizing left hand...")
    l_path = optimize_with_boundaries(l_groups, "Left", max_boundary=split_point)

    print("  Optimizing right hand...")
    r_path = optimize_with_boundaries(r_groups, "Right", min_boundary=split_point + MIN_HAND_GAP)

    if not l_path or not r_path:
        print("❌ FATAL: Optimization failed during path generation.")
        sys.exit(1)

    print("  ✓ Optimization complete")

    # Step 5: Safety validation
    print("\nSTEP 5: Validating output for safety...")
    issues = validate_output(l_path, r_path, note_groups)

    if issues:
        print("\n⚠️  SAFETY WARNINGS:")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"  • {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more warnings.")
        print("\n  Review these warnings before deploying to hardware!")
    else:
        print("  ✓ All safety checks passed")

    # Step 6: Generate servo commands
    print("\nSTEP 6: Generating servo commands...")
    l_cmd, l_shift = generate_servo_commands(l_path, l_groups, "Left", "G1")
    r_cmd, r_shift = generate_servo_commands(r_path, r_groups, "Right", "F7")
    print(f"  ✓ Generated {len(l_cmd)} left hand commands")
    print(f"  ✓ Generated {len(r_cmd)} right hand commands")

    # Report total timeline shift (use maximum of the two hands)
    max_shift = max(l_shift, r_shift)
    if max_shift > 0:
        print(f"  ℹ️  Total timeline shift: {max_shift:.2f}s (song now starts at t={max_shift:.2f}s)")

    # Step 7: Save all outputs
    print("\nSTEP 7: Saving output files...")
    save_outputs(l_cmd, r_cmd, l_path, r_path, l_groups, r_groups, split_point, note_groups, args.output, max_shift)

    print("\n" + "=" * 50)
    print("✓ OPTIMIZATION COMPLETE!")
    print("=" * 50)
    print(f"Split Point:     {index_to_note_name(split_point)} (index {split_point})")
    print(f"Output Files:    {args.output}/")
    print("  • left_hand_commands.txt")
    print("  • right_hand_commands.txt")
    print("  • fingering_plan.csv")
    print("  • fingering_summary.csv")
    print("  • timed_steps.csv")
    if AUTO_TRANSPOSE:
        print("  • black_keys_report.csv (if applicable)")
    print("=" * 50)


if __name__ == '__main__':
    main()