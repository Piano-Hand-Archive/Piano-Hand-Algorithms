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
# AUTO_TRANSPOSE = False  # Changed default: now supports black keys
MAX_KEYS_PER_SECOND = 10.0
VELOCITY_PENALTY = 100
MIN_HAND_GAP = 6  # Minimum keys between Left Hand Max and Right Hand Min

# ==========================================
# SPLAY CONFIGURATION
# Controls the cost/penalty for different finger techniques
# ==========================================
# Splay penalties (higher = avoid more)
INNER_FINGER_BLACK_KEY_PENALTY = 2  # Fingers 2,3,4 playing black keys (preferred)
OUTER_FINGER_BLACK_KEY_PENALTY = 10  # Thumb/pinky playing black keys (avoid)
OUTER_FINGER_SPLAY_PENALTY = 50  # Thumb/pinky reaching beyond natural range (last resort)
MAX_OUTER_SPLAY = 2  # Max white keys thumb/pinky can splay outward

# ==========================================
# LOOK-AHEAD CONFIGURATION
# Prevents "painted into a corner" situations by considering future notes
# ==========================================
LOOKAHEAD_STEPS = 3  # How many future time steps to consider
LOOKAHEAD_UNREACHABLE_PENALTY = 500  # Penalty if a position makes future note unreachable
LOOKAHEAD_DIFFICULT_PENALTY = 25  # Penalty per difficulty point for future positions
LOOKAHEAD_VELOCITY_PENALTY = 50  # Penalty if future movement would exceed speed limit

# ==========================================
# DYNAMIC SPLIT POINT CONFIGURATION
# Allows split point to shift during the piece for better fingering
# ==========================================
DYNAMIC_SPLIT_ENABLED = False  # Enable/disable dynamic split optimization
SPLIT_CHANGE_PENALTY = 100  # Penalty for changing split point between segments
SPLIT_MAX_CHANGE = 3  # Maximum keys split can shift between segments
SEGMENT_SIZE = 8  # Number of time steps per segment for split optimization


# ==========================================
# PART 1: MUSICXML PARSER (Enhanced for black keys)
# ==========================================

def midi_to_key_position(midi):
    """
    Convert MIDI note number to a unified key position system.

    Returns a tuple: (white_key_index, is_black_key, black_key_offset)
    - white_key_index: The white key this note is on or adjacent to (0 = A0)
    - is_black_key: True if this is a black key
    - black_key_offset: For black keys, which adjacent white key to use as reference
                        -1 = use lower white key, +1 = use upper white key, 0 = white key

    Black keys are positioned between white keys:
    - C#/Db is between C and D (can be played from either)
    - D#/Eb is between D and E
    - F#/Gb is between F and G
    - G#/Ab is between G and A
    - A#/Bb is between A and B
    """
    offset = midi - 21  # A0 = MIDI 21
    octave = offset // 12
    note_in_octave = offset % 12

    # Map: semitone -> (white_key_in_octave, is_black, preferred_offset)
    # For black keys, we initially set offset to 0; actual offset determined by context
    key_map = {
        0: (0, False, 0),  # A
        1: (0, True, 0),  # A#/Bb (between A and B)
        2: (1, False, 0),  # B
        3: (2, False, 0),  # C
        4: (2, True, 0),  # C#/Db (between C and D)
        5: (3, False, 0),  # D
        6: (3, True, 0),  # D#/Eb (between D and E)
        7: (4, False, 0),  # E
        8: (5, False, 0),  # F
        9: (5, True, 0),  # F#/Gb (between F and G)
        10: (6, False, 0),  # G
        11: (6, True, 0),  # G#/Ab (between G and A)
    }

    white_key_in_octave, is_black, _ = key_map[note_in_octave]
    white_key_index = octave * 7 + white_key_in_octave

    return (white_key_index, is_black, 0)


def midi_to_white_key_index(midi):
    """Convert MIDI note number to white key index (0-based, 0 = A0).
    Returns None for black keys (legacy behavior for compatibility)."""
    pos, is_black, _ = midi_to_key_position(midi)
    return None if is_black else pos


def index_to_note_name(white_key_index):
    """Convert white key index back to note name (0 = A0, 1 = B0, 2 = C1, etc.)."""
    if white_key_index is None:
        return "None"
    position = white_key_index % 7
    names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

    if position <= 1:  # A or B
        octave = white_key_index // 7
    else:  # C, D, E, F, or G
        octave = white_key_index // 7 + 1

    return f"{names[position]}{octave}"


def midi_to_note_name(midi):
    """Convert MIDI number to full note name including accidentals."""
    if midi is None:
        return "None"
    note_names = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    offset = midi - 21
    octave = offset // 12
    note_in_octave = offset % 12
    note_name = note_names[note_in_octave]

    # Adjust octave for C and above
    if note_in_octave >= 3:  # C or higher
        octave += 1

    return f"{note_name}{octave}"


def parse_musicxml(file, auto_transpose=False):
    """
    Parse MusicXML file, now with full black key support.

    Args:
        file: Path to MusicXML file
        auto_transpose: If True, transposes to C Major/A Minor (legacy mode).
                       If False (default), includes all notes including black keys.
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
                midi_num = music_element.pitch.midi
                is_black = music_element.pitch.alter != 0
                white_key_pos, _, _ = midi_to_key_position(midi_num)

                if is_black and auto_transpose:
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
                              midi_num),
                    'duration': music_element.quarterLength,
                    'white_key_index': white_key_pos,
                    'is_black': is_black,
                    'midi': midi_num,
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
                        'white_key_indices': [midi_to_key_position(n.pitch.midi)[0]
                                              for n in notes_to_include],
                        'is_black_list': [n.pitch.alter != 0 for n in notes_to_include],
                        'midi_list': [n.pitch.midi for n in notes_to_include],
                        'offset': music_element.offset
                    }
                    note_info.append(info)

    if auto_transpose and black_keys_found:
        save_black_key_report(file, black_keys_found, transposed, target_key)
    elif auto_transpose:
        print("\n✓ SUCCESS: All notes converted to white keys only!\n")
    else:
        black_count = sum(1 for n in note_info if n.get('is_black', False) or
                          any(n.get('is_black_list', [])))
        print(f"\n✓ Loaded {len(note_info)} note events ({black_count} contain black keys)\n")

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
    """
    Convert parsed note info to enhanced format with black key information.
    Returns: (time, [(midi, duration, white_key_index, is_black)])
    """
    timed_steps = []
    for n in note_info:
        time_step = []
        if n['type'] == 'chord':
            for i, pitch in enumerate(n['pitches']):
                midi = n['midi_list'][i]
                is_black = n['is_black_list'][i]
                white_key = n['white_key_indices'][i]
                time_step.append((midi, n['duration'], white_key, is_black))
        else:
            midi = n['midi']
            is_black = n.get('is_black', False)
            white_key = n['white_key_index']
            time_step.append((midi, n['duration'], white_key, is_black))

        if time_step:
            timed_steps.append((n['offset'], time_step))

    return timed_steps


def save_timed_steps_csv(timed_steps, output_dir="."):
    """Save intermediate timed steps to CSV for debugging and optimizer input."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "timed_steps.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_time", "midi", "duration", "white_key_index", "is_black"])
        for start_time, step in timed_steps:
            for midi, duration, white_key_index, is_black in step:
                writer.writerow([start_time, midi, duration, white_key_index, int(is_black)])


# ==========================================
# PART 2: FINGERING OPTIMIZER (Enhanced)
# ==========================================

def load_notes_grouped_by_time(filename):
    """Load notes from CSV and group by timestamp, including black key info."""
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
                midi = int(row['midi'])
                is_black = bool(int(row.get('is_black', 0)))
                key_name = midi_to_note_name(midi)

                if current_time_step is None or abs(start_time - current_time_step['time']) > 0.001:
                    current_time_step = {
                        'time': start_time,
                        'notes': [white_key_index],
                        'durations': [duration],
                        'keys': [key_name],
                        'midi_notes': [midi],
                        'is_black': [is_black]
                    }
                    notes_by_time.append(current_time_step)
                else:
                    current_time_step['notes'].append(white_key_index)
                    current_time_step['durations'].append(duration)
                    current_time_step['keys'].append(key_name)
                    current_time_step['midi_notes'].append(midi)
                    current_time_step['is_black'].append(is_black)
            except (ValueError, KeyError) as e:
                pass

    return notes_by_time


def get_active_notes_at_time(note_groups, current_index):
    """
    Get notes that are still being held (sustained) at the current time.
    Returns set of (white_key_index, is_black) tuples.
    """
    if current_index >= len(note_groups):
        return set()

    if note_groups[current_index] is None:
        return set()

    current_time = note_groups[current_index]['time']
    active_notes = set()

    MAX_LOOKBACK_TIME = 10.0

    for i in range(current_index - 1, -1, -1):
        group = note_groups[i]

        if group is None:
            continue

        if current_time - group['time'] > MAX_LOOKBACK_TIME:
            break

        group_start = group['time']

        for idx, (note_idx, duration, is_black) in enumerate(zip(
                group['notes'], group['durations'], group.get('is_black', [False] * len(group['notes'])))):
            note_end_time = group_start + duration
            if note_end_time > current_time + 0.05:
                active_notes.add((note_idx, is_black))

    return active_notes


def calculate_finger_for_note(thumb_pos, note_pos, hand, is_black=False, all_notes_in_chord=None):
    """
    Calculate which finger would play a note given thumb position.

    Returns: (finger_number, technique, penalty, splay_direction, splay_distance)
    - finger_number: 1-5 (1=thumb, 5=pinky), or None if unreachable
    - technique: 'normal', 'black_key_inner', 'black_key_outer', 'splay_thumb', 'splay_pinky'
    - penalty: Additional cost for this fingering
    - splay_direction: For black keys and splays:
        -1 = splay toward lower keys (left on keyboard)
        +1 = splay toward higher keys (right on keyboard)
        0 = no splay (normal white key)
    - splay_distance: Number of keys splayed (0 for normal, 1+ for extended splay)

    For LEFT hand: thumb at highest position, fingers extend DOWN
    For RIGHT hand: thumb at lowest position, fingers extend UP
    """
    if hand == "left":
        # Left hand: thumb is highest, pinky is lowest
        # finger = thumb_pos - note_pos + 1
        natural_finger = thumb_pos - note_pos + 1
    else:
        # Right hand: thumb is lowest, pinky is highest
        # finger = note_pos - thumb_pos + 1
        natural_finger = note_pos - thumb_pos + 1

    # Check natural range (fingers 1-5)
    if 1 <= natural_finger <= 5:
        if is_black:
            # Black key - determine splay direction
            splay_dir = determine_black_key_anchor(natural_finger, note_pos, hand, all_notes_in_chord)

            # Black key - prefer middle fingers
            if 2 <= natural_finger <= 4:
                return (natural_finger, 'black_key_inner', INNER_FINGER_BLACK_KEY_PENALTY, splay_dir, 0)
            else:
                return (natural_finger, 'black_key_outer', OUTER_FINGER_BLACK_KEY_PENALTY, splay_dir, 0)
        else:
            return (natural_finger, 'normal', 0, 0, 0)

    # Check extended range via splaying (thumb and pinky only)
    if natural_finger < 1:
        # Need to reach beyond thumb
        splay_amount = 1 - natural_finger  # How many keys beyond thumb
        if splay_amount <= MAX_OUTER_SPLAY:
            # Thumb can splay to reach this
            # Direction: thumb splays OUTWARD from hand
            # Left hand thumb splays toward HIGHER keys (+1)
            # Right hand thumb splays toward LOWER keys (-1)
            splay_dir = +1 if hand == "left" else -1
            return (1, 'splay_thumb', OUTER_FINGER_SPLAY_PENALTY * splay_amount, splay_dir, splay_amount)

    if natural_finger > 5:
        # Need to reach beyond pinky
        splay_amount = natural_finger - 5  # How many keys beyond pinky
        if splay_amount <= MAX_OUTER_SPLAY:
            # Pinky can splay to reach this
            # Direction: pinky splays OUTWARD from hand
            # Left hand pinky splays toward LOWER keys (-1)
            # Right hand pinky splays toward HIGHER keys (+1)
            splay_dir = -1 if hand == "left" else +1
            return (5, 'splay_pinky', OUTER_FINGER_SPLAY_PENALTY * splay_amount, splay_dir, splay_amount)

    # Unreachable
    return (None, 'unreachable', float('inf'), 0, 0)


def determine_black_key_anchor(finger, note_pos, hand, all_notes_in_chord=None):
    """
    Decide whether to play a black key by splaying from the lower or upper adjacent white key.

    Black key at white_key_index N sits between white keys N and N+1.

    Args:
        finger: Which finger (1-5) is playing this black key
        note_pos: White key index of the black key's position
        hand: "left" or "right"
        all_notes_in_chord: List of (white_key_index, is_black) for all notes in current chord

    Returns:
        -1 = splay from lower white key (toward left/lower on keyboard)
        +1 = splay from upper white key (toward right/higher on keyboard)

    Priority:
    1. If one adjacent white key is already being played, use the OTHER one
    2. Prefer the direction that requires less stretch (toward hand center)
    3. Default: use lower white key
    """
    lower_white = note_pos  # Black key's white_key_index IS the lower adjacent
    upper_white = note_pos + 1  # Next white key up

    # Check if either adjacent white key is being played in this chord
    if all_notes_in_chord:
        white_keys_in_chord = [n for n, is_blk in all_notes_in_chord if not is_blk]

        lower_occupied = lower_white in white_keys_in_chord
        upper_occupied = upper_white in white_keys_in_chord

        # If one is occupied and the other isn't, use the unoccupied one
        if lower_occupied and not upper_occupied:
            return +1  # Splay from upper (toward higher keys)
        if upper_occupied and not lower_occupied:
            return -1  # Splay from lower (toward lower keys)

        # If both or neither are occupied, fall through to preference logic

    # Preference based on finger position and hand
    # Middle fingers (2,3,4): prefer splaying toward hand center for comfort
    # Thumb (1) and Pinky (5): prefer splaying away from hand center

    if hand == "left":
        # Left hand: thumb at top (high keys), pinky at bottom (low keys)
        # Hand center is around finger 3
        if finger <= 2:
            # Fingers closer to thumb - splay toward lower keys feels natural
            return -1
        elif finger >= 4:
            # Fingers closer to pinky - splay toward higher keys feels natural
            return +1
        else:
            # Middle finger - default to lower
            return -1
    else:
        # Right hand: thumb at bottom (low keys), pinky at top (high keys)
        # Hand center is around finger 3
        if finger <= 2:
            # Fingers closer to thumb - splay toward higher keys feels natural
            return +1
        elif finger >= 4:
            # Fingers closer to pinky - splay toward lower keys feels natural
            return -1
        else:
            # Middle finger - default to lower
            return -1


def can_reach_all_notes(thumb_pos, notes_with_black, hand):
    """
    Check if all notes can be reached from this thumb position.

    Args:
        thumb_pos: White key index of thumb position
        notes_with_black: List of (white_key_index, is_black) tuples
        hand: "left" or "right"

    Returns: (can_reach, total_penalty, fingerings)
    - can_reach: True if all notes are reachable
    - total_penalty: Sum of all fingering penalties
    - fingerings: List of (finger, technique, splay_direction, splay_distance) for each note
    """
    total_penalty = 0
    fingerings = []

    for note_pos, is_black in notes_with_black:
        # Pass all notes in chord for black key anchor determination
        finger, technique, penalty, splay_dir, splay_dist = calculate_finger_for_note(
            thumb_pos, note_pos, hand, is_black, notes_with_black
        )

        if finger is None:
            return (False, float('inf'), [])

        total_penalty += penalty
        fingerings.append((finger, technique, splay_dir, splay_dist))

    return (True, total_penalty, fingerings)


def get_possible_states_extended(note_groups, current_index, max_position=None, min_position=None, hand="right"):
    """
    Get possible thumb positions with extended reach via splaying.

    Returns list of (thumb_position, penalty) tuples, sorted by preference.
    Prioritizes positions that don't require splaying.
    """
    note_group = note_groups[current_index]
    if note_group is None:
        return []

    # Get notes that need to start at this time
    new_notes = [(n, b) for n, b in
                 zip(note_group['notes'], note_group.get('is_black', [False] * len(note_group['notes'])))]

    # Get notes that are still being held
    active_notes = get_active_notes_at_time(note_groups, current_index)

    # Combine new and sustained notes
    all_required = set(new_notes) | active_notes

    if not all_required:
        return []

    # Get range of white key positions
    white_positions = [n[0] for n in all_required]
    min_note = min(white_positions)
    max_note = max(white_positions)

    # Calculate natural span (without splaying)
    natural_span = max_note - min_note

    # Maximum possible span with splaying on both ends
    max_span_with_splay = 4 + (2 * MAX_OUTER_SPLAY)  # 4 natural + 2 each side = 8

    if natural_span > max_span_with_splay:
        if not hasattr(get_possible_states_extended, "last_error_time") or \
                get_possible_states_extended.last_error_time != note_group['time']:
            print(f"\n❌ IMPOSSIBLE REACH at Time {note_group['time']:.2f}s:")
            print(f"   Required Notes: {sorted(list(all_required))}")
            print(f"   Span: {natural_span + 1} keys (Max with splay: {max_span_with_splay + 1})")
            get_possible_states_extended.last_error_time = note_group['time']
        return []

    # Generate candidate thumb positions
    # Expand search range to account for splaying
    if hand == "left":
        # Left hand: thumb at highest position
        # Natural range: thumb at max_note to thumb at min_note + 4
        # With splay: thumb can be up to MAX_OUTER_SPLAY lower (pinky splays further down)
        #             or MAX_OUTER_SPLAY higher (thumb splays further up)
        search_start = max(0, max_note - MAX_OUTER_SPLAY)
        search_end = min_note + 4 + MAX_OUTER_SPLAY
    else:
        # Right hand: thumb at lowest position
        # Natural range: thumb at max_note - 4 to thumb at min_note
        # With splay: thumb can be MAX_OUTER_SPLAY lower or higher
        search_start = max(0, max_note - 4 - MAX_OUTER_SPLAY)
        search_end = min_note + MAX_OUTER_SPLAY

    # Apply boundary constraints
    if max_position is not None:
        search_end = min(search_end, max_position)
    if min_position is not None:
        search_start = max(search_start, min_position)

    candidates = []

    for thumb_pos in range(search_start, search_end + 1):
        can_reach, penalty, fingerings = can_reach_all_notes(thumb_pos, list(all_required), hand)

        if can_reach:
            # Add look-ahead penalty to avoid getting "painted into a corner"
            lookahead_penalty = calculate_lookahead_penalty(
                note_groups, current_index, thumb_pos, hand, max_position, min_position
            )
            total_penalty = penalty + lookahead_penalty
            candidates.append((thumb_pos, total_penalty, fingerings))

    # Sort by penalty (prefer positions with less splaying/outer finger black keys)
    candidates.sort(key=lambda x: x[1])

    return candidates


def calculate_lookahead_penalty(note_groups, current_index, thumb_pos, hand, max_position=None, min_position=None):
    """
    Calculate penalty based on how well this thumb position sets up for future notes.

    This prevents the algorithm from choosing a position that works now but makes
    upcoming passages impossible or very difficult.

    Args:
        note_groups: All note groups
        current_index: Current time step
        thumb_pos: Proposed thumb position to evaluate
        hand: "left" or "right"
        max_position: Maximum boundary for this hand
        min_position: Minimum boundary for this hand

    Returns:
        Penalty value (0 = position works well for future, higher = problematic)
    """
    total_penalty = 0
    current_time = note_groups[current_index]['time'] if note_groups[current_index] else 0

    # Look ahead at future time steps
    future_steps_checked = 0

    for future_idx in range(current_index + 1, len(note_groups)):
        if future_steps_checked >= LOOKAHEAD_STEPS:
            break

        future_group = note_groups[future_idx]
        if future_group is None:
            continue

        future_steps_checked += 1
        future_time = future_group['time']
        time_delta = future_time - current_time

        # Get future notes (including any sustained notes)
        future_notes = [(n, b) for n, b in zip(
            future_group['notes'],
            future_group.get('is_black', [False] * len(future_group['notes']))
        )]

        # Also consider notes that will still be sustained at this future time
        future_active = get_active_notes_at_time(note_groups, future_idx)
        all_future_required = set(future_notes) | future_active

        if not all_future_required:
            continue

        # Calculate what thumb positions would be valid for the future notes
        future_positions = get_valid_thumb_positions_for_notes(
            all_future_required, hand, max_position, min_position
        )

        if not future_positions:
            # No valid position exists for future notes - this is bad but not our fault
            continue

        # Check if we can reach ANY valid future position from current thumb_pos
        min_distance_to_valid = min(abs(thumb_pos - fp) for fp in future_positions)

        # Check 1: Is any future position reachable given velocity constraints?
        if time_delta > 0:
            required_velocity = min_distance_to_valid / time_delta
            if required_velocity > MAX_KEYS_PER_SECOND:
                # Would need to move too fast - add heavy penalty
                velocity_excess = required_velocity - MAX_KEYS_PER_SECOND
                total_penalty += LOOKAHEAD_VELOCITY_PENALTY * velocity_excess

        # Check 2: How much movement is required?
        # Penalize positions that require a lot of future movement
        # Weight by how soon the future notes occur (closer = more important)
        time_weight = 1.0 / (future_steps_checked)  # Earlier steps weighted more
        movement_penalty = min_distance_to_valid * MOVE_PENALTY * time_weight * 0.5
        total_penalty += movement_penalty

        # Check 3: Would staying put work? (Ideal case)
        if thumb_pos in future_positions:
            # Current position works for future - small bonus (negative penalty)
            total_penalty -= 2 * time_weight

        # Update current time for velocity calculations in next iteration
        current_time = future_time

    return max(0, total_penalty)  # Don't allow negative total penalty


def get_valid_thumb_positions_for_notes(notes_with_black, hand, max_position=None, min_position=None):
    """
    Get all valid thumb positions that can reach a set of notes.

    This is a simplified version of get_possible_states_extended that doesn't
    need the full note_groups context - just checks reachability.

    Args:
        notes_with_black: Set of (white_key_index, is_black) tuples
        hand: "left" or "right"
        max_position: Maximum boundary
        min_position: Minimum boundary

    Returns:
        List of valid thumb positions
    """
    if not notes_with_black:
        return []

    white_positions = [n[0] for n in notes_with_black]
    min_note = min(white_positions)
    max_note = max(white_positions)

    natural_span = max_note - min_note
    max_span_with_splay = 4 + (2 * MAX_OUTER_SPLAY)

    if natural_span > max_span_with_splay:
        return []  # Impossible span

    # Calculate search range based on hand
    if hand == "left":
        search_start = max(0, max_note - MAX_OUTER_SPLAY)
        search_end = min_note + 4 + MAX_OUTER_SPLAY
    else:
        search_start = max(0, max_note - 4 - MAX_OUTER_SPLAY)
        search_end = min_note + MAX_OUTER_SPLAY

    # Apply boundaries
    if max_position is not None:
        search_end = min(search_end, max_position)
    if min_position is not None:
        search_start = max(search_start, min_position)

    valid_positions = []

    for thumb_pos in range(search_start, search_end + 1):
        can_reach, _, _ = can_reach_all_notes(thumb_pos, list(notes_with_black), hand)
        if can_reach:
            valid_positions.append(thumb_pos)

    return valid_positions


def get_possible_states(note_groups, current_index, max_position=None, min_position=None, hand="right"):
    """
    Wrapper for compatibility - returns just thumb positions (not penalties).
    """
    extended = get_possible_states_extended(note_groups, current_index, max_position, min_position, hand)
    return [pos for pos, _, _ in extended]


def calculate_transition_cost(prev_state, curr_state, time_delta, fingering_penalty=0):
    """
    Calculate the cost of transitioning from one thumb position to another.

    Args:
        prev_state: Previous thumb position (white key index)
        curr_state: Current thumb position (white key index)
        time_delta: Time difference between positions (in beats/seconds)
        fingering_penalty: Additional penalty for difficult fingerings (black keys, splay)

    Returns:
        Total transition cost (distance + penalties + fingering difficulty)
    """
    distance = abs(curr_state - prev_state)
    cost = distance

    if distance > 0:
        cost += MOVE_PENALTY

        if time_delta > 0:
            required_velocity = distance / time_delta
            if required_velocity > MAX_KEYS_PER_SECOND:
                velocity_excess = required_velocity - MAX_KEYS_PER_SECOND
                cost += VELOCITY_PENALTY * velocity_excess

    # Add fingering penalty
    cost += fingering_penalty

    return cost


def optimize_with_boundaries(note_groups, hand_name, max_boundary=None, min_boundary=None):
    """
    Viterbi algorithm to find optimal thumb position path with splay support.

    Now considers fingering penalties in addition to movement costs.
    """
    hand_type = "left" if hand_name.lower() == "left" else "right"

    valid_groups = [(i, g) for i, g in enumerate(note_groups) if g is not None]
    if not valid_groups:
        return [None] * len(note_groups)

    n = len(valid_groups)
    dp = [{} for _ in range(n)]
    backpointer = [{} for _ in range(n)]

    # Initial State - now with penalties
    first_real_idx = valid_groups[0][0]
    first_states = get_possible_states_extended(note_groups, first_real_idx, max_boundary, min_boundary, hand_type)

    if not first_states:
        first_states = get_possible_states_extended(note_groups, first_real_idx, None, None, hand_type)

    if not first_states:
        print(f"❌ {hand_name} hand: No valid starting position found.")
        return []

    for state, penalty, _ in first_states:
        dp[0][state] = penalty  # Start with fingering penalty

    # Viterbi Forward Pass
    for i in range(1, n):
        curr_real_idx = valid_groups[i][0]
        prev_real_idx = valid_groups[i - 1][0]

        possible_states = get_possible_states_extended(note_groups, curr_real_idx, max_boundary, min_boundary,
                                                       hand_type)
        if not possible_states:
            possible_states = get_possible_states_extended(note_groups, curr_real_idx, None, None, hand_type)
        if not possible_states:
            print(f"❌ {hand_name} hand: No valid position at time {note_groups[curr_real_idx]['time']:.2f}s")
            return []

        time_delta = note_groups[curr_real_idx]['time'] - note_groups[prev_real_idx]['time']

        for curr_state, fingering_penalty, _ in possible_states:
            min_cost = float('inf')
            best_prev = None

            for prev_state, prev_cost in dp[i - 1].items():
                cost = prev_cost + calculate_transition_cost(prev_state, curr_state, time_delta, fingering_penalty)

                # Boundary Soft Penalties
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

    # Backtrack
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


def optimize_with_dynamic_boundaries(note_groups, hand_name, split_sequence, is_left=True):
    """
    Viterbi algorithm with time-varying boundaries for dynamic split points.

    This is similar to optimize_with_boundaries, but the boundary changes
    at each time step according to the split_sequence.

    Args:
        note_groups: List of note groups at each time step
        hand_name: "Left" or "Right" for debugging
        split_sequence: List of split points, one per time step
        is_left: True for left hand, False for right hand

    Returns:
        List of optimal thumb positions for each time step
    """
    hand_type = "left" if hand_name.lower() == "left" else "right"

    # Filter out None groups and track their original indices
    valid_groups = [(i, g) for i, g in enumerate(note_groups) if g is not None]
    if not valid_groups:
        return [None] * len(note_groups)

    n = len(valid_groups)
    dp = [{} for _ in range(n)]
    backpointer = [{} for _ in range(n)]

    # Initial State
    first_real_idx = valid_groups[0][0]

    # Get boundary for first time step
    if is_left:
        max_boundary = split_sequence[first_real_idx]
        min_boundary = None
    else:
        max_boundary = None
        min_boundary = split_sequence[first_real_idx] + MIN_HAND_GAP

    first_states = get_possible_states_extended(note_groups, first_real_idx, max_boundary, min_boundary, hand_type)

    if not first_states:
        first_states = get_possible_states_extended(note_groups, first_real_idx, None, None, hand_type)

    if not first_states:
        print(f"❌ {hand_name} hand: No valid starting position found.")
        return []

    for state, penalty, _ in first_states:
        dp[0][state] = penalty

    # Viterbi Forward Pass
    for i in range(1, n):
        curr_real_idx = valid_groups[i][0]
        prev_real_idx = valid_groups[i - 1][0]

        # Get boundary for this time step
        if is_left:
            max_boundary = split_sequence[curr_real_idx]
            min_boundary = None
        else:
            max_boundary = None
            min_boundary = split_sequence[curr_real_idx] + MIN_HAND_GAP

        possible_states = get_possible_states_extended(note_groups, curr_real_idx, max_boundary, min_boundary,
                                                       hand_type)
        if not possible_states:
            possible_states = get_possible_states_extended(note_groups, curr_real_idx, None, None, hand_type)
        if not possible_states:
            print(f"❌ {hand_name} hand: No valid position at time {note_groups[curr_real_idx]['time']:.2f}s")
            return []

        time_delta = note_groups[curr_real_idx]['time'] - note_groups[prev_real_idx]['time']

        for curr_state, fingering_penalty, _ in possible_states:
            min_cost = float('inf')
            best_prev = None

            for prev_state, prev_cost in dp[i - 1].items():
                cost = prev_cost + calculate_transition_cost(prev_state, curr_state, time_delta, fingering_penalty)

                # Boundary Soft Penalties (with dynamic boundaries)
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

    # Backtrack
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


def check_adjacent_conflicts(notes_with_black):
    """
    Detect physically impossible combinations where a white key
    and its immediately adjacent black key must be played simultaneously.

    A black key at white_key_index N sits between white keys N and N+1.
    Conflicts occur when:
    - White key N and black key at N are played together (e.g., C and C#)
    - White key N+1 and black key at N are played together (e.g., D and C#)

    Args:
        notes_with_black: List of (white_key_index, is_black, key_name, midi, duration) tuples

    Returns:
        List of conflict tuples: (white_note_info, black_note_info)
        where each note_info is (white_key_index, is_black, key_name, midi, duration)
    """
    conflicts = []

    white_notes = [n for n in notes_with_black if not n[1]]
    black_notes = [n for n in notes_with_black if n[1]]

    for white_note in white_notes:
        w_pos = white_note[0]
        for black_note in black_notes:
            b_pos = black_note[0]
            # Black key at position N is between white keys N and N+1
            # Conflict if white key is at N (lower adjacent) or N+1 (upper adjacent)
            if b_pos == w_pos or b_pos == w_pos - 1:
                conflicts.append((white_note, black_note))

    return conflicts


def resolve_conflicts_by_splitting(group, split_point, conflict_resolution_log):
    """
    Attempt to resolve adjacent white+black key conflicts by splitting notes between hands.

    Strategy:
    1. For each conflict, try to move the black key to the opposite hand
    2. If that's not possible (wrong side of split), try moving the white key
    3. If neither works, drop the black key (white keys are harmonically primary)

    Args:
        group: Note group dict with 'notes', 'keys', 'durations', 'midi_notes', 'is_black', 'time'
        split_point: Current split point between hands
        conflict_resolution_log: List to append resolution messages to

    Returns:
        Tuple of (left_group, right_group) with conflicts resolved
    """
    if not group:
        return None, None

    time = group['time']

    # Build full note info list
    notes_info = list(zip(
        group['notes'],
        group['is_black'],
        group['keys'],
        group['midi_notes'],
        group['durations']
    ))

    # Check for conflicts
    conflicts = check_adjacent_conflicts(notes_info)

    if not conflicts:
        # No conflicts, return None to signal normal processing
        return None, None

    # We have conflicts - need to resolve them
    # Start by assigning all notes normally
    left_notes = []
    right_notes = []
    dropped_notes = []

    # Track which notes are involved in conflicts
    conflict_black_notes = set()
    conflict_white_notes = set()
    for white_note, black_note in conflicts:
        conflict_white_notes.add(white_note[3])  # midi as unique identifier
        conflict_black_notes.add(black_note[3])

    for note_info in notes_info:
        note_pos, is_black, key_name, midi, dur = note_info

        # Determine natural hand assignment
        natural_hand = "right" if note_pos >= split_point + MIN_HAND_GAP else "left"
        if split_point < note_pos < split_point + MIN_HAND_GAP:
            natural_hand = "right" if note_pos > split_point + (MIN_HAND_GAP / 2) else "left"

        # Check if this note is part of a conflict
        is_conflicting_black = midi in conflict_black_notes
        is_conflicting_white = midi in conflict_white_notes

        if is_conflicting_black:
            # This is a black key in conflict - try to move to opposite hand
            opposite_hand = "left" if natural_hand == "right" else "right"

            # Check if opposite hand can physically reach this note
            if opposite_hand == "left" and note_pos <= split_point + 4:
                # Left hand can reach (with some stretch)
                left_notes.append(note_info)
                conflict_resolution_log.append(
                    f"  Time {time:.2f}s: Moved {key_name} to LEFT hand to resolve conflict"
                )
            elif opposite_hand == "right" and note_pos >= split_point + MIN_HAND_GAP - 4:
                # Right hand can reach (with some stretch)
                right_notes.append(note_info)
                conflict_resolution_log.append(
                    f"  Time {time:.2f}s: Moved {key_name} to RIGHT hand to resolve conflict"
                )
            else:
                # Can't split - drop the black key
                dropped_notes.append(note_info)
                conflict_resolution_log.append(
                    f"  Time {time:.2f}s: DROPPED {key_name} (conflict unresolvable, white key preserved)"
                )
        elif is_conflicting_white:
            # White key in conflict - keep in natural hand (priority)
            if natural_hand == "left":
                left_notes.append(note_info)
            else:
                right_notes.append(note_info)
        else:
            # Not in conflict - normal assignment
            if natural_hand == "left":
                left_notes.append(note_info)
            else:
                right_notes.append(note_info)

    # Build output groups
    def build_group(notes_list, time):
        if not notes_list:
            return None
        return {
            'time': time,
            'notes': [n[0] for n in notes_list],
            'is_black': [n[1] for n in notes_list],
            'keys': [n[2] for n in notes_list],
            'midi_notes': [n[3] for n in notes_list],
            'durations': [n[4] for n in notes_list]
        }

    return build_group(left_notes, time), build_group(right_notes, time)


def assign_hands_to_notes(note_groups, split_point, resolve_conflicts=True):
    """
    Assign notes to left or right hand based on split point.
    Now includes black key information and conflict resolution.

    Args:
        note_groups: List of note groups at each time step
        split_point: White key index dividing left and right hand territories
        resolve_conflicts: If True, attempt to resolve adjacent white+black key conflicts

    Returns:
        Tuple of (l_groups, r_groups, conflict_log)
        - l_groups: Left hand note groups
        - r_groups: Right hand note groups
        - conflict_log: List of conflict resolution messages
    """
    l_groups, r_groups = [], []
    conflict_log = []

    for group in note_groups:
        if not group:
            l_groups.append(None)
            r_groups.append(None)
            continue

        # First, check for and resolve conflicts if enabled
        if resolve_conflicts:
            resolved_left, resolved_right = resolve_conflicts_by_splitting(
                group, split_point, conflict_log
            )
            if resolved_left is not None or resolved_right is not None:
                # Conflicts were found and handled
                l_groups.append(resolved_left)
                r_groups.append(resolved_right)
                continue

        # No conflicts (or resolution disabled) - normal assignment
        l_notes, r_notes = [], []
        l_keys, r_keys = [], []
        l_durs, r_durs = [], []
        l_midi, r_midi = [], []
        l_black, r_black = [], []

        is_black_list = group.get('is_black', [False] * len(group['notes']))
        midi_list = group.get('midi_notes', [0] * len(group['notes']))

        for note, key, dur, midi, is_black in zip(
                group['notes'], group['keys'], group['durations'], midi_list, is_black_list):

            is_right = False
            if note >= split_point + MIN_HAND_GAP:
                is_right = True
            elif note > split_point and note < split_point + MIN_HAND_GAP:
                if note > split_point + (MIN_HAND_GAP / 2):
                    is_right = True

            if is_right:
                r_notes.append(note)
                r_keys.append(key)
                r_durs.append(dur)
                r_midi.append(midi)
                r_black.append(is_black)
            else:
                l_notes.append(note)
                l_keys.append(key)
                l_durs.append(dur)
                l_midi.append(midi)
                l_black.append(is_black)

        l_groups.append({
                            'time': group['time'],
                            'notes': l_notes,
                            'durations': l_durs,
                            'keys': l_keys,
                            'midi_notes': l_midi,
                            'is_black': l_black
                        } if l_notes else None)

        r_groups.append({
                            'time': group['time'],
                            'notes': r_notes,
                            'durations': r_durs,
                            'keys': r_keys,
                            'midi_notes': r_midi,
                            'is_black': r_black
                        } if r_notes else None)

    return l_groups, r_groups, conflict_log


def calculate_path_cost(path, note_groups, hand):
    """Calculate total movement cost of a path including fingering penalties."""
    if not path:
        return 0

    valid_indices = [i for i, p in enumerate(path) if p is not None]
    if len(valid_indices) < 2:
        return 0

    cost = 0
    for k in range(1, len(valid_indices)):
        curr_i, prev_i = valid_indices[k], valid_indices[k - 1]
        dt = note_groups[curr_i]['time'] - note_groups[prev_i]['time']

        # Calculate fingering penalty for current position
        notes_with_black = list(zip(
            note_groups[curr_i]['notes'],
            note_groups[curr_i].get('is_black', [False] * len(note_groups[curr_i]['notes']))
        ))
        _, fingering_penalty, _ = can_reach_all_notes(path[curr_i], notes_with_black, hand)

        cost += calculate_transition_cost(path[prev_i], path[curr_i], dt, fingering_penalty)

    return cost


def find_optimal_split_point(note_groups):
    """
    Find optimal split point using 'Coarse-to-Fine' search.
    """
    all_notes = []
    for g in note_groups:
        all_notes.extend(g['notes'])

    if not all_notes:
        return None

    min_n, max_n = min(all_notes), max(all_notes)

    # Phase 1: Coarse Search
    candidates = []
    step = 3
    print(
        f"  Phase 1: Scanning split points from {index_to_note_name(min_n)} to {index_to_note_name(max_n)} (step={step})...")

    search_range = list(range(min_n, max_n + 1, step))
    total_checks = len(search_range)
    checked = 0

    for split in search_range:
        checked += 1
        # Disable conflict resolution during search for speed
        l_groups, r_groups, _ = assign_hands_to_notes(note_groups, split, resolve_conflicts=False)

        l_path = optimize_with_boundaries(l_groups, "Left", max_boundary=split)
        if not l_path:
            continue

        r_path = optimize_with_boundaries(r_groups, "Right", min_boundary=split + MIN_HAND_GAP)
        if not r_path:
            continue

        cost = calculate_path_cost(l_path, l_groups, "left") + calculate_path_cost(r_path, r_groups, "right")
        candidates.append((split, cost))
        print(f"\r    Progress: {checked}/{total_checks} - Testing {index_to_note_name(split)}: Cost {cost:.0f}  ",
              end="", flush=True)

    print()

    if not candidates:
        print("  ❌ No valid split points found in coarse search.")
        return None

    best_candidate = min(candidates, key=lambda x: x[1])
    best_split, best_cost = best_candidate

    # Phase 2: Refine
    print(f"  Phase 2: Refining around {index_to_note_name(best_split)}...")
    final_best_split = best_split
    final_best_cost = best_cost

    for split in range(best_split - 2, best_split + 3):
        if split == best_split or split < min_n or split > max_n:
            continue

        l_groups, r_groups, _ = assign_hands_to_notes(note_groups, split, resolve_conflicts=False)
        l_path = optimize_with_boundaries(l_groups, "Left", max_boundary=split)
        r_path = optimize_with_boundaries(r_groups, "Right", min_boundary=split + MIN_HAND_GAP)

        if l_path and r_path:
            cost = calculate_path_cost(l_path, l_groups, "left") + calculate_path_cost(r_path, r_groups, "right")
            if cost < final_best_cost:
                final_best_cost = cost
                final_best_split = split
                print(f"    Found better split: {index_to_note_name(split)} (Cost: {cost:.0f})")

    print(f"  ✓ Optimal Split: {index_to_note_name(final_best_split)} (Total Cost: {final_best_cost:.0f})")
    return final_best_split


def find_dynamic_split_points(note_groups, base_split=None):
    """
    Find optimal split point sequence using Viterbi algorithm on segments.

    This allows the split point to shift during the piece, which helps with
    music that has wandering hand positions or wide-ranging passages.

    Args:
        note_groups: List of note groups at each time step
        base_split: Optional starting split point (if None, will find initial)

    Returns:
        List of split points, one per time step (may vary throughout piece)
    """
    if not DYNAMIC_SPLIT_ENABLED:
        # Fall back to static split
        static_split = base_split if base_split else find_optimal_split_point(note_groups)
        return [static_split] * len(note_groups)

    print("  Finding dynamic split points...")

    # Step 1: Divide into segments
    segments = create_segments(note_groups, SEGMENT_SIZE)
    n_segments = len(segments)

    if n_segments == 0:
        return []

    print(f"    Divided into {n_segments} segments of ~{SEGMENT_SIZE} steps each")

    # Step 2: Determine valid split range
    all_notes = []
    for g in note_groups:
        if g:
            all_notes.extend(g['notes'])

    if not all_notes:
        return []

    min_note, max_note = min(all_notes), max(all_notes)

    # Possible split points (with some margin)
    possible_splits = list(range(max(0, min_note - 2), max_note + 3))

    # Filter to feasible splits (must have room for MIN_HAND_GAP)
    possible_splits = [s for s in possible_splits if s + MIN_HAND_GAP <= max_note + 5]

    if not possible_splits:
        print("    ❌ No valid split points found")
        return []

    print(f"    Considering {len(possible_splits)} possible split points")

    # Step 3: Viterbi algorithm on segments
    # dp[segment][split] = minimum cost to reach this split at this segment
    # backpointer[segment][split] = previous split that led here

    dp = [{} for _ in range(n_segments)]
    backpointer = [{} for _ in range(n_segments)]

    # Initialize first segment
    print("    Evaluating segment 1...", end="", flush=True)
    for split in possible_splits:
        cost = evaluate_segment_with_split(segments[0], split)
        if cost < float('inf'):
            dp[0][split] = cost

    if not dp[0]:
        print("\n    ❌ No valid split for first segment")
        return []

    # Forward pass
    for seg_idx in range(1, n_segments):
        print(f"\r    Evaluating segment {seg_idx + 1}/{n_segments}...", end="", flush=True)

        for curr_split in possible_splits:
            # Cost of using this split for current segment
            segment_cost = evaluate_segment_with_split(segments[seg_idx], curr_split)

            if segment_cost >= float('inf'):
                continue

            # Find best previous split
            min_total_cost = float('inf')
            best_prev_split = None

            for prev_split, prev_cost in dp[seg_idx - 1].items():
                # Transition cost
                split_change = abs(curr_split - prev_split)

                # Check if change is within allowed range
                if split_change > SPLIT_MAX_CHANGE:
                    transition_cost = float('inf')  # Disallow large jumps
                else:
                    transition_cost = split_change * SPLIT_CHANGE_PENALTY

                total_cost = prev_cost + transition_cost + segment_cost

                if total_cost < min_total_cost:
                    min_total_cost = total_cost
                    best_prev_split = prev_split

            if best_prev_split is not None:
                dp[seg_idx][curr_split] = min_total_cost
                backpointer[seg_idx][curr_split] = best_prev_split

    print()  # New line after progress

    # Step 4: Backtrack to find optimal sequence
    if not dp[-1]:
        print("    ❌ No valid split sequence found")
        return []

    # Find best final split
    final_split = min(dp[-1], key=dp[-1].get)
    final_cost = dp[-1][final_split]

    # Backtrack
    split_sequence = [final_split]
    for seg_idx in range(n_segments - 1, 0, -1):
        prev_split = backpointer[seg_idx][split_sequence[0]]
        split_sequence.insert(0, prev_split)

    # Step 5: Expand segment splits to per-timestep splits
    per_timestep_splits = expand_splits_to_timesteps(split_sequence, segments, len(note_groups))

    # Report results
    unique_splits = list(set(split_sequence))
    print(f"    ✓ Found dynamic split sequence (Cost: {final_cost:.0f})")
    print(f"    Split points used: {[index_to_note_name(s) for s in sorted(unique_splits)]}")

    if len(unique_splits) > 1:
        changes = sum(1 for i in range(1, len(split_sequence)) if split_sequence[i] != split_sequence[i - 1])
        print(f"    Split changes: {changes} times during piece")

    return per_timestep_splits


def create_segments(note_groups, segment_size):
    """
    Divide note groups into segments for split optimization.

    Args:
        note_groups: List of note groups
        segment_size: Target number of time steps per segment

    Returns:
        List of segments, where each segment is a list of (original_index, note_group) tuples
    """
    segments = []
    current_segment = []

    for i, group in enumerate(note_groups):
        if group is not None:
            current_segment.append((i, group))

        # Check if segment is full
        if len(current_segment) >= segment_size:
            if current_segment:
                segments.append(current_segment)
            current_segment = []

    # Don't forget the last partial segment
    if current_segment:
        segments.append(current_segment)

    return segments


def evaluate_segment_with_split(segment, split_point):
    """
    Evaluate the cost of playing a segment with a given split point.

    This is a simplified evaluation that checks feasibility and estimates cost
    without running full Viterbi optimization (for speed).

    Args:
        segment: List of (original_index, note_group) tuples
        split_point: The split point to evaluate

    Returns:
        Estimated cost (float('inf') if infeasible)
    """
    if not segment:
        return 0

    total_cost = 0

    # Separate notes into left and right hand
    l_notes_all = []
    r_notes_all = []

    for _, group in segment:
        is_black_list = group.get('is_black', [False] * len(group['notes']))

        for note, is_black in zip(group['notes'], is_black_list):
            if note <= split_point:
                l_notes_all.append((note, is_black))
            elif note >= split_point + MIN_HAND_GAP:
                r_notes_all.append((note, is_black))
            else:
                # Note in the gap - assign to closer hand
                if note < split_point + MIN_HAND_GAP / 2:
                    l_notes_all.append((note, is_black))
                else:
                    r_notes_all.append((note, is_black))

    # Check feasibility for each hand
    # Left hand: check if all notes can be reached with any valid thumb position
    if l_notes_all:
        l_positions = get_valid_thumb_positions_for_notes(set(l_notes_all), "left", max_position=split_point)
        if not l_positions:
            return float('inf')  # Infeasible

        # Estimate cost based on range of notes
        l_note_positions = [n[0] for n in l_notes_all]
        l_range = max(l_note_positions) - min(l_note_positions) if l_note_positions else 0
        total_cost += l_range * 2  # Larger range = more movement likely

    # Right hand
    if r_notes_all:
        r_positions = get_valid_thumb_positions_for_notes(set(r_notes_all), "right",
                                                          min_position=split_point + MIN_HAND_GAP)
        if not r_positions:
            return float('inf')  # Infeasible

        r_note_positions = [n[0] for n in r_notes_all]
        r_range = max(r_note_positions) - min(r_note_positions) if r_note_positions else 0
        total_cost += r_range * 2

    # Bonus for balanced workload
    l_count = len(l_notes_all)
    r_count = len(r_notes_all)
    if l_count > 0 and r_count > 0:
        imbalance = abs(l_count - r_count) / max(l_count, r_count)
        total_cost += imbalance * 10  # Penalize very unbalanced splits

    return total_cost


def expand_splits_to_timesteps(split_sequence, segments, total_timesteps):
    """
    Expand segment-level splits to per-timestep splits.

    Args:
        split_sequence: List of splits, one per segment
        segments: List of segments (from create_segments)
        total_timesteps: Total number of time steps in original note_groups

    Returns:
        List of splits, one per timestep
    """
    per_timestep = [None] * total_timesteps

    for seg_idx, segment in enumerate(segments):
        split = split_sequence[seg_idx]
        for orig_idx, _ in segment:
            per_timestep[orig_idx] = split

    # Fill in None values (timesteps with no notes) with nearest split
    last_split = split_sequence[0] if split_sequence else None
    for i in range(total_timesteps):
        if per_timestep[i] is None:
            per_timestep[i] = last_split
        else:
            last_split = per_timestep[i]

    return per_timestep


def assign_hands_with_dynamic_splits(note_groups, split_sequence, resolve_conflicts=True):
    """
    Assign notes to hands using a per-timestep split sequence.

    This is the dynamic split version of assign_hands_to_notes.

    Args:
        note_groups: List of note groups
        split_sequence: List of split points, one per timestep
        resolve_conflicts: Whether to resolve adjacent key conflicts

    Returns:
        Tuple of (l_groups, r_groups, conflict_log)
    """
    l_groups, r_groups = [], []
    conflict_log = []

    for i, group in enumerate(note_groups):
        if not group:
            l_groups.append(None)
            r_groups.append(None)
            continue

        split_point = split_sequence[i]

        # Check for and resolve conflicts if enabled
        if resolve_conflicts:
            resolved_left, resolved_right = resolve_conflicts_by_splitting(
                group, split_point, conflict_log
            )
            if resolved_left is not None or resolved_right is not None:
                l_groups.append(resolved_left)
                r_groups.append(resolved_right)
                continue

        # Normal assignment with this timestep's split
        l_notes, r_notes = [], []
        l_keys, r_keys = [], []
        l_durs, r_durs = [], []
        l_midi, r_midi = [], []
        l_black, r_black = [], []

        is_black_list = group.get('is_black', [False] * len(group['notes']))
        midi_list = group.get('midi_notes', [0] * len(group['notes']))

        for note, key, dur, midi, is_black in zip(
                group['notes'], group['keys'], group['durations'], midi_list, is_black_list):

            is_right = False
            if note >= split_point + MIN_HAND_GAP:
                is_right = True
            elif note > split_point and note < split_point + MIN_HAND_GAP:
                if note > split_point + (MIN_HAND_GAP / 2):
                    is_right = True

            if is_right:
                r_notes.append(note)
                r_keys.append(key)
                r_durs.append(dur)
                r_midi.append(midi)
                r_black.append(is_black)
            else:
                l_notes.append(note)
                l_keys.append(key)
                l_durs.append(dur)
                l_midi.append(midi)
                l_black.append(is_black)

        l_groups.append({
                            'time': group['time'],
                            'notes': l_notes,
                            'durations': l_durs,
                            'keys': l_keys,
                            'midi_notes': l_midi,
                            'is_black': l_black
                        } if l_notes else None)

        r_groups.append({
                            'time': group['time'],
                            'notes': r_notes,
                            'durations': r_durs,
                            'keys': r_keys,
                            'midi_notes': r_midi,
                            'is_black': r_black
                        } if r_notes else None)

    return l_groups, r_groups, conflict_log


def generate_servo_commands(hand_path, hand_groups, hand_name, start_position):
    """
    Generate servo commands with enhanced fingering information.

    Now includes technique information with splay direction:
    - Normal: '1', '2', '3', '4', '5'
    - Black key: '2b-' (splay left), '3b+' (splay right)
    - Extended splay: '1s+2' (thumb splay right 2 keys), '5s-1' (pinky splay left 1 key)
    """
    commands = []
    hand_type = "left" if hand_name.lower() == "left" else "right"

    first_idx = next((i for i, p in enumerate(hand_path) if p is not None), None)
    if first_idx is None:
        return [], 0.0

    curr_thumb = hand_path[first_idx]
    first_note_time = hand_groups[first_idx]['time']

    PREPARATION_TIME = 1.0
    time_shift = 0.0

    if first_note_time < PREPARATION_TIME:
        time_shift = PREPARATION_TIME - first_note_time
        print(f"  ℹ️  {hand_name} hand: Timeline shifted forward by {time_shift:.2f}s for preparation")

    commands.append(f"0.0:step:{start_position}-{index_to_note_name(curr_thumb)}")
    prev_thumb = curr_thumb

    for i in range(first_idx, len(hand_groups)):
        if hand_path[i] is None:
            continue

        curr_thumb = hand_path[i]
        curr_time = hand_groups[i]['time'] + time_shift

        if curr_thumb != prev_thumb:
            commands.append(f"{curr_time:.3f}:step:{index_to_note_name(prev_thumb)}-{index_to_note_name(curr_thumb)}")

        # Calculate finger assignments with technique info
        notes_with_black = list(zip(
            hand_groups[i]['notes'],
            hand_groups[i].get('is_black', [False] * len(hand_groups[i]['notes']))
        ))

        finger_info = []
        for note_pos, is_black in notes_with_black:
            finger, technique, _, splay_dir, splay_dist = calculate_finger_for_note(
                curr_thumb, note_pos, hand_type, is_black, notes_with_black
            )

            # Format finger command with direction and distance
            finger_info.append(format_finger_command(finger, technique, splay_dir, splay_dist))

        commands.append(f"{curr_time:.3f}:servo:{','.join(finger_info)}")

        prev_thumb = curr_thumb

    return commands, time_shift


def format_finger_command(finger, technique, splay_direction, splay_distance):
    """
    Format a finger command string with technique, direction, and distance.

    Args:
        finger: Finger number (1-5)
        technique: 'normal', 'black_key_inner', 'black_key_outer', 'splay_thumb', 'splay_pinky'
        splay_direction: -1 (left/lower), 0 (none), +1 (right/higher)
        splay_distance: Number of keys splayed (0 for black keys, 1+ for extended)

    Returns:
        Formatted string like '2', '3b-', '1s+2'

    Format:
        <finger><type><direction>[distance]

        type:
            (none) = normal white key
            b = black key
            s = extended splay

        direction:
            - = toward lower keys (left on keyboard)
            + = toward higher keys (right on keyboard)

        distance (only for extended splay):
            1, 2, etc. = number of keys beyond natural range
    """
    if finger is None:
        return "X"  # Unreachable marker

    base = str(finger)

    if technique == 'normal':
        return base

    # Determine direction character
    if splay_direction < 0:
        dir_char = '-'
    elif splay_direction > 0:
        dir_char = '+'
    else:
        dir_char = ''

    if technique in ('black_key_inner', 'black_key_outer'):
        return f"{base}b{dir_char}"

    if technique in ('splay_thumb', 'splay_pinky'):
        return f"{base}s{dir_char}{splay_distance}"

    return base


def validate_output(l_path, r_path, note_groups, l_groups=None, r_groups=None):
    """
    Validate generated paths for physical feasibility.
    Now includes check for any remaining adjacent white+black key conflicts.
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

    # 2. Collision Checks (hands crossing)
    for i in range(len(note_groups)):
        if l_path[i] is not None and r_path[i] is not None:
            gap = r_path[i] - l_path[i]
            if gap < MIN_HAND_GAP:
                issues.append(
                    f"Collision Risk: Insufficient gap at Time {note_groups[i]['time']:.2f}s "
                    f"(Left thumb: {l_path[i]}, Right thumb: {r_path[i]}, Gap: {gap}, Min required: {MIN_HAND_GAP})"
                )

    # 3. Adjacent Key Conflicts (within each hand)
    if l_groups and r_groups:
        for hand_name, groups in [("Left", l_groups), ("Right", r_groups)]:
            for group in groups:
                if not group:
                    continue

                # Build note info for conflict check
                notes_info = list(zip(
                    group['notes'],
                    group['is_black'],
                    group['keys'],
                    group.get('midi_notes', [0] * len(group['notes'])),
                    group['durations']
                ))

                conflicts = check_adjacent_conflicts(notes_info)
                for white_note, black_note in conflicts:
                    issues.append(
                        f"Adjacent Key Conflict in {hand_name} hand at Time {group['time']:.2f}s: "
                        f"{white_note[2]} and {black_note[2]} cannot be played simultaneously by same hand"
                    )

    return issues


def save_outputs(l_cmd, r_cmd, l_path, r_path, l_groups, r_groups, split, note_groups, output_dir, time_shift=0.0,
                 conflict_log=None):
    """Save all output files with enhanced fingering information including splay direction."""
    os.makedirs(output_dir, exist_ok=True)

    if conflict_log is None:
        conflict_log = []

    # 1. Servo command files
    with open(os.path.join(output_dir, "left_hand_commands.txt"), 'w') as f:
        f.write('\n'.join(l_cmd))

    with open(os.path.join(output_dir, "right_hand_commands.txt"), 'w') as f:
        f.write('\n'.join(r_cmd))

    # 2. Fingering Plan CSV (enhanced with technique and direction info)
    with open(os.path.join(output_dir, "fingering_plan.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time', 'L_Notes', 'L_Thumb', 'L_Fingers', 'L_Techniques', 'L_Commands',
                         'R_Notes', 'R_Thumb', 'R_Fingers', 'R_Techniques', 'R_Commands'])

        for i in range(len(note_groups)):
            time = note_groups[i]['time'] if note_groups[i] else 0

            # Left hand
            l_n = ';'.join(l_groups[i]['keys']) if l_groups[i] else ""
            l_t = str(l_path[i]) if l_path[i] is not None else ""
            l_f = ""
            l_tech = ""
            l_cmds = ""
            if l_groups[i] and l_path[i] is not None:
                notes_with_black = list(zip(
                    l_groups[i]['notes'],
                    l_groups[i].get('is_black', [False] * len(l_groups[i]['notes']))
                ))
                fingers = []
                techniques = []
                commands = []
                for note_pos, is_black in notes_with_black:
                    finger, technique, _, splay_dir, splay_dist = calculate_finger_for_note(
                        l_path[i], note_pos, "left", is_black, notes_with_black
                    )
                    fingers.append(str(finger))
                    techniques.append(technique)
                    commands.append(format_finger_command(finger, technique, splay_dir, splay_dist))
                l_f = ';'.join(fingers)
                l_tech = ';'.join(techniques)
                l_cmds = ';'.join(commands)

            # Right hand
            r_n = ';'.join(r_groups[i]['keys']) if r_groups[i] else ""
            r_t = str(r_path[i]) if r_path[i] is not None else ""
            r_f = ""
            r_tech = ""
            r_cmds = ""
            if r_groups[i] and r_path[i] is not None:
                notes_with_black = list(zip(
                    r_groups[i]['notes'],
                    r_groups[i].get('is_black', [False] * len(r_groups[i]['notes']))
                ))
                fingers = []
                techniques = []
                commands = []
                for note_pos, is_black in notes_with_black:
                    finger, technique, _, splay_dir, splay_dist = calculate_finger_for_note(
                        r_path[i], note_pos, "right", is_black, notes_with_black
                    )
                    fingers.append(str(finger))
                    techniques.append(technique)
                    commands.append(format_finger_command(finger, technique, splay_dir, splay_dist))
                r_f = ';'.join(fingers)
                r_tech = ';'.join(techniques)
                r_cmds = ';'.join(commands)

            writer.writerow([time, l_n, l_t, l_f, l_tech, l_cmds, r_n, r_t, r_f, r_tech, r_cmds])

    # 3. Summary CSV
    with open(os.path.join(output_dir, "fingering_summary.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Left Hand', 'Right Hand', 'Combined'])

        l_moves = sum(1 for i in range(1, len(l_path))
                      if l_path[i] is not None and l_path[i - 1] is not None and l_path[i] != l_path[i - 1])
        r_moves = sum(1 for i in range(1, len(r_path))
                      if r_path[i] is not None and r_path[i - 1] is not None and r_path[i] != r_path[i - 1])

        # Count black keys and splays with direction tracking
        l_black = 0
        l_black_left = 0
        l_black_right = 0
        l_splay = 0
        r_black = 0
        r_black_left = 0
        r_black_right = 0
        r_splay = 0

        for i, group in enumerate(l_groups):
            if group and l_path[i] is not None:
                notes_with_black = list(zip(
                    group['notes'],
                    group.get('is_black', [False] * len(group['notes']))
                ))
                for note, is_black in zip(group['notes'], group.get('is_black', [])):
                    _, tech, _, splay_dir, _ = calculate_finger_for_note(
                        l_path[i], note, "left", is_black, notes_with_black
                    )
                    if is_black:
                        l_black += 1
                        if splay_dir < 0:
                            l_black_left += 1
                        else:
                            l_black_right += 1
                    if 'splay' in tech:
                        l_splay += 1

        for i, group in enumerate(r_groups):
            if group and r_path[i] is not None:
                notes_with_black = list(zip(
                    group['notes'],
                    group.get('is_black', [False] * len(group['notes']))
                ))
                for note, is_black in zip(group['notes'], group.get('is_black', [])):
                    _, tech, _, splay_dir, _ = calculate_finger_for_note(
                        r_path[i], note, "right", is_black, notes_with_black
                    )
                    if is_black:
                        r_black += 1
                        if splay_dir < 0:
                            r_black_left += 1
                        else:
                            r_black_right += 1
                    if 'splay' in tech:
                        r_splay += 1

        writer.writerow(['Split Point', index_to_note_name(split), split, ''])
        writer.writerow(['Position Changes', l_moves, r_moves, l_moves + r_moves])
        writer.writerow(['Black Key Notes', l_black, r_black, l_black + r_black])
        writer.writerow(['  - Splay Left (-)', l_black_left, r_black_left, l_black_left + r_black_left])
        writer.writerow(['  - Splay Right (+)', l_black_right, r_black_right, l_black_right + r_black_right])
        writer.writerow(['Extended Splay', l_splay, r_splay, l_splay + r_splay])

        # Conflict resolution stats
        if conflict_log:
            moved_count = sum(1 for msg in conflict_log if 'Moved' in msg)
            dropped_count = sum(1 for msg in conflict_log if 'DROPPED' in msg)
            writer.writerow(['Conflict Resolutions', '', '', len(conflict_log)])
            writer.writerow(['  - Notes Moved', '', '', moved_count])
            writer.writerow(['  - Notes Dropped', '', '', dropped_count])

        writer.writerow(['Hardware Limit', f'{MAX_KEYS_PER_SECOND} keys/sec',
                         f'{MAX_KEYS_PER_SECOND} keys/sec', 'Enforced'])
        writer.writerow(['Movement Penalty', MOVE_PENALTY, MOVE_PENALTY, ''])
        writer.writerow(['Hand Gap', '', '', f'{MIN_HAND_GAP} keys'])
        writer.writerow(['Look-Ahead Steps', '', '', LOOKAHEAD_STEPS])
        writer.writerow(['Dynamic Split', '', '', 'Enabled' if DYNAMIC_SPLIT_ENABLED else 'Disabled'])

        if time_shift > 0:
            writer.writerow(['Timeline Shift', f'{time_shift:.2f}s', f'{time_shift:.2f}s', 'Applied for preparation'])
        else:
            writer.writerow(['Timeline Shift', 'None', 'None', 'No shift needed'])

    # 4. Conflict Resolution Log (if any)
    if conflict_log:
        with open(os.path.join(output_dir, "conflict_resolutions.txt"), 'w') as f:
            f.write("Adjacent Key Conflict Resolution Log\n")
            f.write("=" * 50 + "\n\n")
            f.write("These notes had adjacent white+black key conflicts\n")
            f.write("that were resolved by splitting between hands or dropping.\n\n")
            for msg in conflict_log:
                f.write(msg + "\n")


# ==========================================
# MAIN EXECUTION
# ==========================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Robotic Piano Fingering Optimizer (v2 - Black Key Support)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s song.musicxml
  %(prog)s song.musicxml --speed 15 --gap 8
  %(prog)s song.musicxml --penalty 10 --transpose
  %(prog)s song.musicxml --splay-penalty 100 --black-penalty 5
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
    parser.add_argument('--transpose', action='store_true',
                        help='Enable auto-transposition to white keys only (legacy mode)')
    parser.add_argument('--output', default='.',
                        help='Output directory (default: current directory)')

    # New splay/black key configuration
    parser.add_argument('--splay-penalty', type=int, default=50,
                        help='Penalty for thumb/pinky splay (default: 50)')
    parser.add_argument('--black-inner-penalty', type=int, default=2,
                        help='Penalty for middle fingers on black keys (default: 2)')
    parser.add_argument('--black-outer-penalty', type=int, default=10,
                        help='Penalty for thumb/pinky on black keys (default: 10)')
    parser.add_argument('--max-splay', type=int, default=2,
                        help='Max keys thumb/pinky can splay (default: 2)')

    # Look-ahead configuration
    parser.add_argument('--lookahead', type=int, default=3,
                        help='Number of future steps to consider (default: 3, 0 to disable)')
    parser.add_argument('--lookahead-penalty', type=int, default=50,
                        help='Penalty for positions that limit future movement (default: 50)')

    # Dynamic split configuration
    parser.add_argument('--dynamic-split', action='store_true',
                        help='Enable dynamic split point optimization')
    parser.add_argument('--split-change-penalty', type=int, default=100,
                        help='Penalty for changing split point between segments (default: 100)')
    parser.add_argument('--split-max-change', type=int, default=3,
                        help='Maximum keys split can shift between segments (default: 3)')
    parser.add_argument('--segment-size', type=int, default=8,
                        help='Time steps per segment for split optimization (default: 8)')

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

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
    global OUTER_FINGER_SPLAY_PENALTY, INNER_FINGER_BLACK_KEY_PENALTY
    global OUTER_FINGER_BLACK_KEY_PENALTY, MAX_OUTER_SPLAY
    global LOOKAHEAD_STEPS, LOOKAHEAD_VELOCITY_PENALTY
    global DYNAMIC_SPLIT_ENABLED, SPLIT_CHANGE_PENALTY, SPLIT_MAX_CHANGE, SEGMENT_SIZE

    MAX_KEYS_PER_SECOND = args.speed
    MOVE_PENALTY = args.penalty
    MIN_HAND_GAP = args.gap
    AUTO_TRANSPOSE = args.transpose
    OUTER_FINGER_SPLAY_PENALTY = args.splay_penalty
    INNER_FINGER_BLACK_KEY_PENALTY = args.black_inner_penalty
    OUTER_FINGER_BLACK_KEY_PENALTY = args.black_outer_penalty
    MAX_OUTER_SPLAY = args.max_splay
    LOOKAHEAD_STEPS = args.lookahead
    LOOKAHEAD_VELOCITY_PENALTY = args.lookahead_penalty
    DYNAMIC_SPLIT_ENABLED = args.dynamic_split
    SPLIT_CHANGE_PENALTY = args.split_change_penalty
    SPLIT_MAX_CHANGE = args.split_max_change
    SEGMENT_SIZE = args.segment_size

    print("=" * 60)
    print("ROBOTIC PIANO FINGERING OPTIMIZER v2.2")
    print("(Black Keys, Extended Reach, Look-Ahead, Dynamic Split)")
    print("=" * 60)
    print(f"Input File:           {args.file}")
    print(f"Speed Limit:          {MAX_KEYS_PER_SECOND} keys/sec")
    print(f"Movement Penalty:     {MOVE_PENALTY}")
    print(f"Hand Gap:             {MIN_HAND_GAP} keys")
    print(f"Auto-Transpose:       {AUTO_TRANSPOSE}")
    print(f"Max Splay:            {MAX_OUTER_SPLAY} white keys")
    print(f"Splay Penalty:        {OUTER_FINGER_SPLAY_PENALTY}")
    print(f"Black Key (inner):    {INNER_FINGER_BLACK_KEY_PENALTY}")
    print(f"Black Key (outer):    {OUTER_FINGER_BLACK_KEY_PENALTY}")
    print(f"Look-Ahead Steps:     {LOOKAHEAD_STEPS}")
    print(f"Look-Ahead Penalty:   {LOOKAHEAD_VELOCITY_PENALTY}")
    print(f"Dynamic Split:        {DYNAMIC_SPLIT_ENABLED}")
    if DYNAMIC_SPLIT_ENABLED:
        print(f"  Split Change Penalty: {SPLIT_CHANGE_PENALTY}")
        print(f"  Max Split Change:     {SPLIT_MAX_CHANGE} keys")
        print(f"  Segment Size:         {SEGMENT_SIZE} steps")
    print(f"Output Dir:           {args.output}")
    print("=" * 60)
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

    # Step 3: Find optimal split point(s)
    print("\nSTEP 3: Finding optimal split point...")

    if DYNAMIC_SPLIT_ENABLED:
        # Use dynamic split optimization
        split_sequence = find_dynamic_split_points(note_groups)
        if not split_sequence:
            print("❌ FATAL: Could not find valid split sequence.")
            sys.exit(1)
        # Use the most common split as the "primary" for reporting
        from collections import Counter
        split_counts = Counter(split_sequence)
        split_point = split_counts.most_common(1)[0][0]
        unique_splits = len(set(split_sequence))
        print(f"  Primary split: {index_to_note_name(split_point)} ({unique_splits} unique splits used)")
    else:
        # Use static split optimization
        split_point = find_optimal_split_point(note_groups)
        split_sequence = None

    if split_point is None:
        print("❌ FATAL: Could not find any valid split point.")
        print("   The song may exceed the hand span or have other constraints.")
        sys.exit(1)

    # Step 4: Run global optimization with conflict resolution
    print("\nSTEP 4: Running global path optimization...")

    if DYNAMIC_SPLIT_ENABLED and split_sequence:
        # Use dynamic split assignment
        l_groups, r_groups, conflict_log = assign_hands_with_dynamic_splits(
            note_groups, split_sequence, resolve_conflicts=True
        )
    else:
        # Use static split assignment
        l_groups, r_groups, conflict_log = assign_hands_to_notes(
            note_groups, split_point, resolve_conflicts=True
        )

    # Report conflict resolutions
    if conflict_log:
        print(f"\n  ⚠️  Resolved {len(conflict_log)} adjacent key conflicts:")
        for msg in conflict_log[:5]:
            print(msg)
        if len(conflict_log) > 5:
            print(f"  ... and {len(conflict_log) - 5} more resolutions.")

    # For dynamic splits, we need to use varying boundaries
    if DYNAMIC_SPLIT_ENABLED and split_sequence:
        print("  Optimizing left hand (with dynamic boundaries)...")
        l_path = optimize_with_dynamic_boundaries(l_groups, "Left", split_sequence, is_left=True)

        print("  Optimizing right hand (with dynamic boundaries)...")
        r_path = optimize_with_dynamic_boundaries(r_groups, "Right", split_sequence, is_left=False)
    else:
        print("  Optimizing left hand...")
        l_path = optimize_with_boundaries(l_groups, "Left", max_boundary=split_point)

        print("  Optimizing right hand...")
        r_path = optimize_with_boundaries(r_groups, "Right", min_boundary=split_point + MIN_HAND_GAP)

    if not l_path or not r_path:
        print("❌ FATAL: Optimization failed during path generation.")
        sys.exit(1)

    print("  ✓ Optimization complete")

    # Step 5: Safety validation (including remaining conflict check)
    print("\nSTEP 5: Validating output for safety...")
    issues = validate_output(l_path, r_path, note_groups, l_groups, r_groups)

    if issues:
        print("\n⚠️  SAFETY WARNINGS:")
        for issue in issues[:10]:
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

    max_shift = max(l_shift, r_shift)
    if max_shift > 0:
        print(f"  ℹ️  Total timeline shift: {max_shift:.2f}s (song now starts at t={max_shift:.2f}s)")

    # Step 7: Save all outputs
    print("\nSTEP 7: Saving output files...")
    save_outputs(l_cmd, r_cmd, l_path, r_path, l_groups, r_groups, split_point, note_groups, args.output, max_shift,
                 conflict_log)

    print("\n" + "=" * 60)
    print("✓ OPTIMIZATION COMPLETE!")
    print("=" * 60)
    print(f"Split Point:     {index_to_note_name(split_point)} (index {split_point})")
    print(f"Output Files:    {args.output}/")
    print("  • left_hand_commands.txt")
    print("  • right_hand_commands.txt")
    print("  • fingering_plan.csv (with technique & direction info)")
    print("  • fingering_summary.csv")
    print("  • timed_steps.csv")
    if conflict_log:
        print("  • conflict_resolutions.txt (adjacent key conflicts resolved)")
    print("=" * 60)
    print("\nServo Command Format:")
    print("  • Normal white key:  '1', '2', '3', '4', '5'")
    print("  • Black key:         '2b-' (splay left), '3b+' (splay right)")
    print("  • Extended splay:    '1s+2' (thumb right 2 keys), '5s-1' (pinky left 1 key)")
    print("")
    print("Direction Guide:")
    print("  • '-' = toward lower keys (left on keyboard)")
    print("  • '+' = toward higher keys (right on keyboard)")
    print("=" * 60)


if __name__ == '__main__':
    main()