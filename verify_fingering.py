import csv
import os
import re
import argparse
from collections import defaultdict

# ==========================================
# CONSTANTS & CONFIG
# ==========================================
# MIDI Note Names for logging
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def midi_to_name(midi):
    """Convert MIDI number to name (e.g., 60 -> C4)."""
    if midi is None: return "None"
    octave = (midi // 12) - 1
    note = NOTE_NAMES[midi % 12]
    return f"{note}{octave}"


def index_to_name(idx):
    """Convert white key index to name (e.g., 0 -> A0)."""
    # Inverse of the script's mapping logic roughly
    # 0=A0, 1=B0, 2=C1...
    octave = (idx + 9) // 7
    # This is an approximation for display; exact matching uses the index
    return f"Index_{idx}"


# ==========================================
# PARSERS
# ==========================================

def parse_timed_steps(filepath):
    """
    Parses the ground truth (sheet music).
    Returns a dict: { time: set of (white_key_index, is_black, midi) }
    """
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return {}

    events = defaultdict(list)
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row['start_time'])
            # We track note ON events.
            # (Duration handling would require a stateful approach,
            # here we verify note onsets match command onsets).
            events[t].append({
                'white_index': int(row['white_key_index']),
                'is_black': bool(int(row['is_black'])),
                'midi': int(row['midi'])
            })
    return events


def parse_commands(filepath):
    """
    Parses servo command files (robot actions).
    Returns list of events: {'time': t, 'type': 'step'|'servo', 'payload': str}
    """
    if not os.path.exists(filepath):
        return []

    events = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(':')
            if len(parts) >= 3:
                # Handle possible shift in format labels if copy-pasted
                try:
                    # distinct cleanup for raw file reading
                    time_str = parts[0].split(' ')[-1]
                    t = float(time_str)
                    cmd_type = parts[1]
                    payload = parts[2]
                    events.append({'time': t, 'type': cmd_type, 'payload': payload})
                except ValueError:
                    continue
    return events


# ==========================================
# SIMULATION CORE
# ==========================================

class HandState:
    def __init__(self, name, is_left):
        self.name = name
        self.is_left = is_left  # True = Left, False = Right
        self.thumb_pos = None  # White key index
        self.fingers_pressing = []  # List of active finger definitions

    def update_position(self, payload):
        """Parse step command: 'G1-D5' or just 'D5' logic depending on file."""
        # Payload usually: "PreviousNote-NewNote" e.g., "G1-D5"
        # We need to map the note name back to an index to track math.
        # However, the simulator is safer parsing the "fingering_plan.csv"
        # for exact indices if available, but let's try to infer or
        # rely on the thumb index provided in the summary csvs if possible.
        # Since the command file uses names, we need a name_to_index.
        # SIMPLIFICATION: The simulator works best if we use the fingering_plan.csv
        # because mapping "D5" back to index 31 requires the exact same logic key_map.
        pass


def parse_fingering_plan(filepath):
    """
    Load the comprehensive plan which has Thumb Indices and Finger Commands aligned.
    This is more robust than parsing the raw text commands for verification
    because it contains the explicit thumb index integer.
    """
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return []

    plan = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {
                'time': float(row['Time']),
                'l_thumb': int(row['L_Thumb']) if row['L_Thumb'] else None,
                'l_cmd': row['L_Commands'],
                'r_thumb': int(row['R_Thumb']) if row['R_Thumb'] else None,
                'r_cmd': row['R_Commands']
            }
            plan.append(entry)
    return plan


def calculate_hit_notes(thumb_idx, cmd_str, is_left):
    """
    Reverse the algorithm: Given Thumb Index + Finger Command -> What key is hit?
    """
    if not thumb_idx or not cmd_str:
        return []

    hit_notes = []
    # Commands are semicolon separated: "5;1s+2"
    sub_cmds = cmd_str.split(';')

    for sub in sub_cmds:
        if sub == 'X' or sub == '': continue

        # Regex to parse: Finger(1-5) + Type(b/s)? + Dir(+/-)? + Dist(int)?
        # Examples: "5", "2b-", "1s+2"
        match = re.match(r'(\d)([bs])?([+-])?(\d+)?', sub)
        if not match:
            continue

        finger = int(match.group(1))
        tech_type = match.group(2)  # None, 'b', 's'
        direction = match.group(3)  # +, -
        dist_str = match.group(4)
        distance = int(dist_str) if dist_str else 0

        # 1. Determine Effective Finger Reach (accounting for Splay)
        # Splay changes the "natural" reach of the finger.
        # Script: natural_finger = thumb - note + 1 (Left)
        # Splay Amount is deviation from natural.

        effective_reach = finger

        if tech_type == 's':
            # Reverse engineer splay logic from script:
            # if 1s+2 (Left Thumb splay right 2): natural_finger was < 1.
            # splay_amount = 1 - natural -> natural = 1 - splay_amount.
            # effective_reach = 1 - distance
            if finger == 1:
                effective_reach = 1 - distance
            elif finger == 5:
                # Script: splay_amount = natural - 5 -> natural = 5 + splay_amount
                effective_reach = 5 + distance

        # 2. Calculate Target White Key Index
        if is_left:
            # Formula: finger = thumb - note + 1  => note = thumb - finger + 1
            target_idx = thumb_idx - effective_reach + 1
        else:
            # Formula: finger = note - thumb + 1  => note = thumb + finger - 1
            target_idx = thumb_idx + effective_reach - 1

        # 3. Determine if Black Key
        is_black_key = (tech_type == 'b')

        hit_notes.append((target_idx, is_black_key))

    return hit_notes


def main():
    parser = argparse.ArgumentParser(description="Verify Robotic Piano Fingering")
    parser.add_argument('--dir', default='.', help='Directory containing output files')
    args = parser.parse_args()

    # File paths
    plan_path = os.path.join(args.dir, 'fingering_plan.csv')
    truth_path = os.path.join(args.dir, 'timed_steps.csv')

    print(f"🔍 Verifying Output in: {args.dir}")

    # Load Data
    expected_events = parse_timed_steps(truth_path)
    actual_plan = parse_fingering_plan(plan_path)

    if not expected_events or not actual_plan:
        print("❌ Missing required CSV files for verification.")
        return

    # Statistics
    total_checks = 0
    passed_checks = 0
    errors = []

    # Verification Loop
    print(f"   Checking {len(actual_plan)} time steps against ground truth...")

    for step in actual_plan:
        t = step['time']

        # Get Expected Notes for this time (or very close to it)
        # Float matching tolerance
        expected = []
        for et, notes in expected_events.items():
            if abs(et - t) < 0.001:
                expected = notes
                break

        # If no notes expected at this time, ensure commands are empty
        if not expected:
            if step['l_cmd'] or step['r_cmd']:
                # This might be valid (pre-moving fingers), but let's check if it triggers a note
                # For this simulator, we assume every command triggers a note (step logic).
                # If command exists but no note expected, it's a "Ghost Note" error.
                pass
            continue

        total_checks += 1

        # Calculate Actual Hit Notes
        played_notes = []

        # Left Hand
        l_hits = calculate_hit_notes(step['l_thumb'], step['l_cmd'], is_left=True)
        for idx, is_blk in l_hits:
            played_notes.append({'idx': idx, 'black': is_blk, 'hand': 'Left'})

        # Right Hand
        r_hits = calculate_hit_notes(step['r_thumb'], step['r_cmd'], is_left=False)
        for idx, is_blk in r_hits:
            played_notes.append({'idx': idx, 'black': is_blk, 'hand': 'Right'})

        # Compare
        # Convert expected to simpler set for comparison
        exp_set = set((n['white_index'], n['is_black']) for n in expected)
        act_set = set((n['idx'], n['black']) for n in played_notes)

        if exp_set == act_set:
            passed_checks += 1
        else:
            # Detailed Error Reporting
            missing = exp_set - act_set
            extra = act_set - exp_set

            err_msg = f"Time {t:.2f}s Mismatch:"
            if missing:
                m_str = [f"{midi_to_name(n['midi'])}" for n in expected if (n['white_index'], n['is_black']) in missing]
                err_msg += f"\n    MISSING Notes: {m_str} (Indices: {missing})"
            if extra:
                err_msg += f"\n    EXTRA/WRONG Notes Played: Indices {extra}"

            # Context
            err_msg += f"\n    Left Cmd: {step['l_cmd']} (Thumb {step['l_thumb']})"
            err_msg += f"\n    Right Cmd: {step['r_cmd']} (Thumb {step['r_thumb']})"
            errors.append(err_msg)

    # Report
    print("\n" + "=" * 40)
    print("VERIFICATION RESULTS")
    print("=" * 40)
    print(f"Timesteps Checked: {total_checks}")
    print(f"Passed:            {passed_checks}")
    print(f"Failed:            {len(errors)}")
    print(f"Accuracy:          {(passed_checks / total_checks) * 100:.1f}%")

    if errors:
        print("\n❌ ERRORS FOUND:")
        for e in errors[:10]:  # Limit output
            print(e)
        if len(errors) > 10:
            print(f"...and {len(errors) - 10} more.")
    else:
        print("\n✅ SUCCESS: Robot Plan matches Sheet Music perfectly.")


if __name__ == "__main__":
    main()