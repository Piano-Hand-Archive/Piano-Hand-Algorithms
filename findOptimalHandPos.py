import csv
import os

MOVE_PENALTY = 4

def load_notes_grouped_by_time(filename="timed_steps.csv"):
    notes_by_time = []
    current_time_step = None

    with open(filename, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            try:
                start_time = float(row['start_time'])
                white_key_index = int(row['white_key_index'])
                key = row['key']

                if current_time_step is None or abs(start_time - current_time_step['time']) > 0.001:
                    current_time_step = {
                        'time': start_time,
                        'notes': [white_key_index],
                        'keys': [key]
                    }
                    notes_by_time.append(current_time_step)
                else:
                    current_time_step['notes'].append(white_key_index)
                    current_time_step['keys'].append(key)

            except (ValueError, KeyError):
                pass

    return notes_by_time


def calculate_transition_cost(prev_thumb_pos, curr_thumb_pos):
    dist = abs(curr_thumb_pos - prev_thumb_pos)
    if dist == 0:
        return 0
    return dist + MOVE_PENALTY


def calculate_path_total_cost(path):
    if not path:
        return float('inf')

    total_cost = 0
    active_positions = [p for p in path if p is not None]

    for i in range(1, len(active_positions)):
        total_cost += calculate_transition_cost(active_positions[i - 1], active_positions[i])

    return total_cost


def find_optimal_split_point(note_groups, min_gap=6):
    all_notes = []
    for group in note_groups:
        all_notes.extend(group['notes'])

    if not all_notes:
        return None

    min_note = min(all_notes)
    max_note = max(all_notes)

    valid_splits = []

    for split in range(min_note, max_note + 1):
        can_complete = True
        total_collisions = 0

        for group in note_groups:
            left = [n for n in group['notes'] if n <= split]
            right = [n for n in group['notes'] if n >= split + min_gap]

            if left and (max(left) - min(left) > 4):
                can_complete = False
                break
            if right and (max(right) - min(right) > 4):
                can_complete = False
                break

            if left and right:
                if max(left) + min_gap > min(right):
                    total_collisions += 1

        if can_complete:
            valid_splits.append({
                'split': split,
                'collisions': total_collisions
            })

    if not valid_splits:
        return None

    min_collisions = min(s['collisions'] for s in valid_splits)
    candidates = [s for s in valid_splits if s['collisions'] == min_collisions]

    best_split_index = None
    min_total_movement_cost = float('inf')

    for candidate in candidates:
        split = candidate['split']

        l_groups, r_groups, _ = assign_hands_to_notes(note_groups, split, min_gap)

        l_bound = split
        r_bound = split + min_gap

        l_path = optimize_with_boundaries(l_groups, "Left", max_boundary=l_bound, allow_flexibility=False)
        r_path = optimize_with_boundaries(r_groups, "Right", min_boundary=r_bound, allow_flexibility=False)

        if not l_path or not r_path:
            continue

        cost = calculate_path_total_cost(l_path) + calculate_path_total_cost(r_path)

        if cost < min_total_movement_cost:
            min_total_movement_cost = cost
            best_split_index = split

    if best_split_index is not None:
        return best_split_index
    else:
        return candidates[0]['split']


def assign_hands_to_notes(note_groups, split_point, min_gap=6):
    left_groups = []
    right_groups = []
    skipped_notes = []

    for group in note_groups:
        left_notes = []
        right_notes = []
        left_keys = []
        right_keys = []

        for note, key in zip(group['notes'], group['keys']):
            assigned = False

            if note <= split_point:
                left_notes.append(note)
                left_keys.append(key)
                assigned = True
            elif note >= split_point + min_gap:
                right_notes.append(note)
                right_keys.append(key)
                assigned = True
            else:
                if note <= split_point + 4:
                    left_notes.append(note)
                    left_keys.append(key)
                    assigned = True
                elif note >= split_point + min_gap - 4:
                    right_notes.append(note)
                    right_keys.append(key)
                    assigned = True

            if not assigned:
                skipped_notes.append((group['time'], key))

        left_groups.append({
                               'time': group['time'],
                               'notes': left_notes,
                               'keys': left_keys
                           } if left_notes else None)

        right_groups.append({
                                'time': group['time'],
                                'notes': right_notes,
                                'keys': right_keys
                            } if right_notes else None)

    return left_groups, right_groups, split_point


def get_possible_states(note_group, max_position=None, min_position=None):
    if note_group is None:
        return []

    notes = note_group['notes']
    min_note = min(notes)
    max_note = max(notes)

    span = max_note - min_note
    if span > 4:
        return []

    start_range = max(0, max_note - 4)
    end_range = min_note

    if max_position is not None:
        end_range = min(end_range, max_position)
    if min_position is not None:
        start_range = max(start_range, min_position)

    possible = list(range(start_range, end_range + 1))

    return possible if possible else []


def optimize_with_boundaries(note_groups, hand_name="Right", max_boundary=None, min_boundary=None,
                             allow_flexibility=True):
    valid_groups = []
    original_indices = []

    for i, group in enumerate(note_groups):
        if group is not None:
            valid_groups.append(group)
            original_indices.append(i)

    if not valid_groups:
        return [None] * len(note_groups)

    n = len(valid_groups)
    dp = [{} for _ in range(n)]
    backpointer = [{} for _ in range(n)]

    first_states = get_possible_states(valid_groups[0], max_boundary, min_boundary)

    if not first_states and allow_flexibility:
        first_states = get_possible_states(valid_groups[0], None, None)

    if not first_states:
        return []

    for state in first_states:
        dp[0][state] = 0

    for i in range(1, n):
        curr_group = valid_groups[i]
        possible_states = get_possible_states(curr_group, max_boundary, min_boundary)

        if not possible_states and allow_flexibility:
            possible_states = get_possible_states(curr_group, None, None)

        if not possible_states:
            return []

        for curr_state in possible_states:
            min_cost = float('inf')
            best_prev_state = None

            for prev_state, prev_cost in dp[i - 1].items():
                transition = calculate_transition_cost(prev_state, curr_state)

                boundary_penalty = 0
                if max_boundary is not None and curr_state > max_boundary:
                    boundary_penalty = 10
                if min_boundary is not None and curr_state < min_boundary:
                    boundary_penalty = 10

                total_cost = prev_cost + transition + boundary_penalty

                if total_cost < min_cost:
                    min_cost = total_cost
                    best_prev_state = prev_state

            if best_prev_state is not None:
                dp[i][curr_state] = min_cost
                backpointer[i][curr_state] = best_prev_state

    last_costs = dp[-1]
    if not last_costs:
        return []

    current_state = min(last_costs, key=last_costs.get)
    optimal_path = [current_state]

    for i in range(n - 1, 0, -1):
        prev_state = backpointer[i][current_state]
        optimal_path.insert(0, prev_state)
        current_state = prev_state

    full_path = [None] * len(note_groups)
    for i, orig_idx in enumerate(original_indices):
        full_path[orig_idx] = optimal_path[i]

    return full_path


def global_optimization(note_groups, split_point, min_gap=6):
    left_groups, right_groups, final_split = assign_hands_to_notes(note_groups, split_point, min_gap)

    if left_groups is None:
        return None, None, None, None, None

    left_max_boundary = split_point
    right_min_boundary = split_point + min_gap

    left_path = optimize_with_boundaries(left_groups, "Left", max_boundary=left_max_boundary)
    right_path = optimize_with_boundaries(right_groups, "Right", min_boundary=right_min_boundary)

    if not left_path and not right_path:
        return None, None, None, None, None

    return left_path, right_path, final_split, left_groups, right_groups


def index_to_note_name(white_key_index):
    octave = white_key_index // 7
    position_in_octave = white_key_index % 7
    note_names = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    return f"{note_names[position_in_octave]}{octave + 1}"


def generate_servo_commands(hand_path, hand_groups, hand_name, start_position):
    commands = []

    first_playing_idx = None
    for i in range(len(hand_groups)):
        if hand_groups[i] is not None and hand_path[i] is not None:
            first_playing_idx = i
            break

    if first_playing_idx is None:
        return commands

    start_note = start_position
    first_thumb_note = index_to_note_name(hand_path[first_playing_idx])
    first_time = hand_groups[first_playing_idx]['time']

    commands.append(f"{first_time}:step:{start_note}-{first_thumb_note}")

    prev_thumb_pos = hand_path[first_playing_idx]

    for i in range(len(hand_groups)):
        if hand_groups[i] is None or hand_path[i] is None:
            continue

        curr_thumb_pos = hand_path[i]
        curr_time = hand_groups[i]['time']

        prev_note = index_to_note_name(prev_thumb_pos)
        curr_note = index_to_note_name(curr_thumb_pos)
        commands.append(f"{curr_time}:step:{prev_note}-{curr_note}")

        fingers = []
        for note_idx in hand_groups[i]['notes']:
            finger = note_idx - curr_thumb_pos + 1
            fingers.append(str(finger))

        servo_line = f"{curr_time}:servo:{','.join(fingers)}"
        commands.append(servo_line)

        prev_thumb_pos = curr_thumb_pos

    return commands


def save_servo_files(left_commands, right_commands, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "left_hand_commands.txt"), 'w') as f:
        f.write('\n'.join(left_commands))
    with open(os.path.join(output_dir, "right_hand_commands.txt"), 'w') as f:
        f.write('\n'.join(right_commands))


def save_summary_csv(left_path, right_path, left_groups, right_groups, split_point, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)

    left_positions = [p for p in left_path if p is not None]
    right_positions = [p for p in right_path if p is not None]

    left_moves = sum(1 for i in range(1, len(left_path))
                     if left_path[i] is not None and left_path[i - 1] is not None
                     and left_path[i] != left_path[i - 1])
    right_moves = sum(1 for i in range(1, len(right_path))
                      if right_path[i] is not None and right_path[i - 1] is not None
                      and right_path[i] != right_path[i - 1])

    left_max = max(left_positions) if left_positions else 0
    right_min = min(right_positions) if right_positions else 0

    with open(os.path.join(output_dir, "fingering_summary.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Left Hand', 'Right Hand', 'Combined'])
        writer.writerow(['Split Point', index_to_note_name(split_point), split_point, ''])
        writer.writerow(['Position Changes', left_moves, right_moves, left_moves + right_moves])
        writer.writerow(['Max Position', index_to_note_name(left_max), index_to_note_name(right_min),
                         f"Gap: {right_min - left_max}"])


def save_fingering_plan_csv(note_groups, left_path, right_path, left_groups, right_groups, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "fingering_plan.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['Time', 'Left_Notes', 'Left_Thumb', 'Left_Fingers', 'Right_Notes', 'Right_Thumb', 'Right_Fingers'])

        for i in range(len(note_groups)):
            time = note_groups[i]['time']

            l_notes, l_thumb, l_fingers = "", "", ""
            if left_groups[i] and left_path[i] is not None:
                l_notes = ';'.join(left_groups[i]['keys'])
                l_thumb = str(left_path[i])
                l_fingers = ';'.join(str(n - left_path[i] + 1) for n in left_groups[i]['notes'])

            r_notes, r_thumb, r_fingers = "", "", ""
            if right_groups[i] and right_path[i] is not None:
                r_notes = ';'.join(right_groups[i]['keys'])
                r_thumb = str(right_path[i])
                r_fingers = ';'.join(str(n - right_path[i] + 1) for n in right_groups[i]['notes'])

            writer.writerow([time, l_notes, l_thumb, l_fingers, r_notes, r_thumb, r_fingers])


def main():
    note_groups = load_notes_grouped_by_time("timed_steps.csv")
    if not note_groups:
        return

    optimal_split = find_optimal_split_point(note_groups, min_gap=6)

    if optimal_split is None:
        return

    result = global_optimization(note_groups, optimal_split, min_gap=6)

    if result[0] is None:
        return

    left_path, right_path, split_point, left_groups, right_groups = result

    left_commands = generate_servo_commands(left_path, left_groups, "Left", "G1")
    right_commands = generate_servo_commands(right_path, right_groups, "Right", "F7")

    save_servo_files(left_commands, right_commands)
    save_summary_csv(left_path, right_path, left_groups, right_groups, split_point)
    save_fingering_plan_csv(note_groups, left_path, right_path, left_groups, right_groups)


if __name__ == '__main__':
    main()