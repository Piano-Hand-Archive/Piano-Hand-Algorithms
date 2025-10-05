# --- 1. Data Loading and Setup ---
import csv

def load_notes_from_csv(filename="timed_steps.csv"):
    notes = []
    with open(filename, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            try:
                row['start_time'] = float(row['start_time'])
                row['midi'] = int(row['midi'])
                row['duration'] = float(row['duration'])
                row['white_key_index'] = int(row['white_key_index'])
                notes.append(row)
            except (ValueError, KeyError) as e:
                print(f"Skipping row due to error: {row}. Error: {e}")
    return notes

# --- 2. Helper Functions ---
def get_possible_states(note):
    wki = note['white_key_index']
    return [wki - (f - 1) for f in range(1, 6)]

def calculate_transition_cost(prev_thumb_pos, curr_thumb_pos):
    return abs(curr_thumb_pos - prev_thumb_pos)

# --- 3. Main Algorithm Implementation ---
def find_optimal_fingering(notes):
    if not notes:
        return []

    dp = [{} for _ in notes]           # dp[i][state] = min cost
    backpointer = [{} for _ in notes]  # backpointer[i][state] = prev_state

    # Step 1: Initialize first note
    for state in get_possible_states(notes[0]):
        dp[0][state] = 0
    print(f"Note 0 ({notes[0]['key']}): possible thumb positions = {get_possible_states(notes[0])}")
    print(f"Cost table: {dict(sorted(dp[0].items()))}\n")

    # Step 2: Process remaining notes
    for i in range(1, len(notes)):
        current_note = notes[i]
        print(f"--- Processing Note {i} ({current_note['key']}) ---")
        for curr_state in get_possible_states(current_note):
            min_cost = float('inf')
            best_prev_state = None
            for prev_state, prev_cost in dp[i-1].items():
                cost = prev_cost + calculate_transition_cost(prev_state, curr_state)
                if cost < min_cost:
                    min_cost = cost
                    best_prev_state = prev_state
            if best_prev_state is not None:
                dp[i][curr_state] = min_cost
                backpointer[i][curr_state] = best_prev_state
        # Print progression
        sorted_dp = dict(sorted(dp[i].items()))
        sorted_bp = dict(sorted(backpointer[i].items()))
        print(f"Thumb positions (dp[{i}]): {sorted_dp}")
        print(f"Backpointers: {sorted_bp}\n")

    # Step 3: Backtrack to find optimal path
    optimal_path = []
    last_note_costs = dp[-1]
    if not last_note_costs:
        return "No solution found"

    current_state = min(last_note_costs, key=last_note_costs.get)
    optimal_path.append(current_state)

    for i in range(len(notes)-1, 0, -1):
        prev_state = backpointer[i][current_state]
        optimal_path.insert(0, prev_state)
        current_state = prev_state

    return optimal_path

# --- 4. Execution and Output Formatting ---
def main():
    notes = load_notes_from_csv("timed_steps.csv")
    if not notes:
        print("No notes were loaded.")
        return

    optimal_thumb_positions = find_optimal_fingering(notes)

    print("\n\n--- OPTIMAL FINGERING PLAN ---")
    print(f"{'Time':<10} {'Key':<5} {'Thumb Pos':<12} {'Finger'}")
    print("-" * 45)

    for i, note in enumerate(notes):
        thumb_pos = optimal_thumb_positions[i]
        finger = note['white_key_index'] - thumb_pos + 1
        print(f"{note['start_time']:<10.1f} {note['key']:<5} {thumb_pos:<12} {finger}")

    # Count actual state changes
    total_state_changes = 0
    for i in range(1, len(optimal_thumb_positions)):
        if optimal_thumb_positions[i] != optimal_thumb_positions[i-1]:
            total_state_changes += 1

    print(f"\nTotal thumb state changes: {total_state_changes}")

if __name__ == '__main__':
    main()
