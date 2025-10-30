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

# --- 5. Computes RH costs ---  

def compute_rh_cost(notes):
    if not notes: # If song is empty, return nothing.
        return [], 0, 0 
    
    notes = sorted(notes, key=lambda n: (n['start_time'], n['white_key_index']))

    def get_possible_states(note): # gives all possiblel thumb positions
        wki = note['white_key_index']
        return [wki - (f - 1) for f in range(1, 6)]

    def cost(prev, curr): # measures thumb movement between positions
        return abs(prev - curr)

    dp = [{} for _ in notes]  # dp tables
    back = [{} for _ in notes]

    for s in get_possible_states(notes[0]): # start with first note
        dp[0][s] = 0

    for i in range(1, len(notes)): # finds best thumb path
        for s in get_possible_states(notes[i]):
            best_cost = float("inf")
            best_prev = None
            for ps, pcost in dp[i - 1].items():
                move = pcost + cost(ps, s)
                if move < best_cost:
                    best_cost = move
                    best_prev = ps
            if best_prev is not None:
                dp[i][s] = best_cost
                back[i][s] = best_prev

    if not dp[-1]: # 
        return [], 0, 0

    end = min(dp[-1], key=dp[-1].get) # Backtracks to get path
    path = [end]
    for i in range(len(notes) - 1, 0, -1):
        end = back[i][end]
        path.insert(0, end)

    jumps = [abs(path[i] - path[i - 1]) for i in range(1, len(path))] # Returns: thumb_path, total_shift, max_jump = compute_rh_cost(notes)
    return path, sum(jumps) if jumps else 0, max(jumps) if jumps else 0

# --- 6. Left-hand Implemention ---  

def try_assign_left_hand(candidate_note, rh_notes, lh_notes):
    # gets RH for every note
    _, baseline_shift, _ = compute_rh_cost(rh_notes)

    # run note with LH 
    filtered_rh = [
        n for n in rh_notes
        if not (
            n['start_time'] == candidate_note['start_time']
            and n['white_key_index'] == candidate_note['white_key_index']
        )
    ]

    # re-run RH cost without that note
    _, after_shift, _ = compute_rh_cost(filtered_rh)

    # assign to LH if shifts lower
    if after_shift < baseline_shift:
        # add note to LH list
        lh_notes.append(candidate_note)
        return True

    # else: keep RH
    return False

def assign_hands(notes):
    rh_notes = notes.copy()
    lh_notes = []
    for note in notes:
        wki = note['white_key_index']

        # Rule 1: below middle C → automatic LH
        if wki <= 23:
            lh_notes.append(note)
            continue

        # Rule 2: if RH movement >= 5 → test LH
        _, _, max_jump = compute_rh_cost(rh_notes)
        if max_jump >= 5:
            try_assign_left_hand(note, rh_notes, lh_notes)

    # assigns final hand positions
    for note in notes:
        if note in lh_notes:
            note['hand'] = 'L'
        else:
            note['hand'] = 'R'

    return notes

# --- 7. Save Assigned Hands to CSV ---

def save_hand_assignments_to_csv(notes, filename="fingering_plan.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_time", "key", "midi", "duration", "white_key_index", "hand"])
        for note in notes:
            writer.writerow([
                note.get("start_time", ""),
                note.get("key", ""),
                note.get("midi", ""),
                note.get("duration", ""),
                note.get("white_key_index", ""),
                note.get("hand", "R")  # Default to R if not assigned
            ])
    print(f"Saved L/R hand assignments to {filename}")


# --- 8. Run Left-Hand Assignment Only ---
def main_hand_assignment():
    notes = load_notes_from_csv("timed_steps.csv")
    if not notes:
        print("No notes were loaded.")
        return

    assigned_notes = assign_hands(notes)
    save_hand_assignments_to_csv(assigned_notes)

if __name__ == "__main__":
    main_hand_assignment()

