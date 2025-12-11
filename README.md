Yes, you can absolutely copy and paste the content below directly into a file named `README.md`. It is already formatted with valid Markdown syntax.

Here is the block again for easy copying:

````markdown
# Robotic Piano Fingering Optimizer

This Python utility generates optimal fingering plans and servo commands for a pair of robotic piano hands. It uses a simulation-based approach with the Viterbi algorithm to guarantee **completeness** (all notes played), **physical feasibility** (5-key span limit), and **minimal hand movement** (stability).

## Key Features

* **Global Movement Minimization:** Uses a cost-based Viterbi algorithm to find the absolute most efficient path for thumb positions, minimizing total hand shifts across the entire song.
* **Hysteresis (Stability) Control:** Implements a `MOVE_PENALTY` that prioritizes keeping the hand "planted" in one position over making small, unnecessary adjustments. This eliminates "robotic drift."
* **Simulation-Based Split Point:** Instead of guessing, the algorithm simulates playing the entire song for *every possible* split point and selects the one that results in the lowest global movement cost.
* **Strict Collision Prevention:** Defines a hard "No Fly Zone" (Gap) between hands based on the optimal split point, mathematically guaranteeing that hands never cross over or collide.
* **Physical Constraints:** Enforces a strict 5-key maximum span (Thumb to Pinky) for all chords and note reaches.

## Prerequisites

* Python 3.6+
* No external dependencies (uses standard libraries: `csv`, `os`)

## Input Data

The script requires a CSV file named `timed_steps.csv` in the same directory.
**Format:**
```csv
start_time,white_key_index,key
0.0,21,C4
0.0,25,E4
1.0,28,G4
````

  * `start_time`: Time in seconds.
  * `white_key_index`: Integer index of the white key (0 = A0 or C1 depending on your mapping, typically C1=0 for simplicity in this context).
  * `key`: Note name for reference (e.g., "C\#4").

## Usage

Run the script directly from the terminal:

```bash
python findOptimalHandPos.py
```

## Output Files

The script generates four files in the current directory:

1.  **`left_hand_commands.txt`**: Servo commands for the Left Hand.
2.  **`right_hand_commands.txt`**: Servo commands for the Right Hand.
3.  **`fingering_plan.csv`**: A human-readable schedule showing exactly which notes, thumb positions, and fingers are active at every timestamp.
4.  **`fingering_summary.csv`**: Statistics on the optimization, including the selected split point, total moves, and hand efficiency metrics.

## Command Format

The output text files use a specific format for the robot controller:

```text
<time>:step:<prev_note>-<curr_note>
<time>:servo:<finger_indices>
```

  * **`step`**: Moves the **Thumb** (the hand's anchor) from the previous note to the new note.
  * **`servo`**: Activates specific fingers relative to the thumb position (1=Thumb, 5=Pinky).

**Example:**

```text
1.0:step:G3-G3    (Hand stays at G3)
1.0:servo:1,3,5   (Fingers 1, 3, and 5 press down -> Playing G3, B3, D4)
```

## Configuration

You can tune the behavior by modifying the constant at the top of the script:

```python
MOVE_PENALTY = 4
```

  * **Higher Value (e.g., 4-10):** Makes the robot "lazy." It will stretch its fingers to reach notes rather than moving its hand. Results in very stable, blocky movement.
  * **Lower Value (e.g., 0-1):** Makes the robot "fluid." It will move its hand frequently to center itself on notes. Results in constant small adjustments.

## Algorithm Details

1.  **Load & Group:** Reads `timed_steps.csv` and groups simultaneous notes into "Time Steps" (chords).
2.  **Split Point Simulation:**
      * Iterates through every possible key index as a candidate "Split Point."
      * For each candidate, checks if the song is playable without exceeding the 5-key span or crossing hands.
      * If playable, it runs the full Viterbi optimization to calculate the total "Movement Cost."
      * Selects the split point with the lowest cost.
3.  **Global Optimization (Viterbi):**
      * Builds a cost graph (Trellis) where nodes are valid thumb positions.
      * Calculates the cost of moving between positions (Distance + `MOVE_PENALTY`).
      * Finds the lowest-cost path through the entire song graph.
4.  **Command Generation:** Converts the optimized path of Thumb Indices into specific `step` and `servo` commands, inserting an initial move from the "Parked" positions (**Left: G1**, **Right: F7**).

<!-- end list -->

```
```