from music21 import *
import os
import csv


def midi_to_white_key_index(midi):
    offset = midi - 12
    octave = offset // 12
    note_in_octave = offset % 12

    white_key_map = {0: 0, 2: 1, 4: 2, 5: 3, 7: 4, 9: 5, 11: 6}

    if note_in_octave not in white_key_map:
        return None

    return octave * 7 + white_key_map[note_in_octave]


def parse_musicxml(file):
    score = converter.parse(file)
    note_info = []

    for music_element in score.recurse():
        if isinstance(music_element, note.Note):
            if music_element.pitch.alter != 0:
                print(f"Warning: Skipping black key - {music_element.pitch.nameWithOctave}")
                continue
            info = {
                'type': 'note',
                'pitch': (music_element.pitch.step,
                          music_element.pitch.octave,
                          music_element.pitch.alter,
                          music_element.pitch.midi),
                'duration': music_element.quarterLength,
                'white_key_index': midi_to_white_key_index(music_element.pitch.midi)
            }
            note_info.append(info)

        elif isinstance(music_element, chord.Chord):
            white_notes = [n for n in music_element.notes if n.pitch.alter == 0]
            if len(white_notes) < len(music_element.notes):
                black_keys = [n.pitch.nameWithOctave for n in music_element.notes if n.pitch.alter != 0]
                print(f"Warning: Removing black keys from chord - {black_keys}")
            if white_notes:
                info = {
                    'type': 'chord',
                    'pitches': [(n.pitch.step,
                                 n.pitch.octave,
                                 n.pitch.alter,
                                 n.pitch.midi)
                                for n in white_notes],
                    'duration': music_element.quarterLength,
                    'white_key_indices': [midi_to_white_key_index(n.pitch.midi) for n in white_notes]
                }
                note_info.append(info)

    return note_info


def convert_to_time_steps(note_info):
    time_steps = []
    for n in note_info:
        time_step = []
        if n['type'] == 'chord':
            for i, pitch in enumerate(n['pitches']):
                time_step.append((pitch[3], n['duration'], n['white_key_indices'][i]))
        else:
            time_step.append((n['pitch'][3], n['duration'], n['white_key_index']))
        time_steps.append(time_step)
    return time_steps


def convert_to_timed_steps(note_info):
    timed_steps = []
    current_time = 0.0
    for n in note_info:
        time_step = []
        if n['type'] == 'chord':
            for i, pitch in enumerate(n['pitches']):
                time_step.append((pitch[3], n['duration'], n['white_key_indices'][i]))
        else:
            time_step.append((n['pitch'][3], n['duration'], n['white_key_index']))
        timed_steps.append((current_time, time_step))
        current_time += n['duration']
    return timed_steps


def print_parsed_data(note_info, time_steps, timed_steps):
    print("\n" + "=" * 70)
    print("PARSED NOTE INFO")
    print("=" * 70)
    for i, note in enumerate(note_info):
        print(f"\nNote/Chord {i}:")
        print(f"  Type: {note['type']}")
        if note['type'] == 'note':
            step, octave, alter, midi = note['pitch']
            print(f"  Pitch: {step}{octave} (MIDI {midi})")
            print(f"  White Key Index: {note['white_key_index']}")
        else:
            pitches_str = ', '.join([f"{p[0]}{p[1]}" for p in note['pitches']])
            print(f"  Pitches: {pitches_str}")
            print(f"  MIDI: {[p[3] for p in note['pitches']]}")
            print(f"  White Key Indices: {note['white_key_indices']}")
        print(f"  Duration: {note['duration']} quarter notes")

    print("\n" + "=" * 70)
    print("TIME STEPS (simplified)")
    print("=" * 70)
    for i, step in enumerate(time_steps):
        print(f"Step {i}: {step}")

    print("\n" + "=" * 70)
    print("TIMED STEPS (with absolute timing)")
    print("=" * 70)
    for time, step in timed_steps:
        print(f"Time {time}: {step}")


def save_timed_steps_csv(note_info, timed_steps, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "timed_steps.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_time", "key", "midi", "duration", "white_key_index", "hand"])
        note_counter = 0
        for start_time, step in timed_steps:
            for i, (midi, duration, white_key_index) in enumerate(step):
                note_entry = note_info[note_counter]
                if note_entry['type'] == 'note':
                    key_name = f"{note_entry['pitch'][0]}{note_entry['pitch'][1]}"
                    note_counter += 1
                else:
                    key_name = f"{note_entry['pitches'][i][0]}{note_entry['pitches'][i][1]}"
                    if i == len(note_entry['pitches']) - 1:
                        note_counter += 1
                writer.writerow([start_time, key_name, midi, duration, white_key_index, note_entry.get("hand", "R")])
    print(f"\nSaved absolute timed steps CSV to: {csv_path}")


def main():
    file_path = 'happybirthday.musicxml'
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        print(f"Current working directory: {os.getcwd()}")
        return

    print(f"Parsing file: {file_path}")

    note_info = parse_musicxml(file_path)
    time_steps = convert_to_time_steps(note_info)
    timed_steps = convert_to_timed_steps(note_info)

    print_parsed_data(note_info, time_steps, timed_steps)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total notes/chords: {len(note_info)}")
    print(f"Total time steps: {len(time_steps)}")
    single_notes = sum(1 for n in note_info if n['type'] == 'note')
    chords = sum(1 for n in note_info if n['type'] == 'chord')
    print(f"Single notes: {single_notes}")
    print(f"Chords: {chords}")
    total_duration = sum(n['duration'] for n in note_info)
    print(f"Total duration: {total_duration} quarter notes")

    output_dir = os.path.dirname(os.path.abspath(__file__))
    save_timed_steps_csv(note_info, timed_steps, output_dir)


if __name__ == '__main__':
    main()
