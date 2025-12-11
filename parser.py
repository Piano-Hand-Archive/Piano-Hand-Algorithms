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

    for part in score.parts:
        for music_element in part.flatten().notesAndRests:
            if isinstance(music_element, note.Rest):
                continue

            if isinstance(music_element, note.Note):
                if music_element.pitch.alter != 0:
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
                white_notes = [n for n in music_element.notes if n.pitch.alter == 0]
                if white_notes:
                    info = {
                        'type': 'chord',
                        'pitches': [(n.pitch.step,
                                     n.pitch.octave,
                                     n.pitch.alter,
                                     n.pitch.midi)
                                    for n in white_notes],
                        'duration': music_element.quarterLength,
                        'white_key_indices': [midi_to_white_key_index(n.pitch.midi) for n in white_notes],
                        'offset': music_element.offset
                    }
                    note_info.append(info)

    note_info.sort(key=lambda x: x['offset'])
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
    for n in note_info:
        time_step = []
        if n['type'] == 'chord':
            for i, pitch in enumerate(n['pitches']):
                time_step.append((pitch[3], n['duration'], n['white_key_indices'][i]))
        else:
            time_step.append((n['pitch'][3], n['duration'], n['white_key_index']))
        timed_steps.append((n['offset'], time_step))
    return timed_steps


def save_timed_steps_csv(note_info, timed_steps, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "timed_steps.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_time", "key", "midi", "duration", "white_key_index"])
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
                writer.writerow([start_time, key_name, midi, duration, white_key_index])


def main():
    file_path = 'happybirthday.musicxml'
    if not os.path.exists(file_path):
        return

    note_info = parse_musicxml(file_path)
    time_steps = convert_to_time_steps(note_info)
    timed_steps = convert_to_timed_steps(note_info)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    save_timed_steps_csv(note_info, timed_steps, output_dir)

    single_notes = sum(1 for n in note_info if n['type'] == 'note')
    chords = sum(1 for n in note_info if n['type'] == 'chord')
    total_duration = sum(n['duration'] for n in note_info)

    print(f"Total notes/chords: {len(note_info)}")
    print(f"Single notes: {single_notes}")
    print(f"Chords: {chords}")
    print(f"Total duration: {total_duration} quarter notes")


if __name__ == '__main__':
    main()