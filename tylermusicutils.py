import pygame
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from piano_constants import *

def midi_to_white_key_index(midi: int):
	"""Convert MIDI value to white key index (0-based). Returns None if it's a black key."""
	# MIDI note names: C=0, C#=1, D=2, D#=3, E=4, F=5, F#=6, G=7, G#=8, A=9, A#=10, B=11
	offset = midi - 12  # C1 (MIDI 24) should map to index 0
	octave = offset // 12
	note_in_octave = offset % 12
	
	# White keys map: C=0, D=2, E=4, F=5, G=7, A=9, B=11
	white_key_map = {0: 0, 2: 1, 4: 2, 5: 3, 7: 4, 9: 5, 11: 6}
	
	if note_in_octave not in white_key_map:
		return None
	
	return octave * 7 + white_key_map[note_in_octave]

def midi_to_black_key_position(midi: int, white_key_width: float, white_key_start_x: float):
	"""Convert MIDI value to black key position. Returns (x, midi) if it's a black key, None otherwise."""
	# MIDI note names: C=0, C#=1, D=2, D#=3, E=4, F=5, F#=6, G=7, G#=8, A=9, A#=10, B=11
	note_in_octave = midi % 12
	
	# Black keys are: C# (1), D# (3), F# (6), G# (8), A# (10)
	if note_in_octave not in [1, 3, 6, 8, 10]:
		return None
	
	# Use the same mapping as parser.py: offset = midi - 12
	offset = midi - 12
	octave = offset // 12
	note_in_octave_offset = offset % 12
	
	# Find the white key index of the key to the left of this black key
	# C# (1) is after C (0), D# (3) is after D (2), F# (6) is after F (5), etc.
	left_white_note_map = {1: 0, 3: 2, 6: 5, 8: 7, 10: 9}  # Map black key note to left white key note
	left_white_note = left_white_note_map[note_in_octave_offset]
	
	# Map the left white note to its position in the octave
	white_key_map = {0: 0, 2: 1, 4: 2, 5: 3, 7: 4, 9: 5, 11: 6}
	white_pos_in_octave = white_key_map[left_white_note]
	
	# Calculate white key index (same as parser.py)
	white_key_index = octave * 7 + white_pos_in_octave
	
	# Position black key between two white keys
	# Black keys are typically positioned at about 60% across the left white key
	# This creates a natural piano keyboard appearance
	x = white_key_start_x + white_key_index * white_key_width + white_key_width * 0.3
	
	return (x, midi)

def build_white_keys(rect: pygame.Rect, n: int):
	keys = []
	key_w = rect.width // n
	for i in range(n):
		label = str(i)
		r = pygame.Rect(rect.x + i * key_w, rect.y, key_w, rect.height)
		keys.append((label, r))
	return keys


def build_black_keys(white_keys_list, white_key_width: float, white_key_start_x: float, kb_rect: pygame.Rect):
	"""Build black keys positioned correctly between white keys."""
	black_keys = []
	black_key_width = white_key_width * 0.6
	black_key_height = kb_rect.height * 0.6
	
	# Generate black keys for a wide range to cover all 52 white keys
	# White key 0 = MIDI 12 (C0), white key 51 ≈ MIDI 98 (B6)
	# Generate black keys from C#0 (MIDI 13) to A#7 (MIDI 106) to ensure full coverage
	start_midi = 13  # C#0 (first black key)
	end_midi = 106   # A#7 (covers full 52-key range)
	
	for midi in range(start_midi, end_midi + 1):
		pos = midi_to_black_key_position(midi, white_key_width, white_key_start_x)
		if pos is not None:
			x, midi_val = pos
			# Only include black keys that are within the visible keyboard bounds
			# Check if the black key's center is within the keyboard rectangle
			if kb_rect.x - black_key_width/2 <= x <= kb_rect.right + black_key_width/2:
				r = pygame.Rect(
					x - black_key_width / 2,
					kb_rect.y,
					black_key_width,
					black_key_height
				)
				black_keys.append((midi_val, r))
	
	return black_keys

def create_legend_image(filename="legend.png"):
    highlight_patch = mpatches.Patch(
        facecolor=(THUMB_HIGHLIGHT_COLOR[0] / 255, THUMB_HIGHLIGHT_COLOR[1] / 255, THUMB_HIGHLIGHT_COLOR[2] / 255), edgecolor='black', label='Key Played by Thumb'
    )
    right_patch = mpatches.Patch(
        facecolor=(RIGHT_HAND_COLOR[0] / 255, RIGHT_HAND_COLOR[1] / 255, RIGHT_HAND_COLOR[2] / 255),
        edgecolor='black',
        label='Right Hand Position',
    )
    left_patch = mpatches.Patch(
        facecolor=(LEFT_HAND_COLOR[0] / 255, LEFT_HAND_COLOR[1] / 255, LEFT_HAND_COLOR[2] / 255),
        edgecolor='black',
        label='Left Hand Position',
    )

    fig, ax = plt.subplots(figsize=(3, 2))

    # Place the legend to the right
    ax.legend(
        handles=[highlight_patch, right_patch, left_patch],
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        frameon=True,
        title="Visualizer Legend"
    )

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', transparent=True)
    plt.close()