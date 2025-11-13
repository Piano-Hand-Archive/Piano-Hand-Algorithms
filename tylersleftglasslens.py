import sys
from collections import defaultdict
from typing import List

import pandas as pd
import pygame
from tylershand import Hand
from piano_constants import *
from piano_sound import PianoSound, pre_init_audio
from tylermusicutils import build_white_keys, build_black_keys, create_legend_image

LETTER_TO_SEMITONE = {
	'C': 0,
	'D': 2,
	'E': 4,
	'F': 5,
	'G': 7,
	'A': 9,
	'B': 11,
}


#dictionary of key pair:distance
white_keys = ["C", "D", "E", "F", "G", "A", "B"]
octaves = range(1, 8)

# Build full list of white keys across octaves
all_white_keys = [f"{note}{octave}" for octave in octaves for note in white_keys]

# Build dictionary of unique distances (unordered pairs)
distances = {}
for i, key1 in enumerate(all_white_keys):
    for j, key2 in enumerate(all_white_keys):
        if j > i:  # only keep unique pairs
            distances[(key1, key2)] = abs(i - j)

# Print dictionary
for k, v in distances.items():
    print(f"{k}: {v}")

def note_color_for_hands(hand_ids, default_color):
	if not hand_ids:
		return default_color
	if len(hand_ids) == 1:
		hand_id = next(iter(hand_ids))
		return HAND_NOTE_COLORS.get(hand_id, UNKNOWN_HAND_NOTE_COLOR)
	return MIXED_NOTE_COLOR


def blit_thumb_overlay(surface, rect):
	overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
	overlay.fill((THUMB_HIGHLIGHT_COLOR[0], THUMB_HIGHLIGHT_COLOR[1], THUMB_HIGHLIGHT_COLOR[2], THUMB_OVERLAY_ALPHA))
	surface.blit(overlay, rect)

def main():
	create_legend_image("legend.png")  # generate legend image

	# Initialize audio mixer before pygame.init for low latency
	pre_init_audio()
	pygame.init()
	LEGEND_WIDTH = 150
	screen = pygame.display.set_mode((WIDTH + LEGEND_WIDTH, HEIGHT))
	clock = pygame.time.Clock()
	legend_img = pygame.image.load("legend.png")
	legend_img = pygame.transform.scale(legend_img, (250, 175))  # width, height
	legend_rect = legend_img.get_rect(topleft=(WIDTH - 110, MARGIN))
	font = pygame.font.SysFont(None, 18)

	# Initialize simple tone synth
	synth = PianoSound()

	kb_height = HEIGHT - 2 * MARGIN - PROGRESS_BAR_HEIGHT - PROGRESS_BAR_MARGIN_TOP
	kb_rect = pygame.Rect(MARGIN, MARGIN, WIDTH - 2 * MARGIN, kb_height)
	white_keys = build_white_keys(kb_rect, NUM_WHITE_KEYS)
	white_key_width = kb_rect.width / NUM_WHITE_KEYS
	black_keys = build_black_keys(white_keys, white_key_width, kb_rect.x, kb_rect)

	bar_width = kb_rect.width
	bar_x = kb_rect.x
	bar_y = kb_rect.bottom + PROGRESS_BAR_MARGIN_TOP
	bg_rect = pygame.Rect(bar_x, bar_y, bar_width, PROGRESS_BAR_HEIGHT)
	fill_rect = pygame.Rect(bar_x, bar_y, 0, PROGRESS_BAR_HEIGHT)

	finger_df = pd.read_csv('fingering_plan_updated.csv')
	total_duration = float(finger_df['start_time'].max()) + 0.5

	hand_width = max((kb_rect.width // NUM_WHITE_KEYS) * 5, 60)
	hand_height = 100
	base_x = kb_rect.x + 10
	right_y = kb_rect.y + 10
	left_y = right_y + hand_height // 2
	hands = []
	for hand_label, color, y in (('L', LEFT_HAND_COLOR, left_y), ('R', RIGHT_HAND_COLOR, right_y)):
		hands.append(Hand(position=(base_x, y), size=(hand_width, hand_height), color=color, alpha=HAND_ALPHA, hand_label=hand_label, finger_data=finger_df))

	elaspedTime = 0
	running = True
	pause = False
	# Track last set of sounding notes (by MIDI value)
	last_chord_midi_values = set()

	while running:
		dt = clock.tick(FPS) / 1000
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
				break
			elif event.type == pygame.KEYDOWN:
				if event.key in (pygame.K_LEFT, pygame.K_a):
					elaspedTime -= 1
				elif event.key in (pygame.K_RIGHT, pygame.K_d):
					elaspedTime += 1
				elif event.key == pygame.K_SPACE:
					pause = not pause
					# Pause/resume all audio
					if pause:
						try:
							pygame.mixer.pause()
						except pygame.error:
							pass
					else:
						try:
							pygame.mixer.unpause()
						except pygame.error:
							pass

		for hand_obj in hands:
			hand_obj.play(elaspedTime)
		# Update time only when not paused
		if not pause:
			elaspedTime += dt
		elaspedTime = min(max(elaspedTime, 0), total_duration)
		screen.fill(BG_COLOR)

		# Determine which keys to highlight for the current event time window across hands
		white_key_highlights = defaultdict(set)
		midi_highlights = defaultdict(set)
		thumb_highlights = defaultdict(set)
		for hand_obj in hands:
			hand_df = hand_obj.fingerData
			if hand_df.empty:
				continue
			current_idx = min(max(hand_obj.index, 0), len(hand_df) - 1)
			hand_id = (hand_obj.hand_label or 'UNK').upper()

			if 'thumb_pos' in hand_df.columns:
				thumb_value = pd.to_numeric(hand_df.iloc[current_idx]['thumb_pos'], errors='coerce')
				if pd.notna(thumb_value):
					thumb_highlights[int(thumb_value)].add(hand_id)

			if not hand_obj.start_times:
				continue

			if 'note_time' in hand_df.columns:
				note_time_series = hand_df['note_time']
			else:
				note_time_series = pd.Series(0.0, index=hand_df.index)
			if 'note_end_time' in hand_df.columns:
				note_end_series = hand_df['note_end_time']
			else:
				note_end_series = hand_df['start_time'] + note_time_series

			active_mask = (hand_df['start_time'] <= elaspedTime) & (note_end_series > elaspedTime)
			if not active_mask.any():
				continue
			active_rows = hand_df.loc[active_mask]
			for _, row in active_rows.iterrows():
				wi = row.get('white_key_index')
				if pd.notna(wi):
					white_key_highlights[int(wi)].add(hand_id)
				midi = row.get('midi')
				if pd.notna(midi):
					midi_highlights[int(midi)].add(hand_id)

		chord_midi_values = set(midi_highlights.keys())

		# Update audio (start/stop notes as needed)
		if not pause and elaspedTime < total_duration and chord_midi_values != last_chord_midi_values:
			synth.set_active_notes_from_midi(chord_midi_values)
			last_chord_midi_values = chord_midi_values.copy()
		if elaspedTime >= total_duration:
			synth.stop_all()
			last_chord_midi_values = set()
		# Draw white keys
		for label, r in white_keys:
			key_index = int(label)
			fill_color = note_color_for_hands(
				white_key_highlights.get(key_index),
				WHITE_KEY_COLOR
			)
			pygame.draw.rect(screen, fill_color, r)
			if key_index in thumb_highlights:
				blit_thumb_overlay(screen, r)
			pygame.draw.rect(screen, WHITE_KEY_OUTLINE, r, width=2)
			text = font.render(label, True, LABEL_COLOR)
			screen.blit(text, (r.centerx - text.get_width()/2, r.bottom - text.get_height()))
		
		# Draw black keys (on top of white keys)
		for midi_val, r in black_keys:
			fill_color = note_color_for_hands(
				midi_highlights.get(midi_val),
				BLACK_KEY_COLOR
			)
			pygame.draw.rect(screen, fill_color, r)
			pygame.draw.rect(screen, BLACK_KEY_OUTLINE, r, width=1)

		# Progress bar calculations
		progress_base = total_duration if total_duration > 0 else 1
		progress = max(0, min(1, elaspedTime / progress_base))
		fill_rect.size = (int(bar_width * progress), PROGRESS_BAR_HEIGHT)
		# Draw background and fill
		pygame.draw.rect(screen, (255, 255, 255), bg_rect)
		if fill_rect.width > 0:
			pygame.draw.rect(screen, (255, 255, 0), fill_rect)
			pygame.draw.rect(screen, (0, 0, 0), fill_rect, width=1)
		# Outline
		pygame.draw.rect(screen, (0, 0, 0), bg_rect, width=1)
		# Optional progress text
		# Show elapsed / total seconds text instead of percentage
		elapsed_text = font.render(f"{elaspedTime:.1f}s / {total_duration:.1f}s", True, (40, 40, 40))
		screen.blit(elapsed_text, (bg_rect.centerx - elapsed_text.get_width()//2, bg_rect.centery - elapsed_text.get_height()//2))

		for hand_obj in hands:
			hand_obj.draw(screen)
		screen.blit(legend_img, legend_rect)
		pygame.display.flip()

	try:
		synth.stop_all()
	except Exception:
		pass
	pygame.quit()
	return 0


if __name__ == "__main__":
	sys.exit(main())

