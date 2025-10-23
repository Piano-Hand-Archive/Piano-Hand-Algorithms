import sys
import pygame
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tylershand import Hand
from piano_constants import *

# --- Helpers: map note names like "C4" to white key indices (0..NUM_WHITE_KEYS-1) ---
LETTER_TO_SEMITONE = {
	'C': 0,
	'D': 2,
	'E': 4,
	'F': 5,
	'G': 7,
	'A': 9,
	'B': 11,
}

def _note_name_to_midi(note_name: str):
	"""Return MIDI number for note like 'C4'. Returns None if accidental or invalid.

	We intentionally skip accidentals (black keys) because our keyboard only draws white keys.
	"""
	if not note_name or len(note_name) < 2:
		return None
	letter = note_name[0].upper()
	if letter not in LETTER_TO_SEMITONE:
		return None
	# If the second char is accidental, skip (we only support white keys here)
	if len(note_name) >= 3 and note_name[1] in ('#', 'b'):
		return None
	try:
		octave = int(note_name[-1])
	except ValueError:
		return None
	semitone = LETTER_TO_SEMITONE[letter]
	# MIDI mapping: C4 -> 60, so midi = 12 * (octave + 1) + semitone
	return 12 * (octave + 1) + semitone

def _midi_to_white_index(midi: int):
	"""Map MIDI to white key index counting white keys from C0 upward.

	Mirrors parser.midi_to_white_key_index but avoids importing music21.
	Returns None if MIDI is a black key.
	"""
	if midi is None:
		return None
	offset = midi - 12
	octave = offset // 12
	note_in_octave = offset % 12
	white_key_map = {0: 0, 2: 1, 4: 2, 5: 3, 7: 4, 9: 5, 11: 6}
	if note_in_octave not in white_key_map:
		return None
	return octave * 7 + white_key_map[note_in_octave]

def note_name_to_white_index(note_name: str):
	midi = _note_name_to_midi(note_name)
	return _midi_to_white_index(midi) if midi is not None else None
def create_legend_image(filename="legend.png"):
    WHITE_KEY_COLOR = (1, 1, 1)
    HIGHLIGHT_COLOR = (1, 138/255, 138/255)
    HAND_COLOR = (0, 1, 1)

    #white_patch = mpatches.Patch(facecolor=WHITE_KEY_COLOR, edgecolor='black', label='Idle Key')
    highlight_patch = mpatches.Patch(facecolor=HIGHLIGHT_COLOR, edgecolor='black', label='Key Played by Thumb')
    hand_patch = mpatches.Patch(facecolor=HAND_COLOR, edgecolor='black', label='Hand Position')

    fig, ax = plt.subplots(figsize=(3, 2))

    # Place the legend to the right
    ax.legend(
        handles=[highlight_patch, hand_patch],
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        frameon=True,
        title="Visualizer Legend"
    )

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', transparent=True)
    plt.close()

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


PROGRESS_BAR_HEIGHT = 12
PROGRESS_BAR_MARGIN_TOP = 4

BG_COLOR = (255, 255, 255)
WHITE_KEY_COLOR = (255, 255, 255)
WHITE_KEY_OUTLINE = (0, 0, 0)
LABEL_COLOR = (0, 0, 0)

kb_height = HEIGHT - 2 * MARGIN - PROGRESS_BAR_HEIGHT - PROGRESS_BAR_MARGIN_TOP
kb_rect = pygame.Rect(MARGIN, MARGIN, WIDTH - 2 * MARGIN, kb_height)
hand = Hand(position=(kb_rect.x + 10, kb_rect.y + 10), size=(WIDTH//NUM_WHITE_KEYS*5, 100))

# Progress bar 
bar_width = kb_rect.width
bar_x = kb_rect.x
bar_y = kb_rect.bottom + PROGRESS_BAR_MARGIN_TOP
bg_rect = pygame.Rect(bar_x, bar_y, bar_width, PROGRESS_BAR_HEIGHT)
fill_rect = pygame.Rect(bar_x, bar_y, 0, PROGRESS_BAR_HEIGHT)

def build_white_keys(rect: pygame.Rect, n: int):
	keys = []
	key_w = rect.width // n
	for i in range(n):
		label = str(i)
		r = pygame.Rect(rect.x + i * key_w, rect.y, key_w, rect.height)
		keys.append((label, r))
	return keys

def main():
	create_legend_image("legend.png")  # generate legend image

	pygame.init()
	LEGEND_WIDTH = 150
	screen = pygame.display.set_mode((WIDTH + LEGEND_WIDTH, HEIGHT))
	clock = pygame.time.Clock()
	legend_img = pygame.image.load("legend.png")
	legend_img = pygame.transform.scale(legend_img, (250, 175))  # width, height
	legend_rect = legend_img.get_rect(topleft=(WIDTH - 110, MARGIN))
	font = pygame.font.SysFont(None, 18)

	
	white_keys = build_white_keys(kb_rect, NUM_WHITE_KEYS)

	total_duration = hand.start_times[-1]

	elaspedTime = 0
	running = True
	pause = False

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

		hand.play(elaspedTime)
		# Update time only when not paused
		if not pause:
			elaspedTime += dt
		elaspedTime = min(max(elaspedTime, 0), total_duration)
		screen.fill(BG_COLOR)

		# Grabs the currently active chord notes based on hand index
		# Determine which keys to highlight for the current event time window
		i = hand.index
		current_start = hand.start_times[i]
		# Collect contiguous rows with the same start_time as the current index
		indices = [i]
		j = i - 1
		while j >= 0 and hand.start_times[j] == current_start:
			indices.append(j)
			j -= 1
		j = i + 1
		while j < len(hand.start_times) and hand.start_times[j] == current_start:
			indices.append(j)
			j += 1
		# Map these rows to white key indices
		chord_highlights = set()
		for idx_row in indices:
			note_name = str(hand.fingerData.at[idx_row, 'key'])
			wi = note_name_to_white_index(note_name)
			if wi is not None:
				chord_highlights.add(int(wi))

		# Draw white keys
		for label, r in white_keys:
			key_index = int(label)
			if key_index in chord_highlights:
				# If multiple notes start together, use blue; otherwise default single-note color
				color = (120, 180, 255) if len(chord_highlights) > 1 else (255, 240, 160)
				pygame.draw.rect(screen, color, r)

			else:
				pygame.draw.rect(screen, WHITE_KEY_COLOR, r)
			pygame.draw.rect(screen, WHITE_KEY_OUTLINE, r, width=2)
			text = font.render(label, True, LABEL_COLOR)
			screen.blit(text, (r.centerx - text.get_width()/2, r.bottom - text.get_height()))

		# Progress bar calculations
		progress = max(0, min(1, elaspedTime / total_duration))
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

		hand.draw(screen)
		screen.blit(legend_img, legend_rect)
		pygame.display.flip()

	pygame.quit()
	return 0


if __name__ == "__main__":
	sys.exit(main())

