"""
Shared constants for the piano visualization and hand rendering.
Import this module from any file that needs to reference the same values.
"""

# Window / timing
WIDTH = 900
HEIGHT = 500  # Increased to accommodate sheet music
FPS = 60

# Layout
MARGIN = 20
NUM_WHITE_KEYS = 52  # standard full-size keyboard white keys count

# Optional note labels for white keys per octave (unused by default)
WHITE_KEYS = ["C", "D", "E", "F", "G", "A", "B"]
# Keys per second speed
HAND_SPEED = 5 * (WIDTH - MARGIN - (WIDTH // NUM_WHITE_KEYS))

HIGHLIGHT_COLOR = (255, 138, 138)
RIGHT_HAND_COLOR = (80, 170, 255)
LEFT_HAND_COLOR = (255, 180, 90)
HAND_ALPHA = 140

PROGRESS_BAR_HEIGHT = 12
PROGRESS_BAR_MARGIN_TOP = 4

BG_COLOR = (255, 255, 255)
WHITE_KEY_COLOR = (255, 255, 255)
WHITE_KEY_OUTLINE = (0, 0, 0)
BLACK_KEY_COLOR = (0, 0, 0)
BLACK_KEY_OUTLINE = (100, 100, 100)
LABEL_COLOR = (0, 0, 0)

THUMB_HIGHLIGHT_COLOR = HIGHLIGHT_COLOR
HAND_NOTE_COLORS = {
	'L': LEFT_HAND_COLOR,
	'R': RIGHT_HAND_COLOR,
}
UNKNOWN_HAND_NOTE_COLOR = (170, 170, 170)
MIXED_NOTE_COLOR = (200, 130, 240)
THUMB_OVERLAY_ALPHA = 140