"""
Shared constants for the piano visualization and hand rendering.
Import this module from any file that needs to reference the same values.
"""

# Window / timing
WIDTH = 900
HEIGHT = 220
FPS = 60

# Layout
MARGIN = 20
NUM_WHITE_KEYS = 52  # standard full-size keyboard white keys count

# Optional note labels for white keys per octave (unused by default)
WHITE_KEYS = ["C", "D", "E", "F", "G", "A", "B"]
# Keys per second speed
HAND_SPEED = 5 * (WIDTH - MARGIN - (WIDTH // NUM_WHITE_KEYS))