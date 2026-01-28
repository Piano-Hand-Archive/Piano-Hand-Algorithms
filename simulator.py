"""
Standalone Robotic Piano Simulator
No external dependencies except pygame and pandas
"""
import sys
from collections import defaultdict
import pandas as pd
import pygame
import os

# ==========================================
# CONSTANTS (Previously in piano_constants.py)
# ==========================================

# Window Dimensions
WIDTH = 1200
HEIGHT = 400
MARGIN = 20
FPS = 60

# Keyboard
NUM_WHITE_KEYS = 52  # 88 keys piano has 52 white keys
WHITE_KEY_COLOR = (255, 255, 255)
BLACK_KEY_COLOR = (0, 0, 0)
WHITE_KEY_OUTLINE = (0, 0, 0)
BLACK_KEY_OUTLINE = (50, 50, 50)

# Hand Colors
LEFT_HAND_COLOR = (100, 150, 255)  # Blue
RIGHT_HAND_COLOR = (255, 100, 100)  # Red
HAND_ALPHA = 180

# Note Highlighting Colors
HAND_NOTE_COLORS = {
    'L': (150, 200, 255),  # Light Blue
    'R': (255, 150, 150),  # Light Red
}
MIXED_NOTE_COLOR = (200, 150, 255)  # Purple (both hands)
UNKNOWN_HAND_NOTE_COLOR = (200, 200, 200)  # Gray

# Thumb Highlighting
THUMB_HIGHLIGHT_COLOR = (255, 255, 0)  # Yellow
THUMB_OVERLAY_ALPHA = 80

# Progress Bar
PROGRESS_BAR_HEIGHT = 30
PROGRESS_BAR_MARGIN_TOP = 10

# Background
BG_COLOR = (240, 240, 240)


# ==========================================
# AUDIO SYNTHESIS (Previously in piano_sound.py)
# ==========================================

def pre_init_audio():
    """Initialize pygame audio with optimal settings."""
    pygame.mixer.pre_init(44100, -16, 2, 512)


def _note_name_to_midi(note_name):
    """Convert note name (e.g., 'C4') to MIDI number."""
    note_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}

    note_name = note_name.strip()
    note = note_name[0].upper()
    octave = int(note_name[-1])

    # Handle sharps/flats if present
    accidental = 0
    if len(note_name) > 2:
        if '#' in note_name:
            accidental = 1
        elif 'b' in note_name:
            accidental = -1

    midi = (octave + 1) * 12 + note_map[note] + accidental
    return midi


class PianoSound:
    """Simple piano sound synthesizer using pygame."""

    def __init__(self):
        self.active_channels = {}
        self.sounds = {}
        self._generate_tones()

    def _generate_tones(self):
        """Generate simple sine wave tones for each MIDI note."""
        import numpy as np

        sample_rate = 44100
        duration = 2.0  # seconds

        for midi in range(21, 109):  # Piano range
            freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))

            # Generate sine wave
            t = np.linspace(0, duration, int(sample_rate * duration))
            wave = np.sin(2 * np.pi * freq * t)

            # Apply envelope (ADSR)
            attack = int(0.01 * sample_rate)
            decay = int(0.1 * sample_rate)
            sustain_level = 0.7
            release = int(0.2 * sample_rate)

            envelope = np.ones_like(wave)
            if len(envelope) > attack:
                envelope[:attack] = np.linspace(0, 1, attack)
            if len(envelope) > attack + decay:
                envelope[attack:attack + decay] = np.linspace(1, sustain_level, decay)
            if len(envelope) > len(envelope) - release:
                envelope[-release:] = np.linspace(sustain_level, 0, release)

            wave = wave * envelope * 0.3  # Volume control

            # Convert to 16-bit PCM
            wave = (wave * 32767).astype(np.int16)

            # Create stereo
            stereo_wave = np.column_stack((wave, wave))

            # Create pygame Sound
            sound = pygame.sndarray.make_sound(stereo_wave)
            self.sounds[midi] = sound

    def set_active_notes_from_midi(self, midi_set):
        """Update playing notes based on MIDI set."""
        # Stop notes that should no longer play
        for midi in list(self.active_channels.keys()):
            if midi not in midi_set:
                # IMPROVED: Fade out over 50ms instead of instant stop (prevents clicking)
                self.active_channels[midi].fadeout(50)
                del self.active_channels[midi]

        # Start new notes
        for midi in midi_set:
            if midi not in self.active_channels and midi in self.sounds:
                channel = self.sounds[midi].play(-1)  # Loop
                if channel:
                    self.active_channels[midi] = channel

    def stop_all(self):
        """Stop all playing notes."""
        for channel in self.active_channels.values():
            channel.stop()
        self.active_channels.clear()


# ==========================================
# MUSIC UTILITIES (Previously in tylermusicutils.py)
# ==========================================

def midi_to_white_key_index(midi):
    """Convert MIDI note number to white key index (0-based, 0 = A0)."""
    offset = midi - 21  # A0 = MIDI 21
    octave = offset // 12
    note_in_octave = offset % 12
    # Correct map for A-based indexing: A=0, B=2, C=3, D=5, E=7, F=8, G=10
    white_key_map = {0: 0, 2: 1, 3: 2, 5: 3, 7: 4, 8: 5, 10: 6}

    if note_in_octave not in white_key_map:
        return None

    return octave * 7 + white_key_map[note_in_octave]


def build_white_keys(kb_rect, num_keys):
    """Build white key rectangles."""
    keys = []
    key_width = kb_rect.width / num_keys

    for i in range(num_keys):
        x = kb_rect.x + i * key_width
        rect = pygame.Rect(int(x), kb_rect.y, int(key_width), kb_rect.height)
        keys.append((str(i), rect))

    return keys


def build_black_keys(white_keys, white_key_width, kb_x, kb_rect):
    """Build black key rectangles based on white key positions."""
    black_keys = []
    black_width = white_key_width * 0.6
    black_height = kb_rect.height * 0.6

    # Black key pattern: 2-3-2-3 (C#, D#, F#, G#, A#)
    pattern = [1, 1, 0, 1, 1, 1, 0]  # 1 = has black key above

    for i, (label, rect) in enumerate(white_keys):
        octave_pos = i % 7

        if pattern[octave_pos] == 1:
            # Calculate MIDI value for this black key
            white_key_idx = int(label)
            # Estimate MIDI (this is approximate)
            base_midi = 21 + (white_key_idx // 7) * 12

            if octave_pos == 0:  # C#
                midi = base_midi + 1
            elif octave_pos == 1:  # D#
                midi = base_midi + 3
            elif octave_pos == 3:  # F#
                midi = base_midi + 6
            elif octave_pos == 4:  # G#
                midi = base_midi + 8
            elif octave_pos == 5:  # A#
                midi = base_midi + 10
            else:
                continue

            x = rect.right - black_width / 2
            black_rect = pygame.Rect(int(x), kb_rect.y, int(black_width), int(black_height))
            black_keys.append((midi, black_rect))

    return black_keys


def create_legend_image(filename):
    """Create a legend image showing hand colors."""
    img_width, img_height = 200, 150
    surface = pygame.Surface((img_width, img_height))
    surface.fill((255, 255, 255))

    font = pygame.font.SysFont(None, 20)

    # Title
    title = font.render("Legend", True, (0, 0, 0))
    surface.blit(title, (10, 10))

    # Left Hand
    pygame.draw.rect(surface, HAND_NOTE_COLORS['L'], (10, 40, 30, 20))
    text = font.render("Left Hand", True, (0, 0, 0))
    surface.blit(text, (50, 40))

    # Right Hand
    pygame.draw.rect(surface, HAND_NOTE_COLORS['R'], (10, 70, 30, 20))
    text = font.render("Right Hand", True, (0, 0, 0))
    surface.blit(text, (50, 70))

    # Both Hands
    pygame.draw.rect(surface, MIXED_NOTE_COLOR, (10, 100, 30, 20))
    text = font.render("Both Hands", True, (0, 0, 0))
    surface.blit(text, (50, 100))

    # Thumb
    pygame.draw.rect(surface, THUMB_HIGHLIGHT_COLOR, (10, 130, 30, 20))
    text = font.render("Thumb Pos", True, (0, 0, 0))
    surface.blit(text, (50, 130))

    pygame.image.save(surface, filename)


# ==========================================
# HAND CLASS (Previously in tylershand.py)
# ==========================================

class Hand:
    """Represents a robotic hand on the piano."""

    def __init__(self, position, size, color, alpha, hand_label, finger_data):
        self.position = position
        self.size = size
        self.color = color
        self.alpha = alpha
        self.hand_label = hand_label

        # Filter data for this hand
        if not finger_data.empty:
            self.fingerData = finger_data[finger_data['hand'] == hand_label].copy()
            self.fingerData = self.fingerData.sort_values('start_time').reset_index(drop=True)
        else:
            self.fingerData = pd.DataFrame()

        self.index = 0
        self.x_offset = 0

    def play(self, elapsed_time):
        """Update hand position based on elapsed time."""
        if self.fingerData.empty:
            return

        # Find current row based on time
        valid_rows = self.fingerData[self.fingerData['start_time'] <= elapsed_time]

        if not valid_rows.empty:
            self.index = valid_rows.index[-1]

            # Update x position based on thumb position
            if 'thumb_pos' in self.fingerData.columns:
                thumb_pos = self.fingerData.iloc[self.index]['thumb_pos']
                if pd.notna(thumb_pos):
                    # Calculate x offset based on white key position
                    # Each white key is roughly (WIDTH - 2*MARGIN) / NUM_WHITE_KEYS pixels
                    key_width = (WIDTH - 2 * MARGIN) / NUM_WHITE_KEYS
                    self.x_offset = MARGIN + int(thumb_pos) * key_width

    def draw(self, screen):
        """Draw the hand on the screen."""
        if self.fingerData.empty or self.index >= len(self.fingerData):
            return

        # Create semi-transparent surface
        hand_surface = pygame.Surface(self.size, pygame.SRCALPHA)

        # Draw hand body (rectangle with rounded effect)
        pygame.draw.rect(hand_surface, (*self.color, self.alpha),
                         (0, 0, self.size[0], self.size[1]), border_radius=10)

        # Draw fingers (5 small rectangles)
        finger_width = self.size[0] // 6
        finger_height = self.size[1] // 3

        for i in range(5):
            finger_x = (i + 0.5) * finger_width
            finger_y = 0
            pygame.draw.rect(hand_surface, (*self.color, self.alpha + 50),
                             (int(finger_x), int(finger_y), int(finger_width * 0.8), int(finger_height)),
                             border_radius=5)

        # Draw hand label
        font = pygame.font.SysFont(None, 24)
        label_text = font.render(self.hand_label, True, (255, 255, 255))
        label_rect = label_text.get_rect(center=(self.size[0] // 2, self.size[1] // 2))
        hand_surface.blit(label_text, label_rect)

        # Blit to main screen at current position
        screen.blit(hand_surface, (self.x_offset, self.position[1]))


# ==========================================
# DATA PROCESSING
# ==========================================

def process_fingering_data(csv_path: str) -> pd.DataFrame:
    """
    Parses the 'fingering_plan.csv' generated by findOptimalHandPos.py
    and converts it into the Long Format DataFrame expected by the Hand class.
    """
    try:
        raw_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}. Run findOptimalHandPos.py first.")
        sys.exit(1)

    records = []

    for _, row in raw_df.iterrows():
        time = row['Time']

        # Process Left Hand
        if pd.notna(row['L_Notes']) and str(row['L_Notes']).strip():
            notes = str(row['L_Notes']).split(';')
            thumb_pos = row['L_Thumb']
            for note_name in notes:
                midi_val = _note_name_to_midi(note_name)
                wk_index = midi_to_white_key_index(midi_val)
                records.append({
                    'start_time': time,
                    'hand': 'L',
                    'thumb_pos': thumb_pos,
                    'midi': midi_val,
                    'white_key_index': wk_index,
                    'note_name': note_name
                })

        # Process Right Hand
        if pd.notna(row['R_Notes']) and str(row['R_Notes']).strip():
            notes = str(row['R_Notes']).split(';')
            thumb_pos = row['R_Thumb']
            for note_name in notes:
                midi_val = _note_name_to_midi(note_name)
                wk_index = midi_to_white_key_index(midi_val)
                records.append({
                    'start_time': time,
                    'hand': 'R',
                    'thumb_pos': thumb_pos,
                    'midi': midi_val,
                    'white_key_index': wk_index,
                    'note_name': note_name
                })

    if not records:
        print("Warning: Fingering plan is empty.")
        return pd.DataFrame(columns=['start_time', 'hand', 'thumb_pos', 'midi', 'white_key_index'])

    return pd.DataFrame(records)


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def note_color_for_hands(hand_ids, default_color):
    """Determine note color based on which hands are playing it."""
    if not hand_ids:
        return default_color
    if len(hand_ids) == 1:
        hand_id = next(iter(hand_ids))
        return HAND_NOTE_COLORS.get(hand_id, UNKNOWN_HAND_NOTE_COLOR)
    return MIXED_NOTE_COLOR


def blit_thumb_overlay(surface, rect):
    """Draw semi-transparent yellow overlay for thumb position."""
    overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    overlay.fill((THUMB_HIGHLIGHT_COLOR[0], THUMB_HIGHLIGHT_COLOR[1],
                  THUMB_HIGHLIGHT_COLOR[2], THUMB_OVERLAY_ALPHA))
    surface.blit(overlay, rect)


def save_recording(frames):
    """Save recorded frames as video file."""
    if not frames:
        print("⚠️  No frames to save")
        return

    print(f"💾 Processing {len(frames)} frames...")

    # Try to save as MP4 using opencv
    try:
        import cv2
        import numpy as np

        # Transpose from pygame format (width, height, 3) to opencv format (height, width, 3)
        height, width, _ = np.transpose(frames[0], (1, 0, 2)).shape

        filename = f"recording_{int(pygame.time.get_ticks())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 60.0, (width, height))

        for i, frame in enumerate(frames):
            if i % 60 == 0:  # Progress every second
                print(f"  Processing: {i}/{len(frames)} frames ({i/len(frames)*100:.0f}%)")

            # Convert pygame surface to opencv format
            frame = np.transpose(frame, (1, 0, 2))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)

        out.release()
        print(f"✅ Video saved: {filename}")
        return

    except ImportError:
        print("⚠️  opencv-python not installed, trying GIF export...")
    except Exception as e:
        print(f"⚠️  MP4 export failed: {e}, trying GIF export...")

    # Fallback: Save as animated GIF using PIL
    try:
        from PIL import Image
        import numpy as np

        filename = f"recording_{int(pygame.time.get_ticks())}.gif"

        # Convert frames to PIL images
        pil_frames = []
        for i, frame in enumerate(frames[::2]):  # Every other frame to reduce size
            if i % 30 == 0:  # Progress every half second
                print(f"  Processing: {i*2}/{len(frames)} frames ({i*2/len(frames)*100:.0f}%)")

            frame = np.transpose(frame, (1, 0, 2))
            pil_frame = Image.fromarray(frame.astype('uint8'), 'RGB')
            pil_frames.append(pil_frame)

        # Save as GIF (30 FPS instead of 60)
        pil_frames[0].save(
            filename,
            save_all=True,
            append_images=pil_frames[1:],
            duration=33,  # 33ms per frame = ~30 FPS
            loop=0
        )

        print(f"✅ GIF saved: {filename}")
        print(f"   Note: GIF is at 30 FPS (reduced from 60 FPS for file size)")
        return

    except ImportError:
        print("❌ Neither opencv-python nor PIL installed")
        print("   Install with: pip install opencv-python")
        print("   Or: pip install Pillow")
    except Exception as e:
        print(f"❌ Recording save failed: {e}")


# ==========================================
# MAIN SIMULATOR
# ==========================================

def main():
    """Main simulator loop."""

    # Check for numpy (needed for sound generation)
    try:
        import numpy
    except ImportError:
        print("Warning: numpy not found. Running without sound.")
        print("Install with: pip install numpy")
        use_sound = False
    else:
        use_sound = True

    # Generate legend
    pygame.init()  # Need to init pygame before creating images
    create_legend_image("legend.png")

    # Audio Init
    if use_sound:
        pre_init_audio()

    pygame.init()

    # Window Setup
    LEGEND_WIDTH = 150
    screen = pygame.display.set_mode((WIDTH + LEGEND_WIDTH, HEIGHT))
    pygame.display.set_caption("Robotic Piano Simulator (Standalone)")
    clock = pygame.time.Clock()

    # Load assets
    if os.path.exists("legend.png"):
        legend_img = pygame.image.load("legend.png")
        legend_img = pygame.transform.scale(legend_img, (250, 175))
        legend_rect = legend_img.get_rect(topleft=(WIDTH - 110, MARGIN))
    else:
        legend_img = None
        legend_rect = None

    font = pygame.font.SysFont(None, 18)

    # Synth Setup
    if use_sound:
        try:
            synth = PianoSound()
        except Exception as e:
            print(f"Warning: Could not initialize sound: {e}")
            use_sound = False
            synth = None
    else:
        synth = None

    # Keyboard Layout
    kb_height = HEIGHT - 2 * MARGIN - PROGRESS_BAR_HEIGHT - PROGRESS_BAR_MARGIN_TOP
    kb_rect = pygame.Rect(MARGIN, MARGIN, WIDTH - 2 * MARGIN, kb_height)
    white_keys = build_white_keys(kb_rect, NUM_WHITE_KEYS)
    white_key_width = kb_rect.width / NUM_WHITE_KEYS
    black_keys = build_black_keys(white_keys, white_key_width, kb_rect.x, kb_rect)

    # Progress Bar Layout
    bar_width = kb_rect.width
    bar_x = kb_rect.x
    bar_y = kb_rect.bottom + PROGRESS_BAR_MARGIN_TOP
    bg_rect = pygame.Rect(bar_x, bar_y, bar_width, PROGRESS_BAR_HEIGHT)
    fill_rect = pygame.Rect(bar_x, bar_y, 0, PROGRESS_BAR_HEIGHT)

    # --- DATA LOADING ---
    print("Loading fingering plan...")
    finger_df = process_fingering_data('fingering_plan.csv')

    if finger_df.empty:
        total_duration = 10.0
    else:
        total_duration = float(finger_df['start_time'].max()) + 2.0

    # Hand Setup
    hand_width = max((kb_rect.width // NUM_WHITE_KEYS) * 5, 60)
    hand_height = 100
    base_x = kb_rect.x + 10
    right_y = kb_rect.y + 10
    left_y = right_y + hand_height // 2

    hands = []
    for hand_label, color, y in (('L', LEFT_HAND_COLOR, left_y),
                                 ('R', RIGHT_HAND_COLOR, right_y)):
        hands.append(Hand(
            position=(base_x, y),
            size=(hand_width, hand_height),
            color=color,
            alpha=HAND_ALPHA,
            hand_label=hand_label,
            finger_data=finger_df
        ))

    elapsedTime = 0.0
    running = True
    pause = False
    last_chord_midi_values = set()

    # Time management
    start_ticks = pygame.time.get_ticks()
    pause_offset = 0.0
    pause_start_ticks = 0

    # Playback speed control
    playback_speed = 1.0

    # Recording state
    recording = False
    frames = []

    print("\n" + "="*50)
    print("SIMULATOR CONTROLS")
    print("="*50)
    print("  SPACE      - Pause/Resume")
    print("  ← / →      - Seek ±1 second")
    print("  A / D      - Seek ±1 second")
    print("  + / =      - Speed up (+0.25x)")
    print("  -          - Slow down (-0.25x)")
    print("  R          - Start/Stop recording")
    print("  ESC / Q    - Quit")
    print("="*50 + "\n")

    # --- MAIN LOOP ---
    while running:
        clock.tick(FPS)  # Still needed for frame rate limiting

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_LEFT, pygame.K_a):
                    elapsedTime -= 1.0
                    pause_offset += 1.0  # Adjust offset for manual rewind
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    elapsedTime += 1.0
                    pause_offset -= 1.0  # Adjust offset for manual forward
                elif event.key == pygame.K_SPACE:
                    pause = not pause
                    if pause:
                        pause_start_ticks = pygame.time.get_ticks()
                        if use_sound:
                            pygame.mixer.pause()
                    else:
                        pause_offset += (pygame.time.get_ticks() - pause_start_ticks) / 1000.0
                        if use_sound:
                            pygame.mixer.unpause()
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    playback_speed = min(playback_speed + 0.25, 4.0)
                    print(f"⏩ Playback speed: {playback_speed:.2f}x")
                elif event.key == pygame.K_MINUS:
                    playback_speed = max(playback_speed - 0.25, 0.25)
                    print(f"⏪ Playback speed: {playback_speed:.2f}x")
                elif event.key == pygame.K_r:
                    recording = not recording
                    if recording:
                        frames = []
                        print("🔴 Recording started...")
                    else:
                        print("💾 Saving video...")
                        save_recording(frames)
                elif event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

        # Update Hands
        for hand_obj in hands:
            hand_obj.play(elapsedTime)

        # Update Time - FIXED: Use system clock for accuracy with playback speed
        if not pause:
            elapsedTime = ((pygame.time.get_ticks() - start_ticks) / 1000.0 - pause_offset) * playback_speed

        elapsedTime = min(max(elapsedTime, 0.0), total_duration)

        # --- DRAWING ---
        screen.fill(BG_COLOR)

        # Collect highlights
        white_key_highlights = defaultdict(set)
        midi_highlights = defaultdict(set)
        thumb_highlights = defaultdict(set)

        for hand_obj in hands:
            hand_df = hand_obj.fingerData
            if hand_df.empty:
                continue

            current_idx = min(max(hand_obj.index, 0), len(hand_df) - 1)
            hand_id = (hand_obj.hand_label or 'UNK').upper()

            # Thumb position highlighting
            if 'thumb_pos' in hand_df.columns:
                thumb_val = hand_df.iloc[current_idx]['thumb_pos']
                if pd.notna(thumb_val):
                    thumb_highlights[int(thumb_val)].add(hand_id)

            # Find active notes (simple approach: show notes at current time)
            active_mask = hand_df['start_time'] <= elapsedTime
            active_rows = hand_df.loc[active_mask]

            # Only show the most recent notes (within 0.5 seconds)
            if not active_rows.empty:
                recent_time = active_rows['start_time'].max()
                active_rows = active_rows[active_rows['start_time'] >= recent_time - 0.5]

            for _, row in active_rows.iterrows():
                wi = row.get('white_key_index')
                if pd.notna(wi):
                    white_key_highlights[int(wi)].add(hand_id)
                midi = row.get('midi')
                if pd.notna(midi):
                    midi_highlights[int(midi)].add(hand_id)

        # Audio Logic
        if use_sound and synth:
            chord_midi_values = set(midi_highlights.keys())
            if not pause and elapsedTime < total_duration and chord_midi_values != last_chord_midi_values:
                synth.set_active_notes_from_midi(chord_midi_values)
                last_chord_midi_values = chord_midi_values.copy()

            if elapsedTime >= total_duration:
                synth.stop_all()
                last_chord_midi_values = set()

        # Draw White Keys
        for label, r in white_keys:
            key_index = int(label)
            fill_color = note_color_for_hands(white_key_highlights.get(key_index),
                                              WHITE_KEY_COLOR)
            pygame.draw.rect(screen, fill_color, r)

            # Draw Thumb Overlay
            if key_index in thumb_highlights:
                blit_thumb_overlay(screen, r)

            pygame.draw.rect(screen, WHITE_KEY_OUTLINE, r, width=2)

        # Draw Black Keys
        for midi_val, r in black_keys:
            fill_color = note_color_for_hands(midi_highlights.get(midi_val),
                                              BLACK_KEY_COLOR)
            pygame.draw.rect(screen, fill_color, r)
            pygame.draw.rect(screen, BLACK_KEY_OUTLINE, r, width=1)

        # Draw Progress Bar
        progress_base = total_duration if total_duration > 0 else 1
        progress = max(0.0, min(1.0, elapsedTime / progress_base))
        fill_rect.size = (int(bar_width * progress), PROGRESS_BAR_HEIGHT)

        pygame.draw.rect(screen, (255, 255, 255), bg_rect)
        if fill_rect.width > 0:
            pygame.draw.rect(screen, (255, 255, 0), fill_rect)
            pygame.draw.rect(screen, (0, 0, 0), fill_rect, width=1)
        pygame.draw.rect(screen, (0, 0, 0), bg_rect, width=1)

        elapsed_text = font.render(f"{elapsedTime:.1f}s / {total_duration:.1f}s",
                                   True, (40, 40, 40))
        screen.blit(elapsed_text,
                    (bg_rect.centerx - elapsed_text.get_width() // 2,
                     bg_rect.centery - elapsed_text.get_height() // 2))

        # Display playback speed
        speed_text = font.render(f"{playback_speed:.2f}x", True, (40, 40, 40))
        screen.blit(speed_text, (bg_rect.right - 60, bg_rect.centery - 9))

        # Display recording indicator
        if recording:
            rec_text = font.render("🔴 REC", True, (255, 0, 0))
            screen.blit(rec_text, (bg_rect.left + 5, bg_rect.centery - 9))

        # Draw Hands
        for hand_obj in hands:
            hand_obj.draw(screen)

        # Draw Legend
        if legend_img and legend_rect:
            screen.blit(legend_img, legend_rect)

        # Draw on-screen controls (bottom of screen)
        control_font = pygame.font.SysFont(None, 16)
        controls = [
            "SPACE: Pause | ←/→: Seek | +/-: Speed | R: Record | ESC: Quit"
        ]
        y_offset = HEIGHT - 25
        for i, control in enumerate(controls):
            text = control_font.render(control, True, (100, 100, 100))
            text_rect = text.get_rect(center=(WIDTH // 2, y_offset + i * 18))
            screen.blit(text, text_rect)

        # Capture frame if recording
        if recording:
            try:
                frame = pygame.surfarray.array3d(screen)
                frames.append(frame)
            except Exception as e:
                print(f"Warning: Could not capture frame: {e}")

        pygame.display.flip()

    # Cleanup
    if use_sound and synth:
        try:
            synth.stop_all()
        except Exception:
            pass

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()