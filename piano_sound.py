import math
from array import array
import pygame

SAMPLE_RATE = 44100
AUDIO_VOLUME = 0.30  # 0.0 .. 1.0
MIXER_CHANNELS = 32
FADE_IN_MS = 5
FADE_OUT_MS = 50

# Map note letters to semitone offsets (white keys only)
LETTER_TO_SEMITONE = {
    'C': 0,
    'D': 2,
    'E': 4,
    'F': 5,
    'G': 7,
    'A': 9,
    'B': 11,
}

def pre_init_audio(sample_rate: int = SAMPLE_RATE) -> None:
    pygame.mixer.pre_init(sample_rate, size=-16, channels=1, buffer=512)

def _note_name_to_midi(note_name: str):
	letter = note_name[0].upper()
	octave = int(note_name[-1])
	semitone = LETTER_TO_SEMITONE[letter]
	# MIDI mapping: C4 -> 60, so midi = 12 * (octave + 1) + semitone
	return 12 * (octave + 1) + semitone

class PianoSound:
    def __init__(self, sample_rate: int = SAMPLE_RATE, volume: float = AUDIO_VOLUME):
        self.sample_rate = sample_rate
        self.volume = max(0.0, min(1.0, volume))
        self._midi_to_sound = {}
        self._active_notes = {}

        pygame.mixer.set_num_channels(MIXER_CHANNELS)

    @staticmethod
    def _freq_for_midi(midi: int):
        return 440.0 * (2.0 ** ((midi - 69) / 12.0))

    def _build_loopable_sine(self, freq: float):
        samples_per_period = max(64, int(round(self.sample_rate / max(1e-6, freq))))
        amp = int(self.volume * 32767)
        buf = array(
            'h',
            (int(amp * math.sin(2.0 * math.pi * n / samples_per_period)) for n in range(samples_per_period))
        )
        # Create a mono, 16-bit sound; mixer must be pre-initialized accordingly
        return pygame.mixer.Sound(buffer=buf.tobytes())

    def _sound_for_midi(self, midi: int):
        if midi is None:
            return None
        if midi not in self._midi_to_sound:
            freq = self._freq_for_midi(midi)
            self._midi_to_sound[midi] = self._build_loopable_sine(freq)
        return self._midi_to_sound[midi]

    def set_active_notes(self, note_names) -> None:
        """Start/stop notes to match the provided set of note names."""
        to_stop = [n for n in self._active_notes.keys() if n not in note_names]
        for n in to_stop:
            ch = self._active_notes.pop(n, None)
            if ch:
                try:
                    ch.fadeout(FADE_OUT_MS)
                except pygame.error:
                    ch.stop()

        # Start any new notes
        for n in note_names:
            if n in self._active_notes:
                continue
            midi = _note_name_to_midi(n)
            if midi is None:
                continue  # skip accidentals or invalid
            snd = self._sound_for_midi(midi)
            if snd is None:
                continue
            ch = pygame.mixer.find_channel(True)  # force allocate if none are free
            if ch is not None:
                ch.play(snd, loops=-1, fade_ms=FADE_IN_MS)
                self._active_notes[n] = ch

    def stop_all(self):
        for ch in self._active_notes.values():
            try:
                ch.fadeout(FADE_OUT_MS)
            except pygame.error:
                ch.stop()
        self._active_notes.clear()
