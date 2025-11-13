import pygame
import pandas as pd
from bisect import bisect_right
from typing import Dict, List, Optional, Tuple
from piano_constants import *
from tylermusicutils import *

class Hand:
    def __init__(
        self,
        position = (20, 20),
        size = (60, 40),
        color = (0, 255, 255),
        alpha = 100,
        *,
        hand_label: Optional[str] = None,
        csv_path: str = 'fingering_plan_updated.csv',
        finger_data: Optional[pd.DataFrame] = None,
    ) -> None:
        self.x, self.y = position
        self.w, self.h = size
        self.color = color
        self.alpha = max(0, min(255, alpha))
        self.hand_label = hand_label.upper() if hand_label else None
        self.rect = pygame.Rect(self.x, self.y, self.w, self.h)

        # Load and filter fingering data
        if finger_data is not None:
            data = finger_data.copy()
        else:
            data = pd.read_csv(csv_path)

        if self.hand_label and 'hand' in data.columns:
            data = data[
                data['hand'].astype(str).str.upper() == self.hand_label
            ].copy()

        if data.empty:
            raise ValueError(
                f"No fingering data available for hand '{self.hand_label or 'ALL'}'."
            )

        data = data.sort_values('start_time').reset_index(drop=True)
        data['start_time'] = pd.to_numeric(data['start_time'], errors='coerce').fillna(0.0)

        DEFAULT_NOTE_DURATION = 0.5

        def _time_key(value: float) -> float:
            return round(float(value), 5)

        def _build_duration_lookup() -> Dict[Tuple[float, int], float]:
            try:
                duration_df = pd.read_csv('timed_steps.csv')
            except FileNotFoundError:
                return {}
            duration_df = duration_df.copy()
            required_cols = {'start_time', 'midi', 'duration'}
            if not required_cols.issubset(duration_df.columns):
                return {}
            duration_df['start_time'] = pd.to_numeric(duration_df['start_time'], errors='coerce')
            duration_df['midi'] = pd.to_numeric(duration_df.get('midi'), errors='coerce')
            duration_df['duration'] = pd.to_numeric(duration_df.get('duration'), errors='coerce')
            duration_df = duration_df.dropna(subset=['start_time', 'midi', 'duration'])
            lookup: Dict[Tuple[float, int], float] = {}
            for _, row in duration_df.iterrows():
                key = (_time_key(row['start_time']), int(row['midi']))
                lookup[key] = max(float(row['duration']), 0.0)
            return lookup

        duration_lookup = _build_duration_lookup()
        midi_series = pd.to_numeric(data.get('midi'), errors='coerce')
        data['midi'] = midi_series

        note_times: List[float] = []
        for start_val, midi_val in zip(data['start_time'], midi_series):
            note_duration = None
            if pd.notna(midi_val):
                key = (_time_key(start_val), int(midi_val))
                note_duration = duration_lookup.get(key)
            if note_duration is None or note_duration <= 0:
                note_duration = DEFAULT_NOTE_DURATION
            note_times.append(float(note_duration))

        data['note_time'] = note_times
        data['note_end_time'] = data['start_time'] + data['note_time']

        self.fingerData = data
        self.start_times = data['start_time'].astype(float).tolist()

        thumb_default = pd.Series([0] * len(data))
        thumb_series = pd.to_numeric(data.get('thumb_pos', thumb_default), errors='coerce').fillna(0)
        self.thumb_x = (
            (thumb_series * (WIDTH - MARGIN) // NUM_WHITE_KEYS)
            .astype(int)
            .tolist()
        )

        # Movement control
        self.index = 0
        self.max_speed = HAND_SPEED
        self._last_time = 0.0
    def set_position(self, x, y) -> None:
        self.x, self.y = x, y
        self._sync_rect()

    def play(self, elapsed_time) -> None:
        elapsed_time = float(elapsed_time)
        i = bisect_right(self.start_times, elapsed_time) - 1
        if i < 0:
            i = 0
        elif i >= len(self.fingerData):
            i = len(self.fingerData) - 1
        if i != self.index:
            self.index = i
        dt = elapsed_time - self._last_time
        if dt < 0:  # rewind / scrubbing backwards: snap directly
            dt = 0
            self.x = self._target_position(elapsed_time)
            self._sync_rect()
        else:
            self._update_from_index(elapsed_time, dt)
        self._last_time = elapsed_time

    def set_color(self, color) -> None:
        self.color = color

    def draw(self, surface: pygame.Surface) -> None:
        hand_surface = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        fill_color = (*self.color, self.alpha)
        pygame.draw.ellipse(hand_surface, fill_color, hand_surface.get_rect())
        pygame.draw.ellipse(hand_surface, (0, 0, 0), hand_surface.get_rect(), width=2)
        surface.blit(hand_surface, self.rect)

    @property
    def position(self):
        return self.x, self.y

    @property
    def size(self):
        return self.w, self.h

    # private helpers
    def _sync_rect(self) -> None:
        self.rect.x = self.x
        self.rect.y = self.y

    def _target_position(self, elapsed_time: float) -> int:
        """Compute the ideal interpolated x (without speed cap)."""
        if not self.thumb_x:
            return self.x
        i = self.index
        x0 = self.thumb_x[i]
        if i + 1 < len(self.thumb_x):
            x1 = self.thumb_x[i + 1]
            t0 = float(self.start_times[i])
            t1 = float(self.start_times[i + 1])
            if t1 > t0:
                a = (elapsed_time - t0) / (t1 - t0)
                if a < 0:
                    a = 0
                elif a > 1:
                    a = 1
                return int(round(x0 + (x1 - x0) * a))
        return int(x0)

    def _update_from_index(self, elapsed_time, dt) -> None:
        if dt <= 0:
            return
        target = self._target_position(elapsed_time)
        dx = target - self.x
        if dx == 0:
            return  # already there
        max_step = self.max_speed * dt
        if abs(dx) <= max_step:
            self.x = target
        else:
            # move toward target but cap speed
            self.x += int(round(max_step if dx > 0 else -max_step))
        self._sync_rect()

    def current_indices(self) -> List[int]:
        """Return indices that share the same start_time as the current index."""
        if not self.start_times:
            return []
        current_start = self.start_times[self.index]
        indices = [self.index]
        j = self.index - 1
        while j >= 0 and self.start_times[j] == current_start:
            indices.append(j)
            j -= 1
        j = self.index + 1
        while j < len(self.start_times) and self.start_times[j] == current_start:
            indices.append(j)
            j += 1
        return indices
       