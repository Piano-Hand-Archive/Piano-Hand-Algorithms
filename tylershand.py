import pygame
import pandas as pd
from bisect import bisect_right
from piano_constants import *

class Hand:
    def __init__(
        self,
        position = (20, 20),
        size = (60, 40),
        color = (0, 255, 255), 
        alpha = 100,
    ) -> None:
        self.x, self.y = position
        self.w, self.h = size
        self.color = color
        self.alpha = max(0, min(255, alpha))
        self.rect = pygame.Rect(self.x, self.y, self.w, self.h)

        self.index = 0
        self.fingerData = pd.read_csv('fingering_plan_updated.csv')
        self.start_times = self.fingerData['start_time'].tolist()

        thumb = self.fingerData['thumb_pos']
        self.thumb_x = (thumb * (WIDTH - MARGIN) // NUM_WHITE_KEYS).astype(int).tolist()
        # Movement control
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
        # minimal draw: single ellipse with outline
        pygame.draw.ellipse(surface, self.color, self.rect)
        pygame.draw.ellipse(surface, (0, 0, 0), self.rect, width=2)

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
       