import random
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .db import Database


class ImplicitFeedback(Enum):
    LIKE = 1
    OKAY = 0
    SKIP = -1


@dataclass
class Agent:
    db: Database
    preferences: np.ndarray # vector of user's estimated preferences
    alpha: float

    def select_track(self):
        return self._select_greedy()

    def update_preference(self, track: np.ndarray, feedback: ImplicitFeedback):
        self.preferences += self.alpha*(
            feedback.value - self.preferences.dot(track))*track
        self.preferences /= np.linalg.norm(self.preferences) # L2 normalization

    def step(self):
        track = self.select_track()
        feedback = ImplicitFeedback(
            int(input(f"Select feedback for track \"{track["title"]}\":"))
        )
        self.update_preference(np.array(track["embedding"]), feedback)

    def _select_greedy(self):
        return self.db.search(
            self.preferences, 1)[0]
