import random
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .db import songs_db, quality_matrix


class ImplicitFeedback(Enum):
    LIKE = 1
    OKAY = 0
    SKIP = -1


@dataclass
class Agent:
    preferences: np.ndarray # vector of user's estimated preferences
    epsilon: float
    alpha: float

    @property
    def pref_length(self):
        return np.sqrt((self.preferences*self.preferences).sum())

    def select_track(self):
        if random.uniform(0, 1) <= self.epsilon:
            return songs_db[random.randint(0, len(songs_db)-1)] # random track
        return self._select_greedy()

    def update_preference(self, track: np.ndarray, feedback: ImplicitFeedback):
        self.preferences += self.alpha*(
            feedback.value - self.preferences.dot(track))*track
        self.preferences /= self.pref_length # L2 normalization

    def step(self):
        track = self.select_track()
        feedback = ImplicitFeedback(
            int(input(f"Select feedback for track \"{track["title"]}\":"))
        )
        self.update_preference(track["vector_x"], feedback)

    def _select_greedy(self):
        return songs_db[np.argmax(quality_matrix.dot(self.preferences))]
