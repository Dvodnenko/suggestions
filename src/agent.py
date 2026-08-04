import random
from dataclasses import dataclass

import numpy as np

from .db import songs_db, quality_matrix


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
        return songs_db[np.argmax(quality_matrix.dot(self.preferences))]
