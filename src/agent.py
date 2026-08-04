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
