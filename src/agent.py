from dataclasses import dataclass
from enum import Enum

import numpy as np


class ImplicitFeedback(Enum):
    LIKE = 1
    OKAY = 0
    SKIP = -1


@dataclass(frozen=True)
class Environment:
    n_songs: int
    songs_values: np.ndarray

    def pick(self, song: int) -> float:
        value = np.random.normal(self.actual_values[song], 1)
        return value
