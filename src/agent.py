import random
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .db import Database


class ImplicitFeedback(Enum):
    LIKE = 1
    OKAY = 0
    SKIP = -1
