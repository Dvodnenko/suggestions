import numpy as np

from .agent import Agent
from .db import Database
from .sr import SongRecord
from .utils import build_embedding


db = Database()
db.create_collection()

preferences = np.zeros(3) + 1/np.sqrt(3)

agent = Agent(
    db, preferences,
    0, 0.25
)

while True:
    agent.step()
