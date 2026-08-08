import numpy as np

from .agent import Agent
from .db import Database
from .sr import SongRecord
from .utils import build_embedding


db = Database()
db.create_collection()

song1 = SongRecord(
    1, "Bankroll", "Lil Uzi Vert", 
    build_embedding(0.6, 0.8, 0.7), 0.6, 0.8, 0.7)
song2 = SongRecord(
    2, "Love Thy Enemies", "Future", 
    build_embedding(0.3, 0.45, 0.2), 0.3, 0.45, 0.2)

db.insert_songs([song1, song2])
print(db.search([0.3, 0.45, 0.2], top_k=1))
