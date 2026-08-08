import numpy as np


def minmax(x: float, x_max: float, x_min: float) -> float:
    return (x - x_min) / (x_max - x_min + 1e-9)

def build_embedding(danceability: float, energy: float, valence: float) -> list[float]:
    vec = np.array([danceability, energy, valence], dtype=np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-9) # L2 normalization
    return vec.tolist()
