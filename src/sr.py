from dataclasses import dataclass


@dataclass
class SongRecord:

    id: int
    title: str
    main_artist: str
    embedding: list[float]
    danceability: float
    energy: float
    valence: float
    genre: str = "unknown"
