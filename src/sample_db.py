import numpy as np


songs_db = [
    {
        "id": 1,
        "title": "Blinding Lights - The Weeknd",
        "genre": "Synth-Pop (Енергійна, танцювальна)",
        "raw_features": {"danceability": 0.51, "energy": 0.73, "acousticness": 0.001, "valence": 0.65, "tempo": 171.0},
        "vector": np.array([0.395, 0.565, 0.001, 0.503, 0.625])
    },
    {
        "id": 2,
        "title": "Bad Guy - Billie Eilish",
        "genre": "Dark Pop (Танцювальна, низька енергія, електронна)",
        "raw_features": {"danceability": 0.70, "energy": 0.43, "acousticness": 0.32, "valence": 0.56, "tempo": 135.0},
        "vector": np.array([0.627, 0.385, 0.287, 0.502, 0.362])
    },
    {
        "id": 3,
        "title": "Come Away With Me - Norah Jones",
        "genre": "Jazz / Acoustic (Повільна, акустична, меланхолійна)",
        "raw_features": {"danceability": 0.42, "energy": 0.15, "acousticness": 0.88, "valence": 0.31, "tempo": 80.0},
        "vector": np.array([0.413, 0.147, 0.865, 0.305, 0.197])
    },
    {
        "id": 4,
        "title": "Rap God - Eminem",
        "genre": "Hip-Hop (Швидка, агресивна, не акустична)",
        "raw_features": {"danceability": 0.80, "energy": 0.85, "acousticness": 0.04, "valence": 0.63, "tempo": 148.0},
        "vector": np.array([0.551, 0.585, 0.027, 0.434, 0.450])
    }
]


quality_matrix = np.array([song["vector"] for song in songs_db])
