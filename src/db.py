from typing import Any, Optional

import numpy as np
from pymilvus import MilvusClient, DataType

from .sr import SongRecord


class Database:

    def __init__(
            self, 
            db_path: str = "./data/songs.db",
            collection_name: str = "songs",
            dimensions: int = 3,
            metric_type: str = "COSINE"
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.dimensions = dimensions
        self.metric_type = metric_type
        self.client = MilvusClient(uri=db_path)

    @property
    def row_count(self) -> int:
        "number of entries (songs) in the collection"
        return self.client.get_collection_stats(
            self.collection_name)["row_count"]

    def create_collection(self, drop_if_exists: bool = False):
        if self.client.has_collection(self.collection_name):
            if drop_if_exists:
                self.client.drop_collection(self.collection_name)
            else:
                return
 
        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("title", DataType.VARCHAR, max_length=512)
        schema.add_field("main_artist", DataType.VARCHAR, max_length=256)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dimensions)
        schema.add_field("danceability", DataType.FLOAT)
        schema.add_field("energy", DataType.FLOAT)
        schema.add_field("valence", DataType.FLOAT)
        schema.add_field("genre", DataType.VARCHAR, max_length=128)
 
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type=self.metric_type,
            params={"M": 16, "efConstruction": 200},
        )
 
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def insert_songs(self, songs: list[SongRecord]) -> dict[str, Any]:
        rows = [self._record_to_row(s) for s in songs]
        return self.client.insert(collection_name=self.collection_name, data=rows)

    @staticmethod
    def _record_to_row(s: SongRecord) -> dict[str, Any]:
        return {
            "id": s.id,
            "title": s.title,
            "main_artist": s.main_artist,
            "embedding": s.embedding,
            "danceability": s.danceability,
            "energy": s.energy,
            "valence": s.valence,
            "genre": s.genre,
        }

    def search(
        self,
        query_vector: np.ndarray | list[float],
        top_k: int = 20,
        genre: Optional[str] = None,
        exclude_ids: Optional[list[int]] = None,
        min_popularity: Optional[int] = None,
        ef: int = 64,
    ) -> list[dict[str, Any]]:
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        expr_parts = []
        if genre:
            expr_parts.append(f'genre == "{genre}"')
        if min_popularity is not None:
            expr_parts.append(f"popularity >= {min_popularity}")
        if exclude_ids:
            expr_parts.append(f"id not in {exclude_ids}")
        expr = " and ".join(expr_parts) if expr_parts else None

        self.client.load_collection(self.collection_name)

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=top_k,
            filter=expr,
            search_params={"metric_type": self.metric_type, "params": {"ef": ef}},
            output_fields=[
                "title", "main_artist", "genre", "embedding",
                "danceability", "energy", "valence",
            ],
        )

        return results[0]

    def get(self, ids: list[int]):
        self.client.load_collection(self.collection_name)
        return self.client.get(
            collection_name=self.collection_name,
            ids=ids,
            output_fields=[
                "title", "main_artist", "genre", "embedding",
                "danceability", "energy", "valence",
            ],
        )

    def close(self) -> None:
        self.client.close()
