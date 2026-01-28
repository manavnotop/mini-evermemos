from typing import Any, Dict, List, Optional

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from ..models import MemCell, MemScene


class MilvusStorageClient:
    """
    Milvus client for vector storage and retrieval.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        dim: int = 1536,  # Default to OpenAI embedding dimension
    ):
        """
        Initialize the Milvus client.

        Args:
            host: Milvus host
            port: Milvus port
            dim: Dimension of embeddings
        """
        self.host = host
        self.port = port
        self.dim = dim
        self.alias = "default"

        self._connect()
        self._setup_collections()

    def _connect(self) -> None:
        """Connect to Milvus instance."""
        try:
            connections.connect(alias=self.alias, host=self.host, port=self.port)
        except Exception:
            pass

    def _setup_collections(self) -> None:
        """Create collections if they don't exist."""
        # MemCell Collection
        if not utility.has_collection("memcells"):
            self._create_memcell_collection()

        # MemScene Collection
        if not utility.has_collection("memscenes"):
            self._create_memscene_collection()

        self.memcells_col = Collection("memcells")
        self.memscenes_col = Collection("memscenes")

        # Load collections into memory
        self.memcells_col.load()
        self.memscenes_col.load()

    def _create_memcell_collection(self) -> None:
        """Define schema and create memcells collection."""
        fields = [
            FieldSchema(
                name="event_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True
            ),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="scene_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="timestamp", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields, "MemCell vector storage")
        col = Collection("memcells", schema)

        # Create HNSW index
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64},
        }
        col.create_index("embedding", index_params)

    def _create_memscene_collection(self) -> None:
        """Define schema and create memscenes collection."""
        fields = [
            FieldSchema(
                name="scene_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True
            ),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="theme", dtype=DataType.VARCHAR, max_length=64),
        ]
        schema = CollectionSchema(fields, "MemScene vector storage")
        col = Collection("memscenes", schema)

        # Create HNSW index
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64},
        }
        col.create_index("embedding", index_params)

    # --- MemCell Operations ---

    def add_memcell_embedding(self, memcell: MemCell, embedding: List[float]) -> None:
        """Insert a MemCell embedding."""
        data = [
            [memcell.event_id],
            [embedding],
            [memcell.scene_id or ""],
            [int(memcell.timestamp.timestamp())],
        ]
        self.memcells_col.insert(data)
        # Flush meant for immediate consistency in tests, might want to batch in prod
        self.memcells_col.flush()

    def search_memcells(
        self, query_embedding: List[float], top_k: int = 5, expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search MemCells by vector similarity.

        Args:
            query_embedding: Query vector
            top_k: Number of results
            expr: Filtering expression (e.g. "timestamp > 1000")

        Returns:
            List of dicts with 'event_id' and 'score'
        """
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}

        results = self.memcells_col.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["scene_id", "timestamp"],
        )

        hits = []
        for hit in results[0]:
            hits.append(
                {
                    "event_id": hit.id,
                    "score": hit.score,
                    "scene_id": hit.entity.get("scene_id"),
                    "timestamp": hit.entity.get("timestamp"),
                }
            )
        return hits

    def delete_memcell(self, event_id: str) -> None:
        """Delete a MemCell by ID."""
        expr = f'event_id == "{event_id}"'
        self.memcells_col.delete(expr)

    # --- MemScene Operations ---

    def add_memscene_embedding(
        self, memscene: MemScene, embedding: List[float]
    ) -> None:
        """Insert a MemScene embedding."""
        data = [
            [memscene.scene_id],
            [embedding],
            [memscene.theme],
        ]
        self.memscenes_col.insert(data)
        self.memscenes_col.flush()

    def search_memscenes(
        self, query_embedding: List[float], top_k: int = 3, expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search MemScenes by vector similarity."""
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}

        results = self.memscenes_col.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["theme"],
        )

        hits = []
        for hit in results[0]:
            hits.append(
                {
                    "scene_id": hit.id,
                    "score": hit.score,
                    "theme": hit.entity.get("theme"),
                }
            )
        return hits

    def delete_memscene(self, scene_id: str) -> None:
        """Delete a MemScene by ID."""
        expr = f'scene_id == "{scene_id}"'
        self.memscenes_col.delete(expr)

    def clear(self) -> None:
        """Clear all data from collections."""
        # Dropping and recreating is often faster/cleaner for 'clear'
        utility.drop_collection("memcells")
        utility.drop_collection("memscenes")
        self._setup_collections()
