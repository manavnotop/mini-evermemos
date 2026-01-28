"""Mock storage clients for memory system testing."""

from typing import Any, Dict, List, Optional

from src.models import ConflictRecord, MemCell, MemScene, UserProfile


class MockMongoStorageClient:
    """Mock MongoDB client for testing."""

    def __init__(self, uri: str = ""):
        self.memcells: Dict[str, MemCell] = {}
        self.memscenes: Dict[str, MemScene] = {}
        self.conflicts: Dict[str, ConflictRecord] = {}
        self.profiles: Dict[str, UserProfile] = {}

        # Mock collection attributes for compatibility
        self.memcells_col = MockCollection()
        self.memscenes_col = MockCollection()
        self.conflicts_col = MockCollection()
        self.profiles_col = MockCollection()

    def add_memcell(self, memcell: MemCell) -> None:
        self.memcells[memcell.event_id] = memcell

    def get_memcell(self, event_id: str) -> Optional[MemCell]:
        return self.memcells.get(event_id)

    def get_all_memcells(self) -> List[MemCell]:
        return list(self.memcells.values())

    def get_memcells_by_ids(self, event_ids: List[str]) -> List[MemCell]:
        return [self.memcells[eid] for eid in event_ids if eid in self.memcells]

    def add_memscene(self, memscene: MemScene) -> None:
        self.memscenes[memscene.scene_id] = memscene

    def get_memscene(self, scene_id: str) -> Optional[MemScene]:
        return self.memscenes.get(scene_id)

    def get_all_memscenes(self) -> List[MemScene]:
        return list(self.memscenes.values())

    def add_conflict(self, conflict: ConflictRecord) -> None:
        self.conflicts[conflict.conflict_id] = conflict

    def get_conflict(self, conflict_id: str) -> Optional[ConflictRecord]:
        return self.conflicts.get(conflict_id)

    def get_all_conflicts(self) -> List[ConflictRecord]:
        return list(self.conflicts.values())

    def get_unresolved_conflicts(self) -> List[ConflictRecord]:
        return [c for c in self.conflicts.values() if not c.is_resolved]

    def save_user_profile(self, profile: UserProfile) -> None:
        self.profiles[profile.user_id] = profile

    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        return self.profiles.get(user_id)

    def clear(self) -> None:
        self.memcells.clear()
        self.memscenes.clear()
        self.conflicts.clear()
        self.profiles.clear()

    def __getattr__(self, name):
        # Handle raw collection access if needed
        if name == "memcells":
            return self.memcells_col
        if name == "memscenes":
            return self.memscenes_col
        if name == "conflicts":
            return self.conflicts_col
        if name == "profiles":
            return self.profiles_col
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )


class MockCollection:
    """Mock for PyMongo Collection."""

    def count_documents(self, filter: Dict) -> int:
        return 0

    def find(self, filter: Dict = None):
        return []

    def delete_many(self, filter: Dict) -> None:
        pass


class MockMilvusStorageClient:
    """Mock Milvus client for testing."""

    def __init__(self, host: str = "", port: str = ""):
        pass

    def add_memcell_embedding(self, memcell: MemCell, embedding: List[float]) -> None:
        pass

    def search_memcells(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        # Return empty or mocked results
        return []

    def delete_memcell(self, event_id: str) -> None:
        pass

    def add_memscene_embedding(
        self, memscene: MemScene, embedding: List[float]
    ) -> None:
        pass

    def search_memscenes(
        self, query_embedding: List[float], top_k: int = 3
    ) -> List[Dict[str, Any]]:
        return []

    def delete_memscene(self, scene_id: str) -> None:
        pass

    def clear(self) -> None:
        pass
