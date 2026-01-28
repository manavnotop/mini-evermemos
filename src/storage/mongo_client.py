from typing import List, Optional

from pymongo import MongoClient

from ..models import ConflictRecord, MemCell, MemScene, UserProfile


class MongoStorageClient:
    """
    MongoDB client for structured memory storage.
    Handles MemCells, MemScenes, ConflictRecords, and UserProfiles.
    """

    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        db_name: str = "evermemos",
    ):
        """
        Initialize the MongoDB client.

        Args:
            uri: MongoDB connection URI
            db_name: Database name
        """
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

        # Collections
        self.memcells = self.db.memcells
        self.memscenes = self.db.memscenes
        self.conflicts = self.db.conflicts
        self.profiles = self.db.profiles

        self._setup_indexes()

    def _setup_indexes(self) -> None:
        """Create necessary indexes for performance and uniqueness."""
        # MemCells
        self.memcells.create_index("event_id", unique=True)
        self.memcells.create_index("scene_id")
        self.memcells.create_index("timestamp")

        # MemScenes
        self.memscenes.create_index("scene_id", unique=True)
        self.memscenes.create_index("theme")

        # Conflicts
        self.conflicts.create_index("conflict_id", unique=True)

        # Profiles
        self.profiles.create_index("user_id", unique=True)

    # --- MemCell Operations ---

    def add_memcell(self, memcell: MemCell) -> None:
        """Add or update a MemCell."""
        self.memcells.replace_one(
            {"event_id": memcell.event_id},
            memcell.to_dict(),
            upsert=True,
        )

    def get_memcell(self, event_id: str) -> Optional[MemCell]:
        """Retrieve a MemCell by ID."""
        data = self.memcells.find_one({"event_id": event_id})
        if data:
            # MongoDB adds _id, remove it before converting
            data.pop("_id", None)
            return MemCell.from_dict(data)
        return None

    def get_all_memcells(self) -> List[MemCell]:
        """Retrieve all MemCells."""
        cursor = self.memcells.find()
        return [
            MemCell.from_dict({k: v for k, v in doc.items() if k != "_id"})
            for doc in cursor
        ]

    def get_memcells_by_ids(self, event_ids: List[str]) -> List[MemCell]:
        """Retrieve multiple MemCells by their IDs."""
        cursor = self.memcells.find({"event_id": {"$in": event_ids}})
        return [
            MemCell.from_dict({k: v for k, v in doc.items() if k != "_id"})
            for doc in cursor
        ]

    def get_memcells_by_scene(self, scene_id: str) -> List[MemCell]:
        """Retrieve all MemCells associated with a specific scene."""
        # Implemented via MemoryStore.get_memcells_by_scene which fetches scene first
        # and then retrieves cells by IDs.
        return []

    # --- MemScene Operations ---

    def add_memscene(self, memscene: MemScene) -> None:
        """Add or update a MemScene."""
        self.memscenes.replace_one(
            {"scene_id": memscene.scene_id},
            memscene.to_dict(),
            upsert=True,
        )

    def get_memscene(self, scene_id: str) -> Optional[MemScene]:
        """Retrieve a MemScene by ID."""
        data = self.memscenes.find_one({"scene_id": scene_id})
        if data:
            data.pop("_id", None)
            return MemScene.from_dict(data)
        return None

    def get_all_memscenes(self) -> List[MemScene]:
        """Retrieve all MemScenes."""
        cursor = self.memscenes.find()
        return [
            MemScene.from_dict({k: v for k, v in doc.items() if k != "_id"})
            for doc in cursor
        ]

    # --- Conflict Operations ---

    def add_conflict(self, conflict: ConflictRecord) -> None:
        """Add or update a ConflictRecord."""
        self.conflicts.replace_one(
            {"conflict_id": conflict.conflict_id},
            conflict.to_dict(),
            upsert=True,
        )

    def get_conflict(self, conflict_id: str) -> Optional[ConflictRecord]:
        """Retrieve a ConflictRecord by ID."""
        data = self.conflicts.find_one({"conflict_id": conflict_id})
        if data:
            data.pop("_id", None)
            return ConflictRecord.from_dict(data)
        return None

    def get_all_conflicts(self) -> List[ConflictRecord]:
        """Retrieve all ConflictRecords."""
        cursor = self.conflicts.find()
        return [
            ConflictRecord.from_dict({k: v for k, v in doc.items() if k != "_id"})
            for doc in cursor
        ]

    def get_unresolved_conflicts(self) -> List[ConflictRecord]:
        """Retrieve all unresolved conflicts."""
        cursor = self.conflicts.find({"is_resolved": False})
        return [
            ConflictRecord.from_dict({k: v for k, v in doc.items() if k != "_id"})
            for doc in cursor
        ]

    # --- Profile Operations ---

    def save_user_profile(self, profile: UserProfile) -> None:
        """Save or update a UserProfile."""
        self.profiles.replace_one(
            {"user_id": profile.user_id},
            profile.to_dict(),
            upsert=True,
        )

    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Retrieve a UserProfile by ID."""
        data = self.profiles.find_one({"user_id": user_id})
        if data:
            data.pop("_id", None)
            return UserProfile.from_dict(data)
        return None

    def clear(self) -> None:
        """Clear all collections (for testing/reset)."""
        self.memcells.delete_many({})
        self.memscenes.delete_many({})
        self.conflicts.delete_many({})
        self.profiles.delete_many({})
