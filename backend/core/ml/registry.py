"""Model registry and run history backed by SQLite."""

import json
import uuid
import aiosqlite
from datetime import datetime, timezone
from pathlib import Path


class RunRegistry:
    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)

    async def init(self):
        """Create tables if they don't exist."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    params TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    feature_names TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    test_indices TEXT DEFAULT NULL,
                    dataset_hash TEXT DEFAULT NULL
                )
            """)
            # Add columns if migrating from older schema
            try:
                await db.execute("ALTER TABLE runs ADD COLUMN test_indices TEXT DEFAULT NULL")
            except Exception:
                pass  # column already exists
            try:
                await db.execute("ALTER TABLE runs ADD COLUMN dataset_hash TEXT DEFAULT NULL")
            except Exception:
                pass  # column already exists
            await db.commit()

    async def save_run(
        self,
        params: dict,
        metrics: dict,
        feature_names: list[str],
        model_path: str,
        test_indices: list[int] | None = None,
        dataset_hash: str | None = None,
    ) -> str:
        """Save a training run with full reproducibility metadata. Returns run_id."""
        run_id = str(uuid.uuid4())[:8]
        created_at = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO runs (run_id, created_at, params, metrics, feature_names,
                   model_path, test_indices, dataset_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, created_at, json.dumps(params), json.dumps(metrics),
                 json.dumps(feature_names), model_path,
                 json.dumps(test_indices) if test_indices is not None else None,
                 dataset_hash),
            )
            await db.commit()
        return run_id

    async def list_runs(self) -> list[dict]:
        """List all runs, newest first."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM runs ORDER BY created_at DESC")
            rows = await cursor.fetchall()
            return [self._row_to_dict(r) for r in rows]

    async def get_run(self, run_id: str) -> dict | None:
        """Get a single run by ID."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            row = await cursor.fetchone()
            return self._row_to_dict(row) if row else None

    def _row_to_dict(self, row) -> dict:
        d = {
            "run_id": row["run_id"],
            "created_at": row["created_at"],
            "params": json.loads(row["params"]),
            "metrics": json.loads(row["metrics"]),
            "feature_names": json.loads(row["feature_names"]),
            "model_path": row["model_path"],
        }
        # Optional fields (may be NULL for older runs)
        try:
            d["test_indices"] = json.loads(row["test_indices"]) if row["test_indices"] else None
        except (KeyError, TypeError):
            d["test_indices"] = None
        try:
            d["dataset_hash"] = row["dataset_hash"]
        except (KeyError, TypeError):
            d["dataset_hash"] = None
        return d
