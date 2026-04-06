"""Human-in-the-loop annotation store.

Officials review model predictions and mark them as:
- accepted: "Yes, this is a real concern"
- rejected: "No, this is a known clearing / false alarm"

Annotations can feed back into model retraining as curated labels.

Design: one annotation per (lat, lon, run_id) — changing a verdict
updates the existing row rather than appending a duplicate.
"""

import json
import uuid
import aiosqlite
from datetime import datetime, timezone
from pathlib import Path


class AnnotationStore:
    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)

    async def init(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS annotations (
                    id TEXT PRIMARY KEY,
                    lat REAL NOT NULL,
                    lon REAL NOT NULL,
                    run_id TEXT NOT NULL,
                    prediction INTEGER NOT NULL,
                    risk_probability REAL,
                    verdict TEXT NOT NULL CHECK(verdict IN ('accept', 'reject')),
                    note TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            # Unique constraint: one verdict per cell per run
            await db.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_annotations_cell_run
                ON annotations (round(lat, 3), round(lon, 3), run_id)
            """)
            await db.commit()

    async def save_annotation(
        self,
        lat: float,
        lon: float,
        run_id: str,
        prediction: int,
        risk_probability: float,
        verdict: str,
        note: str = "",
    ) -> str:
        """Upsert an annotation — updates if one already exists for this cell+run."""
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            # Check for existing annotation at this cell
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT id FROM annotations
                   WHERE abs(lat - ?) < 0.001 AND abs(lon - ?) < 0.001 AND run_id = ?
                   LIMIT 1""",
                (lat, lon, run_id),
            )
            existing = await cursor.fetchone()

            if existing:
                # Update existing annotation
                annotation_id = existing["id"]
                await db.execute(
                    """UPDATE annotations
                       SET verdict = ?, note = ?, prediction = ?, risk_probability = ?, updated_at = ?
                       WHERE id = ?""",
                    (verdict, note, prediction, risk_probability, now, annotation_id),
                )
            else:
                # Insert new annotation
                annotation_id = str(uuid.uuid4())[:8]
                await db.execute(
                    """INSERT INTO annotations (id, lat, lon, run_id, prediction, risk_probability, verdict, note, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (annotation_id, lat, lon, run_id, prediction, risk_probability, verdict, note, now, now),
                )
            await db.commit()
        return annotation_id

    async def list_annotations(self, run_id: str | None = None) -> list[dict]:
        """List annotations — one per cell (the latest verdict)."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            if run_id:
                cursor = await db.execute(
                    "SELECT * FROM annotations WHERE run_id = ? ORDER BY updated_at DESC", (run_id,)
                )
            else:
                cursor = await db.execute("SELECT * FROM annotations ORDER BY updated_at DESC")
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def get_annotation_for_cell(self, lat: float, lon: float, run_id: str) -> dict | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT * FROM annotations
                   WHERE abs(lat - ?) < 0.001 AND abs(lon - ?) < 0.001 AND run_id = ?
                   LIMIT 1""",
                (lat, lon, run_id),
            )
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def get_stats(self, run_id: str) -> dict:
        """Count annotations by verdict — each cell counted once (latest verdict only)."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT verdict, COUNT(*) as count FROM annotations WHERE run_id = ? GROUP BY verdict",
                (run_id,),
            )
            rows = await cursor.fetchall()
            stats = {"accepted": 0, "rejected": 0, "total": 0}
            for row in rows:
                verdict = row[0]  # "accept" or "reject"
                count = row[1]
                stats[verdict + "ed"] = count
                stats["total"] += count
            return stats
