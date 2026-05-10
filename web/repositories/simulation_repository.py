"""SQLite data access layer for web simulations."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple, cast
import json
import sqlite3
import uuid


def get_db(database_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(database_path))
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db_connection(database_path: Path) -> Iterator[sqlite3.Connection]:
    conn = get_db(database_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(database_path: Path) -> None:
    with get_db_connection(database_path) as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS simulations (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                alloy TEXT NOT NULL,
                voxel_size REAL NOT NULL,
                refinement_level INTEGER NOT NULL,
                analysis_options TEXT,
                status TEXT DEFAULT 'draft',
                total_volume REAL,
                defect_volume REAL,
                voxel_count INTEGER
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS simulation_parts (
                id TEXT PRIMARY KEY,
                simulation_id TEXT NOT NULL,
                name TEXT NOT NULL,
                part_type TEXT NOT NULL,
                filepath TEXT NOT NULL,
                position_x REAL DEFAULT 0,
                position_y REAL DEFAULT 0,
                position_z REAL DEFAULT 0,
                rotation_x REAL DEFAULT 0,
                rotation_y REAL DEFAULT 0,
                rotation_z REAL DEFAULT 0,
                scale REAL DEFAULT 1,
                FOREIGN KEY (simulation_id) REFERENCES simulations(id) ON DELETE CASCADE
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS simulation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulation_id TEXT NOT NULL,
                result_type TEXT NOT NULL,
                data_json TEXT,
                voxel_data_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (simulation_id) REFERENCES simulations(id) ON DELETE CASCADE
            )
            """
        )


def list_simulations(database_path: Path) -> list[Dict[str, Any]]:
    with get_db_connection(database_path) as db:
        rows = db.execute(
            """
            SELECT id, name, description, created_at, updated_at,
                   alloy, voxel_size, status, total_volume, defect_volume, voxel_count
            FROM simulations
            ORDER BY updated_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def create_simulation(database_path: Path, data: Dict[str, Any]) -> str:
    sim_id = str(uuid.uuid4())[:12]
    with get_db_connection(database_path) as db:
        db.execute(
            """
            INSERT INTO simulations (id, name, description, alloy, voxel_size, refinement_level, analysis_options)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sim_id,
                data.get("name", f'Simulation {datetime.now().strftime("%Y-%m-%d %H:%M")}'),
                data.get("description", ""),
                data.get("alloy", "WCB"),
                data.get("voxel_size", 0.1),
                data.get("refinement_level", 4),
                json.dumps(data.get("analysis_options", {})),
            ),
        )
    return sim_id


def get_simulation(database_path: Path, sim_id: str) -> Optional[Dict[str, Any]]:
    with get_db_connection(database_path) as db:
        row = db.execute("SELECT * FROM simulations WHERE id = ?", (sim_id,)).fetchone()
    return dict(row) if row else None


def get_simulation_parts(database_path: Path, sim_id: str) -> list[Dict[str, Any]]:
    with get_db_connection(database_path) as db:
        rows = db.execute("SELECT * FROM simulation_parts WHERE simulation_id = ?", (sim_id,)).fetchall()
    return [dict(row) for row in rows]


def get_simulation_results(database_path: Path, sim_id: str) -> list[Dict[str, Any]]:
    with get_db_connection(database_path) as db:
        rows = db.execute("SELECT * FROM simulation_results WHERE simulation_id = ?", (sim_id,)).fetchall()
    return [dict(row) for row in rows]


def get_simulation_and_parts(database_path: Path, sim_id: str) -> Tuple[Optional[Dict[str, Any]], list[Dict[str, Any]]]:
    return get_simulation(database_path, sim_id), get_simulation_parts(database_path, sim_id)


def update_simulation(database_path: Path, sim_id: str, data: Dict[str, Any]) -> None:
    with get_db_connection(database_path) as db:
        db.execute(
            """
            UPDATE simulations
            SET name = ?, description = ?, alloy = ?, voxel_size = ?,
                refinement_level = ?, analysis_options = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                data.get("name"),
                data.get("description"),
                data.get("alloy"),
                data.get("voxel_size"),
                data.get("refinement_level"),
                json.dumps(data.get("analysis_options", {})),
                sim_id,
            ),
        )


def delete_simulation(database_path: Path, sim_id: str) -> list[str]:
    with get_db_connection(database_path) as db:
        rows = db.execute("SELECT filepath FROM simulation_parts WHERE simulation_id = ?", (sim_id,)).fetchall()
        filepaths = [str(row["filepath"]) for row in rows]
        db.execute("DELETE FROM simulation_results WHERE simulation_id = ?", (sim_id,))
        db.execute("DELETE FROM simulation_parts WHERE simulation_id = ?", (sim_id,))
        db.execute("DELETE FROM simulations WHERE id = ?", (sim_id,))
    return filepaths


def add_part(database_path: Path, sim_id: str, data: Dict[str, Any]) -> str:
    part_id = str(uuid.uuid4())[:8]
    with get_db_connection(database_path) as db:
        db.execute(
            """
            INSERT INTO simulation_parts
            (id, simulation_id, name, part_type, filepath, position_x, position_y, position_z)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                part_id,
                sim_id,
                data.get("name", "Part"),
                data.get("part_type", "cast"),
                data.get("filepath", ""),
                data.get("position_x", 0),
                data.get("position_y", 0),
                data.get("position_z", 0),
            ),
        )
        db.execute("UPDATE simulations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (sim_id,))
    return part_id


def update_part(database_path: Path, sim_id: str, part_id: str, data: Dict[str, Any]) -> None:
    with get_db_connection(database_path) as db:
        db.execute(
            """
            UPDATE simulation_parts
            SET name = ?, part_type = ?, position_x = ?, position_y = ?, position_z = ?,
                rotation_x = ?, rotation_y = ?, rotation_z = ?, scale = ?
            WHERE id = ? AND simulation_id = ?
            """,
            (
                data.get("name"),
                data.get("part_type"),
                data.get("position_x", 0),
                data.get("position_y", 0),
                data.get("position_z", 0),
                data.get("rotation_x", 0),
                data.get("rotation_y", 0),
                data.get("rotation_z", 0),
                data.get("scale", 1),
                part_id,
                sim_id,
            ),
        )


def delete_part(database_path: Path, sim_id: str, part_id: str) -> Optional[str]:
    with get_db_connection(database_path) as db:
        row = db.execute("SELECT filepath FROM simulation_parts WHERE id = ?", (part_id,)).fetchone()
        db.execute("DELETE FROM simulation_parts WHERE id = ? AND simulation_id = ?", (part_id, sim_id))
    return str(row["filepath"]) if row else None


def mark_simulation_status(database_path: Path, sim_id: str, status: str) -> None:
    with get_db_connection(database_path) as db:
        db.execute("UPDATE simulations SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (status, sim_id))


def mark_simulation_completed(database_path: Path, sim_id: str, result_data: Dict[str, Any]) -> None:
    with get_db_connection(database_path) as db:
        db.execute(
            """
            UPDATE simulations
            SET status = 'completed',
                total_volume = ?,
                defect_volume = ?,
                voxel_count = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                result_data["total_volume"],
                result_data.get("shrinkage_stats", {}).get("affected_count", 0),
                result_data["voxel_count"],
                sim_id,
            ),
        )
        db.execute(
            """
            INSERT INTO simulation_results (simulation_id, result_type, data_json)
            VALUES (?, 'summary', ?)
            """,
            (sim_id, json.dumps(result_data)),
        )


def get_simulation_status(database_path: Path, sim_id: str) -> Optional[str]:
    with get_db_connection(database_path) as db:
        row = db.execute("SELECT status FROM simulations WHERE id = ?", (sim_id,)).fetchone()
    if row is None:
        return None
    return str(row["status"])


def get_summary_result(database_path: Path, sim_id: str) -> Optional[Dict[str, Any]]:
    with get_db_connection(database_path) as db:
        row = db.execute(
            "SELECT data_json FROM simulation_results WHERE simulation_id = ? AND result_type = ?",
            (sim_id, "summary"),
        ).fetchone()
    if row is None or row["data_json"] is None:
        return None
    return cast(Dict[str, Any], json.loads(str(row["data_json"])))
