"""SQLite persistence for the React/Babylon web application."""

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
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with get_db_connection(database_path) as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                alloy TEXT NOT NULL,
                voxel_size REAL NOT NULL,
                refinement_level INTEGER NOT NULL,
                memory_mode TEXT NOT NULL,
                analysis_options TEXT NOT NULL,
                performance_options TEXT NOT NULL,
                solidification_params TEXT NOT NULL,
                ui_state TEXT NOT NULL,
                status TEXT DEFAULT 'draft',
                artifact_path TEXT,
                summary_json TEXT
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS parts (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
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
                visible INTEGER DEFAULT 1,
                is_open_top INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            )
            """
        )


def create_project(database_path: Path, data: Dict[str, Any]) -> str:
    project_id = str(uuid.uuid4())[:12]
    now_name = f"Solidification {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    with get_db_connection(database_path) as db:
        db.execute(
            """
            INSERT INTO projects (
                id, name, description, alloy, voxel_size, refinement_level,
                memory_mode, analysis_options, performance_options,
                solidification_params, ui_state
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                data.get("name", now_name),
                data.get("description", ""),
                data.get("alloy", "WCB"),
                float(data.get("voxel_size", 0.1)),
                int(data.get("refinement_level", 3)),
                data.get("memory_mode", "auto"),
                json.dumps(data.get("analysis_options", _default_analysis_options())),
                json.dumps(data.get("performance_options", _default_performance_options())),
                json.dumps(data.get("solidification_params", {})),
                json.dumps(data.get("ui_state", {})),
            ),
        )
    return project_id


def list_projects(database_path: Path) -> list[Dict[str, Any]]:
    with get_db_connection(database_path) as db:
        rows = db.execute(
            """
            SELECT id, name, description, created_at, updated_at, alloy,
                   voxel_size, refinement_level, memory_mode, status, summary_json
            FROM projects
            ORDER BY updated_at DESC
            """
        ).fetchall()
    projects = []
    for row in rows:
        item = dict(row)
        item["summary"] = _loads(item.pop("summary_json"), {})
        projects.append(item)
    return projects


def get_project(database_path: Path, project_id: str) -> Optional[Dict[str, Any]]:
    with get_db_connection(database_path) as db:
        row = db.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    if row is None:
        return None
    project = dict(row)
    project["analysis_options"] = _loads(project.get("analysis_options"), {})
    project["performance_options"] = _loads(project.get("performance_options"), {})
    project["solidification_params"] = _loads(project.get("solidification_params"), {})
    project["ui_state"] = _loads(project.get("ui_state"), {})
    project["summary"] = _loads(project.get("summary_json"), {})
    return project


def update_project(database_path: Path, project_id: str, data: Dict[str, Any]) -> None:
    current = get_project(database_path, project_id)
    if current is None:
        return
    with get_db_connection(database_path) as db:
        db.execute(
            """
            UPDATE projects
            SET name = ?, description = ?, alloy = ?, voxel_size = ?,
                refinement_level = ?, memory_mode = ?, analysis_options = ?,
                performance_options = ?, solidification_params = ?, ui_state = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                data.get("name", current["name"]),
                data.get("description", current.get("description", "")),
                data.get("alloy", current["alloy"]),
                float(data.get("voxel_size", current["voxel_size"])),
                int(data.get("refinement_level", current["refinement_level"])),
                data.get("memory_mode", current["memory_mode"]),
                json.dumps(data.get("analysis_options", current["analysis_options"])),
                json.dumps(data.get("performance_options", current["performance_options"])),
                json.dumps(data.get("solidification_params", current["solidification_params"])),
                json.dumps(data.get("ui_state", current["ui_state"])),
                project_id,
            ),
        )


def delete_project(database_path: Path, project_id: str) -> list[str]:
    with get_db_connection(database_path) as db:
        rows = db.execute("SELECT filepath FROM parts WHERE project_id = ?", (project_id,)).fetchall()
        filepaths = [str(row["filepath"]) for row in rows]
        artifact = db.execute("SELECT artifact_path FROM projects WHERE id = ?", (project_id,)).fetchone()
        if artifact is not None and artifact["artifact_path"]:
            filepaths.append(str(artifact["artifact_path"]))
        db.execute("DELETE FROM parts WHERE project_id = ?", (project_id,))
        db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    return filepaths


def add_part(database_path: Path, project_id: str, data: Dict[str, Any]) -> str:
    part_id = str(uuid.uuid4())[:8]
    with get_db_connection(database_path) as db:
        db.execute(
            """
            INSERT INTO parts (
                id, project_id, name, part_type, filepath,
                position_x, position_y, position_z,
                rotation_x, rotation_y, rotation_z,
                scale, visible, is_open_top
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                part_id,
                project_id,
                data.get("name", "Part"),
                data.get("part_type", "cast"),
                data.get("filepath", ""),
                float(data.get("position_x", 0.0)),
                float(data.get("position_y", 0.0)),
                float(data.get("position_z", 0.0)),
                float(data.get("rotation_x", 0.0)),
                float(data.get("rotation_y", 0.0)),
                float(data.get("rotation_z", 0.0)),
                float(data.get("scale", 1.0)),
                1 if data.get("visible", True) else 0,
                1 if data.get("is_open_top", False) else 0,
            ),
        )
        db.execute("UPDATE projects SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (project_id,))
    return part_id


def update_part(database_path: Path, project_id: str, part_id: str, data: Dict[str, Any]) -> None:
    current = get_part(database_path, project_id, part_id)
    if current is None:
        return
    with get_db_connection(database_path) as db:
        db.execute(
            """
            UPDATE parts
            SET name = ?, part_type = ?, filepath = ?,
                position_x = ?, position_y = ?, position_z = ?,
                rotation_x = ?, rotation_y = ?, rotation_z = ?,
                scale = ?, visible = ?, is_open_top = ?
            WHERE id = ? AND project_id = ?
            """,
            (
                data.get("name", current["name"]),
                data.get("part_type", current["part_type"]),
                data.get("filepath", current["filepath"]),
                float(data.get("position_x", current["position_x"])),
                float(data.get("position_y", current["position_y"])),
                float(data.get("position_z", current["position_z"])),
                float(data.get("rotation_x", current["rotation_x"])),
                float(data.get("rotation_y", current["rotation_y"])),
                float(data.get("rotation_z", current["rotation_z"])),
                float(data.get("scale", current["scale"])),
                1 if data.get("visible", bool(current["visible"])) else 0,
                1 if data.get("is_open_top", bool(current["is_open_top"])) else 0,
                part_id,
                project_id,
            ),
        )


def delete_part(database_path: Path, project_id: str, part_id: str) -> Optional[str]:
    with get_db_connection(database_path) as db:
        row = db.execute(
            "SELECT filepath FROM parts WHERE id = ? AND project_id = ?",
            (part_id, project_id),
        ).fetchone()
        db.execute("DELETE FROM parts WHERE id = ? AND project_id = ?", (part_id, project_id))
    return str(row["filepath"]) if row else None


def get_part(database_path: Path, project_id: str, part_id: str) -> Optional[Dict[str, Any]]:
    with get_db_connection(database_path) as db:
        row = db.execute(
            "SELECT * FROM parts WHERE id = ? AND project_id = ?",
            (part_id, project_id),
        ).fetchone()
    return _normalize_part(dict(row)) if row else None


def get_parts(database_path: Path, project_id: str) -> list[Dict[str, Any]]:
    with get_db_connection(database_path) as db:
        rows = db.execute(
            "SELECT * FROM parts WHERE project_id = ? ORDER BY created_at ASC",
            (project_id,),
        ).fetchall()
    return [_normalize_part(dict(row)) for row in rows]


def get_project_and_parts(
    database_path: Path,
    project_id: str,
) -> Tuple[Optional[Dict[str, Any]], list[Dict[str, Any]]]:
    return get_project(database_path, project_id), get_parts(database_path, project_id)


def mark_project_status(database_path: Path, project_id: str, status: str) -> None:
    with get_db_connection(database_path) as db:
        db.execute(
            "UPDATE projects SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (status, project_id),
        )


def mark_project_completed(
    database_path: Path,
    project_id: str,
    summary: Dict[str, Any],
    artifact_path: str,
) -> None:
    with get_db_connection(database_path) as db:
        db.execute(
            """
            UPDATE projects
            SET status = 'completed', artifact_path = ?, summary_json = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (artifact_path, json.dumps(summary), project_id),
        )


def get_artifact_path(database_path: Path, project_id: str) -> Optional[str]:
    with get_db_connection(database_path) as db:
        row = db.execute("SELECT artifact_path FROM projects WHERE id = ?", (project_id,)).fetchone()
    if row is None or row["artifact_path"] is None:
        return None
    return str(row["artifact_path"])


def _loads(value: Any, default: Dict[str, Any]) -> Dict[str, Any]:
    if not value:
        return dict(default)
    try:
        return cast(Dict[str, Any], json.loads(str(value)))
    except Exception:
        return dict(default)


def _normalize_part(row: Dict[str, Any]) -> Dict[str, Any]:
    row["visible"] = bool(row.get("visible", 1))
    row["is_open_top"] = bool(row.get("is_open_top", 0))
    return row


def _default_analysis_options() -> Dict[str, Any]:
    return {
        "shrinkage": True,
        "hotspot": True,
        "niyama": False,
        "hot_tear": False,
        "sand_burn_in": False,
        "fraction_liquid": True,
        "feeding_path": False,
        "stress_analysis": False,
        "microstructure": False,
        "sand_restraint_index": 0.5,
    }


def _default_performance_options() -> Dict[str, Any]:
    return {
        "use_gpu": False,
        "use_incremental": False,
        "use_chunked_voxelization": False,
        "part_assignment_mode": "auto",
        "part_assignment_workers": 0,
        "part_assignment_batch_size": 2500,
    }

