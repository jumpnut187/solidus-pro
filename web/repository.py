from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple, cast
import json
import os
import re
import sqlite3
import uuid

_UPLOAD_PREFIX_RE = re.compile(r"^[0-9a-f]{8}(?:_b\d+)?_")

DEFAULT_VOXEL_SIZE_IN = 0.1

class UnsafePartFileError(ValueError):
    pass

def resolve_upload_path(filepath: str, upload_root: Path) -> str:
    root = Path(upload_root).resolve()
    candidate = (root / filepath).resolve() if not os.path.isabs(filepath) else Path(filepath).resolve()
    if candidate != root and root not in candidate.parents:
        raise UnsafePartFileError(
            "A part file must live in the server's upload folder. Upload the CAD file instead of "
            "referencing a path."
        )
    return str(candidate)

def strip_upload_prefix(stem: str) -> str:
    cleaned = stem
    while True:
        stripped = _UPLOAD_PREFIX_RE.sub("", cleaned, count=1)
        if stripped == cleaned:
            break
        cleaned = stripped
    return cleaned or stem

def get_db(database_path: "str | os.PathLike") -> sqlite3.Connection:
    conn = sqlite3.connect(str(database_path))
    conn.row_factory = sqlite3.Row
    return conn

@contextmanager
def get_db_connection(database_path: "str | os.PathLike") -> Iterator[sqlite3.Connection]:
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
                riser_sleeve TEXT DEFAULT 'insulating',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                feature_json TEXT,
                base_filepath TEXT,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            )
            """
        )
        _ensure_column(db, "parts", "riser_sleeve", "TEXT DEFAULT 'insulating'")
        _ensure_column(db, "parts", "feature_json", "TEXT")
        _ensure_column(db, "parts", "base_filepath", "TEXT")
        _backfill_part_names(db)

RISER_SLEEVES = ("insulating", "adiabatic", "none")

def _riser_sleeve(value: Any) -> str:
    text = str(value).strip().lower() if value is not None else ""
    return text if text in RISER_SLEEVES else "insulating"

def _ensure_column(db: sqlite3.Connection, table: str, column: str, declaration: str) -> None:
    columns = {str(row["name"]) for row in db.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

def _backfill_part_names(db: sqlite3.Connection) -> None:
    try:
        rows = db.execute("SELECT id, name FROM parts").fetchall()
    except sqlite3.Error:
        return
    for row in rows:
        name = str(row["name"] or "")
        cleaned = strip_upload_prefix(name)
        if cleaned != name:
            db.execute("UPDATE parts SET name = ? WHERE id = ?", (cleaned, row["id"]))

def create_project(database_path: Path, data: Dict[str, Any]) -> str:
    project_id = str(uuid.uuid4())[:12]
    now_name = f"Solidification {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    ui_state = _merge_pour_location(dict(data.get("ui_state") or {}), data)
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
                float(data.get("voxel_size", DEFAULT_VOXEL_SIZE_IN)),
                int(data.get("refinement_level", 3)),
                data.get("memory_mode", "auto"),
                json.dumps(data.get("analysis_options", _default_analysis_options())),
                json.dumps(data.get("performance_options", _default_performance_options())),
                json.dumps(data.get("solidification_params", {})),
                json.dumps(ui_state),
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
    project["pour_location"] = project["ui_state"].get("pour_location")
    return project

ROUTE_OWNED_UI_STATE_KEYS = ("reference_planes", "cut_features", "parting_line")

def update_project(database_path: Path, project_id: str, data: Dict[str, Any]) -> None:
    current = get_project(database_path, project_id)
    if current is None:
        return
    ui_state = _merge_pour_location(dict(data.get("ui_state", current["ui_state"]) or {}), data)
    for key in ROUTE_OWNED_UI_STATE_KEYS:
        if key not in ui_state and key in current["ui_state"]:
            ui_state[key] = current["ui_state"][key]
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
                json.dumps(ui_state),
                project_id,
            ),
        )

def set_ui_state_key(database_path: Path, project_id: str, key: str, value: Any) -> None:
    with get_db_connection(database_path) as db:
        row = db.execute("SELECT ui_state FROM projects WHERE id = ?", (project_id,)).fetchone()
        if row is None:
            return
        ui_state = _loads(row["ui_state"], {})
        ui_state[key] = value
        db.execute(
            "UPDATE projects SET ui_state = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (json.dumps(ui_state), project_id),
        )

def delete_ui_state_key(database_path: Path, project_id: str, key: str) -> None:
    with get_db_connection(database_path) as db:
        row = db.execute("SELECT ui_state FROM projects WHERE id = ?", (project_id,)).fetchone()
        if row is None:
            return
        ui_state = _loads(row["ui_state"], {})
        if key not in ui_state:
            return
        del ui_state[key]
        db.execute(
            "UPDATE projects SET ui_state = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (json.dumps(ui_state), project_id),
        )

def rename_project(database_path: Path, project_id: str, name: str) -> bool:
    with get_db_connection(database_path) as db:
        cursor = db.execute(
            "UPDATE projects SET name = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (name, project_id),
        )
        return cursor.rowcount > 0

def duplicate_project(
    database_path: Path,
    project_id: str,
    name: str,
    part_filepaths: Dict[str, str],
    base_filepaths: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    source = get_project(database_path, project_id)
    if source is None:
        return None
    new_id = str(uuid.uuid4())[:12]
    with get_db_connection(database_path) as db:
        db.execute(
            """
            INSERT INTO projects (
                id, name, description, alloy, voxel_size, refinement_level,
                memory_mode, analysis_options, performance_options,
                solidification_params, ui_state, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'draft')
            """,
            (
                new_id,
                name,
                source.get("description", ""),
                source["alloy"],
                float(source["voxel_size"]),
                int(source["refinement_level"]),
                source["memory_mode"],
                json.dumps(source["analysis_options"]),
                json.dumps(source["performance_options"]),
                json.dumps(source["solidification_params"]),
                json.dumps(source["ui_state"]),
            ),
        )
        rows = db.execute(
            "SELECT * FROM parts WHERE project_id = ? ORDER BY created_at ASC",
            (project_id,),
        ).fetchall()
        id_map = {str(row["id"]): str(uuid.uuid4())[:8] for row in rows}
        db.execute(
            "UPDATE projects SET ui_state = ? WHERE id = ?",
            (json.dumps(remap_cut_part_ids(source["ui_state"], id_map)), new_id),
        )
        for row in rows:
            part = dict(row)
            feature = _loads_feature(part.get("feature_json"))
            db.execute(
                """
                INSERT INTO parts (
                    id, project_id, name, part_type, filepath,
                    position_x, position_y, position_z,
                    rotation_x, rotation_y, rotation_z,
                    scale, visible, is_open_top, riser_sleeve, feature_json, base_filepath
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    id_map[str(part["id"])],
                    new_id,
                    part["name"],
                    part["part_type"],
                    part_filepaths.get(str(part["id"]), str(part["filepath"])),
                    part["position_x"],
                    part["position_y"],
                    part["position_z"],
                    part["rotation_x"],
                    part["rotation_y"],
                    part["rotation_z"],
                    part["scale"],
                    part["visible"],
                    part["is_open_top"],
                    part.get("riser_sleeve") or "insulating",
                    _dumps_feature(remap_feature_part_ids(feature, id_map)),
                    (base_filepaths or {}).get(str(part["id"]), part.get("base_filepath")),
                ),
            )
    return new_id

def delete_project(database_path: Path, project_id: str) -> list[str]:
    with get_db_connection(database_path) as db:
        rows = db.execute("SELECT filepath, base_filepath FROM parts WHERE project_id = ?", (project_id,)).fetchall()
        filepaths = [str(row["filepath"]) for row in rows]
        filepaths += [str(row["base_filepath"]) for row in rows if row["base_filepath"]]
        artifact = db.execute("SELECT artifact_path FROM projects WHERE id = ?", (project_id,)).fetchone()
        if artifact is not None and artifact["artifact_path"]:
            filepaths.append(str(artifact["artifact_path"]))
        db.execute("DELETE FROM parts WHERE project_id = ?", (project_id,))
        db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    return filepaths

PART_WRITABLE_FIELDS = frozenset(
    {
        "name",
        "part_type",
        "filepath",
        "position_x",
        "position_y",
        "position_z",
        "rotation_x",
        "rotation_y",
        "rotation_z",
        "scale",
        "visible",
        "is_open_top",
        "riser_sleeve",
    }
)

def _checked_part_data(data: Dict[str, Any], upload_root: Optional[Path]) -> Dict[str, Any]:
    clean = {key: value for key, value in data.items() if key in PART_WRITABLE_FIELDS}
    if "filepath" in clean and clean["filepath"]:
        if upload_root is None:
            raise UnsafePartFileError("A part file path cannot be accepted without an upload folder.")
        clean["filepath"] = resolve_upload_path(str(clean["filepath"]), upload_root)
    return clean

def add_part(
    database_path: Path,
    project_id: str,
    data: Dict[str, Any],
    *,
    upload_root: Optional[Path] = None,
    feature: Optional[Dict[str, Any]] = None,
) -> str:
    data = _checked_part_data(data, upload_root)
    part_id = str(uuid.uuid4())[:8]
    with get_db_connection(database_path) as db:
        db.execute(
            """
            INSERT INTO parts (
                id, project_id, name, part_type, filepath,
                position_x, position_y, position_z,
                rotation_x, rotation_y, rotation_z,
                scale, visible, is_open_top, riser_sleeve, feature_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                _riser_sleeve(data.get("riser_sleeve")),
                _dumps_feature(feature),
            ),
        )
        db.execute("UPDATE projects SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (project_id,))
    return part_id

def set_part_feature(
    database_path: Path,
    project_id: str,
    part_id: str,
    feature: Dict[str, Any],
) -> None:
    with get_db_connection(database_path) as db:
        db.execute(
            "UPDATE parts SET feature_json = ? WHERE id = ? AND project_id = ?",
            (_dumps_feature(feature), part_id, project_id),
        )
        db.execute("UPDATE projects SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (project_id,))

def count_parts_using_file(database_path: Path, filepath: str) -> int:
    with get_db_connection(database_path) as db:
        row = db.execute(
            "SELECT COUNT(*) AS n FROM parts WHERE filepath = ? OR base_filepath = ?",
            (filepath, filepath),
        ).fetchone()
    return int(row["n"]) if row else 0

def set_part_geometry(
    database_path: Path,
    project_id: str,
    part_id: str,
    filepath: str,
    base_filepath: Optional[str],
) -> None:
    with get_db_connection(database_path) as db:
        db.execute(
            "UPDATE parts SET filepath = ?, base_filepath = ? WHERE id = ? AND project_id = ?",
            (filepath, base_filepath, part_id, project_id),
        )
        db.execute("UPDATE projects SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (project_id,))

def remap_cut_part_ids(ui_state: Any, id_map: Dict[str, str]) -> Dict[str, Any]:
    state = dict(ui_state) if isinstance(ui_state, dict) else {}
    cuts = state.get("cut_features")
    if not isinstance(cuts, list) or not id_map:
        return state
    remapped = []
    for cut in cuts:
        if isinstance(cut, dict) and isinstance(cut.get("partIds"), list):
            cut = {**cut, "partIds": [id_map[pid] for pid in cut["partIds"] if pid in id_map]}
        remapped.append(cut)
    state["cut_features"] = remapped
    return state

def remap_feature_part_ids(
    feature: Optional[Dict[str, Any]], id_map: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    if not feature:
        return feature
    target = feature.get("targetPartId")
    if isinstance(target, str) and target in id_map:
        return {**feature, "targetPartId": id_map[target]}
    return feature

def _dumps_feature(feature: Optional[Dict[str, Any]]) -> Optional[str]:
    return json.dumps(feature) if feature else None

def _loads_feature(value: Any) -> Optional[Dict[str, Any]]:
    if not value:
        return None
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None

def update_part(
    database_path: Path,
    project_id: str,
    part_id: str,
    data: Dict[str, Any],
    *,
    upload_root: Optional[Path] = None,
) -> None:
    current = get_part(database_path, project_id, part_id)
    if current is None:
        return
    data = _checked_part_data(data, upload_root)
    with get_db_connection(database_path) as db:
        db.execute(
            """
            UPDATE parts
            SET name = ?, part_type = ?, filepath = ?,
                position_x = ?, position_y = ?, position_z = ?,
                rotation_x = ?, rotation_y = ?, rotation_z = ?,
                scale = ?, visible = ?, is_open_top = ?, riser_sleeve = ?
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
                _riser_sleeve(data.get("riser_sleeve", current.get("riser_sleeve"))),
                part_id,
                project_id,
            ),
        )

def delete_part(database_path: Path, project_id: str, part_id: str) -> Optional[str]:
    files = delete_part_files(database_path, project_id, part_id)
    return files[0] if files else None

def delete_part_files(database_path: Path, project_id: str, part_id: str) -> list[str]:
    with get_db_connection(database_path) as db:
        row = db.execute(
            "SELECT filepath, base_filepath FROM parts WHERE id = ? AND project_id = ?",
            (part_id, project_id),
        ).fetchone()
        db.execute("DELETE FROM parts WHERE id = ? AND project_id = ?", (part_id, project_id))
    if row is None:
        return []
    return [str(row["filepath"])] + ([str(row["base_filepath"])] if row["base_filepath"] else [])

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

def mark_project_status(database_path: "str | os.PathLike", project_id: str, status: str) -> None:
    with get_db_connection(database_path) as db:
        db.execute(
            "UPDATE projects SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (status, project_id),
        )

def mark_project_completed(
    database_path: "str | os.PathLike",
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

def _merge_pour_location(ui_state: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    if "pour_location" in data:
        ui_state = dict(ui_state)
        ui_state["pour_location"] = data["pour_location"]
    return ui_state

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
    row["riser_sleeve"] = _riser_sleeve(row.get("riser_sleeve"))
    row["feature"] = _loads_feature(row.pop("feature_json", None))
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
        "filling": False,
        "sand_restraint_index": 0.5,
    }

def _default_performance_options() -> Dict[str, Any]:
    return {
        "use_gpu": True,
        "use_incremental": False,
        "use_chunked_voxelization": False,
        "part_assignment_mode": "auto",
        "part_assignment_workers": 0,
        "part_assignment_batch_size": 2500,
        "max_voxels_for_visualization": 90000,
    }
