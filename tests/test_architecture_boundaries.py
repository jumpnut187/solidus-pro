from __future__ import annotations

import ast
import unittest
from pathlib import Path
from typing import Dict, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOCAL_PACKAGES = {"core", "physics", "config", "ui", "web", "tools"}

def _module_name_for(path: Path) -> str:
    rel = path.relative_to(PROJECT_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)

def _imports_in(path: Path) -> Set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return set()

    found: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in LOCAL_PACKAGES:
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            if node.module and node.module.split(".")[0] in LOCAL_PACKAGES:
                found.add(node.module)
    return found

def _source_files(package: str) -> list[Path]:
    return sorted((PROJECT_ROOT / package).rglob("*.py"))

def _build_graph() -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = {}
    for package in LOCAL_PACKAGES:
        root = PROJECT_ROOT / package
        if not root.is_dir():
            continue
        for path in _source_files(package):
            graph[_module_name_for(path)] = _imports_in(path)
    return graph

def _reachable_from(graph: Dict[str, Set[str]], start: str) -> Set[str]:
    seen: Set[str] = set()
    stack = [start]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        for target in graph.get(current, ()):
            if target not in seen:
                stack.append(target)
    return seen

class LiveImportGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = _build_graph()
        cls.live = _reachable_from(cls.graph, "core.simulation_pipeline")

    def test_pipeline_is_the_entry_point(self) -> None:
        self.assertIn("core.simulation_pipeline", self.graph)
        self.assertIn("core.solidification", self.live)
        self.assertIn("core.solidification_loop", self.live)

    def test_physics_modules_are_on_the_live_path(self) -> None:
        for module in (
            "physics.heat_transfer",
            "physics.solidification_physics",
            "physics.units",
        ):
            self.assertIn(
                module,
                self.live,
                msg=f"{module} is no longer reachable from run_pipeline",
            )

    def test_no_duplicate_thermal_stack_reappears(self) -> None:
        banned = {
            "core.thermal_solver",
            "core.thermal_simulation",
            "core.convection",
            "core.mold_interface",
        }
        for module in banned:
            self.assertNotIn(
                module,
                self.graph,
                msg=(
                    f"{module} is back. It duplicated physics/heat_transfer.py and "
                    "was deleted; add to physics/ instead of alongside it."
                ),
            )

class LayeringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = _build_graph()

    def test_physics_is_mesh_agnostic(self) -> None:
        offenders = {
            module: sorted(t for t in targets if t.split(".")[0] in {"core", "config"})
            for module, targets in self.graph.items()
            if module.startswith("physics")
        }
        offenders = {m: t for m, t in offenders.items() if t}
        self.assertEqual(
            offenders,
            {},
            msg=f"physics/ modules importing core/ or config/: {offenders}",
        )

    def test_core_does_not_import_presentation_layers(self) -> None:
        offenders = {
            module: sorted(t for t in targets if t.split(".")[0] in {"ui", "web"})
            for module, targets in self.graph.items()
            if module.split(".")[0] in {"core", "physics", "config"}
        }
        offenders = {m: t for m, t in offenders.items() if t}
        self.assertEqual(
            offenders,
            {},
            msg=f"engine modules importing ui/ or web/: {offenders}",
        )

if __name__ == "__main__":
    unittest.main()
