import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

class ShrinkageDefaultThresholdTests(unittest.TestCase):
    def test_desktop_default_is_70(self) -> None:
        source = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
        match = re.search(r"^SHRINKAGE_DEFAULT_THRESHOLD_PERCENT = (\d+)$", source, re.M)
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), 70)

    def test_desktop_microporosity_default_is_70(self) -> None:
        source = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
        match = re.search(r"^MICROPOROSITY_DEFAULT_THRESHOLD_PERCENT = (\d+)$", source, re.M)
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), 70)

    def test_desktop_recommended_thresholds_are_70(self) -> None:
        from ui.main_window import MainWindow

        window = MainWindow.__new__(MainWindow)
        self.assertEqual(MainWindow._get_recommended_threshold(window, "shrinkage", "WCB"), 70)
        self.assertEqual(MainWindow._get_recommended_threshold(window, "porosity", "WCB"), 70)

    def test_web_default_matches_desktop(self) -> None:
        source = (ROOT / "web_frontend" / "src" / "lib" / "viewState.ts").read_text(encoding="utf-8")
        match = re.search(r"^\s*shrinkage: (\d+),", source, re.M)
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), 70)

    def test_web_porosity_default_is_70(self) -> None:
        source = (ROOT / "web_frontend" / "src" / "lib" / "viewState.ts").read_text(encoding="utf-8")
        match = re.search(r"^\s*porosity: (\d+),", source, re.M)
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), 70)

if __name__ == "__main__":
    unittest.main()
