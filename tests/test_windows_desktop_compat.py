import sys
import unittest

from ui.main_window import clamp_window_position, should_use_console_translucency

class WindowsDesktopCompatTests(unittest.TestCase):
    def test_console_translucency_is_disabled_on_windows(self) -> None:
        self.assertFalse(should_use_console_translucency("win32"))
        self.assertTrue(should_use_console_translucency("darwin"))
        self.assertTrue(should_use_console_translucency("linux"))

    def test_console_position_is_clamped_inside_screen(self) -> None:
        x, y = clamp_window_position(
            preferred_x=1500,
            preferred_y=-100,
            window_width=700,
            window_height=500,
            screen_x=0,
            screen_y=0,
            screen_width=1280,
            screen_height=720,
        )

        self.assertEqual((x, y), (560, 20))

    def test_depth_peeling_is_disabled_on_windows(self) -> None:
        visualizer = sys.modules["ui.visualizer"]

        self.assertFalse(visualizer.should_use_depth_peeling("win32"))
        self.assertTrue(visualizer.should_use_depth_peeling("darwin"))
        self.assertTrue(visualizer.should_use_depth_peeling("linux"))

if __name__ == "__main__":
    unittest.main()
