import unittest

from ui.qt import Qt, is_checked

class IsCheckedTests(unittest.TestCase):
    def test_enum_members(self) -> None:
        self.assertTrue(is_checked(Qt.Checked))
        self.assertFalse(is_checked(Qt.Unchecked))
        self.assertFalse(is_checked(Qt.PartiallyChecked))

    def test_int_payloads_from_state_changed(self) -> None:
        self.assertTrue(is_checked(2))
        self.assertFalse(is_checked(0))
        self.assertFalse(is_checked(1))

    def test_garbage_is_unchecked(self) -> None:
        self.assertFalse(is_checked(None))
        self.assertFalse(is_checked("checked"))

if __name__ == "__main__":
    unittest.main()
