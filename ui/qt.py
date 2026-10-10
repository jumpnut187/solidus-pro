"""The single place where the Qt binding is named.

Everything under ``ui/`` imports its Qt names from here rather than from
``PySide6`` directly, so a future binding change is one edit to this file
instead of 25 edits scattered across the package.

Two Qt 5 -> Qt 6 renames are absorbed here so no caller has to know about
them:

* ``pyqtSignal`` is spelled ``Signal`` in PySide6. Callers import ``Signal``.
  The PyQt name is deliberately *not* aliased -- keeping it alive would leave
  the package half-migrated.
* ``QAction`` moved from ``QtWidgets`` to ``QtGui`` in Qt 6. Importing it from
  here means callers need not care which submodule holds it.

One Qt 6 trap this file cannot absorb, so it is recorded here: enum members are
exposed through the *metaclass*, so ``QPalette.Text`` resolves but
``some_palette.Text`` raises ``AttributeError``. Qt 5 accepted both. Always
reach for the enum on the class, never through an instance.

Names are re-exported explicitly rather than via ``import *``. PySide6 ships
``.pyi`` stubs, and ``ui/workers.py`` and ``ui/dialogs/base.py`` are both
type-checked (see ``mypy.ini``); a star-import would discard every annotation
those two files gain from the migration.
"""

from __future__ import annotations

from PySide6.QtCore import (
    QFileSystemWatcher,
    QObject,
    QPoint,
    QRect,
    Qt,
    QThread,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QCursor,
    QFont,
    QPainter,
    QPalette,
    QPen,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStyle,
    QStyleOptionSlider,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QTextEdit,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)


def is_checked(state: object) -> bool:
    """True when a ``QCheckBox.stateChanged`` payload means "checked".

    Qt 6 trap: ``stateChanged`` emits a plain ``int`` (2), but ``Qt.Checked``
    is a ``Qt.CheckState`` enum member and ``2 == Qt.Checked`` is ``False``
    under PySide6. Every ``state == Qt.Checked`` handler therefore read as
    unchecked on both edges. ``QListWidgetItem.checkState()`` returns the enum,
    so both shapes are accepted here.
    """
    if state == Qt.Checked:
        return True
    checked_value = getattr(Qt.Checked, "value", 2)
    try:
        return int(state) == int(checked_value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return False


__all__ = [
    "is_checked",
    # QtCore
    "QFileSystemWatcher",
    "QObject",
    "QPoint",
    "QRect",
    "Qt",
    "QThread",
    "QTimer",
    "Signal",
    # QtGui
    "QAction",
    "QBrush",
    "QColor",
    "QCursor",
    "QFont",
    "QPainter",
    "QPalette",
    "QPen",
    # QtWidgets
    "QAbstractItemView",
    "QApplication",
    "QCheckBox",
    "QComboBox",
    "QDialog",
    "QDialogButtonBox",
    "QDoubleSpinBox",
    "QFileDialog",
    "QFormLayout",
    "QFrame",
    "QGridLayout",
    "QGroupBox",
    "QHBoxLayout",
    "QHeaderView",
    "QInputDialog",
    "QLabel",
    "QLineEdit",
    "QListWidget",
    "QListWidgetItem",
    "QMainWindow",
    "QMenu",
    "QMessageBox",
    "QPlainTextEdit",
    "QProgressBar",
    "QPushButton",
    "QScrollArea",
    "QSlider",
    "QSpinBox",
    "QSplitter",
    "QStyle",
    "QStyleOptionSlider",
    "QTabWidget",
    "QTableWidget",
    "QTableWidgetItem",
    "QTextBrowser",
    "QTextEdit",
    "QToolButton",
    "QToolTip",
    "QVBoxLayout",
    "QWidget",
    "QWidgetAction",
]
