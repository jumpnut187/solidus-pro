"""Popup window for showing/hiding individual components during review."""

from __future__ import annotations

from ui.qt import (
    QAbstractItemView,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    Qt,
)

from .base import DraggableDialog


class ComponentVisibilityDialog(DraggableDialog):
    """Checklist of components with live show/hide.

    Toggles apply immediately to the 3D view, so this dialog has no
    Confirm/Cancel pair: the only exit is Close. The main window owns the
    check-state logic and connects to ``components_list`` / ``show_all_btn``
    directly, which keeps a single source of truth for the hidden set.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent, "Components")
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowTitleHint
            | Qt.WindowCloseButtonHint
            | Qt.WindowMinMaxButtonsHint
        )
        self.setMinimumWidth(320)
        self.setMinimumHeight(360)
        self._setup_ui()

    def _setup_ui(self) -> None:
        hint = QLabel(
            "Uncheck a component to hide it and its result voxels while reviewing.\n"
            "Hidden components are still included when the simulation runs."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #aaa; font-size: 11px;")
        self.content_layout.addWidget(hint)

        self.components_list = QListWidget()
        self.components_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.content_layout.addWidget(self.components_list)

        button_row = QHBoxLayout()
        self.show_all_btn = QPushButton("Show All")
        self.show_all_btn.setToolTip("Show every component again")
        self.show_all_btn.setEnabled(False)
        button_row.addWidget(self.show_all_btn)
        button_row.addStretch()
        self.content_layout.addLayout(button_row)

        # Live toggles make Confirm/Cancel meaningless; Close is the only exit.
        self.button_box.clear()
        close_btn = self.button_box.addButton(QDialogButtonBox.Close)
        close_btn.clicked.connect(self.close)

    def set_empty_state(self, has_parts: bool) -> None:
        """Grey out the list when there is nothing to toggle."""
        self.components_list.setEnabled(has_parts)
