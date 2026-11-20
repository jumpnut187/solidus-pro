from __future__ import annotations

from typing import Any, Dict

from ui.qt import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    Qt,
)

class DraggableDialog(QDialog):

    def __init__(self, parent=None, title: str = "Dialog") -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint)
        self.setModal(False)
        self._drag_position = None

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_widget.setLayout(self.content_layout)
        self.main_layout.addWidget(self.content_widget)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.main_layout.addWidget(line)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.button(QDialogButtonBox.Ok).setText("Confirm")
        self.button_box.accepted.connect(self._on_confirm)
        self.button_box.rejected.connect(self._on_cancel)
        self.main_layout.addWidget(self.button_box)

        self._original_values: Dict[str, Any] = {}
        self.setStyleSheet(
            """
            DraggableDialog {
                background-color: #2b2b2b;
                border: 2px solid #555;
                border-radius: 8px;
            }
            QLabel {
                color: #ddd;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                color: #aaa;
                subcontrol-origin: margin;
                left: 10px;
            }
            QPushButton {
                background-color: #444;
                border: 1px solid #666;
                border-radius: 4px;
                padding: 6px 12px;
                color: #ddd;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QPushButton:pressed {
                background-color: #333;
            }
            QComboBox {
                background-color: #444;
                border: 1px solid #666;
                border-radius: 4px;
                padding: 4px 8px;
                color: #fff;
            }
            QComboBox:hover {
                border: 1px solid #888;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #2b2b2b;
                color: #fff;
                selection-background-color: #0066cc;
                selection-color: #fff;
                border: 1px solid #555;
                outline: 0;
            }
            QComboBox QAbstractItemView::item {
                color: #fff;
                background-color: transparent;
                padding: 4px;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #0066cc;
                color: #fff;
            }
            """
        )

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_position = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() == Qt.LeftButton and self._drag_position is not None:
            self.move(event.globalPosition().toPoint() - self._drag_position)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._drag_position = None

    def store_original_values(self) -> None:
        pass

    def restore_original_values(self) -> None:
        pass

    def apply_values(self) -> None:
        pass

    def _on_confirm(self) -> None:
        self.apply_values()
        self.accept()

    def _on_cancel(self) -> None:
        self.restore_original_values()
        self.reject()

    def showEvent(self, event) -> None:
        self.store_original_values()
        super().showEvent(event)

    def center_on_parent(self) -> None:
        if self.parent():
            parent_geo = self.parent().geometry()
            x = parent_geo.x() + (parent_geo.width() - self.width()) // 2
            y = parent_geo.y() + (parent_geo.height() - self.height()) // 2
            self.move(x, y)
