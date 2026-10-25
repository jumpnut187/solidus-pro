"""Parts manager dialog."""

from __future__ import annotations

from ui.qt import (
    is_checked,
    QAbstractItemView,
    QBrush,
    QCheckBox,
    QColor,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPalette,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
    Qt,
    Signal,
)

from core.part_manager import (
    PART_TYPE_DISPLAY_NAMES,
    RISER_SLEEVE_INSULATING,
    RISER_SLEEVE_NONE,
    riser_sleeve_from_value,
)
from .base import DraggableDialog


class PartsDialog(DraggableDialog):
    """Modal dialog for managing parts with batch part type assignment"""

    add_part_requested = Signal()
    remove_part_requested = Signal()
    clear_parts_requested = Signal()
    part_selected = Signal(str)
    position_changed = Signal(float, float, float)
    rotation_changed = Signal(float, float, float)
    batch_part_types_confirmed = Signal(dict)
    arrange_grid_requested = Signal()
    arrange_line_requested = Signal()
    duplicate_requested = Signal()
    open_top_changed = Signal(str, bool)
    riser_sleeve_changed = Signal(str, str)

    #: Dropdown entries, derived so a new PartType is selectable automatically.
    PART_TYPE_OPTIONS = list(PART_TYPE_DISPLAY_NAMES.values())

    #: Riser sleeve dropdown entries: (label, Part.riser_sleeve value).
    RISER_SLEEVE_OPTIONS = (
        ("Insulating", RISER_SLEEVE_INSULATING),
        ("None (sand)", RISER_SLEEVE_NONE),
    )

    def __init__(self, parent=None):
        super().__init__(parent, "Parts Manager")
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowTitleHint
            | Qt.WindowCloseButtonHint
            | Qt.WindowMinMaxButtonsHint
        )
        self.setMinimumWidth(450)
        self.setMinimumHeight(450)
        self._setup_ui()

        self._original_part_types = {}
        self._pending_changes = {}

    def _setup_ui(self):
        """Set up the parts dialog UI"""
        btn_layout = QHBoxLayout()
        self.add_part_btn = QPushButton("Add Part")
        self.add_part_btn.clicked.connect(self.add_part_requested.emit)
        btn_layout.addWidget(self.add_part_btn)

        self.remove_part_btn = QPushButton("Remove")
        self.remove_part_btn.clicked.connect(self.remove_part_requested.emit)
        self.remove_part_btn.setEnabled(False)
        btn_layout.addWidget(self.remove_part_btn)

        self.clear_parts_btn = QPushButton("Clear All")
        self.clear_parts_btn.clicked.connect(self.clear_parts_requested.emit)
        self.clear_parts_btn.setEnabled(False)
        btn_layout.addWidget(self.clear_parts_btn)

        self.content_layout.addLayout(btn_layout)

        self.parts_table = QTableWidget()
        self.parts_table.setColumnCount(4)
        self.parts_table.setHorizontalHeaderLabels(["Part Name", "Part Type", "Open Top", "Sleeve"])
        self.parts_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.parts_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.parts_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self.parts_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
        self.parts_table.setColumnWidth(1, 110)
        self.parts_table.setColumnWidth(2, 65)
        self.parts_table.setColumnWidth(3, 105)
        self.parts_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.parts_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.parts_table.setMinimumHeight(150)
        self.parts_table.cellClicked.connect(self._on_row_clicked)
        self.parts_table.setStyleSheet("""
            QTableWidget {
                background-color: #333;
                alternate-background-color: #3a3a3a;
                gridline-color: #555;
                color: #fff;
            }
            QTableWidget::item {
                padding: 4px;
                color: #fff;
                background-color: transparent;
            }
            QTableWidget::item:selected {
                background-color: #0066cc;
                color: #fff;
            }
            QTableWidget::item:alternate {
                color: #fff;
            }
            QHeaderView::section {
                background-color: #444;
                padding: 4px;
                border: 1px solid #555;
                color: #ddd;
            }
            QHeaderView {
                background-color: #2b2b2b;
            }
        """)

        # Qt 6 exposes enum members on the metaclass, so these have to come off
        # QPalette itself -- `palette.Text` raised AttributeError under PySide6.
        palette = self.parts_table.palette()
        palette.setColor(QPalette.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        self.parts_table.setPalette(palette)
        self.content_layout.addWidget(self.parts_table)

        self.confirm_types_btn = QPushButton("Confirm Part Type Assignment")
        self.confirm_types_btn.setToolTip(
            "Apply all pending part type changes at once.\n"
            "This avoids re-rendering for each individual change."
        )
        self.confirm_types_btn.setEnabled(False)
        self.confirm_types_btn.clicked.connect(self._on_confirm_types)
        self.confirm_types_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d5a27;
                border: 1px solid #3d7a37;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3d7a37;
            }
            QPushButton:disabled {
                background-color: #444;
                border: 1px solid #555;
                color: #888;
            }
        """)
        self.content_layout.addWidget(self.confirm_types_btn)

        pos_group = QGroupBox("Position (Selected Part)")
        pos_layout = QGridLayout()

        self.pos_x_spin = QDoubleSpinBox()
        self.pos_x_spin.setRange(-1000, 1000)
        self.pos_x_spin.setSingleStep(0.5)
        self.pos_x_spin.setPrefix("X: ")
        self.pos_x_spin.valueChanged.connect(self._on_position_changed)
        pos_layout.addWidget(self.pos_x_spin, 0, 0)

        self.pos_y_spin = QDoubleSpinBox()
        self.pos_y_spin.setRange(-1000, 1000)
        self.pos_y_spin.setSingleStep(0.5)
        self.pos_y_spin.setPrefix("Y: ")
        self.pos_y_spin.valueChanged.connect(self._on_position_changed)
        pos_layout.addWidget(self.pos_y_spin, 0, 1)

        self.pos_z_spin = QDoubleSpinBox()
        self.pos_z_spin.setRange(-1000, 1000)
        self.pos_z_spin.setSingleStep(0.5)
        self.pos_z_spin.setPrefix("Z: ")
        self.pos_z_spin.valueChanged.connect(self._on_position_changed)
        pos_layout.addWidget(self.pos_z_spin, 0, 2)

        pos_group.setLayout(pos_layout)
        self.content_layout.addWidget(pos_group)

        rot_group = QGroupBox("Rotation (Selected Part)")
        rot_layout = QGridLayout()

        self.rot_x_spin = QDoubleSpinBox()
        self.rot_x_spin.setRange(-360, 360)
        self.rot_x_spin.setSingleStep(5.0)
        self.rot_x_spin.setPrefix("Rx: ")
        self.rot_x_spin.setSuffix("°")
        self.rot_x_spin.valueChanged.connect(self._on_rotation_changed)
        rot_layout.addWidget(self.rot_x_spin, 0, 0)

        self.rot_y_spin = QDoubleSpinBox()
        self.rot_y_spin.setRange(-360, 360)
        self.rot_y_spin.setSingleStep(5.0)
        self.rot_y_spin.setPrefix("Ry: ")
        self.rot_y_spin.setSuffix("°")
        self.rot_y_spin.valueChanged.connect(self._on_rotation_changed)
        rot_layout.addWidget(self.rot_y_spin, 0, 1)

        self.rot_z_spin = QDoubleSpinBox()
        self.rot_z_spin.setRange(-360, 360)
        self.rot_z_spin.setSingleStep(5.0)
        self.rot_z_spin.setPrefix("Rz: ")
        self.rot_z_spin.setSuffix("°")
        self.rot_z_spin.valueChanged.connect(self._on_rotation_changed)
        rot_layout.addWidget(self.rot_z_spin, 0, 2)

        rot_group.setLayout(rot_layout)
        self.content_layout.addWidget(rot_group)

        arrange_layout = QHBoxLayout()
        self.arrange_grid_btn = QPushButton("Grid")
        self.arrange_grid_btn.setToolTip("Arrange parts in a grid")
        self.arrange_grid_btn.clicked.connect(self.arrange_grid_requested.emit)
        arrange_layout.addWidget(self.arrange_grid_btn)

        self.arrange_line_btn = QPushButton("Line")
        self.arrange_line_btn.setToolTip("Arrange parts in a line")
        self.arrange_line_btn.clicked.connect(self.arrange_line_requested.emit)
        arrange_layout.addWidget(self.arrange_line_btn)

        self.duplicate_btn = QPushButton("Duplicate")
        self.duplicate_btn.setToolTip("Duplicate selected part")
        self.duplicate_btn.clicked.connect(self.duplicate_requested.emit)
        self.duplicate_btn.setEnabled(False)
        arrange_layout.addWidget(self.duplicate_btn)

        self.content_layout.addLayout(arrange_layout)

        self.parts_info_label = QLabel("No parts loaded")
        self.parts_info_label.setWordWrap(True)
        self.parts_info_label.setStyleSheet("font-size: 10px; color: #888;")
        self.content_layout.addWidget(self.parts_info_label)

        self._updating = False

    def _on_row_clicked(self, row, column):
        """Handle table row selection"""
        part_id = self._get_part_id_for_row(row)
        if part_id:
            self.part_selected.emit(part_id)
            self.remove_part_btn.setEnabled(True)
            self.duplicate_btn.setEnabled(True)

    def _get_part_id_for_row(self, row):
        """Get the part_id stored in a table row"""
        name_item = self.parts_table.item(row, 0)
        if name_item:
            return name_item.data(Qt.UserRole)
        return None

    def _on_dropdown_changed(self, row, new_type_text):
        """Handle part type dropdown change (no re-render, just track pending)"""
        if self._updating:
            return

        part_id = self._get_part_id_for_row(row)
        if not part_id:
            return

        original_type = self._original_part_types.get(part_id)

        if new_type_text != original_type:
            self._pending_changes[part_id] = new_type_text
        else:
            self._pending_changes.pop(part_id, None)

        self._update_confirm_button_state()

    def _update_confirm_button_state(self):
        """Enable/disable confirm button based on pending changes"""
        has_changes = len(self._pending_changes) > 0
        self.confirm_types_btn.setEnabled(has_changes)

        if has_changes:
            self.confirm_types_btn.setText(f"Confirm Part Type Assignment ({len(self._pending_changes)} change{'s' if len(self._pending_changes) > 1 else ''})")
        else:
            self.confirm_types_btn.setText("Confirm Part Type Assignment")

    def _on_confirm_types(self):
        """Apply all pending part type changes at once"""
        if self._pending_changes:
            self.batch_part_types_confirmed.emit(self._pending_changes.copy())

            for part_id, new_type in self._pending_changes.items():
                self._original_part_types[part_id] = new_type

            self._pending_changes.clear()
            self._update_confirm_button_state()

    def _on_position_changed(self):
        """Forward position change to main window"""
        if not self._updating:
            self.position_changed.emit(
                self.pos_x_spin.value(),
                self.pos_y_spin.value(),
                self.pos_z_spin.value()
            )

    def update_position_spinboxes(self, x, y, z):
        """Update position spinboxes without triggering signals"""
        self._updating = True
        self.pos_x_spin.setValue(x)
        self.pos_y_spin.setValue(y)
        self.pos_z_spin.setValue(z)
        self._updating = False

    def _on_rotation_changed(self):
        """Forward rotation change to main window."""
        if not self._updating:
            self.rotation_changed.emit(
                self.rot_x_spin.value(),
                self.rot_y_spin.value(),
                self.rot_z_spin.value(),
            )

    def update_rotation_spinboxes(self, rx, ry, rz):
        """Update rotation spinboxes without triggering signals."""
        self._updating = True
        self.rot_x_spin.setValue(rx)
        self.rot_y_spin.setValue(ry)
        self.rot_z_spin.setValue(rz)
        self._updating = False

    def _on_open_top_changed(self, part_id: str, is_open: bool):
        """Handle Open Top checkbox state change"""
        if not self._updating:
            self.open_top_changed.emit(part_id, is_open)

    def _on_riser_sleeve_changed(self, part_id: str, combo: QComboBox):
        """Handle Sleeve dropdown change"""
        if not self._updating:
            self.riser_sleeve_changed.emit(part_id, combo.currentData())

    def _update_open_top_checkbox_state(self, row: int, new_type_text: str):
        """Enable/disable Open Top checkbox and Sleeve dropdown based on part type"""
        is_riser = new_type_text == "Riser"
        container_widget = self.parts_table.cellWidget(row, 2)
        if container_widget:
            cb = container_widget.findChild(QCheckBox)
            if cb:
                cb.setEnabled(is_riser)
                if not is_riser:
                    cb.setChecked(False)
        sleeve_combo = self.parts_table.cellWidget(row, 3)
        if isinstance(sleeve_combo, QComboBox):
            sleeve_combo.setEnabled(is_riser)
            if not is_riser:
                sleeve_combo.setCurrentIndex(0)

    def populate_parts(self, parts_data):
        """Populate the table with parts data."""
        self._updating = True
        self.parts_table.setRowCount(0)
        self._original_part_types.clear()
        self._pending_changes.clear()

        type_value_to_text = {
            part_type.value: display
            for part_type, display in PART_TYPE_DISPLAY_NAMES.items()
        }

        for part_data in parts_data:
            part_id, name, part_type_value, *extra = part_data
            is_open_top = bool(extra[0]) if len(extra) > 0 else False
            riser_sleeve = extra[1] if len(extra) > 1 else RISER_SLEEVE_INSULATING

            row = self.parts_table.rowCount()
            self.parts_table.insertRow(row)

            name_item = QTableWidgetItem(name)
            name_item.setData(Qt.UserRole, part_id)
            name_item.setData(Qt.ForegroundRole, QColor(255, 255, 255))
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            name_item.setForeground(QBrush(QColor(255, 255, 255)))
            self.parts_table.setItem(row, 0, name_item)
            self.parts_table.item(row, 0).setForeground(QBrush(QColor(255, 255, 255)))

            type_combo = QComboBox()
            type_combo.addItems(self.PART_TYPE_OPTIONS)
            type_combo.setToolTip(
                "Cast: Actual casting - analyzed for shrink and microporosity\n"
                "Riser: Feeder with exothermic effect - slows cooling\n"
                "Riser Post: Riser neck/post connecting riser to casting\n"
                "Chill: External chill - accelerates cooling"
            )

            display_text = type_value_to_text.get(part_type_value, "Cast")
            type_combo.setCurrentText(display_text)

            self._original_part_types[part_id] = display_text

            type_combo.currentTextChanged.connect(
                lambda text, r=row: self._on_dropdown_changed(r, text)
            )

            self.parts_table.setCellWidget(row, 1, type_combo)

            open_top_cb = QCheckBox()
            open_top_cb.setChecked(is_open_top)
            open_top_cb.setEnabled(part_type_value == "riser")
            open_top_cb.setToolTip(
                "Open Top Riser: Atmospheric pressure pushes metal down for better feeding.\n"
                "Blind Riser (unchecked): Enclosed, may form vacuum reducing feeding by ~20%.\n"
                "With a sand mold (Simulation > Mold && Cores) an open riser stays open to the air\n"
                "at the top of the mold; a blind riser gets the closed-riser sand cover."
            )
            open_top_cb.setProperty("part_id", part_id)
            open_top_cb.stateChanged.connect(
                lambda state, pid=part_id: self._on_open_top_changed(pid, is_checked(state))
            )

            cb_container = QWidget()
            cb_layout = QHBoxLayout(cb_container)
            cb_layout.addWidget(open_top_cb)
            cb_layout.setAlignment(Qt.AlignCenter)
            cb_layout.setContentsMargins(0, 0, 0, 0)
            self.parts_table.setCellWidget(row, 2, cb_container)

            sleeve_combo = QComboBox()
            for label, value in self.RISER_SLEEVE_OPTIONS:
                sleeve_combo.addItem(label, value)
            sleeve_combo.setCurrentIndex(
                max(0, sleeve_combo.findData(riser_sleeve_from_value(riser_sleeve)))
            )
            sleeve_combo.setEnabled(part_type_value == "riser")
            sleeve_combo.setToolTip(
                "Insulating: the riser sits in an insulating sleeve (slower cooling, the default).\n"
                "None (sand): a bare sand riser; its sides cool through the same metal-to-sand\n"
                "contact as the casting."
            )
            sleeve_combo.currentIndexChanged.connect(
                lambda _index, pid=part_id, cb=sleeve_combo: self._on_riser_sleeve_changed(pid, cb)
            )
            self.parts_table.setCellWidget(row, 3, sleeve_combo)

            type_combo.currentTextChanged.connect(
                lambda text, r=row: self._update_open_top_checkbox_state(r, text)
            )

        self._updating = False
        self._update_confirm_button_state()

        has_parts = self.parts_table.rowCount() > 0
        self.clear_parts_btn.setEnabled(has_parts)
        self.parts_info_label.setText(
            f"{self.parts_table.rowCount()} part(s) loaded" if has_parts else "No parts loaded"
        )

    def get_selected_part_id(self):
        """Get the currently selected part ID"""
        selected_rows = self.parts_table.selectionModel().selectedRows()
        if selected_rows:
            row = selected_rows[0].row()
            return self._get_part_id_for_row(row)
        return None

    def select_part_by_id(self, part_id):
        """Select a row by part ID"""
        for row in range(self.parts_table.rowCount()):
            if self._get_part_id_for_row(row) == part_id:
                self.parts_table.selectRow(row)
                return True
        return False
