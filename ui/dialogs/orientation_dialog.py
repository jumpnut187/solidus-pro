from __future__ import annotations

from typing import Dict, Optional

from core.part_manager import PART_TYPE_DISPLAY_NAMES
from ui.orientation_tool import (
    HEIGHT_STEPS_IN,
    ROTATION_STEPS_DEG,
    OrientationController,
    outward_face_normal,
)
from ui.qt import (
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    Qt,
    Signal,
)

from .base import DraggableDialog

PICK_FACE_DOWN = "face_down"
PICK_FACE_UP = "face_up"
PICK_PARTING_HEIGHT = "parting_height"

_PICK_HINTS = {
    PICK_FACE_DOWN: "Face down: click a face of the part in the 3D view. Esc cancels.",
    PICK_FACE_UP: "Face up: click a face of the part in the 3D view. Esc cancels.",
    PICK_PARTING_HEIGHT: "Pick height: click a point on the model to put the parting line there. Esc cancels.",
}

def _named(widget, name: str, tooltip: str = ""):
    widget.setObjectName(name)
    widget.setAccessibleName(name)
    if tooltip:
        widget.setToolTip(tooltip)
    return widget

def _spin(name: str, tooltip: str, low: float, high: float, step: float, suffix: str) -> QDoubleSpinBox:
    spin = _named(QDoubleSpinBox(), name, tooltip)
    spin.setRange(low, high)
    spin.setDecimals(3)
    spin.setSingleStep(step)
    spin.setSuffix(suffix)
    spin.setKeyboardTracking(False)
    return spin

class OrientationDialog(DraggableDialog):

    geometry_changed = Signal()
    parting_changed = Signal()
    pick_requested = Signal(str, object)
    pick_cancelled = Signal()

    TITLE = "Orient & Parting Line"

    def __init__(self, controller: OrientationController, parent=None) -> None:
        super().__init__(parent, self.TITLE)
        self.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.setMinimumWidth(440)
        self.controller = controller
        self.pick_mode: Optional[str] = None
        self._updating = False
        self._snapshot = None
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        layout = self.content_layout

        self.target_label = _named(QLabel("Target: none"), "Target part")
        self.target_label.setStyleSheet("font-weight: bold; color: #fff;")
        layout.addWidget(self.target_label)
        hint = QLabel("Rotates the selected part, or the casting when nothing is selected.")
        hint.setStyleSheet("font-size: 10px; color: #888;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        pick_row = QHBoxLayout()
        self.pick_hint_label = _named(QLabel(""), "Pick mode status")
        self.pick_hint_label.setWordWrap(True)
        self.pick_hint_label.setStyleSheet(
            "background-color: #5a4a10; color: #fff; padding: 6px; border-radius: 4px;"
        )
        pick_row.addWidget(self.pick_hint_label, 1)
        self.cancel_pick_btn = _named(QPushButton("Cancel pick"), "Cancel pick", "Leave pick mode (Esc)")
        self.cancel_pick_btn.clicked.connect(self.cancel_pick)
        pick_row.addWidget(self.cancel_pick_btn)
        self.pick_row_widget = QWidget()
        self.pick_row_widget.setLayout(pick_row)
        self.pick_row_widget.setVisible(False)
        layout.addWidget(self.pick_row_widget)

        layout.addWidget(self._build_orientation_group())
        layout.addWidget(self._build_height_group())
        layout.addWidget(self._build_parting_group())

        self.button_box.button(QDialogButtonBox.Ok).setText("Done")
        _named(self.button_box.button(QDialogButtonBox.Ok), "Done", "Keep the changes and close the tool")
        self.button_box.button(QDialogButtonBox.Cancel).setText("Revert changes")
        _named(
            self.button_box.button(QDialogButtonBox.Cancel),
            "Revert changes",
            "Undo every orientation, height and parting change made since the tool was opened, then close",
        )

    def _build_orientation_group(self) -> QGroupBox:
        group = QGroupBox("Orientation (world axes)")
        box = QVBoxLayout()

        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Rotation step:"))
        self.rotation_step_combo = _named(QComboBox(), "Rotation step", "Angle turned by each rotate button")
        for step in ROTATION_STEPS_DEG:
            self.rotation_step_combo.addItem(f"{step:g}°", step)
        self.rotation_step_combo.currentIndexChanged.connect(self._update_rotate_tooltips)
        step_row.addWidget(self.rotation_step_combo)
        step_row.addStretch(1)
        box.addLayout(step_row)

        grid = QGridLayout()
        self.rotate_buttons: Dict[str, QPushButton] = {}
        for row, axis in enumerate("XYZ"):
            grid.addWidget(QLabel(f"{axis} axis"), row, 0)
            for col, sign in enumerate(("-", "+")):
                name = f"Rotate {axis} {sign}"
                btn = _named(QPushButton(f"{axis} {sign}"), name)
                btn.clicked.connect(
                    lambda _checked=False, a=axis, s=sign: self._on_rotate(a, -1.0 if s == "-" else 1.0)
                )
                grid.addWidget(btn, row, col + 1)
                self.rotate_buttons[name] = btn
        box.addLayout(grid)
        self._update_rotate_tooltips()

        self.keep_on_floor_check = _named(
            QCheckBox("Keep on floor"),
            "Keep on floor",
            "After a rotation, shift the part up or down so its lowest point stays where it was",
        )
        self.keep_on_floor_check.setChecked(self.controller.keep_on_floor)
        self.keep_on_floor_check.toggled.connect(self._on_keep_on_floor_toggled)
        box.addWidget(self.keep_on_floor_check)

        face_row = QHBoxLayout()
        self.face_down_btn = _named(
            QPushButton("Face down"),
            "Face down",
            "Then click a face in the 3D view: the part turns so that face lies flat on the floor (-Z)",
        )
        self.face_down_btn.setCheckable(True)
        self.face_down_btn.clicked.connect(lambda: self._toggle_pick(PICK_FACE_DOWN))
        face_row.addWidget(self.face_down_btn)
        self.face_up_btn = _named(
            QPushButton("Face up"),
            "Face up",
            "Then click a face in the 3D view: the part turns so that face points straight up (+Z)",
        )
        self.face_up_btn.setCheckable(True)
        self.face_up_btn.clicked.connect(lambda: self._toggle_pick(PICK_FACE_UP))
        face_row.addWidget(self.face_up_btn)
        self.reset_orientation_btn = _named(
            QPushButton("Reset orientation"),
            "Reset orientation",
            "Rotation back to 0, 0, 0 (as imported from CAD)",
        )
        self.reset_orientation_btn.clicked.connect(self._on_reset_orientation)
        face_row.addWidget(self.reset_orientation_btn)
        box.addLayout(face_row)

        self.size_label = _named(QLabel("Size (W x D x H): -"), "Target size")
        box.addWidget(self.size_label)

        self.advanced_check = _named(
            QCheckBox("Advanced: exact rotation angles"),
            "Advanced: exact rotation angles",
            "Show the static XYZ Euler angles (degrees) for exact entry",
        )
        box.addWidget(self.advanced_check)
        self.advanced_widget = QWidget()
        adv = QHBoxLayout()
        adv.setContentsMargins(0, 0, 0, 0)
        self.euler_spins = []
        for axis in "XYZ":
            spin = _spin(
                f"Rotation R{axis.lower()}",
                f"Exact rotation about {axis} in degrees (static XYZ Euler, applied X then Y then Z)",
                -360.0,
                360.0,
                5.0,
                "°",
            )
            spin.setPrefix(f"R{axis.lower()}: ")
            spin.valueChanged.connect(self._on_euler_changed)
            adv.addWidget(spin)
            self.euler_spins.append(spin)
        self.advanced_widget.setLayout(adv)
        self.advanced_widget.setVisible(False)
        self.advanced_check.toggled.connect(self.advanced_widget.setVisible)
        box.addWidget(self.advanced_widget)

        group.setLayout(box)
        return group

    def _build_height_group(self) -> QGroupBox:
        group = QGroupBox("Height")
        box = QVBoxLayout()

        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Height step:"))
        self.height_step_combo = _named(
            QComboBox(), "Height step", "Distance moved by the raise/lower buttons (part and parting line)"
        )
        for step in HEIGHT_STEPS_IN:
            self.height_step_combo.addItem(f"{step:g} in", step)
        self.height_step_combo.setCurrentIndex(1)
        step_row.addWidget(self.height_step_combo)
        step_row.addStretch(1)
        box.addLayout(step_row)

        row = QHBoxLayout()
        self.lower_part_btn = _named(QPushButton("Lower part"), "Lower part", "Move the target down by the height step")
        self.lower_part_btn.clicked.connect(lambda: self._on_move_part(-1.0))
        row.addWidget(self.lower_part_btn)
        self.raise_part_btn = _named(QPushButton("Raise part"), "Raise part", "Move the target up by the height step")
        self.raise_part_btn.clicked.connect(lambda: self._on_move_part(1.0))
        row.addWidget(self.raise_part_btn)
        self.drop_to_floor_btn = _named(
            QPushButton("Drop to floor"), "Drop to floor", "Put the target's lowest point at Z = 0"
        )
        self.drop_to_floor_btn.clicked.connect(self._on_drop_to_floor)
        row.addWidget(self.drop_to_floor_btn)
        box.addLayout(row)

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(QLabel("Bottom Z:"))
        self.bottom_z_spin = _spin(
            "Bottom Z", "World Z of the target's lowest point (in)", -1000.0, 1000.0, 0.25, " in"
        )
        self.bottom_z_spin.valueChanged.connect(self._on_bottom_z_changed)
        bottom_row.addWidget(self.bottom_z_spin)
        bottom_row.addStretch(1)
        box.addLayout(bottom_row)

        group.setLayout(box)
        return group

    def _build_parting_group(self) -> QGroupBox:
        group = QGroupBox("Parting line (flat, horizontal)")
        box = QVBoxLayout()

        top_row = QHBoxLayout()
        self.show_parting_check = _named(
            QCheckBox("Show parting line"),
            "Show parting line",
            "Show the parting plane in the 3D view. The first time, it is placed at the widest section.",
        )
        self.show_parting_check.toggled.connect(self._on_show_parting_toggled)
        top_row.addWidget(self.show_parting_check)
        self.lock_parting_check = _named(
            QCheckBox("Lock to casting"),
            "Lock to casting",
            "When the casting is raised or lowered, the parting line moves with it",
        )
        self.lock_parting_check.setChecked(self.controller.lock_default)
        self.lock_parting_check.toggled.connect(self._on_lock_toggled)
        top_row.addWidget(self.lock_parting_check)
        box.addLayout(top_row)

        z_row = QHBoxLayout()
        z_row.addWidget(QLabel("Parting Z:"))
        self.parting_z_spin = _spin(
            "Parting Z", "World Z of the parting plane (in). Above = cope, below = drag.", -1000.0, 1000.0, 0.25, " in"
        )
        self.parting_z_spin.valueChanged.connect(self._on_parting_z_changed)
        z_row.addWidget(self.parting_z_spin)
        self.lower_parting_btn = _named(
            QPushButton("Lower parting"), "Lower parting", "Move the parting line down by the height step"
        )
        self.lower_parting_btn.clicked.connect(lambda: self._on_nudge_parting(-1.0))
        z_row.addWidget(self.lower_parting_btn)
        self.raise_parting_btn = _named(
            QPushButton("Raise parting"), "Raise parting", "Move the parting line up by the height step"
        )
        self.raise_parting_btn.clicked.connect(lambda: self._on_nudge_parting(1.0))
        z_row.addWidget(self.raise_parting_btn)
        box.addLayout(z_row)

        presets = QGridLayout()
        self.preset_buttons: Dict[str, QPushButton] = {}
        preset_specs = (
            ("widest", "Widest section", "Top face of the widest horizontal section (usual flat parting line)"),
            ("top", "Top of casting", "Parting at the casting's top: the whole casting is in the drag"),
            ("bottom", "Bottom of casting", "Parting at the casting's bottom: the whole casting is in the cope"),
            ("center", "Center", "Parting halfway up the casting"),
        )
        for index, (key, label, tip) in enumerate(preset_specs):
            btn = _named(QPushButton(label), label, tip)
            btn.clicked.connect(lambda _checked=False, k=key: self._on_preset(k))
            presets.addWidget(btn, index // 2, index % 2)
            self.preset_buttons[key] = btn
        box.addLayout(presets)

        self.pick_height_btn = _named(
            QPushButton("Pick height"),
            "Pick height",
            "Then click a point on the model: the parting line moves to that point's Z",
        )
        self.pick_height_btn.setCheckable(True)
        self.pick_height_btn.clicked.connect(lambda: self._toggle_pick(PICK_PARTING_HEIGHT))
        box.addWidget(self.pick_height_btn)

        self.split_label = _named(QLabel("Parting line not set"), "Cope and drag readout")
        self.split_label.setWordWrap(True)
        box.addWidget(self.split_label)

        seat_row = QHBoxLayout()
        self.seat_btn = _named(
            QPushButton("Seat on parting line"),
            "Seat on parting line",
            "Move the selected riser/sprue so its bottom sits on the parting plane (cope side)",
        )
        self.seat_btn.clicked.connect(lambda: self._on_seat("above"))
        seat_row.addWidget(self.seat_btn)
        self.hang_btn = _named(
            QPushButton("Hang from parting line"),
            "Hang from parting line",
            "Move the selected runner/gate so its top touches the parting plane (drag side)",
        )
        self.hang_btn.clicked.connect(lambda: self._on_seat("below"))
        seat_row.addWidget(self.hang_btn)
        box.addLayout(seat_row)
        self.seat_hint_label = _named(QLabel(""), "Seat suggestion")
        self.seat_hint_label.setStyleSheet("font-size: 10px; color: #888;")
        self.seat_hint_label.setWordWrap(True)
        box.addWidget(self.seat_hint_label)

        group.setLayout(box)
        return group

    def rotation_step(self) -> float:
        return float(self.rotation_step_combo.currentData())

    def height_step(self) -> float:
        return float(self.height_step_combo.currentData())

    def _update_rotate_tooltips(self, *_args) -> None:
        step = self.rotation_step()
        for name, btn in self.rotate_buttons.items():
            _, axis, sign = name.split(" ")
            btn.setToolTip(f"Turn {sign}{step:g}° about the world {axis} axis, through the part's center")

    def refresh(self) -> None:
        self._updating = True
        try:
            controller = self.controller
            part = controller.target()
            has_target = part is not None
            if part is None:
                self.target_label.setText("Target: none (load a casting first)")
            else:
                kind = PART_TYPE_DISPLAY_NAMES.get(part.part_type, str(part.part_type))
                self.target_label.setText(f"Target: {part.name} ({kind})")

            for widget in (
                *self.rotate_buttons.values(),
                self.face_down_btn,
                self.face_up_btn,
                self.reset_orientation_btn,
                self.lower_part_btn,
                self.raise_part_btn,
                self.drop_to_floor_btn,
                self.bottom_z_spin,
                *self.euler_spins,
            ):
                widget.setEnabled(has_target)

            size = controller.size()
            if size is None:
                self.size_label.setText("Size (W x D x H): -")
            else:
                self.size_label.setText(
                    f"Size (W x D x H): {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} in"
                )
            if part is not None:
                for spin, value in zip(self.euler_spins, part.rotation):
                    spin.setValue(float(value))
                bottom = controller.bottom_z()
                if bottom is not None:
                    self.bottom_z_spin.setValue(bottom)

            reference = controller.reference_part()
            parting = controller.parting
            for widget in (
                self.show_parting_check,
                self.parting_z_spin,
                self.lower_parting_btn,
                self.raise_parting_btn,
                self.pick_height_btn,
                *self.preset_buttons.values(),
            ):
                widget.setEnabled(reference is not None)
            self.show_parting_check.setChecked(bool(parting and parting.visible))
            self.lock_parting_check.setChecked(parting.locked_to_casting if parting else controller.lock_default)
            if parting is not None:
                self.parting_z_spin.setValue(parting.z)
            self.split_label.setText(self._split_text())

            suggested = controller.suggested_seat_side()
            can_seat = suggested is not None and parting is not None
            self.seat_btn.setEnabled(can_seat)
            self.hang_btn.setEnabled(can_seat)
            if suggested is None:
                self.seat_hint_label.setText(
                    "Select a riser, sprue, runner or gate in Parts Manager to seat it on the parting line."
                )
            else:
                action = "Seat on parting line" if suggested == "above" else "Hang from parting line"
                kind = PART_TYPE_DISPLAY_NAMES.get(part.part_type, "") if part else ""
                self.seat_hint_label.setText(f"Suggested for {kind}: {action}.")
            self._sync_pick_buttons()
        finally:
            self._updating = False

    def _split_text(self) -> str:
        parting = self.controller.parting
        if parting is None:
            return "Parting line not set. Pick a preset or turn on Show parting line."
        split = self.controller.parting_split()
        if split is None:
            return f"Parting line Z = {parting.z:.2f} in"
        return (
            f"Parting line Z = {parting.z:.2f} in\n"
            f"Cope depth: {split['cope_depth']:.2f} in    Drag depth: {split['drag_depth']:.2f} in\n"
            f"Cope volume: {split['cope_fraction'] * 100:.0f}%    "
            f"Drag volume: {split['drag_fraction'] * 100:.0f}%"
        )

    def _geometry_done(self, changed: bool) -> None:
        if changed:
            self.geometry_changed.emit()
        self.refresh()

    def _parting_done(self, changed: bool) -> None:
        if changed:
            self.parting_changed.emit()
        self.refresh()

    def _on_rotate(self, axis: str, sign: float) -> None:
        self._geometry_done(self.controller.rotate(axis.lower(), sign * self.rotation_step()))

    def _on_keep_on_floor_toggled(self, checked: bool) -> None:
        self.controller.keep_on_floor = bool(checked)

    def _on_reset_orientation(self) -> None:
        self._geometry_done(self.controller.reset_orientation())

    def _on_euler_changed(self, *_args) -> None:
        if self._updating:
            return
        rx, ry, rz = (spin.value() for spin in self.euler_spins)
        self._geometry_done(self.controller.set_rotation(rx, ry, rz))

    def _on_move_part(self, sign: float) -> None:
        self._geometry_done(self.controller.move_z(sign * self.height_step()))

    def _on_drop_to_floor(self) -> None:
        self._geometry_done(self.controller.drop_to_floor())

    def _on_bottom_z_changed(self, value: float) -> None:
        if self._updating:
            return
        self._geometry_done(self.controller.set_bottom_z(value))

    def _on_show_parting_toggled(self, checked: bool) -> None:
        if self._updating:
            return
        self._parting_done(self.controller.set_parting_visible(bool(checked)))

    def _on_lock_toggled(self, checked: bool) -> None:
        if self._updating:
            return
        self.controller.set_parting_locked(bool(checked))

    def _on_parting_z_changed(self, value: float) -> None:
        if self._updating:
            return
        self._parting_done(self.controller.set_parting_z(value))

    def _on_nudge_parting(self, sign: float) -> None:
        self._parting_done(self.controller.nudge_parting(sign * self.height_step()))

    def _on_preset(self, key: str) -> None:
        self._parting_done(self.controller.apply_parting_preset(key))

    def _on_seat(self, side: str) -> None:
        self._geometry_done(self.controller.seat_target(side))

    def _toggle_pick(self, mode: str) -> None:
        if self.pick_mode == mode:
            self.cancel_pick()
            return
        if self.pick_mode is not None:
            self.pick_cancelled.emit()
        target = self.controller.target()
        if mode in (PICK_FACE_DOWN, PICK_FACE_UP):
            if target is None:
                self._sync_pick_buttons()
                return
            part_ids = [target.id]
        else:
            part_ids = None
        self.pick_mode = mode
        self._sync_pick_buttons()
        self.pick_requested.emit(mode, part_ids)

    def cancel_pick(self) -> None:
        if self.pick_mode is None:
            return
        self.pick_mode = None
        self._sync_pick_buttons()
        self.pick_cancelled.emit()

    def end_pick_mode(self) -> None:
        self.pick_mode = None
        self._sync_pick_buttons()

    def _sync_pick_buttons(self) -> None:
        self.face_down_btn.setChecked(self.pick_mode == PICK_FACE_DOWN)
        self.face_up_btn.setChecked(self.pick_mode == PICK_FACE_UP)
        self.pick_height_btn.setChecked(self.pick_mode == PICK_PARTING_HEIGHT)
        self.pick_row_widget.setVisible(self.pick_mode is not None)
        self.pick_hint_label.setText(_PICK_HINTS.get(self.pick_mode or "", ""))

    def handle_pick(self, payload: dict) -> None:
        mode = self.pick_mode
        self.pick_mode = None
        self._sync_pick_buttons()
        if mode == PICK_PARTING_HEIGHT:
            changed = self.controller.set_parting_z(round(float(payload["point"][2]), 4))
            if self.controller.parting is not None and not self.controller.parting.visible:
                changed = self.controller.set_parting_visible(True) or changed
            self._parting_done(changed)
        elif mode in (PICK_FACE_DOWN, PICK_FACE_UP):
            normal = outward_face_normal(payload["triangle"], payload["toward_viewer"])
            if normal is None:
                self.refresh()
                return
            direction = "down" if mode == PICK_FACE_DOWN else "up"
            self._geometry_done(self.controller.face_to(normal, direction))

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape and self.pick_mode is not None:
            self.cancel_pick()
            event.accept()
            return
        super().keyPressEvent(event)

    def hideEvent(self, event) -> None:
        self.cancel_pick()
        super().hideEvent(event)

    def store_original_values(self) -> None:
        self._snapshot = self.controller.snapshot()
        self.refresh()

    def restore_original_values(self) -> None:
        if self._snapshot is None:
            return
        if self.controller.restore(self._snapshot):
            self.geometry_changed.emit()
        self.refresh()
