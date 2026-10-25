"""Mold & Cores dialog: the explicit sand mold, automatic mold box and cores, and the sands."""

from __future__ import annotations

from typing import Any, Dict

from core.mold_materials import SAND_MATERIALS, sand_choices
from core.solidification_params import (
    CORE_RULE_DRAW,
    CORE_RULE_ENCLOSED,
    MoldParams,
    SAND_MATERIAL_CUSTOM,
)
from ui.qt import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    Qt,
    Signal,
)

from .base import DraggableDialog


def _sand_tooltip(code: int) -> str:
    sand = SAND_MATERIALS.get(code)
    if sand is None:
        return "Constant properties from the Thermal tab of the solidification parameters."
    return f"{sand.uses}\n\n{sand.notes}"


class MoldDialog(DraggableDialog):
    """Edits ``SolidificationParameters.mold``; emits the group as a dict on Confirm."""

    params_changed = Signal(dict)
    #: Emitted on any edit, so the main window can redraw the preview live.
    preview_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent, "Mold && Cores")
        self.setMinimumWidth(420)
        self._setup_ui()

    def _setup_ui(self) -> None:
        mold_group = QGroupBox("Mold")
        mold_layout = QFormLayout()
        self.auto_mold_check = QCheckBox("Generate the sand mold automatically")
        self.auto_mold_check.setToolTip(
            "Puts the mold in the thermal solve as sand cells around the assembly\n"
            "(MAGMASOFT style), instead of the semi-infinite sand boundary.\n"
            "A part typed Mold (flask, cope, drag) is used instead when loaded."
        )
        mold_layout.addRow(self.auto_mold_check)
        self.wall_spin = QDoubleSpinBox()
        self.wall_spin.setRange(0.0, 48.0)
        self.wall_spin.setSpecialValueText("Auto")
        self.wall_spin.setDecimals(2)
        self.wall_spin.setSingleStep(0.5)
        self.wall_spin.setSuffix(" in")
        self.wall_spin.setToolTip(
            "Sand between the assembly and the mold's outer faces (sides and bottom).\n"
            "Auto (0): 3 x the half-thickness of the heaviest section, 0.75-8 in -- three\n"
            "heat-penetration depths at the freeze time; sand further out never warms."
        )
        mold_layout.addRow("Wall thickness", self.wall_spin)
        self.cover_spin = QDoubleSpinBox()
        self.cover_spin.setRange(0.0, 24.0)
        self.cover_spin.setDecimals(2)
        self.cover_spin.setSingleStep(0.5)
        self.cover_spin.setSuffix(" in")
        self.cover_spin.setToolTip(
            "Sand over a closed (blind) riser. Open risers and pour cups stay open\n"
            "to the air at the top of the mold; mark a riser open in the Parts dialog."
        )
        mold_layout.addRow("Closed riser cover", self.cover_spin)
        self.mold_sand_combo = self._sand_combo()
        mold_layout.addRow("Mold sand", self.mold_sand_combo)
        mold_group.setLayout(mold_layout)
        self.content_layout.addWidget(mold_group)

        core_group = QGroupBox("Cores")
        core_layout = QFormLayout()
        self.auto_cores_check = QCheckBox("Find cores automatically")
        self.auto_cores_check.setToolTip(
            "Adds core sand where the mold cannot form the cavity.\n"
            "Loaded core parts override this: with any core part, no cores are added."
        )
        core_layout.addRow(self.auto_cores_check)
        self.core_rule_combo = QComboBox()
        self.core_rule_combo.addItem("Draw direction (undercuts, holes, cavities)", CORE_RULE_DRAW)
        self.core_rule_combo.addItem("Enclosed cavities only (MAGMASOFT)", CORE_RULE_ENCLOSED)
        self.core_rule_combo.setToolTip(
            "Draw direction: a pattern drawn vertically out of cope and drag cannot form\n"
            "sand with the assembly both above and below it.\n"
            "Enclosed: only sand the outside of the mold cannot reach."
        )
        core_layout.addRow("Rule", self.core_rule_combo)
        self.core_sand_combo = self._sand_combo()
        core_layout.addRow("Core sand", self.core_sand_combo)
        core_group.setLayout(core_layout)
        self.content_layout.addWidget(core_group)

        htc_group = QGroupBox("Interfaces")
        htc_layout = QFormLayout()
        self.htc_cast_spin = self._htc_spin(1000.0)
        self.htc_cast_spin.setToolTip("Metal to mold sand above the solidus (MAGMASOFT default C1000).")
        htc_layout.addRow("Metal to sand", self.htc_cast_spin)
        self.air_gap_check = QCheckBox("Air gap below the solidus (steel-sand curve)")
        self.air_gap_check.setToolTip(
            "Scales the metal-sand coefficient down as the skin cools below the solidus,\n"
            "along the curve Williams, Hardin and Beckermann fitted for steel in sand."
        )
        htc_layout.addRow(self.air_gap_check)
        self.htc_chill_spin = self._htc_spin(1500.0)
        htc_layout.addRow("Chill to sand", self.htc_chill_spin)
        self.ambient_spin = QDoubleSpinBox()
        self.ambient_spin.setRange(-40.0, 400.0)
        self.ambient_spin.setSuffix(" °C")
        htc_layout.addRow("Air temperature", self.ambient_spin)
        htc_group.setLayout(htc_layout)
        self.content_layout.addWidget(htc_group)

        note = QLabel(
            "Mold parts draw purple and core parts green, translucent, while you set up;\n"
            "both are hidden while a result is on screen."
        )
        note.setStyleSheet("font-size: 10px; color: #888;")
        self.content_layout.addWidget(note)
        self.content_layout.addStretch()

        for widget in (self.auto_mold_check, self.auto_cores_check, self.air_gap_check):
            widget.toggled.connect(self._changed)
        for widget in (self.wall_spin, self.cover_spin, self.htc_cast_spin, self.htc_chill_spin, self.ambient_spin):
            widget.valueChanged.connect(self._changed)
        for widget in (self.mold_sand_combo, self.core_sand_combo, self.core_rule_combo):
            widget.currentIndexChanged.connect(self._changed)
        self.set_options(MoldParams().__dict__)

    def _sand_combo(self) -> QComboBox:
        combo = QComboBox()
        for code, label in sand_choices():
            combo.addItem(label, code)
            combo.setItemData(combo.count() - 1, _sand_tooltip(code), Qt.ToolTipRole)
        return combo

    @staticmethod
    def _htc_spin(value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(1.0, 100000.0)
        spin.setDecimals(0)
        spin.setSingleStep(100.0)
        spin.setSuffix(" W/m²K")
        spin.setValue(value)
        return spin

    @staticmethod
    def _select(combo: QComboBox, value: int) -> None:
        index = combo.findData(int(value))
        combo.setCurrentIndex(index if index >= 0 else 0)

    def _changed(self, *_args) -> None:
        self.wall_spin.setEnabled(self.auto_mold_check.isChecked())
        self.cover_spin.setEnabled(self.auto_mold_check.isChecked())
        self.core_rule_combo.setEnabled(self.auto_cores_check.isChecked())
        self.preview_requested.emit(self.get_options())

    def set_options(self, options: Dict[str, Any]) -> None:
        params = MoldParams(**{k: v for k, v in (options or {}).items() if k in MoldParams.__dataclass_fields__})
        widgets = (
            self.auto_mold_check, self.auto_cores_check, self.air_gap_check, self.wall_spin, self.cover_spin,
            self.htc_cast_spin, self.htc_chill_spin, self.ambient_spin, self.mold_sand_combo,
            self.core_sand_combo, self.core_rule_combo,
        )
        for widget in widgets:
            widget.blockSignals(True)
        self.auto_mold_check.setChecked(bool(params.auto_mold))
        self.auto_cores_check.setChecked(bool(params.auto_cores))
        self.air_gap_check.setChecked(bool(params.interface_htc_curve))
        self.wall_spin.setValue(params.wall_thickness_in)
        self.cover_spin.setValue(params.closed_riser_cover_in)
        self.htc_cast_spin.setValue(params.htc_cast_mold)
        self.htc_chill_spin.setValue(params.htc_chill_mold)
        self.ambient_spin.setValue(params.ambient_temperature)
        self._select(self.mold_sand_combo, params.mold_material_code)
        self._select(self.core_sand_combo, params.core_material_code)
        self._select(self.core_rule_combo, params.core_rule_code)
        self._extra = {
            k: getattr(params, k)
            for k in ("air_htc", "sand_emissivity", "metal_emissivity", "max_mold_cells", "riser_sleeve_htc")
        }
        for widget in widgets:
            widget.blockSignals(False)
        self.wall_spin.setEnabled(self.auto_mold_check.isChecked())
        self.cover_spin.setEnabled(self.auto_mold_check.isChecked())
        self.core_rule_combo.setEnabled(self.auto_cores_check.isChecked())

    def get_options(self) -> Dict[str, Any]:
        return {
            "auto_mold": int(self.auto_mold_check.isChecked()),
            "auto_cores": int(self.auto_cores_check.isChecked()),
            "core_rule_code": int(self.core_rule_combo.currentData()),
            "wall_thickness_in": float(self.wall_spin.value()),
            "closed_riser_cover_in": float(self.cover_spin.value()),
            "mold_material_code": int(self.mold_sand_combo.currentData()),
            "core_material_code": int(self.core_sand_combo.currentData() or SAND_MATERIAL_CUSTOM),
            "htc_cast_mold": float(self.htc_cast_spin.value()),
            "interface_htc_curve": int(self.air_gap_check.isChecked()),
            "htc_chill_mold": float(self.htc_chill_spin.value()),
            "ambient_temperature": float(self.ambient_spin.value()),
            **self._extra,
        }

    def store_original_values(self) -> None:
        self._original_values = self.get_options()

    def restore_original_values(self) -> None:
        if self._original_values:
            self.set_options(self._original_values)
            self.preview_requested.emit(self._original_values)

    def apply_values(self) -> None:
        self.params_changed.emit(self.get_options())
