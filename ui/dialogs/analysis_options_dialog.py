from __future__ import annotations

from ui.qt import (
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    Signal,
)

from .base import DraggableDialog

class AnalysisOptionsDialog(DraggableDialog):

    options_changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent, "Analysis Options")
        self.setMinimumWidth(350)
        self._setup_ui()
        self._original_values = {}

    def _setup_ui(self):
        analysis_group = QGroupBox("Select Analyses to Run")
        analysis_layout = QVBoxLayout()

        self.shrinkage_check = QCheckBox("Shrink and Microporosity")
        self.shrinkage_check.setChecked(True)
        self.shrinkage_check.setToolTip(
            "Shrink: macro-shrinkage risk from feeding distance,\n"
            "geometric modulus, and solidification sequence.\n"
            "Microporosity: dispersed porosity risk from Niyama,\n"
            "feed-path blockage, and Darcy margin.\n\n"
            "Red regions indicate high risk."
        )
        analysis_layout.addWidget(self.shrinkage_check)

        self.hotspot_check = QCheckBox("Hot Spots (Last to Freeze)")
        self.hotspot_check.setChecked(True)
        self.hotspot_check.setToolTip(
            "Identifies thermal hot spots - regions that solidify last.\n"
            "Hot spots are isolated from feed metal and prone to defects.\n\n"
            "Yellow/orange regions show areas of concern."
        )
        analysis_layout.addWidget(self.hotspot_check)

        self.niyama_check = QCheckBox("Niyama Criterion")
        self.niyama_check.setChecked(False)
        self.niyama_check.setToolTip(
            "Niyama criterion: Ny = G / √R\n"
            "G = thermal gradient (°C/cm)\n"
            "R = cooling rate (°C/s)\n\n"
            "Low Niyama values (< 1.0) indicate centerline shrinkage risk.\n"
            "Material-specific thresholds are applied automatically."
        )
        analysis_layout.addWidget(self.niyama_check)

        self.hot_tear_check = QCheckBox("Hot Tear Risk")
        self.hot_tear_check.setChecked(False)
        self.hot_tear_check.setToolTip(
            "Hot Tear Susceptibility Analysis\n\n"
            "Hot tears (hot cracks) form during final solidification\n"
            "when the alloy is in the Brittle Temperature Range (BTR).\n\n"
            "WHERE HOT TEARS APPEAR:\n"
            "• Inside corners and T-junctions\n"
            "• Where thin sections meet thick sections\n"
            "• Areas with high cooling rate differences\n\n"
            "WHAT TO DO:\n"
            "• Add fillets/radii to inside corners\n"
            "• Smooth section transitions\n"
            "• Consider chills to reduce strain\n\n"
            "Material susceptibility:\n"
            "• CF8 (stainless): HIGH - wide freezing range\n"
            "• WCB (carbon steel): MODERATE\n"
            "• C958 (bronze): LOW - excellent ductility"
        )
        analysis_layout.addWidget(self.hot_tear_check)

        self.sand_burn_in_check = QCheckBox("Sand Burn-In Risk")
        self.sand_burn_in_check.setChecked(False)
        self.sand_burn_in_check.setToolTip(
            "Sand Burn-In (Metal Penetration) Analysis\n\n"
            "Sand burn-in is a SURFACE defect where molten metal\n"
            "penetrates into sand mold pores, causing rough finish.\n\n"
            "RISK FACTORS:\n"
            "• High pouring temperature (superheat)\n"
            "• High metal head (ferrostatic pressure)\n"
            "• Long contact time before solidification\n"
            "• Coarse sand / high permeability mold\n\n"
            "WHAT TO DO:\n"
            "• Reduce pouring temperature\n"
            "• Use finer sand or mold washes/coatings\n"
            "• Add chills to speed surface solidification\n"
            "• Reduce metal head height where possible\n\n"
            "Material susceptibility:\n"
            "• C958 (bronze): HIGH - excellent fluidity\n"
            "• WCB (carbon steel): MODERATE\n"
            "• CF8 (stainless): LOWER - higher viscosity"
        )
        analysis_layout.addWidget(self.sand_burn_in_check)

        self.gas_porosity_check = QCheckBox("Gas Porosity Risk")
        self.gas_porosity_check.setChecked(False)
        self.gas_porosity_check.setToolTip(
            "Gas Porosity Analysis\n\n"
            "Computes a gas porosity risk score based on mold gas\n"
            "generation and entrapment in thick sections.\n\n"
            "Three visualization modes are available:\n"
            "• Gas Generation — raw surface gas field\n"
            "• Gas Entrapment — thick-section trapping\n"
            "• Combined Gas Risk — full R_gas = SGI x E x χ x F_niyama\n\n"
            "RISK FACTORS:\n"
            "• Core surfaces generate the most gas\n"
            "• Thick, slow-freezing sections trap gas\n"
            "• Poor feeding (low Niyama) amplifies risk\n\n"
            "WHAT TO DO:\n"
            "• Improve core venting and binder selection\n"
            "• Reduce section thickness or add chills\n"
            "• Ensure adequate risers for feeding\n\n"
            "Alloy sensitivities:\n"
            "• CA6NM: HIGH (0.75)\n"
            "• CF8M: MEDIUM-HIGH (0.60)\n"
            "• WCB: MEDIUM (0.50)"
        )
        analysis_layout.addWidget(self.gas_porosity_check)

        self.filling_check = QCheckBox("Mold Filling (CFD)")
        self.filling_check.setChecked(False)
        self._filling_disabled_tooltip = (
            "Set a pour location on the 3D model first (see \"Set Pour "
            "Location\" in the sidebar) -- the mold-filling analysis needs "
            "to know where metal enters the mold."
        )
        self._filling_enabled_tooltip = (
            "Simulates mold filling (CFD) from the chosen pour location:\n"
            "melt front advance, air entrainment risk, and fill time.\n\n"
            "Runs after voxelization, before solidification."
        )
        self.set_filling_available(False)
        analysis_layout.addWidget(self.filling_check)

        self.fraction_liquid_check = QCheckBox("Fraction Liquid (Solidification Path)")
        self.fraction_liquid_check.setChecked(False)
        self.fraction_liquid_check.setToolTip(
            "Visualize the solidification path through your casting.\n\n"
            "YELLOW = Liquid metal (still molten)\n"
            "TRANSPARENT = Solid metal (cooled/solidified)\n\n"
            "Use the time slider to watch solidification:\n"
            "• Left = Start of simulated timeline (all liquid)\n"
            "• Right = End of simulated timeline\n"
            "• Labels reflect actual liquid content at each time\n\n"
            "Physics: Edges solidify first (mold contact),\n"
            "bottom solidifies faster (density), hot spots\n"
            "in thick sections stay liquid longest."
        )
        analysis_layout.addWidget(self.fraction_liquid_check)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background-color: #555;")
        analysis_layout.addWidget(sep)

        advanced_label = QLabel("Advanced Analysis:")
        advanced_label.setStyleSheet("font-weight: bold; color: #aaa; margin-top: 5px;")
        analysis_layout.addWidget(advanced_label)

        self.feeding_path_check = QCheckBox("Feeding Path Analysis")
        self.feeding_path_check.setChecked(False)
        self.feeding_path_check.setToolTip(
            "Trace liquid metal feeding paths from risers.\n\n"
            "Traces liquid metal flow paths from risers to casting.\n"
            "Identifies blocked regions that cannot be fed.\n\n"
            "Combines with Niyama for better defect prediction:\n"
            "• Low Niyama + blocked path = HIGH RISK\n"
            "• Low Niyama + good path = MODERATE\n"
            "• Blocked path alone = DEFECT LIKELY\n\n"
            "Requires risers or uses surface as feed source."
        )
        analysis_layout.addWidget(self.feeding_path_check)

        self.stress_check = QCheckBox("Thermal Stress & Distortion")
        self.stress_check.setChecked(False)
        self.stress_check.setToolTip(
            "Predict residual stress and distortion.\n\n"
            "Calculates:\n"
            "• Thermal strain from cooling\n"
            "• Constraint strain from differential solidification\n"
            "• Effective stress magnitude\n"
            "• Crack susceptibility\n"
            "• Distortion vectors\n\n"
            "Note: Simplified model for guidance.\n"
            "For precise stress analysis, use FEA tools."
        )
        analysis_layout.addWidget(self.stress_check)

        restraint_layout = QHBoxLayout()
        restraint_label = QLabel("Sand Restraint Index:")
        restraint_label.setToolTip(
            "Mold restraint/collapse behavior for hot tearing and crack susceptibility.\n"
            "0.0 = highly collapsible mold, 1.0 = very rigid mold."
        )
        restraint_layout.addWidget(restraint_label)
        self.sand_restraint_spin = QDoubleSpinBox()
        self.sand_restraint_spin.setRange(0.0, 1.0)
        self.sand_restraint_spin.setSingleStep(0.05)
        self.sand_restraint_spin.setDecimals(2)
        self.sand_restraint_spin.setValue(0.50)
        self.sand_restraint_spin.setToolTip(
            "Neutral default is 0.50.\n"
            "Higher values increase near-surface restraint in hot tear and crack calculations."
        )
        restraint_layout.addWidget(self.sand_restraint_spin)
        analysis_layout.addLayout(restraint_layout)

        self.microstructure_check = QCheckBox("Microstructure (SDAS) Prediction")
        self.microstructure_check.setChecked(False)
        self.microstructure_check.setToolTip(
            "Predict secondary dendrite arm spacing (SDAS)\n"
            "and mechanical properties from cooling rate.\n\n"
            "Outputs:\n"
            "• SDAS map (μm) - finer = stronger\n"
            "• Predicted UTS (MPa)\n"
            "• Fine/coarse structure zones\n\n"
            "Physics: SDAS = A × cooling_rate^(-n)\n"
            "Faster cooling → finer SDAS → higher strength"
        )
        analysis_layout.addWidget(self.microstructure_check)

        analysis_group.setLayout(analysis_layout)
        self.content_layout.addWidget(analysis_group)

        info_label = QLabel(
            "💡 Tip: Select only the analyses you need to reduce\n"
            "computation time and memory usage."
        )
        info_label.setStyleSheet("font-size: 10px; color: #888; margin-top: 10px;")
        info_label.setWordWrap(True)
        self.content_layout.addWidget(info_label)

        self.niyama_info_label = QLabel("")
        self.niyama_info_label.setStyleSheet("font-size: 10px; color: #aaa;")
        self.niyama_info_label.setWordWrap(True)
        self.content_layout.addWidget(self.niyama_info_label)

        self.content_layout.addStretch()

    def set_filling_available(self, available: bool) -> None:
        self.filling_check.setEnabled(available)
        if available:
            self.filling_check.setToolTip(self._filling_enabled_tooltip)
        else:
            self.filling_check.setChecked(False)
            self.filling_check.setToolTip(self._filling_disabled_tooltip)

    def store_original_values(self):
        self._original_values = {
            'shrinkage': self.shrinkage_check.isChecked(),
            'hotspot': self.hotspot_check.isChecked(),
            'niyama': self.niyama_check.isChecked(),
            'hot_tear': self.hot_tear_check.isChecked(),
            'sand_burn_in': self.sand_burn_in_check.isChecked(),
            'gas_porosity': self.gas_porosity_check.isChecked(),
            'fraction_liquid': self.fraction_liquid_check.isChecked(),
            'filling': self.filling_check.isChecked(),
            'feeding_path': self.feeding_path_check.isChecked(),
            'stress_analysis': self.stress_check.isChecked(),
            'microstructure': self.microstructure_check.isChecked(),
            'sand_restraint_index': self.sand_restraint_spin.value(),
        }

    def restore_original_values(self):
        if self._original_values:
            self.shrinkage_check.setChecked(self._original_values['shrinkage'])
            self.hotspot_check.setChecked(self._original_values['hotspot'])
            self.niyama_check.setChecked(self._original_values['niyama'])
            self.hot_tear_check.setChecked(self._original_values['hot_tear'])
            self.sand_burn_in_check.setChecked(self._original_values['sand_burn_in'])
            self.gas_porosity_check.setChecked(self._original_values['gas_porosity'])
            self.fraction_liquid_check.setChecked(self._original_values['fraction_liquid'])
            if self.filling_check.isEnabled():
                self.filling_check.setChecked(self._original_values.get('filling', False))
            self.feeding_path_check.setChecked(self._original_values.get('feeding_path', False))
            self.stress_check.setChecked(self._original_values.get('stress_analysis', False))
            self.microstructure_check.setChecked(self._original_values.get('microstructure', False))
            self.sand_restraint_spin.setValue(self._original_values.get('sand_restraint_index', 0.5))

    def apply_values(self):
        self.options_changed.emit(self.get_options())

    def get_options(self) -> dict:
        return {
            'shrinkage': self.shrinkage_check.isChecked(),
            'hotspot': self.hotspot_check.isChecked(),
            'niyama': self.niyama_check.isChecked(),
            'hot_tear': self.hot_tear_check.isChecked(),
            'sand_burn_in': self.sand_burn_in_check.isChecked(),
            'gas_porosity': self.gas_porosity_check.isChecked(),
            'fraction_liquid': self.fraction_liquid_check.isChecked(),
            'filling': self.filling_check.isChecked(),
            'feeding_path': self.feeding_path_check.isChecked(),
            'stress_analysis': self.stress_check.isChecked(),
            'microstructure': self.microstructure_check.isChecked(),
            'sand_restraint_index': self.sand_restraint_spin.value(),
        }

    def set_options(self, options: dict):
        self.shrinkage_check.setChecked(options.get('shrinkage', True))
        self.hotspot_check.setChecked(options.get('hotspot', True))
        self.niyama_check.setChecked(options.get('niyama', False))
        self.hot_tear_check.setChecked(options.get('hot_tear', False))
        self.sand_burn_in_check.setChecked(options.get('sand_burn_in', False))
        self.gas_porosity_check.setChecked(options.get('gas_porosity', False))
        self.fraction_liquid_check.setChecked(options.get('fraction_liquid', False))
        self.feeding_path_check.setChecked(options.get('feeding_path', False))
        self.stress_check.setChecked(options.get('stress_analysis', False))
        self.microstructure_check.setChecked(options.get('microstructure', False))
        try:
            restraint_value = float(options.get('sand_restraint_index', 0.5))
        except (TypeError, ValueError):
            restraint_value = 0.5
        self.sand_restraint_spin.setValue(max(0.0, min(1.0, restraint_value)))

    def update_niyama_info(self, alloy_name: str):
        thresholds = {
            'WCB': 1.0,
            'CF8': 0.7,
            'C958': 1.2
        }
        thresh = thresholds.get(alloy_name, 1.0)
        self.niyama_info_label.setText(
            f"Niyama threshold for {alloy_name}: < {thresh} indicates risk"
        )
