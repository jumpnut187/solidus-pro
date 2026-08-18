"""Deep learning model dialogs: train and NPU export."""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
)

from core.surrogate.dataset import DEFAULT_DATASET_DIR, dataset_summary


class DeepTrainDialog(QDialog):
    """Ask for deep training hyperparameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Train Deep Model")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)

        summary = dataset_summary(DEFAULT_DATASET_DIR)
        count = summary["count"]
        alloys = ", ".join(summary["alloys"]) or "none"
        info = QLabel(
            f"Captured training data: <b>{count}</b> run(s)\n"
            f"Alloys: {alloys}\n\n"
            "Trains the 3D U-Net (FiLM conditioning) on all captured runs at "
            "the selected grid resolution.  The model is saved to\n"
            "<b>models/deep_model/trained</b> (weights.pt + manifest.json)."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        from config.settings import get_settings

        settings = get_settings()
        d = settings.deep

        form = QFormLayout()
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(int(d.epochs))
        form.addRow("Epochs:", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(int(d.batch_size))
        form.addRow("Batch size:", self.batch_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(6)
        self.lr_spin.setRange(1e-7, 1e-1)
        self.lr_spin.setValue(float(d.learning_rate))
        self.lr_spin.setSingleStep(0.0005)
        form.addRow("Learning rate:", self.lr_spin)

        self.resolution_combo = QComboBox()
        for res in (16, 32, 64, 128):
            self.resolution_combo.addItem(f"{res}^3", res)
        idx = self.resolution_combo.findData(int(d.grid_resolution))
        self.resolution_combo.setCurrentIndex(max(idx, 0))
        form.addRow("Grid resolution:", self.resolution_combo)
        layout.addLayout(form)

        self.load_check = QCheckBox("Load the newly trained model automatically")
        self.load_check.setChecked(True)
        layout.addWidget(self.load_check)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Train Model")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def options(self) -> dict:
        """Return the user's choices as a dict."""
        return {
            "epochs": int(self.epochs_spin.value()),
            "batch_size": int(self.batch_spin.value()),
            "learning_rate": float(self.lr_spin.value()),
            "grid_resolution": int(self.resolution_combo.currentData()),
            "load_model": bool(self.load_check.isChecked()),
        }


class DeepExportDialog(QDialog):
    """Pick an NPU target and export the trained deep model."""

    TARGET_LABELS = {
        "coreml": "Apple ANE — Core ML (.mlpackage)",
        "onnx": "Snapdragon — ONNX fp16 (QNN EP)",
        "tflite": "Google Tensor — TFLite int8 (NNAPI)",
    }
    TARGET_NOTES = {
        "coreml": "Exports to a .mlpackage run on the Apple Neural Engine "
                  "(compute_units=all). Validated directly on this Mac.",
        "onnx": "Exports a fp16 ONNX graph for the Snapdragon QNN EP "
                "(Hexagon/HTP). Validated on the CPU EP in this repo.",
        "tflite": "Exports an int8 post-training-quantized TFLite model for "
                  "NNAPI / Edge TPU. Requires the onnx2tf toolchain.",
    }

    def __init__(self, parent=None, model_dir: str = "models/deep_model/trained"):
        super().__init__(parent)
        self.setWindowTitle("Export Deep Model for NPU")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.model_edit = QLineEdit(model_dir)
        form.addRow("Model:", self.model_edit)

        self.target_combo = QComboBox()
        for value, label in self.TARGET_LABELS.items():
            self.target_combo.addItem(label, value)
        form.addRow("Target:", self.target_combo)
        layout.addLayout(form)

        self.note_label = QLabel(self.TARGET_NOTES["coreml"])
        self.note_label.setWordWrap(True)
        self.note_label.setStyleSheet("color: gray;")
        layout.addWidget(self.note_label)
        self.target_combo.currentIndexChanged.connect(self._update_note)

        from config.settings import get_settings

        settings = get_settings()
        self.export_edit = QLineEdit(str(settings.deep.export_dir))
        export_form = QFormLayout()
        export_form.addRow("Output directory:", self.export_edit)
        layout.addLayout(export_form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Export")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_note(self) -> None:
        self.note_label.setText(self.TARGET_NOTES[str(self.target_combo.currentData())])

    def options(self) -> dict:
        return {
            "target": str(self.target_combo.currentData()),
            "model_dir": str(self.model_edit.text()).strip(),
            "out_dir": str(self.export_edit.text()).strip(),
        }
