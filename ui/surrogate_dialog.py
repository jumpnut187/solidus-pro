"""Surrogate model info dialog."""

from __future__ import annotations

from typing import Optional

from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QVBoxLayout,
    QTextBrowser,
    QFormLayout,
    QComboBox,
    QSpinBox,
    QCheckBox,
    QLabel,
)

from core.surrogate.backend import backend_device_name
from core.surrogate.dataset import DEFAULT_DATASET_DIR, dataset_summary
from core.surrogate.model import SurrogateModel


class SurrogateModelInfoDialog(QDialog):
    """Displays the loaded surrogate model's manifest and capabilities."""

    def __init__(self, model: Optional[SurrogateModel], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Surrogate Model Info")
        self.resize(560, 420)

        layout = QVBoxLayout(self)
        browser = QTextBrowser(self)
        browser.setOpenExternalLinks(True)

        if model is None:
            browser.setPlainText("No surrogate model loaded.")
        else:
            device = backend_device_name(model.backend)
            lines = [
                f"<h3>{model.manifest.name}</h3>",
                f"<b>Version:</b> {model.manifest.version}",
                f"<b>Backend:</b> {model.manifest.backend} <i>({device})</i>",
                f"<b>Grid resolution:</b> {model.manifest.grid_resolution}^3",
                f"<b>Outputs:</b> {', '.join(model.manifest.outputs)}",
                f"<b>Alloy coverage:</b> {', '.join(model.manifest.alloy_coverage) or 'any'}",
                f"<b>Supported analyses:</b> "
                f"{', '.join(model.manifest.supported_analyses) or 'all schedule-derived'}",
                f"<b>Units:</b> {model.manifest.units}",
            ]
            accuracy = model.manifest.accuracy
            if accuracy:
                lines.append(
                    f"<b>Accuracy:</b> RMSE {accuracy.get('rmse_s', 0):.1f}s · "
                    f"MAE {accuracy.get('mae_s', 0):.1f}s · "
                    f"confidence {accuracy.get('confidence', 0):.2f}"
                )
            if model.manifest.notes:
                lines.append(f"<p><i>{model.manifest.notes}</i></p>")
            browser.setHtml("<br>".join(lines))

        layout.addWidget(browser)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class SurrogateSettingsDialog(QDialog):
    """Edit surrogate backend and training-data settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Surrogate Settings")
        self.setMinimumWidth(380)

        from config.settings import get_settings

        self._settings = get_settings()
        s = self._settings.surrogate

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.backend_combo = QComboBox()
        self.backend_combo.addItem("auto (best available)", "auto")
        self.backend_combo.addItem("Core ML (Apple Neural Engine)", "coreml")
        self.backend_combo.addItem("MLX (GPU)", "mlx")
        self.backend_combo.addItem("NumPy (CPU reference)", "numpy")
        idx = self.backend_combo.findData(s.backend)
        self.backend_combo.setCurrentIndex(max(idx, 0))
        form.addRow("Backend:", self.backend_combo)

        self.resolution_spin = QSpinBox()
        self.resolution_spin.setRange(16, 128)
        self.resolution_spin.setValue(int(s.grid_resolution))
        self.resolution_spin.setSingleStep(16)
        form.addRow("Grid resolution:", self.resolution_spin)

        self.auto_capture_check = QCheckBox("Capture training data after every traditional simulation")
        self.auto_capture_check.setChecked(bool(s.auto_capture))
        layout.addLayout(form)
        layout.addWidget(self.auto_capture_check)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _save(self) -> None:
        s = self._settings.surrogate
        s.backend = str(self.backend_combo.currentData())
        s.grid_resolution = int(self.resolution_spin.value())
        s.auto_capture = bool(self.auto_capture_check.isChecked())
        try:
            self._settings.save_user_override("surrogate", "backend", s.backend)
            self._settings.save_user_override("surrogate", "grid_resolution", s.grid_resolution)
            self._settings.save_user_override("surrogate", "auto_capture", s.auto_capture)
        except Exception:
            pass
        self.accept()
class SurrogateTrainDialog(QDialog):
    """Ask how to train the surrogate model from captured data."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Retrain Surrogate Model")
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)

        summary = dataset_summary(DEFAULT_DATASET_DIR)
        count = summary["count"]
        alloys = ", ".join(summary["alloys"]) or "none"
        info = QLabel(
            f"Captured training data: <b>{count}</b> run(s)\n"
            f"Alloys: {alloys}\n\n"
            "The model is fitted from these runs and saved to\n"
            "<b>models/surrogate/trained</b> (manifest.json + weights.npz)."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.generate_check = QCheckBox(
            "Add synthetic sample runs from the full solver (recommended when you have little data)"
        )
        self.generate_check.setChecked(count < 10)
        layout.addWidget(self.generate_check)

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
            "generate_first": bool(self.generate_check.isChecked()),
            "load_model": bool(self.load_check.isChecked()),
        }
