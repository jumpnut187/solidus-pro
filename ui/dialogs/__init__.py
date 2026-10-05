"""Dialog components for the desktop UI."""

from .analysis_options_dialog import AnalysisOptionsDialog
from .base import DraggableDialog
from .component_visibility_dialog import ComponentVisibilityDialog
from .orientation_dialog import OrientationDialog
from .parts_dialog import PartsDialog

__all__ = [
    "AnalysisOptionsDialog",
    "ComponentVisibilityDialog",
    "DraggableDialog",
    "OrientationDialog",
    "PartsDialog",
]
