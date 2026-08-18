"""
Main Application Window
Clean UI with left sidebar for parameters
Supports multiple STL parts with positioning
Uses modal dialogs for Parts, Material, and Mesh settings

Enhanced Visualization Features:
- Cross-section/slice views
- Animation export (MP4/GIF)
- Multiple colormaps
- Color bar/legend
- Measurement tools
"""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSlider, QGroupBox,
    QFileDialog, QProgressBar, QDoubleSpinBox, QCheckBox,
    QMessageBox, QSplitter, QGridLayout, QListWidget, QListWidgetItem,
    QInputDialog, QMenu, QAction, QDialog, QDialogButtonBox, QFrame,
    QTextEdit, QApplication, QSpinBox, QTabWidget, QScrollArea,
    QPlainTextEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QWidgetAction
)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QObject, QTimer, QFileSystemWatcher
from PyQt5.QtGui import QFont, QCursor
import json
import sys
import os
import time
import gc
from typing import Optional
from datetime import datetime

import numpy as np

from core.part_manager import PartManager, MultiPartSTLLoader, PartType, PART_TYPE_COLORS
from core.materials import get_all_alloy_names, get_alloy
from core.gpu_compute import GPUComputeManager
from core.memory_limits import (
    validate_mesh_parameters,
    auto_adjust_voxel_size,
    MemorySeverity,
    get_max_voxels_for_system,
    estimate_voxel_count,
    get_system_memory_info,
    get_voxel_size_recommendations,
)
from .visualizer import CastingVisualizer, COLOR_MAPS
from .custom_slider import MarkedSlider
from .dialogs import AnalysisOptionsDialog, DraggableDialog, PartsDialog
from .project_state import build_project_state, load_project_state
from .workers import (
    DeepAutoTrainWorker,
    DeepExportWorker,
    DeepTrainWorker,
    MeshSuggestionWorker,
    SimulationWorker,
    SurrogateTrainWorker,
    SurrogateWorker,
)
from .surrogate_dialog import SurrogateModelInfoDialog
from config.hardware_config import hardware
from core.mesh_ml import MeshRecommendation
from core.solidification_params import SolidificationParameters
from core.memory_manager import reset_memory_manager

try:
    from core.error_logger import logger
except ImportError:
    from core.error_logger_dummy import logger


FRACTION_LIQUID_TIME_SLIDER_MAX = 1000

FRACTION_LIQUID_PLAYBACK_INTERVAL_MS = 90
FRACTION_LIQUID_PLAYBACK_BASE_STEP = 10
FRACTION_LIQUID_PLAYBACK_SPEEDS = (0.5, 1.0, 2.0, 4.0)

# Playback debug prints every Nth tick; the per-tick lines otherwise flood
# the console and slow the QPlainTextEdit document as it grows.
FRACTION_LIQUID_DEBUG_PRINT_INTERVAL = 10

# Cap console history so append cost stays flat during long playback sessions.
CONSOLE_MAX_BLOCKS = 5000


def should_use_console_translucency(platform: str = sys.platform) -> bool:
    """Windows often fails to show translucent Qt top-level dialogs reliably."""
    return platform != "win32"


def clamp_window_position(
    preferred_x: int,
    preferred_y: int,
    window_width: int,
    window_height: int,
    screen_x: int,
    screen_y: int,
    screen_width: int,
    screen_height: int,
    margin: int = 20,
) -> tuple[int, int]:
    """Keep a floating window inside a screen's available geometry."""
    min_x = screen_x + margin
    min_y = screen_y + margin
    max_x = screen_x + max(margin, screen_width - window_width - margin)
    max_y = screen_y + max(margin, screen_height - window_height - margin)
    return (
        max(min_x, min(preferred_x, max_x)),
        max(min_y, min(preferred_y, max_y)),
    )



# ==================== Material Properties Dialog ====================

class MaterialDialog(DraggableDialog):
    """Modal dialog for material selection with detailed property display"""
    
    alloy_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent, "Material Properties")
        self.setMinimumWidth(380)
        self.setMinimumHeight(500)
        self._setup_ui()
        self._original_alloy = None
    
    def _setup_ui(self):
        """Set up the material dialog UI"""
        self.content_layout.addWidget(QLabel("Select Alloy:"))
        
        self.alloy_combo = QComboBox()
        self.alloy_combo.addItems(get_all_alloy_names())
        self.alloy_combo.currentTextChanged.connect(self._on_alloy_changed)
        self.content_layout.addWidget(self.alloy_combo)
        
        # Alloy info display - with scroll area for detailed info
        self.alloy_info_label = QLabel()
        self.alloy_info_label.setWordWrap(True)
        self.alloy_info_label.setStyleSheet("font-size: 10px; color: #bbb; margin-top: 10px; line-height: 1.3;")
        self.content_layout.addWidget(self.alloy_info_label)
        
        self.content_layout.addStretch()
        
        # Update initial display
        self._update_alloy_info(self.alloy_combo.currentText())
    
    def _on_alloy_changed(self, alloy_name):
        """Update display when alloy changes"""
        self._update_alloy_info(alloy_name)
    
    def _update_alloy_info(self, alloy_name):
        """Update alloy info display with calculated properties"""
        try:
            alloy = get_alloy(alloy_name)
            
            # Basic properties
            info_lines = [
                f"<b>Temperature Range:</b>",
                f"  Liquidus: {alloy.liquidus_temp}°C",
                f"  Solidus: {alloy.solidus_temp}°C",
                f"  Freezing Range: {alloy.liquidus_temp - alloy.solidus_temp}°C",
                f"  Pouring: {alloy.pouring_temp}°C",
                "",
                f"<b>Shrinkage Properties:</b>",
                f"  Volumetric: {alloy.volumetric_shrinkage:.1f}%",
                f"  Linear: {alloy.linear_shrinkage:.1f}%",
                f"  Feeding Multiplier: {alloy.feeding_distance_multiplier:.1f}",
                "",
                f"<b>Thermal Properties:</b>",
                f"  Thermal Diffusivity: {alloy.thermal_diffusivity:.2e} m²/s",
                f"  Conductivity: {alloy.thermal_conductivity} W/m·K",
                f"  Latent Heat: {alloy.latent_heat/1000:.0f} kJ/kg",
                "",
                f"<b>Density Change:</b>",
                f"  Liquid: {alloy.density_liquid} kg/m³",
                f"  Solid: {alloy.density_solid} kg/m³",
                f"  Ratio: {alloy.density_change_ratio:.3f}",
                "",
                f"<b>Feeding Physics:</b>",
                f"  Critical fs: {alloy.critical_fraction_solid:.2f}",
                f"  SDAS: {alloy.dendrite_arm_spacing} μm",
            ]
            
            info = "<br>".join(info_lines)
            self.alloy_info_label.setTextFormat(Qt.RichText)
            self.alloy_info_label.setText(info)
        except Exception as e:
            self.alloy_info_label.setText(f"Unknown alloy: {e}")
    
    def store_original_values(self):
        """Store current alloy selection"""
        self._original_alloy = self.alloy_combo.currentText()
    
    def restore_original_values(self):
        """Restore original alloy on cancel"""
        if self._original_alloy:
            self.alloy_combo.setCurrentText(self._original_alloy)
    
    def apply_values(self):
        """Emit signal with selected alloy"""
        self.alloy_changed.emit(self.alloy_combo.currentText())
    
    def get_current_alloy(self):
        """Get current alloy selection"""
        return self.alloy_combo.currentText()
    
    def set_alloy(self, alloy_name):
        """Set current alloy"""
        self.alloy_combo.setCurrentText(alloy_name)


# ==================== Mesh Parameters Dialog ====================

class MeshDialog(DraggableDialog):
    """Modal dialog for mesh parameters"""
    
    settings_changed = pyqtSignal(float, int, str)  # voxel_size, refinement, memory_mode
    apply_ml_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent, "Mesh Parameters")
        self.setMinimumWidth(320)
        self._setup_ui()
        self._original_values = {}
    
    def _setup_ui(self):
        """Set up the mesh dialog UI"""
        # Voxel size
        self.content_layout.addWidget(QLabel("Base Voxel Size (inches):"))
        self.voxel_size_spin = QDoubleSpinBox()
        self.voxel_size_spin.setRange(0.02, 0.5)
        self.voxel_size_spin.setValue(0.1)
        self.voxel_size_spin.setSingleStep(0.02)
        self.voxel_size_spin.setDecimals(3)
        self.content_layout.addWidget(self.voxel_size_spin)
        
        # Refinement level
        self.content_layout.addWidget(QLabel("Refinement Level:"))
        self.refinement_spin = QDoubleSpinBox()
        self.refinement_spin.setRange(1, 4)
        self.refinement_spin.setValue(3)
        self.refinement_spin.setDecimals(0)
        self.content_layout.addWidget(self.refinement_spin)
        
        # Memory mode
        self.content_layout.addWidget(QLabel("Memory Mode:"))
        self.memory_mode_combo = QComboBox()
        self.memory_mode_combo.addItems(["Auto (Recommended)", "Force RAM", "Force Disk"])
        self.memory_mode_combo.setToolTip(
            "Auto: Automatically uses disk for large models\n"
            "Force RAM: Keep all data in RAM (faster but may crash)\n"
            "Force Disk: Use disk for simulation arrays (slower)\n\n"
            "Note: Some RAM is always needed for mesh processing,\n"
            "KD-trees, and visualization. Use larger voxel size\n"
            "to reduce overall memory usage."
        )
        self.content_layout.addWidget(self.memory_mode_combo)
        
        # ML recommendation button
        self.apply_ml_button = QPushButton("Apply Machine Learning Mesh")
        self.apply_ml_button.setEnabled(False)
        self.apply_ml_button.clicked.connect(self.apply_ml_requested.emit)
        self.content_layout.addWidget(self.apply_ml_button)
        
        # ML status label
        self.ml_status_label = QLabel("")
        self.ml_status_label.setWordWrap(True)
        self.ml_status_label.setStyleSheet("font-size: 10px; color: #888;")
        self.content_layout.addWidget(self.ml_status_label)
        
        self.content_layout.addStretch()
    
    def store_original_values(self):
        """Store current settings"""
        self._original_values = {
            'voxel_size': self.voxel_size_spin.value(),
            'refinement': int(self.refinement_spin.value()),
            'memory_mode': self.memory_mode_combo.currentIndex()
        }
    
    def restore_original_values(self):
        """Restore original settings on cancel"""
        if self._original_values:
            self.voxel_size_spin.setValue(self._original_values['voxel_size'])
            self.refinement_spin.setValue(self._original_values['refinement'])
            self.memory_mode_combo.setCurrentIndex(self._original_values['memory_mode'])
    
    def apply_values(self):
        """Emit signal with current settings"""
        memory_modes = {"Auto (Recommended)": "auto", "Force RAM": "ram", "Force Disk": "disk"}
        memory_mode = memory_modes.get(self.memory_mode_combo.currentText(), "auto")
        self.settings_changed.emit(
            self.voxel_size_spin.value(),
            int(self.refinement_spin.value()),
            memory_mode
        )
    
    def get_voxel_size(self):
        return self.voxel_size_spin.value()
    
    def get_refinement_level(self):
        return int(self.refinement_spin.value())
    
    def get_memory_mode(self):
        memory_modes = {"Auto (Recommended)": "auto", "Force RAM": "ram", "Force Disk": "disk"}
        return memory_modes.get(self.memory_mode_combo.currentText(), "auto")
    
    def set_values(self, voxel_size, refinement, memory_mode="auto"):
        """Set current values"""
        self.voxel_size_spin.setValue(voxel_size)
        self.refinement_spin.setValue(refinement)
        mode_map = {"auto": 0, "ram": 1, "disk": 2}
        self.memory_mode_combo.setCurrentIndex(mode_map.get(memory_mode, 0))



# ==================== Performance Options Dialog ====================

class PerformanceDialog(DraggableDialog):
    """Modal dialog for performance/acceleration options"""
    
    options_changed = pyqtSignal(dict)  # Emit dict of performance settings
    
    def __init__(self, parent=None):
        super().__init__(parent, "Performance Options")
        self.setMinimumWidth(400)
        self._setup_ui()
        self._original_values = {}
    
    def _setup_ui(self):
        """Set up the performance options dialog UI"""
        import platform
        
        # Get GPU status
        gpu_manager = GPUComputeManager()
        gpu_available = gpu_manager.is_gpu_available
        gpu_backend = gpu_manager.backend.name if gpu_available else "None"
        gpu_device = gpu_manager.device_name
        
        # GPU Acceleration Group
        gpu_group = QGroupBox("GPU Acceleration")
        gpu_layout = QVBoxLayout()
        
        # GPU status info with platform-specific messaging
        if gpu_available:
            if gpu_backend == 'METAL':
                gpu_status = QLabel("✓ Apple Silicon GPU active (MLX + Taichi)")
                gpu_status.setStyleSheet("color: #4a4; font-weight: bold;")
                gpu_layout.addWidget(gpu_status)
            else:
                gpu_status = QLabel(f"✓ {gpu_backend} available: {gpu_device}")
                gpu_status.setStyleSheet("color: #4a4; font-weight: bold;")
                gpu_layout.addWidget(gpu_status)
        else:
            gpu_status = QLabel("✗ GPU acceleration not available")
            gpu_status.setStyleSheet("color: #a44; font-weight: bold;")
            gpu_layout.addWidget(gpu_status)

            # Platform-specific explanation
            system = platform.system()
            if system == 'Darwin':  # macOS
                reason_label = QLabel(
                    "Metal GPU acceleration not active. Install: pip install mlx taichi  (then restart)\n"
                    "Using optimized Numba JIT (CPU) instead."
                )
            elif system == 'Windows':
                reason_label = QLabel(
                    "For NVIDIA GPUs: pip install cupy-cuda12x (requires CUDA toolkit)\n"
                    "Currently using optimized Numba JIT (CPU) acceleration."
                )
            else:
                reason_label = QLabel(
                    "Install pyopencl or cupy for GPU support.\n"
                    "Currently using optimized Numba JIT (CPU) acceleration."
                )
            reason_label.setStyleSheet("font-size: 10px; color: #888; margin-left: 10px;")
            reason_label.setWordWrap(True)
            gpu_layout.addWidget(reason_label)

            # Show Numba thread info as the fallback
            numba_threads = hardware.get_optimal_thread_count('compute')
            fallback_label = QLabel(f"✓ Numba JIT active: {numba_threads} threads")
            fallback_label.setStyleSheet("color: #6a6; font-size: 11px;")
            gpu_layout.addWidget(fallback_label)

        gpu_group.setLayout(gpu_layout)
        self.content_layout.addWidget(gpu_group)

        # Use GPU toggle (lives below the status group so it is always visible)
        self.use_gpu_check = QCheckBox("Use GPU acceleration")
        parent_opts = getattr(self.parent(), '_performance_options', {}) if self.parent() is not None else {}
        self.use_gpu_check.setChecked(bool(parent_opts.get('use_gpu', True)))
        self.use_gpu_check.setEnabled(bool(gpu_available))
        if gpu_available:
            self.use_gpu_check.setToolTip(
                "Enable GPU acceleration for distance computations, solidification,\n"
                "and containment. Disable to force CPU (Numba JIT) — useful for\n"
                "benchmarking or when GPU results are unstable."
            )
        else:
            self.use_gpu_check.setToolTip("No GPU detected — install mlx and taichi to enable.")
        self.content_layout.addWidget(self.use_gpu_check)
        
        # Optimization Features Group
        opt_group = QGroupBox("Optimization Features")
        opt_layout = QVBoxLayout()
        
        # Chunked Voxelization
        self.chunked_voxel_check = QCheckBox("Chunked Voxelization")
        self.chunked_voxel_check.setChecked(False)
        self.chunked_voxel_check.setToolTip(
            "Process voxelization in chunks for very large models.\n"
            "Reduces memory usage but adds some overhead.\n"
            "Recommended for models > 10 million voxels."
        )
        opt_layout.addWidget(self.chunked_voxel_check)
        
        # Incremental Solidification
        self.incremental_check = QCheckBox("Incremental Solidification")
        self.incremental_check.setChecked(False)
        self.incremental_check.setToolTip(
            "Track only the active solidification front instead of full arrays.\n"
            "Can reduce computation for very large models.\n"
            "Adds overhead for smaller models - use with caution."
        )
        opt_layout.addWidget(self.incremental_check)
        
        # Octree Spatial Index
        self.octree_check = QCheckBox("Octree Spatial Index")
        self.octree_check.setChecked(True)
        self.octree_check.setToolTip(
            "Use octree for O(log n) spatial queries.\n"
            "Faster than KD-tree for non-uniform voxel distributions.\n"
            "Recommended for most use cases."
        )
        opt_layout.addWidget(self.octree_check)
        
        opt_group.setLayout(opt_layout)
        self.content_layout.addWidget(opt_group)
        
        # Info about hardware
        info_group = QGroupBox("Hardware Info")
        info_layout = QVBoxLayout()
        
        info_text = QLabel(
            f"CPU: {hardware.cpu_brand}\n"
            f"Cores: {hardware.physical_cores} physical, {hardware.logical_cores} logical\n"
            f"RAM: {hardware.available_memory / (1024**3):.1f} GB\n"
            f"Numba Threads: {hardware.get_optimal_thread_count('compute')}"
        )
        info_text.setStyleSheet("font-size: 10px; color: #aaa;")
        info_layout.addWidget(info_text)
        
        info_group.setLayout(info_layout)
        self.content_layout.addWidget(info_group)
        
        # Tip
        tip_label = QLabel(
            "💡 Tip: Keep defaults unless working with very large models\n"
            "(>5M voxels). GPU acceleration helps most use cases."
        )
        tip_label.setStyleSheet("font-size: 10px; color: #888; margin-top: 10px;")
        tip_label.setWordWrap(True)
        self.content_layout.addWidget(tip_label)
        
        self.content_layout.addStretch()
    
    def store_original_values(self):
        """Store current settings"""
        self._original_values = {
            'chunked_voxelization': self.chunked_voxel_check.isChecked(),
            'incremental_solidification': self.incremental_check.isChecked(),
            'use_octree': self.octree_check.isChecked(),
            'use_gpu': self.use_gpu_check.isChecked(),
        }

    def restore_original_values(self):
        """Restore original settings on cancel"""
        if self._original_values:
            self.chunked_voxel_check.setChecked(self._original_values.get('chunked_voxelization', False))
            self.incremental_check.setChecked(self._original_values.get('incremental_solidification', False))
            self.octree_check.setChecked(self._original_values.get('use_octree', True))
            self.use_gpu_check.setChecked(self._original_values.get('use_gpu', True))

    def apply_values(self):
        """Emit signal with current settings"""
        self.options_changed.emit(self.get_options())

    def get_options(self) -> dict:
        """Get current performance options"""
        return {
            'chunked_voxelization': self.chunked_voxel_check.isChecked(),
            'incremental_solidification': self.incremental_check.isChecked(),
            'use_octree': self.octree_check.isChecked(),
            'use_gpu': self.use_gpu_check.isChecked(),
        }

    def set_options(self, options: dict):
        """Set performance options from dict"""
        self.chunked_voxel_check.setChecked(options.get('chunked_voxelization', False))
        self.incremental_check.setChecked(options.get('incremental_solidification', False))
        self.octree_check.setChecked(options.get('use_octree', True))
        self.use_gpu_check.setChecked(options.get('use_gpu', True))


# ==================== Visualization Settings Dialog ====================

class VisualizationDialog(DraggableDialog):
    """Modal dialog for visualization settings including cross-sections, colormaps, and legend"""
    
    colormap_changed = pyqtSignal(str)
    color_scale_changed = pyqtSignal(float, float)
    cross_section_changed = pyqtSignal(str, bool, float, int)  # axis, enabled, position, direction
    cross_section_interaction_started = pyqtSignal(str)  # axis
    cross_section_interaction_ended = pyqtSignal(str)  # axis
    show_legend_changed = pyqtSignal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent, "Visualization Settings")
        self.setMinimumWidth(400)
        self.setMinimumHeight(500)
        # Direction state per axis: +1 (default) or -1 (flipped)
        self.x_direction = 1
        self.y_direction = 1
        self.z_direction = 1
        self._setup_ui()
        self._original_values = {}
    
    def _setup_ui(self):
        """Set up the visualization settings dialog UI"""
        # Create tab widget
        tabs = QTabWidget()
        
        # ==================== Cross-Section Tab ====================
        cross_section_tab = QWidget()
        cs_layout = QVBoxLayout()
        
        cs_info = QLabel(
            "Cross-section views let you see inside the model.\n"
            "Enable a cutting plane, then drag the slider to move it.\n"
            "The blue plane indicator is shown while adjusting and hidden when you let go."
        )
        cs_info.setStyleSheet("font-size: 10px; color: #888;")
        cs_info.setWordWrap(True)
        cs_layout.addWidget(cs_info)
        
        # X-axis plane
        x_group = QGroupBox("X Cutting Plane")
        x_layout = QVBoxLayout()
        
        self.x_plane_check = QCheckBox("Enable X Plane")
        self.x_plane_check.stateChanged.connect(lambda s: self._on_plane_toggle('X', s))
        x_layout.addWidget(self.x_plane_check)
        
        x_slider_layout = QHBoxLayout()
        x_slider_layout.addWidget(QLabel("Position:"))
        self.x_plane_slider = QSlider(Qt.Horizontal)
        self.x_plane_slider.setRange(0, 1000)
        self.x_plane_slider.setValue(500)
        self.x_plane_slider.valueChanged.connect(lambda v: self._on_plane_position('X', v))
        self.x_plane_slider.sliderPressed.connect(lambda: self.cross_section_interaction_started.emit('X'))
        self.x_plane_slider.sliderReleased.connect(lambda: self.cross_section_interaction_ended.emit('X'))
        self.x_plane_slider.setEnabled(False)
        x_slider_layout.addWidget(self.x_plane_slider)
        self.x_pos_label = QLabel("50%")
        x_slider_layout.addWidget(self.x_pos_label)
        x_layout.addLayout(x_slider_layout)
        
        self.x_flip_check = QCheckBox("Flip Direction")
        self.x_flip_check.setChecked(False)
        self.x_flip_check.stateChanged.connect(lambda s: self._on_direction_changed('X', s))
        x_layout.addWidget(self.x_flip_check)

        x_group.setLayout(x_layout)
        cs_layout.addWidget(x_group)

        # Y-axis plane
        y_group = QGroupBox("Y Cutting Plane")
        y_layout = QVBoxLayout()
        
        self.y_plane_check = QCheckBox("Enable Y Plane")
        self.y_plane_check.stateChanged.connect(lambda s: self._on_plane_toggle('Y', s))
        y_layout.addWidget(self.y_plane_check)
        
        y_slider_layout = QHBoxLayout()
        y_slider_layout.addWidget(QLabel("Position:"))
        self.y_plane_slider = QSlider(Qt.Horizontal)
        self.y_plane_slider.setRange(0, 1000)
        self.y_plane_slider.setValue(500)
        self.y_plane_slider.valueChanged.connect(lambda v: self._on_plane_position('Y', v))
        self.y_plane_slider.sliderPressed.connect(lambda: self.cross_section_interaction_started.emit('Y'))
        self.y_plane_slider.sliderReleased.connect(lambda: self.cross_section_interaction_ended.emit('Y'))
        self.y_plane_slider.setEnabled(False)
        y_slider_layout.addWidget(self.y_plane_slider)
        self.y_pos_label = QLabel("50%")
        y_slider_layout.addWidget(self.y_pos_label)
        y_layout.addLayout(y_slider_layout)
        
        self.y_flip_check = QCheckBox("Flip Direction")
        self.y_flip_check.setChecked(False)
        self.y_flip_check.stateChanged.connect(lambda s: self._on_direction_changed('Y', s))
        y_layout.addWidget(self.y_flip_check)

        y_group.setLayout(y_layout)
        cs_layout.addWidget(y_group)

        # Z-axis plane
        z_group = QGroupBox("Z Cutting Plane")
        z_layout = QVBoxLayout()
        
        self.z_plane_check = QCheckBox("Enable Z Plane")
        self.z_plane_check.stateChanged.connect(lambda s: self._on_plane_toggle('Z', s))
        z_layout.addWidget(self.z_plane_check)
        
        z_slider_layout = QHBoxLayout()
        z_slider_layout.addWidget(QLabel("Position:"))
        self.z_plane_slider = QSlider(Qt.Horizontal)
        self.z_plane_slider.setRange(0, 1000)
        self.z_plane_slider.setValue(500)
        self.z_plane_slider.valueChanged.connect(lambda v: self._on_plane_position('Z', v))
        self.z_plane_slider.sliderPressed.connect(lambda: self.cross_section_interaction_started.emit('Z'))
        self.z_plane_slider.sliderReleased.connect(lambda: self.cross_section_interaction_ended.emit('Z'))
        self.z_plane_slider.setEnabled(False)
        z_slider_layout.addWidget(self.z_plane_slider)
        self.z_pos_label = QLabel("50%")
        z_slider_layout.addWidget(self.z_pos_label)
        z_layout.addLayout(z_slider_layout)
        
        self.z_flip_check = QCheckBox("Flip Direction")
        self.z_flip_check.setChecked(False)
        self.z_flip_check.stateChanged.connect(lambda s: self._on_direction_changed('Z', s))
        z_layout.addWidget(self.z_flip_check)

        z_group.setLayout(z_layout)
        cs_layout.addWidget(z_group)

        cs_layout.addStretch()
        cross_section_tab.setLayout(cs_layout)
        tabs.addTab(cross_section_tab, "Cross-Section")
        
        # ==================== Colors Tab ====================
        colors_tab = QWidget()
        colors_layout = QVBoxLayout()
        
        # Colormap selection
        cmap_group = QGroupBox("Color Map")
        cmap_layout = QVBoxLayout()
        
        cmap_layout.addWidget(QLabel("Select color scheme:"))
        self.colormap_combo = QComboBox()
        for key, value in COLOR_MAPS.items():
            self.colormap_combo.addItem(value['name'], key)
        self.colormap_combo.currentIndexChanged.connect(self._on_colormap_changed)
        cmap_layout.addWidget(self.colormap_combo)
        
        self.colormap_desc_label = QLabel("")
        self.colormap_desc_label.setStyleSheet("font-size: 10px; color: #888;")
        self.colormap_desc_label.setWordWrap(True)
        cmap_layout.addWidget(self.colormap_desc_label)
        self._update_colormap_description()
        
        cmap_group.setLayout(cmap_layout)
        colors_layout.addWidget(cmap_group)
        
        # Color scale bounds
        scale_group = QGroupBox("Color Scale Bounds")
        scale_layout = QGridLayout()
        
        scale_layout.addWidget(QLabel("Minimum:"), 0, 0)
        self.scale_min_spin = QDoubleSpinBox()
        self.scale_min_spin.setRange(0.0, 1.0)
        self.scale_min_spin.setValue(0.0)
        self.scale_min_spin.setSingleStep(0.05)
        self.scale_min_spin.setDecimals(2)
        self.scale_min_spin.valueChanged.connect(self._on_scale_changed)
        scale_layout.addWidget(self.scale_min_spin, 0, 1)
        
        scale_layout.addWidget(QLabel("Maximum:"), 1, 0)
        self.scale_max_spin = QDoubleSpinBox()
        self.scale_max_spin.setRange(0.0, 1.0)
        self.scale_max_spin.setValue(1.0)
        self.scale_max_spin.setSingleStep(0.05)
        self.scale_max_spin.setDecimals(2)
        self.scale_max_spin.valueChanged.connect(self._on_scale_changed)
        scale_layout.addWidget(self.scale_max_spin, 1, 1)
        
        self.reset_scale_btn = QPushButton("Reset to 0-1")
        self.reset_scale_btn.clicked.connect(self._reset_scale)
        scale_layout.addWidget(self.reset_scale_btn, 2, 0, 1, 2)
        
        scale_group.setLayout(scale_layout)
        colors_layout.addWidget(scale_group)
        
        colors_layout.addStretch()
        colors_tab.setLayout(colors_layout)
        tabs.addTab(colors_tab, "Colors")
        
        # ==================== Legend Tab ====================
        legend_tab = QWidget()
        legend_layout = QVBoxLayout()
        
        legend_group = QGroupBox("Color Bar Legend")
        lg_layout = QVBoxLayout()
        
        self.show_legend_check = QCheckBox("Show Color Bar")
        self.show_legend_check.setChecked(False)
        self.show_legend_check.stateChanged.connect(self._on_legend_toggle)
        lg_layout.addWidget(self.show_legend_check)
        
        lg_layout.addWidget(QLabel("Legend Title:"))
        self.legend_title_combo = QComboBox()
        self.legend_title_combo.addItems([
            "Intensity", "Shrinkage/Porosity Risk", "Defect Risk",
            "Niyama Criterion", "Hot Spot Intensity", "Fraction Liquid"
        ])
        lg_layout.addWidget(self.legend_title_combo)
        
        legend_info = QLabel(
            "The color bar shows the relationship between\n"
            "colors and values in the visualization."
        )
        legend_info.setStyleSheet("font-size: 10px; color: #888;")
        lg_layout.addWidget(legend_info)
        
        legend_group.setLayout(lg_layout)
        legend_layout.addWidget(legend_group)
        
        legend_layout.addStretch()
        legend_tab.setLayout(legend_layout)
        tabs.addTab(legend_tab, "Legend")
        
        self.content_layout.addWidget(tabs)
    
    def _on_plane_toggle(self, axis: str, state: int):
        """Handle plane enable/disable"""
        enabled = state == Qt.Checked

        # Enable/disable slider
        slider = getattr(self, f"{axis.lower()}_plane_slider")
        slider.setEnabled(enabled)

        # Emit signal with current direction
        position = slider.value() / 1000.0
        direction = getattr(self, f"{axis.lower()}_direction")
        self.cross_section_changed.emit(axis, enabled, position, direction)
    
    def _on_plane_position(self, axis: str, value: int):
        """Handle plane position change"""
        position = value / 1000.0  # 0-1 range

        # Update label
        label = getattr(self, f"{axis.lower()}_pos_label")
        label.setText(f"{int(position * 100)}%")

        # Emit signal if enabled
        check = getattr(self, f"{axis.lower()}_plane_check")
        if check.isChecked():
            direction = getattr(self, f"{axis.lower()}_direction")
            self.cross_section_changed.emit(axis, True, position, direction)
    
    def _on_direction_changed(self, axis: str, state: int):
        """Handle flip direction change"""
        direction = -1 if state == Qt.Checked else 1
        setattr(self, f"{axis.lower()}_direction", direction)

        # Re-emit with new direction if plane is currently enabled
        check = getattr(self, f"{axis.lower()}_plane_check")
        slider = getattr(self, f"{axis.lower()}_plane_slider")
        if check.isChecked():
            position = slider.value() / 1000.0
            self.cross_section_changed.emit(axis, True, position, direction)

    def _on_colormap_changed(self, index: int):
        """Handle colormap selection change"""
        key = self.colormap_combo.currentData()
        self._update_colormap_description()
        self.colormap_changed.emit(key)
    
    def _update_colormap_description(self):
        """Update colormap description label"""
        key = self.colormap_combo.currentData()
        if key and key in COLOR_MAPS:
            self.colormap_desc_label.setText(COLOR_MAPS[key]['description'])
    
    def _on_scale_changed(self):
        """Handle color scale bounds change"""
        min_val = self.scale_min_spin.value()
        max_val = self.scale_max_spin.value()
        if min_val < max_val:
            self.color_scale_changed.emit(min_val, max_val)
    
    def _reset_scale(self):
        """Reset color scale to 0-1"""
        self.scale_min_spin.setValue(0.0)
        self.scale_max_spin.setValue(1.0)
    
    def _on_legend_toggle(self, state: int):
        """Handle legend show/hide"""
        self.show_legend_changed.emit(state == Qt.Checked)
    
    def get_colormap(self) -> str:
        """Get current colormap key"""
        return self.colormap_combo.currentData() or 'red_gradient'
    
    def set_colormap(self, key: str):
        """Set current colormap"""
        index = self.colormap_combo.findData(key)
        if index >= 0:
            self.colormap_combo.setCurrentIndex(index)
    
    def set_bounds_range(self, axis: str, min_val: float, max_val: float):
        """Set the position range for a cross-section slider"""
        # Store for position calculations
        setattr(self, f"_{axis.lower()}_min", min_val)
        setattr(self, f"_{axis.lower()}_max", max_val)


# ==================== Measurement Dialog ====================

class MeasurementDialog(DraggableDialog):
    """Modal dialog for measurement tools"""
    
    measure_distance_requested = pyqtSignal()
    clear_measurements_requested = pyqtSignal()
    calculate_volume_requested = pyqtSignal(float)  # threshold
    dialog_closed = pyqtSignal()  # Emitted when dialog is closed
    
    def __init__(self, parent=None):
        super().__init__(parent, "Measurement Tools")
        self.setMinimumWidth(350)
        self._setup_ui()
    
    def _on_confirm(self):
        """Handle confirm button - stop measurement mode"""
        self.dialog_closed.emit()
        self.accept()
    
    def _on_cancel(self):
        """Handle cancel button - stop measurement mode"""
        self.dialog_closed.emit()
        self.reject()
    
    def closeEvent(self, event):
        """Handle window close button (X) - stop measurement mode"""
        self.dialog_closed.emit()
        super().closeEvent(event)
    
    def _setup_ui(self):
        """Set up the measurement tools dialog UI"""
        # Distance measurement
        dist_group = QGroupBox("Distance Measurement")
        dist_layout = QVBoxLayout()
        
        dist_info = QLabel(
            "Measure point-to-point distance on the model.\n"
            "Click two points on the surface to measure."
        )
        dist_info.setStyleSheet("font-size: 10px; color: #888;")
        dist_info.setWordWrap(True)
        dist_layout.addWidget(dist_info)
        
        self.measure_dist_btn = QPushButton("📏 Start Distance Measurement")
        self.measure_dist_btn.clicked.connect(self.measure_distance_requested.emit)
        dist_layout.addWidget(self.measure_dist_btn)
        
        dist_group.setLayout(dist_layout)
        self.content_layout.addWidget(dist_group)
        
        # Volume calculation
        vol_group = QGroupBox("Defect Volume Calculation")
        vol_layout = QVBoxLayout()
        
        vol_info = QLabel(
            "Calculate the total volume of defect regions\n"
            "above a specified intensity threshold."
        )
        vol_info.setStyleSheet("font-size: 10px; color: #888;")
        vol_info.setWordWrap(True)
        vol_layout.addWidget(vol_info)
        
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("Threshold:"))
        self.volume_threshold_spin = QDoubleSpinBox()
        self.volume_threshold_spin.setRange(0.0, 1.0)
        self.volume_threshold_spin.setValue(0.3)
        self.volume_threshold_spin.setSingleStep(0.05)
        thresh_layout.addWidget(self.volume_threshold_spin)
        vol_layout.addLayout(thresh_layout)
        
        self.calc_volume_btn = QPushButton("📊 Calculate Defect Volume")
        self.calc_volume_btn.clicked.connect(
            lambda: self.calculate_volume_requested.emit(self.volume_threshold_spin.value())
        )
        vol_layout.addWidget(self.calc_volume_btn)
        
        self.volume_result_label = QLabel("")
        self.volume_result_label.setStyleSheet("font-weight: bold; color: #4a4;")
        vol_layout.addWidget(self.volume_result_label)
        
        vol_group.setLayout(vol_layout)
        self.content_layout.addWidget(vol_group)
        
        # Clear measurements
        self.clear_btn = QPushButton("🗑 Clear All Measurements")
        self.clear_btn.clicked.connect(self.clear_measurements_requested.emit)
        self.content_layout.addWidget(self.clear_btn)
        
        self.content_layout.addStretch()
    
    def set_volume_result(self, volume: float):
        """Display volume calculation result"""
        self.volume_result_label.setText(f"Defect Volume: {volume:.4f} in³")


# ==================== Animation Export Dialog ====================

class AnimationExportDialog(DraggableDialog):
    """Modal dialog for animation export"""
    
    start_recording = pyqtSignal()
    stop_recording = pyqtSignal()
    export_gif = pyqtSignal(str, int)  # filepath, fps
    export_mp4 = pyqtSignal(str, int)  # filepath, fps
    
    def __init__(self, parent=None):
        super().__init__(parent, "Animation Export")
        self.setMinimumWidth(400)
        self._is_recording = False
        self._frame_count = 0
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the animation export dialog UI"""
        # Info
        info = QLabel(
            "Export solidification animations as video files.\n\n"
            "1. Click 'Start Recording' to begin capturing frames\n"
            "2. Use the time slider to advance through solidification\n"
            "3. Click 'Stop Recording' when done\n"
            "4. Export as GIF or MP4"
        )
        info.setStyleSheet("font-size: 10px; color: #888;")
        info.setWordWrap(True)
        self.content_layout.addWidget(info)
        
        # Recording controls
        rec_group = QGroupBox("Recording")
        rec_layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("⏺ Start Recording")
        self.start_btn.clicked.connect(self._start_recording)
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹ Stop Recording")
        self.stop_btn.clicked.connect(self._stop_recording)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)
        rec_layout.addLayout(btn_layout)
        
        self.status_label = QLabel("Not recording")
        self.status_label.setStyleSheet("color: #888;")
        rec_layout.addWidget(self.status_label)
        
        self.frame_count_label = QLabel("Frames: 0")
        rec_layout.addWidget(self.frame_count_label)
        
        rec_group.setLayout(rec_layout)
        self.content_layout.addWidget(rec_group)
        
        # Export settings
        export_group = QGroupBox("Export Settings")
        export_layout = QVBoxLayout()
        
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("Frame Rate (FPS):"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(10)
        fps_layout.addWidget(self.fps_spin)
        export_layout.addLayout(fps_layout)
        
        export_btn_layout = QHBoxLayout()
        self.export_gif_btn = QPushButton("Export as GIF")
        self.export_gif_btn.clicked.connect(self._export_gif)
        self.export_gif_btn.setEnabled(False)
        export_btn_layout.addWidget(self.export_gif_btn)
        
        self.export_mp4_btn = QPushButton("Export as MP4")
        self.export_mp4_btn.clicked.connect(self._export_mp4)
        self.export_mp4_btn.setEnabled(False)
        export_btn_layout.addWidget(self.export_mp4_btn)
        export_layout.addLayout(export_btn_layout)
        
        export_group.setLayout(export_layout)
        self.content_layout.addWidget(export_group)
        
        # Tips
        tips = QLabel(
            "💡 Tips:\n"
            "• GIF: Smaller file size, works everywhere\n"
            "• MP4: Better quality, requires OpenCV\n"
            "• Use 10-15 FPS for smooth playback"
        )
        tips.setStyleSheet("font-size: 10px; color: #888; margin-top: 10px;")
        tips.setWordWrap(True)
        self.content_layout.addWidget(tips)
        
        self.content_layout.addStretch()
    
    def _start_recording(self):
        """Start recording frames"""
        self._is_recording = True
        self._frame_count = 0
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.export_gif_btn.setEnabled(False)
        self.export_mp4_btn.setEnabled(False)
        self.status_label.setText("🔴 Recording...")
        self.status_label.setStyleSheet("color: #f44;")
        self.start_recording.emit()
    
    def _stop_recording(self):
        """Stop recording frames"""
        self._is_recording = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Recording stopped")
        self.status_label.setStyleSheet("color: #888;")
        self.stop_recording.emit()
        
        if self._frame_count > 0:
            self.export_gif_btn.setEnabled(True)
            self.export_mp4_btn.setEnabled(True)
    
    def update_frame_count(self, count: int):
        """Update displayed frame count"""
        self._frame_count = count
        self.frame_count_label.setText(f"Frames: {count}")
    
    def _export_gif(self):
        """Export as GIF"""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Animation as GIF", "", "GIF Files (*.gif)"
        )
        if filepath:
            if not filepath.endswith('.gif'):
                filepath += '.gif'
            self.export_gif.emit(filepath, self.fps_spin.value())
    
    def _export_mp4(self):
        """Export as MP4"""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Animation as MP4", "", "MP4 Files (*.mp4)"
        )
        if filepath:
            if not filepath.endswith('.mp4'):
                filepath += '.mp4'
            self.export_mp4.emit(filepath, self.fps_spin.value())


# ==================== Console Output Stream ====================

class ConsoleOutputStream(QObject):
    """Custom stream to capture stdout/stderr and emit signals"""
    text_written = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.buffer = ""
    
    def write(self, text):
        self.buffer += text
        self.text_written.emit(text)
    
    def flush(self):
        pass


# ==================== Help Guide Dialog ====================


class HelpGuideDialog(QDialog):
    """Dialog that documents alloy defaults and result interpretation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Help / Guide")
        self.setWindowFlags(
            Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.WindowMinMaxButtonsHint
        )
        self.setMinimumSize(860, 520)
        self.resize(980, 640)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.help_text = QTextEdit()
        self.help_text.setReadOnly(True)
        self.help_text.setFont(QFont("Consolas", 10) if sys.platform != "darwin" else QFont("Menlo", 11))
        self.help_text.setLineWrapMode(QTextEdit.WidgetWidth)
        self.help_text.setHtml(self._build_help_html(None))
        layout.addWidget(self.help_text)

        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.close)
        button_box.accepted.connect(self.close)
        layout.addWidget(button_box)
        self.setLayout(layout)

    def refresh_content(self, selected_alloy: Optional[str]) -> None:
        self.help_text.setHtml(self._build_help_html(selected_alloy))

    def _build_help_html(self, selected_alloy: Optional[str]) -> str:
        rows = []
        for alloy_name in sorted(get_all_alloy_names()):
            try:
                alloy = get_alloy(alloy_name)
            except Exception:
                continue

            current = " ★" if selected_alloy and selected_alloy == alloy.name else ""
            full_name = getattr(alloy, "full_name", None) or alloy.name
            liquidus = getattr(alloy, "liquidus_temp", 0)
            solidus = getattr(alloy, "solidus_temp", 0)
            shrinkage = getattr(alloy, "volumetric_shrinkage", 0.0)
            feeding = getattr(alloy, "feeding_distance_multiplier", 0.0)
            niyama = getattr(alloy, "niyama_threshold", 0.0)
            freeze_range = liquidus - solidus

            rows.append(
                "<tr>"
                f"<td>{alloy_name}{current}</td>"
                f"<td>{full_name}</td>"
                f"<td>{shrinkage:.2f}%</td>"
                f"<td>{feeding:.2f}x</td>"
                f"<td>{niyama:.2f}</td>"
                f"<td>{liquidus:.0f} / {solidus:.0f} / {freeze_range:.0f} °C</td>"
                "</tr>"
            )

        alloy_rows = "".join(rows) if rows else "<tr><td colspan='6'>No alloy data available.</td></tr>"
        selected_text = (
            f"<p><b>Selected Alloy:</b> {selected_alloy}</p>"
            if selected_alloy
            else "<p><b>Selected Alloy:</b> Not selected yet.</p>"
        )

        return f"""
            <h2>Help / How to Use</h2>
            <p>Use this dialog to compare materials and check what each result mode means.</p>
            {selected_text}

            <h3>Alloy Comparison</h3>
            <table border='1' cellpadding='6' cellspacing='0'>
                <thead>
                    <tr>
                        <th>Alloy</th>
                        <th>Display Name</th>
                        <th>Volumetric Shrinkage</th>
                        <th>Feeding Multiplier</th>
                        <th>Niyama Threshold</th>
                        <th>Liquidus / Solidus / Freeze Range (°C)</th>
                    </tr>
                </thead>
                <tbody>
                    {alloy_rows}
                </tbody>
            </table>

            <h3>Result View Interpretation</h3>
            <ul>
                <li><b>Model Only:</b> Geometry with no overlay. Use to verify orientation and labels.</li>
                <li><b>Shrinkage/Porosity:</b> Higher values indicate higher risk likelihood.</li>
                <li><b>Hot Spots:</b> Higher values indicate trapped late-solidification zones.</li>
                <li><b>Niyama Criterion:</b> Directional solidification risk map. Lower threshold = stricter risk view.</li>
                <li><b>Hot Tear Risk:</b> Uses brittle-temperature range behavior for tearing risk.</li>
                <li><b>Sand Burn-In:</b> Flags likely sand-contacted interfaces under high thermal stress.</li>
                <li><b>Fraction Liquid:</b> Time sweep of mushy-zone progression over simulation.</li>
            </ul>
        """


# ==================== Log Viewer Window ====================

class LogViewerDialog(QDialog):
    """Dialog to view error logs from the application"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Error Log Viewer")
        self.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.WindowMinMaxButtonsHint)
        self.setMinimumSize(800, 500)
        self.resize(900, 600)
        
        self._setup_ui()
        self._auto_refresh_timer = QTimer(self)
        self._auto_refresh_timer.timeout.connect(self._refresh_logs)
    
    def _setup_ui(self):
        """Set up the log viewer UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Header with filter controls
        header_layout = QHBoxLayout()
        
        header_layout.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "Errors Only", "Warnings & Errors", "Info & Above"])
        self.filter_combo.currentIndexChanged.connect(self._refresh_logs)
        header_layout.addWidget(self.filter_combo)
        
        header_layout.addStretch()
        
        self.auto_refresh_check = QCheckBox("Auto-refresh")
        self.auto_refresh_check.setChecked(False)
        self.auto_refresh_check.stateChanged.connect(self._toggle_auto_refresh)
        header_layout.addWidget(self.auto_refresh_check)
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_logs)
        header_layout.addWidget(self.refresh_btn)
        
        layout.addLayout(header_layout)
        
        # Tab widget for different views
        self.tab_widget = QTabWidget()
        
        # Recent entries tab (in-memory)
        self.recent_text = QTextEdit()
        self.recent_text.setReadOnly(True)
        self.recent_text.setFont(QFont("Monaco", 11) if sys.platform == "darwin" else QFont("Consolas", 10))
        self.recent_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 5px;
            }
        """)
        self.tab_widget.addTab(self.recent_text, "Recent Entries")
        
        # Full log file tab
        self.file_text = QTextEdit()
        self.file_text.setReadOnly(True)
        self.file_text.setFont(QFont("Monaco", 11) if sys.platform == "darwin" else QFont("Consolas", 10))
        self.file_text.setStyleSheet(self.recent_text.styleSheet())
        self.tab_widget.addTab(self.file_text, "Full Log File")
        
        layout.addWidget(self.tab_widget)
        
        # Log file path display
        self.path_label = QLabel("")
        self.path_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.path_label)
        
        # Button row
        btn_layout = QHBoxLayout()
        
        self.open_file_btn = QPushButton("Open in Editor")
        self.open_file_btn.clicked.connect(self._open_in_editor)
        btn_layout.addWidget(self.open_file_btn)
        
        self.open_folder_btn = QPushButton("Open Logs Folder")
        self.open_folder_btn.clicked.connect(self._open_logs_folder)
        btn_layout.addWidget(self.open_folder_btn)
        
        self.copy_btn = QPushButton("Copy Selected")
        self.copy_btn.clicked.connect(self._copy_selected)
        btn_layout.addWidget(self.copy_btn)
        
        btn_layout.addStretch()
        
        self.clear_btn = QPushButton("Clear Log")
        self.clear_btn.setStyleSheet("color: #ff6b6b;")
        self.clear_btn.clicked.connect(self._clear_log)
        btn_layout.addWidget(self.clear_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.close_btn)
        
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
    
    def showEvent(self, event):
        """Refresh logs when dialog is shown"""
        super().showEvent(event)
        self._refresh_logs()
    
    def _toggle_auto_refresh(self, state):
        """Toggle auto-refresh timer"""
        if state == Qt.Checked:
            self._auto_refresh_timer.start(2000)  # Refresh every 2 seconds
        else:
            self._auto_refresh_timer.stop()
    
    def _refresh_logs(self):
        """Refresh log display"""
        try:
            from core.error_logger import logger
            
            # Update path label
            log_path = logger.get_log_file_path()
            self.path_label.setText(f"Log file: {log_path}")
            
            # Get filter level
            filter_map = {
                0: None,      # All
                1: 'ERROR',   # Errors Only
                2: 'WARNING', # Warnings & Errors
                3: 'INFO',    # Info & Above
            }
            level_filter = filter_map.get(self.filter_combo.currentIndex())
            
            # Update recent entries tab
            entries = logger.get_recent_entries(200, level_filter)
            recent_html = self._format_entries_html(entries)
            self.recent_text.setHtml(recent_html)
            
            # Update full file tab
            file_contents = logger.get_log_file_contents(1000)
            self.file_text.setPlainText(file_contents)
            
        except ImportError:
            self.recent_text.setPlainText("Error logger module not loaded.")
            self.file_text.setPlainText("Error logger module not loaded.")
    
    def _format_entries_html(self, entries):
        """Format log entries as HTML with color coding"""
        if not entries:
            return "<p style='color: #888;'>No log entries found.</p>"
        
        html = "<style>"
        html += "body { font-family: Monaco, Consolas, monospace; font-size: 12px; }"
        html += ".entry { margin-bottom: 8px; padding: 4px; border-left: 3px solid; padding-left: 8px; }"
        html += ".DEBUG { border-color: #666; color: #888; }"
        html += ".INFO { border-color: #4fc3f7; color: #b3e5fc; }"
        html += ".WARNING { border-color: #ffb74d; color: #ffe0b2; }"
        html += ".ERROR { border-color: #ef5350; color: #ffcdd2; }"
        html += ".CRITICAL { border-color: #ff1744; color: #ff8a80; background-color: #2c1215; }"
        html += ".timestamp { color: #888; font-size: 10px; }"
        html += ".message { white-space: pre-wrap; }"
        html += "</style>"
        
        # Show newest first
        for entry in reversed(entries):
            level = entry.get('level', 'INFO')
            timestamp = entry.get('timestamp', '')[:19]  # Trim microseconds
            message = entry.get('message', '').replace('<', '&lt;').replace('>', '&gt;')
            
            html += f'<div class="entry {level}">'
            html += f'<span class="timestamp">{timestamp}</span> '
            html += f'<b>[{level}]</b><br>'
            html += f'<span class="message">{message}</span>'
            html += '</div>'
        
        return html
    
    def _open_in_editor(self):
        """Open log file in default text editor"""
        try:
            from core.error_logger import logger
            import subprocess
            log_path = str(logger.get_log_file_path())
            
            if sys.platform == 'darwin':
                subprocess.run(['open', log_path])
            elif sys.platform == 'win32':
                os.startfile(log_path)
            else:
                subprocess.run(['xdg-open', log_path])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open log file: {e}")
    
    def _open_logs_folder(self):
        """Open the logs folder in file manager"""
        try:
            from core.error_logger import LOG_DIR
            import subprocess
            
            if sys.platform == 'darwin':
                subprocess.run(['open', str(LOG_DIR)])
            elif sys.platform == 'win32':
                os.startfile(str(LOG_DIR))
            else:
                subprocess.run(['xdg-open', str(LOG_DIR)])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open logs folder: {e}")
    
    def _copy_selected(self):
        """Copy selected text to clipboard"""
        current_tab = self.tab_widget.currentWidget()
        if isinstance(current_tab, QTextEdit):
            cursor = current_tab.textCursor()
            if cursor.hasSelection():
                QApplication.clipboard().setText(cursor.selectedText())
    
    def _clear_log(self):
        """Clear the log file after confirmation"""
        reply = QMessageBox.question(
            self, "Clear Log",
            "Are you sure you want to clear the log file?\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                from core.error_logger import logger
                logger.clear_log_file()
                self._refresh_logs()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not clear log: {e}")


# ==================== Console Window ====================

class ConsoleWindow(QDialog):
    """Floating console window to display application output"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Console Output")
        self.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.WindowMinMaxButtonsHint)
        self.setMinimumSize(600, 400)
        self.resize(700, 500)
        
        if should_use_console_translucency():
            self.setAttribute(Qt.WA_TranslucentBackground)
            self.setWindowOpacity(0.85)
        
        # For dragging (optional, window already has title bar)
        self._drag_position = None
        
        # Set up UI
        self._setup_ui()
        
        # Redirect stdout/stderr
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._output_stream = ConsoleOutputStream()
        self._output_stream.text_written.connect(self._append_text)
        
        # Don't redirect immediately - let main window control this
        self._is_redirecting = False
    
    def _setup_ui(self):
        """Set up the console window UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Console text area (QPlainTextEdit: fast plain-text layout with a
        # bounded block count so appends stay cheap during long playback)
        self.console_text = QPlainTextEdit()
        self.console_text.setReadOnly(True)
        self.console_text.setMaximumBlockCount(CONSOLE_MAX_BLOCKS)
        self.console_text.document().setUndoRedoEnabled(False)
        self.console_text.setFont(QFont("Monaco", 11) if sys.platform == "darwin" else QFont("Consolas", 10))
        self.console_text.setStyleSheet("""
            QPlainTextEdit {
                background-color: rgba(30, 30, 30, 220);
                color: #d4d4d4;
                border: 1px solid rgba(80, 80, 80, 200);
                border-radius: 4px;
                padding: 5px;
                selection-background-color: #264f78;
                selection-color: #ffffff;
            }
        """)
        # Enable text selection and copying
        self.console_text.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
        )
        layout.addWidget(self.console_text)
        
        # Button row
        btn_layout = QHBoxLayout()
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_console)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(68, 68, 68, 220);
                border: 1px solid rgba(102, 102, 102, 200);
                border-radius: 4px;
                padding: 6px 12px;
                color: #ddd;
            }
            QPushButton:hover {
                background-color: rgba(85, 85, 85, 230);
            }
        """)
        btn_layout.addWidget(self.clear_btn)
        
        self.copy_btn = QPushButton("Copy All")
        self.copy_btn.clicked.connect(self._copy_all)
        self.copy_btn.setStyleSheet(self.clear_btn.styleSheet())
        btn_layout.addWidget(self.copy_btn)
        
        btn_layout.addStretch()
        
        self.scroll_lock_check = QCheckBox("Auto-scroll")
        self.scroll_lock_check.setChecked(True)
        self.scroll_lock_check.setStyleSheet("QCheckBox { color: #ddd; }")
        btn_layout.addWidget(self.scroll_lock_check)
        
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
        
        # Window styling - semi-transparent background
        self.setStyleSheet("""
            ConsoleWindow {
                background-color: rgba(43, 43, 43, 230);
                border-radius: 8px;
            }
        """)
    
    def start_capture(self):
        """Start capturing stdout/stderr"""
        if not self._is_redirecting:
            sys.stdout = self._output_stream
            sys.stderr = self._output_stream
            self._is_redirecting = True
    
    def stop_capture(self):
        """Stop capturing and restore original streams"""
        if self._is_redirecting:
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr
            self._is_redirecting = False
    
    def _append_text(self, text):
        """Append text to the console"""
        # appendPlainText appends at the end without cursor churn or reflow of
        # the whole document; the block cap drops the oldest output.
        self.console_text.appendPlainText(text)

        # Auto-scroll if enabled
        if self.scroll_lock_check.isChecked():
            scrollbar = self.console_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
    
    def _clear_console(self):
        """Clear the console output"""
        self.console_text.clear()
    
    def _copy_all(self):
        """Copy all console text to clipboard"""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.console_text.toPlainText())
        # Brief visual feedback could be added here
    
    def write_message(self, text):
        """Write a message directly (without going through stdout)"""
        self._append_text(text)
    
    def closeEvent(self, event):
        """Handle window close - just hide, don't destroy"""
        self.hide()
        event.ignore()  # Don't actually close, just hide
    
    def showEvent(self, event):
        """When shown, start capturing output"""
        self.start_capture()
        super().showEvent(event)


# ==================== Riser Design Dialog ====================

class RiserDesignDialog(DraggableDialog):
    """Modal dialog for automatic riser design using Heuvers' circle method"""
    
    add_risers_requested = pyqtSignal(list)  # List of riser recommendations to add
    
    def __init__(self, parent=None):
        super().__init__(parent, "Automatic Riser Design")
        self.setMinimumWidth(500)
        self.setMinimumHeight(600)
        self._recommendations = []
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the riser design dialog UI"""
        # Info section
        info_label = QLabel(
            "🎯 <b>Automatic Riser Design</b><br><br>"
            "This tool analyzes your casting geometry to recommend optimal riser "
            "placement using industry-standard methods:<br><br>"
            "• <b>Heuvers' Circle Method</b> - Identifies thermal hotspots<br>"
            "• <b>Caine's Method</b> - Sizes risers based on modulus<br>"
            "• <b>Pellini's Rule</b> - Calculates feeding distances"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-size: 11px; color: #ccc; margin-bottom: 10px;")
        self.content_layout.addWidget(info_label)
        
        # Parameters group
        params_group = QGroupBox("Design Parameters")
        params_layout = QGridLayout()
        
        params_layout.addWidget(QLabel("Max Risers:"), 0, 0)
        self.max_risers_spin = QSpinBox()
        self.max_risers_spin.setRange(1, 10)
        self.max_risers_spin.setValue(5)
        self.max_risers_spin.setToolTip("Maximum number of risers to recommend")
        params_layout.addWidget(self.max_risers_spin, 0, 1)
        
        params_layout.addWidget(QLabel("Modulus Factor:"), 1, 0)
        self.modulus_factor_spin = QDoubleSpinBox()
        self.modulus_factor_spin.setRange(1.0, 2.0)
        self.modulus_factor_spin.setValue(1.2)
        self.modulus_factor_spin.setSingleStep(0.1)
        self.modulus_factor_spin.setToolTip("Riser modulus / casting modulus ratio (typically 1.2)")
        params_layout.addWidget(self.modulus_factor_spin, 1, 1)
        
        params_layout.addWidget(QLabel("Feeding Constant:"), 2, 0)
        self.feeding_const_spin = QDoubleSpinBox()
        self.feeding_const_spin.setRange(2.0, 6.0)
        self.feeding_const_spin.setValue(4.5)
        self.feeding_const_spin.setSingleStep(0.5)
        self.feeding_const_spin.setToolTip("Pellini constant: 4.5 for steel, 3.0 for aluminum")
        params_layout.addWidget(self.feeding_const_spin, 2, 1)
        
        params_group.setLayout(params_layout)
        self.content_layout.addWidget(params_group)
        
        # Analyze button
        self.analyze_btn = QPushButton("🔍 Analyze Geometry")
        self.analyze_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        self.analyze_btn.clicked.connect(self._run_analysis)
        self.content_layout.addWidget(self.analyze_btn)
        
        # Results group
        results_group = QGroupBox("Recommended Risers")
        results_layout = QVBoxLayout()
        
        self.results_list = QListWidget()
        self.results_list.setMinimumHeight(150)
        self.results_list.itemClicked.connect(self._on_riser_selected)
        results_layout.addWidget(self.results_list)
        
        # Riser details
        self.details_label = QLabel("Select a riser to see details")
        self.details_label.setWordWrap(True)
        self.details_label.setStyleSheet("font-size: 10px; color: #aaa; padding: 10px; background: #333; border-radius: 4px;")
        results_layout.addWidget(self.details_label)
        
        results_group.setLayout(results_layout)
        self.content_layout.addWidget(results_group)
        
        # Coverage info
        self.coverage_label = QLabel("")
        self.coverage_label.setStyleSheet("font-size: 11px; color: #4a4; font-weight: bold;")
        self.content_layout.addWidget(self.coverage_label)
        
        # Add risers button
        self.add_risers_btn = QPushButton("➕ Add Selected Risers as Parts")
        self.add_risers_btn.setEnabled(False)
        self.add_risers_btn.setToolTip("Add recommended risers as new parts in the simulation")
        self.add_risers_btn.clicked.connect(self._add_selected_risers)
        self.content_layout.addWidget(self.add_risers_btn)
        
        # Store reference to parent for accessing mesh data
        self._voxel_mesh = None
        self._alloy = None
    
    def set_mesh_data(self, voxel_mesh, alloy):
        """Set the voxel mesh and alloy for analysis"""
        self._voxel_mesh = voxel_mesh
        self._alloy = alloy
        self.analyze_btn.setEnabled(True)
    
    def _run_analysis(self):
        """Run the riser design analysis"""
        # Get mesh data from parent window if not already set
        if self._voxel_mesh is None:
            try:
                self._prepare_mesh_from_parent()
            except Exception as e:
                QMessageBox.warning(self, "Mesh Error", f"Failed to prepare mesh: {str(e)}")
                return
        
        if self._voxel_mesh is None:
            QMessageBox.warning(self, "No Data", "Could not generate mesh data.")
            return
        
        try:
            from core.riser_design import RiserDesigner, RiserDesignParameters
            
            self.analyze_btn.setEnabled(False)
            self.analyze_btn.setText("Analyzing...")
            QApplication.processEvents()
            
            # Get parameters (all dimensions in inches to match voxel mesh units)
            # Riser sizes come from industry standard database
            voxel_size = self._voxel_mesh.base_voxel_size
            params = RiserDesignParameters(
                modulus_factor=self.modulus_factor_spin.value(),
                feeding_constant=self.feeding_const_spin.value(),
                safety_factor=1.15,
                min_diameter=0.5,   # 0.5 inch minimum
                max_diameter=12.0   # 12 inch maximum
            )
            
            # Create designer
            designer = RiserDesigner(
                voxel_positions=self._voxel_mesh.voxel_centers,
                voxel_size=self._voxel_mesh.base_voxel_size,
                neighbors=self._voxel_mesh.neighbors if hasattr(self._voxel_mesh, 'neighbors') else self._create_neighbor_array(),
                surface_mask=self._voxel_mesh.surface_voxels if hasattr(self._voxel_mesh, 'surface_voxels') else self._create_surface_mask(),
                alloy=self._alloy,
                params=params
            )
            
            # Debug: print modulus info
            modulus = designer.calculate_local_modulus()
            print(f"\n=== RISER DESIGN DEBUG ===")
            print(f"Voxel size: {self._voxel_mesh.base_voxel_size} in")
            print(f"Num voxels: {len(modulus)}")
            print(f"Modulus range: {modulus.min():.4f} to {modulus.max():.4f} in")
            print(f"Max modulus location: {self._voxel_mesh.voxel_centers[modulus.argmax()]}")
            
            # Run analysis
            self._recommendations = designer.design_risers(max_risers=self.max_risers_spin.value())
            
            # Debug: print hotspot info
            for i, hs in enumerate(designer.hotspots):
                print(f"Hotspot {i+1}: modulus={hs.modulus:.4f} in, pos={hs.center_position}")
            print(f"==========================\n")
            
            # Update results list (units are in inches)
            self.results_list.clear()
            for i, riser in enumerate(self._recommendations, 1):
                item = QListWidgetItem(
                    f"Riser {i}: Ø{riser.diameter:.2f}\" × {riser.height:.2f}\" "
                    f"@ ({riser.position[0]:.2f}, {riser.position[1]:.2f}, {riser.position[2]:.2f})"
                )
                item.setData(Qt.UserRole, i - 1)  # Store index
                item.setCheckState(Qt.Checked)
                self.results_list.addItem(item)
            
            # Update coverage (volume is in cubic inches)
            coverage = designer.analyze_coverage()
            self.coverage_label.setText(
                f"Coverage: {coverage['covered_fraction']:.1%} of casting fed | "
                f"Total riser volume: {coverage['total_riser_volume']:.2f} in³"
            )
            
            self.add_risers_btn.setEnabled(len(self._recommendations) > 0)
            
            # Print report to console
            print("\n" + designer.generate_report())
            
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"Failed to analyze geometry:\n{str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.analyze_btn.setEnabled(True)
            self.analyze_btn.setText("🔍 Analyze Geometry")
    
    def _prepare_mesh_from_parent(self):
        """Generate voxel mesh from parent window's part data"""
        from core.voxel_mesh import AdaptiveVoxelMesh
        from core.materials import get_alloy
        from core.part_manager import PartType
        import numpy as np
        
        main_window = self.parent()
        if main_window is None:
            raise ValueError("No parent window")
        
        # Get the casting part
        casting_part = None
        for part in main_window.part_manager.get_all_parts():
            if part.part_type == PartType.CAST:
                casting_part = part
                break
        
        if casting_part is None:
            parts = main_window.part_manager.get_all_parts()
            if parts:
                casting_part = parts[0]
            else:
                raise ValueError("No parts loaded")
        
        # Get the trimesh object from the part
        mesh = casting_part.get_transformed_mesh()
        bounds = np.array([mesh.bounds[0], mesh.bounds[1]]).T
        
        # Create voxel mesh
        voxel_size = main_window._current_voxel_size
        self._voxel_mesh = AdaptiveVoxelMesh(
            base_voxel_size=voxel_size,
            max_refinement_level=3
        )
        success = self._voxel_mesh.generate(mesh, bounds)
        
        if not success or self._voxel_mesh.voxel_centers is None:
            raise ValueError("Failed to create voxel mesh")
        
        # Get alloy properties
        try:
            self._alloy = get_alloy(main_window._current_alloy)
        except KeyError:
            self._alloy = get_alloy("WCB")
    
    def _create_neighbor_array(self):
        """Create a simple neighbor array if not available"""
        import numpy as np
        n = len(self._voxel_mesh.voxel_centers)
        # Return empty neighbors (analysis will still work but less accurate)
        return np.full((n, 6), -1, dtype=np.int32)
    
    def _create_surface_mask(self):
        """Create a surface mask if not available"""
        import numpy as np
        n = len(self._voxel_mesh.voxel_centers)
        # Assume all voxels could be surface (conservative)
        return np.ones(n, dtype=bool)
    
    def _on_riser_selected(self, item):
        """Show details for selected riser"""
        idx = item.data(Qt.UserRole)
        if idx is not None and idx < len(self._recommendations):
            riser = self._recommendations[idx]
            details = (
                f"<b>Riser {idx + 1} Details:</b><br><br>"
                f"<b>Type:</b> {riser.riser_type.name}<br>"
                f"<b>Shape:</b> {riser.shape.name}<br>"
                f"<b>Position:</b> ({riser.position[0]:.3f}, {riser.position[1]:.3f}, {riser.position[2]:.3f}) in<br>"
                f"<b>Diameter:</b> {riser.diameter:.3f} in<br>"
                f"<b>Height:</b> {riser.height:.3f} in<br>"
                f"<b>Volume:</b> {riser.volume:.3f} in³<br>"
                f"<b>Modulus:</b> {riser.modulus:.4f} in<br>"
                f"<b>Voxels Fed:</b> {len(riser.feeding_zone)}<br>"
                f"<b>Confidence:</b> {riser.confidence:.0%}<br><br>"
                f"<i>{riser.rationale}</i>"
            )
            self.details_label.setText(details)
    
    def _add_selected_risers(self):
        """Add checked risers as new parts"""
        selected = []
        for i in range(self.results_list.count()):
            item = self.results_list.item(i)
            if item.checkState() == Qt.Checked:
                idx = item.data(Qt.UserRole)
                if idx < len(self._recommendations):
                    selected.append(self._recommendations[idx])
        
        if selected:
            self.add_risers_requested.emit(selected)
            self.accept()


# ==================== Solidification Parameters Dialog ====================

class SolidificationParamsDialog(DraggableDialog):
    """
    Modal dialog for configuring solidification simulation parameters.

    Organized into 6 tab groups:
      - Feeding: Riser feeding distance and shrinkage reduction near risers
      - Chills: Chill thermal boost and shrinkage reduction near chills
      - Exothermic: Exothermic riser sleeve timing and effectiveness
      - Air Gap: Mold air-gap development and surface influence
      - Thermal: Mold/chill temperatures and heat-transfer coefficients
      - Timing: Chvorinov simulation time and safety factors

    Emits params_changed(dict) on Confirm with the full nested dict.
    """

    params_changed = pyqtSignal(dict)

    # Field definitions: (group_key, field_key, label, default, min, max, step, decimals, unit, tooltip)
    # A field is rendered as a slider+spin pair when its key is in SLIDER_FIELDS.
    SLIDER_FIELDS: set = {"feeding_zone_vertical_bias"}
    FIELD_DEFS: list = [
        # Feeding
        ("feeding", "base_feeding_distance", "Base Feeding Distance", 6.0, 2.0, 24.0, 0.1, 2, "inches",
         "Base riser influence radius. Final radius = this × (alloy feeding_multiplier / 3.0)."),
        ("feeding", "min_influence_factor", "Min Influence Factor", 0.6, 0.01, 1.0, 0.01, 2, "fraction",
         "At edge of riser influence, solidification is slowed to this fraction of normal rate."),
        ("feeding", "riser_max_reduction", "Riser Max Shrinkage Reduction", 0.10, 0.01, 0.5, 0.01, 2, "fraction",
         "At riser contact, only this fraction of calculated shrinkage remains."),
        ("feeding", "riser_min_reduction", "Riser Min Shrinkage Reduction", 1.0, 0.5, 1.0, 0.01, 2, "fraction",
         "At edge of feeding distance, full shrinkage (no reduction)."),
        ("feeding", "post_contact_reduction", "Feeder Neck Contact Reduction", 0.35, 0.1, 0.8, 0.01, 2, "fraction",
         "Shrinkage reduction at feeder neck contact with casting."),
        ("feeding", "disconnected_neck_floor", "Disconnected Neck Floor", 0.70, 0.3, 1.0, 0.01, 2, "fraction",
         "Minimum shrinkage reduction for feeder necks not connected to a riser system."),
        ("feeding", "shrink_scalar", "Shrink Scalar", 1.0, 0.25, 4.0, 0.05, 2, "",
         "Global Shrinkage/Porosity calibration. Values above 1 reduce displayed risk; values below 1 increase it."),
        ("feeding", "feeding_zone_vertical_bias", "Feeding-Zone +Z Bias", 0.5, 0.0, 2.0, 0.05, 2, "",
         "Strength of +Z directional solidification inside the riser feeding zone. "
         "Higher = bottom solidifies faster in the feeding zone, hardening directional "
         "freezing toward risers. 0 disables the feeding-zone extra (a constant 0.3 "
         "baseline directional bias still applies to the entire cast so it solidifies "
         "bottom-up). 2 is very aggressive. "
         "Default 0.5 matches the legacy directional-solidification behavior. "
         "Applied by the active solver (thermal_coupled or schedule_legacy); "
         "this slider's extra contribution is a no-op when no RISER voxels are present."),
        # Chills
        ("chill", "max_chill_boost", "Max Chill Boost", 3.0, 1.0, 10.0, 0.1, 1, "multiplier",
         "Maximum solidification acceleration near chills."),
        ("chill", "shrinkage_radius_multiplier", "Shrinkage Radius Multiplier", 1.5, 0.5, 5.0, 0.1, 2, "× chill thickness",
         "Shrinkage reduction zone = chill thickness × this value."),
        ("chill", "shrinkage_max_reduction", "Chill Max Shrinkage Reduction", 0.15, 0.01, 0.5, 0.01, 2, "fraction",
         "At chill contact, only this fraction of calculated shrinkage remains."),
        ("chill", "shrinkage_min_reduction", "Chill Min Shrinkage Reduction", 1.0, 0.5, 1.0, 0.01, 2, "fraction",
         "At edge of chill influence, full shrinkage (no reduction)."),
        ("chill", "thickness_fallback_voxels", "Chill Thickness Fallback", 5.0, 1.0, 20.0, 1.0, 0, "voxels",
         "When no explicit chill thickness is available, estimate as this × voxel_size."),
        # Exothermic
        ("exothermic", "ignition_delay", "Ignition Delay", 10.0, 0.0, 60.0, 1.0, 0, "seconds",
         "Seconds for exothermic sleeve to reach ignition temperature."),
        ("exothermic", "burn_duration", "Burn Duration", 120.0, 30.0, 600.0, 10.0, 0, "seconds",
         "Duration of the active exothermic reaction."),
        ("exothermic", "insulation_only_factor", "Insulation-Only Factor", 0.7, 0.1, 1.0, 0.01, 2, "fraction",
         "Post-burnout: solidification rate as fraction of normal (0.7 = 1.4× slower)."),
        ("exothermic", "full_exothermic_factor", "Full Exothermic Factor", 0.5, 0.1, 0.9, 0.01, 2, "fraction",
         "Active burn: solidification rate as fraction of normal (0.5 = 2× slower)."),
        ("exothermic", "decay_time_constant", "Decay Time Constant", 60.0, 10.0, 300.0, 5.0, 0, "seconds",
         "Time constant for exothermic effect decay after burnout."),
        # Air Gap
        ("air_gap", "time_constant", "Air Gap Time Constant", 60.0, 10.0, 600.0, 10.0, 0, "seconds",
         "Time over which the air gap develops."),
        ("air_gap", "surface_influence_depth", "Surface Influence Depth", 0.5, 0.1, 3.0, 0.1, 2, "inches",
         "How far the air-gap effect extends from the casting surface."),
        ("air_gap", "max_slowdown", "Max Air Gap Slowdown", 0.75, 0.1, 0.95, 0.01, 2, "fraction",
         "Maximum slowdown at surface due to air gap (0.75 = 25% slower)."),
        # Thermal Boundary
        ("thermal_boundary", "mold_temperature", "Mold Temperature", 21.0, 20.0, 500.0, 5.0, 0, "°C",
         "Mold surface boundary condition temperature."),
        ("thermal_boundary", "chill_temperature", "Chill Temperature", 21.0, 0.0, 100.0, 5.0, 0, "°C",
         "Chill fixed temperature boundary condition."),
        ("thermal_boundary", "htc_cast_interface", "HTC Cast Interface", 500.0, 100.0, 2000.0, 50.0, 0, "W/m²K",
         "Heat-transfer coefficient at cast-mold interface."),
        ("thermal_boundary", "htc_riser_post", "HTC Riser Post", 350.0, 100.0, 2000.0, 50.0, 0, "W/m²K",
         "Heat-transfer coefficient at riser-post-mold interface."),
        ("thermal_boundary", "htc_riser", "HTC Riser", 220.0, 100.0, 2000.0, 50.0, 0, "W/m²K",
         "Heat-transfer coefficient at riser-mold interface."),
        # Timing
        ("timing", "chvorinov_mold_constant", "Chvorinov Mold Constant", 280.0, 50.0, 1000.0, 10.0, 0, "s/in²",
         "Chvorinov mold constant B (t = B·M²). ~280 s/in² for WCB in sand."),
        ("timing", "safety_factor_with_risers", "Safety Factor (with Risers)", 2.5, 1.0, 10.0, 0.1, 2, "multiplier",
         "Time safety factor when risers are present."),
        ("timing", "safety_factor_no_risers", "Safety Factor (no Risers)", 1.5, 1.0, 10.0, 0.1, 2, "multiplier",
         "Time safety factor when no risers are present."),
        ("timing", "min_simulation_time", "Min Simulation Time", 120.0, 10.0, 600.0, 10.0, 0, "seconds",
         "Minimum simulation time for any casting."),
        ("timing", "max_simulation_time", "Max Simulation Time", 14400.0, 600.0, 86400.0, 600.0, 0, "seconds",
         "Maximum simulation time cap (thermal_coupled mode). 14400s = 4 hours."),
        ("timing", "sand_saturation_time", "Sand Saturation Time", 30.0, 5.0, 300.0, 5.0, 0, "seconds",
         "Time constant for sand heat saturation around riser posts."),
        ("timing", "max_hot_sand_slowdown", "Max Hot Sand Slowdown", 0.6, 0.1, 0.95, 0.01, 2, "fraction",
         "Maximum slowdown from heat-saturated sand (0.6 = 1.7x slower)."),
        # Hot Tear
        ("hot_tear", "mold_restraint_index", "Mold Restraint Index", 0.5, 0.0, 1.0, 0.01, 2, "fraction",
         "Mold restraint level at casting surface (0.0=free, 1.0=fully restrained)."),
        ("hot_tear", "metal_mold_friction_index", "Metal/Mold Friction", 0.5, 0.0, 1.0, 0.01, 2, "fraction",
         "Friction coefficient at metal-mold interface."),
        ("hot_tear", "core_restraint_index", "Core Restraint Index", 0.0, 0.0, 1.0, 0.01, 2, "fraction",
         "Core restraint level at core voxels (0.0=no restraint, 1.0=fully restrained)."),
        ("hot_tear", "low_melting_impurity_index", "Impurity Index", 0.0, 0.0, 1.0, 0.01, 2, "fraction",
         "Low-melting impurity level affecting hot tear susceptibility."),
        ("hot_tear", "hot_tear_zone_threshold", "Hot Tear Threshold", 0.0, 0.0, 1.0, 0.01, 2, "fraction",
         "Threshold for hot tear zone detection (0.0 = adaptive based on alloy)."),
        ("hot_tear", "healing_relief_strength", "Healing Relief Strength", 0.65, 0.0, 1.0, 0.01, 2, "fraction",
         "Feeding healing effectiveness at reducing hot tear risk."),
        ("hot_tear", "surface_restraint_depth_inches", "Surface Restraint Depth", 0.5, 0.1, 3.0, 0.1, 2, "inches",
         "Depth from surface where mold restraint applies."),
        ("hot_tear", "solid_path_search_depth", "Solid Path Search Depth", 8, 2, 20, 1, 0, "voxels",
         "Search depth for solidification path strain calculation."),
    ]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent, title="Solidification Parameters")
        self._spinboxes: dict = {}
        self._sliders: dict = {}
        self._original_values: dict = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setWindowTitle("Solidification Parameters")
        self.setMinimumSize(700, 520)

        # Tab widget
        self.tab_widget = QTabWidget()
        self.content_layout.addWidget(self.tab_widget)

        # Group fields by tab
        groups = {}
        for fd in self.FIELD_DEFS:
            grp = fd[0]
            if grp not in groups:
                groups[grp] = []
            groups[grp].append(fd)

        tab_labels = {
            "feeding": "Feeding",
            "chill": "Chills",
            "exothermic": "Exothermic Sleeves",
            "air_gap": "Air Gap",
            "thermal_boundary": "Thermal",
            "timing": "Timing",
            "hot_tear": "Hot Tear",
        }

        for grp_key, grp_fields in groups.items():
            tab = QScrollArea()
            tab.setWidgetResizable(True)
            tab.setFrameShape(QFrame.NoFrame)
            inner = QWidget()
            vlayout = QVBoxLayout(inner)
            vlayout.setContentsMargins(6, 6, 6, 6)
            vlayout.addSpacing(0)

            for fd in grp_fields:
                (_, field_key, label, default, min_v, max_v, step, decimals, unit, tip) = fd
                row = self._make_field_row(label, field_key, default, min_v, max_v, step, decimals, unit, tip)
                vlayout.addLayout(row)

            vlayout.addStretch()
            tab.setWidget(inner)
            self.tab_widget.addTab(tab, tab_labels.get(grp_key, grp_key.title()))

        # Button row
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Reset
        ))
        button_box.accepted.connect(self._on_confirm)
        button_box.rejected.connect(self._on_cancel)
        # Find Reset button and connect it
        for button in button_box.buttons():
            if button_box.buttonRole(button) == QDialogButtonBox.ResetRole:
                button.clicked.connect(self._on_reset)
                break
        self.content_layout.addWidget(button_box)

    def _make_field_row(self, label: str, field_key: str, default: float,
                        min_v: float, max_v: float, step: float,
                        decimals: int, unit: str, tooltip: str) -> QHBoxLayout:
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setMinimumWidth(220)
        lbl.setToolTip(tooltip)
        spin = QDoubleSpinBox()
        spin.setRange(min_v, max_v)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setSuffix(f" {unit}" if unit else "")
        spin.setToolTip(tooltip)
        row.addWidget(lbl)
        if field_key in self.SLIDER_FIELDS:
            slider = QSlider(Qt.Horizontal)
            slider_steps = max(2, int(round((max_v - min_v) / step)))
            slider.setRange(0, slider_steps)
            slider.setSingleStep(1)
            slider.setPageStep(max(1, slider_steps // 10))
            slider.setToolTip(tooltip)
            slider.setMinimumWidth(180)
            row.addWidget(slider, 2)
            row.addWidget(spin, 1)

            def _spin_to_slider(value: float) -> None:
                slider_steps_now = slider.maximum() - slider.minimum()
                span = max_v - min_v
                if span <= 0 or slider_steps_now <= 0:
                    return
                ratio = (value - min_v) / span
                slider.setValue(int(round(ratio * slider_steps_now)))

            def _slider_to_spin() -> None:
                slider_steps_now = slider.maximum() - slider.minimum()
                span = max_v - min_v
                if slider_steps_now <= 0:
                    return
                ratio = (slider.value() - slider.minimum()) / slider_steps_now
                spin.setValue(min_v + ratio * span)

            spin.valueChanged.connect(_spin_to_slider)
            slider.valueChanged.connect(_slider_to_spin)
            _spin_to_slider(default)
            self._sliders[(label, field_key)] = slider
        else:
            row.addWidget(spin, 1)
        self._spinboxes[(label, field_key)] = spin  # keyed by (group_label, field_key)
        return row

    def store_original_values(self) -> None:
        self._original_values: dict = self.get_options()

    def restore_original_values(self) -> None:
        if hasattr(self, "_original_values"):
            self.set_options(self._original_values)

    def apply_values(self) -> None:
        options = self.get_options()
        self.params_changed.emit(options)
        self._persist_to_user_config(options)
        self.accept()

    @staticmethod
    def _persist_to_user_config(options: dict) -> None:
        """Write solidification params to ~/.castsim/config.yaml so changes
        persist across app sessions. Only writes if yaml is importable.
        """
        try:
            import yaml
        except ImportError:
            return
        try:
            from pathlib import Path
            config_dir = Path.home() / ".castsim"
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / "config.yaml"
            existing: dict = {}
            if config_path.exists():
                try:
                    existing = yaml.safe_load(config_path.read_text()) or {}
                except Exception:
                    existing = {}
            if not isinstance(existing, dict):
                existing = {}
            solid_section = existing.get("solidification")
            if not isinstance(solid_section, dict):
                solid_section = {}
            for grp_key, grp_values in options.items():
                if not isinstance(grp_values, dict):
                    continue
                bucket = solid_section.get(grp_key)
                if not isinstance(bucket, dict):
                    bucket = {}
                bucket.update(grp_values)
                solid_section[grp_key] = bucket
            existing["solidification"] = solid_section
            config_path.write_text(yaml.safe_dump(existing, sort_keys=False))
        except Exception as exc:
            print(f"[SolidificationParamsDialog] failed to persist config: {exc}")

    def get_options(self) -> dict:
        result = {}
        for (grp_label, field_key), spin in self._spinboxes.items():
            grp = None
            for fd in self.FIELD_DEFS:
                if fd[1] == field_key:
                    grp = fd[0]
                    break
            if grp is None:
                continue
            result.setdefault(grp, {})
            result[grp][field_key] = spin.value()
        return result

    def set_options(self, options: dict) -> None:
        if options is None:
            return
        for (grp_label, field_key), spin in self._spinboxes.items():
            grp = None
            for fd in self.FIELD_DEFS:
                if fd[1] == field_key:
                    grp = fd[0]
                    break
            if grp is None:
                continue
            if grp in options and field_key in options[grp]:
                spin.setValue(options[grp][field_key])

    def _on_confirm(self) -> None:
        self.apply_values()

    def _on_cancel(self) -> None:
        self.restore_original_values()
        self.reject()

    def _on_reset(self) -> None:
        defaults = SolidificationParameters().to_dict()
        self.set_options(defaults)


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        # Multi-part support
        self.part_manager = PartManager()
        self.stl_loader = MultiPartSTLLoader()  # Backwards-compatible loader
        self.stl_loader.part_manager = self.part_manager  # Share the manager
        
        self.simulation_worker = None
        self.mesh_suggestion_worker = None
        self.current_results = None
        self.mesh_setting_origin = "manual"
        self.last_ml_recommendation: Optional[MeshRecommendation] = None
        self._applying_ml_suggestion = False
        self._selected_part_id: Optional[str] = None
        self._current_project_path: Optional[str] = None
        
        # Analysis options
        self._analysis_options = {
            'shrinkage': True,
            'hotspot': True,
            'niyama': False,
            'fraction_liquid': True,
            'sand_restraint_index': 0.5,
        }
        
        # Performance options
        self._performance_options = {
            'chunked_voxelization': False,
            'incremental_solidification': False,
            'use_octree': True,
            'part_assignment_mode': 'parallel',
            'part_assignment_workers': 0,
            'part_assignment_batch_size': 2500,
            'use_gpu': True,
        }

        # Solidification parameters (defaults merged with user-config overrides
        # at ~/.castsim/config.yaml so changes persist across app sessions).
        self._solidification_params = self._load_solidification_params_with_user_override()

        # Current visualization mode
        self._current_vis_mode = 'shrinkage'  # 'shrinkage', 'hotspot', 'niyama', 'fraction_liquid'
        
        # Fraction liquid playback state
        self._fraction_liquid_playback_timer = QTimer()
        self._fraction_liquid_playback_timer.setInterval(FRACTION_LIQUID_PLAYBACK_INTERVAL_MS)
        self._fraction_liquid_playback_timer.timeout.connect(self._on_fraction_liquid_playback_tick)
        self._is_playing_fraction_liquid = False
        self._fraction_liquid_playback_speed = 1.0
        self._fraction_liquid_render_count = 0

        # Progress tracking
        self.slice_start_times = []
        self.last_slice_index = 0
        self.simulation_start_time = None
        
        self.setWindowTitle("Solidus Pro - Casting Solidification Simulator")
        self.setGeometry(100, 100, 1400, 900)
        
        self._init_ui()
        self._create_menu_bar()
        self._autoload_surrogate_model()
        self._setup_deep_autotrain()
        self._show_hardware_info()
    
    def _init_ui(self):
        """Initialize UI components"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        self.main_splitter = splitter
        
        # Left sidebar (parameters)
        sidebar = self._create_sidebar()
        splitter.addWidget(sidebar)
        
        # Right panel (3D visualization)
        right_panel = self._create_visualization_panel()
        splitter.addWidget(right_panel)
        
        # Set initial sizes (250px sidebar, rest for visualization)
        splitter.setSizes([250, 1150])
        
        main_layout.addWidget(splitter)
        
        # Create modal dialogs
        self._create_dialogs()

        # Seed use_gpu from persisted settings and render real label
        from config.settings import settings as _settings
        self._performance_options['use_gpu'] = bool(
            getattr(_settings, 'use_gpu', self._performance_options.get('use_gpu', True))
        )
        self._update_performance_summary()
    
    def _create_menu_bar(self):
        """Create the application menu bar"""
        menubar = self.menuBar()
        
        # ==================== File Menu ====================
        file_menu = menubar.addMenu("&File")
        
        # Add Part
        self.add_part_action = QAction("📦 Add Part...", self)
        self.add_part_action.setShortcut("Ctrl+O")
        self.add_part_action.setStatusTip("Add a CAD part (STL or STEP) to the simulation")
        self.add_part_action.triggered.connect(self._add_part)
        file_menu.addAction(self.add_part_action)
        
        # Parts Manager
        self.parts_manager_action = QAction("📋 Parts Manager...", self)
        self.parts_manager_action.setShortcut("Ctrl+P")
        self.parts_manager_action.setStatusTip("Manage all loaded parts")
        self.parts_manager_action.triggered.connect(self._open_parts_dialog)
        file_menu.addAction(self.parts_manager_action)

        self.open_project_action = QAction("📂 Open Project...", self)
        self.open_project_action.setShortcut("Ctrl+Shift+O")
        self.open_project_action.setStatusTip("Open a saved Solidus project")
        self.open_project_action.triggered.connect(self._load_project)
        file_menu.addAction(self.open_project_action)

        self.save_project_action = QAction("💾 Save Project...", self)
        self.save_project_action.setShortcut("Ctrl+Shift+S")
        self.save_project_action.setStatusTip("Save the current Solidus project")
        self.save_project_action.triggered.connect(self._save_project)
        file_menu.addAction(self.save_project_action)
        
        file_menu.addSeparator()
        
        # Export Screenshot
        screenshot_action = QAction("📸 Export Screenshot...", self)
        screenshot_action.setShortcut("Ctrl+S")
        screenshot_action.setStatusTip("Save a screenshot of the current view")
        screenshot_action.triggered.connect(self._take_screenshot)
        file_menu.addAction(screenshot_action)
        
        # Export Animation
        animation_action = QAction("🎬 Export Animation...", self)
        animation_action.setStatusTip("Export solidification animation as GIF or MP4")
        animation_action.triggered.connect(self._open_animation_dialog)
        file_menu.addAction(animation_action)
        
        # Generate Report
        report_action = QAction("📄 Generate Report...", self)
        report_action.setShortcut("Ctrl+R")
        report_action.setStatusTip("Generate PDF/HTML simulation report")
        report_action.triggered.connect(self._generate_report)
        file_menu.addAction(report_action)
        
        file_menu.addSeparator()
        
        # Exit
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # ==================== Simulation Menu ====================
        sim_menu = menubar.addMenu("&Simulation")
        
        # Run Simulation
        self.run_sim_action = QAction("▶ Run Simulation", self)
        self.run_sim_action.setShortcut("F5")
        self.run_sim_action.setStatusTip("Run the solidification simulation")
        self.run_sim_action.triggered.connect(self._run_simulation)
        self.run_sim_action.setEnabled(False)
        sim_menu.addAction(self.run_sim_action)

        self.stop_sim_action = QAction("■ Stop Simulation", self)
        self.stop_sim_action.setShortcut("Shift+F5")
        self.stop_sim_action.setStatusTip("Cancel the active simulation")
        self.stop_sim_action.triggered.connect(self._cancel_simulation)
        self.stop_sim_action.setEnabled(False)
        sim_menu.addAction(self.stop_sim_action)
        
        sim_menu.addSeparator()
        
        # Material Properties
        material_action = QAction("🔧 Material Properties...", self)
        material_action.setShortcut("Ctrl+M")
        material_action.setStatusTip("Select alloy material")
        material_action.triggered.connect(self._open_material_dialog)
        sim_menu.addAction(material_action)
        
        # Mesh Parameters
        mesh_action = QAction("🔲 Mesh Parameters...", self)
        mesh_action.setStatusTip("Configure voxel mesh settings")
        mesh_action.triggered.connect(self._open_mesh_dialog)
        sim_menu.addAction(mesh_action)
        
        # Analysis Options
        analysis_action = QAction("📊 Analysis Options...", self)
        analysis_action.setStatusTip("Select analyses to run")
        analysis_action.triggered.connect(self._open_analysis_dialog)
        sim_menu.addAction(analysis_action)
        
        # Performance Options
        performance_action = QAction("⚡ Performance Options...", self)
        performance_action.setStatusTip("Configure GPU and optimization settings")
        performance_action.triggered.connect(self._open_performance_dialog)
        sim_menu.addAction(performance_action)

        # GPU Distance Chunk Size (popover)
        gpu_chunk_action = QAction("🖥 GPU Distance Chunk Size...", self)
        gpu_chunk_action.setStatusTip(
            "Tune the chunk size for GPU distance computations (RAM vs speed)"
        )
        gpu_chunk_action.triggered.connect(self._open_gpu_chunk_popover)
        sim_menu.addAction(gpu_chunk_action)

        # Solidification Parameters
        solidification_action = QAction("🔧 Solidification Parameters...", self)
        solidification_action.setStatusTip("Configure feeding distances, chill reduction, exothermic sleeve timing, and more")
        solidification_action.triggered.connect(self._open_solidification_params_dialog)
        sim_menu.addAction(solidification_action)

        # ==================== Surrogate Menu ====================
        surrogate_menu = menubar.addMenu("&Surrogate")

        self.surrogate_model = None
        self.surrogate_worker = None

        # Load Model
        self.surrogate_load_action = QAction("📥 Load Surrogate Model...", self)
        self.surrogate_load_action.setStatusTip("Load a trained surrogate solidification model (.mlmodel / .npz / .json)")
        self.surrogate_load_action.triggered.connect(self._load_surrogate_model)
        surrogate_menu.addAction(self.surrogate_load_action)

        # Model Info
        self.surrogate_info_action = QAction("ℹ️ Model Info...", self)
        self.surrogate_info_action.setStatusTip("Show the loaded surrogate model manifest")
        self.surrogate_info_action.triggered.connect(self._show_surrogate_model_info)
        self.surrogate_info_action.setEnabled(False)
        surrogate_menu.addAction(self.surrogate_info_action)

        # Settings
        surrogate_settings_action = QAction("⚙️ Surrogate Settings...", self)
        surrogate_settings_action.setStatusTip("Configure the surrogate backend and training-data capture")
        surrogate_settings_action.triggered.connect(self._open_surrogate_settings_dialog)
        surrogate_menu.addAction(surrogate_settings_action)

        # Retrain from captured data
        self.surrogate_train_action = QAction("🔁 Retrain Surrogate Model...", self)
        self.surrogate_train_action.setStatusTip("Train the surrogate model from captured simulation data")
        self.surrogate_train_action.triggered.connect(self._open_surrogate_train_dialog)
        surrogate_menu.addAction(self.surrogate_train_action)

        surrogate_menu.addSeparator()

        # Run Surrogate
        self.run_surrogate_action = QAction("▶ Run Surrogate Simulation", self)
        self.run_surrogate_action.setShortcut("Ctrl+Alt+S")
        self.run_surrogate_action.setStatusTip("Run the surrogate (NPU/ML) solidification simulation")
        self.run_surrogate_action.triggered.connect(self._run_surrogate)
        self.run_surrogate_action.setEnabled(False)
        surrogate_menu.addAction(self.run_surrogate_action)

        # Stop Surrogate
        self.stop_surrogate_action = QAction("■ Stop Surrogate", self)
        self.stop_surrogate_action.setStatusTip("Cancel the active surrogate simulation")
        self.stop_surrogate_action.triggered.connect(self._cancel_surrogate)
        self.stop_surrogate_action.setEnabled(False)
        surrogate_menu.addAction(self.stop_surrogate_action)

        # ==================== Deep Model Menu ====================
        # Optional torch module (core/deep_model).  Loading a deep model sets
        # self.surrogate_model so the existing Surrogate run path works
        # unchanged.  Auto-train watches data/surrogate_dataset for new runs.
        deep_menu = menubar.addMenu("&Deep Model")

        self.deep_worker = None  # shared by manual + auto training (mutual exclusion)
        self.deep_export_worker = None
        self.deep_exporting = False
        self.deep_dataset_watcher = None
        self.deep_debounce_timer = QTimer(self)
        self.deep_debounce_timer.setSingleShot(True)
        self.deep_debounce_timer.setInterval(10_000)
        self.deep_debounce_timer.timeout.connect(self._start_deep_auto_train)

        self.deep_train_action = QAction("🧠 Train Deep Model...", self)
        self.deep_train_action.setStatusTip("Train the 3D U-Net surrogate from captured runs (requires torch)")
        self.deep_train_action.triggered.connect(self._open_deep_train_dialog)
        deep_menu.addAction(self.deep_train_action)

        self.deep_load_action = QAction("📥 Load Deep Model...", self)
        self.deep_load_action.setStatusTip("Load a trained deep model (models/deep_model/trained)")
        self.deep_load_action.triggered.connect(self._load_deep_model)
        deep_menu.addAction(self.deep_load_action)

        self.deep_export_action = QAction("📦 Export for NPU...", self)
        self.deep_export_action.setStatusTip("Export the trained model to Apple ANE / Snapdragon / Google Tensor")
        self.deep_export_action.triggered.connect(self._export_deep_model)
        deep_menu.addAction(self.deep_export_action)

        self.deep_autotrain_action = QAction("🔁 Auto-Train on New Captures", self)
        self.deep_autotrain_action.setCheckable(True)
        self.deep_autotrain_action.setStatusTip(
            "Watch data/surrogate_dataset and retrain the deep model after new captures "
            "(cooldown + change detection; never erases captured data)"
        )
        self.deep_autotrain_action.toggled.connect(self._toggle_deep_autotrain)
        deep_menu.addAction(self.deep_autotrain_action)

        # ==================== Tools Menu ====================
        tools_menu = menubar.addMenu("&Tools")
        
        # Auto Riser Design
        self.riser_design_action = QAction("🎯 Auto Riser Design...", self)
        self.riser_design_action.setShortcut("Ctrl+Shift+R")
        self.riser_design_action.setStatusTip("Automatically design risers using Heuvers' circle method")
        self.riser_design_action.triggered.connect(self._open_riser_design_dialog)
        self.riser_design_action.setEnabled(False)  # Enable after parts loaded
        tools_menu.addAction(self.riser_design_action)
        
        tools_menu.addSeparator()
        
        # Measurement Tools
        measurement_action = QAction("📏 Measurement Tools...", self)
        measurement_action.setStatusTip("Distance and volume measurement tools")
        measurement_action.triggered.connect(self._open_measurement_dialog)
        tools_menu.addAction(measurement_action)
        
        # Defect Volume Calculator
        defect_vol_action = QAction("📊 Calculate Defect Volume", self)
        defect_vol_action.setStatusTip("Calculate total defect volume above threshold")
        defect_vol_action.triggered.connect(lambda: self._calculate_defect_volume(
            self.threshold_slider.value() / 100.0
        ))
        tools_menu.addAction(defect_vol_action)
        
        # ==================== View Menu ====================
        view_menu = menubar.addMenu("&View")
        
        # Visualization Settings
        vis_settings_action = QAction("🎨 Visualization Settings...", self)
        vis_settings_action.setStatusTip("Configure colors, cross-sections, and legend")
        vis_settings_action.triggered.connect(self._open_visualization_dialog)
        view_menu.addAction(vis_settings_action)
        
        view_menu.addSeparator()
        
        # Show/Hide Model
        self.show_model_action = QAction("Show Model", self)
        self.show_model_action.setCheckable(True)
        self.show_model_action.setChecked(True)
        self.show_model_action.triggered.connect(lambda checked: self._toggle_mesh_visibility(
            Qt.Checked if checked else Qt.Unchecked
        ))
        view_menu.addAction(self.show_model_action)
        
        # Show/Hide Results
        self.show_results_action = QAction("Show Results", self)
        self.show_results_action.setCheckable(True)
        self.show_results_action.setChecked(True)
        self.show_results_action.triggered.connect(lambda checked: self._toggle_shrinkage_visibility(
            Qt.Checked if checked else Qt.Unchecked
        ))
        view_menu.addAction(self.show_results_action)
        
        view_menu.addSeparator()
        
        # Reset View
        reset_view_action = QAction("Reset Camera View", self)
        reset_view_action.setShortcut("Home")
        reset_view_action.setStatusTip("Reset camera to default position")
        reset_view_action.triggered.connect(self._reset_camera_view)
        view_menu.addAction(reset_view_action)

        view_menu.addSeparator()

        # Projection toggle (default off = orthographic)
        self.perspective_projection_action = QAction("Perspective Projection", self)
        self.perspective_projection_action.setCheckable(True)
        self.perspective_projection_action.setChecked(False)
        self.perspective_projection_action.setStatusTip("Switch between orthographic (default) and perspective projection")
        self.perspective_projection_action.triggered.connect(self._toggle_projection_mode)
        view_menu.addAction(self.perspective_projection_action)
        
        # ==================== Window Menu ====================
        window_menu = menubar.addMenu("&Window")
        
        # Console
        console_action = QAction("🖥 Console Output", self)
        console_action.setShortcut("Ctrl+`")
        console_action.setStatusTip("Show/hide console output window")
        console_action.triggered.connect(self._toggle_console)
        window_menu.addAction(console_action)
        
        # ==================== Help Menu ====================
        help_menu = menubar.addMenu("&Help")
        
        # Help Guide
        help_guide_action = QAction("Help / Guide", self)
        help_guide_action.setShortcut("F1")
        help_guide_action.setStatusTip("Open simulation help and alloy/result guidance")
        help_guide_action.triggered.connect(self._show_help_guide_dialog)
        help_menu.addAction(help_guide_action)

        help_menu.addSeparator()

        # Error Log Viewer
        log_viewer_action = QAction("View Error Log", self)
        log_viewer_action.setShortcut("Ctrl+L")
        log_viewer_action.setStatusTip("View application error log")
        log_viewer_action.triggered.connect(self._show_log_viewer)
        help_menu.addAction(log_viewer_action)
        
        help_menu.addSeparator()
        
        # About
        about_action = QAction("About", self)
        about_action.setStatusTip("About this application")
        about_action.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_action)
        
        # Hardware Info
        hardware_info_action = QAction("Hardware Info", self)
        hardware_info_action.setStatusTip("Show detected hardware information")
        hardware_info_action.triggered.connect(self._show_hardware_info)
        help_menu.addAction(hardware_info_action)

    def _create_dialogs(self):
        """Create modal dialogs for Parts, Material, Mesh, and Analysis settings"""
        # Parts dialog
        self.parts_dialog = PartsDialog(self)
        self.parts_dialog.add_part_requested.connect(self._add_part)
        self.parts_dialog.remove_part_requested.connect(self._remove_selected_part)
        self.parts_dialog.clear_parts_requested.connect(self._clear_all_parts)
        self.parts_dialog.part_selected.connect(self._on_part_selected_from_dialog)
        self.parts_dialog.position_changed.connect(self._on_part_position_from_dialog)
        self.parts_dialog.rotation_changed.connect(self._on_part_rotation_from_dialog)
        self.parts_dialog.batch_part_types_confirmed.connect(self._on_batch_part_types_confirmed)
        self.parts_dialog.arrange_grid_requested.connect(self._arrange_parts_grid)
        self.parts_dialog.arrange_line_requested.connect(self._arrange_parts_line)
        self.parts_dialog.duplicate_requested.connect(self._duplicate_selected_part)
        self.parts_dialog.open_top_changed.connect(self._on_part_open_top_changed)
        
        # Material dialog
        self.material_dialog = MaterialDialog(self)
        self.material_dialog.alloy_changed.connect(self._on_alloy_confirmed)
        
        # Analysis options dialog
        self.analysis_dialog = AnalysisOptionsDialog(self)
        self.analysis_dialog.options_changed.connect(self._on_analysis_options_confirmed)
        
        # Mesh dialog
        self.mesh_dialog = MeshDialog(self)
        self.mesh_dialog.settings_changed.connect(self._on_mesh_settings_confirmed)
        self.mesh_dialog.apply_ml_requested.connect(self._apply_ml_mesh)
        
        # Performance dialog
        self.performance_dialog = PerformanceDialog(self)
        self.performance_dialog.options_changed.connect(self._on_performance_options_confirmed)

        # Solidification parameters dialog
        self.solidification_dialog = SolidificationParamsDialog(self)
        self.solidification_dialog.params_changed.connect(self._on_solidification_params_confirmed)

        # Console window
        self.console_window = ConsoleWindow(self)
        # Start capturing output immediately so we don't miss anything
        self.console_window.start_capture()
        
        # Log viewer dialog
        self.log_viewer_dialog = LogViewerDialog(self)
        
        # Help guide dialog
        self.help_guide_dialog = HelpGuideDialog(self)
        
        # ==================== NEW VISUALIZATION DIALOGS ====================
        
        # Visualization settings dialog (cross-section, colors, legend)
        self.visualization_dialog = VisualizationDialog(self)
        self.visualization_dialog.colormap_changed.connect(self._on_colormap_changed)
        self.visualization_dialog.color_scale_changed.connect(self._on_color_scale_changed)
        self.visualization_dialog.cross_section_changed.connect(self._on_cross_section_changed)
        self.visualization_dialog.cross_section_interaction_started.connect(
            self.visualizer.show_cross_section_plane_visual
        )
        self.visualization_dialog.cross_section_interaction_ended.connect(
            self.visualizer.hide_cross_section_plane_visual
        )
        self.visualization_dialog.show_legend_changed.connect(self._on_legend_toggle)
        
        # Measurement dialog
        self.measurement_dialog = MeasurementDialog(self)
        self.measurement_dialog.measure_distance_requested.connect(self._start_distance_measurement)
        self.measurement_dialog.clear_measurements_requested.connect(self._clear_measurements)
        self.measurement_dialog.calculate_volume_requested.connect(self._calculate_defect_volume)
        self.measurement_dialog.dialog_closed.connect(self._stop_measurement_mode)
        
        # Animation export dialog
        self.animation_dialog = AnimationExportDialog(self)
        self.animation_dialog.start_recording.connect(self._start_animation_recording)
        self.animation_dialog.stop_recording.connect(self._stop_animation_recording)
        self.animation_dialog.export_gif.connect(self._export_animation_gif)
        self.animation_dialog.export_mp4.connect(self._export_animation_mp4)
        
        # ==================== DESIGN TOOLS DIALOGS ====================
        
        # Auto Riser Design dialog
        self.riser_dialog = RiserDesignDialog(self)
        self.riser_dialog.add_risers_requested.connect(self._apply_riser_recommendations)
        
        # Store current values for dialogs
        self._current_voxel_size = 0.1
        self._current_refinement = 3
        self._current_memory_mode = "auto"
        self._current_alloy = "WCB"
    
    def _create_sidebar(self) -> QWidget:
        """Create compact left sidebar with buttons to open modal dialogs"""
        sidebar = QWidget()
        sidebar.setMaximumWidth(300)
        sidebar.setMinimumWidth(200)
        
        layout = QVBoxLayout()
        layout.setSpacing(8)
        
        # Title
        title = QLabel("Solidus Pro")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Settings Buttons Section (open modal dialogs)
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout()
        settings_layout.setSpacing(6)
        
        # Parts button
        self.open_parts_btn = QPushButton("📦 Parts Manager...")
        self.open_parts_btn.setToolTip("Manage CAD parts (Add, Remove, Position)")
        self.open_parts_btn.clicked.connect(self._open_parts_dialog)
        settings_layout.addWidget(self.open_parts_btn)
        
        # Parts summary label
        self.parts_summary_label = QLabel("No parts loaded")
        self.parts_summary_label.setStyleSheet("font-size: 10px; color: #888; margin-left: 10px;")
        settings_layout.addWidget(self.parts_summary_label)
        
        # Material button
        self.open_material_btn = QPushButton("🔧 Material Properties...")
        self.open_material_btn.setToolTip("Select alloy material")
        self.open_material_btn.clicked.connect(self._open_material_dialog)
        settings_layout.addWidget(self.open_material_btn)
        
        # Material summary label
        self.material_summary_label = QLabel("Alloy: WCB")
        self.material_summary_label.setStyleSheet("font-size: 10px; color: #888; margin-left: 10px;")
        settings_layout.addWidget(self.material_summary_label)
        
        # Analysis Options button
        self.open_analysis_btn = QPushButton("📊 Analysis Options...")
        self.open_analysis_btn.setToolTip("Select analyses: Shrinkage, Hot Spots, Niyama, Fraction Liquid")
        self.open_analysis_btn.clicked.connect(self._open_analysis_dialog)
        settings_layout.addWidget(self.open_analysis_btn)
        
        # Analysis summary label
        self.analysis_summary_label = QLabel("Analyses: Shrink, Hot Spots")
        self.analysis_summary_label.setStyleSheet("font-size: 10px; color: #888; margin-left: 10px;")
        settings_layout.addWidget(self.analysis_summary_label)
        
        # Mesh button
        self.open_mesh_btn = QPushButton("🔲 Mesh Parameters...")
        self.open_mesh_btn.setToolTip("Configure voxel mesh settings")
        self.open_mesh_btn.clicked.connect(self._open_mesh_dialog)
        settings_layout.addWidget(self.open_mesh_btn)
        
        # Mesh summary label
        self.mesh_summary_label = QLabel("Voxel: 0.100\", Ref: 3")
        self.mesh_summary_label.setStyleSheet("font-size: 10px; color: #888; margin-left: 10px;")
        settings_layout.addWidget(self.mesh_summary_label)
        
        # Performance button
        self.open_performance_btn = QPushButton("⚡ Performance Options...")
        self.open_performance_btn.setToolTip("Configure GPU acceleration and optimization features")
        self.open_performance_btn.clicked.connect(self._open_performance_dialog)
        settings_layout.addWidget(self.open_performance_btn)
        
        # Performance summary label
        self.performance_summary_label = QLabel("GPU: Detecting…")
        self.performance_summary_label.setStyleSheet("font-size: 10px; color: #888; margin-left: 10px;")
        settings_layout.addWidget(self.performance_summary_label)
        
        # Solidification Parameters button
        self.open_solidification_btn = QPushButton("🔧 Solidification Params...")
        self.open_solidification_btn.setToolTip("Configure feeding distances, chill reduction, exothermic sleeve timing, and more")
        self.open_solidification_btn.clicked.connect(self._open_solidification_params_dialog)
        settings_layout.addWidget(self.open_solidification_btn)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Simulation Control
        sim_group = QGroupBox("Simulation")
        sim_layout = QVBoxLayout()
        
        self.run_sim_btn = QPushButton("▶ Run Simulation")
        self.run_sim_btn.clicked.connect(self._run_simulation)
        self.run_sim_btn.setEnabled(False)
        self.run_sim_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        sim_layout.addWidget(self.run_sim_btn)

        self.stop_sim_btn = QPushButton("■ Stop Simulation")
        self.stop_sim_btn.clicked.connect(self._cancel_simulation)
        self.stop_sim_btn.setEnabled(False)
        sim_layout.addWidget(self.stop_sim_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% - %v/%m")
        sim_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        sim_layout.addWidget(self.status_label)
        
        sim_group.setLayout(sim_layout)
        layout.addWidget(sim_group)
        
        # Visualization Controls
        vis_group = QGroupBox("Visualization")
        vis_layout = QVBoxLayout()
        
        self.show_mesh_check = QCheckBox("Show Model")
        self.show_mesh_check.setChecked(True)
        self.show_mesh_check.stateChanged.connect(self._toggle_mesh_visibility)
        vis_layout.addWidget(self.show_mesh_check)
        
        self.show_results_check = QCheckBox("Show Results")
        self.show_results_check.setChecked(True)
        self.show_results_check.stateChanged.connect(self._toggle_shrinkage_visibility)
        vis_layout.addWidget(self.show_results_check)
        
        # Result type selector
        vis_layout.addWidget(QLabel("Result Type:"))
        self.result_type_combo = QComboBox()
        self.result_type_combo.addItems(["Shrinkage/Porosity", "Feed Modulus", "Hot Spots", "Niyama Criterion", "Fraction Liquid"])
        self.result_type_combo.currentTextChanged.connect(self._on_result_type_changed)
        self.result_type_combo.setEnabled(False)  # Enable after simulation
        vis_layout.addWidget(self.result_type_combo)
        
        # Threshold/intensity slider
        self.threshold_label_title = QLabel("X-Ray Level %:")
        vis_layout.addWidget(self.threshold_label_title)
        self.threshold_slider = MarkedSlider(Qt.Horizontal)
        self.threshold_slider.setRange(1, 100)
        self.threshold_slider.setValue(30)
        self.threshold_slider.setMarkerPosition(30)
        self.threshold_slider.valueChanged.connect(self._update_threshold)
        vis_layout.addWidget(self.threshold_slider)

        self.threshold_label = QLabel("30%")
        vis_layout.addWidget(self.threshold_label)
        
        # Time slider for Fraction Liquid (hidden by default)
        self.time_slider_widget = QWidget()
        time_slider_layout = QVBoxLayout()
        time_slider_layout.setContentsMargins(0, 0, 0, 0)
        
        # Header with liquid/solid indicators
        time_header_layout = QHBoxLayout()
        time_header_layout.addWidget(QLabel("🟡 Liquid"))
        time_header_layout.addStretch()
        time_header_layout.addWidget(QLabel("Solidification Time"))
        time_header_layout.addStretch()
        time_header_layout.addWidget(QLabel("Solid ⬜"))
        time_slider_layout.addLayout(time_header_layout)
        
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, FRACTION_LIQUID_TIME_SLIDER_MAX)
        self.time_slider.setValue(0)
        self.time_slider.setToolTip(
            "Drag to step through the simulated solidification timeline:\n"
            "• Left = Start of simulation (all liquid)\n"
            "• Right = End of simulated timeline\n"
            "Labels reflect actual liquid content at each time."
        )
        self.time_slider.valueChanged.connect(self._update_time_slider)
        time_slider_layout.addWidget(self.time_slider)

        playback_controls_layout = QHBoxLayout()
        playback_controls_layout.setSpacing(6)

        self.fraction_liquid_play_pause_btn = QPushButton("▶ Play")
        self.fraction_liquid_play_pause_btn.setToolTip("Play/Pause fraction liquid animation")
        self.fraction_liquid_play_pause_btn.clicked.connect(self._toggle_fraction_liquid_playback)
        self.fraction_liquid_play_pause_btn.setEnabled(False)
        playback_controls_layout.addWidget(self.fraction_liquid_play_pause_btn)

        playback_controls_layout.addWidget(QLabel("Speed:"))
        self.fraction_liquid_speed_combo = QComboBox()
        for speed in FRACTION_LIQUID_PLAYBACK_SPEEDS:
            self.fraction_liquid_speed_combo.addItem(f"{speed}x", speed)
        self.fraction_liquid_speed_combo.setCurrentIndex(
            self.fraction_liquid_speed_combo.findData(1.0)
        )
        self.fraction_liquid_speed_combo.setToolTip("Playback speed for fraction liquid animation")
        self.fraction_liquid_speed_combo.currentIndexChanged.connect(self._on_fraction_liquid_speed_changed)
        playback_controls_layout.addWidget(self.fraction_liquid_speed_combo)

        playback_controls_layout.addStretch()
        time_slider_layout.addLayout(playback_controls_layout)

        self.time_label = QLabel("0.0s (Liquid)")
        time_slider_layout.addWidget(self.time_label)
        self.time_slider_widget.setLayout(time_slider_layout)
        self.time_slider_widget.setVisible(False)  # Show only for fraction liquid
        vis_layout.addWidget(self.time_slider_widget)
        
        vis_group.setLayout(vis_layout)
        layout.addWidget(vis_group)
        
        # Camera Controls Section
        camera_group = QGroupBox("Camera Views")
        camera_layout = QVBoxLayout()
        
        # Create grid for camera buttons
        camera_grid = QGridLayout()
        camera_grid.setSpacing(4)
        
        top_btn = QPushButton("Top")
        top_btn.clicked.connect(lambda: self.visualizer.set_camera_view('top'))
        camera_grid.addWidget(top_btn, 0, 1)
        
        left_btn = QPushButton("Left")
        left_btn.clicked.connect(lambda: self.visualizer.set_camera_view('left'))
        camera_grid.addWidget(left_btn, 1, 0)
        
        right_btn = QPushButton("Right")
        right_btn.clicked.connect(lambda: self.visualizer.set_camera_view('right'))
        camera_grid.addWidget(right_btn, 1, 2)
        
        front_btn = QPushButton("Front")
        front_btn.clicked.connect(lambda: self.visualizer.set_camera_view('front'))
        camera_grid.addWidget(front_btn, 2, 1)
        
        bottom_btn = QPushButton("Bottom")
        bottom_btn.clicked.connect(lambda: self.visualizer.set_camera_view('bottom'))
        camera_grid.addWidget(bottom_btn, 3, 1)
        
        camera_layout.addLayout(camera_grid)
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        # Stretch at bottom
        layout.addStretch()
        
        sidebar.setLayout(layout)
        return sidebar
    
    # ==================== Dialog Open Methods ====================
    
    def _open_parts_dialog(self):
        """Open the parts manager dialog"""
        self._sync_parts_to_dialog()
        self.parts_dialog.center_on_parent()
        self.parts_dialog.show()
        self.parts_dialog.raise_()
    
    def _open_material_dialog(self):
        """Open the material properties dialog"""
        self.material_dialog.set_alloy(self._current_alloy)
        self.material_dialog.center_on_parent()
        self.material_dialog.show()
        self.material_dialog.raise_()
    
    def _open_analysis_dialog(self):
        """Open the analysis options dialog"""
        self.analysis_dialog.set_options(self._analysis_options)
        self.analysis_dialog.update_niyama_info(self._current_alloy)
        self.analysis_dialog.center_on_parent()
        self.analysis_dialog.show()
        self.analysis_dialog.raise_()

    def _open_solidification_params_dialog(self):
        """Open the solidification parameters dialog"""
        self.solidification_dialog.set_options(self._solidification_params)
        self.solidification_dialog.center_on_parent()
        self.solidification_dialog.show()
        self.solidification_dialog.raise_()

    def _on_solidification_params_confirmed(self, options):
        """Handle solidification params confirmation from dialog"""
        self._solidification_params = options

    @staticmethod
    def _load_solidification_params_with_user_override() -> dict:
        """Build SolidificationParameters defaults and apply user overrides from
        ``~/.castsim/config.yaml`` so any prior dialog session persists.
        """
        defaults = SolidificationParameters().to_dict()
        try:
            from pathlib import Path
            config_path = Path.home() / ".castsim" / "config.yaml"
            if not config_path.exists():
                return defaults
            try:
                import yaml
            except ImportError:
                return defaults
            try:
                data = yaml.safe_load(config_path.read_text()) or {}
            except Exception:
                return defaults
            if not isinstance(data, dict):
                return defaults
            solid = data.get("solidification")
            if not isinstance(solid, dict):
                return defaults
            for grp_key, grp_values in solid.items():
                if not isinstance(grp_values, dict):
                    continue
                bucket = defaults.get(grp_key)
                if not isinstance(bucket, dict):
                    continue
                for field_key, value in grp_values.items():
                    if field_key in bucket:
                        bucket[field_key] = value
                defaults[grp_key] = bucket
            return defaults
        except Exception as exc:
            print(f"[MainWindow] failed to read user overrides: {exc}")
            return defaults

    def _open_mesh_dialog(self):
        """Open the mesh parameters dialog"""
        self.mesh_dialog.set_values(
            self._current_voxel_size,
            self._current_refinement,
            self._current_memory_mode
        )
        # Enable ML button if STL is loaded
        self.mesh_dialog.apply_ml_button.setEnabled(self.stl_loader.is_valid())
        self.mesh_dialog.center_on_parent()
        self.mesh_dialog.show()
        self.mesh_dialog.raise_()
    
    def _open_performance_dialog(self):
        """Open the performance options dialog"""
        self._update_performance_summary()
        self.performance_dialog.set_options(self._performance_options)
        self.performance_dialog.center_on_parent()
        self.performance_dialog.show()
        self.performance_dialog.raise_()
    
    # GPU distance chunk values exposed by the popover slider
    GPU_CHUNK_VALUES = (500, 1000, 2000, 4000, 8000, 10000)
    
    def _get_gpu_chunk_reference_count(self) -> int:
        """Return reference (surface voxel) count from the last sim, or a default."""
        try:
            simulation = self.simulation_worker.simulation
            mesh = simulation.mesh
            surface = mesh.surface_voxels
            if surface is not None and len(surface) > 0:
                count = int(np.sum(surface))
                if count > 0:
                    return count
        except Exception:
            pass
        return 15000
    
    def _gpu_chunk_peak_gb(self, chunk_size: int, refs: int) -> float:
        """Estimate peak GPU/unified memory per chunk (diff + sq + sum + sqrt)."""
        return chunk_size * refs * 32 / (1024 ** 3)
    
    def _open_gpu_chunk_popover(self):
        """Show a popover to tune the GPU distance computation chunk size."""
        from config.settings import get_settings as _get_settings
        settings = _get_settings()
        current = max(1, int(getattr(settings.performance, 'gpu_distance_chunk_size', 10000)))
        
        # Snap current value to the closest slider stop.
        current_idx = 0
        for i, val in enumerate(self.GPU_CHUNK_VALUES):
            if abs(val - current) <= abs(self.GPU_CHUNK_VALUES[current_idx] - current):
                current_idx = i
        
        refs = self._get_gpu_chunk_reference_count()
        default_refs = refs >= 15000 and refs <= 16000
        initial_idx = current_idx
        changed = {"value": False}
        
        menu = QMenu(self)
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 10, 12, 10)
        
        title = QLabel("GPU Distance Chunk Size")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)
        
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, len(self.GPU_CHUNK_VALUES) - 1)
        slider.setValue(current_idx)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(1)
        layout.addWidget(slider)
        
        value_label = QLabel("")
        value_label.setStyleSheet("font-size: 11px; color: #eee;")
        layout.addWidget(value_label)
        
        gb_label = QLabel("")
        gb_label.setStyleSheet("font-size: 11px; color: #8cf;")
        layout.addWidget(gb_label)
        
        note_label = QLabel("")
        note_label.setStyleSheet("font-size: 10px; color: #888;")
        note_label.setWordWrap(True)
        layout.addWidget(note_label)
        
        reset_btn = QPushButton("Reset to Default (10,000)")
        reset_btn.setStyleSheet("font-size: 10px; padding: 3px 8px;")
        layout.addWidget(reset_btn)
        
        def update_labels(idx):
            value = self.GPU_CHUNK_VALUES[idx]
            peak_gb = self._gpu_chunk_peak_gb(value, refs)
            value_label.setText(
                f"Chunk size: {value:,} points per chunk  "
                f"(smaller = less RAM, slightly slower)"
            )
            if default_refs:
                ref_text = f"at ~{refs:,} reference points (typical last-run value)"
            else:
                ref_text = f"at {refs:,} reference points from your last simulation"
            gb_label.setText(f"Peak GPU memory per chunk: ~{peak_gb:.2f} GB ({ref_text})")
            note_label.setText(
                "Smaller chunk = less RAM but more kernel launches. "
                "Results are identical at any size. Applies to the next simulation."
            )
            if idx != initial_idx:
                changed["value"] = True
            apply_gpu_chunk_size(value, idx)
        
        def apply_gpu_chunk_size(value, idx):
            settings.performance.gpu_distance_chunk_size = value
            if changed["value"]:
                try:
                    settings.save_user_override("performance", "gpu_distance_chunk_size", value)
                except Exception as e:
                    logger.error(f"Could not persist GPU chunk size: {e}")
            self._update_performance_summary()
        
        def on_reset():
            reset_btn.setEnabled(False)
            default_idx = self.GPU_CHUNK_VALUES.index(10000)
            slider.setValue(default_idx)
            update_labels(default_idx)
            QTimer.singleShot(800, lambda: reset_btn.setEnabled(True))
        
        slider.valueChanged.connect(update_labels)
        reset_btn.clicked.connect(on_reset)
        
        update_labels(current_idx)
        
        action = QWidgetAction(menu)
        action.setDefaultWidget(widget)
        menu.addAction(action)
        menu.setStyleSheet(
            "QMenu { padding: 4px; }"
        )
        menu.exec_(QCursor.pos())

    def _toggle_console(self):
        """Toggle the console output window visibility"""
        if self.console_window.isVisible():
            self.console_window.hide()
        else:
            # Position near the main window if first time
            if not self.console_window.pos().x():
                main_geo = self.geometry()
                screen_geo = QApplication.desktop().availableGeometry(self)
                x, y = clamp_window_position(
                    main_geo.x() + main_geo.width() + 10,
                    main_geo.y(),
                    self.console_window.width(),
                    self.console_window.height(),
                    screen_geo.x(),
                    screen_geo.y(),
                    screen_geo.width(),
                    screen_geo.height(),
                )
                self.console_window.move(x, y)
            self.console_window.show()
            self.console_window.raise_()
            self.console_window.activateWindow()
    
    def _sync_parts_to_dialog(self):
        """Sync current parts to the parts dialog using the new table format"""
        # Build parts data list: (part_id, name, part_type_value, is_open_top)
        parts_data = []
        for part_id, part in self.part_manager.parts.items():
            parts_data.append((part_id, part.name, part.part_type.value, part.is_open_top))
        
        # Populate the table (this handles all UI updates internally)
        self.parts_dialog.populate_parts(parts_data)
    
    # ==================== Dialog Callback Methods ====================
    
    def _on_part_selected_from_dialog(self, part_id):
        """Handle part selection from dialog (now receives part_id directly)"""
        # Unhighlight previous selection
        if self._selected_part_id and self._selected_part_id != part_id:
            self.visualizer.highlight_part(self._selected_part_id, False)
        
        self._selected_part_id = part_id
        part = self.part_manager.get_part(part_id)
        
        if part:
            # Update dialog position spinboxes
            self.parts_dialog.update_position_spinboxes(
                part.position[0], part.position[1], part.position[2]
            )
            self.parts_dialog.update_rotation_spinboxes(
                part.rotation[0], part.rotation[1], part.rotation[2]
            )
            
            # Highlight in visualization
            self.visualizer.highlight_part(part_id, True)
            
            # Enable controls (buttons are now enabled in the dialog itself)
            self.parts_dialog.remove_part_btn.setEnabled(True)
            self.parts_dialog.duplicate_btn.setEnabled(True)
    
    def _on_part_position_from_dialog(self, x, y, z):
        """Handle position change from dialog"""
        if self._selected_part_id:
            part = self.part_manager.get_part(self._selected_part_id)
            if part:
                part.set_position(x, y, z)
                self._update_visualization()

    def _on_part_rotation_from_dialog(self, rx, ry, rz):
        """Handle rotation change from dialog."""
        if self._selected_part_id:
            part = self.part_manager.get_part(self._selected_part_id)
            if part:
                part.set_rotation(rx, ry, rz)
                self._update_visualization()
    
    def _on_batch_part_types_confirmed(self, changes):
        """Handle batch part type changes from dialog - applies all changes at once"""
        if not changes:
            return
        
        type_map = {
            "Cast": PartType.CAST, 
            "Riser": PartType.RISER, 
            "Riser Post": PartType.RISER_POST, 
            "Chill": PartType.CHILL,
            "Core": PartType.CORE,
        }
        
        # Apply all part type changes
        for part_id, type_text in changes.items():
            part = self.part_manager.get_part(part_id)
            if part:
                new_type = type_map.get(type_text, PartType.CAST)
                part.set_part_type(new_type)
                print(f"Changed '{part.name}' to {new_type.value.upper()}")
        
        # Single visualization update after all changes
        self._update_visualization()
        print(f"Applied {len(changes)} part type change(s).")
    
    def _on_part_open_top_changed(self, part_id: str, is_open_top: bool):
        """Handle Open Top checkbox change from Parts dialog"""
        part = self.part_manager.get_part(part_id)
        if part:
            part.is_open_top = is_open_top
            status = "OPEN" if is_open_top else "BLIND"
            print(f"Riser '{part.name}' set to {status} top")
    
    def _on_alloy_confirmed(self, alloy_name):
        """Handle alloy confirmation from dialog"""
        self._current_alloy = alloy_name
        self.material_summary_label.setText(f"Alloy: {alloy_name}")
        # Update Niyama info in analysis dialog
        self.analysis_dialog.update_niyama_info(alloy_name)
        
        # Update dynamic threshold if currently viewing Niyama
        if hasattr(self, '_current_vis_mode') and self._current_vis_mode == "niyama":
            rec_threshold = self._get_recommended_threshold("niyama", self._current_alloy)
            self.threshold_slider.setValue(rec_threshold)
            self.threshold_slider.setMarkerPosition(rec_threshold)
            alloy_key = self._current_alloy.split()[0] if self._current_alloy else 'WCB'
            self.threshold_label_title.setText(f"Min Defect Risk % ({alloy_key}):")
    
    def _on_analysis_options_confirmed(self, options):
        """Handle analysis options confirmation from dialog"""
        self._analysis_options = options
        self._update_analysis_summary()
    
    def _update_analysis_summary(self):
        """Update analysis summary label in sidebar"""
        analyses = []
        if self._analysis_options.get('shrinkage', False):
            analyses.append("Shrink")
        if self._analysis_options.get('hotspot', False):
            analyses.append("Hot Spots")
        if self._analysis_options.get('niyama', False):
            analyses.append("Niyama")
        if self._analysis_options.get('fraction_liquid', False):
            analyses.append("Frac. Liq.")
        
        if analyses:
            self.analysis_summary_label.setText(f"Analyses: {', '.join(analyses)}")
        else:
            self.analysis_summary_label.setText("No analyses selected")
    
    def _current_simulation_has_risers(self) -> bool:
        """Return True when the current simulation contains RISER voxels."""
        if self.simulation_worker is None or self.simulation_worker.simulation is None:
            return False

        simulation = self.simulation_worker.simulation
        mesh = getattr(simulation, "mesh", None)
        part_types = getattr(mesh, "voxel_part_types", None) if mesh is not None else None
        if part_types is None:
            return False

        return bool(np.any(np.asarray(part_types, dtype=np.int32) == 1))

    def _get_recommended_threshold(self, vis_mode, alloy_name):
        """Get recommended threshold value (0-100) based on visualization mode and alloy"""
        if vis_mode == "shrinkage":
            return 30
        elif vis_mode == "hotspot":
            return 30  # Top 30% hottest areas
        elif vis_mode == "niyama":
            # Alloy-specific Niyama thresholds
            alloy_default_thresholds = {
                'WCB': 30,    # Carbon steel
                'CF8': 20,    # Stainless
                'C958': 40,   # Bronze
            }
            # Extract alloy key (e.g., "WCB" from "WCB - Carbon Steel")
            alloy_key = alloy_name.split()[0] if alloy_name else 'WCB'
            return alloy_default_thresholds.get(alloy_key, 30)
        elif vis_mode == "sand_burn_in":
            return 20
        elif vis_mode == "gas_porosity":
            return 10
        # Default for others
        return 20

    def _get_effective_analysis_options(self) -> dict:
        """Return desktop-normalized analysis options."""
        options = dict(self._analysis_options)
        options.pop("solver_mode", None)
        return options

    def _get_fraction_liquid_slider_fraction(self, slider_value: int) -> float:
        slider_clamped = max(0, min(int(slider_value), FRACTION_LIQUID_TIME_SLIDER_MAX))
        return slider_clamped / float(FRACTION_LIQUID_TIME_SLIDER_MAX)

    @staticmethod
    def _map_fraction_liquid_slider_time(slider_value: int, min_time: float, max_time: float) -> float:
        """Map the fraction-liquid slider to time, preserving exact endpoints."""
        slider_clamped = max(0, min(int(slider_value), FRACTION_LIQUID_TIME_SLIDER_MAX))
        if slider_clamped <= 0:
            return float(min_time)
        if slider_clamped >= FRACTION_LIQUID_TIME_SLIDER_MAX:
            return float(max_time)

        slider_fraction = slider_clamped / float(FRACTION_LIQUID_TIME_SLIDER_MAX)
        return float(min_time + slider_fraction * (max_time - min_time))

    def _coerce_fraction_liquid_slider_value(self, raw_value: int) -> int:
        """Support loading older projects that persisted a 0-100 slider range."""
        value = int(raw_value)
        if value <= 100:
            scaled = int(round(max(0, min(value, 100)) * (FRACTION_LIQUID_TIME_SLIDER_MAX / 100.0)))
            return scaled
        return max(0, min(value, FRACTION_LIQUID_TIME_SLIDER_MAX))

    def _toggle_fraction_liquid_playback(self) -> None:
        """Toggle play/pause state of fraction liquid playback."""
        if self._is_playing_fraction_liquid:
            self._stop_fraction_liquid_playback()
        else:
            self._start_fraction_liquid_playback()

    def _start_fraction_liquid_playback(self) -> None:
        """Start fraction liquid playback."""
        if self.simulation_worker is None or self.simulation_worker.simulation is None:
            return
        sim = self.simulation_worker.simulation
        if not sim.fraction_liquid_times:
            return

        self._is_playing_fraction_liquid = True
        self.fraction_liquid_play_pause_btn.setText("⏸ Pause")

        effective_interval = int(FRACTION_LIQUID_PLAYBACK_INTERVAL_MS / self._fraction_liquid_playback_speed)
        self._fraction_liquid_playback_timer.setInterval(effective_interval)
        self._fraction_liquid_playback_timer.start()

    def _stop_fraction_liquid_playback(self) -> None:
        """Stop fraction liquid playback."""
        self._is_playing_fraction_liquid = False
        self.fraction_liquid_play_pause_btn.setText("▶ Play")
        self._fraction_liquid_playback_timer.stop()

    def _on_fraction_liquid_playback_tick(self) -> None:
        """Advance the fraction liquid slider by one playback step."""
        current = self.time_slider.value()
        step = FRACTION_LIQUID_PLAYBACK_BASE_STEP
        new_value = self._advance_fraction_liquid_slider_value(current, step, FRACTION_LIQUID_TIME_SLIDER_MAX)
        self.time_slider.setValue(new_value)

    @staticmethod
    def _advance_fraction_liquid_slider_value(current: int, step: int, max_value: int) -> int:
        """Advance slider value with looping.

        The final frame (at max_value) is displayed before the slider wraps to 0.
        This means advance is only clamped to 0 when next_value > max_value.
        """
        next_val = current + step
        if next_val > max_value:
            return 0
        return next_val

    def _on_fraction_liquid_speed_changed(self, index: int) -> None:
        """Handle speed combo change."""
        speed_data = self.fraction_liquid_speed_combo.currentData()
        if speed_data is not None:
            self._fraction_liquid_playback_speed = float(speed_data)
        else:
            self._fraction_liquid_playback_speed = 1.0

        if self._is_playing_fraction_liquid:
            self._stop_fraction_liquid_playback()
            self._start_fraction_liquid_playback()

    def _sync_fraction_liquid_playback_controls(self) -> None:
        """Enable/disable playback controls based on mode and data readiness."""
        is_fl = self._current_vis_mode == "fraction_liquid"
        has_data = (
            self.simulation_worker is not None
            and self.simulation_worker.simulation is not None
            and bool(self.simulation_worker.simulation.fraction_liquid_times)
        )
        enabled = bool(is_fl and has_data)
        self.fraction_liquid_play_pause_btn.setEnabled(enabled)
        self.fraction_liquid_speed_combo.setEnabled(enabled)

    def _on_result_type_changed(self, result_type):
        """Handle result type selection change"""
        print(f"DEBUG: Result type changed to: {result_type}")
        self._current_vis_mode = {
            "Shrinkage/Porosity": "shrinkage",
            "Feed Modulus": "feed_modulus",
            "Hot Spots": "hotspot",
            "Niyama Criterion": "niyama",
            "Hot Tear Risk": "hot_tear",
            "Sand Burn-In": "sand_burn_in",
            "Gas Generation": "gas_generation",
            "Gas Entrapment": "gas_entrapment",
            "Gas Porosity Risk": "gas_porosity",
            "Fraction Liquid": "fraction_liquid",
            # Advanced analysis modes
            "Feeding Quality": "feeding_quality",
            "Thermal Stress": "thermal_stress",
            "Distortion": "distortion",
            "Crack Susceptibility": "crack_susceptibility",
            "SDAS (Microstructure)": "sdas",
            "Predicted UTS": "predicted_uts"
        }.get(result_type, "shrinkage")
        
        print(f"DEBUG: Visualization mode set to: {self._current_vis_mode}")
        
        # Show/hide time slider for fraction liquid
        is_fraction_liquid = (self._current_vis_mode == "fraction_liquid")
        is_hot_tear = (self._current_vis_mode == "hot_tear")
        print(f"DEBUG: Setting time_slider_widget visible: {is_fraction_liquid}")
        self.time_slider_widget.setVisible(is_fraction_liquid)
        
        # Only hide threshold slider for fraction liquid (uses time slider instead)
        # Hot tear now uses threshold to filter within identified zones
        hide_threshold = is_fraction_liquid
        self.threshold_slider.setVisible(not hide_threshold)
        self.threshold_label.setVisible(not hide_threshold)
        self.threshold_label_title.setVisible(not hide_threshold)
        
        if is_fraction_liquid:
            self.threshold_slider.setValue(1)  # Show all voxels with >1% liquid
            # Check if simulation has fraction liquid data
            if self.simulation_worker and self.simulation_worker.simulation:
                sim = self.simulation_worker.simulation
                if sim.fraction_liquid_times:
                    print(f"DEBUG: Fraction liquid snapshots: {len(sim.fraction_liquid_times)}")
                    print(f"DEBUG: Time range: {sim.fraction_liquid_times[0]:.1f}s to {sim.fraction_liquid_times[-1]:.1f}s")
                    # Trigger initial visualization at slider position
                    self._update_time_slider(self.time_slider.value())
                else:
                    print("DEBUG: NO fraction liquid snapshots found!")
            else:
                print("DEBUG: No simulation worker or simulation object!")
        elif is_hot_tear:
            # Hot tear: start with low threshold (10%) to show only highest-risk zones
            # Users can increase to see more zones or decrease to focus on worst areas
            self.threshold_slider.setValue(10)
            self.threshold_slider.setMarkerPosition(10)
            self.threshold_label_title.setText("Min Risk %:")
        elif self._current_vis_mode == "niyama":
            # Niyama: set alloy-specific default threshold
            rec_threshold = self._get_recommended_threshold("niyama", self._current_alloy)
            self.threshold_slider.setValue(rec_threshold)
            self.threshold_slider.setMarkerPosition(rec_threshold)
            
            alloy_key = self._current_alloy.split()[0] if self._current_alloy else 'WCB'
            self.threshold_label_title.setText(f"Min Defect Risk % ({alloy_key}):")
            self.visualizer.set_colormap('niyama')
            
        elif self._current_vis_mode == "sand_burn_in":
            # Sand burn-in
            rec_threshold = self._get_recommended_threshold("sand_burn_in", self._current_alloy)
            self.threshold_slider.setValue(rec_threshold)
            self.threshold_slider.setMarkerPosition(rec_threshold)
            self.threshold_label_title.setText("Min Burn-In Risk %:")
        
        elif self._current_vis_mode in ("gas_generation", "gas_entrapment", "gas_porosity"):
            rec_threshold = self._get_recommended_threshold("gas_porosity", self._current_alloy)
            self.threshold_slider.setValue(rec_threshold)
            self.threshold_slider.setMarkerPosition(rec_threshold)
            mode_label = {
                "gas_generation": "Min Gas Generation %:",
                "gas_entrapment": "Min Entrapment %:",
                "gas_porosity": "Min Gas Risk %:",
            }.get(self._current_vis_mode, "Min Risk %:")
            self.threshold_label_title.setText(mode_label)
            self.visualizer.set_colormap('defect_industry')
            
        elif self._current_vis_mode == "feed_modulus":
            self.threshold_slider.setValue(1)
            self.threshold_slider.setMarkerPosition(1)
            self.threshold_label_title.setText("Feed Modulus Range %:")
            self.visualizer.set_colormap('feedmod')
            self.visualizer.enable_feed_modulus_leader(True)
        elif self._current_vis_mode in ("feeding_quality", "thermal_stress", "distortion", 
                                         "crack_susceptibility", "sdas", "predicted_uts"):
            # Advanced analysis modes
            # ... existing label logic ...
            labels = {
                "feeding_quality": "Min Risk %:",
                "thermal_stress": "Min Stress %:",
                "distortion": "Min Distortion %:",
                "crack_susceptibility": "Min Crack Risk %:",
                "sdas": "Min SDAS %:",
                "predicted_uts": "Min Low-UTS %:"
            }
            self.threshold_label_title.setText(labels.get(self._current_vis_mode, "Threshold:"))
            self.threshold_slider.setValue(20)
            self.threshold_slider.setMarkerPosition(20)
            
        else:
            # Default (Shrinkage, Hotspot, etc.)
            rec_threshold = self._get_recommended_threshold(self._current_vis_mode, self._current_alloy)
            self.threshold_slider.setValue(rec_threshold)
            self.threshold_slider.setMarkerPosition(rec_threshold)
            
            # Update threshold label based on result type
            labels = {
                "shrinkage": "X-Ray Level %:",
                "hotspot": "Min Intensity %:",
            }
            self.threshold_label_title.setText(labels.get(self._current_vis_mode, "Threshold:"))
            # Reset colormap to default for non-Niyama modes
            self.visualizer.set_colormap('red_gradient')
            self.visualizer.enable_feed_modulus_leader(False)


        
        # Update visualization with current results
        if self.current_results is not None:
            self._update_results_visualization()

        self._stop_fraction_liquid_playback()
        self._sync_fraction_liquid_playback_controls()
    
    @staticmethod
    def _get_fraction_liquid_state_label(fraction_liquid) -> str:
        """Return a descriptive label based on actual fraction-liquid voxel data."""
        if fraction_liquid is None or fraction_liquid.size == 0:
            return "Fully Solid"

        mean_liquid = float(np.mean(fraction_liquid))
        max_liquid = float(np.max(fraction_liquid))
        voxels_with_liquid = float(np.count_nonzero(fraction_liquid > 0.01))
        liquid_fraction_of_voxels = voxels_with_liquid / fraction_liquid.size if fraction_liquid.size > 0 else 0.0

        if max_liquid <= 0.01:
            return "Fully Solid"

        if mean_liquid < 0.005 and liquid_fraction_of_voxels < 0.05:
            return "Trace Liquid"

        if liquid_fraction_of_voxels < 0.15 and max_liquid > 0.5:
            return "Risers/Hot Spots Liquid"

        if mean_liquid < 0.25:
            return "Mostly Solid"

        if mean_liquid < 0.75:
            return "Solidifying"

        return "Mostly Liquid"

    @staticmethod
    def _get_fraction_liquid_stats(fraction_liquid) -> tuple[float, float, int]:
        """Return mean, max, and visible-liquid voxel count for debug/status output."""
        if fraction_liquid is None or fraction_liquid.size == 0:
            return 0.0, 0.0, 0

        fraction_liquid_arr = np.asarray(fraction_liquid, dtype=np.float32)
        return (
            float(np.mean(fraction_liquid_arr)),
            float(np.max(fraction_liquid_arr)),
            int(np.count_nonzero(fraction_liquid_arr > 0.01)),
        )

    @staticmethod
    def _get_effective_shrinkage_threshold(
        intensities: np.ndarray,
        requested_threshold: float,
        minimum_display_threshold: float = 0.01,
    ) -> tuple[float, bool]:
        """
        Return a display threshold that avoids blanking real low-level risks.

        The boolean indicates whether the threshold was lowered from the user's
        requested value.
        """
        if intensities.size == 0:
            return requested_threshold, False

        finite_intensities = np.asarray(intensities, dtype=np.float32)
        finite_intensities = finite_intensities[np.isfinite(finite_intensities)]
        if finite_intensities.size == 0:
            return requested_threshold, False

        if np.any(finite_intensities >= requested_threshold):
            return requested_threshold, False

        max_intensity = float(np.max(finite_intensities))
        if max_intensity < minimum_display_threshold:
            return requested_threshold, False

        return max(minimum_display_threshold, min(max_intensity, requested_threshold)), True

    def _update_time_slider(self, value):
        """Handle time slider change for fraction liquid visualization"""
        if self.simulation_worker is None or self.simulation_worker.simulation is None:
            return

        sim = self.simulation_worker.simulation
        if sim.fraction_liquid_times is None or len(sim.fraction_liquid_times) == 0:
            print(f"DEBUG: No fraction liquid snapshots available!")
            return

        # Map slider value to actual time range
        slider_fraction = self._get_fraction_liquid_slider_fraction(value)
        min_time, max_time = sim.get_fraction_liquid_time_range()
        time_value = self._map_fraction_liquid_slider_time(value, min_time, max_time)

        # Get actual fraction liquid data for labeling
        fraction_liquid = sim.get_fraction_liquid_at_time(time_value)
        mean_liquid, max_liquid, visible_liquid = self._get_fraction_liquid_stats(fraction_liquid)

        # Throttled debug output (playback advances 10 slider steps per tick,
        # so print roughly every 10th tick rather than every tick)
        if value % (FRACTION_LIQUID_PLAYBACK_BASE_STEP * FRACTION_LIQUID_DEBUG_PRINT_INTERVAL) == 0:
            print(
                f"DEBUG: SliderRaw={value}, Slider={slider_fraction * 100:.1f}%, Time={time_value:.1f}s, "
                f"Snapshots={len(sim.fraction_liquid_times)}, TimeRange=[{min_time:.1f}, {max_time:.1f}], "
                f"LiquidMean={mean_liquid:.4f}, LiquidMax={max_liquid:.4f}, LiquidVoxels>1%={visible_liquid}"
            )

        # Show descriptive label based on actual liquid field
        state_desc = self._get_fraction_liquid_state_label(fraction_liquid)

        self.time_label.setText(f"{time_value:.1f}s - {state_desc}")

        # Update visualization (pass pre-computed array to avoid querying twice)
        self._update_fraction_liquid_visualization(
            time_value, fraction_liquid=fraction_liquid, slider_value=value
        )

        # Capture frame if animation recording is active
        if hasattr(self.visualizer, '_is_recording') and self.visualizer._is_recording:
            self._capture_animation_frame()
    
    def _update_fraction_liquid_visualization(self, time_seconds, fraction_liquid=None, slider_value=None):
        """Update fraction liquid visualization at given time"""
        if self.simulation_worker is None or self.simulation_worker.simulation is None:
            return

        # Use the voxel mesh from simulation worker (not self._voxel_mesh which doesn't exist)
        voxel_mesh = self.simulation_worker.voxel_mesh
        if voxel_mesh is None or voxel_mesh.voxel_centers is None:
            return

        sim = self.simulation_worker.simulation
        if fraction_liquid is None:
            fraction_liquid = sim.get_fraction_liquid_at_time(time_seconds)

        if fraction_liquid is not None:
            self._fraction_liquid_render_count += 1
            if self._fraction_liquid_render_count % FRACTION_LIQUID_DEBUG_PRINT_INTERVAL == 0:
                mean_liquid, max_liquid, visible_liquid = self._get_fraction_liquid_stats(fraction_liquid)
                _, max_time = sim.get_fraction_liquid_time_range()
                print(
                    f"DEBUG: Fraction liquid render Time={time_seconds:.1f}s, MaxTime={max_time:.1f}s, "
                    f"LiquidMean={mean_liquid:.4f}, LiquidMax={max_liquid:.4f}, LiquidVoxels>1%={visible_liquid}"
                )
            self.visualizer.load_fraction_liquid(
                voxel_mesh.voxel_centers,
                voxel_mesh.voxel_sizes,
                fraction_liquid,
                cache_key=slider_value,
            )
            # Update legend without a second full render (load_fraction_liquid
            # already rendered this tick).
            self.visualizer.update_legend("Fraction Liquid", 0.0, 1.0, render=False)
    
    def _on_mesh_settings_confirmed(self, voxel_size, refinement, memory_mode):
        """Handle mesh settings confirmation from dialog"""
        self._current_voxel_size = voxel_size
        self._current_refinement = refinement
        self._current_memory_mode = memory_mode
        self._update_mesh_summary()
        self.mesh_setting_origin = "manual"
    
    def _on_performance_options_confirmed(self, options: dict):
        """Handle performance options confirmation from dialog"""
        self._performance_options.update(options)
        if 'use_gpu' in options:
            from config.settings import settings as _settings
            _settings.use_gpu = bool(options['use_gpu'])
        self._update_performance_summary()
    
    def _update_performance_summary(self):
        """Update performance summary label in sidebar"""
        gpu_mgr = GPUComputeManager()
        gpu_available = gpu_mgr.is_gpu_available
        use_gpu = bool(self._performance_options.get('use_gpu', True))

        if gpu_available and use_gpu:
            backend = gpu_mgr.backend.name
            device = gpu_mgr.device_name
            short = "Apple Silicon GPU" if "Apple Silicon" in device else device
            gpu_text = f"GPU: {backend} ({short})"
            color = "#4a4"
        else:
            gpu_text = "GPU: Off"
            color = "#888"

        parts = [gpu_text]
        if self._performance_options.get('chunked_voxelization', False):
            parts.append("Chunk")
        if self._performance_options.get('incremental_solidification', False):
            parts.append("Incr")
        try:
            from config.settings import get_settings as _get_settings
            chunk = int(_get_settings().performance.gpu_distance_chunk_size)
            if chunk > 0 and chunk != 10000:
                parts.append(f"DistChunk {chunk:,}")
        except Exception:
            pass
        self.performance_summary_label.setText(", ".join(parts))
        self.performance_summary_label.setStyleSheet(
            f"font-size: 10px; color: {color}; margin-left: 10px;"
        )
    
    def _update_mesh_summary(self):
        """Update mesh summary label in sidebar"""
        self.mesh_summary_label.setText(
            f"Voxel: {self._current_voxel_size:.3f}\", Ref: {int(self._current_refinement)}"
        )
    
    def _update_parts_summary(self):
        """Update the parts summary label in sidebar"""
        count = len(self.part_manager.parts)
        if count == 0:
            self.parts_summary_label.setText("No parts loaded")
        else:
            types = {"cast": 0, "riser": 0, "riser_post": 0, "chill": 0}
            for part in self.part_manager.parts.values():
                types[part.part_type.value] += 1
            summary = f"{count} part(s): "
            parts = []
            if types["cast"] > 0:
                parts.append(f"{types['cast']}C")
            if types["riser"] > 0:
                parts.append(f"{types['riser']}R")
            if types["riser_post"] > 0:
                parts.append(f"{types['riser_post']}RP")
            if types["chill"] > 0:
                parts.append(f"{types['chill']}Ch")
            self.parts_summary_label.setText(summary + ", ".join(parts))
    
    def _create_visualization_panel(self) -> QWidget:
        """Create 3D visualization panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.visualizer = CastingVisualizer()
        layout.addWidget(self.visualizer)
        
        panel.setLayout(layout)
        return panel

    def _apply_ml_mesh(self):
        """Invoke ML-based mesh recommendation."""
        if not self.stl_loader.is_valid():
            QMessageBox.warning(self, "Machine Learning Mesh", "Please load a CAD file first.")
            return

        if self.mesh_suggestion_worker is not None and self.mesh_suggestion_worker.isRunning():
            return

        self.mesh_dialog.apply_ml_button.setEnabled(False)
        self.mesh_dialog.ml_status_label.setText("Computing recommendation...")

        self.mesh_suggestion_worker = MeshSuggestionWorker(self.stl_loader)
        self.mesh_suggestion_worker.result_ready.connect(self._on_ml_recommendation)
        self.mesh_suggestion_worker.error.connect(self._on_ml_error)
        self.mesh_suggestion_worker.finished.connect(self._on_mesh_worker_finished)
        self.mesh_suggestion_worker.start()

    def _on_ml_recommendation(self, recommendation: MeshRecommendation):
        """Handle ML mesh recommendation result."""
        self._applying_ml_suggestion = True
        try:
            self.mesh_dialog.voxel_size_spin.setValue(recommendation.base_voxel_size)
            self.mesh_dialog.refinement_spin.setValue(recommendation.refinement_level)
            # Also update stored values
            self._current_voxel_size = recommendation.base_voxel_size
            self._current_refinement = recommendation.refinement_level
            self._update_mesh_summary()
        finally:
            self._applying_ml_suggestion = False

        self.mesh_setting_origin = "ml"
        self.last_ml_recommendation = recommendation

        confidence_pct = f"{recommendation.confidence * 100:.0f}%"
        status_lines = [f"{recommendation.reason} (confidence {confidence_pct})"]
        if recommendation.warning:
            status_lines.append(recommendation.warning)
        self.mesh_dialog.ml_status_label.setText("\n".join(status_lines))
        self.mesh_dialog.apply_ml_button.setEnabled(True)

    def _on_ml_error(self, message: str):
        """Handle ML recommendation errors."""
        self.mesh_dialog.ml_status_label.setText(f"Recommendation failed: {message}")
        self.mesh_dialog.apply_ml_button.setEnabled(True)

    def _on_mesh_worker_finished(self):
        """Clean up mesh suggestion worker after thread has fully finished."""
        if self.mesh_suggestion_worker is not None:
            self.mesh_suggestion_worker = None

    def _show_hardware_info(self):
        """Display hardware info in status"""
        gpu_manager = GPUComputeManager()
        use_gpu = bool(self._performance_options.get('use_gpu', True))
        if gpu_manager.is_gpu_available and use_gpu:
            gpu_status = f"{gpu_manager.backend.name} ({gpu_manager.device_name})"
        else:
            gpu_status = "Off"
        info = (f"Hardware: {hardware.cpu_count} cores "
                f"({hardware.performance_cores}P+{hardware.efficiency_cores}E), "
                f"GPU: {gpu_status}")
        self.statusBar().showMessage(info)
    
    # ==================== Part Management Methods ====================
    
    def _add_part(self):
        """Add new parts from CAD files (supports multi-select).

        STL files load as single parts. STEP (``.step`` / ``.stp``)
        files are treated as multi-body assemblies: each body in the
        file is registered as its own Part with the CAD body name and
        auto-inferred part type.
        """
        filepaths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add CAD Parts (Select Multiple)",
            "",
            "CAD Files (*.stl *.step *.stp);;STL Files (*.stl);;STEP Files (*.step *.stp);;All Files (*)"
        )

        if filepaths:
            self.status_label.setText(f"Loading {len(filepaths)} file(s)...")
            self.progress_bar.setRange(0, len(filepaths))

            loaded_count = 0
            for i, filepath in enumerate(filepaths):
                self.progress_bar.setValue(i)

                ext = os.path.splitext(filepath)[1].lower()
                if ext in (".step", ".stp"):
                    # Multi-body assembly: one Part per body.
                    new_ids = self.part_manager.load_assembly_parts(filepath)
                else:
                    # Single-body CAD file (STL/OBJ/PLY/...).
                    part_id = self.part_manager.load_part(
                        filepath, position=None, preserve_origin=True
                    )
                    new_ids = [part_id] if part_id else []

                loaded_count += len(new_ids)

            self.progress_bar.setValue(len(filepaths))
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)

            if loaded_count > 0:
                self._update_parts_ui()
                self._update_visualization()

                if loaded_count > 1:
                    self.status_label.setText(f"Added {loaded_count} parts - positions preserved from CAD file")
                else:
                    self.status_label.setText(f"Added {loaded_count} part")
            else:
                self.status_label.setText("Failed to load parts")
    
    def _remove_selected_part(self):
        """Remove the currently selected part"""
        if self._selected_part_id:
            part = self.part_manager.get_part(self._selected_part_id)
            if part:
                name = part.name
                self.part_manager.remove_part(self._selected_part_id)
                self.visualizer.remove_part(self._selected_part_id)
                self._selected_part_id = None
                self._update_parts_ui()
                self._update_visualization()
                self.status_label.setText(f"Removed part: {name}")
    
    def _clear_all_parts(self):
        """Clear all parts"""
        reply = QMessageBox.question(
            self,
            "Clear All Parts",
            "Are you sure you want to remove all parts?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.part_manager.clear_all()
            self.visualizer.clear_visualization()
            self._selected_part_id = None
            self._update_parts_ui()
            self.status_label.setText("Cleared all parts")
    
    def _show_part_context_menu(self, pos):
        """Show context menu for part list"""
        item = self.parts_dialog.parts_list.itemAt(pos)
        if not item:
            return
        
        part_id = item.data(Qt.UserRole)
        part = self.part_manager.get_part(part_id)
        if not part:
            return
        
        menu = QMenu(self)
        
        # Duplicate action
        duplicate_action = QAction("Duplicate", self)
        duplicate_action.triggered.connect(lambda: self._duplicate_part(part_id))
        menu.addAction(duplicate_action)
        
        # Rename action
        rename_action = QAction("Rename", self)
        rename_action.triggered.connect(lambda: self._rename_part(part_id))
        menu.addAction(rename_action)
        
        menu.addSeparator()
        
        # Visibility toggle
        vis_action = QAction("Hide" if part.visible else "Show", self)
        vis_action.triggered.connect(lambda: self._toggle_part_visibility(part_id))
        menu.addAction(vis_action)
        
        menu.addSeparator()
        
        # Remove action
        remove_action = QAction("Remove", self)
        remove_action.triggered.connect(lambda: self._remove_part(part_id))
        menu.addAction(remove_action)
        
        menu.exec_(self.parts_dialog.parts_list.mapToGlobal(pos))
    
    def _duplicate_part(self, part_id: str):
        """Duplicate a specific part"""
        new_id = self.part_manager.duplicate_part(part_id, offset=(2.0, 0.0, 0.0))
        if new_id:
            self._update_parts_ui()
            self._update_visualization()
    
    def _duplicate_selected_part(self):
        """Duplicate the currently selected part"""
        if self._selected_part_id:
            self._duplicate_part(self._selected_part_id)
    
    def _rename_part(self, part_id: str):
        """Rename a part"""
        part = self.part_manager.get_part(part_id)
        if not part:
            return
        
        new_name, ok = QInputDialog.getText(
            self,
            "Rename Part",
            "Enter new name:",
            text=part.name
        )
        
        if ok and new_name:
            part.name = new_name
            self._update_parts_ui()
    
    def _toggle_part_visibility(self, part_id: str):
        """Toggle visibility of a part"""
        part = self.part_manager.get_part(part_id)
        if part:
            part.visible = not part.visible
            self.visualizer.set_part_visibility(part_id, part.visible)
            self._update_parts_ui()
    
    def _remove_part(self, part_id: str):
        """Remove a specific part"""
        self.part_manager.remove_part(part_id)
        self.visualizer.remove_part(part_id)
        if self._selected_part_id == part_id:
            self._selected_part_id = None
        self._update_parts_ui()
        self._update_visualization()
    
    def _arrange_parts_grid(self):
        """Arrange parts in a grid"""
        spacing, ok = QInputDialog.getDouble(
            self,
            "Grid Spacing",
            "Space between parts (inches):",
            value=1.0,
            min=0.1,
            max=100.0,
            decimals=2
        )
        
        if ok:
            self.part_manager.auto_arrange_grid(spacing)
            self._update_visualization()
    
    def _arrange_parts_line(self):
        """Arrange parts in a line"""
        axis, ok = QInputDialog.getItem(
            self,
            "Line Axis",
            "Arrange along axis:",
            ["X", "Y", "Z"],
            0,
            False
        )
        
        if ok:
            self.part_manager.auto_arrange_line(axis.lower(), spacing=1.0)
            self._update_visualization()
    
    def _update_parts_ui(self):
        """Update the parts list and info in both sidebar and dialog"""
        # Update dialog parts list
        self._sync_parts_to_dialog()
        
        # Count parts by type
        type_counts = {"cast": 0, "riser": 0, "riser_post": 0, "chill": 0}
        
        for part in self.part_manager.get_all_parts():
            type_counts[part.part_type.value] += 1
        
        # Update sidebar summary
        self._update_parts_summary()
        
        # Update info label in dialog
        count = self.part_manager.get_part_count()
        if count == 0:
            self.parts_dialog.parts_info_label.setText("No parts loaded")
            self.run_sim_btn.setEnabled(False)
            self.run_sim_action.setEnabled(False)
            self.riser_design_action.setEnabled(False)
            self.mesh_dialog.apply_ml_button.setEnabled(False)
            self.parts_dialog.remove_part_btn.setEnabled(False)
            self.parts_dialog.duplicate_btn.setEnabled(False)
            self.parts_dialog.clear_parts_btn.setEnabled(False)
            self.parts_dialog.confirm_types_btn.setEnabled(False)
        else:
            total_vol = self.part_manager.get_total_volume()
            bounds = self.part_manager.get_combined_bounds()
            if bounds is not None:
                dims = f"{bounds[0][1]-bounds[0][0]:.2f}\" x {bounds[1][1]-bounds[1][0]:.2f}\" x {bounds[2][1]-bounds[2][0]:.2f}\""
            else:
                dims = "N/A"
            
            # Show type breakdown
            type_info = []
            if type_counts["cast"] > 0:
                type_info.append(f"🔩{type_counts['cast']}")
            if type_counts["riser"] > 0:
                type_info.append(f"🔥{type_counts['riser']}")
            if type_counts["riser_post"] > 0:
                type_info.append(f"📍{type_counts['riser_post']}")
            if type_counts["chill"] > 0:
                type_info.append(f"❄️{type_counts['chill']}")
            
            self.parts_dialog.parts_info_label.setText(
                f"{count} part(s) | {' '.join(type_info)}\n"
                f"Total: {total_vol:.2f} in³ | {dims}"
            )
            self.run_sim_btn.setEnabled(self.simulation_worker is None or not self.simulation_worker.isRunning())
            self.run_sim_action.setEnabled(self.simulation_worker is None or not self.simulation_worker.isRunning())
            self.riser_design_action.setEnabled(True)  # Enable riser design tool
            self.mesh_dialog.apply_ml_button.setEnabled(True)
            self.parts_dialog.clear_parts_btn.setEnabled(True)
            self._refresh_surrogate_enabled()
        
        # Update stl_loader for backwards compatibility
        self.stl_loader._update_legacy_properties()
    
    def _update_visualization(self):
        """Update the 3D visualization with all parts"""
        try:
            self.visualizer.load_parts_from_manager(self.part_manager)
        except Exception as exc:
            logger.exception(f"Failed to update 3D visualization: {exc}")
            self.status_label.setText("Failed to update 3D view. See error log.")
    
    # ==================== Legacy STL Loading (Single Part) ====================
    
    def _load_stl_file(self):
        """Load STL file (legacy single-part mode - clears existing parts).

        Deprecated: kept for backwards compatibility but unreferenced by
        any menu/button. The file dialog now accepts STEP files too, but
        the single-part mode always loads only the first body if a STEP
        is selected. New code should use :meth:`_add_part`.
        """
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open CAD File",
            "",
            "CAD Files (*.stl *.step *.stp);;STL Files (*.stl);;STEP Files (*.step *.stp);;All Files (*)"
        )
        
        if filepath:
            self.status_label.setText("Loading STL file...")
            self.progress_bar.setRange(0, 0)  # Indeterminate
            
            # Clear existing parts and add new one
            self.part_manager.clear_all()
            part_id = self.part_manager.load_part(filepath)
            
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            
            if part_id:
                self._update_parts_ui()
                self._update_visualization()
                
                part = self.part_manager.get_part(part_id)
                self.status_label.setText("STL loaded successfully")
                self.mesh_setting_origin = "manual"
                self.last_ml_recommendation = None
                self.ml_status_label.setText("")
                
                # Show voxel size recommendations based on model dimensions
                self._show_voxel_recommendations(part_id)
            else:
                self.status_label.setText("Error loading STL")
                self.apply_ml_button.setEnabled(False)
                self.ml_status_label.setText("")
    
    def _show_voxel_recommendations(self, part_id: str):
        """Show voxel size recommendations after loading an STL"""
        try:
            part = self.part_manager.get_part(part_id)
            if not part or not part.loader:
                return
            
            bounds = part.loader.get_bounds_array()
            bounds_tuple = (
                (float(bounds[0][0]), float(bounds[0][1])),
                (float(bounds[1][0]), float(bounds[1][1])),
                (float(bounds[2][0]), float(bounds[2][1])),
            )
            
            # Get recommendations
            rec = get_voxel_size_recommendations(bounds_tuple)
            
            # Build info message
            dims = rec.model_dimensions
            current_size = self._current_voxel_size
            current_voxels = estimate_voxel_count(bounds_tuple, current_size, 0.4)
            
            # Print to console
            print(f"\n{'='*60}")
            print(f"Model: {dims[0]:.1f}\" x {dims[1]:.1f}\" x {dims[2]:.1f}\"")
            print(f"Current voxel size: {current_size}\" ({current_voxels:,} estimated voxels)")
            print(f"\n{rec.message}")
            print(f"Recommended: {rec.recommended_voxel_size}\" ({rec.estimated_voxels:,} voxels, ~{rec.estimated_memory_mb:.0f}MB)")
            print(f"\nAlternatives:")
            for vs, vox, mem, qual in rec.alternatives[:5]:
                marker = " ◀ recommended" if vs == rec.recommended_voxel_size else ""
                print(f"  {vs:.2f}\": {vox:,} voxels, {mem:.0f}MB ({qual}){marker}")
            print(f"{'='*60}\n")
            
            # Show dialog if current setting will cause issues
            if current_voxels > rec.estimated_voxels * 2:
                reply = QMessageBox.question(
                    self,
                    "Voxel Size Recommendation",
                    f"Model size: {dims[0]:.1f}\" × {dims[1]:.1f}\" × {dims[2]:.1f}\"\n\n"
                    f"Current voxel size ({current_size}\") will create ~{current_voxels:,} voxels.\n\n"
                    f"Recommended: {rec.recommended_voxel_size}\" ({rec.estimated_voxels:,} voxels)\n"
                    f"This will use ~{rec.estimated_memory_mb:.0f}MB memory.\n\n"
                    f"Would you like to use the recommended voxel size?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.Yes:
                    self._current_voxel_size = rec.recommended_voxel_size
                    self.status_label.setText(f"Voxel size set to {rec.recommended_voxel_size}\"")
                    print(f"Voxel size updated to {rec.recommended_voxel_size}\"")
        except Exception as e:
            print(f"Warning: Could not calculate recommendations: {e}")

    def _has_loaded_parts(self) -> bool:
        """Return True when parts are available for simulation."""
        return self.part_manager.get_part_count() > 0

    def _set_simulation_running(self, running: bool) -> None:
        """Toggle controls that should be disabled during a simulation."""
        can_run = (not running) and self._has_loaded_parts()
        self.run_sim_btn.setEnabled(can_run)
        self.run_sim_action.setEnabled(can_run)
        self.stop_sim_btn.setEnabled(running)
        self.stop_sim_action.setEnabled(running)

        self._refresh_surrogate_enabled()
        self.run_surrogate_action.setEnabled(can_run and self.surrogate_model is not None)
        self.stop_surrogate_action.setEnabled(running)
        self.surrogate_load_action.setEnabled(not running)
        self.surrogate_train_action.setEnabled(not running)
        self.surrogate_info_action.setEnabled((not running) and self.surrogate_model is not None)

        for control in (
            self.add_part_action,
            self.parts_manager_action,
            self.open_project_action,
            self.save_project_action,
            self.open_parts_btn,
            self.open_material_btn,
            self.open_analysis_btn,
            self.open_mesh_btn,
            self.open_performance_btn,
        ):
            control.setEnabled(not running)

        self.parts_dialog.add_part_btn.setEnabled(not running)

    def _cancel_simulation(self) -> None:
        """Cancel the active simulation if one is running."""
        if self.simulation_worker is None or not self.simulation_worker.isRunning():
            return

        self.simulation_worker.cancel()
        self.stop_sim_btn.setEnabled(False)
        self.stop_sim_action.setEnabled(False)
        self.status_label.setText("Cancelling simulation...")
        self.progress_bar.setFormat("Cancelling...")

    def _collect_project_state(self) -> dict:
        """Capture the current desktop session state."""
        result_type = self.result_type_combo.currentText() or "Shrinkage/Porosity"
        ui_state = {
            "show_model": self.show_mesh_check.isChecked(),
            "show_results": self.show_results_check.isChecked(),
            "result_type": result_type,
            "threshold": int(self.threshold_slider.value()),
            "time_slider": int(self.time_slider.value()),
            "fraction_liquid_playback_speed": self._fraction_liquid_playback_speed,
            "splitter_sizes": list(self.main_splitter.sizes()) if hasattr(self, "main_splitter") else [],
        }
        return build_project_state(
            self.part_manager,
            alloy_name=self._current_alloy,
            voxel_size=self._current_voxel_size,
            refinement_level=int(self._current_refinement),
            memory_mode=self._current_memory_mode,
            analysis_options=self._get_effective_analysis_options(),
            performance_options=self._performance_options,
            solidification_params=self._solidification_params,
            mesh_setting_origin=self.mesh_setting_origin,
            current_vis_mode=self._current_vis_mode,
            ui_state=ui_state,
        ).to_dict()

    def _save_project(self) -> None:
        """Save the current desktop session to a project file."""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project",
            self._current_project_path or "",
            "Solidus Project (*.solidus-project.json);;JSON Files (*.json)",
        )
        if not filepath:
            return

        payload = self._collect_project_state()
        project_dir = os.path.dirname(filepath)
        for part in payload["parts"]:
            try:
                part["filepath"] = os.path.relpath(part["filepath"], project_dir)
            except ValueError:
                # Keep absolute path when a relative path cannot be formed.
                pass

        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        self._current_project_path = filepath
        self.status_label.setText(f"Project saved: {os.path.basename(filepath)}")

    def _load_project(self) -> None:
        """Load a desktop project file."""
        if self.simulation_worker is not None and self.simulation_worker.isRunning():
            QMessageBox.warning(self, "Simulation Running", "Stop the current simulation before loading a project.")
            return

        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            self._current_project_path or "",
            "Solidus Project (*.solidus-project.json);;JSON Files (*.json)",
        )
        if not filepath:
            return

        with open(filepath, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        project_state = load_project_state(payload).to_dict()
        self._apply_project_state(project_state, filepath)

    def _apply_project_state(self, project_state: dict, filepath: str) -> None:
        """Apply a normalized project state to the current window."""
        simulation_state = project_state["simulation"]
        ui_state = project_state.get("ui_state", {})
        project_dir = os.path.dirname(filepath)
        type_map = {
            "cast": PartType.CAST,
            "riser": PartType.RISER,
            "riser_post": PartType.RISER_POST,
            "chill": PartType.CHILL,
            "core": PartType.CORE,
        }

        self.current_results = None
        self.result_type_combo.setEnabled(False)
        self.visualizer.clear_visualization()
        self.part_manager.clear_all()
        self._selected_part_id = None

        missing_parts = []
        for part_data in project_state["parts"]:
            resolved_path = self._resolve_project_part_path(project_dir, part_data["filepath"])
            if resolved_path is None:
                missing_parts.append(part_data["filepath"])
                continue

            part_id = self.part_manager.load_part(resolved_path, name=part_data["name"], preserve_origin=True)
            if not part_id:
                missing_parts.append(part_data["filepath"])
                continue

            part = self.part_manager.get_part(part_id)
            if part is None:
                missing_parts.append(part_data["filepath"])
                continue

            part.set_part_type(type_map.get(part_data["part_type"], PartType.CAST))
            part.set_position(*part_data["position"])
            part.set_rotation(*part_data["rotation"])
            part.set_scale(part_data["scale"])
            part.color = tuple(part_data["color"])
            part.visible = part_data["visible"]
            part.is_open_top = part_data["is_open_top"]

        self._on_alloy_confirmed(simulation_state["alloy_name"])
        self.material_dialog.set_alloy(simulation_state["alloy_name"])
        self.mesh_dialog.set_values(
            simulation_state["voxel_size"],
            simulation_state["refinement_level"],
            simulation_state["memory_mode"],
        )
        self._on_mesh_settings_confirmed(
            simulation_state["voxel_size"],
            simulation_state["refinement_level"],
            simulation_state["memory_mode"],
        )
        self.analysis_dialog.set_options(simulation_state["analysis_options"])
        self._on_analysis_options_confirmed(simulation_state["analysis_options"])
        self.performance_dialog.set_options(simulation_state["performance_options"])
        self._on_performance_options_confirmed(simulation_state["performance_options"])
        solid_params = simulation_state.get("solidification_params", {})
        if solid_params:
            self.solidification_dialog.set_options(solid_params)
            self._on_solidification_params_confirmed(solid_params)
        self.mesh_setting_origin = simulation_state.get("mesh_setting_origin", "manual")
        self._current_vis_mode = simulation_state.get("current_vis_mode", "shrinkage")

        self._update_parts_ui()
        self._update_visualization()
        self.show_mesh_check.setChecked(bool(ui_state.get("show_model", True)))
        self.show_results_check.setChecked(bool(ui_state.get("show_results", True)))
        self.threshold_slider.setValue(int(ui_state.get("threshold", 30)))
        self.time_slider.setValue(
            self._coerce_fraction_liquid_slider_value(int(ui_state.get("time_slider", 0)))
        )

        try:
            saved_speed = float(ui_state.get("fraction_liquid_playback_speed", 1.0))
            if saved_speed in FRACTION_LIQUID_PLAYBACK_SPEEDS:
                self._fraction_liquid_playback_speed = saved_speed
                idx = self.fraction_liquid_speed_combo.findData(saved_speed)
                if idx >= 0:
                    self.fraction_liquid_speed_combo.setCurrentIndex(idx)
            else:
                self._fraction_liquid_playback_speed = 1.0
                default_idx = self.fraction_liquid_speed_combo.findData(1.0)
                if default_idx >= 0:
                    self.fraction_liquid_speed_combo.setCurrentIndex(default_idx)
        except (TypeError, ValueError):
            self._fraction_liquid_playback_speed = 1.0
            default_idx = self.fraction_liquid_speed_combo.findData(1.0)
            if default_idx >= 0:
                self.fraction_liquid_speed_combo.setCurrentIndex(default_idx)

        result_type = str(ui_state.get("result_type", "Shrinkage/Porosity"))
        combo_index = self.result_type_combo.findText(result_type)
        if combo_index >= 0:
            self.result_type_combo.setCurrentIndex(combo_index)

        splitter_sizes = ui_state.get("splitter_sizes")
        if isinstance(splitter_sizes, list) and len(splitter_sizes) == 2:
            self.main_splitter.setSizes([int(splitter_sizes[0]), int(splitter_sizes[1])])

        self._current_project_path = filepath
        self.status_label.setText(f"Project loaded: {os.path.basename(filepath)}")

        if missing_parts:
            QMessageBox.warning(
                self,
                "Missing Project Parts",
                "Some part files could not be loaded:\n\n" + "\n".join(missing_parts),
            )

    def _resolve_project_part_path(self, project_dir: str, stored_path: str) -> Optional[str]:
        """Resolve a stored part path relative to a project file."""
        if os.path.isabs(stored_path) and os.path.exists(stored_path):
            return stored_path

        candidate = os.path.normpath(os.path.join(project_dir, stored_path))
        if os.path.exists(candidate):
            return candidate
        return None
    
    def _on_alloy_changed(self, alloy_name: str):
        """Update alloy info display"""
        try:
            alloy = get_alloy(alloy_name)
            liquidus_f = (alloy.liquidus_temp * 9/5) + 32
            info = (
                f"Feeding: {alloy.feeding_distance_multiplier}x thickness\n"
                f"Liquidus: {liquidus_f:.0f}°F"
            )
            self.alloy_info_label.setText(info)
        except:
            self.alloy_info_label.setText("")

    def _purge_previous_simulation_memory(self) -> None:
        """Release prior simulation memory while preserving loaded parts."""
        previous_worker = self.simulation_worker
        if previous_worker is not None and previous_worker.isRunning():
            return

        self.current_results = None

        if hasattr(self, "result_type_combo"):
            self.result_type_combo.setEnabled(False)

        if hasattr(self, "visualizer"):
            self.visualizer.clear_visualization()

        if previous_worker is not None:
            previous_worker.simulation = None
            previous_worker.voxel_mesh = None
            self.simulation_worker = None

        try:
            reset_memory_manager()
        except Exception as exc:
            logger.warning("Failed to reset simulation memory manager", context={"error": str(exc)})

        gc.collect()
    
    def _run_simulation(self):
        """Run solidification simulation"""
        if not self.stl_loader.is_valid():
            QMessageBox.warning(self, "Error", "Please load a CAD file first")
            return
        
        # PRE-VALIDATION: Check memory requirements before starting
        bounds = self.stl_loader.get_bounds_array()
        bounds_tuple = (
            (float(bounds[0][0]), float(bounds[0][1])),
            (float(bounds[1][0]), float(bounds[1][1])),
            (float(bounds[2][0]), float(bounds[2][1])),
        )
        
        # Estimate fill ratio from mesh volume
        bbox_volume = (bounds[0][1] - bounds[0][0]) * (bounds[1][1] - bounds[1][0]) * (bounds[2][1] - bounds[2][0])
        mesh_volume = self.stl_loader.volume if self.stl_loader.volume > 0 else bbox_volume * 0.5
        fill_ratio = max(0.1, min(mesh_volume / bbox_volume, 0.9)) if bbox_volume > 0 else 0.5
        
        voxel_size = self._current_voxel_size
        
        # Validate memory requirements
        validation = validate_mesh_parameters(bounds_tuple, voxel_size, fill_ratio)
        
        if not validation.can_proceed:
            # CRITICAL or ERROR - Cannot proceed
            if validation.recommended_voxel_size:
                reply = QMessageBox.critical(
                    self,
                    "Memory Limit Exceeded",
                    f"{validation.message}\n\n"
                    f"Estimated voxels: {validation.estimated_voxels:,}\n"
                    f"Maximum safe: {validation.max_allowed_voxels:,}\n\n"
                    f"Would you like to automatically adjust the voxel size to {validation.recommended_voxel_size:.3f}\"?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.Yes:
                    self._current_voxel_size = validation.recommended_voxel_size
                    self.mesh_dialog.voxel_size_spin.setValue(validation.recommended_voxel_size)
                    voxel_size = validation.recommended_voxel_size
                    self._update_mesh_summary()  # Update sidebar summary
                    # Re-validate with new size
                    validation = validate_mesh_parameters(bounds_tuple, voxel_size, fill_ratio)
                    if not validation.can_proceed:
                        QMessageBox.critical(
                            self,
                            "Still Too Large",
                            "Even with adjusted voxel size, the mesh is too large. "
                            "Please manually increase voxel size further."
                        )
                        return
                else:
                    return
            else:
                QMessageBox.critical(
                    self,
                    "Memory Limit Exceeded",
                    f"{validation.message}\n\n"
                    f"Please increase the voxel size to reduce memory usage."
                )
                return
        
        elif validation.severity == MemorySeverity.WARNING:
            # Warning - Ask user to confirm
            mem_info = get_system_memory_info()
            reply = QMessageBox.warning(
                self,
                "Large Mesh Warning",
                f"{validation.message}\n\n"
                f"Estimated voxels: {validation.estimated_voxels:,}\n"
                f"Estimated memory: ~{validation.estimated_memory_mb:.0f} MB\n"
                f"Available RAM: ~{mem_info['available_ram_gb']:.1f} GB\n\n"
                f"Do you want to continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
        
        # Disable controls
        self._set_simulation_running(True)

        # Stop any active fraction liquid playback
        self._stop_fraction_liquid_playback()

        # Update status with memory info
        self.status_label.setText(f"Starting simulation ({validation.estimated_voxels:,} estimated voxels)...")
        
        # Clear previous simulation outputs without removing loaded parts.
        self._purge_previous_simulation_memory()
        
        # Show combined mesh (includes all parts: casting, risers, chills)
        combined_mesh = self.part_manager.get_combined_mesh()
        if combined_mesh is not None:
            self.visualizer.load_mesh(
                combined_mesh.vertices,
                combined_mesh.faces
            )
        else:
            # Fall back to just the STL loader mesh
            self.visualizer.load_mesh(
                self.stl_loader.get_vertices(),
                self.stl_loader.get_faces()
            )
        
        # Use stored memory mode
        memory_mode = self._current_memory_mode
        
        # Log simulation start
        logger.info("Starting simulation", context={
            'alloy': self._current_alloy,
            'voxel_size': voxel_size,
            'refinement': int(self._current_refinement),
            'memory_mode': memory_mode,
            'estimated_voxels': validation.estimated_voxels,
            'estimated_memory_mb': validation.estimated_memory_mb
        })
        
        # Start simulation in background thread
        self.simulation_worker = SimulationWorker(
            self.stl_loader,
            self._current_alloy,
            voxel_size,
            int(self._current_refinement),
            memory_mode=memory_mode,
            settings_origin=self.mesh_setting_origin,
            part_manager=self.part_manager,  # Pass part manager for type tracking
            analysis_options=self._get_effective_analysis_options(),  # Pass normalized desktop options
            performance_options=self._performance_options,  # Pass performance options
            solidification_params=self._solidification_params  # Pass solidification parameters
        )
        
        self.simulation_worker.progress.connect(self._on_simulation_progress)
        self.simulation_worker.detailed_progress.connect(self._on_detailed_progress)
        self.simulation_worker.finished.connect(self._on_simulation_finished)
        
        # Reset progress tracking
        self.slice_start_times = []
        self.last_slice_index = 0
        self.simulation_start_time = time.time()
        
        self.progress_bar.setRange(0, 0)  # Indeterminate initially
        self.progress_bar.setFormat("Starting...")
        self.simulation_worker.start()
    
    def _on_simulation_progress(self, message: str):
        """Handle simulation progress updates"""
        self.status_label.setText(message)
    
    def _on_detailed_progress(self, current: int, total: int, stage: str):
        """Handle detailed Z-slice progress with time estimation"""
        if total == 0:
            return
        
        # Update progress bar range if needed
        if self.progress_bar.maximum() != total:
            self.progress_bar.setRange(0, total)
        
        self.progress_bar.setValue(current)
        
        # Calculate time estimation
        current_time = time.time()
        
        # Track slice times for better ETA calculation
        if current > self.last_slice_index:
            self.slice_start_times.append(current_time)
            self.last_slice_index = current
        
        # Calculate ETA based on average slice time
        if len(self.slice_start_times) >= 2:
            # Calculate average time per slice from recent slices
            recent_slices = min(10, len(self.slice_start_times))
            time_diffs = []
            for i in range(1, recent_slices):
                time_diffs.append(self.slice_start_times[-i] - self.slice_start_times[-(i+1)])
            
            avg_time_per_slice = sum(time_diffs) / len(time_diffs) if time_diffs else 0
            remaining_slices = total - current
            estimated_seconds = avg_time_per_slice * remaining_slices
            
            # Format ETA
            if estimated_seconds < 60:
                eta_str = f"{int(estimated_seconds)}s"
            elif estimated_seconds < 3600:
                minutes = int(estimated_seconds / 60)
                seconds = int(estimated_seconds % 60)
                eta_str = f"{minutes}m {seconds}s"
            else:
                hours = int(estimated_seconds / 3600)
                minutes = int((estimated_seconds % 3600) / 60)
                eta_str = f"{hours}h {minutes}m"
            
            # Update progress bar format with ETA
            percentage = int((current / total) * 100)
            self.progress_bar.setFormat(f"{percentage}% - ETA: {eta_str}")
        else:
            # Not enough data for ETA yet
            percentage = int((current / total) * 100)
            self.progress_bar.setFormat(f"{percentage}%")
        
        # Update status label
        self.status_label.setText(f"Processing {stage}: Z-slice {current}/{total}")
    
    def _on_simulation_finished(self, success: bool, status: str):
        """Handle simulation completion"""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100 if success else 0)
        self._set_simulation_running(False)
        
        if success:
            # Log simulation completion
            elapsed_time = time.time() - self.simulation_start_time if hasattr(self, 'simulation_start_time') else 0
            logger.info("Simulation completed successfully", context={
                'elapsed_time_seconds': elapsed_time,
                'alloy': self._current_alloy
            })
            logger.log_performance("Complete simulation", elapsed_time, 
                                   self.simulation_worker.voxel_mesh.get_voxel_count() if self.simulation_worker.voxel_mesh else 0)
            
            # Get results
            simulation = self.simulation_worker.simulation
            voxel_mesh = self.simulation_worker.voxel_mesh
            
            # Store results and prepare visualizations
            self.current_results = simulation.export_results()
            
            # Enable result type selector and configure available options
            self._setup_result_type_options()
            
            # Visualize first available result type
            self._update_results_visualization()
            
            self.status_label.setText("✓ Visualization complete")
            self.progress_bar.setFormat("Complete")
        elif status == "cancelled":
            logger.info("Simulation cancelled", context={'alloy': self._current_alloy})
            self.current_results = None
            self.status_label.setText("Simulation cancelled")
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Cancelled")
        else:
            logger.error("Simulation failed", context={'alloy': self._current_alloy})
            self.status_label.setText("Simulation failed")
            self.progress_bar.setFormat("Failed")
            QMessageBox.critical(self, "Error", "Simulation failed. Check console for details.")

    # ------------------------------------------------------------------
    # Surrogate simulation handlers
    # ------------------------------------------------------------------
    def _load_surrogate_model(self) -> None:
        """Load a surrogate model artifact (.json manifest or model dir)."""
        from config.settings import get_settings

        settings = get_settings()
        model_dir = str(getattr(settings.surrogate, "model_dir", "models/surrogate"))
        start_dir = os.path.abspath(model_dir)
        if not os.path.isdir(start_dir):
            start_dir = os.path.abspath("models")

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Surrogate Model",
            start_dir,
            "Surrogate models (*.json *.npz *.mlmodel *.mlx);;All files (*.*)",
        )
        if not path:
            return

        # STEP/STL are CAD geometry for File > Add Part, not surrogate models.
        if path.lower().endswith((".stl", ".step", ".stp")):
            QMessageBox.information(
                self,
                "Surrogate Model",
                "STEP/STL files are CAD geometry — load them via File > Add Part, "
                "not as a surrogate model.\n\n"
                "The surrogate model is a trained artifact (a folder with "
                "manifest.json, or a .json manifest). Bundled models:\n"
                "  • models/surrogate/demo\n"
                "  • models/surrogate/trained",
            )
            return

        from core import surrogate

        result = surrogate.load_model(path)
        if result.error or result.model is None:
            QMessageBox.critical(self, "Surrogate Model", f"Failed to load model:\n{result.error}")
            return

        self.surrogate_model = result.model
        self._persist_surrogate_model_path(path)
        backend = self.surrogate_model.backend
        device = surrogate.backend_device_name(backend)
        self.surrogate_info_action.setEnabled(True)
        self._refresh_surrogate_enabled()
        logger.info("Surrogate model loaded", context={
            'name': self.surrogate_model.manifest.name,
            'backend': backend.value,
        })
        QMessageBox.information(
            self,
            "Surrogate Model Loaded",
            f"Loaded: {self.surrogate_model.manifest.name}\n"
            f"Backend: {backend.value} ({device})\n\n"
            f"{surrogate.format_metrics(self.surrogate_model.manifest.accuracy)}",
        )

    def _show_surrogate_model_info(self) -> None:
        dialog = SurrogateModelInfoDialog(self.surrogate_model, self)
        dialog.exec_()

    def _open_surrogate_settings_dialog(self) -> None:
        from ui.surrogate_dialog import SurrogateSettingsDialog

        dialog = SurrogateSettingsDialog(self)
        dialog.exec_()

    def _autoload_surrogate_model(self) -> None:
        """Restore the last-used surrogate model at startup (no dialogs).

        Preference order:
        1. The model path persisted from the last load/train (if it exists).
        2. ``<model_dir>/trained`` (most recent training output).
        3. ``<model_dir>/demo`` (bundled reference model).
        When the deep model feature is enabled, ``models/deep_model/trained``
        is also a candidate (checked before the numpy demo model).
        """
        try:
            from config.settings import get_settings
            from core import surrogate

            settings = get_settings()
            model_dir = os.path.abspath(str(getattr(settings.surrogate, "model_dir", "models/surrogate")))

            candidates = []
            last = str(getattr(settings.surrogate, "last_model_path", "") or "")
            if last and os.path.exists(last):
                candidates.append(last)
            for name in ("trained", "demo"):
                path = os.path.join(model_dir, name)
                if os.path.isdir(path) and os.path.exists(os.path.join(path, "manifest.json")):
                    candidates.append(path)
            if bool(getattr(settings.deep, "enabled", False)):
                deep_trained = os.path.join(
                    os.path.abspath(str(getattr(settings.deep, "model_dir", "models/deep_model"))),
                    "trained",
                )
                if os.path.isdir(deep_trained) and os.path.exists(
                    os.path.join(deep_trained, "manifest.json")
                ):
                    candidates.insert(0, deep_trained)

            for path in candidates:
                result = surrogate.load_model(path)
                if result.model is None:
                    continue
                self.surrogate_model = result.model
                self.surrogate_info_action.setEnabled(True)
                self._refresh_surrogate_enabled()
                self.status_label.setText(
                    f"Surrogate model loaded: {result.model.manifest.name} ({result.model.manifest.backend})"
                )
                return
            self.status_label.setText("Surrogate: no model found — use Surrogate > Load or Retrain")
        except Exception:
            pass

    def _persist_surrogate_model_path(self, path: str) -> None:
        """Remember the last-used model so it auto-loads next launch."""
        try:
            from config.settings import get_settings

            settings = get_settings()
            settings.surrogate.last_model_path = str(path)
            settings.save_user_override("surrogate", "last_model_path", str(path))
        except Exception:
            pass

    def _open_surrogate_train_dialog(self) -> None:
        """Ask for training options, then train in a background thread."""
        from config.settings import get_settings
        from core.surrogate.dataset import DEFAULT_DATASET_DIR
        from core.surrogate.training import DEFAULT_MODEL_OUTPUT
        from ui.surrogate_dialog import SurrogateTrainDialog

        if self.surrogate_worker is not None and self.surrogate_worker.isRunning():
            QMessageBox.information(self, "Surrogate Model", "A surrogate run is in progress.")
            return

        dialog = SurrogateTrainDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return
        options = dialog.options()

        settings = get_settings()
        model_dir = str(getattr(settings.surrogate, "model_dir", "models/surrogate"))
        output_dir = os.path.join(os.path.abspath(model_dir), "trained")

        self._set_simulation_running(True)
        self._stop_fraction_liquid_playback()
        self.status_label.setText("Training surrogate model...")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("Training...")

        self.surrogate_train_worker = SurrogateTrainWorker(
            data_dir=DEFAULT_DATASET_DIR,
            output_dir=output_dir,
            grid_resolution=int(getattr(settings.surrogate, "grid_resolution", 64) or 64),
            generate_first=options["generate_first"],
            model_name="Trained Linear Surrogate",
        )
        self.surrogate_train_worker.progress.connect(self._on_simulation_progress)
        self.surrogate_train_worker.detailed_progress.connect(self._on_detailed_progress)
        self.surrogate_train_worker.finished.connect(
            lambda success, status, summary: self._on_surrogate_train_finished(
                success, status, summary, load_model=options["load_model"]
            )
        )
        self.surrogate_train_worker.start()

    def _on_surrogate_train_finished(self, success: bool, status: str, summary: dict, load_model: bool = True) -> None:
        """Handle surrogate training completion."""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100 if success else 0)
        self._set_simulation_running(False)

        if not success:
            self.status_label.setText("Surrogate training failed")
            self.progress_bar.setFormat("Failed")
            if status == "cancelled":
                self.status_label.setText("Surrogate training cancelled")
                self.progress_bar.setFormat("Cancelled")
            return

        headline = (summary or {}).get("headline", {})
        records = (summary or {}).get("records", 0)
        train = (summary or {}).get("train", 0)
        test = (summary or {}).get("test", 0)
        saved_to = (summary or {}).get("saved_to", "")

        lines = [
            f"Surrogate model trained from {records} run(s) ({train} train / {test} test).",
            "",
            f"  RMSE: {headline.get('rmse_s', 0):.1f}s",
            f"  MAE:  {headline.get('mae_s', 0):.1f}s",
            f"  Confidence: {headline.get('confidence', 0):.2f}",
            "",
            f"Saved to: {saved_to}",
        ]

        loaded_name = None
        if load_model and saved_to:
            from core import surrogate

            result = surrogate.load_model(saved_to)
            if result.model is not None:
                self.surrogate_model = result.model
                self._persist_surrogate_model_path(saved_to)
                self._refresh_surrogate_enabled()
                loaded_name = result.model.manifest.name

        if loaded_name:
            lines.append("")
            lines.append(f"The new model is loaded and ready: {loaded_name}")

        self.status_label.setText("✓ Surrogate model trained")
        self.progress_bar.setFormat("Complete")
        QMessageBox.information(self, "Surrogate Model Trained", "\n".join(lines))

    def _refresh_surrogate_enabled(self) -> None:
        has_parts = self._has_loaded_parts()
        has_model = self.surrogate_model is not None
        self.run_surrogate_action.setEnabled(has_parts and has_model)
        self.surrogate_info_action.setEnabled(has_model)

    # ------------------------------------------------------------------
    # Deep model handlers (core/deep_model — optional torch module)
    # ------------------------------------------------------------------
    def _deep_worker_busy(self) -> bool:
        """True when a deep train or export worker is active."""
        train_busy = self.deep_worker is not None and self.deep_worker.isRunning()
        export_busy = self.deep_export_worker is not None and self.deep_export_worker.isRunning()
        return bool(train_busy or export_busy)

    def _open_deep_train_dialog(self) -> None:
        """Ask for hyperparameters, then train in a background thread."""
        from config.settings import get_settings
        from core.deep_model.autotrain import load_state
        from core.surrogate.dataset import DEFAULT_DATASET_DIR
        from core.deep_model.training import DEFAULT_MODEL_OUTPUT
        from ui.deep_model_dialog import DeepTrainDialog

        if self._deep_worker_busy():
            QMessageBox.information(self, "Deep Model", "A deep training or export is already in progress.")
            return

        dialog = DeepTrainDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return
        options = dialog.options()

        settings = get_settings()
        model_dir = str(getattr(settings.deep, "model_dir", "models/deep_model"))
        output_dir = os.path.join(os.path.abspath(model_dir), "trained")

        self._set_simulation_running(True)
        self._stop_fraction_liquid_playback()
        self.status_label.setText("Training deep model...")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("Training...")

        self.deep_worker = DeepTrainWorker(
            data_dir=DEFAULT_DATASET_DIR,
            output_dir=output_dir,
            grid_resolution=options["grid_resolution"],
            epochs=options["epochs"],
            batch_size=options["batch_size"],
            learning_rate=options["learning_rate"],
            min_records=int(getattr(settings.deep, "min_records", 4) or 4),
            model_name="Trained Deep Surrogate (3D U-Net)",
        )
        self.deep_worker.progress.connect(self._on_simulation_progress)
        self.deep_worker.detailed_progress.connect(self._on_detailed_progress)
        self.deep_worker.finished.connect(
            lambda success, status, summary: self._on_deep_train_finished(
                success, status, summary, load_model=options["load_model"]
            )
        )
        self.deep_worker.start()

    def _on_deep_train_finished(self, success: bool, status: str, summary: dict, load_model: bool = True) -> None:
        """Handle deep training completion (manual or auto-train path)."""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100 if success else 0)
        self._set_simulation_running(False)

        if not success:
            self.status_label.setText("Deep training failed")
            self.progress_bar.setFormat("Failed")
            if status == "cancelled":
                self.status_label.setText("Deep training cancelled")
                self.progress_bar.setFormat("Cancelled")
            return

        headline = (summary or {}).get("headline", {})
        records = (summary or {}).get("records", 0)
        train = (summary or {}).get("train", 0)
        test = (summary or {}).get("test", 0)
        saved_to = (summary or {}).get("saved_to", "")

        lines = [
            f"Deep model trained from {records} run(s) ({train} train / {test} test).",
            "",
            f"  RMSE: {headline.get('rmse_s', 0):.1f}s",
            f"  MAE:  {headline.get('mae_s', 0):.1f}s",
            f"  Confidence: {headline.get('confidence', 0):.2f}",
            "",
            f"Saved to: {saved_to}",
        ]

        loaded_name = None
        if load_model and saved_to:
            from core import surrogate

            result = surrogate.load_model(saved_to)
            if result.model is not None:
                self.surrogate_model = result.model
                self._persist_deep_model_path(saved_to)
                self._refresh_surrogate_enabled()
                loaded_name = result.model.manifest.name

        if loaded_name:
            lines.append("")
            lines.append(f"The new deep model is loaded and ready: {loaded_name}")

        self.status_label.setText("✓ Deep model trained")
        self.progress_bar.setFormat("Complete")
        QMessageBox.information(self, "Deep Model Trained", "\n".join(lines))

    def _persist_deep_model_path(self, path: str) -> None:
        """Remember the last-used deep model so it auto-loads next launch."""
        try:
            from config.settings import get_settings

            settings = get_settings()
            settings.deep.last_model_path = str(path)
            settings.save_user_override("deep", "last_model_path", str(path))
        except Exception:
            pass

    def _load_deep_model(self) -> None:
        """Load a deep model artifact directory into the surrogate slot."""
        from config.settings import get_settings

        settings = get_settings()
        model_dir = str(getattr(settings.deep, "model_dir", "models/deep_model"))
        start_dir = os.path.abspath(model_dir)
        if not os.path.isdir(start_dir):
            start_dir = os.path.abspath("models")

        path = QFileDialog.getExistingDirectory(
            self, "Load Deep Model (directory with manifest.json)", start_dir
        )
        if not path:
            return

        from core import surrogate

        result = surrogate.load_model(path)
        if result.error or result.model is None:
            QMessageBox.critical(self, "Deep Model", f"Failed to load model:\n{result.error}")
            return

        self.surrogate_model = result.model
        self._persist_deep_model_path(path)
        self.surrogate_info_action.setEnabled(True)
        self._refresh_surrogate_enabled()
        QMessageBox.information(
            self,
            "Deep Model Loaded",
            f"Loaded: {self.surrogate_model.manifest.name}\n"
            f"Backend: torch (deep 3D U-Net)\n\n"
            f"{surrogate.format_metrics(self.surrogate_model.manifest.accuracy)}\n\n"
            "Run it with Surrogate > Run Surrogate Simulation.",
        )

    def _export_deep_model(self) -> None:
        """Pick an NPU target and export the trained deep model."""
        from config.settings import get_settings
        from ui.deep_model_dialog import DeepExportDialog

        if self._deep_worker_busy():
            QMessageBox.information(self, "Deep Model", "A deep training or export is already in progress.")
            return

        settings = get_settings()
        model_dir = os.path.join(
            os.path.abspath(str(getattr(settings.deep, "model_dir", "models/deep_model"))),
            "trained",
        )
        if not os.path.isdir(model_dir):
            QMessageBox.information(
                self,
                "Deep Model",
                "No trained deep model found. Train one first (Deep Model > Train Deep Model...).",
            )
            return

        dialog = DeepExportDialog(self, model_dir=model_dir)
        if dialog.exec_() != QDialog.Accepted:
            return
        options = dialog.options()

        self.deep_exporting = True
        self.status_label.setText(f"Exporting deep model for {options['target']}...")
        self.deep_export_worker = DeepExportWorker(
            model_dir=options["model_dir"],
            target=options["target"],
            out_dir=options["out_dir"],
        )
        self.deep_export_worker.progress.connect(self._on_simulation_progress)
        self.deep_export_worker.finished.connect(self._on_deep_export_finished)
        self.deep_export_worker.start()

    def _on_deep_export_finished(self, success: bool, status: str, result: dict) -> None:
        self.deep_exporting = False
        if not success:
            self.status_label.setText("Deep model export failed")
            return
        lines = [
            f"Export complete: {str(result.get('target', '')).upper()}",
            "",
            f"  Artifact: {result.get('artifact_path', '')}",
            f"  Quantization: {result.get('quantization', '')}",
            f"  Validated: {'yes' if result.get('validated') else 'no'}",
        ]
        note = result.get("note", "")
        if note:
            lines.append("")
            lines.append(note)
        self.status_label.setText("✓ Deep model export complete")
        QMessageBox.information(self, "Deep Model Export", "\n".join(lines))

    def _setup_deep_autotrain(self) -> None:
        """Set up the dataset watcher + initial policy check when enabled."""
        try:
            from config.settings import get_settings

            settings = get_settings()
            self.deep_autotrain_action.setChecked(bool(settings.deep.enabled))
            if settings.deep.enabled:
                self._start_deep_dataset_watcher()
                self._start_deep_auto_train()
        except Exception:
            pass

    def _toggle_deep_autotrain(self, checked: bool) -> None:
        """Enable/disable the auto-train watcher from the menu."""
        try:
            from config.settings import get_settings

            settings = get_settings()
            settings.deep.enabled = bool(checked)
            settings.save_user_override("deep", "enabled", settings.deep.enabled)
            if checked:
                self._start_deep_dataset_watcher()
                self.status_label.setText("Deep auto-train enabled — watching captured runs")
                self._start_deep_auto_train()
            else:
                self._stop_deep_dataset_watcher()
                self.status_label.setText("Deep auto-train disabled")
        except Exception:
            pass

    def _start_deep_dataset_watcher(self) -> None:
        """Watch the surrogate dataset directory for new captured runs."""
        try:
            from core.surrogate.dataset import DEFAULT_DATASET_DIR

            if self.deep_dataset_watcher is None:
                self.deep_dataset_watcher = QFileSystemWatcher(self)
                self.deep_dataset_watcher.directoryChanged.connect(
                    lambda _path: self.deep_debounce_timer.start()
                )
            if not self.deep_dataset_watcher.directories():
                os.makedirs(DEFAULT_DATASET_DIR, exist_ok=True)
                self.deep_dataset_watcher.addPath(DEFAULT_DATASET_DIR)
        except Exception:
            pass

    def _stop_deep_dataset_watcher(self) -> None:
        if self.deep_dataset_watcher is not None:
            self.deep_dataset_watcher.removePaths(self.deep_dataset_watcher.directories())
        self.deep_debounce_timer.stop()

    def _start_deep_auto_train(self) -> None:
        """Run the auto-train policy in a background thread (debounced)."""
        if self._deep_worker_busy():
            return
        try:
            from config.settings import get_settings
            from core.deep_model.autotrain import load_state
            from core.surrogate.dataset import DEFAULT_DATASET_DIR

            settings = get_settings()
            if not settings.deep.enabled:
                return
            model_dir = str(getattr(settings.deep, "model_dir", "models/deep_model"))
            output_dir = os.path.join(os.path.abspath(model_dir), "trained")
            state = load_state(output_dir)

            self.deep_worker = DeepAutoTrainWorker(
                data_dir=DEFAULT_DATASET_DIR,
                output_dir=output_dir,
                grid_resolution=int(getattr(settings.deep, "grid_resolution", 32) or 32),
                epochs=int(getattr(settings.deep, "epochs", 100) or 100),
                batch_size=int(getattr(settings.deep, "batch_size", 4) or 4),
                learning_rate=float(getattr(settings.deep, "learning_rate", 1e-3) or 1e-3),
                min_records=int(getattr(settings.deep, "min_records", 4) or 4),
                cooldown_seconds=int(getattr(settings.deep, "cooldown_seconds", 300) or 300),
                state=state,
                export_in_progress=self.deep_exporting,
            )
            self.deep_worker.progress.connect(self._on_simulation_progress)
            self.deep_worker.detailed_progress.connect(self._on_detailed_progress)
            self.deep_worker.skipped.connect(
                lambda reason: self.status_label.setText(f"Deep auto-train: {reason}")
            )
            self.deep_worker.finished.connect(
                lambda success, status, summary: self._on_deep_autotrain_finished(
                    success, status, summary
                )
            )
            self.deep_worker.start()
        except Exception:
            pass

    def _on_deep_autotrain_finished(self, success: bool, status: str, summary: dict) -> None:
        """Handle auto-train completion — the active model is left untouched."""
        if not success:
            if status != "skipped":
                self.status_label.setText("Deep auto-train failed")
            return
        records = (summary or {}).get("records", 0)
        saved_to = (summary or {}).get("saved_to", "")
        self.status_label.setText(
            f"✓ Deep auto-train complete: {records} records -> {saved_to}"
        )


    def _run_surrogate(self) -> None:
        """Run the surrogate solidification simulation in a background thread."""
        from core import surrogate

        if not self.stl_loader.is_valid():
            QMessageBox.warning(self, "Surrogate Simulation", "Please load a CAD file first.")
            return
        if self.surrogate_model is None:
            QMessageBox.warning(self, "Surrogate Simulation", "Please load a surrogate model first.")
            return

        self._set_simulation_running(True)
        self._stop_fraction_liquid_playback()
        self._purge_previous_simulation_memory()

        combined_mesh = self.part_manager.get_combined_mesh()
        if combined_mesh is not None:
            self.visualizer.load_mesh(combined_mesh.vertices, combined_mesh.faces)
        else:
            self.visualizer.load_mesh(self.stl_loader.get_vertices(), self.stl_loader.get_faces())

        voxel_size = self._current_voxel_size
        self.status_label.setText("Starting surrogate simulation...")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("Starting...")

        self.surrogate_worker = SurrogateWorker(
            self.stl_loader,
            self.surrogate_model,
            self._current_alloy,
            voxel_size,
            int(self._current_refinement),
            memory_mode=self._current_memory_mode,
            part_manager=self.part_manager,
            analysis_options=self._get_effective_analysis_options(),
            solidification_params=self._solidification_params,
        )
        self.surrogate_worker.progress.connect(self._on_simulation_progress)
        self.surrogate_worker.detailed_progress.connect(self._on_detailed_progress)
        self.surrogate_worker.finished.connect(self._on_surrogate_finished)
        self.simulation_start_time = time.time()
        self.surrogate_worker.start()

    def _cancel_surrogate(self) -> None:
        if self.surrogate_worker is not None and self.surrogate_worker.isRunning():
            self.surrogate_worker.cancel()
            self.status_label.setText("Cancelling surrogate simulation...")

    def _on_surrogate_finished(self, success: bool, status: str) -> None:
        """Handle surrogate simulation completion."""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100 if success else 0)
        self._set_simulation_running(False)

        if not success:
            self.current_results = None
            self.status_label.setText("Surrogate simulation failed")
            self.progress_bar.setFormat("Failed")
            if status != "cancelled":
                QMessageBox.critical(self, "Surrogate Simulation", "Surrogate simulation failed. Check console for details.")
            return

        # Reuse the traditional result pipeline by pointing simulation_worker
        # at the surrogate worker's outputs.
        self.simulation_worker = self.surrogate_worker
        simulation = self.simulation_worker.simulation
        self.current_results = simulation.export_results()
        self._setup_result_type_options()
        self._update_results_visualization()

        badge = f"Surrogate ({self.surrogate_model.manifest.backend}) · {self.surrogate_model.manifest.name}"
        confidence = getattr(simulation, "surrogate_confidence", 0.0)
        if confidence > 0:
            badge += f" · conf {confidence:.2f}"
        if getattr(simulation, "surrogate_disabled_analyses", None):
            badge += " · disabled: " + ", ".join(simulation.surrogate_disabled_analyses)
        self.status_label.setText(f"✓ {badge}")
        self.progress_bar.setFormat("Complete")

    def _setup_result_type_options(self):
        """Configure result type combo based on available analyses"""
        self.result_type_combo.setEnabled(True)
        
        # Build list of available result types
        available_types = []
        if self._analysis_options.get('shrinkage', False):
            available_types.append("Shrinkage/Porosity")
        if self.simulation_worker is not None and self.simulation_worker.simulation is not None:
            if getattr(self.simulation_worker.simulation, "local_modulus", None) is not None:
                available_types.append("Feed Modulus")
        if self._analysis_options.get('hotspot', False):
            available_types.append("Hot Spots")
        if self._analysis_options.get('niyama', False):
            available_types.append("Niyama Criterion")
        if self._analysis_options.get('hot_tear', False):
            available_types.append("Hot Tear Risk")
        if self._analysis_options.get('sand_burn_in', False):
            available_types.append("Sand Burn-In")
        if self._analysis_options.get('gas_porosity', False):
            available_types.append("Gas Generation")
            available_types.append("Gas Entrapment")
            available_types.append("Gas Porosity Risk")
        if self._analysis_options.get('fraction_liquid', False):
            available_types.append("Fraction Liquid")
        # Advanced analysis types
        if self._analysis_options.get('feeding_path', False):
            available_types.append("Feeding Quality")
        if self._analysis_options.get('stress_analysis', False):
            available_types.append("Thermal Stress")
            available_types.append("Distortion")
            available_types.append("Crack Susceptibility")
        if self._analysis_options.get('microstructure', False):
            available_types.append("SDAS (Microstructure)")
            available_types.append("Predicted UTS")
        
        # Update combo box
        self.result_type_combo.blockSignals(True)
        self.result_type_combo.clear()
        self.result_type_combo.addItems(available_types)
        self.result_type_combo.blockSignals(False)
        
        # Set time slider range for fraction liquid
        if self._analysis_options.get('fraction_liquid', False) and self.simulation_worker.simulation is not None:
            min_time, max_time = self.simulation_worker.simulation.get_fraction_liquid_time_range()
            self.time_slider.setRange(0, FRACTION_LIQUID_TIME_SLIDER_MAX)
            self.time_label.setText(f"{min_time:.1f}s")
        
        # Select first available type AND trigger the handler
        if available_types:
            self._current_vis_mode = {
                "Shrinkage/Porosity": "shrinkage",
                "Feed Modulus": "feed_modulus",
                "Hot Spots": "hotspot",
                "Niyama Criterion": "niyama",
                "Hot Tear Risk": "hot_tear",
                "Sand Burn-In": "sand_burn_in",
                "Gas Generation": "gas_generation",
                "Gas Entrapment": "gas_entrapment",
                "Gas Porosity Risk": "gas_porosity",
                "Fraction Liquid": "fraction_liquid",
                "Feeding Quality": "feeding_quality",
                "Thermal Stress": "thermal_stress",
                "Distortion": "distortion",
                "Crack Susceptibility": "crack_susceptibility",
                "SDAS (Microstructure)": "sdas",
                "Predicted UTS": "predicted_uts"
            }.get(available_types[0], "shrinkage")
            # Manually trigger to ensure UI is updated
            self._on_result_type_changed(available_types[0])
    
    def _update_results_visualization(self):
        """Update visualization based on current result type"""
        if self.simulation_worker is None or self.simulation_worker.simulation is None:
            return
        
        simulation = self.simulation_worker.simulation
        voxel_mesh = self.simulation_worker.voxel_mesh
        threshold = self.threshold_slider.value() / 100.0
        
        # Clear ALL previous result actors before loading new visualization
        # This ensures we don't have overlapping visualizations
        self.visualizer.clear_result_actors()
        
        if self._current_vis_mode == 'shrinkage':
            self._visualize_shrinkage(simulation, voxel_mesh, threshold)
        elif self._current_vis_mode == 'feed_modulus':
            self._visualize_feed_modulus(simulation, voxel_mesh, threshold)
        elif self._current_vis_mode == 'hotspot':
            self._visualize_hotspots(simulation, voxel_mesh, threshold)
        elif self._current_vis_mode == 'niyama':
            self._visualize_niyama(simulation, voxel_mesh, threshold)
        elif self._current_vis_mode == 'hot_tear':
            self._visualize_hot_tear(simulation, voxel_mesh, threshold)
        elif self._current_vis_mode == 'sand_burn_in':
            self._visualize_sand_burn_in(simulation, voxel_mesh, threshold)
        elif self._current_vis_mode in ('gas_generation', 'gas_entrapment', 'gas_porosity'):
            self._visualize_gas_porosity(simulation, voxel_mesh, threshold)
        elif self._current_vis_mode == 'fraction_liquid':
            min_time, max_time = simulation.get_fraction_liquid_time_range()
            time_value = self._map_fraction_liquid_slider_time(
                self.time_slider.value(),
                min_time,
                max_time,
            )
            self._update_fraction_liquid_visualization(
                time_value, slider_value=self.time_slider.value()
            )
        # Advanced analysis visualizations
        elif self._current_vis_mode == 'feeding_quality':
            self._visualize_feeding_quality(simulation, voxel_mesh, threshold)
        elif self._current_vis_mode == 'thermal_stress':
            self._visualize_thermal_stress(simulation, voxel_mesh, threshold)
        elif self._current_vis_mode == 'distortion':
            self._visualize_distortion(simulation, voxel_mesh, threshold)
        elif self._current_vis_mode == 'crack_susceptibility':
            self._visualize_crack_susceptibility(simulation, voxel_mesh, threshold)
        elif self._current_vis_mode == 'sdas':
            self._visualize_sdas(simulation, voxel_mesh, threshold)
        elif self._current_vis_mode == 'predicted_uts':
            self._visualize_predicted_uts(simulation, voxel_mesh, threshold)
    
    def _visualize_shrinkage(self, simulation, voxel_mesh, threshold):
        """Visualize shrinkage/porosity results"""
        if simulation.shrinkage_probability is None:
            print("No shrinkage data available")
            return

        if hasattr(simulation, "get_shrinkage_display_intensity"):
            intensities = np.asarray(simulation.get_shrinkage_display_intensity(), dtype=np.float32)
        else:
            intensities = np.array([
                simulation.get_shrinkage_intensity(i)
                for i in range(len(voxel_mesh.voxel_centers))
            ], dtype=np.float32)

        if len(intensities) != len(voxel_mesh.voxel_centers):
            print("No shrinkage data available")
            return
        effective_threshold, lowered_threshold = self._get_effective_shrinkage_threshold(
            intensities,
            threshold,
        )
        if lowered_threshold:
            threshold_percent = max(1, min(100, int(round(effective_threshold * 100.0))))
            self.threshold_slider.blockSignals(True)
            self.threshold_slider.setValue(threshold_percent)
            self.threshold_slider.setMarkerPosition(threshold_percent)
            self.threshold_slider.blockSignals(False)
            self.threshold_label.setText(f"{threshold_percent}%")
            print(
                "DEBUG: Lowered shrinkage display threshold "
                f"from {threshold:.3f} to {effective_threshold:.3f} "
                f"(max risk={float(np.max(intensities)):.3f})"
            )

        if not np.any(intensities >= effective_threshold):
            print("No shrinkage voxels to display")
            return

        self.visualizer.load_shrinkage(
            voxel_mesh.voxel_centers,
            voxel_mesh.voxel_sizes,
            intensities,
            effective_threshold
        )

        # Update legend
        self.visualizer.update_legend("Shrinkage/Porosity Risk", 0.0, 1.0)

        # Store for threshold updates
        self.visualizer.current_shrinkage_data = (
            voxel_mesh.voxel_centers,
            voxel_mesh.voxel_sizes,
            intensities
        )

        # Register shrink/porosity inspector data so clicks reveal the
        # connected defect pocket volume in cubic inches.
        shrinkage_risk = getattr(simulation, "shrinkage_probability", None)
        porosity_risk = getattr(simulation, "porosity_probability", None)
        self.visualizer.set_shrink_inspector_data(
            display_intensity=intensities,
            mesh=voxel_mesh,
            shrinkage_risk=shrinkage_risk,
            porosity_risk=porosity_risk,
            threshold=effective_threshold,
        )

    def _visualize_feed_modulus(self, simulation, voxel_mesh, threshold):
        """Visualize local feed modulus in inches."""
        local_modulus = getattr(simulation, "local_modulus", None)
        if local_modulus is None:
            local_modulus = getattr(voxel_mesh, "voxel_moduli", None)
        if local_modulus is None:
            print("No feed modulus data available")
            return

        feed_modulus_in = np.asarray(local_modulus, dtype=np.float32)
        if len(feed_modulus_in) != len(voxel_mesh.voxel_centers):
            print("No feed modulus data available")
            return

        # Feed modulus is only meaningful for cast metal; exclude chills.
        part_types = getattr(voxel_mesh, "voxel_part_types", None)
        if part_types is not None and len(part_types) == len(feed_modulus_in):
            chill_mask = np.asarray(part_types, dtype=np.int32) == 2
            if np.any(chill_mask):
                feed_modulus_in = feed_modulus_in.copy()
                feed_modulus_in[chill_mask] = 0.0

        finite = feed_modulus_in[np.isfinite(feed_modulus_in) & (feed_modulus_in > 0.0)]
        if finite.size == 0:
            print("No feed modulus data available")
            return

        min_in = float(np.min(finite))
        max_in = float(np.max(finite))
        threshold_in = min_in + (float(threshold) * max(max_in - min_in, 0.0))
        self.visualizer.update_legend("Feed Modulus", min_in, max_in, "in")
        self.visualizer.load_shrinkage(
            voxel_mesh.voxel_centers,
            voxel_mesh.voxel_sizes,
            np.where(np.isfinite(feed_modulus_in), feed_modulus_in, 0.0).astype(np.float32),
            threshold_in,
        )
        self.visualizer.current_shrinkage_data = (
            voxel_mesh.voxel_centers,
            voxel_mesh.voxel_sizes,
            np.where(np.isfinite(feed_modulus_in), feed_modulus_in, 0.0).astype(np.float32),
        )
    
    def _visualize_hotspots(self, simulation, voxel_mesh, threshold):
        """Visualize hot spot results"""
        if simulation.hotspot_intensity is None:
            print("No hot spot data available")
            return
        
        self.visualizer.load_hotspots(
            voxel_mesh.voxel_centers,
            voxel_mesh.voxel_sizes,
            simulation.hotspot_intensity,
            threshold
        )
        
        # Update legend
        self.visualizer.update_legend("Hot Spot Intensity", 0.0, 1.0)
        
        # Store for threshold updates
        self.visualizer.current_hotspot_data = (
            voxel_mesh.voxel_centers,
            voxel_mesh.voxel_sizes,
            simulation.hotspot_intensity
        )
    
    def _visualize_niyama(self, simulation, voxel_mesh, threshold):
        """Visualize Niyama criterion results with isolation-enhanced defect detection"""
        if simulation.niyama_values is None:
            print("No Niyama data available")
            return
        
        # For Niyama, lower values indicate higher risk
        # Invert for visualization (so red = low Niyama = high risk)
        alloy_key = simulation.alloy.name.split()[0]
        from core.solidification import NIYAMA_THRESHOLDS
        niyama_threshold = NIYAMA_THRESHOLDS.get(alloy_key, 1.0)
        
        # Use Niyama-only risk for the Niyama view. Feeding/isolation blending is
        # exposed separately as combined feed risk in export/web payloads.
        if hasattr(simulation, 'niyama_risk') and simulation.niyama_risk is not None:
            risk_intensity = simulation.niyama_risk.astype(np.float32)
            print("  Using Niyama-only risk visualization")
        else:
            # Fallback: Convert Niyama values to risk intensity (0-1)
            max_niyama = max(np.max(simulation.niyama_values), niyama_threshold * 2)
            risk_intensity = 1.0 - np.clip(simulation.niyama_values / max_niyama, 0, 1)

        # Niyama defects are only meaningful in casting metal.
        # Exclude risers/riser-post/chill voxels from visualization.
        part_types = getattr(voxel_mesh, "voxel_part_types", None)
        if part_types is not None and len(part_types) == len(risk_intensity):
            cast_mask = np.asarray(part_types, dtype=np.int32) == 0  # 0 = CAST
            excluded = int(np.size(cast_mask) - np.sum(cast_mask))
            if excluded > 0:
                risk_intensity = np.where(cast_mask, risk_intensity, 0.0).astype(np.float32, copy=False)
                print(f"  Excluded {excluded:,} non-cast voxels (riser/chill) from Niyama display")
        
        # Use risk_intensity for BOTH parameters - this allows proper threshold control
        # The old approach used boolean niyama_defect_risk which only showed True voxels
        # regardless of slider position
        
        # Debug output: show risk intensity stats
        print(f"  Niyama risk intensity range: {np.min(risk_intensity):.3f} - {np.max(risk_intensity):.3f}")
        print(f"  Voxels above threshold ({threshold:.0%}): {np.sum(risk_intensity > threshold):,}")
        
        self.visualizer.load_niyama(
            voxel_mesh.voxel_centers,
            voxel_mesh.voxel_sizes,
            risk_intensity,
            risk_intensity,  # Use float 0-1 for threshold filtering, not boolean
            threshold
        )

        
        # Update legend
        self.visualizer.update_legend("Defect Risk (Niyama)", 0.0, 1.0)
        
        # Store for threshold updates - use risk_intensity for both
        self.visualizer.current_niyama_data = (
            voxel_mesh.voxel_centers,
            voxel_mesh.voxel_sizes,
            risk_intensity,
            risk_intensity  # Use float 0-1 for threshold filtering
        )
    
    def _visualize_hot_tear(self, simulation, voxel_mesh, threshold):
        """Visualize hot tear risk zones"""
        if simulation.hot_tear_risk is None:
            print("No hot tear data available")
            return
        
        # Hot tear risk is already 0-1 scale
        # Use cyan/purple color scheme to distinguish from shrinkage (red/yellow)
        risk_intensity = simulation.hot_tear_risk.astype(np.float32)
        
        print(f"  Hot tear risk range: {np.min(risk_intensity):.3f} - {np.max(risk_intensity):.3f}")
        print(f"  Hot tear zone voxels: {np.sum(simulation.hot_tear_zones):,}")
        
        # Use the hotspot visualization method but with different color mapping
        # We'll treat hot tear as a special hotspot type
        self.visualizer.load_hot_tear(
            voxel_mesh.voxel_centers,
            voxel_mesh.voxel_sizes,
            risk_intensity,
            simulation.hot_tear_zones,
            threshold
        )
        # Update legend
        self.visualizer.update_legend("Hot Tear Risk", 0.0, 1.0)
        
        # Store for threshold updates
        self.visualizer.current_hot_tear_data = (
            voxel_mesh.voxel_centers,
            voxel_mesh.voxel_sizes,
            risk_intensity,
            simulation.hot_tear_zones
        )
    
    def _visualize_sand_burn_in(self, simulation, voxel_mesh, threshold):
        """Visualize sand burn-in risk zones (surface defects)"""
        if simulation.sand_burn_in_risk is None:
            print("No sand burn-in data available")
            return
        
        # Sand burn-in risk is already 0-1 scale
        # Uses tan/orange/brown color scheme to evoke burned sand appearance
        risk_intensity = simulation.sand_burn_in_risk.astype(np.float32)
        
        print(f"  Sand burn-in risk range: {np.min(risk_intensity):.3f} - {np.max(risk_intensity):.3f}")
        print(f"  Sand burn-in zone voxels: {np.sum(simulation.sand_burn_in_zones):,}")
        
        self.visualizer.load_sand_burn_in(
            voxel_mesh.voxel_centers,
            voxel_mesh.voxel_sizes,
            risk_intensity,
            simulation.sand_burn_in_zones,
            threshold
        )
        # Update legend
        self.visualizer.update_legend("Sand Burn-In Risk", 0.0, 1.0)
        
        # Store for threshold updates
        self.visualizer.current_sand_burn_in_data = (
            voxel_mesh.voxel_centers,
            voxel_mesh.voxel_sizes,
            risk_intensity,
            simulation.sand_burn_in_zones
        )
    
    def _visualize_gas_porosity(self, simulation, voxel_mesh, threshold):
        """Visualize gas porosity results (three sub-modes)."""
        if simulation.gas_porosity_risk is None:
            print("No gas porosity data available")
            return

        mode = self._current_vis_mode

        if mode == "gas_generation":
            if simulation.gas_generation is None:
                print("No gas generation data available")
                return
            intensities = simulation.gas_generation.astype(np.float32)
            legend_title = "Gas Generation (SGI)"
        elif mode == "gas_entrapment":
            if simulation.gas_entrapment is None:
                print("No gas entrapment data available")
                return
            intensities = simulation.gas_entrapment.astype(np.float32)
            legend_title = "Gas Entrapment (E)"
        else:
            intensities = simulation.gas_porosity_risk.astype(np.float32)
            legend_title = "Gas Porosity Risk (R_gas)"

        print(f"  Gas porosity ({mode}) range: {np.min(intensities):.4f} - {np.max(intensities):.4f}")
        if simulation.gas_porosity_hotspots is not None:
            print(f"  Gas porosity hotspots: {np.sum(simulation.gas_porosity_hotspots):,}")

        self.visualizer.load_gas_porosity(
            voxel_mesh.voxel_centers,
            voxel_mesh.voxel_sizes,
            intensities,
            threshold,
            mode=mode,
        )
        self.visualizer.update_legend(legend_title, 0.0, 1.0)

        self.visualizer.current_gas_porosity_data = (
            voxel_mesh.voxel_centers,
            voxel_mesh.voxel_sizes,
            intensities,
            mode,
        )
    
    def _visualize_feeding_quality(self, simulation, voxel_mesh, threshold):
        """Visualize feeding path quality results"""
        if simulation.feeding_quality is None:
            print("No feeding quality data available - was 'Feeding Path Analysis' enabled?")
            # Show a message in the status bar
            self.status_label.setText("⚠️ No feeding quality data - enable 'Feeding Path Analysis' and re-run")
            return
        
        # Feeding quality: 1 = good (green), 0 = blocked (red)
        # Invert for consistent "red = bad" visualization
        risk_intensity = 1.0 - simulation.feeding_quality.astype(np.float32)
        
        print(f"  Feeding quality range: {np.min(simulation.feeding_quality):.3f} - {np.max(simulation.feeding_quality):.3f}")
        print(f"  Blocked voxels: {simulation.feeding_blocked_count:,}")
        
        # Update legend
        self.visualizer.update_legend("Feeding Quality", 0.0, 1.0, "0=Blocked, 1=Good")
        
        # Use hotspot visualization with inverted data
        self.visualizer.load_hotspots(
            voxel_mesh.voxel_centers,
            voxel_mesh.voxel_sizes,
            risk_intensity,
            threshold
        )
        
        self.visualizer.current_hotspot_data = (
            voxel_mesh.voxel_centers,
            voxel_mesh.voxel_sizes,
            risk_intensity
        )
    
    def _visualize_thermal_stress(self, simulation, voxel_mesh, threshold):
        """Visualize thermal stress results"""
        if simulation.effective_stress is None:
            print("No thermal stress data available - was 'Thermal Stress & Distortion' enabled?")
            self.status_label.setText("⚠️ No stress data - enable 'Thermal Stress & Distortion' and re-run")
            return
        
        # Normalize stress to 0-1 range based on max stress
        max_stress = np.max(simulation.effective_stress)
        if max_stress > 0:
            normalized_stress = simulation.effective_stress / max_stress
        else:
            normalized_stress = np.zeros_like(simulation.effective_stress)
        
        print(f"  Stress range: 0 - {max_stress:.1f} MPa")
        print(f"  Mean stress: {np.mean(simulation.effective_stress):.1f} MPa")
        
        # Update legend
        self.visualizer.update_legend("Thermal Stress", 0.0, max_stress, "MPa")
        
        self.visualizer.load_hotspots(
            voxel_mesh.voxel_centers,
            voxel_mesh.voxel_sizes,
            normalized_stress.astype(np.float32),
            threshold
        )
        
        self.visualizer.current_hotspot_data = (
            voxel_mesh.voxel_centers,
            voxel_mesh.voxel_sizes,
            normalized_stress
        )
    
    def _visualize_distortion(self, simulation, voxel_mesh, threshold):
        """Visualize distortion magnitude results"""
        if simulation.distortion_magnitude is None:
            print("No distortion data available - was 'Thermal Stress & Distortion' enabled?")
            self.status_label.setText("⚠️ No distortion data - enable 'Thermal Stress & Distortion' and re-run")
            return
        
        # Normalize distortion to 0-1 range
        max_dist = np.max(simulation.distortion_magnitude)
        if max_dist > 0:
            normalized_dist = simulation.distortion_magnitude / max_dist
        else:
            normalized_dist = np.zeros_like(simulation.distortion_magnitude)
        
        print(f"  Distortion range: 0 - {max_dist*1000:.3f} mils")
        
        # Update legend
        self.visualizer.update_legend("Distortion", 0.0, max_dist*1000, "mils")
        
        self.visualizer.load_hotspots(
            voxel_mesh.voxel_centers,
            voxel_mesh.voxel_sizes,
            normalized_dist.astype(np.float32),
            threshold
        )
        
        self.visualizer.current_hotspot_data = (
            voxel_mesh.voxel_centers,
            voxel_mesh.voxel_sizes,
            normalized_dist
        )
    
    def _visualize_crack_susceptibility(self, simulation, voxel_mesh, threshold):
        """Visualize crack susceptibility results"""
        if simulation.crack_susceptibility is None:
            print("No crack susceptibility data available - was 'Thermal Stress & Distortion' enabled?")
            self.status_label.setText("⚠️ No crack data - enable 'Thermal Stress & Distortion' and re-run")
            return
        
        # Already 0-1 range
        risk_intensity = simulation.crack_susceptibility.astype(np.float32)
        
        print(f"  Crack susceptibility range: {np.min(risk_intensity):.3f} - {np.max(risk_intensity):.3f}")
        
        # Update legend
        self.visualizer.update_legend("Crack Risk", 0.0, 1.0, "Probability")
        
        self.visualizer.load_hotspots(
            voxel_mesh.voxel_centers,
            voxel_mesh.voxel_sizes,
            risk_intensity,
            threshold
        )
        
        self.visualizer.current_hotspot_data = (
            voxel_mesh.voxel_centers,
            voxel_mesh.voxel_sizes,
            risk_intensity
        )
    
    def _visualize_sdas(self, simulation, voxel_mesh, threshold):
        """Visualize SDAS (microstructure) results"""
        if simulation.sdas is None:
            print("No SDAS data available - was 'Microstructure (SDAS) Prediction' enabled?")
            self.status_label.setText("⚠️ No SDAS data - enable 'Microstructure (SDAS) Prediction' and re-run")
            return
        
        # Normalize SDAS: higher values = coarser = worse = more red
        # Lower SDAS = finer = better = less red
        min_sdas = np.min(simulation.sdas)
        max_sdas = np.max(simulation.sdas)
        
        if max_sdas > min_sdas:
            # Normalize so that MAX SDAS = 1 (red/bad), MIN SDAS = 0 (green/good)
            normalized_sdas = (simulation.sdas - min_sdas) / (max_sdas - min_sdas)
        else:
            normalized_sdas = np.zeros_like(simulation.sdas)
        
        print(f"  SDAS range: {min_sdas:.0f} - {max_sdas:.0f} μm")
        print(f"  Average SDAS: {np.mean(simulation.sdas):.0f} μm")
        
        # Update legend
        self.visualizer.update_legend("SDAS", min_sdas, max_sdas, "μm")
        
        self.visualizer.load_hotspots(
            voxel_mesh.voxel_centers,
            voxel_mesh.voxel_sizes,
            normalized_sdas.astype(np.float32),
            threshold
        )
        
        self.visualizer.current_hotspot_data = (
            voxel_mesh.voxel_centers,
            voxel_mesh.voxel_sizes,
            normalized_sdas
        )
    
    def _visualize_predicted_uts(self, simulation, voxel_mesh, threshold):
        """Visualize predicted UTS (tensile strength) results"""
        if simulation.predicted_uts is None:
            print("No predicted UTS data available - was 'Microstructure (SDAS) Prediction' enabled?")
            self.status_label.setText("⚠️ No UTS data - enable 'Microstructure (SDAS) Prediction' and re-run")
            return
        
        # Normalize UTS: higher is better, so invert for visualization
        # Low UTS = red (bad), High UTS = green (good)
        min_uts = np.min(simulation.predicted_uts)
        max_uts = np.max(simulation.predicted_uts)
        
        if max_uts > min_uts:
            # Invert: low UTS gets high value (red)
            normalized_uts = 1.0 - (simulation.predicted_uts - min_uts) / (max_uts - min_uts)
        else:
            normalized_uts = np.zeros_like(simulation.predicted_uts)
        
        print(f"  UTS range: {min_uts:.0f} - {max_uts:.0f} MPa")
        print(f"  Average UTS: {np.mean(simulation.predicted_uts):.0f} MPa")
        
        # Update legend (Note: color scale is inverted, red=low UTS)
        self.visualizer.update_legend("Predicted UTS", max_uts, min_uts, "MPa")
        
        self.visualizer.load_hotspots(
            voxel_mesh.voxel_centers,
            voxel_mesh.voxel_sizes,
            normalized_uts.astype(np.float32),
            threshold
        )
        
        self.visualizer.current_hotspot_data = (
            voxel_mesh.voxel_centers,
            voxel_mesh.voxel_sizes,
            normalized_uts
        )
    
    def _update_threshold(self, value):
        """Update visualization threshold for current analysis type"""
        threshold = value / 100.0
        self.threshold_label.setText(f"{value}%")

        # Update based on current visualization mode
        mode = getattr(self, '_current_vis_mode', 'shrinkage')

        if mode == 'shrinkage':
            if hasattr(self.visualizer, 'current_shrinkage_data') and \
               self.visualizer.current_shrinkage_data is not None:
                self.visualizer.update_shrinkage_threshold(threshold)
                # The shrink/porosity inspector stores its own threshold for
                # pocket extraction. The visualizer clears the leader on
                # threshold change so the next click uses the new range.
                shrink_data = getattr(
                    self.visualizer, "_shrink_inspector_data", None
                )
                if shrink_data is not None:
                    shrink_data["threshold"] = float(threshold)
        elif mode == 'feed_modulus':
            if hasattr(self.visualizer, 'current_shrinkage_data') and \
               self.visualizer.current_shrinkage_data is not None:
                _, _, feed_modulus_in = self.visualizer.current_shrinkage_data
                finite = np.asarray(feed_modulus_in, dtype=np.float32)
                finite = finite[np.isfinite(finite)]
                if finite.size > 0:
                    min_in = float(np.min(finite))
                    max_in = float(np.max(finite))
                    threshold_in = min_in + (threshold * max(max_in - min_in, 0.0))
                    self.visualizer.update_shrinkage_threshold(threshold_in)
        elif mode == 'hotspot':
            if hasattr(self.visualizer, 'current_hotspot_data') and \
               self.visualizer.current_hotspot_data is not None:
                self.visualizer.update_hotspot_threshold(threshold)
        elif mode == 'niyama':
            if hasattr(self.visualizer, 'current_niyama_data') and \
               self.visualizer.current_niyama_data is not None:
                self.visualizer.update_niyama_threshold(threshold)
        elif mode == 'hot_tear':
            if hasattr(self.visualizer, 'current_hot_tear_data') and \
               self.visualizer.current_hot_tear_data is not None:
                self.visualizer.update_hot_tear_threshold(threshold)
        elif mode == 'sand_burn_in':
            if hasattr(self.visualizer, 'current_sand_burn_in_data') and \
               self.visualizer.current_sand_burn_in_data is not None:
                self.visualizer.update_sand_burn_in_threshold(threshold)
        elif mode in ('gas_generation', 'gas_entrapment', 'gas_porosity'):
            if hasattr(self.visualizer, 'current_gas_porosity_data') and \
               self.visualizer.current_gas_porosity_data is not None:
                self.visualizer.update_gas_porosity_threshold(threshold)
        elif mode in ('feeding_quality', 'thermal_stress', 'distortion',
                      'crack_susceptibility', 'sdas', 'predicted_uts'):
            if hasattr(self.visualizer, 'current_hotspot_data') and \
               self.visualizer.current_hotspot_data is not None:
                self.visualizer.update_hotspot_threshold(threshold)
        elif mode == 'fraction_liquid':
            # For fraction liquid, threshold controls minimum liquid fraction shown
            if self.simulation_worker is not None and self.simulation_worker.simulation is not None:
                min_time, max_time = self.simulation_worker.simulation.get_fraction_liquid_time_range()
                time_value = self._map_fraction_liquid_slider_time(
                    self.time_slider.value(),
                    min_time,
                    max_time,
                )
                self._update_fraction_liquid_visualization(
                    time_value, slider_value=self.time_slider.value()
                )
    
    def _toggle_mesh_visibility(self, state):
        """Toggle mesh visibility"""
        self.visualizer.toggle_mesh_visibility(state == Qt.Checked)
    
    def _toggle_shrinkage_visibility(self, state):
        """Toggle shrinkage visibility"""
        self.visualizer.toggle_shrinkage_visibility(state == Qt.Checked)
    
    # ==================== NEW VISUALIZATION FEATURE HANDLERS ====================
    
    def _open_visualization_dialog(self):
        """Open the visualization settings dialog"""
        # Update cross-section ranges if we have model bounds
        if self.visualizer.model_bounds:
            bounds = self.visualizer.model_bounds
            self.visualization_dialog.set_bounds_range('X', bounds[0], bounds[1])
            self.visualization_dialog.set_bounds_range('Y', bounds[2], bounds[3])
            self.visualization_dialog.set_bounds_range('Z', bounds[4], bounds[5])
        
        self.visualization_dialog.center_on_parent()
        self.visualization_dialog.show()
        self.visualization_dialog.raise_()
    
    def _open_measurement_dialog(self):
        """Open the measurement tools dialog"""
        self.measurement_dialog.center_on_parent()
        self.measurement_dialog.show()
        self.measurement_dialog.raise_()
    
    def _open_animation_dialog(self):
        """Open the animation export dialog"""
        self.animation_dialog.center_on_parent()
        self.animation_dialog.show()
        self.animation_dialog.raise_()
    
    def _take_screenshot(self):
        """Take a screenshot of the current view"""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Screenshot",
            f"casting_screenshot_{int(time.time())}.png",
            "PNG Files (*.png);;All Files (*)"
        )
        if filepath:
            if not filepath.endswith('.png'):
                filepath += '.png'
            self.visualizer.export_screenshot(filepath)
            self.status_label.setText(f"Screenshot saved: {os.path.basename(filepath)}")
    
    def _on_colormap_changed(self, colormap: str):
        """Handle colormap change from visualization dialog"""
        self.visualizer.set_colormap(colormap)
        
        # Update legend title based on current result type
        if self._current_vis_mode == 'shrinkage':
            self.visualizer.show_color_bar("Shrinkage/Porosity Risk",
                                           self.visualization_dialog.show_legend_check.isChecked())
        elif self._current_vis_mode == 'hotspot':
            self.visualizer.show_color_bar("Hot Spot Intensity",
                                           self.visualization_dialog.show_legend_check.isChecked())
        elif self._current_vis_mode == 'niyama':
            self.visualizer.show_color_bar("Niyama Defect Risk",
                                           self.visualization_dialog.show_legend_check.isChecked())
        elif self._current_vis_mode == 'hot_tear':
            self.visualizer.show_color_bar("Hot Tear Risk",
                                           self.visualization_dialog.show_legend_check.isChecked())
        
        # Re-render current visualization
        if self.current_results is not None:
            self._update_results_visualization()
    
    def _on_color_scale_changed(self, min_val: float, max_val: float):
        """Handle color scale bounds change"""
        self.visualizer.set_color_scale_bounds(min_val, max_val)
        
        # Re-render current visualization
        if self.current_results is not None:
            self._update_results_visualization()
    
    def _on_cross_section_changed(self, axis: str, enabled: bool, position: float, direction: int = 1):
        """Handle cross-section plane change"""
        if enabled:
            # Convert normalized position (0-1) to actual position
            min_pos, max_pos = self.visualizer.get_cross_section_range(axis)
            actual_pos = min_pos + position * (max_pos - min_pos)
            self.visualizer.enable_cross_section(axis, actual_pos, direction=direction)
        else:
            self.visualizer.disable_cross_section(axis)

    def _on_legend_toggle(self, show: bool):
        """Handle legend show/hide toggle"""
        title = "Intensity"
        if self._current_vis_mode == 'shrinkage':
            title = "Shrinkage/Porosity Risk"
        elif self._current_vis_mode == 'hotspot':
            title = "Hot Spot Intensity"
        elif self._current_vis_mode == 'niyama':
            title = "Niyama Defect Risk"
        elif self._current_vis_mode == 'hot_tear':
            title = "Hot Tear Risk"
        elif self._current_vis_mode == 'fraction_liquid':
            title = "Fraction Liquid"
        
        self.visualizer.show_color_bar(title, show)
        
        # Set additional legend info
        if show:
            sim_time = 0.0
            if self.simulation_worker and self.simulation_worker.simulation:
                sim_time = self.simulation_worker.simulation.time
            self.visualizer.set_legend_info(self._current_alloy, sim_time)
    
    def _start_distance_measurement(self):
        """Start distance measurement mode"""
        self.visualizer.enable_measurement_mode('distance')
        self.status_label.setText("📏 Distance measurement: Click two points on the model")
    
    def _stop_measurement_mode(self):
        """Stop measurement mode when dialog is closed"""
        self.visualizer.disable_measurement_mode()
        self.status_label.setText("Ready")
    
    def _clear_measurements(self):
        """Clear all measurements"""
        self.visualizer.clear_measurements()
        self.status_label.setText("Measurements cleared")
    
    def _calculate_defect_volume(self, threshold: float):
        """Calculate defect volume above threshold"""
        volume = self.visualizer.calculate_defect_volume(threshold)
        if volume is not None:
            self.measurement_dialog.set_volume_result(volume)
            print(f"Defect volume (threshold={threshold:.2f}): {volume:.4f} in³")
        else:
            self.measurement_dialog.volume_result_label.setText("No defect data available")
    
    def _start_animation_recording(self):
        """Start recording animation frames"""
        self.visualizer.start_animation_recording()
        self._animation_frame_count = 0
        self.status_label.setText("Recording animation frames...")
    
    def _stop_animation_recording(self):
        """Stop recording animation frames"""
        count = self.visualizer.stop_animation_recording()
        self.animation_dialog.update_frame_count(count)
        self.status_label.setText(f"Recording stopped: {count} frames")
    
    def _capture_animation_frame(self):
        """Capture a single animation frame"""
        self.visualizer.capture_frame()
        self._animation_frame_count = len(self.visualizer._animation_frames)
        self.animation_dialog.update_frame_count(self._animation_frame_count)
    
    def _export_animation_gif(self, filepath: str, fps: int):
        """Export animation as GIF"""
        success = self.visualizer.export_animation_gif(filepath, fps)
        if success:
            self.status_label.setText(f"Animation exported: {os.path.basename(filepath)}")
            QMessageBox.information(self, "Export Complete", 
                                    f"Animation exported to:\n{filepath}")
        else:
            QMessageBox.warning(self, "Export Failed", 
                               "Failed to export GIF. Check console for details.")
    
    def _export_animation_mp4(self, filepath: str, fps: int):
        """Export animation as MP4"""
        success = self.visualizer.export_animation_mp4(filepath, fps)
        if success:
            self.status_label.setText(f"Animation exported: {os.path.basename(filepath)}")
            QMessageBox.information(self, "Export Complete", 
                                    f"Animation exported to:\n{filepath}")
        else:
            QMessageBox.warning(self, "Export Failed", 
                               "Failed to export MP4. Make sure OpenCV is installed.")
    
    # ==================== RISER DESIGN METHODS ====================
    
    def _open_riser_design_dialog(self):
        """Open the Auto Riser Design dialog"""
        # Check if we have parts loaded
        if not self.part_manager.get_all_parts():
            QMessageBox.warning(self, "No Parts Loaded", 
                               "Please load a CAD part before opening riser design.")
            return
        
        # Just show the dialog - mesh will be generated when user clicks Analyze
        self.riser_dialog.show()
        self.riser_dialog.raise_()
        self.riser_dialog.activateWindow()
    
    def _run_auto_riser(self):
        """Run automatic riser design analysis"""
        # Check if we have a mesh to analyze
        if not self.part_manager.get_all_parts():
            QMessageBox.warning(self, "No Parts Loaded", 
                               "Please load a CAD part before running riser design analysis.")
            return
        
        try:
            from core.riser_design import (
                RiserDesigner, RiserDesignParameters, MaterialRiserProperties,
                RiserRecommendation, RiserType, RiserShape
            )
            from core.voxel_mesh import AdaptiveVoxelMesh
            from core.materials import get_alloy
            from core.part_manager import PartType
            
            # Get the casting part (first casting-type part)
            casting_part = None
            for part in self.part_manager.get_all_parts():
                if part.part_type == PartType.CAST:
                    casting_part = part
                    break
            
            if casting_part is None:
                # Use first part if no casting specifically marked
                casting_part = self.part_manager.get_all_parts()[0]
            
            # Get mesh data from the part
            stl_mesh = casting_part.get_transformed_mesh()
            if stl_mesh is None:
                QMessageBox.warning(self, "Mesh Not Available", 
                                   "The part mesh is not available for analysis.")
                return
            
            self.status_label.setText("🔄 Creating voxel mesh for analysis...")
            QApplication.processEvents()
            
            # Create voxel mesh with current settings
            voxel_size = self._current_voxel_size
            bounds = np.array([stl_mesh.bounds[0], stl_mesh.bounds[1]]).T
            
            voxel_mesh = AdaptiveVoxelMesh(base_voxel_size=voxel_size, max_refinement_level=3)
            success = voxel_mesh.generate(stl_mesh, bounds)
            
            if not success or voxel_mesh.voxel_centers is None:
                QMessageBox.warning(self, "Mesh Error", 
                                   "Failed to create voxel mesh. Try a different voxel size.")
                return
            
            self.status_label.setText("🔄 Running riser design analysis...")
            QApplication.processEvents()
            
            # Get alloy properties - they already have the right attributes
            try:
                alloy_props = get_alloy(self._current_alloy)
            except KeyError:
                alloy_props = get_alloy("WCB")  # Default fallback
            
            # Configure riser design parameters
            params = RiserDesignParameters(
                safety_factor=1.2,
                modulus_factor=1.2,  # Riser modulus should be 1.2x casting modulus
                height_ratio=1.5,    # H/D ratio for cylindrical risers
                min_diameter=voxel_size * 3,
                max_diameter=voxel_size * 30
            )
            
            # Get voxel data for the designer
            voxel_positions = voxel_mesh.voxel_centers
            n_voxels = len(voxel_positions)
            surface_mask = voxel_mesh.surface_voxels if hasattr(voxel_mesh, 'surface_voxels') and voxel_mesh.surface_voxels is not None else np.ones(n_voxels, dtype=bool)
            neighbors = voxel_mesh.voxel_neighbors if hasattr(voxel_mesh, 'voxel_neighbors') and voxel_mesh.voxel_neighbors is not None else np.full((n_voxels, 6), -1, dtype=np.int32)
            
            # Create designer and run analysis
            designer = RiserDesigner(
                voxel_positions=voxel_positions,
                voxel_size=voxel_size,
                neighbors=neighbors,
                surface_mask=surface_mask,
                alloy=alloy_props,
                params=params
            )
            
            recommendations = designer.design_risers(max_risers=5)
            
            # Store results
            self._riser_recommendations = recommendations
            
            self.status_label.setText(f"✅ Riser analysis complete: {len(recommendations)} risers recommended")
            print(f"\n========== RISER DESIGN ANALYSIS ==========")
            print(f"Part: {casting_part.name}")
            print(f"Alloy: {self._current_alloy}")
            print(f"Voxel size: {voxel_size} in")
            print(f"Voxel count: {n_voxels}")
            print(f"Recommendations: {len(recommendations)} risers")
            for i, rec in enumerate(recommendations):
                print(f"\n  Riser {i+1}:")
                print(f"    Type: {rec.riser_type.name}")
                print(f"    Position: ({rec.position[0]:.2f}, {rec.position[1]:.2f}, {rec.position[2]:.2f})")
                print(f"    Size: {rec.diameter:.2f} x {rec.height:.2f}")
                print(f"    Confidence: {rec.confidence:.0%}")
            print("==========================================\n")
            
        except ImportError as e:
            QMessageBox.critical(self, "Module Error", 
                                f"Riser design module not found:\n{str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", 
                                f"Error during riser analysis:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def _apply_riser_recommendations(self, recommendations: list):
        """Add recommended risers as parts to the simulation"""
        if not recommendations:
            QMessageBox.information(self, "No Recommendations", 
                                   "No riser recommendations to apply.")
            return
        
        try:
            import trimesh
            import tempfile
            from core.part_manager import PartType
            
            added_count = 0
            for i, rec in enumerate(recommendations):
                # Create a cylinder mesh for the riser
                riser_mesh = trimesh.creation.cylinder(
                    radius=rec.diameter / 2.0,
                    height=rec.height,
                    sections=32
                )
                
                # Position the riser with substantial overlap into the casting
                # The riser needs to penetrate deeply into the casting to maintain a 
                # liquid feeding path during solidification. The connection area must
                # have enough thermal mass to stay liquid longer than surrounding metal.
                # 
                # For a plate casting, the riser should penetrate at least 30-50% of
                # the plate thickness to ensure proper feeding connectivity.
                # Using 40% of riser diameter as overlap (substantial penetration)
                overlap = rec.diameter * 0.40  # 40% of diameter penetration
                
                # Cylinder is created centered at origin, so translate to position
                # Base goes at (position.z - overlap) so riser penetrates into casting
                riser_mesh.apply_translation([
                    rec.position[0],
                    rec.position[1],
                    rec.position[2] - overlap + rec.height / 2.0
                ])
                
                # Create a unique name (riser_type is an enum, use .name)
                riser_type_name = rec.riser_type.name if hasattr(rec.riser_type, 'name') else str(rec.riser_type)
                riser_name = f"AutoRiser_{i+1}_{riser_type_name}"
                
                # Save mesh to temp file then load it via PartManager
                with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    riser_mesh.export(tmp_path)
                
                # Load the part using PartManager
                part_id = self.part_manager.load_part(
                    filepath=tmp_path,
                    name=riser_name,
                    preserve_origin=True
                )
                
                # Set part type to RISER
                if part_id:
                    part = self.part_manager.get_part(part_id)
                    if part:
                        part.set_part_type(PartType.RISER)
                    added_count += 1
                    print(f"Added {riser_name} at position {rec.position}")
                
                # Clean up temp file
                try:
                    os.remove(tmp_path)
                except:
                    pass
            
            if added_count > 0:
                # Update visualization
                self._update_parts_summary()
                self._update_visualization()
                
                self.status_label.setText(f"✅ Added {added_count} risers to simulation")
                QMessageBox.information(self, "Risers Applied", 
                                       f"Successfully added {added_count} riser(s) to the simulation.\n\n"
                                       "You can view and adjust them in the Parts Manager.")
            
        except ImportError as e:
            QMessageBox.critical(self, "Module Error", 
                                f"trimesh module not found:\n{str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error Adding Risers", 
                                f"Failed to add risers:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    # ==================== REPORT GENERATION METHODS ====================
    
    def _generate_report(self):
        """Generate a simulation report in HTML or PDF format"""
        if self.current_results is None:
            QMessageBox.warning(self, "No Results", 
                               "Please run a simulation first before generating a report.")
            return
        
        try:
            from core.report_generator import ReportGenerator, SimulationReport
            from PyQt5.QtWidgets import QFileDialog
            
            # Get save path from user
            filepath, filter_used = QFileDialog.getSaveFileName(
                self,
                "Save Simulation Report",
                os.path.expanduser("~/casting_report.html"),
                "HTML Report (*.html);;Markdown Report (*.md);;PDF Report (*.pdf)"
            )
            
            if not filepath:
                return
            
            self.status_label.setText("🔄 Generating report...")
            QApplication.processEvents()
            
            # Determine format from extension
            ext = os.path.splitext(filepath)[1].lower()
            
            # Build report data
            report_data = SimulationReport(
                title=f"Solidus Pro - Simulation Report",
                timestamp=datetime.now(),
                alloy=self._current_alloy,
                mesh_params={
                    'voxel_size': self._current_voxel_size,
                    'refinement_level': self._current_refinement
                },
                results=self.current_results,
                analysis_options=self._get_effective_analysis_options()
            )
            
            # Generate report
            generator = ReportGenerator()
            
            if ext == '.pdf':
                success = generator.generate_pdf(report_data, filepath)
            elif ext == '.md':
                success = generator.generate_markdown(report_data, filepath)
            else:
                success = generator.generate_html(report_data, filepath)
            
            if success:
                self.status_label.setText(f"✅ Report saved: {os.path.basename(filepath)}")
                QMessageBox.information(self, "Report Generated", 
                                       f"Report saved to:\n{filepath}")
            else:
                QMessageBox.warning(self, "Report Error", 
                                   "Failed to generate report. Check console for details.")
                
        except ImportError as e:
            QMessageBox.critical(self, "Module Error", 
                                f"Report generator module not found:\n{str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Report Error", 
                                f"Error generating report:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def _refresh_parts_visualization(self):
        """Refresh the visualization of all parts"""
        self.visualizer.clear_all()
        for part in self.part_manager.get_all_parts():
            self.visualizer.add_stl_mesh(
                part.mesh, 
                name=part.part_id,
                part_type=part.part_type
            )
        self.visualizer.reset_camera()
    
    def _reset_camera_view(self):
        """Reset the camera to default view"""
        self.visualizer.reset_camera()
        self.status_label.setText("Camera view reset")

    def _toggle_projection_mode(self, checked: bool):
        """Toggle between perspective (checked) and orthographic (unchecked) projection"""
        mode = 'perspective' if checked else 'orthographic'
        self.visualizer.set_projection_mode(mode)
        label = "Perspective" if checked else "Orthographic"
        self.status_label.setText(f"Projection mode: {label}")

    def _show_about_dialog(self):
        """Show the about dialog"""
        QMessageBox.about(
            self,
            "About Solidus Pro",
            "<h2>Solidus Pro</h2>"
            "<p>Professional Casting Solidification Simulator</p>"
            "<p>Version 2.0</p>"
            "<p>A powerful tool for simulating and analyzing casting solidification.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>3D visualization of casting geometry</li>"
            "<li>Shrinkage porosity prediction</li>"
            "<li>Hotspot and Niyama analysis</li>"
            "<li>Automatic riser design with Heuvers' method</li>"
            "<li>Multi-part support with risers and chills</li>"
            "<li>PDF/HTML report generation</li>"
            "</ul>"
            "<p>Optimized for Apple Silicon M-series chips.</p>"
        )

    def _show_help_guide_dialog(self):
        """Show the alloy and results interpretation help dialog."""
        selected_alloy = getattr(self, "_current_alloy", None)
        self.help_guide_dialog.refresh_content(selected_alloy)
        self.help_guide_dialog.show()
        self.help_guide_dialog.raise_()
        self.help_guide_dialog.activateWindow()
    
    def _show_log_viewer(self):
        """Show the error log viewer dialog"""
        self.log_viewer_dialog.show()
        self.log_viewer_dialog.raise_()
        self.log_viewer_dialog.activateWindow()

    def closeEvent(self, event):
        """Prevent closing while a simulation is still running."""
        if self.simulation_worker is not None and self.simulation_worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Simulation Running",
                "A simulation is still running. Stop it and keep the application open?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self._cancel_simulation()
            event.ignore()
            return
        self._stop_fraction_liquid_playback()
        super().closeEvent(event)
