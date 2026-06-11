"""
Integrated Thermal Simulation Engine
Combines all physics modules into a unified simulation:

1. Temperature Field Calculation (thermal_solver.py)
   - Finite difference heat equation
   - Latent heat release
   - Temperature-dependent properties

2. Mold-Metal Interface (mold_interface.py)
   - Interfacial heat transfer coefficient
   - Mold thermal properties
   - Air gap formation

3. Convection in Liquid Metal (convection.py)
   - Natural convection enhancement
   - Buoyancy-driven mixing
   - Temperature homogenization

This engine provides a physics-based alternative to the geometric
modulus-based solidification in the original SolidificationEngine.

Usage:
    engine = ThermalSimulationEngine(mesh, alloy, config)
    engine.initialize()
    engine.run_simulation()
    
    # Access results
    temperatures = engine.get_temperature_field()
    fraction_solid = engine.get_fraction_solid()
    niyama = engine.get_niyama_criterion()

Author: GitHub Copilot
Date: November 2025
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from core.thermal_solver import (
    ThermalSolver, ThermalSolverConfig, ThermalProperties,
    create_thermal_properties_from_alloy
)
from core.mold_interface import (
    MoldInterfaceModel, MoldType, MoldProperties, get_mold_properties
)
from core.convection import (
    ConvectionModel, LiquidMetalProperties, create_liquid_properties_from_alloy
)
from core.materials import AlloyProperties
from core.memory_manager import create_managed_array


class SimulationMode(str, Enum):
    """Available simulation modes"""
    FULL_THERMAL = "full_thermal"      # Full heat equation + convection + IHTC
    THERMAL_ONLY = "thermal_only"      # Heat equation + IHTC, no convection
    FAST_MODULUS = "fast_modulus"      # Original geometric modulus method


@dataclass
class ThermalSimConfig:
    """
    Configuration for integrated thermal simulation.
    """
    # Simulation mode
    mode: str = "full_thermal"
    
    # Time stepping
    max_time: float = 1000.0           # Maximum simulation time (s)
    output_interval: float = 10.0      # Time between output snapshots (s)
    
    # Convergence
    target_fraction_solid: float = 0.99  # Stop when this fraction is solid
    
    # Thermal solver settings
    cfl_safety: float = 0.4            # CFL number for stability
    use_temperature_dependent: bool = True
    include_latent_heat: bool = True
    
    # Mold settings
    mold_type: str = "sand_silica"
    mold_temperature: float = 25.0     # °C
    
    # Convection settings
    include_convection: bool = True
    liquid_threshold: Optional[float] = None     # Fraction solid for convection cutoff (None = use alloy's critical_fraction_solid)
    mixing_coefficient: float = 0.1    # Buoyancy mixing strength
    
    # Performance
    use_gpu: bool = False
    
    # Output
    save_history: bool = True          # Save temperature history for animation


@dataclass
class SimulationResults:
    """
    Results from thermal simulation.
    """
    # Final state
    temperatures: np.ndarray = None
    fraction_solid: np.ndarray = None
    solidification_times: np.ndarray = None
    
    # Thermal analysis
    thermal_gradients: np.ndarray = None
    cooling_rates: np.ndarray = None
    niyama_values: np.ndarray = None
    
    # Interface data
    final_ihtc: np.ndarray = None
    final_air_gap: np.ndarray = None
    mold_surface_temps: np.ndarray = None
    
    # History (if save_history=True)
    temperature_history: List[Tuple[float, np.ndarray]] = field(default_factory=list)
    fraction_solid_history: List[Tuple[float, np.ndarray]] = field(default_factory=list)
    
    # Statistics
    total_time: float = 0.0
    total_steps: int = 0
    total_latent_heat: float = 0.0
    avg_solidification_time: float = 0.0


class ThermalSimulationEngine:
    """
    Integrated thermal simulation engine for casting solidification.
    
    Combines:
    - Finite difference heat equation solver
    - Mold-metal interface heat transfer with IHTC
    - Natural convection in liquid metal
    
    This provides physically accurate temperature fields for:
    - Niyama criterion calculation
    - Hot spot prediction
    - Shrinkage prediction
    - Solidification sequence visualization
    """
    
    def __init__(self,
                 voxel_mesh,  # AdaptiveVoxelMesh
                 alloy: AlloyProperties,
                 config: Optional[ThermalSimConfig] = None):
        """
        Initialize thermal simulation engine.
        
        Args:
            voxel_mesh: AdaptiveVoxelMesh with voxel data
            alloy: AlloyProperties for the casting material
            config: Simulation configuration
        """
        self.mesh = voxel_mesh
        self.alloy = alloy
        self.config = config or ThermalSimConfig()
        
        # Convert units (mesh is in inches, physics uses meters)
        self.voxel_size_m = voxel_mesh.base_voxel_size * 0.0254
        
        # Initialize component models
        self._thermal_solver: Optional[ThermalSolver] = None
        self._mold_interface: Optional[MoldInterfaceModel] = None
        self._convection: Optional[ConvectionModel] = None
        
        # Mapping arrays
        self._surface_indices: np.ndarray = None
        self._interior_indices: np.ndarray = None
        
        # Results
        self.results = SimulationResults()
        
        # State tracking
        self._is_initialized = False
        self._current_time = 0.0
        
    def initialize(self, show_progress: bool = True):
        """
        Initialize all simulation components.
        
        Args:
            show_progress: Print initialization progress
        """
        n_voxels = self.mesh.get_voxel_count()
        
        if show_progress:
            print("\n" + "="*60)
            print("Initializing Thermal Simulation Engine")
            print("="*60)
            print(f"Mode: {self.config.mode}")
            print(f"Voxels: {n_voxels:,}")
            print(f"Alloy: {self.alloy.name}")
        
        # Get surface and interior voxel indices
        self._surface_indices = self.mesh.get_surface_voxel_indices()
        all_indices = np.arange(n_voxels)
        surface_set = set(self._surface_indices)
        self._interior_indices = np.array([i for i in all_indices if i not in surface_set])
        
        if show_progress:
            print(f"Surface voxels: {len(self._surface_indices):,}")
            print(f"Interior voxels: {len(self._interior_indices):,}")
        
        # Initialize thermal properties
        thermal_props = create_thermal_properties_from_alloy(self.alloy)
        
        # Initialize thermal solver
        thermal_config = ThermalSolverConfig()
        thermal_config.cfl_safety_factor = self.config.cfl_safety
        thermal_config.use_temperature_dependent_props = self.config.use_temperature_dependent
        thermal_config.include_latent_heat = self.config.include_latent_heat
        thermal_config.pouring_temperature = self.alloy.pouring_temp
        thermal_config.mold_temperature = self.config.mold_temperature
        thermal_config.use_gpu = self.config.use_gpu
        
        self._thermal_solver = ThermalSolver(
            n_voxels=n_voxels,
            voxel_size_m=self.voxel_size_m,
            thermal_props=thermal_props,
            config=thermal_config
        )
        
        # Initialize with mesh data
        surface_mask = np.zeros(n_voxels, dtype=np.bool_)
        surface_mask[self._surface_indices] = True
        
        self._thermal_solver.initialize(
            neighbors=self.mesh.voxel_neighbors,
            surface_mask=surface_mask,
            pouring_temp=self.alloy.pouring_temp
        )
        
        # Initialize mold interface model
        mold_type = MoldType(self.config.mold_type)
        mold_props = get_mold_properties(mold_type)
        self._mold_interface = MoldInterfaceModel(
            n_surface_voxels=len(self._surface_indices),
            mold_props=mold_props
        )
        
        surface_temps = np.full(len(self._surface_indices), 
                               self.alloy.pouring_temp, 
                               dtype=np.float32)
        self._mold_interface.initialize(
            surface_temps,
            self.alloy.solidus_temp,
            self.alloy.liquidus_temp
        )
        
        # Initialize convection model (if enabled)
        if self.config.include_convection and self.config.mode == "full_thermal":
            liquid_props = create_liquid_properties_from_alloy(self.alloy)
            
            # Use alloy's critical_fraction_solid as liquid threshold if not specified
            liquid_threshold = self.config.liquid_threshold
            if liquid_threshold is None:
                liquid_threshold = self.alloy.critical_fraction_solid
            
            self._convection = ConvectionModel(
                n_voxels=n_voxels,
                liquid_props=liquid_props,
                liquid_threshold=liquid_threshold,
                mixing_coefficient=self.config.mixing_coefficient
            )
            self._convection.initialize()
        
        # Initialize result arrays
        self.results.solidification_times = create_managed_array(
            n_voxels, dtype=np.float32, name="solidification_times"
        )
        
        self._is_initialized = True
        self._current_time = 0.0
        
        if show_progress:
            print("="*60)
            print("Thermal simulation initialized successfully")
            print("="*60 + "\n")
    
    def _require_initialized_components(self) -> Tuple[ThermalSolver, MoldInterfaceModel]:
        """
        Return initialized runtime components and fail fast on invalid call order.
        """
        if self._thermal_solver is None or self._mold_interface is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        return self._thermal_solver, self._mold_interface

    def step(self) -> Tuple[float, float, float]:
        """
        Advance simulation by one time step.
        
        Returns:
            Tuple of (time_step, avg_fraction_solid, max_temp_change)
        """
        solver, mold_interface = self._require_initialized_components()

        # Get current surface temperatures for mold interface
        surface_temps = solver.temperatures[self._surface_indices]
        
        # Update mold interface and get new boundary conditions
        heat_flux = mold_interface.update(surface_temps, solver.time_step)
        
        # Update boundary temperatures based on mold surface
        mold_temps = mold_interface.get_boundary_temperature()
        solver.boundary_temps[self._surface_indices] = mold_temps
        
        # Apply convection enhancement if enabled
        if self._convection is not None:
            # Get effective conductivity with convection
            k_eff = self._convection.get_effective_conductivity(
                solver.temperatures,
                solver.fraction_solid,
                solver.neighbors,
                self.voxel_size_m
            )
            
            # Update diffusivity based on enhanced conductivity
            rho = solver.thermal_props.get_density(
                np.mean(solver.temperatures)
            )
            cp = solver.thermal_props.get_specific_heat(
                np.mean(solver.temperatures)
            )
            solver.diffusivity = k_eff / (rho * cp)
            
            # Apply buoyancy mixing
            heights = self._get_normalized_heights()
            solver.temperatures = self._convection.apply_mixing(
                solver.temperatures,
                solver.fraction_solid,
                solver.neighbors,
                heights,
                solver.time_step
            )
        
        # Take thermal solver step
        dt, max_change = solver.step()
        
        # Record solidification times
        newly_solid = (
            (solver.fraction_solid >= 1.0) & 
            (self.results.solidification_times == 0)
        )
        self.results.solidification_times[newly_solid] = self._current_time
        
        self._current_time += dt
        
        avg_fs = np.mean(solver.fraction_solid)
        
        return dt, avg_fs, max_change
    
    def run_simulation(self,
                       show_progress: bool = True,
                       progress_callback=None) -> bool:
        """
        Run complete thermal simulation until solidification complete.
        
        Args:
            show_progress: Print progress to console
            progress_callback: Optional callback(current, total, stage)
        
        Returns:
            True if simulation completed successfully
        """
        if not self._is_initialized:
            self.initialize(show_progress)
            solver, mold_interface = self._require_initialized_components()
        else:
            solver, mold_interface = self._require_initialized_components()
        
        if show_progress:
            print("\n" + "="*60)
            print("Running Thermal Simulation")
            print("="*60)
        
        step_count = 0
        last_output_time = 0.0
        last_progress_pct = 0
        
        while self._current_time < self.config.max_time:
            dt, avg_fs, max_change = self.step()
            step_count += 1
            
            # Check if solidification complete
            if avg_fs >= self.config.target_fraction_solid:
                break
            
            # Save history snapshot
            if (self.config.save_history and 
                self._current_time - last_output_time >= self.config.output_interval):
                self.results.temperature_history.append(
                    (self._current_time, solver.temperatures.copy())
                )
                self.results.fraction_solid_history.append(
                    (self._current_time, solver.fraction_solid.copy())
                )
                last_output_time = self._current_time
            
            # Progress reporting
            progress_pct = int(avg_fs / self.config.target_fraction_solid * 100)
            if progress_pct > last_progress_pct:
                if progress_callback:
                    progress_callback(progress_pct, 100, "Thermal Simulation")
                
                if show_progress and progress_pct % 10 == 0:
                    T_min = np.min(solver.temperatures)
                    T_max = np.max(solver.temperatures[~solver.boundary_mask])
                    print(f"  t={self._current_time:.1f}s: "
                          f"fs={avg_fs*100:.1f}%, "
                          f"T=[{T_min:.0f}, {T_max:.0f}]°C, "
                          f"dt={dt:.4f}s")
                
                last_progress_pct = progress_pct
        
        # Finalize results
        self._finalize_results()
        
        if show_progress:
            self._print_summary()
        
        return True
    
    def _finalize_results(self):
        """Compute final results and statistics"""
        solver, mold_interface = self._require_initialized_components()

        # Copy final state
        self.results.temperatures = solver.temperatures.copy()
        self.results.fraction_solid = solver.fraction_solid.copy()
        
        # Compute thermal gradient and cooling rate
        self.results.thermal_gradients = solver.get_thermal_gradient()
        self.results.cooling_rates = solver.get_cooling_rate()
        
        # Compute Niyama criterion
        self.results.niyama_values = solver.get_niyama_criterion()
        
        # Interface data
        self.results.final_ihtc = mold_interface.get_ihtc()
        self.results.final_air_gap = mold_interface.get_air_gap()
        self.results.mold_surface_temps = mold_interface.get_boundary_temperature()
        
        # Statistics
        self.results.total_time = self._current_time
        self.results.total_steps = solver.total_steps
        self.results.total_latent_heat = solver.total_latent_heat_released
        
        solidified_mask = self.results.solidification_times > 0
        if np.any(solidified_mask):
            self.results.avg_solidification_time = np.mean(
                self.results.solidification_times[solidified_mask]
            )
    
    def _print_summary(self):
        """Print simulation summary"""
        _, mold_interface = self._require_initialized_components()

        print("\n" + "="*60)
        print("Thermal Simulation Results")
        print("="*60)
        print(f"Total time: {self.results.total_time:.1f}s")
        print(f"Total steps: {self.results.total_steps:,}")
        print(f"Average solidification time: {self.results.avg_solidification_time:.1f}s")
        print(f"\nFinal state:")
        print(f"  Fraction solid: {np.mean(self.results.fraction_solid)*100:.1f}%")
        print(f"  Temperature range: [{np.min(self.results.temperatures):.0f}, "
              f"{np.max(self.results.temperatures):.0f}]°C")
        print(f"\nNiyama analysis:")
        print(f"  Range: [{np.min(self.results.niyama_values):.2f}, "
              f"{np.max(self.results.niyama_values):.2f}]")
        print(f"  Mean: {np.mean(self.results.niyama_values):.2f}")
        
        if mold_interface is not None:
            stats = mold_interface.get_statistics()
            print(f"\nMold interface:")
            print(f"  Average IHTC: {stats['avg_ihtc']:.0f} W/m²·K")
            print(f"  Max air gap: {stats['max_air_gap_um']:.1f} μm")
            print(f"  Max mold temp: {stats['max_mold_surface_temp']:.0f}°C")
        
        if self._convection is not None:
            stats = self._convection.get_statistics()
            print(f"\nConvection:")
            print(f"  Average Nusselt: {stats['average_nusselt']:.2f}")
            print(f"  Final liquid fraction: {stats['liquid_fraction']*100:.1f}%")
        
        print("="*60 + "\n")
    
    def _get_normalized_heights(self) -> np.ndarray:
        """Get normalized height (0=bottom, 1=top) for each voxel"""
        z_coords = self.mesh.voxel_centers[:, 2]
        z_min = np.min(z_coords)
        z_max = np.max(z_coords)
        
        if z_max > z_min:
            return ((z_coords - z_min) / (z_max - z_min)).astype(np.float32)
        else:
            return np.zeros(len(z_coords), dtype=np.float32)
    
    # === Results Access Methods ===
    
    def get_temperature_field(self) -> np.ndarray:
        """Get final temperature field"""
        return self.results.temperatures
    
    def get_fraction_solid(self) -> np.ndarray:
        """Get final fraction solid"""
        return self.results.fraction_solid
    
    def get_solidification_times(self) -> np.ndarray:
        """Get solidification time for each voxel"""
        return self.results.solidification_times
    
    def get_niyama_criterion(self) -> np.ndarray:
        """Get Niyama criterion values"""
        return self.results.niyama_values
    
    def get_thermal_gradients(self) -> np.ndarray:
        """Get thermal gradient magnitude (°C/m)"""
        return self.results.thermal_gradients
    
    def get_cooling_rates(self) -> np.ndarray:
        """Get cooling rates (°C/s)"""
        return self.results.cooling_rates
    
    def get_niyama_defect_risk(self, threshold: float = 1.0) -> np.ndarray:
        """
        Get boolean mask of voxels with Niyama below threshold.
        
        Args:
            threshold: Niyama threshold for defect risk
        
        Returns:
            Boolean array (True = defect risk)
        """
        return self.results.niyama_values < threshold
    
    def get_temperature_at_time(self, time: float) -> Optional[np.ndarray]:
        """
        Get temperature field at specific time (from history).
        
        Args:
            time: Time in seconds
        
        Returns:
            Temperature array, or None if not available
        """
        if not self.results.temperature_history:
            return None
        
        # Find closest snapshot
        times = [t for t, _ in self.results.temperature_history]
        idx = np.argmin(np.abs(np.array(times) - time))
        
        return self.results.temperature_history[idx][1]
    
    def get_fraction_liquid_at_time(self, time: float) -> Optional[np.ndarray]:
        """
        Get fraction liquid at specific time (from history).
        
        Args:
            time: Time in seconds
        
        Returns:
            Fraction liquid array, or None if not available
        """
        if not self.results.fraction_solid_history:
            return None
        
        # Find closest snapshot
        times = [t for t, _ in self.results.fraction_solid_history]
        idx = np.argmin(np.abs(np.array(times) - time))
        
        return 1.0 - self.results.fraction_solid_history[idx][1]
    
    def export_results(self) -> Dict:
        """Export all results as dictionary"""
        return {
            'mode': self.config.mode,
            'alloy': self.alloy.name,
            'mold_type': self.config.mold_type,
            'total_time': self.results.total_time,
            'total_steps': self.results.total_steps,
            'temperatures': self.results.temperatures,
            'fraction_solid': self.results.fraction_solid,
            'solidification_times': self.results.solidification_times,
            'thermal_gradients': self.results.thermal_gradients,
            'cooling_rates': self.results.cooling_rates,
            'niyama_values': self.results.niyama_values,
            'avg_solidification_time': self.results.avg_solidification_time,
            'n_snapshots': len(self.results.temperature_history)
        }


# Convenience function for quick setup
def create_thermal_simulation(
    mesh,  # AdaptiveVoxelMesh
    alloy: AlloyProperties,
    mold_type: str | MoldType = MoldType.SAND_SILICA,
    include_convection: bool = True,
    save_history: bool = True
) -> ThermalSimulationEngine:
    """
    Create and configure thermal simulation engine.
    
    Args:
        mesh: AdaptiveVoxelMesh
        alloy: AlloyProperties
        mold_type: Type of mold material
        include_convection: Include natural convection
        save_history: Save snapshots for animation
    
    Returns:
        Configured ThermalSimulationEngine
    """
    config = ThermalSimConfig()
    config.mode = (
            "full_thermal"
        if include_convection
            else "thermal_only"
    )
    config.mold_type = mold_type.value if isinstance(mold_type, MoldType) else str(mold_type)
    config.include_convection = include_convection
    config.save_history = save_history
    
    return ThermalSimulationEngine(mesh, alloy, config)
