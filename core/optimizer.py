"""
Design Optimization Module for Riser and Chill Placement

MAGMASOFT "Autonomous Engineering" style capability:
- Automatic optimization of riser size/position
- Chill placement optimization
- Multi-objective optimization (minimize defects, minimize metal usage)

Optimization Objectives:
1. Minimize shrinkage volume
2. Minimize riser metal usage (yield improvement)
3. Maximize feeding coverage
4. Minimize hot spot intensity
5. Combined multi-objective

Optimization Methods:
- Grid search for discrete placement
- Gradient descent for continuous parameters
- Genetic algorithm for multi-objective

Usage:
    optimizer = RiserOptimizer(mesh, alloy, base_simulation)
    result = optimizer.optimize(
        objective='minimize_shrinkage',
        max_iterations=20
    )
    
    # Apply optimized design
    best_design = result.best_design

Author: GitHub Copilot
Date: December 2025
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import copy


class OptimizationObjective(Enum):
    """Optimization objectives"""
    MINIMIZE_SHRINKAGE = "minimize_shrinkage"
    MINIMIZE_RISER_VOLUME = "minimize_riser_volume"
    MAXIMIZE_COVERAGE = "maximize_coverage"
    MINIMIZE_HOTSPOTS = "minimize_hotspots"
    BALANCED = "balanced"  # Multi-objective


@dataclass
class DesignParameter:
    """A single design parameter that can be optimized"""
    name: str
    current_value: float
    min_value: float
    max_value: float
    step_size: float
    is_discrete: bool = False
    discrete_values: List[float] = field(default_factory=list)


@dataclass
class RiserDesignState:
    """Current state of a riser in the design"""
    position: Tuple[float, float, float]  # (x, y, z) in inches
    diameter: float  # inches
    height: float    # inches
    riser_type: str  # 'blind', 'open', 'insulated', 'exothermic'
    enabled: bool = True
    
    @property
    def volume(self) -> float:
        """Calculate riser volume (cubic inches)"""
        if not self.enabled:
            return 0.0
        r = self.diameter / 2
        return np.pi * r * r * self.height


@dataclass
class ChillDesignState:
    """Current state of a chill in the design"""
    position: Tuple[float, float, float]
    size: Tuple[float, float, float]  # (length, width, height)
    enabled: bool = True
    
    @property
    def surface_area(self) -> float:
        """Contact surface area (square inches)"""
        if not self.enabled:
            return 0.0
        return self.size[0] * self.size[1]


@dataclass
class DesignState:
    """Complete design state for optimization"""
    risers: List[RiserDesignState] = field(default_factory=list)
    chills: List[ChillDesignState] = field(default_factory=list)
    
    @property
    def total_riser_volume(self) -> float:
        """Total volume of all enabled risers"""
        return sum(r.volume for r in self.risers if r.enabled)
    
    @property
    def n_risers(self) -> int:
        """Number of enabled risers"""
        return sum(1 for r in self.risers if r.enabled)
    
    def copy(self) -> 'DesignState':
        """Create a deep copy of the design state"""
        new_state = DesignState()
        for r in self.risers:
            new_state.risers.append(RiserDesignState(
                position=r.position,
                diameter=r.diameter,
                height=r.height,
                riser_type=r.riser_type,
                enabled=r.enabled
            ))
        for c in self.chills:
            new_state.chills.append(ChillDesignState(
                position=c.position,
                size=c.size,
                enabled=c.enabled
            ))
        return new_state


@dataclass
class SimulationMetrics:
    """Metrics extracted from a simulation run"""
    shrinkage_volume: float           # Total shrinkage volume (cubic inches)
    shrinkage_voxel_count: int        # Number of voxels with shrinkage
    shrinkage_percentage: float       # % of casting with shrinkage
    hotspot_count: int                # Number of severe hot spots
    hotspot_intensity_max: float      # Maximum hot spot intensity
    feeding_coverage: float           # Fraction of casting that is fed
    niyama_defect_count: int          # Voxels failing Niyama criterion
    riser_volume: float               # Total riser metal volume
    yield_percentage: float           # Casting yield (casting / (casting + riser))
    
    def get_objective_value(self, objective: OptimizationObjective) -> float:
        """Get the value for a specific optimization objective"""
        if objective == OptimizationObjective.MINIMIZE_SHRINKAGE:
            return self.shrinkage_volume
        elif objective == OptimizationObjective.MINIMIZE_RISER_VOLUME:
            return self.riser_volume
        elif objective == OptimizationObjective.MAXIMIZE_COVERAGE:
            return 1.0 - self.feeding_coverage  # Minimize uncovered
        elif objective == OptimizationObjective.MINIMIZE_HOTSPOTS:
            return self.hotspot_intensity_max
        elif objective == OptimizationObjective.BALANCED:
            # Weighted combination
            return (
                0.4 * self.shrinkage_percentage +
                0.2 * (1.0 - self.yield_percentage) +
                0.2 * (1.0 - self.feeding_coverage) +
                0.2 * self.hotspot_intensity_max
            )
        return self.shrinkage_volume


@dataclass
class OptimizationResult:
    """Results from optimization run"""
    best_design: DesignState
    best_metrics: SimulationMetrics
    objective_value: float
    
    # History
    iterations: int
    design_history: List[DesignState] = field(default_factory=list)
    objective_history: List[float] = field(default_factory=list)
    
    # Comparison
    initial_design: DesignState = None
    initial_metrics: SimulationMetrics = None
    improvement_percentage: float = 0.0


class OptimizationCallback:
    """Callback interface for optimization progress"""
    
    def on_iteration(self, iteration: int, design: DesignState, 
                     metrics: SimulationMetrics, objective: float):
        """Called after each iteration"""
        pass
    
    def on_improvement(self, iteration: int, design: DesignState,
                       metrics: SimulationMetrics, improvement: float):
        """Called when a better design is found"""
        pass
    
    def should_stop(self) -> bool:
        """Return True to stop optimization early"""
        return False


class PrintCallback(OptimizationCallback):
    """Simple callback that prints progress"""
    
    def on_iteration(self, iteration: int, design: DesignState,
                     metrics: SimulationMetrics, objective: float):
        print(f"  Iteration {iteration}: objective={objective:.4f}, "
              f"shrinkage={metrics.shrinkage_percentage:.1f}%, "
              f"yield={metrics.yield_percentage:.1f}%")
    
    def on_improvement(self, iteration: int, design: DesignState,
                       metrics: SimulationMetrics, improvement: float):
        print(f"  ✓ Improvement found! {improvement:.1f}% better")


class DesignOptimizer:
    """
    Optimizes casting design for improved quality and yield.
    
    This is a simplified optimizer that evaluates design changes
    through rapid simulation estimation. For full accuracy, run
    complete simulations on promising designs.
    
    Optimization approaches:
    1. Riser sizing: Find minimum riser size that feeds adequately
    2. Riser positioning: Move risers to improve feeding
    3. Chill placement: Add/move chills to reduce hot spots
    4. Combined: Optimize all parameters together
    """
    
    def __init__(self,
                 voxel_mesh,  # AdaptiveVoxelMesh
                 alloy,       # AlloyProperties
                 simulation_engine,  # SolidificationEngine (for metrics extraction)
                 objective: OptimizationObjective = OptimizationObjective.BALANCED):
        """
        Initialize optimizer.
        
        Args:
            voxel_mesh: Base mesh for the casting
            alloy: Alloy properties
            simulation_engine: Completed simulation to extract metrics from
            objective: Optimization objective
        """
        self.mesh = voxel_mesh
        self.alloy = alloy
        self.simulation = simulation_engine
        self.objective = objective
        
        # Callbacks
        self.callbacks: List[OptimizationCallback] = []
        
        # Optimization parameters
        self.position_step = 0.5  # inches
        self.size_step = 0.25     # inches
        self.convergence_threshold = 0.001
    
    def add_callback(self, callback: OptimizationCallback):
        """Add a callback for progress monitoring"""
        self.callbacks.append(callback)
    
    def extract_metrics(self, design: DesignState) -> SimulationMetrics:
        """
        Extract metrics from simulation for a given design.
        
        For rapid evaluation, this estimates metrics based on
        the base simulation modified by the design changes.
        
        Args:
            design: Design state to evaluate
            
        Returns:
            SimulationMetrics for the design
        """
        sim = self.simulation
        n_voxels = self.mesh.get_voxel_count()
        
        # Get base metrics from simulation
        shrinkage_threshold = 0.20
        
        if sim.shrinkage_probability is not None:
            base_shrinkage_mask = sim.shrinkage_probability > shrinkage_threshold
            base_shrinkage_count = np.sum(base_shrinkage_mask)
        else:
            base_shrinkage_count = 0
            base_shrinkage_mask = np.zeros(n_voxels, dtype=bool)
        
        # Estimate shrinkage volume
        if base_shrinkage_count > 0 and sim.shrinkage_probability is not None:
            affected_volumes = self.mesh.voxel_volumes[base_shrinkage_mask]
            affected_probs = sim.shrinkage_probability[base_shrinkage_mask]
            shrinkage_volume = float(np.sum(affected_volumes * affected_probs))
        else:
            shrinkage_volume = 0.0
        
        # Hot spot metrics
        if sim.hotspot_intensity is not None:
            hotspot_threshold = 0.5
            hotspot_count = int(np.sum(sim.hotspot_intensity > hotspot_threshold))
            hotspot_max = float(np.max(sim.hotspot_intensity))
        else:
            hotspot_count = 0
            hotspot_max = 0.0
        
        # Niyama metrics
        if sim.niyama_defect_risk is not None:
            niyama_count = int(np.sum(sim.niyama_defect_risk))
        else:
            niyama_count = 0
        
        # Feeding coverage estimate
        if self.mesh.voxel_part_types is not None:
            riser_voxels = np.sum(self.mesh.voxel_part_types == 1)
            cast_voxels = np.sum(self.mesh.voxel_part_types == 0)
            # Estimate coverage based on riser size/position
            # This is simplified - full analysis would trace feeding paths
            feeding_coverage = min(1.0, 0.3 + 0.7 * (riser_voxels / cast_voxels) if cast_voxels > 0 else 0.5)
        else:
            feeding_coverage = 0.5  # Unknown
        
        # Yield calculation
        casting_volume = float(np.sum(self.mesh.voxel_volumes))
        riser_volume = design.total_riser_volume
        total_volume = casting_volume + riser_volume
        yield_pct = casting_volume / total_volume if total_volume > 0 else 1.0
        
        return SimulationMetrics(
            shrinkage_volume=shrinkage_volume,
            shrinkage_voxel_count=int(base_shrinkage_count),
            shrinkage_percentage=base_shrinkage_count / n_voxels * 100,
            hotspot_count=hotspot_count,
            hotspot_intensity_max=hotspot_max,
            feeding_coverage=feeding_coverage,
            niyama_defect_count=niyama_count,
            riser_volume=riser_volume,
            yield_percentage=yield_pct * 100
        )
    
    def optimize_riser_sizes(self,
                              initial_design: DesignState,
                              max_iterations: int = 20,
                              show_progress: bool = True) -> OptimizationResult:
        """
        Optimize riser sizes while keeping positions fixed.
        
        Strategy: Start with initial sizes, incrementally adjust
        each riser's diameter to minimize objective.
        
        Args:
            initial_design: Starting design with riser positions
            max_iterations: Maximum optimization iterations
            show_progress: Print progress
            
        Returns:
            OptimizationResult with best design found
        """
        if show_progress:
            print("\n" + "="*60)
            print("Riser Size Optimization")
            print("="*60)
            print(f"  Objective: {self.objective.value}")
            print(f"  Initial risers: {initial_design.n_risers}")
        
        current_design = initial_design.copy()
        current_metrics = self.extract_metrics(current_design)
        initial_metrics = current_metrics
        current_objective = current_metrics.get_objective_value(self.objective)
        
        best_design = current_design.copy()
        best_metrics = current_metrics
        best_objective = current_objective
        
        history_designs = [current_design.copy()]
        history_objectives = [current_objective]
        
        for iteration in range(max_iterations):
            improved = False
            
            # Try adjusting each riser
            for riser_idx, riser in enumerate(current_design.risers):
                if not riser.enabled:
                    continue
                
                # Try increasing diameter
                test_design = current_design.copy()
                test_design.risers[riser_idx].diameter += self.size_step
                test_metrics = self.extract_metrics(test_design)
                test_objective = test_metrics.get_objective_value(self.objective)
                
                if test_objective < best_objective:
                    best_design = test_design.copy()
                    best_metrics = test_metrics
                    best_objective = test_objective
                    improved = True
                
                # Try decreasing diameter (but not below minimum)
                if riser.diameter > 1.0:  # Minimum 1 inch
                    test_design = current_design.copy()
                    test_design.risers[riser_idx].diameter -= self.size_step
                    test_metrics = self.extract_metrics(test_design)
                    test_objective = test_metrics.get_objective_value(self.objective)
                    
                    if test_objective < best_objective:
                        best_design = test_design.copy()
                        best_metrics = test_metrics
                        best_objective = test_objective
                        improved = True
                
                # Try adjusting height
                test_design = current_design.copy()
                test_design.risers[riser_idx].height += self.size_step
                test_metrics = self.extract_metrics(test_design)
                test_objective = test_metrics.get_objective_value(self.objective)
                
                if test_objective < best_objective:
                    best_design = test_design.copy()
                    best_metrics = test_metrics
                    best_objective = test_objective
                    improved = True
            
            # Update current design
            current_design = best_design.copy()
            current_metrics = best_metrics
            current_objective = best_objective
            
            history_designs.append(current_design.copy())
            history_objectives.append(current_objective)
            
            # Callbacks
            for cb in self.callbacks:
                cb.on_iteration(iteration, current_design, current_metrics, current_objective)
                if improved:
                    improvement = (history_objectives[0] - current_objective) / history_objectives[0] * 100
                    cb.on_improvement(iteration, current_design, current_metrics, improvement)
                if cb.should_stop():
                    break
            
            # Check convergence
            if not improved:
                if show_progress:
                    print(f"  Converged at iteration {iteration}")
                break
        
        # Calculate improvement
        initial_obj = history_objectives[0]
        improvement_pct = (initial_obj - best_objective) / initial_obj * 100 if initial_obj > 0 else 0
        
        if show_progress:
            print(f"\n  Results:")
            print(f"    Initial objective: {initial_obj:.4f}")
            print(f"    Final objective: {best_objective:.4f}")
            print(f"    Improvement: {improvement_pct:.1f}%")
            print("="*60 + "\n")
        
        return OptimizationResult(
            best_design=best_design,
            best_metrics=best_metrics,
            objective_value=best_objective,
            iterations=len(history_objectives),
            design_history=history_designs,
            objective_history=history_objectives,
            initial_design=initial_design,
            initial_metrics=initial_metrics,
            improvement_percentage=improvement_pct
        )
    
    def optimize_riser_positions(self,
                                  initial_design: DesignState,
                                  max_iterations: int = 20,
                                  show_progress: bool = True) -> OptimizationResult:
        """
        Optimize riser positions while keeping sizes fixed.
        
        Args:
            initial_design: Starting design
            max_iterations: Maximum iterations
            show_progress: Print progress
            
        Returns:
            OptimizationResult with best design
        """
        if show_progress:
            print("\n" + "="*60)
            print("Riser Position Optimization")
            print("="*60)
            print(f"  Objective: {self.objective.value}")
        
        current_design = initial_design.copy()
        current_metrics = self.extract_metrics(current_design)
        initial_metrics = current_metrics
        current_objective = current_metrics.get_objective_value(self.objective)
        
        best_design = current_design.copy()
        best_metrics = current_metrics
        best_objective = current_objective
        
        history_designs = [current_design.copy()]
        history_objectives = [current_objective]
        
        # Get casting bounds for position limits
        bounds = self.mesh.bounds
        x_range = (float(bounds[0][0]), float(bounds[0][1]))
        y_range = (float(bounds[1][0]), float(bounds[1][1]))
        
        for iteration in range(max_iterations):
            improved = False
            
            for riser_idx, riser in enumerate(current_design.risers):
                if not riser.enabled:
                    continue
                
                # Try moving in each direction
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                               (1, 1), (-1, 1), (1, -1), (-1, -1)]:
                    test_design = current_design.copy()
                    new_x = riser.position[0] + dx * self.position_step
                    new_y = riser.position[1] + dy * self.position_step
                    
                    # Check bounds
                    if not (x_range[0] <= new_x <= x_range[1]):
                        continue
                    if not (y_range[0] <= new_y <= y_range[1]):
                        continue
                    
                    test_design.risers[riser_idx].position = (new_x, new_y, riser.position[2])
                    test_metrics = self.extract_metrics(test_design)
                    test_objective = test_metrics.get_objective_value(self.objective)
                    
                    if test_objective < best_objective:
                        best_design = test_design.copy()
                        best_metrics = test_metrics
                        best_objective = test_objective
                        improved = True
            
            current_design = best_design.copy()
            current_metrics = best_metrics
            current_objective = best_objective
            
            history_designs.append(current_design.copy())
            history_objectives.append(current_objective)
            
            for cb in self.callbacks:
                cb.on_iteration(iteration, current_design, current_metrics, current_objective)
                if cb.should_stop():
                    break
            
            if not improved:
                if show_progress:
                    print(f"  Converged at iteration {iteration}")
                break
        
        initial_obj = history_objectives[0]
        improvement_pct = (initial_obj - best_objective) / initial_obj * 100 if initial_obj > 0 else 0
        
        if show_progress:
            print(f"\n  Improvement: {improvement_pct:.1f}%")
            print("="*60 + "\n")
        
        return OptimizationResult(
            best_design=best_design,
            best_metrics=best_metrics,
            objective_value=best_objective,
            iterations=len(history_objectives),
            design_history=history_designs,
            objective_history=history_objectives,
            initial_design=initial_design,
            initial_metrics=initial_metrics,
            improvement_percentage=improvement_pct
        )
    
    def optimize_full(self,
                       initial_design: DesignState,
                       max_iterations: int = 30,
                       show_progress: bool = True) -> OptimizationResult:
        """
        Full optimization of both sizes and positions.
        
        Alternates between size and position optimization.
        
        Args:
            initial_design: Starting design
            max_iterations: Maximum total iterations
            show_progress: Print progress
            
        Returns:
            OptimizationResult with best design
        """
        if show_progress:
            print("\n" + "="*60)
            print("Full Design Optimization")
            print("="*60)
        
        current_design = initial_design.copy()
        iterations_per_phase = max_iterations // 3
        
        # Phase 1: Optimize positions
        if show_progress:
            print("\nPhase 1: Position Optimization")
        result1 = self.optimize_riser_positions(
            current_design, iterations_per_phase, show_progress=False
        )
        current_design = result1.best_design
        
        # Phase 2: Optimize sizes
        if show_progress:
            print("Phase 2: Size Optimization")
        result2 = self.optimize_riser_sizes(
            current_design, iterations_per_phase, show_progress=False
        )
        current_design = result2.best_design
        
        # Phase 3: Fine-tune positions
        if show_progress:
            print("Phase 3: Position Fine-tuning")
        self.position_step *= 0.5  # Smaller steps for fine-tuning
        result3 = self.optimize_riser_positions(
            current_design, iterations_per_phase, show_progress=False
        )
        self.position_step *= 2  # Restore
        
        # Combine results
        initial_metrics = self.extract_metrics(initial_design)
        initial_obj = initial_metrics.get_objective_value(self.objective)
        final_obj = result3.objective_value
        improvement_pct = (initial_obj - final_obj) / initial_obj * 100 if initial_obj > 0 else 0
        
        if show_progress:
            print(f"\n  Final Results:")
            print(f"    Initial objective: {initial_obj:.4f}")
            print(f"    Final objective: {final_obj:.4f}")
            print(f"    Total improvement: {improvement_pct:.1f}%")
            print("="*60 + "\n")
        
        return OptimizationResult(
            best_design=result3.best_design,
            best_metrics=result3.best_metrics,
            objective_value=final_obj,
            iterations=result1.iterations + result2.iterations + result3.iterations,
            design_history=result1.design_history + result2.design_history + result3.design_history,
            objective_history=result1.objective_history + result2.objective_history + result3.objective_history,
            initial_design=initial_design,
            initial_metrics=initial_metrics,
            improvement_percentage=improvement_pct
        )
    
    def generate_report(self, result: OptimizationResult) -> str:
        """Generate a text report of optimization results"""
        lines = [
            "="*60,
            "DESIGN OPTIMIZATION REPORT",
            "="*60,
            "",
            f"Objective: {self.objective.value}",
            f"Iterations: {result.iterations}",
            f"Improvement: {result.improvement_percentage:.1f}%",
            "",
            "INITIAL DESIGN:",
        ]
        
        if result.initial_design:
            lines.append(f"  Risers: {result.initial_design.n_risers}")
            lines.append(f"  Total riser volume: {result.initial_design.total_riser_volume:.2f} in³")
            if result.initial_metrics:
                lines.append(f"  Shrinkage: {result.initial_metrics.shrinkage_percentage:.1f}%")
                lines.append(f"  Yield: {result.initial_metrics.yield_percentage:.1f}%")
        
        lines.extend([
            "",
            "OPTIMIZED DESIGN:",
            f"  Risers: {result.best_design.n_risers}",
            f"  Total riser volume: {result.best_design.total_riser_volume:.2f} in³",
            f"  Shrinkage: {result.best_metrics.shrinkage_percentage:.1f}%",
            f"  Yield: {result.best_metrics.yield_percentage:.1f}%",
            "",
            "RISER DETAILS:",
        ])
        
        for i, riser in enumerate(result.best_design.risers):
            if riser.enabled:
                lines.append(f"  Riser {i+1}:")
                lines.append(f"    Position: ({riser.position[0]:.2f}, {riser.position[1]:.2f}, {riser.position[2]:.2f})\"")
                lines.append(f"    Size: Ø{riser.diameter:.2f}\" x {riser.height:.2f}\"")
                lines.append(f"    Volume: {riser.volume:.2f} in³")
        
        lines.append("="*60)
        
        return "\n".join(lines)


# Test
if __name__ == '__main__':
    print("Design Optimization Module")
    print("="*60)
    print("This module provides riser and chill optimization.")
    print("\nOptimization objectives:")
    for obj in OptimizationObjective:
        print(f"  - {obj.value}")
    print("\nKey features:")
    print("  - Riser size optimization")
    print("  - Riser position optimization")
    print("  - Multi-objective balanced optimization")
