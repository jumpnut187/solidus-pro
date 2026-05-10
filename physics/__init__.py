"""
Physics Engine Package for Casting Solidification Simulation

Provides modular physics components that can be swapped or extended:
- heat_transfer.py: Thermal conductivity, diffusion, boundary conditions
- solidification_physics.py: Phase change, latent heat, fraction solid
- shrinkage.py: Volumetric shrinkage, porosity prediction
- niyama.py: Niyama criterion for centerline shrinkage prediction

Design Principles:
- Each module is independent and testable
- Common interfaces allow physics model swapping
- Vectorized NumPy/Numba operations for performance
- Type annotations throughout

Usage:
    from physics import HeatTransferModel, SolidificationModel, ShrinkageModel
    
    # Create physics models
    heat = HeatTransferModel(mesh, alloy)
    solidification = SolidificationModel(mesh, alloy)
    shrinkage = ShrinkageModel(mesh, alloy)
    
    # Run simulation
    for t in time_steps:
        heat.step(dt)
        solidification.update(heat.temperatures)
        shrinkage.calculate(solidification.fraction_solid)
"""

from physics.heat_transfer import (
    HeatTransferModel,
    BoundaryCondition,
    BoundaryType,
    TemperatureField,
)

from physics.solidification_physics import (
    SolidificationModel,
    PhaseState,
    FractionSolidCalculator,
)

from physics.shrinkage import (
    ShrinkageModel,
    ShrinkagePrediction,
    FeedingDistanceCalculator,
)

from physics.niyama import (
    NiyamaModel,
    NiyamaCriterion,
    ThermalGradientCalculator,
)

__all__ = [
    # Heat Transfer
    "HeatTransferModel",
    "BoundaryCondition", 
    "BoundaryType",
    "TemperatureField",
    
    # Solidification
    "SolidificationModel",
    "PhaseState",
    "FractionSolidCalculator",
    
    # Shrinkage
    "ShrinkageModel",
    "ShrinkagePrediction",
    "FeedingDistanceCalculator",
    
    # Niyama
    "NiyamaModel",
    "NiyamaCriterion",
    "ThermalGradientCalculator",
]
