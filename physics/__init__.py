"""
Physics Engine Package for Casting Solidification Simulation

Provides modular physics components that can be swapped or extended:
- heat_transfer.py: Thermal conductivity, diffusion, boundary conditions
- solidification_physics.py: Phase change, latent heat, fraction solid
- gas_porosity.py: Gas porosity risk modeling

Design Principles:
- Each module is independent and testable
- Common interfaces allow physics model swapping
- Vectorized NumPy/Numba operations for performance
- Type annotations throughout

Usage:
    from physics import HeatTransferModel, SolidificationModel

    heat = HeatTransferModel(mesh, alloy)
    solidification = SolidificationModel(mesh, alloy)

    for t in time_steps:
        heat.step(dt)
        solidification.update(heat.temperatures)
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
]  
