#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SatUpdate - Satellite Constellation Simulation and Visualization

A framework for simulating and visualizing satellite constellations,
with support for Walker-Delta, Walker-Star, and random constellation patterns.

The package is divided into two main subpackages:
- simulation: Numerical simulation of orbital mechanics and satellites
- visualization: Pygame-based 3D visualization (optional)

Example usage:

    # Simulation only (no visualization)
    from SatUpdate.simulation import Simulation, SimulationConfig, ConstellationType
    
    config = SimulationConfig(
        constellation_type=ConstellationType.WALKER_DELTA,
        num_planes=3,
        sats_per_plane=4,
        altitude=550
    )
    sim = Simulation(config)
    sim.initialize()
    sim.step(60)  # Step 60 seconds
    
    # Save simulation log
    sim.save_log("simulation_log.json")
"""

__version__ = "1.0.0"
__author__ = "Jonathan"

# Re-export commonly used items for convenience
from .simulation import (
    Simulation,
    SimulationConfig,
    ConstellationType,
    Satellite,
    EllipticalOrbit,
    EARTH_RADIUS_KM,
    EARTH_MASS_KG,
    SimulationLogger,
    load_simulation_log,
)

__all__ = [
    # Main classes
    "Simulation",
    "SimulationConfig",
    "ConstellationType",
    "Satellite",
    "EllipticalOrbit",
    
    # Constants
    "EARTH_RADIUS_KM",
    "EARTH_MASS_KG",
    
    # Logging
    "SimulationLogger",
    "load_simulation_log",
    
    # Subpackages
    "simulation",
    "visualization",
]