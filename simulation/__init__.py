#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SatUpdate Simulation Package

This package provides all the numerical simulation components for
satellite constellation simulation, including orbital mechanics,
satellite dynamics, constellation generation, and logging.

The simulation can be run independently of any visualization.
"""

from .orbit import (
    EllipticalOrbit,
    EARTH_RADIUS_KM,
    EARTH_MASS_KG,
    G,
)

from .satellite import (
    Satellite,
    GeospatialPosition,
)

from .constellation import (
    ConstellationFactory,
    create_circular_orbit,
    create_random_orbit,
    create_random_constellation,
    create_walker_delta_constellation,
    create_walker_star_constellation,
    create_starlink_like_constellation,
    create_gps_like_constellation,
)

from .base_station import (
    BaseStation,
    BaseStationConfig,
)

from .simulation import (
    Simulation,
    SimulationConfig,
    SimulationState,
    AgentStatistics,
    ConstellationType,
    create_simulation,
)

from .logging import (
    SimulationLogger,
    SimulationLogHeader,
    TimestepRecord,
    RequestRecord,
    load_simulation_log,
    create_logger_from_simulation,
)


__all__ = [
    # Orbit
    "EllipticalOrbit",
    "EARTH_RADIUS_KM",
    "EARTH_MASS_KG",
    "G",
    
    # Satellite
    "Satellite",
    "GeospatialPosition",
    
    # Constellation
    "ConstellationFactory",
    "create_circular_orbit",
    "create_random_orbit",
    "create_random_constellation",
    "create_walker_delta_constellation",
    "create_walker_star_constellation",
    "create_starlink_like_constellation",
    "create_gps_like_constellation",
    
    # Base Station
    "BaseStation",
    "BaseStationConfig",
    
    # Simulation
    "Simulation",
    "SimulationConfig",
    "SimulationState",
    "AgentStatistics",
    "ConstellationType",
    "create_simulation",
    
    # Logging
    "SimulationLogger",
    "SimulationLogHeader",
    "TimestepRecord",
    "RequestRecord",
    "load_simulation_log",
    "create_logger_from_simulation",
]

__version__ = "1.0.0"