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

# Step 1: TrajectoryProvider interface
from .trajectory import (
    TrajectoryProvider,
    TrajectoryState,
    KeplerianProvider,
    create_keplerian_provider,
)

# Step 2: SPICE Provider (optional dependency)
from .spice_provider import (
    SpiceProvider,
    SpiceKernelSet,
    SpiceConstellationConfig,
    SpiceDatasetLoader,
    create_spice_provider,
    is_spice_available,
    SPICE_AVAILABLE,
)

# Step 4: Network Backend Interface
from .network_backend import (
    NetworkBackend,
    NativeNetworkBackend,
    DelayedNetworkBackend,
    PacketTransfer,
    NetworkStatistics,
    DropReason,
    PendingTransfer,
    create_native_backend,
    create_delayed_backend,
)

# Step 5: NS-3 Backend (File Mode)
from .ns3_backend import (
    NS3Backend,
    NS3Config,
    NS3Mode,
    NS3ErrorModel,
    NS3PropagationModel,
    NS3Node,
    NS3SendCommand,
    create_ns3_backend,
    check_ns3_available,
    is_ns3_available,
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
    
    # Trajectory (Step 1)
    "TrajectoryProvider",
    "TrajectoryState",
    "KeplerianProvider",
    "create_keplerian_provider",
    
    # SPICE Provider (Step 2)
    "SpiceProvider",
    "SpiceKernelSet",
    "SpiceConstellationConfig",
    "SpiceDatasetLoader",
    "create_spice_provider",
    "is_spice_available",
    "SPICE_AVAILABLE",
    
    # Network Backend (Step 4)
    "NetworkBackend",
    "NativeNetworkBackend",
    "DelayedNetworkBackend",
    "PacketTransfer",
    "NetworkStatistics",
    "DropReason",
    "PendingTransfer",
    "create_native_backend",
    "create_delayed_backend",
    
    # NS-3 Backend (Step 5)
    "NS3Backend",
    "NS3Config",
    "NS3Mode",
    "NS3ErrorModel",
    "NS3PropagationModel",
    "NS3Node",
    "NS3SendCommand",
    "create_ns3_backend",
    "check_ns3_available",
    "is_ns3_available",
]

__version__ = "1.0.0"