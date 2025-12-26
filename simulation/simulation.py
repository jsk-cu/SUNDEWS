#!/usr/bin/env python3
"""
Simulation Module

Main simulation class for satellite constellation software update distribution.
Provides a unified interface for running simulations with integrated agent-based
packet distribution protocol and optional logging.

The simulation can run independently of visualization for batch processing,
testing, or analysis.
"""

import math
from typing import List, Optional, Tuple, Dict, Any, Set, Type
from dataclasses import dataclass, field
from enum import Enum

from .orbit import EllipticalOrbit, EARTH_RADIUS_KM, EARTH_MASS_KG
from .satellite import Satellite, GeospatialPosition
from .constellation import (
    create_random_constellation,
    create_walker_delta_constellation,
    create_walker_star_constellation,
)
from .base_station import BaseStation, BaseStationConfig
from .logging import SimulationLogger, create_logger_from_simulation


class ConstellationType(Enum):
    """Supported constellation types."""

    RANDOM = "random"
    WALKER_DELTA = "walker_delta"
    WALKER_STAR = "walker_star"
    CUSTOM = "custom"


@dataclass
class SimulationConfig:
    """
    Configuration for a simulation.
    """

    constellation_type: ConstellationType = ConstellationType.WALKER_DELTA
    num_planes: int = 3
    sats_per_plane: int = 4
    num_satellites: int = 12
    altitude: float = 550.0
    inclination: float = math.radians(53)
    phasing_parameter: int = 1
    min_periapsis_altitude: float = 300.0
    max_periapsis_altitude: float = 2000.0
    max_apoapsis_altitude: float = 40000.0
    earth_radius: float = EARTH_RADIUS_KM
    earth_mass: float = EARTH_MASS_KG
    random_seed: Optional[int] = None
    communication_range: Optional[float] = None
    num_packets: int = 100
    agent_class: Optional[Type] = None
    base_station_latitude: float = 0.0
    base_station_longitude: float = 0.0
    base_station_altitude: float = 0.0
    base_station_range: float = 10000.0


@dataclass
class AgentStatistics:
    """
    Statistics about agent packet distribution.
    """

    total_packets: int = 0
    packets_per_agent: Dict[int, int] = field(default_factory=dict)
    completion_percentage: Dict[int, float] = field(default_factory=dict)
    fully_updated_count: int = 0
    average_completion: float = 0.0


@dataclass
class SimulationState:
    """
    Current simulation state.
    """

    time: float = 0.0
    step_count: int = 0
    satellite_positions: Dict[str, GeospatialPosition] = field(default_factory=dict)
    active_links: Set[Tuple[str, str]] = field(default_factory=set)
    base_station_links: Set[Tuple[str, str]] = field(default_factory=set)
    agent_statistics: AgentStatistics = field(default_factory=AgentStatistics)


class Simulation:
    """
    Main simulation class for satellite constellation updates.

    Encapsulates constellation, base stations, agents, and the packet
    distribution protocol. Can run independently of visualization.

    Parameters
    ----------
    config : SimulationConfig, optional
        Simulation configuration.
    enable_logging : bool
        Whether to enable simulation logging (default True).

    Attributes
    ----------
    config : SimulationConfig
        Current configuration.
    orbits : list
        Orbital elements for each orbit.
    satellites : list
        Satellites in the constellation.
    base_stations : list
        Ground stations.
    state : SimulationState
        Current simulation state.
    agents : dict
        Agent instances (agent_id -> Agent).
    logger : SimulationLogger
        Logger for capturing simulation events.
    """

    # Earth's sidereal rotation rate (rad/s)
    EARTH_ROTATION_RATE = 2 * math.pi / 86164.0905

    # Base station agent ID
    BASE_STATION_AGENT_ID = 0

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        enable_logging: bool = False
    ):
        self.config = config or SimulationConfig()
        self.orbits: List[EllipticalOrbit] = []
        self.satellites: List[Satellite] = []
        self.base_stations: List[BaseStation] = []
        self.state = SimulationState()
        self.earth_rotation_rate = self.EARTH_ROTATION_RATE

        # Agent system
        self.agents: Dict[int, Any] = {}
        self.satellite_id_to_agent_id: Dict[str, int] = {}
        self.agent_id_to_satellite_id: Dict[int, str] = {}
        self.base_station_agent_id = self.BASE_STATION_AGENT_ID

        # Logging
        self.logger = SimulationLogger(enabled=enable_logging)
        self._timestep: float = 60.0  # Default timestep for logging header

        self._initialized = False

    def initialize(self, timestep: float = 60.0) -> None:
        """
        Initialize simulation by creating constellation, base stations, and agents.

        Parameters
        ----------
        timestep : float
            The timestep that will be used for simulation steps.
            Used for logging header metadata.
        """
        self._timestep = timestep
        self._create_constellation()
        self._create_base_stations()
        self._create_agents()
        self._update_state()
        self._update_active_links()
        self._update_base_station_links()
        self._update_agent_statistics()
        
        # Initialize logging
        if self.logger.enabled:
            agent_type = "unknown"
            if self.config.agent_class is not None:
                agent_type = getattr(self.config.agent_class, 'name', 'custom')
            else:
                agent_type = "min"  # default agent
            
            self.logger.set_header_from_config(
                self.config,
                agent_type=agent_type,
                timestep=timestep
            )
            
            # Record initial state (step 0, before any simulation)
            self._record_timestep_log()
        
        self._initialized = True

    def _create_constellation(self) -> None:
        """Create satellite constellation based on configuration."""
        config = self.config

        if config.constellation_type == ConstellationType.RANDOM:
            self.orbits, self.satellites = create_random_constellation(
                num_satellites=config.num_satellites,
                min_periapsis_altitude=config.min_periapsis_altitude,
                max_periapsis_altitude=config.max_periapsis_altitude,
                max_apoapsis_altitude=config.max_apoapsis_altitude,
                earth_radius=config.earth_radius,
                earth_mass=config.earth_mass,
                seed=config.random_seed,
            )

        elif config.constellation_type == ConstellationType.WALKER_DELTA:
            self.orbits, self.satellites = create_walker_delta_constellation(
                num_planes=config.num_planes,
                sats_per_plane=config.sats_per_plane,
                altitude=config.altitude,
                inclination=config.inclination,
                phasing_parameter=config.phasing_parameter,
                earth_radius=config.earth_radius,
                earth_mass=config.earth_mass,
            )

        elif config.constellation_type == ConstellationType.WALKER_STAR:
            self.orbits, self.satellites = create_walker_star_constellation(
                num_planes=config.num_planes,
                sats_per_plane=config.sats_per_plane,
                altitude=config.altitude,
                inclination=config.inclination,
                phasing_parameter=config.phasing_parameter,
                earth_radius=config.earth_radius,
                earth_mass=config.earth_mass,
            )

        elif config.constellation_type == ConstellationType.CUSTOM:
            pass  # Orbits and satellites set externally

    def _create_base_stations(self) -> None:
        """Create base stations from configuration."""
        config = self.config
        base_station = BaseStation.at_coordinates(
            latitude_deg=config.base_station_latitude,
            longitude_deg=config.base_station_longitude,
            altitude=config.base_station_altitude,
            communication_range=config.base_station_range,
            name="BASE-1",
        )
        self.base_stations = [base_station]

    def _create_agents(self) -> None:
        """Create agents for base station and all satellites."""
        if self.config.agent_class is not None:
            AgentClass = self.config.agent_class
        else:
            from agents import Agent

            AgentClass = Agent

        num_satellites = len(self.satellites)
        num_packets = self.config.num_packets

        self.agents = {}
        self.satellite_id_to_agent_id = {}
        self.agent_id_to_satellite_id = {}

        # Base station agent (ID 0)
        self.agents[self.BASE_STATION_AGENT_ID] = AgentClass(
            agent_id=self.BASE_STATION_AGENT_ID,
            num_packets=num_packets,
            num_satellites=num_satellites,
            is_base_station=True,
        )

        # Satellite agents (IDs 1 to N)
        for idx, satellite in enumerate(self.satellites):
            agent_id = idx + 1

            self.agents[agent_id] = AgentClass(
                agent_id=agent_id,
                num_packets=num_packets,
                num_satellites=num_satellites,
                is_base_station=False,
            )

            self.satellite_id_to_agent_id[satellite.satellite_id] = agent_id
            self.agent_id_to_satellite_id[agent_id] = satellite.satellite_id

    def _get_agent_neighbors(self, agent_id: int) -> Set[int]:
        """Get neighbor agent IDs for a given agent."""
        neighbors = set()

        if agent_id == self.BASE_STATION_AGENT_ID:
            # Base station's neighbors are satellites it can communicate with
            for bs_name, sat_id in self.state.base_station_links:
                if sat_id in self.satellite_id_to_agent_id:
                    neighbors.add(self.satellite_id_to_agent_id[sat_id])
        else:
            # Satellite's neighbors
            sat_id = self.agent_id_to_satellite_id.get(agent_id)
            if sat_id is None:
                return neighbors

            # Other satellites with active links
            for id1, id2 in self.state.active_links:
                if id1 == sat_id:
                    if id2 in self.satellite_id_to_agent_id:
                        neighbors.add(self.satellite_id_to_agent_id[id2])
                elif id2 == sat_id:
                    if id1 in self.satellite_id_to_agent_id:
                        neighbors.add(self.satellite_id_to_agent_id[id1])

            # Base station if in range
            for bs_name, linked_sat_id in self.state.base_station_links:
                if linked_sat_id == sat_id:
                    neighbors.add(self.BASE_STATION_AGENT_ID)
                    break

        return neighbors

    def _agent_id_to_entity_id(self, agent_id: int) -> str:
        """Convert agent ID to entity ID (satellite ID or base station name)."""
        if agent_id == self.BASE_STATION_AGENT_ID:
            return self.base_stations[0].name if self.base_stations else "BASE-1"
        return self.agent_id_to_satellite_id.get(agent_id, f"AGENT-{agent_id}")

    def _run_agent_protocol(self) -> List[Tuple[str, str, int, bool]]:
        """
        Execute the 4-phase agent communication protocol.

        Returns
        -------
        List[Tuple[str, str, int, bool]]
            List of request records: (requester_id, requestee_id, packet_idx, successful)
        """
        request_records: List[Tuple[str, str, int, bool]] = []

        # Phase 1: Broadcast
        broadcasts: Dict[int, Any] = {}
        for agent_id, agent in self.agents.items():
            broadcasts[agent_id] = agent.broadcast_state()

        # Phase 2: Request
        all_requests: Dict[int, Dict[int, int]] = {}
        for agent_id, agent in self.agents.items():
            neighbors = self._get_agent_neighbors(agent_id)
            neighbor_broadcasts = {
                neighbor_id: broadcasts[neighbor_id]
                for neighbor_id in neighbors
                if neighbor_id in broadcasts
            }
            all_requests[agent_id] = agent.make_requests(neighbor_broadcasts)

        # Phase 3: Respond
        requests_to_agent: Dict[int, Dict[int, int]] = {
            agent_id: {} for agent_id in self.agents
        }
        for requester_id, requests in all_requests.items():
            for requestee_id, packet_idx in requests.items():
                if requestee_id in requests_to_agent:
                    requests_to_agent[requestee_id][requester_id] = packet_idx

        responses: Dict[int, Dict[int, Optional[int]]] = {}
        for agent_id, agent in self.agents.items():
            responses[agent_id] = agent.receive_requests_and_update(
                requests_to_agent[agent_id]
            )

        # Phase 4: Receive and record requests
        packets_to_agent: Dict[int, Dict[int, Optional[int]]] = {
            agent_id: {} for agent_id in self.agents
        }
        for requester_id, requests in all_requests.items():
            for requestee_id, requested_packet in requests.items():
                if requestee_id in responses:
                    sent_packet = responses[requestee_id].get(requester_id)
                    packets_to_agent[requester_id][requestee_id] = sent_packet
                    
                    # Record the request
                    requester_entity = self._agent_id_to_entity_id(requester_id)
                    requestee_entity = self._agent_id_to_entity_id(requestee_id)
                    was_successful = sent_packet is not None
                    request_records.append((
                        requester_entity,
                        requestee_entity,
                        requested_packet,
                        was_successful
                    ))

        for agent_id, agent in self.agents.items():
            agent.receive_packets_and_update(packets_to_agent[agent_id])

        return request_records

    def _update_state(self) -> None:
        """Update satellite positions in state."""
        earth_rotation = self.state.time * self.earth_rotation_rate

        self.state.satellite_positions = {
            sat.satellite_id: sat.get_geospatial_position(earth_rotation)
            for sat in self.satellites
        }

    def _update_active_links(self) -> None:
        """Update active inter-satellite communication links."""
        active_links: Set[Tuple[str, str]] = set()
        comm_range = self.config.communication_range
        n = len(self.satellites)

        for i in range(n):
            for j in range(i + 1, n):
                sat_i = self.satellites[i]
                sat_j = self.satellites[j]

                if not sat_i.has_line_of_sight(sat_j):
                    continue

                if comm_range is not None:
                    if sat_i.distance_to(sat_j) > comm_range:
                        continue

                # Consistent ordering
                id_i, id_j = sat_i.satellite_id, sat_j.satellite_id
                if id_i < id_j:
                    active_links.add((id_i, id_j))
                else:
                    active_links.add((id_j, id_i))

        self.state.active_links = active_links

    def _update_base_station_links(self) -> None:
        """Update active base station to satellite links."""
        base_station_links: Set[Tuple[str, str]] = set()
        earth_rotation = self.state.time * self.earth_rotation_rate

        for base_station in self.base_stations:
            for satellite in self.satellites:
                if base_station.can_communicate(satellite, earth_rotation):
                    base_station_links.add((base_station.name, satellite.satellite_id))

        self.state.base_station_links = base_station_links

    def _update_agent_statistics(self) -> None:
        """Update agent packet distribution statistics."""
        stats = AgentStatistics()
        stats.total_packets = self.config.num_packets

        satellite_completions = []
        fully_updated = 0

        for agent_id, agent in self.agents.items():
            packet_count = agent.get_packet_count()
            completion = agent.get_completion_percentage()

            stats.packets_per_agent[agent_id] = packet_count
            stats.completion_percentage[agent_id] = completion

            if agent_id != self.BASE_STATION_AGENT_ID:
                satellite_completions.append(completion)
                if agent.has_all_packets():
                    fully_updated += 1

        stats.fully_updated_count = fully_updated
        if satellite_completions:
            stats.average_completion = sum(satellite_completions) / len(
                satellite_completions
            )

        self.state.agent_statistics = stats

    def _record_timestep_log(self, request_records: Optional[List[Tuple[str, str, int, bool]]] = None) -> None:
        """Record the current timestep to the log."""
        if not self.logger.enabled:
            return
        
        # Start timestep record
        self.logger.start_timestep(
            step=self.state.step_count,
            time=self.state.time
        )
        
        # Record packet counts for satellites only
        packet_counts = {}
        for satellite in self.satellites:
            agent_id = self.satellite_id_to_agent_id.get(satellite.satellite_id)
            if agent_id is not None and agent_id in self.agents:
                packet_counts[satellite.satellite_id] = self.agents[agent_id].get_packet_count()
        
        self.logger.record_packet_counts(packet_counts)
        
        # Record communication pairs (inter-satellite links only)
        self.logger.record_communication_pairs_from_set(self.state.active_links)
        
        # Record requests if provided
        if request_records:
            self.logger.record_requests_batch(request_records)
        
        # Finalize timestep
        self.logger.end_timestep()

    def step(self, timestep: float) -> SimulationState:
        """
        Advance simulation by one timestep.

        Parameters
        ----------
        timestep : float
            Time to advance (seconds).

        Returns
        -------
        SimulationState
            Updated simulation state.

        Raises
        ------
        RuntimeError
            If simulation not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Simulation not initialized. Call initialize() first.")

        for satellite in self.satellites:
            satellite.step(timestep)

        self.state.time += timestep
        self.state.step_count += 1

        self._update_state()
        self._update_active_links()
        self._update_base_station_links()
        
        # Run agent protocol and capture request records
        request_records = self._run_agent_protocol()
        
        self._update_agent_statistics()
        
        # Record to log
        self._record_timestep_log(request_records)

        return self.state

    def run(self, duration: float, timestep: float) -> List[SimulationState]:
        """
        Run simulation for specified duration.

        Parameters
        ----------
        duration : float
            Total simulation time (seconds).
        timestep : float
            Time step (seconds).

        Returns
        -------
        list
            List of states at each timestep.
        """
        if not self._initialized:
            self.initialize(timestep=timestep)

        states = []
        elapsed = 0.0

        while elapsed < duration:
            self.step(timestep)
            states.append(
                SimulationState(
                    time=self.state.time,
                    step_count=self.state.step_count,
                    satellite_positions=dict(self.state.satellite_positions),
                    active_links=set(self.state.active_links),
                    base_station_links=set(self.state.base_station_links),
                    agent_statistics=AgentStatistics(
                        total_packets=self.state.agent_statistics.total_packets,
                        packets_per_agent=dict(
                            self.state.agent_statistics.packets_per_agent
                        ),
                        completion_percentage=dict(
                            self.state.agent_statistics.completion_percentage
                        ),
                        fully_updated_count=self.state.agent_statistics.fully_updated_count,
                        average_completion=self.state.agent_statistics.average_completion,
                    ),
                )
            )
            elapsed += timestep

        return states

    def reset(self) -> None:
        """Reset simulation to initial state."""
        self.state = SimulationState()
        self.logger.reset()
        self._create_constellation()
        self._create_base_stations()
        self._create_agents()
        self._update_state()
        self._update_active_links()
        self._update_base_station_links()
        self._update_agent_statistics()

    def regenerate(self, new_seed: Optional[int] = None) -> None:
        """
        Regenerate constellation with new random seed.
        """
        if new_seed is not None:
            self.config.random_seed = new_seed
        else:
            import random
            self.config.random_seed = random.randint(0, 2**31)

        self.reset()

    def set_custom_constellation(
        self, orbits: List[EllipticalOrbit], satellites: List[Satellite]
    ) -> None:
        """Set a custom constellation."""
        self.config.constellation_type = ConstellationType.CUSTOM
        self.orbits = orbits
        self.satellites = satellites
        self.state = SimulationState()
        self._create_base_stations()
        self._create_agents()
        self._update_state()
        self._update_active_links()
        self._update_base_station_links()
        self._update_agent_statistics()
        self._initialized = True

    def get_satellite(self, satellite_id: str) -> Optional[Satellite]:
        """Get satellite by ID."""
        for sat in self.satellites:
            if sat.satellite_id == satellite_id:
                return sat
        return None

    def get_agent(self, agent_id: int) -> Optional[Any]:
        """Get agent by ID."""
        return self.agents.get(agent_id)

    def get_satellite_agent(self, satellite_id: str) -> Optional[Any]:
        """Get agent for a specific satellite."""
        agent_id = self.satellite_id_to_agent_id.get(satellite_id)
        if agent_id is not None:
            return self.agents.get(agent_id)
        return None

    def get_base_station_agent(self) -> Optional[Any]:
        """Get the base station's agent."""
        return self.agents.get(self.BASE_STATION_AGENT_ID)

    def get_inter_satellite_distances(self) -> Dict[Tuple[str, str], float]:
        """Calculate distances between all satellite pairs."""
        distances = {}
        n = len(self.satellites)

        for i in range(n):
            for j in range(i + 1, n):
                sat_i = self.satellites[i]
                sat_j = self.satellites[j]
                distances[(sat_i.satellite_id, sat_j.satellite_id)] = sat_i.distance_to(
                    sat_j
                )

        return distances

    def get_line_of_sight_matrix(self) -> Dict[Tuple[str, str], bool]:
        """Calculate line-of-sight status for all satellite pairs."""
        los_matrix = {}
        n = len(self.satellites)

        for i in range(n):
            for j in range(i + 1, n):
                sat_i = self.satellites[i]
                sat_j = self.satellites[j]
                los_matrix[
                    (sat_i.satellite_id, sat_j.satellite_id)
                ] = sat_i.has_line_of_sight(sat_j)

        return los_matrix

    def is_update_complete(self) -> bool:
        """Check if all satellites have received all packets."""
        return self.state.agent_statistics.fully_updated_count == len(self.satellites)

    def save_log(self, filepath: str, indent: int = 2) -> None:
        """
        Save the simulation log to a JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to output file
        indent : int
            JSON indentation level
        """
        self.logger.save(filepath, indent=indent)

    def get_log(self) -> Dict[str, Any]:
        """
        Get the simulation log as a dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Log with 'header' and 'time_series' keys
        """
        return self.logger.get_log()

    @property
    def num_satellites(self) -> int:
        """Number of satellites."""
        return len(self.satellites)

    @property
    def num_orbits(self) -> int:
        """Number of unique orbits."""
        return len(self.orbits)

    @property
    def simulation_time(self) -> float:
        """Current simulation time (seconds)."""
        return self.state.time

    def get_summary(self) -> Dict[str, Any]:
        """Get simulation summary."""
        return {
            "constellation_type": self.config.constellation_type.value,
            "num_satellites": self.num_satellites,
            "num_orbits": self.num_orbits,
            "num_base_stations": len(self.base_stations),
            "num_packets": self.config.num_packets,
            "simulation_time": self.state.time,
            "step_count": self.state.step_count,
            "average_completion": self.state.agent_statistics.average_completion,
            "fully_updated_satellites": self.state.agent_statistics.fully_updated_count,
            "update_complete": self.is_update_complete(),
            "initialized": self._initialized,
        }

    def __repr__(self) -> str:
        return (
            f"Simulation(\n"
            f"  type={self.config.constellation_type.value},\n"
            f"  satellites={self.num_satellites},\n"
            f"  orbits={self.num_orbits},\n"
            f"  base_stations={len(self.base_stations)},\n"
            f"  packets={self.config.num_packets},\n"
            f"  time={self.state.time:.2f}s,\n"
            f"  steps={self.state.step_count},\n"
            f"  avg_completion={self.state.agent_statistics.average_completion:.1f}%\n"
            f")"
        )


def create_simulation(constellation_type: str = "walker_delta", **kwargs) -> Simulation:
    """
    Create simulation with specified constellation type.
    """
    type_map = {
        "random": ConstellationType.RANDOM,
        "walker_delta": ConstellationType.WALKER_DELTA,
        "walker_star": ConstellationType.WALKER_STAR,
    }

    if constellation_type not in type_map:
        raise ValueError(f"Unknown constellation type: {constellation_type}")

    config = SimulationConfig(constellation_type=type_map[constellation_type])

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return Simulation(config)


if __name__ == "__main__":
    print("Simulation Demo with Logging")
    print("=" * 60)

    config = SimulationConfig(
        constellation_type=ConstellationType.WALKER_DELTA,
        num_planes=3,
        sats_per_plane=4,
        altitude=550,
        inclination=math.radians(53),
        num_packets=50,
    )

    sim = Simulation(config, enable_logging=True)
    sim.initialize(timestep=60.0)

    print(f"\nCreated simulation: {sim}")

    print(f"\nRunning for 10 minutes...")
    for _ in range(10):
        sim.step(60)

    stats = sim.state.agent_statistics
    print(f"\nAfter 10 minutes:")
    print(f"  Average completion: {stats.average_completion:.1f}%")
    print(f"  Fully updated: {stats.fully_updated_count}/{len(sim.satellites)}")
    print(f"  Log timesteps: {sim.logger.num_timesteps}")
    
    # Save log
    sim.save_log("/tmp/simulation_log.json")
    print(f"\nLog saved to /tmp/simulation_log.json")