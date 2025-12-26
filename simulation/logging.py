#!/usr/bin/env python3
"""
Simulation Logging Module

Provides logging functionality for capturing simulation state and events
in a structured JSON format for analysis and reproducibility.

The log format consists of:
- header: Configuration and metadata for simulation reproducibility
- time_series: List of timestep records with state and events

Each timestep record contains:
1. packet_counts: {satellite_id: num_packets} for all satellites
2. communication_pairs: [(sat_a, sat_b), ...] active links
3. requests: [(requester, requestee, packet_idx, was_successful), ...]
"""

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Set
from pathlib import Path


@dataclass
class RequestRecord:
    """
    Record of a single packet request.
    
    Attributes
    ----------
    requester_id : str
        ID of the agent/satellite making the request
    requestee_id : str
        ID of the agent/satellite being asked
    packet_idx : int
        Index of the packet being requested
    was_successful : bool
        Whether the request was fulfilled
    """
    requester_id: str
    requestee_id: str
    packet_idx: int
    was_successful: bool
    
    def to_tuple(self) -> Tuple[str, str, int, bool]:
        """Convert to tuple format for JSON serialization."""
        return (self.requester_id, self.requestee_id, self.packet_idx, self.was_successful)


@dataclass
class TimestepRecord:
    """
    Record of simulation state at a single timestep.
    
    Attributes
    ----------
    step : int
        The timestep number (0-indexed)
    time : float
        Simulation time in seconds
    packet_counts : Dict[str, int]
        Packet count per satellite {satellite_id: num_packets}
    communication_pairs : List[Tuple[str, str]]
        Active inter-satellite links [(sat_a, sat_b), ...]
    requests : List[Tuple[str, str, int, bool]]
        Request records [(requester, requestee, packet_idx, success), ...]
    """
    step: int
    time: float
    packet_counts: Dict[str, int] = field(default_factory=dict)
    communication_pairs: List[Tuple[str, str]] = field(default_factory=list)
    requests: List[Tuple[str, str, int, bool]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "step": self.step,
            "time": self.time,
            "packet_counts": self.packet_counts,
            "communication_pairs": self.communication_pairs,
            "requests": self.requests
        }


@dataclass
class SimulationLogHeader:
    """
    Header containing all information needed to reproduce a simulation.
    
    Attributes
    ----------
    constellation_type : str
        Type of constellation (walker_delta, walker_star, random)
    num_planes : int
        Number of orbital planes (Walker constellations)
    sats_per_plane : int
        Satellites per plane (Walker constellations)
    num_satellites : int
        Total number of satellites
    altitude : float
        Orbital altitude in km
    inclination : float
        Orbital inclination in radians
    phasing_parameter : int
        Walker phasing parameter F
    random_seed : Optional[int]
        Random seed for reproducibility
    communication_range : Optional[float]
        Inter-satellite communication range in km
    num_packets : int
        Number of packets in the software update
    agent_type : str
        Type of agent controller used
    base_station_latitude : float
        Base station latitude in degrees
    base_station_longitude : float
        Base station longitude in degrees
    base_station_altitude : float
        Base station altitude in km
    base_station_range : float
        Base station communication range in km
    timestep : float
        Simulation timestep in seconds
    earth_radius : float
        Earth radius in km
    earth_mass : float
        Earth mass in kg
    created_at : str
        ISO format timestamp of log creation
    version : str
        Log format version
    """
    constellation_type: str
    num_planes: int
    sats_per_plane: int
    num_satellites: int
    altitude: float
    inclination: float
    phasing_parameter: int
    random_seed: Optional[int]
    communication_range: Optional[float]
    num_packets: int
    agent_type: str
    base_station_latitude: float
    base_station_longitude: float
    base_station_altitude: float
    base_station_range: float
    timestep: float
    earth_radius: float
    earth_mass: float
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class SimulationLogger:
    """
    Logger for capturing simulation state and events.
    
    Captures packet distribution state, communication topology, and
    request/response events at each timestep for later analysis.
    
    Parameters
    ----------
    enabled : bool
        Whether logging is enabled (default True)
    
    Attributes
    ----------
    header : SimulationLogHeader
        Configuration metadata
    time_series : List[TimestepRecord]
        Recorded timestep data
    enabled : bool
        Whether logging is active
    
    Example
    -------
    >>> logger = SimulationLogger()
    >>> logger.set_header_from_config(config, agent_type="min")
    >>> 
    >>> # During simulation step:
    >>> logger.start_timestep(step=0, time=0.0)
    >>> logger.record_packet_counts({"SAT-1": 5, "SAT-2": 10})
    >>> logger.record_communication_pairs([("SAT-1", "SAT-2")])
    >>> logger.record_request("SAT-1", "SAT-2", packet_idx=3, successful=True)
    >>> logger.end_timestep()
    >>> 
    >>> # Export
    >>> logger.save("simulation_log.json")
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.header: Optional[SimulationLogHeader] = None
        self.time_series: List[TimestepRecord] = []
        self._current_record: Optional[TimestepRecord] = None
        self._pending_requests: List[Tuple[str, str, int, bool]] = []
    
    def set_header(self, header: SimulationLogHeader) -> None:
        """
        Set the log header directly.
        
        Parameters
        ----------
        header : SimulationLogHeader
            The header containing simulation configuration
        """
        if not self.enabled:
            return
        self.header = header
    
    def set_header_from_config(
        self,
        config: Any,
        agent_type: str = "unknown",
        timestep: float = 60.0
    ) -> None:
        """
        Create and set header from a SimulationConfig object.
        
        Parameters
        ----------
        config : SimulationConfig
            The simulation configuration
        agent_type : str
            Name of the agent controller type
        timestep : float
            Simulation timestep in seconds
        """
        if not self.enabled:
            return
        
        self.header = SimulationLogHeader(
            constellation_type=config.constellation_type.value,
            num_planes=config.num_planes,
            sats_per_plane=config.sats_per_plane,
            num_satellites=config.num_satellites,
            altitude=config.altitude,
            inclination=config.inclination,
            phasing_parameter=config.phasing_parameter,
            random_seed=config.random_seed,
            communication_range=config.communication_range,
            num_packets=config.num_packets,
            agent_type=agent_type,
            base_station_latitude=config.base_station_latitude,
            base_station_longitude=config.base_station_longitude,
            base_station_altitude=config.base_station_altitude,
            base_station_range=config.base_station_range,
            timestep=timestep,
            earth_radius=config.earth_radius,
            earth_mass=config.earth_mass,
        )
    
    def start_timestep(self, step: int, time: float) -> None:
        """
        Begin recording a new timestep.
        
        Parameters
        ----------
        step : int
            The timestep number (0-indexed)
        time : float
            Current simulation time in seconds
        """
        if not self.enabled:
            return
        
        self._current_record = TimestepRecord(step=step, time=time)
        self._pending_requests = []
    
    def record_packet_counts(self, counts: Dict[str, int]) -> None:
        """
        Record packet counts for all satellites.
        
        Parameters
        ----------
        counts : Dict[str, int]
            Mapping of satellite_id -> num_packets
        """
        if not self.enabled or self._current_record is None:
            return
        
        self._current_record.packet_counts = dict(counts)
    
    def record_communication_pairs(self, pairs: List[Tuple[str, str]]) -> None:
        """
        Record active inter-satellite communication links.
        
        Parameters
        ----------
        pairs : List[Tuple[str, str]]
            List of (satellite_a_id, satellite_b_id) tuples
        """
        if not self.enabled or self._current_record is None:
            return
        
        self._current_record.communication_pairs = list(pairs)
    
    def record_communication_pairs_from_set(self, pairs: Set[Tuple[str, str]]) -> None:
        """
        Record active inter-satellite communication links from a set.
        
        Parameters
        ----------
        pairs : Set[Tuple[str, str]]
            Set of (satellite_a_id, satellite_b_id) tuples
        """
        if not self.enabled or self._current_record is None:
            return
        
        # Sort for deterministic output
        sorted_pairs = sorted(pairs, key=lambda x: (x[0], x[1]))
        self._current_record.communication_pairs = sorted_pairs
    
    def record_request(
        self,
        requester_id: str,
        requestee_id: str,
        packet_idx: int,
        successful: bool
    ) -> None:
        """
        Record a single packet request.
        
        Parameters
        ----------
        requester_id : str
            ID of the requesting satellite/agent
        requestee_id : str
            ID of the satellite/agent being asked
        packet_idx : int
            Index of the packet requested
        successful : bool
            Whether the request was fulfilled
        """
        if not self.enabled:
            return
        
        self._pending_requests.append((requester_id, requestee_id, packet_idx, successful))
    
    def record_requests_batch(
        self,
        requests: List[Tuple[str, str, int, bool]]
    ) -> None:
        """
        Record multiple requests at once.
        
        Parameters
        ----------
        requests : List[Tuple[str, str, int, bool]]
            List of (requester_id, requestee_id, packet_idx, successful) tuples
        """
        if not self.enabled:
            return
        
        self._pending_requests.extend(requests)
    
    def end_timestep(self) -> None:
        """
        Finalize the current timestep record and add to time series.
        """
        if not self.enabled or self._current_record is None:
            return
        
        # Sort requests for deterministic output
        sorted_requests = sorted(
            self._pending_requests,
            key=lambda x: (x[0], x[1], x[2])
        )
        self._current_record.requests = sorted_requests
        
        self.time_series.append(self._current_record)
        self._current_record = None
        self._pending_requests = []
    
    def get_log(self) -> Dict[str, Any]:
        """
        Get the complete log as a dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with 'header' and 'time_series' keys
        """
        return {
            "header": self.header.to_dict() if self.header else {},
            "time_series": [record.to_dict() for record in self.time_series]
        }
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert the log to a JSON string.
        
        Parameters
        ----------
        indent : int
            JSON indentation level (default 2)
        
        Returns
        -------
        str
            JSON string representation
        """
        return json.dumps(self.get_log(), indent=indent)
    
    def save(self, filepath: str, indent: int = 2) -> None:
        """
        Save the log to a JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to output file
        indent : int
            JSON indentation level (default 2)
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.get_log(), f, indent=indent)
    
    def reset(self) -> None:
        """
        Clear all recorded data while preserving the header.
        """
        self.time_series = []
        self._current_record = None
        self._pending_requests = []
    
    def clear(self) -> None:
        """
        Clear all data including the header.
        """
        self.header = None
        self.reset()
    
    @property
    def num_timesteps(self) -> int:
        """Number of recorded timesteps."""
        return len(self.time_series)
    
    def __repr__(self) -> str:
        header_info = self.header.constellation_type if self.header else "no header"
        return f"SimulationLogger(timesteps={self.num_timesteps}, header={header_info})"


def load_simulation_log(filepath: str) -> Dict[str, Any]:
    """
    Load a simulation log from a JSON file.
    
    Parameters
    ----------
    filepath : str
        Path to the log file
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with 'header' and 'time_series' keys
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def create_logger_from_simulation(
    simulation: Any,
    agent_type: str = "unknown",
    timestep: float = 60.0,
    enabled: bool = True
) -> SimulationLogger:
    """
    Create a logger pre-configured from a Simulation object.
    
    Parameters
    ----------
    simulation : Simulation
        The simulation to log
    agent_type : str
        Name of the agent controller type
    timestep : float
        Simulation timestep in seconds
    enabled : bool
        Whether logging is enabled
    
    Returns
    -------
    SimulationLogger
        Configured logger instance
    """
    logger = SimulationLogger(enabled=enabled)
    logger.set_header_from_config(simulation.config, agent_type, timestep)
    return logger