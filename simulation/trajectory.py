#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory Provider Module

Provides an abstract interface for satellite position computation that
decouples trajectory calculation from the core simulation. This enables
pluggable trajectory sources (Keplerian, SPICE, etc.) without changing
core simulation logic.

This module implements Step 1 of the NS-3/SPICE integration plan.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from .satellite import Satellite


@dataclass
class TrajectoryState:
    """
    State vector for a satellite at a specific time.
    
    Contains position and velocity in Earth-Centered Inertial (ECI)
    coordinates, along with epoch and reference frame information.
    
    Attributes
    ----------
    position_eci : np.ndarray
        Position vector [x, y, z] in kilometers in ECI frame
    velocity_eci : np.ndarray
        Velocity vector [vx, vy, vz] in km/s in ECI frame
    epoch : datetime
        The time at which this state is valid
    reference_frame : str
        Reference frame identifier (default "J2000")
    
    Examples
    --------
    >>> state = TrajectoryState(
    ...     position_eci=np.array([7000.0, 0.0, 0.0]),
    ...     velocity_eci=np.array([0.0, 7.5, 0.0]),
    ...     epoch=datetime(2025, 1, 1, 0, 0, 0)
    ... )
    >>> state.position_eci.shape
    (3,)
    """
    position_eci: np.ndarray
    velocity_eci: np.ndarray
    epoch: datetime
    reference_frame: str = "J2000"
    
    def __post_init__(self):
        """Validate state vector dimensions."""
        if not isinstance(self.position_eci, np.ndarray):
            self.position_eci = np.array(self.position_eci, dtype=float)
        if not isinstance(self.velocity_eci, np.ndarray):
            self.velocity_eci = np.array(self.velocity_eci, dtype=float)
        
        if self.position_eci.shape != (3,):
            raise ValueError(
                f"position_eci must have shape (3,), got {self.position_eci.shape}"
            )
        if self.velocity_eci.shape != (3,):
            raise ValueError(
                f"velocity_eci must have shape (3,), got {self.velocity_eci.shape}"
            )
    
    @property
    def speed(self) -> float:
        """Orbital speed in km/s."""
        return float(np.linalg.norm(self.velocity_eci))
    
    @property
    def radius(self) -> float:
        """Distance from Earth's center in km."""
        return float(np.linalg.norm(self.position_eci))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "position_eci": self.position_eci.tolist(),
            "velocity_eci": self.velocity_eci.tolist(),
            "epoch": self.epoch.isoformat(),
            "reference_frame": self.reference_frame,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryState":
        """Create from dictionary."""
        return cls(
            position_eci=np.array(data["position_eci"]),
            velocity_eci=np.array(data["velocity_eci"]),
            epoch=datetime.fromisoformat(data["epoch"]),
            reference_frame=data.get("reference_frame", "J2000"),
        )


class TrajectoryProvider(ABC):
    """
    Abstract base class for satellite trajectory providers.
    
    Defines the interface for computing satellite positions and velocities
    at arbitrary times. Implementations may use different trajectory models
    (Keplerian elements, SPICE kernels, TLE propagation, etc.).
    
    All implementations must support:
    - Getting full state vectors (position + velocity)
    - Getting position only (for efficiency)
    - Listing available satellites
    - Reporting valid time bounds
    
    Examples
    --------
    >>> class MyProvider(TrajectoryProvider):
    ...     def get_state(self, satellite_id, time):
    ...         # Implementation
    ...         pass
    ...     # ... other methods
    
    >>> provider = MyProvider()
    >>> state = provider.get_state("SAT-001", datetime.now())
    """
    
    @abstractmethod
    def get_state(self, satellite_id: str, time: datetime) -> TrajectoryState:
        """
        Get full state vector for a satellite at a specific time.
        
        Parameters
        ----------
        satellite_id : str
            Unique identifier of the satellite
        time : datetime
            Time at which to compute the state
        
        Returns
        -------
        TrajectoryState
            Position and velocity in ECI coordinates
        
        Raises
        ------
        KeyError
            If satellite_id is not found
        ValueError
            If time is outside valid bounds
        """
        pass
    
    @abstractmethod
    def get_position_eci(self, satellite_id: str, time: datetime) -> np.ndarray:
        """
        Get position in ECI coordinates for a satellite at a specific time.
        
        This method may be more efficient than get_state() when only
        position is needed.
        
        Parameters
        ----------
        satellite_id : str
            Unique identifier of the satellite
        time : datetime
            Time at which to compute the position
        
        Returns
        -------
        np.ndarray
            Position vector [x, y, z] in kilometers
        
        Raises
        ------
        KeyError
            If satellite_id is not found
        ValueError
            If time is outside valid bounds
        """
        pass
    
    @abstractmethod
    def get_satellite_ids(self) -> List[str]:
        """
        Get list of available satellite identifiers.
        
        Returns
        -------
        List[str]
            List of satellite IDs that can be queried
        """
        pass
    
    @abstractmethod
    def get_time_bounds(self, satellite_id: str) -> Tuple[datetime, datetime]:
        """
        Get valid time range for a satellite.
        
        Parameters
        ----------
        satellite_id : str
            Unique identifier of the satellite
        
        Returns
        -------
        Tuple[datetime, datetime]
            (start_time, end_time) tuple defining valid range
            For unbounded trajectories, returns (datetime.min, datetime.max)
        
        Raises
        ------
        KeyError
            If satellite_id is not found
        """
        pass
    
    def get_velocity_eci(self, satellite_id: str, time: datetime) -> np.ndarray:
        """
        Get velocity in ECI coordinates for a satellite at a specific time.
        
        Default implementation uses get_state(), but subclasses may
        override for efficiency.
        
        Parameters
        ----------
        satellite_id : str
            Unique identifier of the satellite
        time : datetime
            Time at which to compute the velocity
        
        Returns
        -------
        np.ndarray
            Velocity vector [vx, vy, vz] in km/s
        """
        return self.get_state(satellite_id, time).velocity_eci
    
    def is_valid_time(self, satellite_id: str, time: datetime) -> bool:
        """
        Check if a time is within valid bounds for a satellite.
        
        Parameters
        ----------
        satellite_id : str
            Unique identifier of the satellite
        time : datetime
            Time to check
        
        Returns
        -------
        bool
            True if time is within valid bounds
        """
        start, end = self.get_time_bounds(satellite_id)
        return start <= time <= end
    
    def __contains__(self, satellite_id: str) -> bool:
        """Check if satellite ID is available."""
        return satellite_id in self.get_satellite_ids()
    
    def __len__(self) -> int:
        """Return number of satellites."""
        return len(self.get_satellite_ids())


class KeplerianProvider(TrajectoryProvider):
    """
    Trajectory provider that wraps existing Satellite objects.
    
    This provider uses Keplerian orbital mechanics to compute satellite
    positions and velocities. It wraps existing Satellite instances,
    ensuring zero behavioral change when used as the default provider.
    
    The provider manages internal time tracking relative to an epoch,
    propagating satellite positions forward from the initial state.
    
    Parameters
    ----------
    satellites : List[Satellite]
        List of Satellite objects to wrap
    epoch : datetime
        Reference epoch for time calculations
    
    Attributes
    ----------
    satellites : Dict[str, Satellite]
        Dictionary mapping satellite_id to Satellite object
    epoch : datetime
        Reference epoch
    
    Examples
    --------
    >>> from simulation import Satellite, EllipticalOrbit, EARTH_RADIUS_KM
    >>> orbit = EllipticalOrbit(
    ...     apoapsis=EARTH_RADIUS_KM + 550,
    ...     periapsis=EARTH_RADIUS_KM + 550,
    ...     inclination=0.9,
    ...     longitude_of_ascending_node=0,
    ...     argument_of_periapsis=0,
    ... )
    >>> sat = Satellite(orbit, satellite_id="SAT-001")
    >>> provider = KeplerianProvider([sat], datetime(2025, 1, 1))
    >>> state = provider.get_state("SAT-001", datetime(2025, 1, 1, 0, 1))
    """
    
    def __init__(
        self,
        satellites: List[Satellite],
        epoch: datetime,
    ):
        """
        Initialize KeplerianProvider with satellites and epoch.
        
        Parameters
        ----------
        satellites : List[Satellite]
            List of Satellite objects to manage
        epoch : datetime
            Reference epoch for time calculations. All satellites are
            assumed to be at their initial positions at this time.
        """
        self.epoch = epoch
        self._satellites: Dict[str, Satellite] = {
            sat.satellite_id: sat for sat in satellites
        }
        
        # Store initial positions for reset capability
        self._initial_positions: Dict[str, float] = {
            sat.satellite_id: sat.position for sat in satellites
        }
        
        # Track current simulation time relative to epoch
        self._current_time: datetime = epoch
        
    @property
    def satellites(self) -> Dict[str, Satellite]:
        """Dictionary of satellite_id -> Satellite."""
        return self._satellites
    
    def get_satellite(self, satellite_id: str) -> Satellite:
        """
        Get the underlying Satellite object.
        
        Parameters
        ----------
        satellite_id : str
            Unique identifier of the satellite
        
        Returns
        -------
        Satellite
            The wrapped Satellite object
        
        Raises
        ------
        KeyError
            If satellite_id is not found
        """
        if satellite_id not in self._satellites:
            raise KeyError(f"Satellite '{satellite_id}' not found")
        return self._satellites[satellite_id]
    
    def _propagate_to_time(self, satellite_id: str, time: datetime) -> None:
        """
        Propagate a satellite to the specified time.
        
        This method handles time propagation relative to the epoch,
        computing the satellite's position at the requested time.
        
        Parameters
        ----------
        satellite_id : str
            Satellite to propagate
        time : datetime
            Target time
        """
        sat = self._satellites[satellite_id]
        
        # Calculate elapsed time from epoch in seconds
        elapsed_seconds = (time - self.epoch).total_seconds()
        
        # Calculate orbital position at this time
        # Position = initial_position + (elapsed_time / period)
        orbit_fraction = elapsed_seconds / sat.orbit.period
        new_position = (self._initial_positions[satellite_id] + orbit_fraction) % 1.0
        
        # Temporarily set the satellite position for computation
        sat._position = new_position
    
    def get_state(self, satellite_id: str, time: datetime) -> TrajectoryState:
        """
        Get full state vector for a satellite at a specific time.
        
        Parameters
        ----------
        satellite_id : str
            Unique identifier of the satellite
        time : datetime
            Time at which to compute the state
        
        Returns
        -------
        TrajectoryState
            Position and velocity in ECI coordinates
        
        Raises
        ------
        KeyError
            If satellite_id is not found
        """
        if satellite_id not in self._satellites:
            raise KeyError(f"Satellite '{satellite_id}' not found")
        
        # Propagate to requested time
        self._propagate_to_time(satellite_id, time)
        
        sat = self._satellites[satellite_id]
        
        return TrajectoryState(
            position_eci=sat.get_position_eci(),
            velocity_eci=sat.get_velocity_eci(),
            epoch=time,
            reference_frame="J2000",
        )
    
    def get_position_eci(self, satellite_id: str, time: datetime) -> np.ndarray:
        """
        Get position in ECI coordinates for a satellite at a specific time.
        
        Parameters
        ----------
        satellite_id : str
            Unique identifier of the satellite
        time : datetime
            Time at which to compute the position
        
        Returns
        -------
        np.ndarray
            Position vector [x, y, z] in kilometers
        
        Raises
        ------
        KeyError
            If satellite_id is not found
        """
        if satellite_id not in self._satellites:
            raise KeyError(f"Satellite '{satellite_id}' not found")
        
        # Propagate to requested time
        self._propagate_to_time(satellite_id, time)
        
        return self._satellites[satellite_id].get_position_eci()
    
    def get_velocity_eci(self, satellite_id: str, time: datetime) -> np.ndarray:
        """
        Get velocity in ECI coordinates for a satellite at a specific time.
        
        Parameters
        ----------
        satellite_id : str
            Unique identifier of the satellite
        time : datetime
            Time at which to compute the velocity
        
        Returns
        -------
        np.ndarray
            Velocity vector [vx, vy, vz] in km/s
        
        Raises
        ------
        KeyError
            If satellite_id is not found
        """
        if satellite_id not in self._satellites:
            raise KeyError(f"Satellite '{satellite_id}' not found")
        
        # Propagate to requested time
        self._propagate_to_time(satellite_id, time)
        
        return self._satellites[satellite_id].get_velocity_eci()
    
    def get_satellite_ids(self) -> List[str]:
        """
        Get list of available satellite identifiers.
        
        Returns
        -------
        List[str]
            List of satellite IDs
        """
        return list(self._satellites.keys())
    
    def get_time_bounds(self, satellite_id: str) -> Tuple[datetime, datetime]:
        """
        Get valid time range for a satellite.
        
        Keplerian orbits are mathematically valid for all time, so this
        returns the minimum and maximum datetime values.
        
        Parameters
        ----------
        satellite_id : str
            Unique identifier of the satellite
        
        Returns
        -------
        Tuple[datetime, datetime]
            (datetime.min, datetime.max) - unbounded range
        
        Raises
        ------
        KeyError
            If satellite_id is not found
        """
        if satellite_id not in self._satellites:
            raise KeyError(f"Satellite '{satellite_id}' not found")
        
        return (datetime.min, datetime.max)
    
    def step_all(self, timestep: float) -> None:
        """
        Advance all satellites by the given timestep.
        
        This method provides compatibility with the existing simulation
        approach of stepping satellites forward in time.
        
        Parameters
        ----------
        timestep : float
            Time to advance in seconds
        """
        for sat in self._satellites.values():
            sat.step(timestep)
        
        # Update initial positions to current state for next computation
        for sat_id, sat in self._satellites.items():
            self._initial_positions[sat_id] = sat.position
        
        # Update epoch to reflect stepped time
        from datetime import timedelta
        self.epoch = self.epoch + timedelta(seconds=timestep)
    
    def reset(self) -> None:
        """
        Reset all satellites to their initial positions.
        
        This restores the satellites to the state they were in when
        the provider was created.
        """
        for sat_id, initial_pos in self._initial_positions.items():
            self._satellites[sat_id]._position = initial_pos
            self._satellites[sat_id].elapsed_time = 0.0
    
    def get_current_state_dict(self) -> Dict[str, TrajectoryState]:
        """
        Get current state of all satellites.
        
        Returns
        -------
        Dict[str, TrajectoryState]
            Dictionary mapping satellite_id to current TrajectoryState
        """
        return {
            sat_id: TrajectoryState(
                position_eci=sat.get_position_eci(),
                velocity_eci=sat.get_velocity_eci(),
                epoch=self.epoch,
                reference_frame="J2000",
            )
            for sat_id, sat in self._satellites.items()
        }


# Factory function for creating providers
def create_keplerian_provider(
    satellites: List[Satellite],
    epoch: Optional[datetime] = None,
) -> KeplerianProvider:
    """
    Create a KeplerianProvider from a list of satellites.
    
    Parameters
    ----------
    satellites : List[Satellite]
        List of Satellite objects
    epoch : datetime, optional
        Reference epoch. If not provided, uses current time.
    
    Returns
    -------
    KeplerianProvider
        Configured trajectory provider
    
    Examples
    --------
    >>> provider = create_keplerian_provider(satellites)
    >>> state = provider.get_state("SAT-001", datetime.now())
    """
    if epoch is None:
        epoch = datetime.now()
    
    return KeplerianProvider(satellites, epoch)