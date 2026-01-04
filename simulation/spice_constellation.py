#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPICE Constellation Module

Provides utilities for creating satellite constellations from SPICE ephemeris
kernels. This module handles:
- Discovering bodies in SPK files
- Converting SPICE state vectors to Keplerian orbital elements
- Creating Satellite objects from SPICE data

This enables loading real satellite ephemeris data and visualizing/simulating
constellations defined in SPICE SPK files.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
import math

import numpy as np

from .orbit import EllipticalOrbit, EARTH_RADIUS_KM
from .satellite import Satellite
from .spice_provider import (
    SpiceProvider,
    SpiceKernelSet,
    SPICE_AVAILABLE,
)

if SPICE_AVAILABLE:
    import spiceypy as spice

logger = logging.getLogger(__name__)


# Earth gravitational parameter (km^3/s^2)
EARTH_MU = 398600.4418


@dataclass
class SPKBodyInfo:
    """
    Information about a body discovered in an SPK file.
    
    Attributes
    ----------
    naif_id : int
        NAIF ID of the body
    name : str
        Name of the body (from SPICE or generated)
    coverage_start : datetime
        Start of ephemeris coverage
    coverage_end : datetime
        End of ephemeris coverage
    """
    naif_id: int
    name: str
    coverage_start: Optional[datetime] = None
    coverage_end: Optional[datetime] = None


@dataclass
class OrbitalElements:
    """
    Classical Keplerian orbital elements.
    
    Attributes
    ----------
    semi_major_axis : float
        Semi-major axis in km
    eccentricity : float
        Orbital eccentricity (0 = circular, <1 = elliptical)
    inclination : float
        Orbital inclination in radians
    raan : float
        Right ascension of ascending node in radians
    argument_of_periapsis : float
        Argument of periapsis in radians
    true_anomaly : float
        True anomaly in radians
    mean_anomaly : float
        Mean anomaly in radians
    """
    semi_major_axis: float
    eccentricity: float
    inclination: float
    raan: float
    argument_of_periapsis: float
    true_anomaly: float
    mean_anomaly: float
    
    @property
    def apoapsis(self) -> float:
        """Apoapsis distance from Earth's center (km)."""
        return self.semi_major_axis * (1 + self.eccentricity)
    
    @property
    def periapsis(self) -> float:
        """Periapsis distance from Earth's center (km)."""
        return self.semi_major_axis * (1 - self.eccentricity)
    
    @property
    def altitude(self) -> float:
        """Altitude above Earth surface at semi-major axis (km)."""
        return self.semi_major_axis - EARTH_RADIUS_KM
    
    @property
    def apoapsis_altitude(self) -> float:
        """Altitude above Earth surface at apoapsis (km)."""
        return self.apoapsis - EARTH_RADIUS_KM
    
    @property
    def periapsis_altitude(self) -> float:
        """Altitude above Earth surface at periapsis (km)."""
        return self.periapsis - EARTH_RADIUS_KM
    
    @property
    def inclination_deg(self) -> float:
        """Inclination in degrees."""
        return math.degrees(self.inclination)
    
    @property
    def period_seconds(self) -> float:
        """Orbital period in seconds."""
        return 2 * math.pi * math.sqrt(self.semi_major_axis**3 / EARTH_MU)
    
    @property
    def position_parameter(self) -> float:
        """Position parameter (0 to 1) for Satellite class."""
        return (self.mean_anomaly / (2 * math.pi)) % 1.0


def discover_spk_bodies(
    spk_path: Union[str, Path],
    leapseconds_path: Optional[Union[str, Path]] = None,
) -> List[SPKBodyInfo]:
    """
    Discover all bodies contained in an SPK file.
    
    Parameters
    ----------
    spk_path : str or Path
        Path to the SPK ephemeris file (.bsp)
    leapseconds_path : str or Path, optional
        Path to leapseconds kernel (.tls). Required for time conversion.
    
    Returns
    -------
    List[SPKBodyInfo]
        List of body information for each body in the SPK file
    
    Raises
    ------
    ImportError
        If SpiceyPy is not installed
    FileNotFoundError
        If SPK file does not exist
    
    Examples
    --------
    >>> bodies = discover_spk_bodies("constellation.bsp", "naif0012.tls")
    >>> for body in bodies:
    ...     print(f"{body.name}: NAIF ID {body.naif_id}")
    """
    if not SPICE_AVAILABLE:
        raise ImportError(
            "SpiceyPy not installed. Install with: pip install spiceypy"
        )
    
    spk_path = Path(spk_path)
    if not spk_path.exists():
        raise FileNotFoundError(f"SPK file not found: {spk_path}")
    
    bodies = []
    kernels_loaded = []
    
    try:
        # Load leapseconds if provided (needed for time conversion)
        if leapseconds_path:
            leapseconds_path = Path(leapseconds_path)
            if leapseconds_path.exists():
                spice.furnsh(str(leapseconds_path))
                kernels_loaded.append(leapseconds_path)
        
        # Get all body IDs in the SPK file
        spk_ids = spice.spkobj(str(spk_path))
        num_bodies = spice.card(spk_ids)
        
        for i in range(num_bodies):
            naif_id = int(spk_ids[i])
            
            # Try to get body name from SPICE
            try:
                body_name = spice.bodc2n(naif_id)
                sat_id = body_name.replace(" ", "-")
            except:
                # Generate name if not found
                sat_id = f"SAT-{abs(naif_id)}"
            
            # Get coverage times if leapseconds loaded
            coverage_start = None
            coverage_end = None
            
            if leapseconds_path and kernels_loaded:
                try:
                    coverage = spice.spkcov(str(spk_path), naif_id)
                    if spice.wncard(coverage) > 0:
                        start_et, end_et = spice.wnfetd(coverage, 0)
                        # Convert ET to datetime
                        start_str = spice.et2utc(start_et, "ISOC", 3)
                        end_str = spice.et2utc(end_et, "ISOC", 3)
                        coverage_start = datetime.fromisoformat(
                            start_str.replace("Z", "+00:00")
                        )
                        coverage_end = datetime.fromisoformat(
                            end_str.replace("Z", "+00:00")
                        )
                except Exception as e:
                    logger.debug(f"Could not get coverage for {naif_id}: {e}")
            
            bodies.append(SPKBodyInfo(
                naif_id=naif_id,
                name=sat_id,
                coverage_start=coverage_start,
                coverage_end=coverage_end,
            ))
        
        logger.info(f"Discovered {len(bodies)} bodies in {spk_path}")
        
    finally:
        # Clean up loaded kernels
        spice.kclear()
    
    return bodies


def create_naif_mapping(bodies: List[SPKBodyInfo]) -> Dict[str, int]:
    """
    Create a NAIF ID mapping from discovered bodies.
    
    Parameters
    ----------
    bodies : List[SPKBodyInfo]
        List of body information from discover_spk_bodies()
    
    Returns
    -------
    Dict[str, int]
        Mapping from satellite name to NAIF ID
    """
    return {body.name: body.naif_id for body in bodies}


def state_vector_to_elements(
    position: np.ndarray,
    velocity: np.ndarray,
    mu: float = EARTH_MU,
) -> OrbitalElements:
    """
    Convert state vector (position, velocity) to Keplerian orbital elements.
    
    Parameters
    ----------
    position : np.ndarray
        Position vector [x, y, z] in km (ECI frame)
    velocity : np.ndarray
        Velocity vector [vx, vy, vz] in km/s (ECI frame)
    mu : float, optional
        Gravitational parameter in km^3/s^2 (default: Earth)
    
    Returns
    -------
    OrbitalElements
        Computed Keplerian orbital elements
    
    Examples
    --------
    >>> pos = np.array([7000, 0, 0])  # km
    >>> vel = np.array([0, 7.5, 0])   # km/s
    >>> elements = state_vector_to_elements(pos, vel)
    >>> print(f"Semi-major axis: {elements.semi_major_axis:.0f} km")
    """
    r = np.linalg.norm(position)
    v = np.linalg.norm(velocity)
    
    # Specific orbital energy
    energy = (v**2 / 2) - (mu / r)
    
    # Semi-major axis
    if energy < 0:
        a = -mu / (2 * energy)
    else:
        # Parabolic or hyperbolic - use current radius as approximation
        a = r
        logger.warning(f"Non-elliptical orbit detected (energy={energy:.2f}), using r={r:.0f} km")
    
    # Angular momentum vector
    h = np.cross(position, velocity)
    h_mag = np.linalg.norm(h)
    
    # Eccentricity vector
    e_vec = (np.cross(velocity, h) / mu) - (position / r)
    e = np.linalg.norm(e_vec)
    
    # Clamp eccentricity to valid range
    e = max(0.0, min(e, 0.99))
    
    # Inclination
    i = np.arccos(np.clip(h[2] / h_mag, -1, 1))
    
    # Node vector (points to ascending node)
    n = np.cross([0, 0, 1], h)
    n_mag = np.linalg.norm(n)
    
    # Right ascension of ascending node (RAAN)
    if n_mag > 1e-10:
        raan = np.arccos(np.clip(n[0] / n_mag, -1, 1))
        if n[1] < 0:
            raan = 2 * np.pi - raan
    else:
        raan = 0.0
    
    # Argument of periapsis
    if n_mag > 1e-10 and e > 1e-10:
        arg_periapsis = np.arccos(np.clip(np.dot(n, e_vec) / (n_mag * e), -1, 1))
        if e_vec[2] < 0:
            arg_periapsis = 2 * np.pi - arg_periapsis
    else:
        arg_periapsis = 0.0
    
    # True anomaly
    if e > 1e-10:
        true_anomaly = np.arccos(np.clip(np.dot(e_vec, position) / (e * r), -1, 1))
        if np.dot(position, velocity) < 0:
            true_anomaly = 2 * np.pi - true_anomaly
    else:
        # Circular orbit - use argument of latitude
        if n_mag > 1e-10:
            true_anomaly = np.arccos(np.clip(np.dot(n, position) / (n_mag * r), -1, 1))
            if position[2] < 0:
                true_anomaly = 2 * np.pi - true_anomaly
        else:
            true_anomaly = 0.0
    
    # Convert true anomaly to mean anomaly
    if e < 1e-10:
        mean_anomaly = true_anomaly
    else:
        # Eccentric anomaly from true anomaly
        half_ta = true_anomaly / 2
        tan_half_E = np.sqrt((1 - e) / (1 + e)) * np.tan(half_ta)
        E = 2 * np.arctan(tan_half_E)
        # Mean anomaly from eccentric anomaly
        mean_anomaly = E - e * np.sin(E)
    
    # Normalize mean anomaly to [0, 2*pi)
    mean_anomaly = mean_anomaly % (2 * np.pi)
    
    return OrbitalElements(
        semi_major_axis=a,
        eccentricity=e,
        inclination=i,
        raan=raan,
        argument_of_periapsis=arg_periapsis,
        true_anomaly=true_anomaly,
        mean_anomaly=mean_anomaly,
    )


def create_satellite_from_spice(
    provider: SpiceProvider,
    satellite_id: str,
    epoch: Optional[datetime] = None,
) -> Tuple[Satellite, EllipticalOrbit, OrbitalElements]:
    """
    Create a Satellite object from SPICE ephemeris data.
    
    Parameters
    ----------
    provider : SpiceProvider
        Initialized SPICE provider with loaded kernels
    satellite_id : str
        Satellite ID to query from the provider
    epoch : datetime, optional
        Time at which to compute orbital elements. Default: now (UTC)
    
    Returns
    -------
    Tuple[Satellite, EllipticalOrbit, OrbitalElements]
        Created Satellite, its orbit, and computed orbital elements
    
    Raises
    ------
    KeyError
        If satellite_id not found in provider
    ValueError
        If epoch is outside kernel coverage
    
    Examples
    --------
    >>> provider = SpiceProvider(kernel_set, naif_mapping)
    >>> sat, orbit, elements = create_satellite_from_spice(provider, "TDRS-3")
    >>> print(f"Altitude: {elements.altitude:.0f} km")
    """
    # Use naive datetime to avoid timezone comparison issues in SpiceProvider
    if epoch is None:
        epoch = datetime.utcnow()
    elif epoch.tzinfo is not None:
        # Convert to naive datetime (assume UTC)
        epoch = epoch.replace(tzinfo=None)
    
    # Get state from SPICE
    state = provider.get_state(satellite_id, epoch)
    
    # Convert to orbital elements
    elements = state_vector_to_elements(state.position_eci, state.velocity_eci)
    
    # Convert from semi_major_axis/eccentricity to apoapsis/periapsis
    # apoapsis = a * (1 + e)
    # periapsis = a * (1 - e)
    apoapsis = elements.semi_major_axis * (1 + elements.eccentricity)
    periapsis = elements.semi_major_axis * (1 - elements.eccentricity)
    
    # Create orbit object using apoapsis/periapsis (the actual EllipticalOrbit interface)
    orbit = EllipticalOrbit(
        apoapsis=apoapsis,
        periapsis=periapsis,
        inclination=elements.inclination,
        longitude_of_ascending_node=elements.raan,
        argument_of_periapsis=elements.argument_of_periapsis,
    )
    
    # Create satellite
    satellite = Satellite(
        orbit=orbit,
        initial_position=elements.position_parameter,
        satellite_id=satellite_id,
    )
    
    return satellite, orbit, elements


def create_constellation_from_spice(
    spk_path: Union[str, Path],
    leapseconds_path: Union[str, Path],
    epoch: Optional[datetime] = None,
    planetary_path: Optional[Union[str, Path]] = None,
) -> Tuple[List[Satellite], List[EllipticalOrbit], SpiceProvider]:
    """
    Create a complete satellite constellation from an SPK file.
    
    This is the main entry point for loading SPICE-defined constellations.
    It discovers all bodies in the SPK file, creates orbital elements,
    and returns Satellite objects ready for simulation.
    
    Parameters
    ----------
    spk_path : str or Path
        Path to spacecraft ephemeris kernel (.bsp)
    leapseconds_path : str or Path
        Path to leapseconds kernel (.tls)
    epoch : datetime, optional
        Time at which to compute orbital elements. Default: now (UTC)
    planetary_path : str or Path, optional
        Path to planetary ephemeris kernel (.bsp)
    
    Returns
    -------
    Tuple[List[Satellite], List[EllipticalOrbit], SpiceProvider]
        - List of Satellite objects
        - List of corresponding orbits
        - The SpiceProvider (for continued access to ephemeris)
    
    Raises
    ------
    ImportError
        If SpiceyPy is not installed
    FileNotFoundError
        If kernel files not found
    ValueError
        If no valid satellites could be created
    
    Examples
    --------
    >>> satellites, orbits, provider = create_constellation_from_spice(
    ...     "TDRSS.bsp",
    ...     "naif0012.tls"
    ... )
    >>> print(f"Loaded {len(satellites)} satellites")
    >>> for sat in satellites:
    ...     pos = sat.get_geospatial_position()
    ...     print(f"  {sat.satellite_id}: alt={pos.altitude:.0f} km")
    """
    if not SPICE_AVAILABLE:
        raise ImportError(
            "SpiceyPy not installed. Install with: pip install spiceypy"
        )
    
    spk_path = Path(spk_path)
    leapseconds_path = Path(leapseconds_path)
    
    if not spk_path.exists():
        raise FileNotFoundError(f"SPK file not found: {spk_path}")
    if not leapseconds_path.exists():
        raise FileNotFoundError(f"Leapseconds file not found: {leapseconds_path}")
    
    # Use naive datetime to avoid timezone comparison issues in SpiceProvider
    if epoch is None:
        epoch = datetime.utcnow()  # Naive UTC datetime
    elif epoch.tzinfo is not None:
        # Convert to naive UTC
        epoch = epoch.replace(tzinfo=None)
    
    # Discover bodies in the SPK file
    logger.info(f"Discovering bodies in {spk_path}")
    bodies = discover_spk_bodies(spk_path, leapseconds_path)
    
    if not bodies:
        raise ValueError(f"No bodies found in SPK file: {spk_path}")
    
    logger.info(f"Found {len(bodies)} bodies: {[b.name for b in bodies]}")
    
    # Create NAIF ID mapping
    naif_mapping = create_naif_mapping(bodies)
    
    # Build kernel set
    planetary = []
    if planetary_path:
        planetary_path = Path(planetary_path)
        if planetary_path.exists():
            planetary.append(planetary_path)
    
    kernel_set = SpiceKernelSet(
        leapseconds=leapseconds_path,
        spacecraft=[spk_path],
        planetary=planetary,
    )
    
    # Create SPICE provider
    provider = SpiceProvider(
        kernel_set=kernel_set,
        naif_id_mapping=naif_mapping,
    )
    
    # Create satellites from each body
    satellites = []
    orbits = []
    
    for body in bodies:
        try:
            sat, orbit, elements = create_satellite_from_spice(
                provider, body.name, epoch
            )
            satellites.append(sat)
            orbits.append(orbit)
            
            logger.info(
                f"Created {body.name}: alt={elements.altitude:.0f} km, "
                f"inc={elements.inclination_deg:.1f}°, e={elements.eccentricity:.4f}"
            )
            
        except Exception as e:
            logger.warning(f"Could not create satellite {body.name}: {e}")
            # Print to console as well for visibility
            print(f"Could not create satellite {body.name}: {e}")
    
    if not satellites:
        raise ValueError("No satellites could be created from SPICE data")
    
    logger.info(f"Created {len(satellites)} satellites from SPICE ephemeris")
    
    return satellites, orbits, provider


def load_spice_for_simulation(
    spk_path: Union[str, Path],
    leapseconds_path: Union[str, Path],
    epoch: Optional[datetime] = None,
    planetary_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Load SPICE data and prepare it for use with the Simulation class.
    
    This is a convenience wrapper that returns all necessary components
    for integrating SPICE data with the simulation.
    
    Parameters
    ----------
    spk_path : str or Path
        Path to spacecraft ephemeris kernel (.bsp)
    leapseconds_path : str or Path
        Path to leapseconds kernel (.tls)
    epoch : datetime, optional
        Time at which to compute orbital elements
    planetary_path : str or Path, optional
        Path to planetary ephemeris kernel (.bsp)
    verbose : bool
        Whether to print progress information
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'satellites': List[Satellite]
        - 'orbits': List[EllipticalOrbit]
        - 'provider': SpiceProvider
        - 'bodies': List[SPKBodyInfo]
        - 'naif_mapping': Dict[str, int]
        - 'epoch': datetime
    
    Examples
    --------
    >>> spice_data = load_spice_for_simulation("TDRSS.bsp", "naif0012.tls")
    >>> sim.set_custom_constellation(
    ...     spice_data['orbits'],
    ...     spice_data['satellites']
    ... )
    """
    if verbose:
        print(f"\nLoading SPICE data from: {spk_path}")
    
    # Discover bodies first
    bodies = discover_spk_bodies(spk_path, leapseconds_path)
    naif_mapping = create_naif_mapping(bodies)
    
    if verbose:
        print(f"  Found {len(bodies)} bodies:")
        for body in bodies:
            print(f"    {body.name}: NAIF ID {body.naif_id}")
    
    # Use naive datetime to avoid timezone issues
    if epoch is None:
        epoch = datetime.utcnow()
    elif epoch.tzinfo is not None:
        epoch = epoch.replace(tzinfo=None)
    
    # Create constellation
    satellites, orbits, provider = create_constellation_from_spice(
        spk_path,
        leapseconds_path,
        epoch=epoch,
        planetary_path=planetary_path,
    )
    
    if verbose:
        print(f"\nCreated {len(satellites)} satellites:")
        for sat in satellites:
            pos = sat.get_geospatial_position()
            print(f"    {sat.satellite_id}: alt={pos.altitude:.0f} km")
    
    return {
        'satellites': satellites,
        'orbits': orbits,
        'provider': provider,
        'bodies': bodies,
        'naif_mapping': naif_mapping,
        'epoch': epoch,
    }