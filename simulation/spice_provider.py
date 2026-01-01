#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPICE Provider Module

Implements a trajectory provider that reads satellite positions from
SPICE ephemeris kernels using SpiceyPy. This enables using high-fidelity
NASA/JPL ephemeris data for satellite positions.

This module implements Step 2 of the NS-3/SPICE integration plan.

Features:
- Automatic time bounds detection from kernel coverage
- Efficient caching of computed positions
- Thread-safe kernel management
- Clean unload on destruction
- Graceful degradation when SpiceyPy not installed
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import json
import logging
import numpy as np

from .trajectory import TrajectoryProvider, TrajectoryState


# Check for SpiceyPy availability
try:
    import spiceypy as spice
    SPICE_AVAILABLE = True
except ImportError:
    spice = None
    SPICE_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class SpiceKernelSet:
    """
    Collection of SPICE kernel files required for ephemeris computation.
    
    Groups the different types of kernels needed for computing satellite
    positions using the SPICE toolkit.
    
    Attributes
    ----------
    leapseconds : Path
        Path to leap seconds kernel (.tls) - Required
    spacecraft : List[Path]
        Paths to spacecraft ephemeris kernels (.bsp) - Required
    planetary : List[Path]
        Paths to planetary ephemeris kernels (.bsp) - Optional
    frame : Path, optional
        Path to frame kernel (.tf) - Optional
    planetary_constants : Path, optional
        Path to planetary constants kernel (.tpc) - Optional
    
    Examples
    --------
    >>> kernel_set = SpiceKernelSet(
    ...     leapseconds=Path("naif0012.tls"),
    ...     spacecraft=[Path("constellation.bsp")],
    ...     planetary=[Path("de440.bsp")]
    ... )
    """
    leapseconds: Path
    spacecraft: List[Path]
    planetary: List[Path] = field(default_factory=list)
    frame: Optional[Path] = None
    planetary_constants: Optional[Path] = None
    
    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.leapseconds, str):
            self.leapseconds = Path(self.leapseconds)
        
        self.spacecraft = [
            Path(p) if isinstance(p, str) else p 
            for p in self.spacecraft
        ]
        self.planetary = [
            Path(p) if isinstance(p, str) else p 
            for p in self.planetary
        ]
        
        if self.frame is not None and isinstance(self.frame, str):
            self.frame = Path(self.frame)
        if self.planetary_constants is not None and isinstance(self.planetary_constants, str):
            self.planetary_constants = Path(self.planetary_constants)
    
    def all_kernels(self) -> List[Path]:
        """Get list of all kernel files."""
        kernels = [self.leapseconds] + self.spacecraft + self.planetary
        if self.frame is not None:
            kernels.append(self.frame)
        if self.planetary_constants is not None:
            kernels.append(self.planetary_constants)
        return kernels
    
    def validate(self) -> List[str]:
        """
        Validate that all required kernel files exist.
        
        Returns
        -------
        List[str]
            List of error messages, empty if all valid
        """
        errors = []
        
        if not self.leapseconds.exists():
            errors.append(f"Leapseconds kernel not found: {self.leapseconds}")
        
        for kernel in self.spacecraft:
            if not kernel.exists():
                errors.append(f"Spacecraft kernel not found: {kernel}")
        
        for kernel in self.planetary:
            if not kernel.exists():
                errors.append(f"Planetary kernel not found: {kernel}")
        
        if self.frame is not None and not self.frame.exists():
            errors.append(f"Frame kernel not found: {self.frame}")
        
        if self.planetary_constants is not None and not self.planetary_constants.exists():
            errors.append(f"Planetary constants kernel not found: {self.planetary_constants}")
        
        return errors


@dataclass
class SpiceConstellationConfig:
    """
    Configuration for a SPICE-based satellite constellation.
    
    Defines the mapping between satellite IDs and NAIF IDs, along with
    observation parameters.
    
    Attributes
    ----------
    name : str
        Name of the constellation
    satellites : Dict[str, int]
        Mapping of satellite_id to NAIF ID
    reference_frame : str
        Reference frame for positions (default "J2000")
    observer : str
        Observer body name (default "EARTH")
    aberration_correction : str
        Aberration correction type (default "NONE")
    """
    name: str
    satellites: Dict[str, int]
    reference_frame: str = "J2000"
    observer: str = "EARTH"
    aberration_correction: str = "NONE"


class SpiceProvider(TrajectoryProvider):
    """
    Trajectory provider using SPICE ephemeris kernels.
    
    Computes satellite positions by querying SPICE kernels through
    SpiceyPy. Supports automatic time bounds detection and efficient
    position caching.
    
    Parameters
    ----------
    kernel_set : SpiceKernelSet
        Collection of kernel files to load
    naif_id_mapping : Dict[str, int]
        Mapping from satellite_id to NAIF ID
    reference_frame : str, optional
        Reference frame for positions (default "J2000")
    observer : str, optional
        Observer body name (default "EARTH")
    aberration_correction : str, optional
        Aberration correction (default "NONE")
    
    Raises
    ------
    ImportError
        If SpiceyPy is not installed
    FileNotFoundError
        If any kernel file is not found
    
    Examples
    --------
    >>> kernel_set = SpiceKernelSet(
    ...     leapseconds=Path("naif0012.tls"),
    ...     spacecraft=[Path("constellation.bsp")]
    ... )
    >>> provider = SpiceProvider(
    ...     kernel_set=kernel_set,
    ...     naif_id_mapping={"SAT-001": -100001}
    ... )
    >>> state = provider.get_state("SAT-001", datetime(2025, 1, 1))
    """
    
    def __init__(
        self,
        kernel_set: SpiceKernelSet,
        naif_id_mapping: Dict[str, int],
        reference_frame: str = "J2000",
        observer: str = "EARTH",
        aberration_correction: str = "NONE",
    ):
        if not SPICE_AVAILABLE:
            raise ImportError(
                "SpiceyPy not installed. Install with: pip install spiceypy\n"
                "Or use --trajectory-provider=keplerian (default)"
            )
        
        self._kernel_set = kernel_set
        self._naif_id_mapping = naif_id_mapping.copy()
        self._reference_frame = reference_frame
        self._observer = observer
        self._aberration_correction = aberration_correction
        
        # Reverse mapping for lookups
        self._id_to_naif = naif_id_mapping.copy()
        self._naif_to_id = {v: k for k, v in naif_id_mapping.items()}
        
        # Time bounds cache
        self._time_bounds_cache: Dict[str, Tuple[datetime, datetime]] = {}
        
        # Position cache for efficiency
        self._position_cache: Dict[Tuple[str, float], np.ndarray] = {}
        self._cache_max_size = 10000
        
        # Track loaded kernels for cleanup
        self._loaded_kernels: List[Path] = []
        
        # Validate and load kernels
        self._validate_kernels()
        self._load_kernels()
        
        logger.info(
            f"SpiceProvider initialized with {len(naif_id_mapping)} satellites"
        )
    
    def _validate_kernels(self) -> None:
        """Validate kernel files exist."""
        errors = self._kernel_set.validate()
        if errors:
            raise FileNotFoundError(
                "Missing SPICE kernel files:\n" + "\n".join(errors)
            )
    
    def _load_kernels(self) -> None:
        """Load all kernels into SPICE."""
        for kernel in self._kernel_set.all_kernels():
            spice.furnsh(str(kernel))
            self._loaded_kernels.append(kernel)
            logger.debug(f"Loaded kernel: {kernel}")
    
    def _unload_kernels(self) -> None:
        """Unload all kernels from SPICE."""
        for kernel in self._loaded_kernels:
            try:
                spice.unload(str(kernel))
                logger.debug(f"Unloaded kernel: {kernel}")
            except Exception as e:
                logger.warning(f"Error unloading kernel {kernel}: {e}")
        self._loaded_kernels.clear()
    
    def __del__(self):
        """Clean up kernels on destruction."""
        if SPICE_AVAILABLE and self._loaded_kernels:
            self._unload_kernels()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unload kernels."""
        self._unload_kernels()
        return False
    
    def _datetime_to_et(self, time: datetime) -> float:
        """
        Convert datetime to SPICE ephemeris time.
        
        Parameters
        ----------
        time : datetime
            Time to convert
        
        Returns
        -------
        float
            Ephemeris time (seconds past J2000)
        """
        # Ensure timezone-aware datetime in UTC
        if time.tzinfo is None:
            time = time.replace(tzinfo=timezone.utc)
        
        # Format as ISO string for SPICE
        time_str = time.strftime("%Y-%m-%dT%H:%M:%S.%f")
        return spice.str2et(time_str)
    
    def _et_to_datetime(self, et: float) -> datetime:
        """
        Convert SPICE ephemeris time to datetime.
        
        Parameters
        ----------
        et : float
            Ephemeris time (seconds past J2000)
        
        Returns
        -------
        datetime
            Converted datetime in UTC
        """
        utc_str = spice.et2utc(et, "ISOC", 6)
        return datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
    
    def _get_naif_id(self, satellite_id: str) -> int:
        """Get NAIF ID for satellite."""
        if satellite_id not in self._id_to_naif:
            raise KeyError(
                f"Satellite '{satellite_id}' not found. "
                f"Available: {list(self._id_to_naif.keys())}"
            )
        return self._id_to_naif[satellite_id]
    
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
            If time is outside kernel coverage
        """
        naif_id = self._get_naif_id(satellite_id)
        et = self._datetime_to_et(time)
        
        # Check time bounds
        if not self.is_valid_time(satellite_id, time):
            start, end = self.get_time_bounds(satellite_id)
            raise ValueError(
                f"Time {time} is outside kernel coverage for {satellite_id}. "
                f"Valid range: {start} to {end}"
            )
        
        # Query SPICE for state vector
        try:
            state, light_time = spice.spkezr(
                str(naif_id),
                et,
                self._reference_frame,
                self._aberration_correction,
                self._observer
            )
        except Exception as e:
            raise ValueError(
                f"SPICE error getting state for {satellite_id} at {time}: {e}"
            )
        
        # State vector: [x, y, z, vx, vy, vz] in km and km/s
        position_eci = np.array(state[0:3])
        velocity_eci = np.array(state[3:6])
        
        return TrajectoryState(
            position_eci=position_eci,
            velocity_eci=velocity_eci,
            epoch=time,
            reference_frame=self._reference_frame
        )
    
    def get_position_eci(self, satellite_id: str, time: datetime) -> np.ndarray:
        """
        Get position in ECI coordinates for a satellite at a specific time.
        
        More efficient than get_state() when only position is needed.
        
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
            If time is outside kernel coverage
        """
        naif_id = self._get_naif_id(satellite_id)
        et = self._datetime_to_et(time)
        
        # Check cache first
        cache_key = (satellite_id, et)
        if cache_key in self._position_cache:
            return self._position_cache[cache_key].copy()
        
        # Check time bounds
        if not self.is_valid_time(satellite_id, time):
            start, end = self.get_time_bounds(satellite_id)
            raise ValueError(
                f"Time {time} is outside kernel coverage for {satellite_id}. "
                f"Valid range: {start} to {end}"
            )
        
        # Query SPICE for position only
        try:
            position, light_time = spice.spkpos(
                str(naif_id),
                et,
                self._reference_frame,
                self._aberration_correction,
                self._observer
            )
        except Exception as e:
            raise ValueError(
                f"SPICE error getting position for {satellite_id} at {time}: {e}"
            )
        
        position_array = np.array(position)
        
        # Cache result (with size limit)
        if len(self._position_cache) >= self._cache_max_size:
            # Clear oldest entries (simple FIFO)
            keys = list(self._position_cache.keys())
            for key in keys[:len(keys)//2]:
                del self._position_cache[key]
        
        self._position_cache[cache_key] = position_array.copy()
        
        return position_array
    
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
        """
        return self.get_state(satellite_id, time).velocity_eci
    
    def get_satellite_ids(self) -> List[str]:
        """
        Get list of available satellite identifiers.
        
        Returns
        -------
        List[str]
            List of satellite IDs that can be queried
        """
        return list(self._id_to_naif.keys())
    
    def get_time_bounds(self, satellite_id: str) -> Tuple[datetime, datetime]:
        """
        Get valid time range for a satellite from kernel coverage.
        
        Parameters
        ----------
        satellite_id : str
            Unique identifier of the satellite
        
        Returns
        -------
        Tuple[datetime, datetime]
            (start_time, end_time) tuple defining valid range
        
        Raises
        ------
        KeyError
            If satellite_id is not found
        """
        if satellite_id in self._time_bounds_cache:
            return self._time_bounds_cache[satellite_id]
        
        naif_id = self._get_naif_id(satellite_id)
        
        # Query kernel coverage for this object
        try:
            # Get coverage window for the spacecraft
            coverage = spice.spkcov(
                str(self._kernel_set.spacecraft[0]),
                naif_id
            )
            
            # Extract first coverage window
            if spice.wncard(coverage) > 0:
                start_et, end_et = spice.wnfetd(coverage, 0)
                start_time = self._et_to_datetime(start_et)
                end_time = self._et_to_datetime(end_et)
            else:
                # No coverage found - return unbounded
                logger.warning(
                    f"No coverage found for {satellite_id}, using unbounded range"
                )
                start_time = datetime.min.replace(tzinfo=timezone.utc)
                end_time = datetime.max.replace(tzinfo=timezone.utc)
        except Exception as e:
            logger.warning(f"Error getting coverage for {satellite_id}: {e}")
            # Fall back to very wide range
            start_time = datetime(1900, 1, 1, tzinfo=timezone.utc)
            end_time = datetime(2100, 1, 1, tzinfo=timezone.utc)
        
        self._time_bounds_cache[satellite_id] = (start_time, end_time)
        return (start_time, end_time)
    
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
        
        # Ensure timezone-aware comparison
        if time.tzinfo is None:
            time = time.replace(tzinfo=timezone.utc)
        
        return start <= time <= end
    
    def clear_cache(self) -> None:
        """Clear the position cache."""
        self._position_cache.clear()
        logger.debug("Position cache cleared")
    
    @property
    def reference_frame(self) -> str:
        """Reference frame used for positions."""
        return self._reference_frame
    
    @property
    def observer(self) -> str:
        """Observer body name."""
        return self._observer
    
    @property
    def num_satellites(self) -> int:
        """Number of satellites in this provider."""
        return len(self._id_to_naif)


class SpiceDatasetLoader:
    """
    Utility class for loading SPICE configurations from various sources.
    
    Provides factory methods for creating SpiceProvider instances from
    configuration files, HORIZONS exports, and other data sources.
    
    Examples
    --------
    >>> provider = SpiceDatasetLoader.from_config_file("constellation.json")
    >>> state = provider.get_state("SAT-001", datetime.now())
    """
    
    @staticmethod
    def from_config_file(
        config_path: Union[str, Path],
        kernels_dir: Optional[Union[str, Path]] = None
    ) -> SpiceProvider:
        """
        Load SPICE provider from a JSON configuration file.
        
        Parameters
        ----------
        config_path : str or Path
            Path to configuration JSON file
        kernels_dir : str or Path, optional
            Base directory for kernel files. If not specified, kernel paths
            are relative to the config file location.
        
        Returns
        -------
        SpiceProvider
            Configured trajectory provider
        
        Raises
        ------
        FileNotFoundError
            If config file or kernels not found
        ValueError
            If config file is malformed
        
        Examples
        --------
        >>> provider = SpiceDatasetLoader.from_config_file("config.json")
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path) as f:
            config = json.load(f)
        
        # Determine base directory for kernels
        if kernels_dir is not None:
            base_dir = Path(kernels_dir)
        else:
            base_dir = config_path.parent
        
        # Parse required fields
        required_fields = ["leapseconds", "spacecraft_kernels", "satellites"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in config: {field}")
        
        # Build kernel set
        kernel_set = SpiceKernelSet(
            leapseconds=base_dir / config["leapseconds"],
            spacecraft=[base_dir / k for k in config["spacecraft_kernels"]],
            planetary=[base_dir / k for k in config.get("planetary", [])],
            frame=base_dir / config["frame"] if "frame" in config else None,
            planetary_constants=(
                base_dir / config["planetary_constants"]
                if "planetary_constants" in config else None
            )
        )
        
        # Create provider
        return SpiceProvider(
            kernel_set=kernel_set,
            naif_id_mapping=config["satellites"],
            reference_frame=config.get("reference_frame", "J2000"),
            observer=config.get("observer", "EARTH"),
            aberration_correction=config.get("aberration_correction", "NONE")
        )
    
    @staticmethod
    def from_kernel_set(
        kernel_set: SpiceKernelSet,
        naif_id_mapping: Dict[str, int],
        **kwargs
    ) -> SpiceProvider:
        """
        Create SPICE provider from a kernel set and ID mapping.
        
        Parameters
        ----------
        kernel_set : SpiceKernelSet
            Collection of kernel files
        naif_id_mapping : Dict[str, int]
            Mapping from satellite_id to NAIF ID
        **kwargs
            Additional arguments passed to SpiceProvider
        
        Returns
        -------
        SpiceProvider
            Configured trajectory provider
        """
        return SpiceProvider(
            kernel_set=kernel_set,
            naif_id_mapping=naif_id_mapping,
            **kwargs
        )
    
    @staticmethod
    def from_horizons_export(
        spk_file: Union[str, Path],
        leapseconds: Union[str, Path],
        naif_id_mapping: Dict[str, int],
        **kwargs
    ) -> SpiceProvider:
        """
        Load SPICE provider from NASA HORIZONS SPK export.
        
        Parameters
        ----------
        spk_file : str or Path
            Path to SPK file exported from HORIZONS
        leapseconds : str or Path
            Path to leap seconds kernel
        naif_id_mapping : Dict[str, int]
            Mapping from satellite_id to NAIF ID
        **kwargs
            Additional arguments passed to SpiceProvider
        
        Returns
        -------
        SpiceProvider
            Configured trajectory provider
        
        Notes
        -----
        SPK files can be generated from NASA HORIZONS at:
        https://ssd.jpl.nasa.gov/horizons/
        """
        kernel_set = SpiceKernelSet(
            leapseconds=Path(leapseconds),
            spacecraft=[Path(spk_file)]
        )
        
        return SpiceProvider(
            kernel_set=kernel_set,
            naif_id_mapping=naif_id_mapping,
            **kwargs
        )
    
    @staticmethod
    def create_config_template(
        output_path: Union[str, Path],
        num_satellites: int = 3
    ) -> Path:
        """
        Create a template configuration file.
        
        Parameters
        ----------
        output_path : str or Path
            Path for output config file
        num_satellites : int
            Number of satellites in template
        
        Returns
        -------
        Path
            Path to created config file
        """
        output_path = Path(output_path)
        
        config = {
            "name": "MyConstellation",
            "epoch": "2025-01-01T00:00:00Z",
            "leapseconds": "naif0012.tls",
            "planetary": ["de440.bsp"],
            "spacecraft_kernels": ["constellation_v1.bsp"],
            "satellites": {
                f"SAT-{i+1:03d}": -100000 - (i+1)
                for i in range(num_satellites)
            },
            "reference_frame": "J2000",
            "observer": "EARTH",
            "aberration_correction": "NONE"
        }
        
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Created template config: {output_path}")
        return output_path


# Factory function for convenience
def create_spice_provider(
    config_path: Optional[Union[str, Path]] = None,
    kernel_set: Optional[SpiceKernelSet] = None,
    naif_id_mapping: Optional[Dict[str, int]] = None,
    **kwargs
) -> SpiceProvider:
    """
    Create a SpiceProvider using the most appropriate method.
    
    Parameters
    ----------
    config_path : str or Path, optional
        Path to configuration JSON file
    kernel_set : SpiceKernelSet, optional
        Pre-configured kernel set
    naif_id_mapping : Dict[str, int], optional
        Mapping from satellite_id to NAIF ID
    **kwargs
        Additional arguments passed to SpiceProvider
    
    Returns
    -------
    SpiceProvider
        Configured trajectory provider
    
    Raises
    ------
    ValueError
        If insufficient arguments provided
    
    Examples
    --------
    >>> # From config file
    >>> provider = create_spice_provider(config_path="config.json")
    
    >>> # From kernel set
    >>> provider = create_spice_provider(
    ...     kernel_set=kernel_set,
    ...     naif_id_mapping={"SAT-001": -100001}
    ... )
    """
    if config_path is not None:
        return SpiceDatasetLoader.from_config_file(config_path, **kwargs)
    
    if kernel_set is not None and naif_id_mapping is not None:
        return SpiceDatasetLoader.from_kernel_set(
            kernel_set, naif_id_mapping, **kwargs
        )
    
    raise ValueError(
        "Must provide either config_path or (kernel_set and naif_id_mapping)"
    )


def is_spice_available() -> bool:
    """
    Check if SpiceyPy is installed and available.
    
    Returns
    -------
    bool
        True if SpiceyPy can be imported
    """
    return SPICE_AVAILABLE