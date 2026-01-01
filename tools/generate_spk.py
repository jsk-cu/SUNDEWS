#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPK Export Utility

Enables exporting SatUpdate constellation definitions to SPICE SPK format
for interoperability with other tools. Supports two export paths:

1. mkspk Export (Recommended): Exports state vectors and setup files
   compatible with NAIF's mkspk tool for creating standard-compliant SPK files.

2. Direct Export (Limited): Uses SpiceyPy's spkw09 for Type 9 segments.
   Suitable for short-duration, high-accuracy needs.

This module implements Step 3 of the NS-3/SPICE integration plan.

Usage
-----
Command-line:
    python -m tools.generate_spk --simulation walker_delta \\
        --output ./spk_output --duration 24 --step 60

Programmatic:
    from tools.generate_spk import SPKGenerator
    
    gen = SPKGenerator(output_dir)
    gen.add_from_simulation(sim, start_time, duration_hours=24)
    gen.export_for_mkspk()
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any
import json
import logging
import math
import sys

import numpy as np


logger = logging.getLogger(__name__)


# NAIF ID ranges for user-defined spacecraft
# Using range -100000 to -199999 for SatUpdate constellations
NAIF_ID_BASE = -100000
NAIF_ID_MIN = -199999
NAIF_ID_MAX = -100001

# Standard NAIF body IDs
EARTH_NAIF_ID = 399
EARTH_BARYCENTER_NAIF_ID = 3
MOON_NAIF_ID = 301
SUN_NAIF_ID = 10


@dataclass
class StateVector:
    """
    State vector at a specific time.
    
    Attributes
    ----------
    epoch : datetime
        Time of the state vector (UTC)
    position : np.ndarray
        Position [x, y, z] in kilometers
    velocity : np.ndarray
        Velocity [vx, vy, vz] in km/s
    """
    epoch: datetime
    position: np.ndarray
    velocity: np.ndarray
    
    def __post_init__(self):
        """Convert to numpy arrays if needed."""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=float)
        if not isinstance(self.velocity, np.ndarray):
            self.velocity = np.array(self.velocity, dtype=float)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "epoch": self.epoch.isoformat(),
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateVector":
        """Create from dictionary."""
        return cls(
            epoch=datetime.fromisoformat(data["epoch"]),
            position=np.array(data["position"]),
            velocity=np.array(data["velocity"])
        )


@dataclass
class SPKSegment:
    """
    SPK segment for a single spacecraft.
    
    Contains the state vectors and metadata needed to generate
    an SPK segment for one satellite.
    
    Attributes
    ----------
    satellite_id : str
        User-defined satellite identifier
    naif_id : int
        NAIF ID for the spacecraft (negative integer)
    states : List[StateVector]
        Time-ordered state vectors
    center_body : int
        NAIF ID of central body (default 399 = Earth)
    reference_frame : str
        Reference frame name (default "J2000")
    """
    satellite_id: str
    naif_id: int
    states: List[StateVector] = field(default_factory=list)
    center_body: int = EARTH_NAIF_ID
    reference_frame: str = "J2000"
    
    @property
    def start_time(self) -> Optional[datetime]:
        """Start time of coverage."""
        if not self.states:
            return None
        return self.states[0].epoch
    
    @property
    def end_time(self) -> Optional[datetime]:
        """End time of coverage."""
        if not self.states:
            return None
        return self.states[-1].epoch
    
    @property
    def duration_seconds(self) -> float:
        """Duration of coverage in seconds."""
        if len(self.states) < 2:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()
    
    def validate(self) -> List[str]:
        """
        Validate segment data.
        
        Returns
        -------
        List[str]
            List of validation errors, empty if valid
        """
        errors = []
        
        if not self.satellite_id:
            errors.append("satellite_id is required")
        
        if self.naif_id >= 0:
            errors.append(f"naif_id must be negative for spacecraft, got {self.naif_id}")
        
        if len(self.states) < 2:
            errors.append(f"At least 2 states required, got {len(self.states)}")
        
        # Check time ordering
        for i in range(1, len(self.states)):
            if self.states[i].epoch <= self.states[i-1].epoch:
                errors.append(
                    f"States must be time-ordered: state {i} ({self.states[i].epoch}) "
                    f"is not after state {i-1} ({self.states[i-1].epoch})"
                )
                break
        
        return errors


class NAIFIDManager:
    """
    Manages NAIF ID assignment for satellites.
    
    Provides consistent, reproducible NAIF ID assignment for
    satellites in a constellation.
    
    Attributes
    ----------
    base_id : int
        Base NAIF ID to start from (default -100001)
    
    Examples
    --------
    >>> manager = NAIFIDManager()
    >>> naif_id = manager.get_id("SAT-001")
    >>> print(naif_id)  # -100001
    """
    
    def __init__(self, base_id: int = NAIF_ID_MAX):
        self._base_id = base_id
        self._assignments: Dict[str, int] = {}
        self._next_id = base_id
    
    def get_id(self, satellite_id: str) -> int:
        """
        Get or assign NAIF ID for a satellite.
        
        Parameters
        ----------
        satellite_id : str
            User-defined satellite identifier
        
        Returns
        -------
        int
            NAIF ID for the satellite
        """
        if satellite_id not in self._assignments:
            self._assignments[satellite_id] = self._next_id
            self._next_id -= 1
            
            if self._next_id < NAIF_ID_MIN:
                raise ValueError(
                    f"Exceeded maximum number of satellites "
                    f"({NAIF_ID_MAX - NAIF_ID_MIN + 1})"
                )
        
        return self._assignments[satellite_id]
    
    def get_mapping(self) -> Dict[str, int]:
        """Get complete satellite ID to NAIF ID mapping."""
        return self._assignments.copy()
    
    def set_mapping(self, mapping: Dict[str, int]) -> None:
        """
        Set explicit NAIF ID mapping.
        
        Parameters
        ----------
        mapping : Dict[str, int]
            Mapping from satellite_id to NAIF ID
        """
        self._assignments = mapping.copy()
        if self._assignments:
            self._next_id = min(self._assignments.values()) - 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary for serialization."""
        return {
            "base_id": self._base_id,
            "assignments": self._assignments.copy()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NAIFIDManager":
        """Create from dictionary."""
        manager = cls(base_id=data.get("base_id", NAIF_ID_MAX))
        manager.set_mapping(data.get("assignments", {}))
        return manager


class SPKGenerator:
    """
    Generator for SPICE SPK ephemeris files.
    
    Collects satellite state vectors and exports them in formats
    compatible with NAIF's mkspk tool or directly via SpiceyPy.
    
    Parameters
    ----------
    output_dir : Path or str
        Directory for output files
    producer_id : str, optional
        Producer identification string (default "SatUpdate")
    
    Examples
    --------
    >>> gen = SPKGenerator("./spk_output")
    >>> gen.add_satellite_trajectory("SAT-001", -100001, states)
    >>> gen.export_for_mkspk()
    """
    
    def __init__(
        self,
        output_dir: Union[Path, str],
        producer_id: str = "SatUpdate"
    ):
        self.output_dir = Path(output_dir)
        self.producer_id = producer_id
        self.segments: List[SPKSegment] = []
        self.naif_manager = NAIFIDManager()
        
        self._metadata: Dict[str, Any] = {
            "created": datetime.now(timezone.utc).isoformat(),
            "producer": producer_id,
            "version": "1.0"
        }
    
    def add_satellite_trajectory(
        self,
        satellite_id: str,
        naif_id: Optional[int] = None,
        states: Optional[List[Union[StateVector, Dict]]] = None,
        center_body: int = EARTH_NAIF_ID,
        reference_frame: str = "J2000"
    ) -> SPKSegment:
        """
        Add a satellite trajectory to the generator.
        
        Parameters
        ----------
        satellite_id : str
            User-defined satellite identifier
        naif_id : int, optional
            NAIF ID for the spacecraft. If not provided, one will be assigned.
        states : List[StateVector or Dict], optional
            State vectors. If None, empty segment is created.
        center_body : int, optional
            NAIF ID of central body (default 399 = Earth)
        reference_frame : str, optional
            Reference frame name (default "J2000")
        
        Returns
        -------
        SPKSegment
            The created segment (for further modification if needed)
        """
        # Assign NAIF ID if not provided
        if naif_id is None:
            naif_id = self.naif_manager.get_id(satellite_id)
        else:
            # Register explicit mapping
            self.naif_manager._assignments[satellite_id] = naif_id
        
        # Convert dict states to StateVector objects
        converted_states = []
        if states:
            for state in states:
                if isinstance(state, dict):
                    # Handle both formats: {"time": ..., "pos": ..., "vel": ...}
                    # and {"epoch": ..., "position": ..., "velocity": ...}
                    if "time" in state:
                        converted_states.append(StateVector(
                            epoch=state["time"],
                            position=np.array(state["pos"]),
                            velocity=np.array(state["vel"])
                        ))
                    else:
                        converted_states.append(StateVector.from_dict(state))
                else:
                    converted_states.append(state)
        
        segment = SPKSegment(
            satellite_id=satellite_id,
            naif_id=naif_id,
            states=converted_states,
            center_body=center_body,
            reference_frame=reference_frame
        )
        
        self.segments.append(segment)
        logger.debug(f"Added trajectory for {satellite_id} (NAIF ID: {naif_id})")
        
        return segment
    
    def add_from_simulation(
        self,
        simulation,
        start_time: Optional[datetime] = None,
        duration_hours: float = 24.0,
        step_seconds: float = 60.0,
        center_body: int = EARTH_NAIF_ID,
        reference_frame: str = "J2000"
    ) -> List[SPKSegment]:
        """
        Add trajectories from a simulation.
        
        Propagates the simulation and records state vectors at regular
        intervals.
        
        Parameters
        ----------
        simulation : Simulation
            Initialized simulation instance
        start_time : datetime, optional
            Start time for ephemeris (default: now)
        duration_hours : float, optional
            Duration in hours (default 24)
        step_seconds : float, optional
            Time step in seconds (default 60)
        center_body : int, optional
            NAIF ID of central body (default 399 = Earth)
        reference_frame : str, optional
            Reference frame name (default "J2000")
        
        Returns
        -------
        List[SPKSegment]
            Created segments for all satellites
        """
        if start_time is None:
            start_time = datetime.now(timezone.utc)
        
        # Ensure timezone-aware
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        
        duration_seconds = duration_hours * 3600.0
        num_steps = int(duration_seconds / step_seconds)
        
        # Initialize state storage for each satellite
        satellite_states: Dict[str, List[StateVector]] = {
            sat.satellite_id: [] for sat in simulation.satellites
        }
        
        # Record initial states
        current_time = start_time
        for sat in simulation.satellites:
            satellite_states[sat.satellite_id].append(StateVector(
                epoch=current_time,
                position=sat.get_position_eci().copy(),
                velocity=sat.get_velocity_eci().copy()
            ))
        
        # Propagate and record
        for step in range(num_steps):
            simulation.step(step_seconds)
            current_time = current_time + timedelta(seconds=step_seconds)
            
            for sat in simulation.satellites:
                satellite_states[sat.satellite_id].append(StateVector(
                    epoch=current_time,
                    position=sat.get_position_eci().copy(),
                    velocity=sat.get_velocity_eci().copy()
                ))
        
        # Create segments
        segments = []
        for sat in simulation.satellites:
            segment = self.add_satellite_trajectory(
                satellite_id=sat.satellite_id,
                states=satellite_states[sat.satellite_id],
                center_body=center_body,
                reference_frame=reference_frame
            )
            segments.append(segment)
        
        logger.info(
            f"Added {len(segments)} satellite trajectories "
            f"({num_steps + 1} states each, {duration_hours}h duration)"
        )
        
        return segments
    
    def export_for_mkspk(self) -> Path:
        """
        Export state vectors and setup files for NAIF mkspk tool.
        
        Creates the following structure:
        - {satellite_id}_states.txt: State vectors
        - {satellite_id}_setup.txt: mkspk configuration
        - generate_all.sh: Script to run mkspk for all satellites
        - naif_ids.json: NAIF ID mapping documentation
        - metadata.json: Export metadata
        
        Returns
        -------
        Path
            Output directory containing all files
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        for segment in self.segments:
            # Validate segment
            errors = segment.validate()
            if errors:
                logger.warning(
                    f"Skipping invalid segment {segment.satellite_id}: "
                    f"{'; '.join(errors)}"
                )
                continue
            
            # Write state vectors file
            states_file = self._write_states_file(segment)
            generated_files.append(states_file)
            
            # Write mkspk setup file
            setup_file = self._write_setup_file(segment, states_file)
            generated_files.append(setup_file)
        
        # Write shell script to run mkspk
        script_file = self._write_generate_script()
        generated_files.append(script_file)
        
        # Write NAIF ID mapping
        mapping_file = self._write_naif_mapping()
        generated_files.append(mapping_file)
        
        # Write metadata
        metadata_file = self._write_metadata()
        generated_files.append(metadata_file)
        
        logger.info(
            f"Exported {len(self.segments)} segments to {self.output_dir}"
        )
        
        return self.output_dir
    
    def _write_states_file(self, segment: SPKSegment) -> Path:
        """Write state vectors file for mkspk."""
        filename = f"{segment.satellite_id}_states.txt"
        filepath = self.output_dir / filename
        
        with open(filepath, "w") as f:
            f.write(f"# State vectors for {segment.satellite_id}\n")
            f.write(f"# NAIF ID: {segment.naif_id}\n")
            f.write(f"# Center body: {segment.center_body}\n")
            f.write(f"# Reference frame: {segment.reference_frame}\n")
            f.write(f"# Format: EPOCH X Y Z VX VY VZ\n")
            f.write(f"# Units: UTC, km, km/s\n")
            f.write("#\n")
            
            for state in segment.states:
                # Format epoch as ISO string
                epoch_str = state.epoch.strftime("%Y-%m-%dT%H:%M:%S.%f")
                
                # Write state vector
                f.write(
                    f"{epoch_str} "
                    f"{state.position[0]:.10e} "
                    f"{state.position[1]:.10e} "
                    f"{state.position[2]:.10e} "
                    f"{state.velocity[0]:.10e} "
                    f"{state.velocity[1]:.10e} "
                    f"{state.velocity[2]:.10e}\n"
                )
        
        logger.debug(f"Wrote states file: {filepath}")
        return filepath
    
    def _write_setup_file(self, segment: SPKSegment, states_file: Path) -> Path:
        """Write mkspk setup file."""
        filename = f"{segment.satellite_id}_setup.txt"
        filepath = self.output_dir / filename
        
        # Determine output SPK filename
        spk_filename = f"{segment.satellite_id}.bsp"
        
        with open(filepath, "w") as f:
            f.write("\\begindata\n\n")
            
            # Input configuration
            f.write("   INPUT_DATA_TYPE   = 'STATES'\n")
            f.write(f"   INPUT_DATA_FILE   = '{states_file.name}'\n")
            f.write("   DATA_ORDER        = 'EPOCH X Y Z VX VY VZ'\n")
            f.write("   DATA_DELIMITER    = ' '\n")
            f.write("   LINES_PER_RECORD  = 1\n")
            f.write("   COMMENT_DELIMITER = '#'\n\n")
            
            # Time configuration
            f.write("   TIME_WRAPPER      = '# ETIME'\n")
            f.write("   INPUT_DATA_UNITS  = ('ANGLES=DEGREES' 'DISTANCES=km')\n\n")
            
            # Output configuration
            f.write("   OUTPUT_SPK_TYPE   = 9\n")
            f.write(f"   OUTPUT_FILE       = '{spk_filename}'\n")
            f.write("   POLYNOM_DEGREE    = 7\n\n")
            
            # Segment configuration
            f.write(f"   OBJECT_ID         = {segment.naif_id}\n")
            f.write(f"   OBJECT_NAME       = '{segment.satellite_id}'\n")
            f.write(f"   CENTER_ID         = {segment.center_body}\n")
            f.write("   CENTER_NAME       = 'EARTH'\n")
            f.write(f"   REF_FRAME_NAME    = '{segment.reference_frame}'\n\n")
            
            # Producer information
            f.write(f"   PRODUCER_ID       = '{self.producer_id}'\n\n")
            
            # Segment ID (first 40 chars of description)
            segment_id = f"{segment.satellite_id} ephemeris"[:40]
            f.write(f"   SEGMENT_ID        = '{segment_id}'\n\n")
            
            f.write("\\begintext\n\n")
            
            # Add comments
            f.write(f"This setup file was generated by {self.producer_id}.\n")
            f.write(f"Satellite: {segment.satellite_id}\n")
            f.write(f"NAIF ID: {segment.naif_id}\n")
            if segment.start_time and segment.end_time:
                f.write(f"Coverage: {segment.start_time} to {segment.end_time}\n")
                f.write(f"Duration: {segment.duration_seconds / 3600:.2f} hours\n")
                f.write(f"States: {len(segment.states)}\n")
        
        logger.debug(f"Wrote setup file: {filepath}")
        return filepath
    
    def _write_generate_script(self) -> Path:
        """Write shell script to run mkspk for all satellites."""
        filepath = self.output_dir / "generate_all.sh"
        
        with open(filepath, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("# Generated by SatUpdate SPK Export\n")
            f.write(f"# Created: {datetime.now(timezone.utc).isoformat()}\n")
            f.write("#\n")
            f.write("# This script generates SPK files using NAIF's mkspk tool.\n")
            f.write("# Ensure mkspk is in your PATH or set MKSPK_PATH.\n")
            f.write("#\n")
            f.write("# Usage: ./generate_all.sh\n")
            f.write("#\n\n")
            
            f.write("MKSPK=${MKSPK_PATH:-mkspk}\n\n")
            
            f.write("# Check if mkspk is available\n")
            f.write("if ! command -v $MKSPK &> /dev/null; then\n")
            f.write("    echo \"Error: mkspk not found. Install NAIF Toolkit or set MKSPK_PATH.\"\n")
            f.write("    exit 1\n")
            f.write("fi\n\n")
            
            f.write("cd \"$(dirname \"$0\")\"\n\n")
            
            f.write("echo \"Generating SPK files...\"\n\n")
            
            for segment in self.segments:
                setup_file = f"{segment.satellite_id}_setup.txt"
                f.write(f"echo \"Processing {segment.satellite_id}...\"\n")
                f.write(f"$MKSPK -setup {setup_file} -input {segment.satellite_id}_states.txt\n")
                f.write("if [ $? -ne 0 ]; then\n")
                f.write(f"    echo \"Error generating SPK for {segment.satellite_id}\"\n")
                f.write("    exit 1\n")
                f.write("fi\n\n")
            
            f.write("echo \"Done! Generated SPK files:\"\n")
            for segment in self.segments:
                f.write(f"echo \"  - {segment.satellite_id}.bsp\"\n")
            
            f.write("\n# Optional: Merge all SPK files into one\n")
            f.write("# echo \"Merging SPK files...\"\n")
            f.write("# spkmerge constellation.bsp ")
            f.write(" ".join(f"{s.satellite_id}.bsp" for s in self.segments))
            f.write("\n")
        
        # Make executable
        filepath.chmod(0o755)
        
        logger.debug(f"Wrote generation script: {filepath}")
        return filepath
    
    def _write_naif_mapping(self) -> Path:
        """Write NAIF ID mapping documentation."""
        filepath = self.output_dir / "naif_ids.json"
        
        mapping = {
            "description": "NAIF ID assignments for SatUpdate constellation",
            "created": datetime.now(timezone.utc).isoformat(),
            "center_body": {
                "name": "EARTH",
                "naif_id": EARTH_NAIF_ID
            },
            "satellites": {}
        }
        
        for segment in self.segments:
            mapping["satellites"][segment.satellite_id] = {
                "naif_id": segment.naif_id,
                "reference_frame": segment.reference_frame,
                "num_states": len(segment.states),
                "start_time": segment.start_time.isoformat() if segment.start_time else None,
                "end_time": segment.end_time.isoformat() if segment.end_time else None
            }
        
        with open(filepath, "w") as f:
            json.dump(mapping, f, indent=2)
        
        logger.debug(f"Wrote NAIF ID mapping: {filepath}")
        return filepath
    
    def _write_metadata(self) -> Path:
        """Write export metadata."""
        filepath = self.output_dir / "metadata.json"
        
        metadata = {
            **self._metadata,
            "num_segments": len(self.segments),
            "satellites": [s.satellite_id for s in self.segments],
            "total_states": sum(len(s.states) for s in self.segments)
        }
        
        # Add time bounds across all segments
        start_times = [s.start_time for s in self.segments if s.start_time]
        end_times = [s.end_time for s in self.segments if s.end_time]
        
        if start_times:
            metadata["earliest_epoch"] = min(start_times).isoformat()
        if end_times:
            metadata["latest_epoch"] = max(end_times).isoformat()
        
        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.debug(f"Wrote metadata: {filepath}")
        return filepath
    
    def write_direct(
        self,
        output_file: Union[Path, str],
        leapseconds_kernel: Optional[Union[Path, str]] = None
    ) -> Path:
        """
        Write SPK file directly using SpiceyPy (limited support).
        
        This method uses SpiceyPy's spkw09 to write Type 9 SPK segments.
        For better compatibility, prefer export_for_mkspk().
        
        Parameters
        ----------
        output_file : Path or str
            Output SPK file path
        leapseconds_kernel : Path or str, optional
            Path to leapseconds kernel (required for time conversion)
        
        Returns
        -------
        Path
            Path to created SPK file
        
        Raises
        ------
        ImportError
            If SpiceyPy is not available
        ValueError
            If no valid segments to export
        """
        try:
            import spiceypy as spice
        except ImportError:
            raise ImportError(
                "SpiceyPy required for direct SPK writing. "
                "Install with: pip install spiceypy\n"
                "Or use export_for_mkspk() which doesn't require SpiceyPy."
            )
        
        output_file = Path(output_file)
        
        # Load leapseconds if provided
        if leapseconds_kernel:
            spice.furnsh(str(leapseconds_kernel))
        
        valid_segments = [s for s in self.segments if not s.validate()]
        if not valid_segments:
            raise ValueError("No valid segments to export")
        
        try:
            # Create new SPK file
            handle = spice.spkopn(
                str(output_file),
                f"SatUpdate Constellation SPK",
                1000  # Comment area size
            )
            
            for segment in valid_segments:
                # Convert states to format needed by spkw09
                epochs = []
                states = []
                
                for state in segment.states:
                    # Convert datetime to ET
                    time_str = state.epoch.strftime("%Y-%m-%dT%H:%M:%S.%f")
                    et = spice.str2et(time_str)
                    epochs.append(et)
                    
                    # State vector: [x, y, z, vx, vy, vz]
                    states.append(list(state.position) + list(state.velocity))
                
                # Write Type 9 segment
                spice.spkw09(
                    handle,
                    segment.naif_id,
                    segment.center_body,
                    segment.reference_frame,
                    epochs[0],
                    epochs[-1],
                    f"{segment.satellite_id} ephemeris",
                    7,  # Polynomial degree
                    len(epochs),
                    states,
                    epochs
                )
                
                logger.debug(f"Wrote segment for {segment.satellite_id}")
            
            # Close SPK file
            spice.spkcls(handle)
            
            logger.info(f"Created SPK file: {output_file}")
            
        finally:
            # Unload leapseconds if we loaded it
            if leapseconds_kernel:
                spice.unload(str(leapseconds_kernel))
        
        return output_file
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of generator contents.
        
        Returns
        -------
        Dict[str, Any]
            Summary including segment count, time coverage, etc.
        """
        summary = {
            "output_dir": str(self.output_dir),
            "producer_id": self.producer_id,
            "num_segments": len(self.segments),
            "segments": []
        }
        
        for segment in self.segments:
            seg_info = {
                "satellite_id": segment.satellite_id,
                "naif_id": segment.naif_id,
                "num_states": len(segment.states),
                "center_body": segment.center_body,
                "reference_frame": segment.reference_frame
            }
            
            if segment.start_time:
                seg_info["start_time"] = segment.start_time.isoformat()
            if segment.end_time:
                seg_info["end_time"] = segment.end_time.isoformat()
            if segment.duration_seconds > 0:
                seg_info["duration_hours"] = segment.duration_seconds / 3600
            
            summary["segments"].append(seg_info)
        
        return summary


def create_spk_from_simulation(
    simulation,
    output_dir: Union[Path, str],
    duration_hours: float = 24.0,
    step_seconds: float = 60.0,
    start_time: Optional[datetime] = None
) -> Path:
    """
    Convenience function to create SPK export from a simulation.
    
    Parameters
    ----------
    simulation : Simulation
        Initialized simulation instance
    output_dir : Path or str
        Output directory for SPK files
    duration_hours : float, optional
        Duration in hours (default 24)
    step_seconds : float, optional
        Time step in seconds (default 60)
    start_time : datetime, optional
        Start time (default: now)
    
    Returns
    -------
    Path
        Output directory containing all files
    
    Examples
    --------
    >>> from simulation import Simulation, SimulationConfig
    >>> from tools.generate_spk import create_spk_from_simulation
    >>> 
    >>> sim = Simulation(config)
    >>> sim.initialize()
    >>> output = create_spk_from_simulation(sim, "./spk_output")
    """
    generator = SPKGenerator(output_dir)
    generator.add_from_simulation(
        simulation,
        start_time=start_time,
        duration_hours=duration_hours,
        step_seconds=step_seconds
    )
    return generator.export_for_mkspk()


# Command-line interface
def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Export SatUpdate constellation to SPICE SPK format"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for SPK files"
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        default=24.0,
        help="Duration in hours (default: 24)"
    )
    
    parser.add_argument(
        "--step",
        type=float,
        default=60.0,
        help="Time step in seconds (default: 60)"
    )
    
    parser.add_argument(
        "--constellation",
        choices=["walker_delta", "walker_star", "random"],
        default="walker_delta",
        help="Constellation type (default: walker_delta)"
    )
    
    parser.add_argument(
        "--planes",
        type=int,
        default=3,
        help="Number of orbital planes (default: 3)"
    )
    
    parser.add_argument(
        "--sats-per-plane",
        type=int,
        default=4,
        help="Satellites per plane (default: 4)"
    )
    
    parser.add_argument(
        "--altitude",
        type=float,
        default=550.0,
        help="Orbital altitude in km (default: 550)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s"
    )
    
    # Import simulation modules
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from simulation import Simulation, SimulationConfig, ConstellationType
    except ImportError as e:
        logger.error(f"Could not import simulation module: {e}")
        sys.exit(1)
    
    # Map constellation type
    const_type_map = {
        "walker_delta": ConstellationType.WALKER_DELTA,
        "walker_star": ConstellationType.WALKER_STAR,
        "random": ConstellationType.RANDOM
    }
    
    # Create simulation
    config = SimulationConfig(
        constellation_type=const_type_map[args.constellation],
        num_planes=args.planes,
        sats_per_plane=args.sats_per_plane,
        altitude=args.altitude,
        num_packets=1,  # Not used for SPK export
    )
    
    sim = Simulation(config)
    sim.initialize()
    
    logger.info(
        f"Created {args.constellation} constellation with "
        f"{sim.num_satellites} satellites"
    )
    
    # Generate SPK files
    output_dir = create_spk_from_simulation(
        sim,
        args.output,
        duration_hours=args.duration,
        step_seconds=args.step
    )
    
    logger.info(f"SPK files exported to: {output_dir}")
    logger.info("Run ./generate_all.sh to create SPK binary files with mkspk")


if __name__ == "__main__":
    main()