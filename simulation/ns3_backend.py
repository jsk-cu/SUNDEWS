#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NS-3 Network Backend Module

Provides NS-3 integration for high-fidelity network simulation using:
- File Mode: Batch processing via subprocess and JSON files
- Socket Mode: Real-time communication via TCP (Step 6)
- Bindings Mode: Direct Python bindings (Step 7)

This module implements Step 5 of the NS-3/SPICE integration plan.

Features:
- JSON-based communication protocol
- Automatic NS-3 installation detection
- Graceful fallback to mock mode for testing
- Configurable network parameters (data rate, delay, error models)
- Proper cleanup of temporary files
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    List, Dict, Set, Tuple, Optional, Any, Union, Callable
)
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time

import numpy as np

from .network_backend import (
    NetworkBackend,
    PacketTransfer,
    NetworkStatistics,
    PendingTransfer,
    DropReason,
)


logger = logging.getLogger(__name__)


class NS3Mode(Enum):
    """NS-3 communication modes."""
    FILE = "file"       # Subprocess with JSON files
    SOCKET = "socket"   # TCP socket communication
    BINDINGS = "bindings"  # Direct Python bindings
    MOCK = "mock"       # Mock mode for testing


class NS3ErrorModel(Enum):
    """NS-3 error models for link simulation."""
    NONE = "none"
    RATE = "rate"           # Fixed error rate
    BURST = "burst"         # Bursty errors
    GILBERT_ELLIOT = "gilbert_elliot"  # Two-state Markov


class NS3PropagationModel(Enum):
    """NS-3 propagation delay models."""
    CONSTANT_SPEED = "constant_speed"  # Speed of light
    FIXED = "fixed"                    # Fixed delay
    RANDOM = "random"                  # Random delay


@dataclass
class NS3Config:
    """
    Configuration for NS-3 network simulation.
    
    Attributes
    ----------
    data_rate : str
        Link data rate (e.g., "10Mbps", "1Gbps")
    propagation_model : NS3PropagationModel
        Propagation delay model
    error_model : NS3ErrorModel
        Packet error model
    error_rate : float
        Error rate for rate-based error model (0.0 to 1.0)
    queue_size : int
        Queue size in packets
    mtu : int
        Maximum transmission unit in bytes
    propagation_speed : float
        Signal propagation speed in m/s (default: speed of light)
    fixed_delay_ms : float
        Fixed delay in ms (for FIXED propagation model)
    """
    data_rate: str = "10Mbps"
    propagation_model: NS3PropagationModel = NS3PropagationModel.CONSTANT_SPEED
    error_model: NS3ErrorModel = NS3ErrorModel.NONE
    error_rate: float = 0.0
    queue_size: int = 100
    mtu: int = 1500
    propagation_speed: float = 299792458.0  # Speed of light in m/s
    fixed_delay_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "data_rate": self.data_rate,
            "propagation_model": self.propagation_model.value,
            "error_model": self.error_model.value,
            "error_rate": self.error_rate,
            "queue_size": self.queue_size,
            "mtu": self.mtu,
            "propagation_speed": self.propagation_speed,
            "fixed_delay_ms": self.fixed_delay_ms,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NS3Config":
        """Create from dictionary."""
        return cls(
            data_rate=data.get("data_rate", "10Mbps"),
            propagation_model=NS3PropagationModel(
                data.get("propagation_model", "constant_speed")
            ),
            error_model=NS3ErrorModel(data.get("error_model", "none")),
            error_rate=data.get("error_rate", 0.0),
            queue_size=data.get("queue_size", 100),
            mtu=data.get("mtu", 1500),
            propagation_speed=data.get("propagation_speed", 299792458.0),
            fixed_delay_ms=data.get("fixed_delay_ms", 0.0),
        )


@dataclass 
class NS3Node:
    """
    Node specification for NS-3 topology.
    
    Attributes
    ----------
    id : str
        Unique node identifier
    node_type : str
        Node type: "satellite" or "ground"
    position : np.ndarray
        Position [x, y, z] in meters
    """
    id: str
    node_type: str  # "satellite" or "ground"
    position: np.ndarray
    
    def __post_init__(self):
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=float)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.node_type,
            "position": self.position.tolist()
        }


@dataclass
class NS3SendCommand:
    """
    Packet send command for NS-3.
    
    Attributes
    ----------
    source : str
        Source node ID
    destination : str
        Destination node ID
    packet_id : int
        Unique packet identifier
    size : int
        Packet size in bytes
    """
    source: str
    destination: str
    packet_id: int
    size: int = 1024
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "destination": self.destination,
            "packet_id": self.packet_id,
            "size": self.size,
        }


class NS3Backend(NetworkBackend):
    """
    NS-3 network backend for high-fidelity network simulation.
    
    Supports multiple communication modes:
    - FILE: Subprocess with JSON file I/O (default, most compatible)
    - SOCKET: TCP socket for real-time simulation (Step 6)
    - BINDINGS: Direct Python bindings (Step 7)
    - MOCK: Mock mode for testing without NS-3
    
    Parameters
    ----------
    mode : NS3Mode or str
        Communication mode
    ns3_path : Path or str, optional
        Path to NS-3 installation directory
    work_dir : Path or str, optional
        Working directory for temporary files
    config : NS3Config, optional
        Network simulation configuration
    scenario_name : str
        Name of NS-3 scenario to run
    timeout : float
        Subprocess timeout in seconds
    cleanup : bool
        Whether to clean up temporary files
    
    Examples
    --------
    >>> backend = NS3Backend(mode="file", ns3_path="/opt/ns3")
    >>> backend.initialize(topology)
    >>> backend.send_packet("SAT-001", "SAT-002", packet_id=1)
    >>> transfers = backend.step(60.0)
    """
    
    DEFAULT_NS3_PATH = Path("/home/jonathan/ns-allinone-3.45/ns-3.45")
    DEFAULT_SCENARIO = "satellite-update-scenario"
    
    def __init__(
        self,
        mode: Union[NS3Mode, str] = NS3Mode.FILE,
        ns3_path: Optional[Union[Path, str]] = None,
        work_dir: Optional[Union[Path, str]] = None,
        config: Optional[NS3Config] = None,
        scenario_name: str = DEFAULT_SCENARIO,
        timeout: float = 300.0,
        cleanup: bool = True,
    ):
        # Parse mode
        if isinstance(mode, str):
            mode = NS3Mode(mode.lower())
        self._mode = mode
        
        # Paths
        self._ns3_path = Path(ns3_path) if ns3_path else self.DEFAULT_NS3_PATH
        self._work_dir: Optional[Path] = Path(work_dir) if work_dir else None
        self._temp_dir: Optional[Path] = None
        
        # Configuration
        self._config = config or NS3Config()
        self._scenario_name = scenario_name
        self._timeout = timeout
        self._cleanup = cleanup
        
        # State
        self._nodes: Dict[str, NS3Node] = {}
        self._active_links: Set[Tuple[str, str]] = set()
        self._pending_sends: List[NS3SendCommand] = []
        self._statistics = NetworkStatistics()
        self._current_time: float = 0.0
        self._initialized = False
        
        # File paths (set during initialization)
        self._input_file: Optional[Path] = None
        self._output_file: Optional[Path] = None
        
        # Check NS-3 availability
        self._ns3_available = self._check_ns3_installation()
        #assert self._ns3_available
        if not self._ns3_available and mode != NS3Mode.MOCK:
            logger.warning(
                f"NS-3 not found at {self._ns3_path}. "
                f"Using mock mode. Install NS-3 or set ns3_path correctly."
            )
            self._mode = NS3Mode.MOCK
    
    @property
    def mode(self) -> NS3Mode:
        """Current communication mode."""
        return self._mode
    
    @property
    def ns3_available(self) -> bool:
        """Whether NS-3 is available."""
        return self._ns3_available
    
    @property
    def config(self) -> NS3Config:
        """Network configuration."""
        return self._config
    
    def _check_ns3_installation(self) -> bool:
        """
        Check if NS-3 is installed and configured.
        
        Returns
        -------
        bool
            True if NS-3 is available
        """
        ns3_exe = self._ns3_path / "ns3"
        
        if not ns3_exe.exists():
            # Try common alternative locations
            alternatives = [
                self._ns3_path / "build" / "ns3",
                self._ns3_path / "waf",
            ]
            for alt in alternatives:
                if alt.exists():
                    return True
            return False
        
        # Try running ns3 --version
        try:
            result = subprocess.run(
                [str(ns3_exe), "show", "version"],  # CORRECT
                capture_output=True,
                timeout=10,
                cwd=str(ns3_exe.parent)  # Run from NS-3 directory
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
            return False
    
    def _setup_work_dir(self) -> None:
        """Set up working directory for file I/O."""
        if self._work_dir:
            self._work_dir.mkdir(parents=True, exist_ok=True)
            self._input_file = self._work_dir / "ns3_input.json"
            self._output_file = self._work_dir / "ns3_output.json"
        else:
            # Create temporary directory
            self._temp_dir = Path(tempfile.mkdtemp(prefix="ns3_satupdate_"))
            self._input_file = self._temp_dir / "input.json"
            self._output_file = self._temp_dir / "output.json"
    
    def initialize(self, topology: Dict[str, Any]) -> None:
        """
        Initialize backend with network topology.
        
        Parameters
        ----------
        topology : Dict
            Topology specification with:
            - nodes: List of node specs (id, type, position)
            - links: List of active links
            - config: Optional NS3Config overrides
        """
        self._setup_work_dir()
        
        # Parse nodes
        self._nodes.clear()
        if "nodes" in topology:
            for node_data in topology["nodes"]:
                node = NS3Node(
                    id=node_data["id"],
                    node_type=node_data.get("type", "satellite"),
                    position=np.array(node_data.get("position", [0, 0, 0]))
                )
                self._nodes[node.id] = node
        
        # Parse links
        self._active_links.clear()
        if "links" in topology:
            for link in topology["links"]:
                if isinstance(link, (list, tuple)) and len(link) >= 2:
                    self._active_links.add((link[0], link[1]))
        
        # Parse config overrides
        if "config" in topology:
            self._config = NS3Config.from_dict({
                **self._config.to_dict(),
                **topology["config"]
            })
        
        self._pending_sends.clear()
        self._statistics.reset()
        self._current_time = 0.0
        self._initialized = True
        
        # Send initialization command to NS-3 (file mode)
        if self._mode == NS3Mode.FILE and self._ns3_available:
            self._send_init_command()
        
        logger.debug(
            f"NS3Backend initialized: {len(self._nodes)} nodes, "
            f"{len(self._active_links)} links, mode={self._mode.value}"
        )
    
    def update_topology(self, active_links: Set[Tuple[str, str]]) -> None:
        """Update active links."""
        self._active_links = active_links.copy()
    
    def update_positions(self, positions: Dict[str, np.ndarray]) -> None:
        """
        Update node positions.
        
        Parameters
        ----------
        positions : Dict[str, np.ndarray]
            Mapping of node ID to [x, y, z] position in km
        """
        for node_id, position in positions.items():
            if node_id in self._nodes:
                # Convert km to meters for NS-3
                self._nodes[node_id].position = np.array(position) * 1000.0
            else:
                # Create new node
                self._nodes[node_id] = NS3Node(
                    id=node_id,
                    node_type="satellite",
                    position=np.array(position) * 1000.0
                )
    
    def send_packet(
        self,
        source: str,
        destination: str,
        packet_id: int,
        size_bytes: int = 1024,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Queue a packet for transmission.
        
        Parameters
        ----------
        source : str
            Source node ID
        destination : str  
            Destination node ID
        packet_id : int
            Packet content identifier
        size_bytes : int
            Packet size in bytes
        metadata : Dict, optional
            Additional metadata (ignored in NS-3 mode)
        
        Returns
        -------
        bool
            True if packet was queued
        """
        cmd = NS3SendCommand(
            source=source,
            destination=destination,
            packet_id=packet_id,
            size=size_bytes
        )
        self._pending_sends.append(cmd)
        return True
    
    def step(self, timestep: float) -> List[PacketTransfer]:
        """
        Advance simulation and process transfers.
        
        Parameters
        ----------
        timestep : float
            Time step in seconds
        
        Returns
        -------
        List[PacketTransfer]
            Completed packet transfers
        """
        self._current_time += timestep
        
        if self._mode == NS3Mode.MOCK:
            return self._step_mock(timestep)
        elif self._mode == NS3Mode.FILE:
            return self._step_file_mode(timestep)
        elif self._mode == NS3Mode.SOCKET:
            return self._step_socket_mode(timestep)
        elif self._mode == NS3Mode.BINDINGS:
            return self._step_bindings_mode(timestep)
        else:
            raise ValueError(f"Unknown mode: {self._mode}")
    
    def _step_mock(self, timestep: float) -> List[PacketTransfer]:
        """
        Mock step for testing without NS-3.
        
        Simulates realistic latency based on node positions and
        applies configured error model.
        """
        transfers: List[PacketTransfer] = []
        
        for cmd in self._pending_sends:
            # Check link exists
            link_exists = self._has_link(cmd.source, cmd.destination)
            
            # Calculate latency if positions available
            latency_ms = self._calculate_mock_latency(cmd.source, cmd.destination)
            
            # Apply error model
            success = link_exists and self._apply_error_model()
            
            if success:
                transfer = PacketTransfer(
                    source_id=cmd.source,
                    destination_id=cmd.destination,
                    packet_id=cmd.packet_id,
                    timestamp=self._current_time,
                    success=True,
                    latency_ms=latency_ms,
                    size_bytes=cmd.size,
                    dropped_reason=DropReason.NONE,
                )
                self._statistics.total_packets_received += 1
                self._statistics.total_bytes_received += cmd.size
            else:
                drop_reason = DropReason.NO_ROUTE if not link_exists else DropReason.INTERFERENCE
                transfer = PacketTransfer(
                    source_id=cmd.source,
                    destination_id=cmd.destination,
                    packet_id=cmd.packet_id,
                    timestamp=self._current_time,
                    success=False,
                    latency_ms=None,
                    size_bytes=cmd.size,
                    dropped_reason=drop_reason,
                )
                self._statistics.total_packets_dropped += 1
                self._statistics.drop_reasons[drop_reason] = \
                    self._statistics.drop_reasons.get(drop_reason, 0) + 1
            
            self._statistics.total_packets_sent += 1
            self._statistics.total_bytes_sent += cmd.size
            transfers.append(transfer)
        
        self._pending_sends.clear()
        return transfers
    
    def _step_file_mode(self, timestep: float) -> List[PacketTransfer]:
        """
        Execute step using file-based communication.
        
        1. Write input JSON file
        2. Invoke NS-3 subprocess
        3. Read output JSON file
        4. Parse and return transfers
        """
        if not self._ns3_available:
            logger.warning("NS-3 not available, falling back to mock mode")
            return self._step_mock(timestep)
        
        # Write input file
        input_data = self._create_input_data(timestep)
        with open(self._input_file, 'w') as f:
            json.dump(input_data, f, indent=2)
        
        # Build command
        cmd = self._build_ns3_command()
        
        try:
            # Run NS-3 scenario
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=str(self._ns3_path)
            )
            
            if result.returncode != 0:
                logger.error(f"NS-3 failed: {result.stderr}")
                return self._step_mock(timestep)
            
            # Read output file
            if not self._output_file.exists():
                logger.error("NS-3 did not create output file")
                return self._step_mock(timestep)
            
            with open(self._output_file) as f:
                output_data = json.load(f)
            
            # Parse transfers
            return self._parse_output(output_data)
            
        except subprocess.TimeoutExpired:
            logger.error(f"NS-3 timed out after {self._timeout}s")
            return self._step_mock(timestep)
        except Exception as e:
            logger.error(f"NS-3 error: {e}")
            return self._step_mock(timestep)
        finally:
            self._pending_sends.clear()
    
    def _step_socket_mode(self, timestep: float) -> List[PacketTransfer]:
        """Socket mode - implemented in Step 6."""
        raise NotImplementedError(
            "Socket mode not yet implemented. Use file mode or mock mode."
        )
    
    def _step_bindings_mode(self, timestep: float) -> List[PacketTransfer]:
        """Bindings mode - implemented in Step 7."""
        raise NotImplementedError(
            "Bindings mode not yet implemented. Use file mode or mock mode."
        )
    
    def _create_input_data(self, timestep: float) -> Dict[str, Any]:
        """Create input JSON data for NS-3."""
        return {
            "command": "step",
            "timestep": timestep,
            "simulation_time": self._current_time,
            "topology": {
                "nodes": [node.to_dict() for node in self._nodes.values()],
                "links": list(self._active_links),
            },
            "sends": [cmd.to_dict() for cmd in self._pending_sends],
            "config": self._config.to_dict(),
        }
    
    def _build_ns3_command(self) -> List[str]:
        """Build NS-3 subprocess command."""
        ns3_exe = self._ns3_path / "ns3"
        
        # Build scenario arguments
        scenario_args = (
            f"{self._scenario_name} "
            f"--input={self._input_file} "
            f"--output={self._output_file}"
        )
        
        return [str(ns3_exe), "run", scenario_args]
    
    def _parse_output(self, output_data: Dict[str, Any]) -> List[PacketTransfer]:
        """Parse NS-3 output JSON into PacketTransfer objects."""
        transfers: List[PacketTransfer] = []
        
        if output_data.get("status") != "success":
            logger.warning(f"NS-3 reported error: {output_data.get('error')}")
        
        for transfer_data in output_data.get("transfers", []):
            # Map dropped_reason string to enum
            drop_reason = DropReason.NONE
            if not transfer_data.get("success", True):
                reason_str = transfer_data.get("dropped_reason", "no_route")
                try:
                    drop_reason = DropReason(reason_str)
                except ValueError:
                    drop_reason = DropReason.NO_ROUTE
            
            transfer = PacketTransfer(
                source_id=transfer_data["source"],
                destination_id=transfer_data["destination"],
                packet_id=transfer_data["packet_id"],
                timestamp=transfer_data.get("timestamp", self._current_time),
                success=transfer_data.get("success", True),
                latency_ms=transfer_data.get("latency_ms"),
                size_bytes=transfer_data.get("size", 1024),
                dropped_reason=drop_reason,
            )
            transfers.append(transfer)
            
            # Update statistics
            self._statistics.total_packets_sent += 1
            self._statistics.total_bytes_sent += transfer.size_bytes
            
            if transfer.success:
                self._statistics.total_packets_received += 1
                self._statistics.total_bytes_received += transfer.size_bytes
            else:
                self._statistics.total_packets_dropped += 1
                self._statistics.drop_reasons[drop_reason] = \
                    self._statistics.drop_reasons.get(drop_reason, 0) + 1
        
        # Update statistics from NS-3 output
        if "statistics" in output_data:
            ns3_stats = output_data["statistics"]
            if "average_latency_ms" in ns3_stats:
                self._statistics.average_latency_ms = ns3_stats["average_latency_ms"]
            if "throughput_bps" in ns3_stats:
                self._statistics.throughput_bps = ns3_stats["throughput_bps"]
        
        return transfers
    
    def _send_init_command(self) -> None:
        """Send initialization command to NS-3 (for persistent modes)."""
        init_data = {
            "command": "initialize",
            "topology": {
                "nodes": [node.to_dict() for node in self._nodes.values()],
                "links": list(self._active_links),
            },
            "config": self._config.to_dict(),
        }
        
        with open(self._input_file, 'w') as f:
            json.dump(init_data, f, indent=2)
    
    def _has_link(self, node1: str, node2: str) -> bool:
        """Check if link exists (bidirectional)."""
        return (node1, node2) in self._active_links or \
               (node2, node1) in self._active_links
    
    def _calculate_mock_latency(self, source: str, destination: str) -> float:
        """
        Calculate mock latency based on positions.
        
        Returns latency in milliseconds.
        """
        if self._config.propagation_model == NS3PropagationModel.FIXED:
            return self._config.fixed_delay_ms
        
        # Get positions
        if source not in self._nodes or destination not in self._nodes:
            return 0.0
        
        src_pos = self._nodes[source].position
        dst_pos = self._nodes[destination].position
        
        # Calculate distance (positions in meters)
        distance_m = np.linalg.norm(src_pos - dst_pos)
        
        # Propagation delay
        delay_s = distance_m / self._config.propagation_speed
        delay_ms = delay_s * 1000.0
        
        # Add random component if configured
        if self._config.propagation_model == NS3PropagationModel.RANDOM:
            delay_ms *= np.random.uniform(0.9, 1.1)
        
        return delay_ms
    
    def _apply_error_model(self) -> bool:
        """
        Apply error model to determine if packet succeeds.
        
        Returns True if packet should succeed.
        """
        if self._config.error_model == NS3ErrorModel.NONE:
            return True
        elif self._config.error_model == NS3ErrorModel.RATE:
            return np.random.random() > self._config.error_rate
        else:
            # Default: no error
            return True
    
    def get_statistics(self) -> NetworkStatistics:
        """Get network statistics."""
        return self._statistics
    
    def reset(self) -> None:
        """Reset backend state."""
        self._nodes.clear()
        self._active_links.clear()
        self._pending_sends.clear()
        self._statistics.reset()
        self._current_time = 0.0
    
    def shutdown(self) -> None:
        """Clean up resources."""
        if self._cleanup and self._temp_dir and self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir)
                logger.debug(f"Cleaned up temp directory: {self._temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp dir: {e}")
    
    def __del__(self):
        """Destructor - ensure cleanup."""
        self.shutdown()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False
    
    def is_link_active(self, node1: str, node2: str) -> bool:
        """Check if link is active."""
        return self._has_link(node1, node2)
    
    def get_pending_count(self) -> int:
        """Get number of pending transfers."""
        return len(self._pending_sends)
    
    @property
    def name(self) -> str:
        """Backend name."""
        return f"NS3Backend({self._mode.value})"


def check_ns3_available(ns3_path: Optional[Union[Path, str]] = None) -> bool:
    """
    Check if NS-3 is available.
    
    Parameters
    ----------
    ns3_path : Path or str, optional
        Path to NS-3 installation
    
    Returns
    -------
    bool
        True if NS-3 is available
    """
    backend = NS3Backend(mode="mock", ns3_path=ns3_path)
    return backend._check_ns3_installation()


def create_ns3_backend(
    mode: str = "file",
    ns3_path: Optional[Union[Path, str]] = None,
    config: Optional[NS3Config] = None,
    **kwargs
) -> NS3Backend:
    """
    Create an NS-3 network backend.
    
    Parameters
    ----------
    mode : str
        Communication mode: "file", "socket", "bindings", or "mock"
    ns3_path : Path or str, optional
        Path to NS-3 installation
    config : NS3Config, optional
        Network configuration
    **kwargs
        Additional arguments passed to NS3Backend
    
    Returns
    -------
    NS3Backend
        Configured NS-3 backend
    """
    return NS3Backend(
        mode=mode,
        ns3_path=ns3_path,
        config=config,
        **kwargs
    )


# Export convenience function for checking NS-3
def is_ns3_available() -> bool:
    """Check if NS-3 is available at the default location."""
    return check_ns3_available()