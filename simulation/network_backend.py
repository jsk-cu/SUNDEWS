#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network Backend Module

Provides an abstract interface for network simulation that allows plugging
in different network models. The default NativeNetworkBackend preserves
existing behavior with instant, perfect packet delivery.

This module implements Step 4 of the NS-3/SPICE integration plan.

Features:
- Abstract NetworkBackend interface for pluggable network models
- NativeNetworkBackend with zero-latency perfect delivery (default)
- Topology-aware packet routing
- Network statistics collection
- Ready for NS-3 backend integration (Steps 5-7)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    List, Dict, Set, Tuple, Optional, Any, Union, Callable
)
import logging
import numpy as np


logger = logging.getLogger(__name__)


class DropReason(Enum):
    """Reasons for packet drops."""
    NONE = "none"
    NO_ROUTE = "no_route"
    LINK_DOWN = "link_down"
    QUEUE_FULL = "queue_full"
    TIMEOUT = "timeout"
    COLLISION = "collision"
    OUT_OF_RANGE = "out_of_range"
    INTERFERENCE = "interference"


@dataclass
class PacketTransfer:
    """
    Represents a packet transfer between two nodes.
    
    Attributes
    ----------
    source_id : str
        ID of the sending node
    destination_id : str
        ID of the receiving node
    packet_id : int
        Unique identifier of the packet content
    timestamp : float
        Simulation time when transfer completed
    success : bool
        Whether the transfer was successful
    latency_ms : float, optional
        Transfer latency in milliseconds (None for instant)
    size_bytes : int
        Packet size in bytes
    dropped_reason : DropReason, optional
        Reason for drop if success is False
    metadata : Dict, optional
        Additional transfer metadata
    """
    source_id: str
    destination_id: str
    packet_id: int
    timestamp: float
    success: bool
    latency_ms: Optional[float] = None
    size_bytes: int = 1024
    dropped_reason: DropReason = DropReason.NONE
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and set defaults."""
        if not self.success and self.dropped_reason == DropReason.NONE:
            self.dropped_reason = DropReason.NO_ROUTE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_id": self.source_id,
            "destination_id": self.destination_id,
            "packet_id": self.packet_id,
            "timestamp": self.timestamp,
            "success": self.success,
            "latency_ms": self.latency_ms,
            "size_bytes": self.size_bytes,
            "dropped_reason": self.dropped_reason.value,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PacketTransfer":
        """Create from dictionary."""
        return cls(
            source_id=data["source_id"],
            destination_id=data["destination_id"],
            packet_id=data["packet_id"],
            timestamp=data["timestamp"],
            success=data["success"],
            latency_ms=data.get("latency_ms"),
            size_bytes=data.get("size_bytes", 1024),
            dropped_reason=DropReason(data.get("dropped_reason", "none")),
            metadata=data.get("metadata", {}),
        )


@dataclass
class NetworkStatistics:
    """
    Network performance statistics.
    
    Attributes
    ----------
    total_packets_sent : int
        Total number of packets sent
    total_packets_received : int
        Total number of packets successfully received
    total_packets_dropped : int
        Total number of packets dropped
    total_bytes_sent : int
        Total bytes sent
    total_bytes_received : int
        Total bytes successfully received
    average_latency_ms : float
        Average packet latency in milliseconds
    min_latency_ms : float
        Minimum observed latency
    max_latency_ms : float
        Maximum observed latency
    throughput_bps : float
        Current throughput in bits per second
    link_utilization : Dict[Tuple[str, str], float]
        Utilization per link (0.0 to 1.0)
    drop_reasons : Dict[DropReason, int]
        Count of drops by reason
    """
    total_packets_sent: int = 0
    total_packets_received: int = 0
    total_packets_dropped: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    average_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    throughput_bps: float = 0.0
    link_utilization: Dict[Tuple[str, str], float] = field(default_factory=dict)
    drop_reasons: Dict[DropReason, int] = field(default_factory=dict)
    
    @property
    def delivery_ratio(self) -> float:
        """Packet delivery ratio (0.0 to 1.0)."""
        if self.total_packets_sent == 0:
            return 1.0
        return self.total_packets_received / self.total_packets_sent
    
    @property
    def drop_ratio(self) -> float:
        """Packet drop ratio (0.0 to 1.0)."""
        if self.total_packets_sent == 0:
            return 0.0
        return self.total_packets_dropped / self.total_packets_sent
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.total_packets_sent = 0
        self.total_packets_received = 0
        self.total_packets_dropped = 0
        self.total_bytes_sent = 0
        self.total_bytes_received = 0
        self.average_latency_ms = 0.0
        self.min_latency_ms = float('inf')
        self.max_latency_ms = 0.0
        self.throughput_bps = 0.0
        self.link_utilization.clear()
        self.drop_reasons.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_packets_sent": self.total_packets_sent,
            "total_packets_received": self.total_packets_received,
            "total_packets_dropped": self.total_packets_dropped,
            "total_bytes_sent": self.total_bytes_sent,
            "total_bytes_received": self.total_bytes_received,
            "average_latency_ms": self.average_latency_ms,
            "min_latency_ms": self.min_latency_ms if self.min_latency_ms != float('inf') else None,
            "max_latency_ms": self.max_latency_ms,
            "throughput_bps": self.throughput_bps,
            "delivery_ratio": self.delivery_ratio,
            "drop_ratio": self.drop_ratio,
            "drop_reasons": {k.value: v for k, v in self.drop_reasons.items()},
        }


@dataclass
class PendingTransfer:
    """
    A packet transfer that is in progress.
    
    Used internally by network backends to track packets in flight.
    """
    source_id: str
    destination_id: str
    packet_id: int
    size_bytes: int
    send_time: float
    expected_arrival: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class NetworkBackend(ABC):
    """
    Abstract base class for network simulation backends.
    
    Defines the interface for network simulation, allowing different
    implementations (native, NS-3, etc.) to be plugged in without
    changing the simulation logic.
    
    All implementations must support:
    - Topology initialization and updates
    - Packet sending and receiving
    - Statistics collection
    
    Examples
    --------
    >>> class MyBackend(NetworkBackend):
    ...     def initialize(self, topology):
    ...         pass
    ...     # ... other methods
    
    >>> backend = MyBackend()
    >>> backend.initialize(topology)
    >>> backend.send_packet("SAT-001", "SAT-002", packet_id=1)
    >>> transfers = backend.step(60.0)
    """
    
    @abstractmethod
    def initialize(self, topology: Dict[str, Any]) -> None:
        """
        Initialize the network with a topology.
        
        Parameters
        ----------
        topology : Dict
            Network topology specification including:
            - nodes: List of node specifications
            - links: List of initial active links
            - config: Network configuration parameters
        """
        pass
    
    @abstractmethod
    def update_topology(self, active_links: Set[Tuple[str, str]]) -> None:
        """
        Update the set of active links.
        
        Called when satellite positions change and link availability
        needs to be updated.
        
        Parameters
        ----------
        active_links : Set[Tuple[str, str]]
            Set of currently active bidirectional links as (node1, node2) tuples
        """
        pass
    
    @abstractmethod
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
            Unique packet identifier
        size_bytes : int, optional
            Packet size in bytes (default 1024)
        metadata : Dict, optional
            Additional packet metadata
        
        Returns
        -------
        bool
            True if packet was queued successfully, False if rejected
        """
        pass
    
    @abstractmethod
    def step(self, timestep: float) -> List[PacketTransfer]:
        """
        Advance network simulation by one timestep.
        
        Processes all queued packets and returns completed transfers.
        
        Parameters
        ----------
        timestep : float
            Time to advance in seconds
        
        Returns
        -------
        List[PacketTransfer]
            List of completed packet transfers (both successful and failed)
        """
        pass
    
    @abstractmethod
    def get_statistics(self) -> NetworkStatistics:
        """
        Get current network statistics.
        
        Returns
        -------
        NetworkStatistics
            Current network performance statistics
        """
        pass
    
    def reset(self) -> None:
        """
        Reset the network backend to initial state.
        
        Default implementation does nothing. Subclasses should override
        if they maintain state that needs to be reset.
        """
        pass
    
    def shutdown(self) -> None:
        """
        Clean up resources.
        
        Called when the backend is no longer needed. Default implementation
        does nothing. Subclasses should override if cleanup is needed.
        """
        pass
    
    def is_link_active(self, node1: str, node2: str) -> bool:
        """
        Check if a link is currently active.
        
        Parameters
        ----------
        node1 : str
            First node ID
        node2 : str
            Second node ID
        
        Returns
        -------
        bool
            True if link is active in either direction
        """
        # Default implementation - subclasses should override for efficiency
        return False
    
    def get_pending_count(self) -> int:
        """
        Get number of packets currently in transit.
        
        Returns
        -------
        int
            Number of pending packet transfers
        """
        return 0
    
    @property
    def name(self) -> str:
        """Backend implementation name."""
        return self.__class__.__name__


class NativeNetworkBackend(NetworkBackend):
    """
    Native network backend with instant, perfect packet delivery.
    
    This is the default backend that preserves existing simulation behavior:
    - Zero latency (instant delivery)
    - Perfect reliability (no drops except for missing links)
    - Unlimited bandwidth
    - Topology-aware (respects line-of-sight and range)
    
    Parameters
    ----------
    allow_unlinked : bool, optional
        If True, allow packets between nodes without active links (default False)
    
    Examples
    --------
    >>> backend = NativeNetworkBackend()
    >>> backend.initialize({"links": [("SAT-001", "SAT-002")]})
    >>> backend.send_packet("SAT-001", "SAT-002", packet_id=1)
    >>> transfers = backend.step(60.0)
    >>> print(transfers[0].success)
    True
    """
    
    def __init__(self, allow_unlinked: bool = False):
        self._active_links: Set[Tuple[str, str]] = set()
        self._pending_transfers: List[PendingTransfer] = []
        self._statistics = NetworkStatistics()
        self._current_time: float = 0.0
        self._allow_unlinked = allow_unlinked
        self._initialized = False
        
        # Latency sum for computing average
        self._latency_sum: float = 0.0
        self._latency_count: int = 0
    
    def initialize(self, topology: Dict[str, Any]) -> None:
        """
        Initialize with topology.
        
        Parameters
        ----------
        topology : Dict
            Topology with optional 'links' key containing initial active links
        """
        self._active_links = set()
        
        # Extract links from topology
        if "links" in topology:
            for link in topology["links"]:
                if isinstance(link, (list, tuple)) and len(link) >= 2:
                    self._active_links.add((link[0], link[1]))
        
        self._pending_transfers.clear()
        self._statistics.reset()
        self._current_time = 0.0
        self._latency_sum = 0.0
        self._latency_count = 0
        self._initialized = True
        
        logger.debug(
            f"NativeNetworkBackend initialized with {len(self._active_links)} links"
        )
    
    def update_topology(self, active_links: Set[Tuple[str, str]]) -> None:
        """
        Update active links.
        
        Parameters
        ----------
        active_links : Set[Tuple[str, str]]
            New set of active bidirectional links
        """
        self._active_links = active_links.copy()
        logger.debug(f"Topology updated: {len(self._active_links)} active links")
    
    def send_packet(
        self,
        source: str,
        destination: str,
        packet_id: int,
        size_bytes: int = 1024,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Queue a packet for instant delivery.
        
        In the native backend, packets are delivered instantly in the next
        step() call if a link exists between source and destination.
        
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
            Additional metadata
        
        Returns
        -------
        bool
            Always True (packets are never rejected at queue time)
        """
        transfer = PendingTransfer(
            source_id=source,
            destination_id=destination,
            packet_id=packet_id,
            size_bytes=size_bytes,
            send_time=self._current_time,
            expected_arrival=self._current_time,  # Instant delivery
            metadata=metadata or {},
        )
        self._pending_transfers.append(transfer)
        return True
    
    def step(self, timestep: float) -> List[PacketTransfer]:
        """
        Process all pending transfers.
        
        In the native backend, all queued packets are delivered instantly
        if a link exists between source and destination.
        
        Parameters
        ----------
        timestep : float
            Time step in seconds (used for time tracking)
        
        Returns
        -------
        List[PacketTransfer]
            List of completed transfers
        """
        self._current_time += timestep
        completed: List[PacketTransfer] = []
        
        for pending in self._pending_transfers:
            # Check if link exists (bidirectional check)
            link_exists = self._has_link(pending.source_id, pending.destination_id)
            
            if link_exists or self._allow_unlinked:
                # Successful delivery
                transfer = PacketTransfer(
                    source_id=pending.source_id,
                    destination_id=pending.destination_id,
                    packet_id=pending.packet_id,
                    timestamp=self._current_time,
                    success=True,
                    latency_ms=0.0,  # Instant delivery
                    size_bytes=pending.size_bytes,
                    dropped_reason=DropReason.NONE,
                    metadata=pending.metadata,
                )
                
                self._statistics.total_packets_received += 1
                self._statistics.total_bytes_received += pending.size_bytes
                
                # Update latency stats (0 for native)
                self._latency_count += 1
                # Average remains 0 for instant delivery
                
            else:
                # No link - packet dropped
                transfer = PacketTransfer(
                    source_id=pending.source_id,
                    destination_id=pending.destination_id,
                    packet_id=pending.packet_id,
                    timestamp=self._current_time,
                    success=False,
                    latency_ms=None,
                    size_bytes=pending.size_bytes,
                    dropped_reason=DropReason.NO_ROUTE,
                    metadata=pending.metadata,
                )
                
                self._statistics.total_packets_dropped += 1
                self._statistics.drop_reasons[DropReason.NO_ROUTE] = \
                    self._statistics.drop_reasons.get(DropReason.NO_ROUTE, 0) + 1
            
            # Update sent statistics
            self._statistics.total_packets_sent += 1
            self._statistics.total_bytes_sent += pending.size_bytes
            
            completed.append(transfer)
        
        # Clear pending transfers
        self._pending_transfers.clear()
        
        return completed
    
    def get_statistics(self) -> NetworkStatistics:
        """
        Get network statistics.
        
        Returns
        -------
        NetworkStatistics
            Current statistics (perfect delivery for native backend)
        """
        return self._statistics
    
    def reset(self) -> None:
        """Reset backend state."""
        self._active_links.clear()
        self._pending_transfers.clear()
        self._statistics.reset()
        self._current_time = 0.0
        self._latency_sum = 0.0
        self._latency_count = 0
    
    def is_link_active(self, node1: str, node2: str) -> bool:
        """Check if link is active (bidirectional check)."""
        return self._has_link(node1, node2)
    
    def get_pending_count(self) -> int:
        """Get number of pending transfers."""
        return len(self._pending_transfers)
    
    def _has_link(self, node1: str, node2: str) -> bool:
        """Check for link in either direction."""
        return (node1, node2) in self._active_links or \
               (node2, node1) in self._active_links
    
    @property
    def active_links(self) -> Set[Tuple[str, str]]:
        """Get current active links."""
        return self._active_links.copy()
    
    @property
    def current_time(self) -> float:
        """Get current simulation time."""
        return self._current_time


class DelayedNetworkBackend(NetworkBackend):
    """
    Network backend with configurable latency.
    
    Extends NativeNetworkBackend with propagation delay based on
    distance between nodes. Useful for testing latency-aware protocols.
    
    Parameters
    ----------
    propagation_speed : float
        Signal propagation speed in km/s (default: speed of light ~299792 km/s)
    processing_delay_ms : float
        Fixed processing delay per hop in milliseconds (default: 0)
    position_provider : Callable, optional
        Function that returns position for a node ID
    
    Examples
    --------
    >>> backend = DelayedNetworkBackend(processing_delay_ms=1.0)
    >>> backend.set_positions({"SAT-001": [7000, 0, 0], "SAT-002": [0, 7000, 0]})
    """
    
    SPEED_OF_LIGHT_KM_S = 299792.458
    
    def __init__(
        self,
        propagation_speed: float = SPEED_OF_LIGHT_KM_S,
        processing_delay_ms: float = 0.0,
        position_provider: Optional[Callable[[str], np.ndarray]] = None
    ):
        self._propagation_speed = propagation_speed
        self._processing_delay_ms = processing_delay_ms
        self._position_provider = position_provider
        
        self._active_links: Set[Tuple[str, str]] = set()
        self._pending_transfers: List[PendingTransfer] = []
        self._statistics = NetworkStatistics()
        self._current_time: float = 0.0
        self._positions: Dict[str, np.ndarray] = {}
        
        self._latency_sum: float = 0.0
        self._latency_count: int = 0
    
    def set_positions(self, positions: Dict[str, Union[List, np.ndarray]]) -> None:
        """
        Set node positions for delay calculation.
        
        Parameters
        ----------
        positions : Dict[str, array-like]
            Mapping of node ID to [x, y, z] position in km
        """
        self._positions = {
            k: np.array(v) if not isinstance(v, np.ndarray) else v
            for k, v in positions.items()
        }
    
    def initialize(self, topology: Dict[str, Any]) -> None:
        """Initialize with topology."""
        self._active_links = set()
        
        if "links" in topology:
            for link in topology["links"]:
                if isinstance(link, (list, tuple)) and len(link) >= 2:
                    self._active_links.add((link[0], link[1]))
        
        # Extract positions from nodes if available
        if "nodes" in topology:
            for node in topology["nodes"]:
                if "id" in node and "position" in node:
                    self._positions[node["id"]] = np.array(node["position"])
        
        self._pending_transfers.clear()
        self._statistics.reset()
        self._current_time = 0.0
        self._latency_sum = 0.0
        self._latency_count = 0
    
    def update_topology(self, active_links: Set[Tuple[str, str]]) -> None:
        """Update active links."""
        self._active_links = active_links.copy()
    
    def send_packet(
        self,
        source: str,
        destination: str,
        packet_id: int,
        size_bytes: int = 1024,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Queue packet with calculated delay."""
        delay_ms = self._calculate_delay(source, destination)
        arrival_time = self._current_time + (delay_ms / 1000.0)
        
        transfer = PendingTransfer(
            source_id=source,
            destination_id=destination,
            packet_id=packet_id,
            size_bytes=size_bytes,
            send_time=self._current_time,
            expected_arrival=arrival_time,
            metadata=metadata or {},
        )
        self._pending_transfers.append(transfer)
        return True
    
    def step(self, timestep: float) -> List[PacketTransfer]:
        """Process transfers that have completed."""
        self._current_time += timestep
        completed: List[PacketTransfer] = []
        remaining: List[PendingTransfer] = []
        
        for pending in self._pending_transfers:
            if pending.expected_arrival <= self._current_time:
                # Check if link still exists
                link_exists = self._has_link(pending.source_id, pending.destination_id)
                
                latency_ms = (pending.expected_arrival - pending.send_time) * 1000.0
                
                if link_exists:
                    transfer = PacketTransfer(
                        source_id=pending.source_id,
                        destination_id=pending.destination_id,
                        packet_id=pending.packet_id,
                        timestamp=self._current_time,
                        success=True,
                        latency_ms=latency_ms,
                        size_bytes=pending.size_bytes,
                        dropped_reason=DropReason.NONE,
                        metadata=pending.metadata,
                    )
                    
                    self._statistics.total_packets_received += 1
                    self._statistics.total_bytes_received += pending.size_bytes
                    
                    # Update latency stats
                    self._latency_sum += latency_ms
                    self._latency_count += 1
                    self._statistics.average_latency_ms = \
                        self._latency_sum / self._latency_count
                    self._statistics.min_latency_ms = min(
                        self._statistics.min_latency_ms, latency_ms
                    )
                    self._statistics.max_latency_ms = max(
                        self._statistics.max_latency_ms, latency_ms
                    )
                else:
                    transfer = PacketTransfer(
                        source_id=pending.source_id,
                        destination_id=pending.destination_id,
                        packet_id=pending.packet_id,
                        timestamp=self._current_time,
                        success=False,
                        latency_ms=None,
                        size_bytes=pending.size_bytes,
                        dropped_reason=DropReason.LINK_DOWN,
                        metadata=pending.metadata,
                    )
                    
                    self._statistics.total_packets_dropped += 1
                    self._statistics.drop_reasons[DropReason.LINK_DOWN] = \
                        self._statistics.drop_reasons.get(DropReason.LINK_DOWN, 0) + 1
                
                self._statistics.total_packets_sent += 1
                self._statistics.total_bytes_sent += pending.size_bytes
                completed.append(transfer)
            else:
                # Still in transit
                remaining.append(pending)
        
        self._pending_transfers = remaining
        return completed
    
    def get_statistics(self) -> NetworkStatistics:
        """Get network statistics."""
        return self._statistics
    
    def reset(self) -> None:
        """Reset backend state."""
        self._active_links.clear()
        self._pending_transfers.clear()
        self._statistics.reset()
        self._current_time = 0.0
        self._latency_sum = 0.0
        self._latency_count = 0
    
    def is_link_active(self, node1: str, node2: str) -> bool:
        """Check if link is active."""
        return self._has_link(node1, node2)
    
    def get_pending_count(self) -> int:
        """Get number of pending transfers."""
        return len(self._pending_transfers)
    
    def _has_link(self, node1: str, node2: str) -> bool:
        """Check for link in either direction."""
        return (node1, node2) in self._active_links or \
               (node2, node1) in self._active_links
    
    def _calculate_delay(self, source: str, destination: str) -> float:
        """
        Calculate propagation delay between nodes.
        
        Returns delay in milliseconds.
        """
        delay_ms = self._processing_delay_ms
        
        # Add propagation delay if positions available
        if source in self._positions and destination in self._positions:
            distance = np.linalg.norm(
                self._positions[source] - self._positions[destination]
            )
            propagation_delay_s = distance / self._propagation_speed
            delay_ms += propagation_delay_s * 1000.0
        elif self._position_provider is not None:
            try:
                pos_src = self._position_provider(source)
                pos_dst = self._position_provider(destination)
                distance = np.linalg.norm(pos_src - pos_dst)
                propagation_delay_s = distance / self._propagation_speed
                delay_ms += propagation_delay_s * 1000.0
            except Exception:
                pass  # Use only processing delay
        
        return delay_ms


def create_native_backend() -> NativeNetworkBackend:
    """
    Create a native network backend with default settings.
    
    Returns
    -------
    NativeNetworkBackend
        Configured backend with instant, perfect delivery
    """
    return NativeNetworkBackend()


def create_delayed_backend(
    propagation_speed: float = DelayedNetworkBackend.SPEED_OF_LIGHT_KM_S,
    processing_delay_ms: float = 0.0
) -> DelayedNetworkBackend:
    """
    Create a delayed network backend.
    
    Parameters
    ----------
    propagation_speed : float
        Signal propagation speed in km/s
    processing_delay_ms : float
        Fixed processing delay per hop in milliseconds
    
    Returns
    -------
    DelayedNetworkBackend
        Configured backend with latency simulation
    """
    return DelayedNetworkBackend(
        propagation_speed=propagation_speed,
        processing_delay_ms=processing_delay_ms
    )