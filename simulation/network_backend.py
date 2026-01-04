#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network Backend Module

Provides abstract interface and implementations for network simulation backends.
This module implements Step 4 of the NS-3/SPICE integration plan.

Features:
- Abstract NetworkBackend interface
- NativeNetworkBackend for instant delivery
- DelayedNetworkBackend for latency simulation
- Statistics tracking and reporting
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Set, Tuple, Optional, Any

import numpy as np


class DropReason(Enum):
    """Reasons for packet drops."""
    NONE = "none"
    NO_ROUTE = "no_route"
    LINK_DOWN = "link_down"
    QUEUE_FULL = "queue_full"
    TIMEOUT = "timeout"
    ERROR = "error"
    COLLISION = "collision"


@dataclass
class PacketTransfer:
    """
    Record of a completed packet transfer.
    
    Attributes
    ----------
    source_id : str
        Source node identifier
    destination_id : str
        Destination node identifier
    packet_id : int
        Unique packet content identifier
    timestamp : float
        Simulation time when transfer completed
    success : bool
        Whether transfer succeeded
    latency_ms : float, optional
        Transfer latency in milliseconds (None if not measured)
    size_bytes : int
        Packet size in bytes (default 1024)
    drop_reason : DropReason, optional
        Reason for drop if not successful
    metadata : Dict, optional
        Additional metadata
    """
    source_id: str
    destination_id: str
    packet_id: int
    timestamp: float
    success: bool
    latency_ms: Optional[float] = None
    size_bytes: int = 1024
    drop_reason: Optional[DropReason] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set default drop reason based on success status."""
        if not self.success and self.drop_reason is None:
            self.drop_reason = DropReason.NO_ROUTE
        if self.success and self.drop_reason is None:
            self.drop_reason = DropReason.NONE
    
    @property
    def dropped_reason(self) -> DropReason:
        """Alias for drop_reason for backward compatibility."""
        return self.drop_reason if self.drop_reason else DropReason.NONE
    
    @dropped_reason.setter
    def dropped_reason(self, value: DropReason):
        """Set drop_reason via alias."""
        self.drop_reason = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "destination_id": self.destination_id,
            "source": self.source_id,  # Backward compatibility
            "destination": self.destination_id,  # Backward compatibility
            "packet_id": self.packet_id,
            "timestamp": self.timestamp,
            "success": self.success,
            "latency_ms": self.latency_ms,
            "size_bytes": self.size_bytes,
            "drop_reason": self.drop_reason.value if self.drop_reason else None,
            "dropped_reason": self.drop_reason.value if self.drop_reason else None,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PacketTransfer":
        """Create from dictionary."""
        drop_reason = None
        if data.get("drop_reason") or data.get("dropped_reason"):
            reason_str = data.get("drop_reason") or data.get("dropped_reason")
            try:
                drop_reason = DropReason(reason_str)
            except ValueError:
                drop_reason = DropReason.ERROR
        
        return cls(
            source_id=data.get("source_id", data.get("source", "")),
            destination_id=data.get("destination_id", data.get("destination", "")),
            packet_id=data.get("packet_id", 0),
            timestamp=data.get("timestamp", 0.0),
            success=data.get("success", True),
            latency_ms=data.get("latency_ms"),
            size_bytes=data.get("size_bytes", data.get("size", 1024)),
            drop_reason=drop_reason,
            metadata=data.get("metadata", {}),
        )


class NetworkStatistics:
    """
    Network performance statistics.
    
    Tracks packets sent, received, dropped, and latency metrics.
    
    Accepts both `avg_latency_ms` and `average_latency_ms` as parameters
    for backward compatibility.
    """
    
    def __init__(
        self,
        total_packets_sent: int = 0,
        total_packets_received: int = 0,
        total_packets_dropped: int = 0,
        total_bytes_sent: int = 0,
        total_bytes_received: int = 0,
        min_latency_ms: float = float('inf'),
        max_latency_ms: float = 0.0,
        avg_latency_ms: float = 0.0,
        average_latency_ms: Optional[float] = None,  # Alias parameter
        _latency_sum: float = 0.0,
        _latency_count: int = 0,
    ):
        self.total_packets_sent = total_packets_sent
        self.total_packets_received = total_packets_received
        self.total_packets_dropped = total_packets_dropped
        self.total_bytes_sent = total_bytes_sent
        self.total_bytes_received = total_bytes_received
        self.min_latency_ms = min_latency_ms
        self.max_latency_ms = max_latency_ms
        
        # Handle alias: average_latency_ms takes precedence if provided
        if average_latency_ms is not None:
            self.avg_latency_ms = average_latency_ms
        else:
            self.avg_latency_ms = avg_latency_ms
        
        self._latency_sum = _latency_sum
        self._latency_count = _latency_count
    
    @property
    def average_latency_ms(self) -> float:
        """Alias for avg_latency_ms for backward compatibility."""
        return self.avg_latency_ms
    
    @average_latency_ms.setter
    def average_latency_ms(self, value: float):
        """Set avg_latency_ms via alias."""
        self.avg_latency_ms = value
    
    def record_send(self, size_bytes: int) -> None:
        """Record a packet send."""
        self.total_packets_sent += 1
        self.total_bytes_sent += size_bytes
    
    def record_receive(self, size_bytes: int, latency_ms: float) -> None:
        """Record a successful packet receive."""
        self.total_packets_received += 1
        self.total_bytes_received += size_bytes
        
        self._latency_sum += latency_ms
        self._latency_count += 1
        
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        self.avg_latency_ms = self._latency_sum / self._latency_count
    
    def record_drop(self) -> None:
        """Record a packet drop."""
        self.total_packets_dropped += 1
    
    @property
    def delivery_ratio(self) -> float:
        """Ratio of received to sent packets."""
        if self.total_packets_sent == 0:
            return 1.0
        return self.total_packets_received / self.total_packets_sent
    
    @property
    def drop_ratio(self) -> float:
        """Ratio of dropped to sent packets."""
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
        self.min_latency_ms = float('inf')
        self.max_latency_ms = 0.0
        self.avg_latency_ms = 0.0
        self._latency_sum = 0.0
        self._latency_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_packets_sent": self.total_packets_sent,
            "total_packets_received": self.total_packets_received,
            "total_packets_dropped": self.total_packets_dropped,
            "total_bytes_sent": self.total_bytes_sent,
            "total_bytes_received": self.total_bytes_received,
            "min_latency_ms": self.min_latency_ms if self.min_latency_ms != float('inf') else 0,
            "max_latency_ms": self.max_latency_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "average_latency_ms": self.avg_latency_ms,  # Alias
            "delivery_ratio": self.delivery_ratio,
            "drop_ratio": self.drop_ratio,
        }


@dataclass
class PendingTransfer:
    """Internal tracking of in-flight packets."""
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
    
    Defines the interface for simulating network packet transfers.
    Implementations may use different simulation approaches
    (instant delivery, delayed delivery, NS-3 simulation, etc.).
    """
    
    @abstractmethod
    def initialize(self, topology: Dict[str, Any]) -> None:
        """
        Initialize backend with network topology.
        
        Parameters
        ----------
        topology : Dict
            Network topology with nodes and links
        """
        pass
    
    @abstractmethod
    def update_topology(self, active_links: Set[Tuple[str, str]]) -> None:
        """
        Update the set of active links.
        
        Parameters
        ----------
        active_links : Set[Tuple[str, str]]
            Set of (node1, node2) tuples for active links
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
            Packet content identifier
        size_bytes : int
            Packet size in bytes
        metadata : Dict, optional
            Additional metadata
        
        Returns
        -------
        bool
            True if packet was queued successfully
        """
        pass
    
    @abstractmethod
    def step(self, timestep: float) -> List[PacketTransfer]:
        """
        Advance simulation and return completed transfers.
        
        Parameters
        ----------
        timestep : float
            Time step in seconds
        
        Returns
        -------
        List[PacketTransfer]
            Completed packet transfers
        """
        pass
    
    @abstractmethod
    def get_statistics(self) -> NetworkStatistics:
        """
        Get network performance statistics.
        
        Returns
        -------
        NetworkStatistics
            Current statistics
        """
        pass
    
    def reset(self) -> None:
        """Reset backend state."""
        pass
    
    def shutdown(self) -> None:
        """Clean up resources."""
        pass
    
    def is_link_active(self, node1: str, node2: str) -> bool:
        """Check if a link is currently active."""
        return False
    
    def get_pending_count(self) -> int:
        """Get number of packets currently in transit."""
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
    - Topology-aware (respects active links)
    """
    
    def __init__(self, allow_unlinked: bool = False):
        self._active_links: Set[Tuple[str, str]] = set()
        self._pending_transfers: List[PendingTransfer] = []
        self._statistics = NetworkStatistics()
        self._current_time: float = 0.0
        self._allow_unlinked = allow_unlinked
        self._initialized = False
    
    @property
    def active_links(self) -> Set[Tuple[str, str]]:
        """Get the set of active links."""
        return self._active_links
    
    def initialize(self, topology: Dict[str, Any]) -> None:
        """Initialize with topology."""
        links = topology.get("links", [])
        self._active_links = set()
        for link in links:
            if isinstance(link, (list, tuple)) and len(link) == 2:
                self._active_links.add((link[0], link[1]))
        self._initialized = True
    
    def update_topology(self, active_links: Set[Tuple[str, str]]) -> None:
        """Update active links."""
        self._active_links = active_links
    
    def _has_link(self, node1: str, node2: str) -> bool:
        """Check if link exists (bidirectional)."""
        return (node1, node2) in self._active_links or \
               (node2, node1) in self._active_links
    
    def send_packet(
        self,
        source: str,
        destination: str,
        packet_id: int,
        size_bytes: int = 1024,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Queue a packet for transmission."""
        self._pending_transfers.append(PendingTransfer(
            source_id=source,
            destination_id=destination,
            packet_id=packet_id,
            size_bytes=size_bytes,
            send_time=self._current_time,
            expected_arrival=self._current_time,  # Instant delivery
            metadata=metadata or {},
        ))
        self._statistics.record_send(size_bytes)
        return True
    
    def step(self, timestep: float) -> List[PacketTransfer]:
        """Process pending transfers."""
        self._current_time += timestep
        transfers = []
        
        for pending in self._pending_transfers:
            # Check if link exists
            has_link = self._has_link(pending.source_id, pending.destination_id)
            
            if has_link or self._allow_unlinked:
                transfer = PacketTransfer(
                    source_id=pending.source_id,
                    destination_id=pending.destination_id,
                    packet_id=pending.packet_id,
                    timestamp=self._current_time,
                    success=True,
                    latency_ms=0.0,
                    size_bytes=pending.size_bytes,
                    drop_reason=DropReason.NONE,
                    metadata=pending.metadata,
                )
                self._statistics.record_receive(pending.size_bytes, 0.0)
            else:
                transfer = PacketTransfer(
                    source_id=pending.source_id,
                    destination_id=pending.destination_id,
                    packet_id=pending.packet_id,
                    timestamp=self._current_time,
                    success=False,
                    latency_ms=0.0,
                    size_bytes=pending.size_bytes,
                    drop_reason=DropReason.NO_ROUTE,
                    metadata=pending.metadata,
                )
                self._statistics.record_drop()
            
            transfers.append(transfer)
        
        self._pending_transfers.clear()
        return transfers
    
    def get_statistics(self) -> NetworkStatistics:
        """Get statistics."""
        return self._statistics
    
    def reset(self) -> None:
        """Reset state."""
        self._pending_transfers.clear()
        self._active_links.clear()
        self._statistics.reset()
        self._current_time = 0.0
    
    def is_link_active(self, node1: str, node2: str) -> bool:
        """Check if link is active."""
        return self._has_link(node1, node2)
    
    def get_pending_count(self) -> int:
        """Get pending transfer count."""
        return len(self._pending_transfers)


class DelayedNetworkBackend(NetworkBackend):
    """
    Network backend with propagation delay simulation.
    
    Simulates realistic latency based on distance between nodes
    or a fixed processing delay.
    
    Parameters
    ----------
    propagation_speed : float
        Speed of signal propagation in m/s (default: speed of light)
    fixed_delay_ms : float
        Fixed delay added to all transfers in milliseconds
    processing_delay_ms : float
        Alias for fixed_delay_ms for backward compatibility
    """
    
    def __init__(
        self,
        propagation_speed: float = 299792458.0,  # Speed of light m/s
        fixed_delay_ms: float = 0.0,
        processing_delay_ms: Optional[float] = None,
    ):
        self._active_links: Set[Tuple[str, str]] = set()
        self._node_positions: Dict[str, np.ndarray] = {}
        self._pending_transfers: List[PendingTransfer] = []
        self._statistics = NetworkStatistics()
        self._current_time: float = 0.0
        self._propagation_speed = propagation_speed
        # processing_delay_ms is an alias for fixed_delay_ms
        self._fixed_delay_ms = processing_delay_ms if processing_delay_ms is not None else fixed_delay_ms
        self._initialized = False
    
    @property
    def active_links(self) -> Set[Tuple[str, str]]:
        """Get the set of active links."""
        return self._active_links
    
    def initialize(self, topology: Dict[str, Any]) -> None:
        """Initialize with topology."""
        # Parse links
        links = topology.get("links", [])
        self._active_links = set()
        for link in links:
            if isinstance(link, (list, tuple)) and len(link) == 2:
                self._active_links.add((link[0], link[1]))
        
        # Parse node positions
        nodes = topology.get("nodes", [])
        self._node_positions = {}
        for node in nodes:
            node_id = node.get("id", "")
            position = node.get("position", [0, 0, 0])
            self._node_positions[node_id] = np.array(position, dtype=float)
        
        self._initialized = True
    
    def update_topology(self, active_links: Set[Tuple[str, str]]) -> None:
        """Update active links."""
        self._active_links = active_links
    
    def set_positions(self, positions: Dict[str, np.ndarray]) -> None:
        """
        Set node positions for distance-based latency calculation.
        
        Parameters
        ----------
        positions : Dict[str, np.ndarray]
            Dictionary mapping node IDs to position vectors (in km)
        """
        for node_id, pos in positions.items():
            # Convert km to meters for propagation calculation
            self._node_positions[node_id] = np.array(pos, dtype=float) * 1000.0
    
    def _has_link(self, node1: str, node2: str) -> bool:
        """Check if link exists."""
        return (node1, node2) in self._active_links or \
               (node2, node1) in self._active_links
    
    def _calculate_latency(self, source: str, destination: str) -> float:
        """Calculate propagation latency in milliseconds."""
        latency = self._fixed_delay_ms
        
        # Add distance-based delay if positions are known
        src_pos = self._node_positions.get(source)
        dst_pos = self._node_positions.get(destination)
        
        if src_pos is not None and dst_pos is not None:
            distance_m = np.linalg.norm(dst_pos - src_pos)
            delay_s = distance_m / self._propagation_speed
            latency += delay_s * 1000.0
        
        return latency
    
    def send_packet(
        self,
        source: str,
        destination: str,
        packet_id: int,
        size_bytes: int = 1024,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Queue a packet for transmission."""
        latency_ms = self._calculate_latency(source, destination)
        arrival_time = self._current_time + (latency_ms / 1000.0)
        
        self._pending_transfers.append(PendingTransfer(
            source_id=source,
            destination_id=destination,
            packet_id=packet_id,
            size_bytes=size_bytes,
            send_time=self._current_time,
            expected_arrival=arrival_time,
            metadata=metadata or {},
        ))
        self._statistics.record_send(size_bytes)
        return True
    
    def step(self, timestep: float) -> List[PacketTransfer]:
        """Process transfers that arrive during this timestep."""
        self._current_time += timestep
        transfers = []
        remaining = []
        
        for pending in self._pending_transfers:
            if pending.expected_arrival <= self._current_time:
                has_link = self._has_link(pending.source_id, pending.destination_id)
                latency_ms = (pending.expected_arrival - pending.send_time) * 1000.0
                
                if has_link:
                    transfer = PacketTransfer(
                        source_id=pending.source_id,
                        destination_id=pending.destination_id,
                        packet_id=pending.packet_id,
                        timestamp=pending.expected_arrival,
                        success=True,
                        latency_ms=latency_ms,
                        size_bytes=pending.size_bytes,
                        drop_reason=DropReason.NONE,
                        metadata=pending.metadata,
                    )
                    self._statistics.record_receive(pending.size_bytes, latency_ms)
                else:
                    transfer = PacketTransfer(
                        source_id=pending.source_id,
                        destination_id=pending.destination_id,
                        packet_id=pending.packet_id,
                        timestamp=self._current_time,
                        success=False,
                        latency_ms=latency_ms,
                        size_bytes=pending.size_bytes,
                        drop_reason=DropReason.LINK_DOWN,
                        metadata=pending.metadata,
                    )
                    self._statistics.record_drop()
                
                transfers.append(transfer)
            else:
                remaining.append(pending)
        
        self._pending_transfers = remaining
        return transfers
    
    def get_statistics(self) -> NetworkStatistics:
        """Get statistics."""
        return self._statistics
    
    def reset(self) -> None:
        """Reset state."""
        self._pending_transfers.clear()
        self._statistics.reset()
        self._current_time = 0.0
    
    def is_link_active(self, node1: str, node2: str) -> bool:
        """Check if link is active."""
        return self._has_link(node1, node2)
    
    def get_pending_count(self) -> int:
        """Get pending transfer count."""
        return len(self._pending_transfers)


def create_native_backend(**kwargs) -> NativeNetworkBackend:
    """Create a native network backend."""
    return NativeNetworkBackend(**kwargs)


def create_delayed_backend(**kwargs) -> DelayedNetworkBackend:
    """Create a delayed network backend."""
    return DelayedNetworkBackend(**kwargs)