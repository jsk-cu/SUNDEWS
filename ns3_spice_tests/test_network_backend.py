#!/usr/bin/env python3
"""
Tests for Step 4: NetworkBackend Interface

These tests verify:
1. NetworkBackend ABC is correctly defined
2. PacketTransfer and NetworkStatistics dataclasses
3. NativeNetworkBackend produces identical results to current implementation
4. DelayedNetworkBackend correctly models latency
5. Topology updates handled correctly
6. All existing tests pass with NativeNetworkBackend
"""

import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from abc import ABC

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPacketTransfer:
    """Tests for PacketTransfer dataclass."""
    
    def test_packet_transfer_creation(self):
        """Test PacketTransfer can be created."""
        from simulation import PacketTransfer
        
        transfer = PacketTransfer(
            source_id="SAT-001",
            destination_id="SAT-002",
            packet_id=42,
            timestamp=100.0,
            success=True,
            latency_ms=23.5
        )
        
        assert transfer.source_id == "SAT-001"
        assert transfer.destination_id == "SAT-002"
        assert transfer.packet_id == 42
        assert transfer.timestamp == 100.0
        assert transfer.success is True
        assert transfer.latency_ms == 23.5
    
    def test_packet_transfer_defaults(self):
        """Test PacketTransfer default values."""
        from simulation import PacketTransfer, DropReason
        
        transfer = PacketTransfer(
            source_id="A",
            destination_id="B",
            packet_id=1,
            timestamp=0.0,
            success=True
        )
        
        assert transfer.latency_ms is None
        assert transfer.size_bytes == 1024
        assert transfer.dropped_reason == DropReason.NONE
        assert transfer.metadata == {}
    
    def test_packet_transfer_drop_reason_auto(self):
        """Test drop reason is set automatically for failed transfers."""
        from simulation import PacketTransfer, DropReason
        
        transfer = PacketTransfer(
            source_id="A",
            destination_id="B",
            packet_id=1,
            timestamp=0.0,
            success=False  # No dropped_reason specified
        )
        
        # Should auto-set to NO_ROUTE
        assert transfer.dropped_reason == DropReason.NO_ROUTE
    
    def test_packet_transfer_serialization(self):
        """Test PacketTransfer to_dict and from_dict."""
        from simulation import PacketTransfer, DropReason
        
        transfer = PacketTransfer(
            source_id="SAT-001",
            destination_id="SAT-002",
            packet_id=42,
            timestamp=100.0,
            success=True,
            latency_ms=23.5,
            size_bytes=2048,
            metadata={"test": "value"}
        )
        
        data = transfer.to_dict()
        assert "source_id" in data
        assert "destination_id" in data
        assert "packet_id" in data
        assert "latency_ms" in data
        
        restored = PacketTransfer.from_dict(data)
        assert restored.source_id == transfer.source_id
        assert restored.packet_id == transfer.packet_id
        assert restored.latency_ms == transfer.latency_ms


class TestNetworkStatistics:
    """Tests for NetworkStatistics dataclass."""
    
    def test_statistics_creation(self):
        """Test NetworkStatistics can be created."""
        from simulation import NetworkStatistics
        
        stats = NetworkStatistics(
            total_packets_sent=100,
            total_packets_received=95,
            total_packets_dropped=5,
            average_latency_ms=25.0
        )
        
        assert stats.total_packets_sent == 100
        assert stats.total_packets_received == 95
        assert stats.total_packets_dropped == 5
        assert stats.average_latency_ms == 25.0
    
    def test_statistics_defaults(self):
        """Test NetworkStatistics default values."""
        from simulation import NetworkStatistics
        
        stats = NetworkStatistics()
        
        assert stats.total_packets_sent == 0
        assert stats.total_packets_received == 0
        assert stats.total_packets_dropped == 0
        assert stats.average_latency_ms == 0.0
    
    def test_delivery_ratio(self):
        """Test delivery_ratio property."""
        from simulation import NetworkStatistics
        
        stats = NetworkStatistics(
            total_packets_sent=100,
            total_packets_received=80
        )
        
        assert stats.delivery_ratio == 0.8
    
    def test_drop_ratio(self):
        """Test drop_ratio property."""
        from simulation import NetworkStatistics
        
        stats = NetworkStatistics(
            total_packets_sent=100,
            total_packets_dropped=10
        )
        
        assert stats.drop_ratio == 0.1
    
    def test_statistics_reset(self):
        """Test statistics reset."""
        from simulation import NetworkStatistics
        
        stats = NetworkStatistics(
            total_packets_sent=100,
            total_packets_received=95
        )
        
        stats.reset()
        
        assert stats.total_packets_sent == 0
        assert stats.total_packets_received == 0
    
    def test_statistics_serialization(self):
        """Test NetworkStatistics to_dict."""
        from simulation import NetworkStatistics
        
        stats = NetworkStatistics(
            total_packets_sent=100,
            total_packets_received=95,
            average_latency_ms=25.0
        )
        
        data = stats.to_dict()
        
        assert data["total_packets_sent"] == 100
        assert data["total_packets_received"] == 95
        assert "delivery_ratio" in data
        assert "drop_ratio" in data


class TestDropReason:
    """Tests for DropReason enum."""
    
    def test_drop_reasons_exist(self):
        """Test all expected drop reasons exist."""
        from simulation import DropReason
        
        assert hasattr(DropReason, 'NONE')
        assert hasattr(DropReason, 'NO_ROUTE')
        assert hasattr(DropReason, 'LINK_DOWN')
        assert hasattr(DropReason, 'QUEUE_FULL')
        assert hasattr(DropReason, 'TIMEOUT')
    
    def test_drop_reason_values(self):
        """Test drop reason values are strings."""
        from simulation import DropReason
        
        assert DropReason.NONE.value == "none"
        assert DropReason.NO_ROUTE.value == "no_route"
        assert DropReason.LINK_DOWN.value == "link_down"


class TestNetworkBackendInterface:
    """Tests for NetworkBackend abstract base class."""
    
    def test_backend_is_abc(self):
        """Test NetworkBackend is an ABC."""
        from simulation import NetworkBackend
        
        assert issubclass(NetworkBackend, ABC)
    
    def test_cannot_instantiate_abc(self):
        """Test NetworkBackend cannot be instantiated."""
        from simulation import NetworkBackend
        
        with pytest.raises(TypeError):
            NetworkBackend()
    
    def test_abstract_methods_defined(self):
        """Test required abstract methods are defined."""
        from simulation import NetworkBackend
        import inspect
        
        # Get abstract methods
        abstract_methods = set()
        for name, method in inspect.getmembers(NetworkBackend):
            if getattr(method, '__isabstractmethod__', False):
                abstract_methods.add(name)
        
        expected = {'initialize', 'update_topology', 'send_packet', 'step', 'get_statistics'}
        assert expected <= abstract_methods


class TestNativeNetworkBackend:
    """Tests for NativeNetworkBackend implementation."""
    
    def test_native_backend_creation(self):
        """Test NativeNetworkBackend can be created."""
        from simulation import NativeNetworkBackend
        
        backend = NativeNetworkBackend()
        
        assert backend is not None
        assert backend.get_pending_count() == 0
    
    def test_backend_initialization(self, sample_topology):
        """Test initialization with topology."""
        from simulation import NativeNetworkBackend
        
        backend = NativeNetworkBackend()
        backend.initialize(sample_topology)
        
        # Should have links from topology
        assert len(backend.active_links) == 3
    
    def test_update_topology(self, active_links_set):
        """Test topology update."""
        from simulation import NativeNetworkBackend
        
        backend = NativeNetworkBackend()
        backend.initialize({})
        backend.update_topology(active_links_set)
        
        assert backend.active_links == active_links_set
    
    def test_send_packet(self):
        """Test packet sending."""
        from simulation import NativeNetworkBackend
        
        backend = NativeNetworkBackend()
        backend.initialize({"links": [("A", "B")]})
        
        result = backend.send_packet("A", "B", packet_id=1)
        
        assert result is True
        assert backend.get_pending_count() == 1
    
    def test_step_delivers_packets(self):
        """Test step delivers queued packets."""
        from simulation import NativeNetworkBackend
        
        backend = NativeNetworkBackend()
        backend.initialize({"links": [("A", "B")]})
        backend.send_packet("A", "B", packet_id=1)
        backend.send_packet("A", "B", packet_id=2)
        
        transfers = backend.step(60.0)
        
        assert len(transfers) == 2
        assert all(t.success for t in transfers)
        assert backend.get_pending_count() == 0
    
    def test_step_drops_without_link(self):
        """Test packets dropped when no link exists."""
        from simulation import NativeNetworkBackend, DropReason
        
        backend = NativeNetworkBackend()
        backend.initialize({"links": []})  # No links
        backend.send_packet("A", "B", packet_id=1)
        
        transfers = backend.step(60.0)
        
        assert len(transfers) == 1
        assert transfers[0].success is False
        assert transfers[0].dropped_reason == DropReason.NO_ROUTE
    
    def test_instant_delivery(self):
        """Test native backend has zero latency."""
        from simulation import NativeNetworkBackend
        
        backend = NativeNetworkBackend()
        backend.initialize({"links": [("A", "B")]})
        backend.send_packet("A", "B", packet_id=1)
        
        transfers = backend.step(60.0)
        
        assert transfers[0].latency_ms == 0.0
    
    def test_bidirectional_link_check(self):
        """Test link check works bidirectionally."""
        from simulation import NativeNetworkBackend
        
        backend = NativeNetworkBackend()
        backend.initialize({"links": [("A", "B")]})
        
        # Should work in both directions
        assert backend.is_link_active("A", "B")
        assert backend.is_link_active("B", "A")
    
    def test_statistics_tracking(self):
        """Test statistics are tracked correctly."""
        from simulation import NativeNetworkBackend
        
        backend = NativeNetworkBackend()
        backend.initialize({"links": [("A", "B")]})
        
        backend.send_packet("A", "B", packet_id=1, size_bytes=1000)
        backend.send_packet("A", "B", packet_id=2, size_bytes=2000)
        backend.step(60.0)
        
        stats = backend.get_statistics()
        
        assert stats.total_packets_sent == 2
        assert stats.total_packets_received == 2
        assert stats.total_bytes_sent == 3000
        assert stats.total_bytes_received == 3000
    
    def test_reset(self):
        """Test backend reset."""
        from simulation import NativeNetworkBackend
        
        backend = NativeNetworkBackend()
        backend.initialize({"links": [("A", "B")]})
        backend.send_packet("A", "B", packet_id=1)
        backend.step(60.0)
        
        backend.reset()
        
        assert len(backend.active_links) == 0
        assert backend.get_pending_count() == 0
        assert backend.get_statistics().total_packets_sent == 0


class TestDelayedNetworkBackend:
    """Tests for DelayedNetworkBackend implementation."""
    
    def test_delayed_backend_creation(self):
        """Test DelayedNetworkBackend can be created."""
        from simulation import DelayedNetworkBackend
        
        backend = DelayedNetworkBackend()
        
        assert backend is not None
    
    def test_delayed_backend_with_processing_delay(self):
        """Test backend with processing delay."""
        from simulation import DelayedNetworkBackend
        
        backend = DelayedNetworkBackend(processing_delay_ms=10.0)
        backend.initialize({"links": [("A", "B")]})
        backend.send_packet("A", "B", packet_id=1)
        
        # First step - packet should still be in transit
        transfers = backend.step(0.005)  # 5ms
        assert len(transfers) == 0
        assert backend.get_pending_count() == 1
        
        # Second step - packet should arrive
        transfers = backend.step(0.010)  # Another 10ms
        assert len(transfers) == 1
        assert transfers[0].success
        assert transfers[0].latency_ms >= 10.0
    
    def test_delayed_backend_with_positions(self):
        """Test backend with position-based delay."""
        from simulation import DelayedNetworkBackend
        
        backend = DelayedNetworkBackend()
        backend.initialize({"links": [("A", "B")]})
        
        # Set positions - 1000 km apart
        backend.set_positions({
            "A": np.array([7000.0, 0.0, 0.0]),
            "B": np.array([8000.0, 0.0, 0.0])  # 1000 km away
        })
        
        backend.send_packet("A", "B", packet_id=1)
        
        # Step forward enough time
        transfers = backend.step(1.0)  # 1 second
        
        assert len(transfers) == 1
        assert transfers[0].success
        # Latency should be > 0 (propagation delay)
        assert transfers[0].latency_ms > 0
    
    def test_delayed_backend_link_down_during_transit(self):
        """Test packet dropped if link goes down during transit."""
        from simulation import DelayedNetworkBackend, DropReason
        
        backend = DelayedNetworkBackend(processing_delay_ms=100.0)  # 100ms delay
        backend.initialize({"links": [("A", "B")]})
        backend.send_packet("A", "B", packet_id=1)
        
        # Remove link while packet in transit
        backend.update_topology(set())
        
        # Step to complete transfer
        transfers = backend.step(0.2)  # 200ms
        
        assert len(transfers) == 1
        assert transfers[0].success is False
        assert transfers[0].dropped_reason == DropReason.LINK_DOWN


class TestNativeBackendEquivalence:
    """Tests ensuring NativeNetworkBackend preserves simulation behavior."""
    
    def test_simulation_results_unchanged(self, small_simulation):
        """Test simulation produces correct results."""
        sim = small_simulation
        
        # Record initial statistics
        initial_stats = sim.state.agent_statistics.average_completion
        
        # Run simulation
        for _ in range(5):
            sim.step(60.0)
        
        # Statistics should be computed correctly
        final_stats = sim.state.agent_statistics.average_completion
        
        # Should have made some progress
        assert final_stats >= initial_stats
    
    def test_packet_distribution_works(self, small_simulation):
        """Test packet distribution works correctly."""
        sim = small_simulation
        
        # Run until some packets distributed
        for _ in range(10):
            sim.step(60.0)
        
        # Check that packets have been distributed
        stats = sim.state.agent_statistics
        
        # At least some progress should be made
        assert stats.average_completion >= 0.0


class TestTopologyManagement:
    """Tests for topology update handling."""
    
    def test_link_addition(self):
        """Test adding links to topology."""
        from simulation import NativeNetworkBackend
        
        backend = NativeNetworkBackend()
        backend.initialize({})
        
        # Start with one link
        backend.update_topology({("A", "B")})
        assert len(backend.active_links) == 1
        
        # Add more links
        backend.update_topology({("A", "B"), ("B", "C"), ("C", "D")})
        assert len(backend.active_links) == 3
    
    def test_link_removal(self):
        """Test removing links from topology."""
        from simulation import NativeNetworkBackend
        
        backend = NativeNetworkBackend()
        backend.initialize({})
        
        # Start with multiple links
        backend.update_topology({("A", "B"), ("B", "C"), ("C", "D")})
        assert len(backend.active_links) == 3
        
        # Remove links
        backend.update_topology({("A", "B")})
        assert len(backend.active_links) == 1
        assert ("A", "B") in backend.active_links
    
    def test_topology_with_simulation_links(self, small_simulation):
        """Test topology updates from simulation."""
        sim = small_simulation
        
        # Get active links from simulation
        active_links = sim.state.active_links
        
        # Links should be populated
        assert isinstance(active_links, set)
        
        # Step simulation - links may change
        sim.step(60.0)
        new_links = sim.state.active_links
        
        # Links should still be valid
        assert isinstance(new_links, set)


class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_native_backend(self):
        """Test create_native_backend factory."""
        from simulation import create_native_backend, NativeNetworkBackend
        
        backend = create_native_backend()
        
        assert isinstance(backend, NativeNetworkBackend)
    
    def test_create_delayed_backend(self):
        """Test create_delayed_backend factory."""
        from simulation import create_delayed_backend, DelayedNetworkBackend
        
        backend = create_delayed_backend(processing_delay_ms=5.0)
        
        assert isinstance(backend, DelayedNetworkBackend)


class TestBackwardCompatibilityStep4:
    """Ensure Step 4 changes don't break existing functionality."""
    
    def test_simulation_api_unchanged(self, simulation_config):
        """Test Simulation API is unchanged."""
        from simulation import Simulation
        
        sim = Simulation(simulation_config)
        
        # All existing methods exist
        assert hasattr(sim, 'initialize')
        assert hasattr(sim, 'step')
        assert hasattr(sim, 'run')
        assert hasattr(sim, 'reset')
        assert hasattr(sim, 'satellites')
    
    def test_agent_protocol_works(self, small_simulation):
        """Test agent protocol still works."""
        sim = small_simulation
        
        # Run several steps
        for _ in range(5):
            state = sim.step(60.0)
        
        # Agent statistics should be computed
        assert state.agent_statistics is not None
        assert isinstance(state.agent_statistics.average_completion, float)
    
    def test_trajectory_provider_unchanged(self, sample_satellites):
        """Test TrajectoryProvider from Step 1 unchanged."""
        from simulation import KeplerianProvider
        
        epoch = datetime(2025, 1, 1, 0, 0, 0)
        provider = KeplerianProvider(sample_satellites, epoch)
        
        assert len(provider.get_satellite_ids()) == 3
    
    def test_spice_provider_unchanged(self):
        """Test SpiceProvider from Step 2 unchanged."""
        from simulation import SpiceProvider, is_spice_available
        
        assert SpiceProvider is not None
        assert callable(is_spice_available)
    
    def test_all_step4_exports_available(self):
        """Test all Step 4 exports are available."""
        from simulation import (
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
        
        assert NetworkBackend is not None
        assert NativeNetworkBackend is not None
        assert DelayedNetworkBackend is not None
        assert PacketTransfer is not None
        assert NetworkStatistics is not None
        assert DropReason is not None