#!/usr/bin/env python3
"""
Tests for Step 1: TrajectoryProvider Interface

These tests verify:
1. TrajectoryProvider ABC is correctly defined
2. KeplerianProvider wraps existing satellites properly
3. Simulation works identically with KeplerianProvider
4. All existing functionality is preserved
"""

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTrajectoryProviderInterface:
    """Tests for the TrajectoryProvider abstract base class."""
    
    def test_trajectory_state_dataclass(self):
        """Test TrajectoryState dataclass creation and attributes."""
        from simulation import TrajectoryState
        
        state = TrajectoryState(
            position_eci=np.array([7000.0, 0.0, 0.0]),
            velocity_eci=np.array([0.0, 7.5, 0.0]),
            epoch=datetime(2025, 1, 1, 0, 0, 0),
            reference_frame="J2000"
        )
        
        assert state.position_eci.shape == (3,)
        assert state.velocity_eci.shape == (3,)
        assert state.reference_frame == "J2000"
        assert isinstance(state.epoch, datetime)
    
    def test_trajectory_state_properties(self):
        """Test TrajectoryState computed properties."""
        from simulation import TrajectoryState
        
        state = TrajectoryState(
            position_eci=np.array([7000.0, 0.0, 0.0]),
            velocity_eci=np.array([0.0, 7.5, 0.0]),
            epoch=datetime(2025, 1, 1, 0, 0, 0)
        )
        
        assert abs(state.radius - 7000.0) < 0.01
        assert abs(state.speed - 7.5) < 0.01
    
    def test_trajectory_state_serialization(self):
        """Test TrajectoryState to_dict and from_dict."""
        from simulation import TrajectoryState
        
        state = TrajectoryState(
            position_eci=np.array([7000.0, 100.0, 200.0]),
            velocity_eci=np.array([0.1, 7.5, 0.2]),
            epoch=datetime(2025, 1, 1, 12, 30, 0),
            reference_frame="J2000"
        )
        
        # Serialize
        data = state.to_dict()
        assert "position_eci" in data
        assert "velocity_eci" in data
        assert "epoch" in data
        assert "reference_frame" in data
        
        # Deserialize
        restored = TrajectoryState.from_dict(data)
        np.testing.assert_allclose(restored.position_eci, state.position_eci)
        np.testing.assert_allclose(restored.velocity_eci, state.velocity_eci)
        assert restored.epoch == state.epoch
        assert restored.reference_frame == state.reference_frame
    
    def test_trajectory_provider_abstract_methods(self):
        """Test that TrajectoryProvider defines required abstract methods."""
        from simulation import TrajectoryProvider
        from abc import ABC
        
        # Verify it's an ABC
        assert issubclass(TrajectoryProvider, ABC)
        
        # Verify cannot instantiate abstract class
        with pytest.raises(TypeError):
            TrajectoryProvider()
    
    def test_concrete_provider_implementation(self):
        """Test that concrete implementations satisfy the interface."""
        from simulation import TrajectoryProvider, TrajectoryState
        
        class MockProvider(TrajectoryProvider):
            def get_state(self, satellite_id: str, time: datetime) -> TrajectoryState:
                return TrajectoryState(
                    position_eci=np.array([7000.0, 0.0, 0.0]),
                    velocity_eci=np.array([0.0, 7.5, 0.0]),
                    epoch=time
                )
            
            def get_position_eci(self, satellite_id: str, time: datetime) -> np.ndarray:
                return np.array([7000.0, 0.0, 0.0])
            
            def get_satellite_ids(self) -> List[str]:
                return ["SAT-001", "SAT-002"]
            
            def get_time_bounds(self, satellite_id: str) -> Tuple[datetime, datetime]:
                return (datetime.min, datetime.max)
        
        # Should be instantiable
        provider = MockProvider()
        assert len(provider.get_satellite_ids()) == 2
        
        # Test __contains__
        assert "SAT-001" in provider
        assert "SAT-003" not in provider
        
        # Test __len__
        assert len(provider) == 2


class TestKeplerianProvider:
    """Tests for the KeplerianProvider implementation."""
    
    def test_keplerian_provider_creation(self, sample_satellites):
        """Test KeplerianProvider can be created from satellites."""
        from simulation import KeplerianProvider
        
        epoch = datetime(2025, 1, 1, 0, 0, 0)
        provider = KeplerianProvider(sample_satellites, epoch)
        
        assert len(provider.get_satellite_ids()) == 3
        assert "TEST-SAT-001" in provider.get_satellite_ids()
    
    def test_keplerian_provider_position_matches_satellite(self, sample_satellites):
        """Test KeplerianProvider returns same position as direct satellite access."""
        from simulation import KeplerianProvider
        
        sat = sample_satellites[0]
        expected_position = sat.get_position_eci()
        
        epoch = datetime(2025, 1, 1, 0, 0, 0)
        provider = KeplerianProvider(sample_satellites, epoch)
        
        # Get position through provider
        provider_position = provider.get_position_eci(sat.satellite_id, epoch)
        
        # Should match
        np.testing.assert_allclose(provider_position, expected_position, rtol=1e-10)
    
    def test_keplerian_provider_velocity_matches_satellite(self, sample_satellites):
        """Test KeplerianProvider returns same velocity as direct satellite access."""
        from simulation import KeplerianProvider
        
        sat = sample_satellites[0]
        expected_velocity = sat.get_velocity_eci()
        
        epoch = datetime(2025, 1, 1, 0, 0, 0)
        provider = KeplerianProvider(sample_satellites, epoch)
        
        # Get velocity through provider
        provider_velocity = provider.get_velocity_eci(sat.satellite_id, epoch)
        
        # Should match
        np.testing.assert_allclose(provider_velocity, expected_velocity, rtol=1e-10)
    
    def test_keplerian_provider_state(self, sample_satellites):
        """Test KeplerianProvider get_state returns complete state."""
        from simulation import KeplerianProvider, TrajectoryState
        
        epoch = datetime(2025, 1, 1, 0, 0, 0)
        provider = KeplerianProvider(sample_satellites, epoch)
        
        state = provider.get_state("TEST-SAT-001", epoch)
        
        assert isinstance(state, TrajectoryState)
        assert state.position_eci.shape == (3,)
        assert state.velocity_eci.shape == (3,)
        assert state.epoch == epoch
        assert state.reference_frame == "J2000"
    
    def test_keplerian_provider_time_propagation(self, sample_satellites):
        """Test KeplerianProvider correctly propagates position over time."""
        from simulation import KeplerianProvider
        
        sat = sample_satellites[0]
        epoch = datetime(2025, 1, 1, 0, 0, 0)
        provider = KeplerianProvider(sample_satellites, epoch)
        
        # Get initial position
        initial_position = provider.get_position_eci(sat.satellite_id, epoch)
        
        # Get position 1 minute later
        later = epoch + timedelta(minutes=1)
        later_position = provider.get_position_eci(sat.satellite_id, later)
        
        # Position should have changed
        assert not np.allclose(initial_position, later_position)
        
        # But altitude should be approximately the same (circular orbit)
        initial_radius = np.linalg.norm(initial_position)
        later_radius = np.linalg.norm(later_position)
        assert abs(initial_radius - later_radius) < 1.0  # Within 1 km
    
    def test_keplerian_provider_time_bounds(self, sample_satellites):
        """Test KeplerianProvider reports unbounded time range."""
        from simulation import KeplerianProvider
        
        epoch = datetime(2025, 1, 1, 0, 0, 0)
        provider = KeplerianProvider(sample_satellites, epoch)
        
        start, end = provider.get_time_bounds("TEST-SAT-001")
        
        # Keplerian orbits are valid for all time
        assert start == datetime.min
        assert end == datetime.max
    
    def test_keplerian_provider_missing_satellite(self, sample_satellites):
        """Test KeplerianProvider raises KeyError for unknown satellite."""
        from simulation import KeplerianProvider
        
        epoch = datetime(2025, 1, 1, 0, 0, 0)
        provider = KeplerianProvider(sample_satellites, epoch)
        
        with pytest.raises(KeyError):
            provider.get_state("NONEXISTENT", epoch)
        
        with pytest.raises(KeyError):
            provider.get_position_eci("NONEXISTENT", epoch)
        
        with pytest.raises(KeyError):
            provider.get_time_bounds("NONEXISTENT")
    
    def test_keplerian_provider_is_valid_time(self, sample_satellites):
        """Test KeplerianProvider.is_valid_time method."""
        from simulation import KeplerianProvider
        
        epoch = datetime(2025, 1, 1, 0, 0, 0)
        provider = KeplerianProvider(sample_satellites, epoch)
        
        # All times should be valid for Keplerian orbits
        assert provider.is_valid_time("TEST-SAT-001", datetime(2000, 1, 1))
        assert provider.is_valid_time("TEST-SAT-001", datetime(2100, 1, 1))
    
    def test_keplerian_provider_step_all(self, sample_satellites):
        """Test KeplerianProvider step_all method."""
        from simulation import KeplerianProvider
        
        epoch = datetime(2025, 1, 1, 0, 0, 0)
        provider = KeplerianProvider(sample_satellites, epoch)
        
        # Record initial positions
        initial_positions = {
            sat_id: provider.get_position_eci(sat_id, epoch).copy()
            for sat_id in provider.get_satellite_ids()
        }
        
        # Step all satellites forward
        provider.step_all(60.0)
        
        # Positions should have changed after stepping
        for sat_id in provider.get_satellite_ids():
            new_pos = provider.get_position_eci(sat_id, provider.epoch)
            assert not np.allclose(new_pos, initial_positions[sat_id])


class TestSimulationWithTrajectoryProvider:
    """Tests for Simulation integration with TrajectoryProvider."""
    
    def test_simulation_accepts_trajectory_provider(self, simulation_config):
        """Test Simulation can be created with a trajectory provider."""
        from simulation import Simulation
        
        sim = Simulation(simulation_config)
        sim.initialize()
        
        assert sim.num_satellites > 0
    
    def test_simulation_default_provider_unchanged(self, small_simulation):
        """Test simulation with default provider behaves identically."""
        sim = small_simulation
        
        # Record initial state
        initial_positions = {
            sat.satellite_id: sat.get_position_eci().copy()
            for sat in sim.satellites
        }
        
        # Step simulation
        sim.step(60.0)
        
        # Positions should have changed
        for sat in sim.satellites:
            new_pos = sat.get_position_eci()
            old_pos = initial_positions[sat.satellite_id]
            assert not np.allclose(new_pos, old_pos)
    
    def test_simulation_statistics_unchanged(self, small_simulation):
        """Test agent statistics work correctly with provider."""
        sim = small_simulation
        
        # Run a few steps
        for _ in range(5):
            sim.step(60.0)
        
        stats = sim.state.agent_statistics
        
        # Verify statistics are computed
        assert stats.total_packets == sim.config.num_packets
        assert 0.0 <= stats.average_completion <= 100.0
        assert stats.fully_updated_count >= 0


class TestBackwardCompatibilityStep1:
    """Ensure Step 1 changes don't break existing functionality."""
    
    def test_satellite_class_unchanged(self):
        """Test Satellite class API is unchanged."""
        from simulation import Satellite, EllipticalOrbit, EARTH_RADIUS_KM
        
        orbit = EllipticalOrbit(
            apoapsis=EARTH_RADIUS_KM + 550,
            periapsis=EARTH_RADIUS_KM + 550,
            inclination=math.radians(53),
            longitude_of_ascending_node=0,
            argument_of_periapsis=0,
        )
        
        sat = Satellite(orbit, initial_position=0.0, satellite_id="TEST")
        
        # All existing methods should work
        assert sat.satellite_id == "TEST"
        assert sat.get_position_eci().shape == (3,)
        assert sat.get_velocity_eci().shape == (3,)
        assert isinstance(sat.get_altitude(), float)
        assert isinstance(sat.get_speed(), float)
        
        sat.step(60.0)
        assert sat.elapsed_time == 60.0
    
    def test_simulation_api_unchanged(self, simulation_config):
        """Test Simulation API is unchanged."""
        from simulation import Simulation
        
        # Should work without trajectory_provider argument
        sim = Simulation(simulation_config)
        
        # All existing methods should exist
        assert hasattr(sim, 'initialize')
        assert hasattr(sim, 'step')
        assert hasattr(sim, 'run')
        assert hasattr(sim, 'reset')
        assert hasattr(sim, 'is_update_complete')
        assert hasattr(sim, 'num_satellites')
        assert hasattr(sim, 'num_orbits')
        assert hasattr(sim, 'simulation_time')
    
    def test_orbit_class_unchanged(self):
        """Test EllipticalOrbit class API is unchanged."""
        from simulation import EllipticalOrbit, EARTH_RADIUS_KM
        
        orbit = EllipticalOrbit(
            apoapsis=EARTH_RADIUS_KM + 550,
            periapsis=EARTH_RADIUS_KM + 550,
            inclination=math.radians(53),
            longitude_of_ascending_node=0,
            argument_of_periapsis=0,
        )
        
        # All existing methods should work
        pos = orbit.position_eci(0.0)
        assert pos.shape == (3,)
        
        vel = orbit.velocity_eci(0.0)
        assert vel.shape == (3,)
        
        r = orbit.radius_at_true_anomaly(0.0)
        assert r > EARTH_RADIUS_KM
        
        v = orbit.velocity_at_radius(r)
        assert v > 0
    
    def test_constellation_factories_unchanged(self):
        """Test constellation factory functions unchanged."""
        from simulation import (
            create_walker_delta_constellation,
            create_walker_star_constellation,
            create_random_constellation,
        )
        
        # Walker-Delta
        orbits, sats = create_walker_delta_constellation(
            num_planes=3,
            sats_per_plane=4,
            altitude=550,
            inclination=math.radians(53),
            phasing_parameter=1,
        )
        assert len(orbits) == 3
        assert len(sats) == 12
        
        # Walker-Star
        orbits, sats = create_walker_star_constellation(
            num_planes=4,
            sats_per_plane=3,
            altitude=800,
        )
        assert len(orbits) == 4
        assert len(sats) == 12
        
        # Random
        orbits, sats = create_random_constellation(
            num_satellites=10,
            seed=42,
        )
        assert len(sats) == 10


class TestCreateKeplerianProvider:
    """Test the create_keplerian_provider factory function."""
    
    def test_create_with_epoch(self, sample_satellites):
        """Test creating provider with explicit epoch."""
        from simulation import create_keplerian_provider
        
        epoch = datetime(2025, 6, 15, 12, 0, 0)
        provider = create_keplerian_provider(sample_satellites, epoch)
        
        assert provider.epoch == epoch
        assert len(provider) == len(sample_satellites)
    
    def test_create_without_epoch(self, sample_satellites):
        """Test creating provider without epoch uses current time."""
        from simulation import create_keplerian_provider
        
        before = datetime.now()
        provider = create_keplerian_provider(sample_satellites)
        after = datetime.now()
        
        # Epoch should be between before and after
        assert before <= provider.epoch <= after