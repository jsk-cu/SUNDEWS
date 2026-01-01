#!/usr/bin/env python3
"""
Pytest Configuration and Shared Fixtures

Provides common fixtures, markers, and utilities for testing the
NS-3 and SPICE integration components.
"""

import math
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import pytest
import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# PYTEST MARKERS
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_spice: mark test as requiring SpiceyPy installation"
    )
    config.addinivalue_line(
        "markers", "requires_ns3: mark test as requiring NS-3 installation"
    )
    config.addinivalue_line(
        "markers", "requires_ns3_bindings: mark test as requiring NS-3 Python bindings"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark as integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on available dependencies."""
    # Check for SpiceyPy
    try:
        import spiceypy
        spice_available = True
    except ImportError:
        spice_available = False
    
    # Check for NS-3 (basic check)
    ns3_available = Path("/usr/local/ns3/ns3").exists() or \
                    Path("/opt/ns3/ns3").exists()
    
    # Check for NS-3 Python bindings
    try:
        import ns.core
        ns3_bindings_available = True
    except ImportError:
        ns3_bindings_available = False
    
    skip_spice = pytest.mark.skip(reason="SpiceyPy not installed")
    skip_ns3 = pytest.mark.skip(reason="NS-3 not installed")
    skip_ns3_bindings = pytest.mark.skip(reason="NS-3 Python bindings not available")
    
    for item in items:
        if "requires_spice" in item.keywords and not spice_available:
            item.add_marker(skip_spice)
        if "requires_ns3" in item.keywords and not ns3_available:
            item.add_marker(skip_ns3)
        if "requires_ns3_bindings" in item.keywords and not ns3_bindings_available:
            item.add_marker(skip_ns3_bindings)


# =============================================================================
# SIMULATION FIXTURES
# =============================================================================

@pytest.fixture
def simulation_config():
    """Default simulation configuration for testing."""
    from simulation import SimulationConfig, ConstellationType
    
    return SimulationConfig(
        constellation_type=ConstellationType.WALKER_DELTA,
        num_planes=2,
        sats_per_plane=3,
        altitude=550.0,
        inclination=math.radians(53),
        num_packets=10,
        random_seed=42,
    )


@pytest.fixture
def small_simulation(simulation_config):
    """Small simulation instance for quick tests."""
    from simulation import Simulation
    
    sim = Simulation(simulation_config, enable_logging=True)
    sim.initialize(timestep=60.0)
    return sim


@pytest.fixture
def medium_simulation():
    """Medium-sized simulation for integration tests."""
    from simulation import Simulation, SimulationConfig, ConstellationType
    
    config = SimulationConfig(
        constellation_type=ConstellationType.WALKER_DELTA,
        num_planes=4,
        sats_per_plane=4,
        altitude=550.0,
        inclination=math.radians(53),
        num_packets=50,
        random_seed=42,
    )
    sim = Simulation(config, enable_logging=True)
    sim.initialize(timestep=60.0)
    return sim


@pytest.fixture
def sample_satellites():
    """Sample satellite objects for testing."""
    from simulation import (
        EllipticalOrbit, 
        Satellite, 
        EARTH_RADIUS_KM
    )
    
    satellites = []
    for i in range(3):
        orbit = EllipticalOrbit(
            apoapsis=EARTH_RADIUS_KM + 550,
            periapsis=EARTH_RADIUS_KM + 550,
            inclination=math.radians(53),
            longitude_of_ascending_node=math.radians(i * 120),
            argument_of_periapsis=0,
        )
        sat = Satellite(
            orbit=orbit,
            initial_position=i / 3,
            satellite_id=f"TEST-SAT-{i+1:03d}"
        )
        satellites.append(sat)
    
    return satellites


# =============================================================================
# TRAJECTORY PROVIDER FIXTURES
# =============================================================================

@pytest.fixture
def sample_trajectory_states():
    """Sample trajectory states for testing."""
    from dataclasses import dataclass
    
    @dataclass
    class TrajectoryState:
        position_eci: np.ndarray
        velocity_eci: np.ndarray
        epoch: datetime
        reference_frame: str = "J2000"
    
    states = []
    base_time = datetime(2025, 1, 1, 0, 0, 0)
    
    for i in range(10):
        # Simple circular orbit approximation
        angle = math.radians(i * 36)  # 36 degrees per step
        radius = 6921.0  # 550 km altitude
        
        states.append(TrajectoryState(
            position_eci=np.array([
                radius * math.cos(angle),
                radius * math.sin(angle),
                0.0
            ]),
            velocity_eci=np.array([
                -7.6 * math.sin(angle),
                7.6 * math.cos(angle),
                0.0
            ]),
            epoch=base_time + timedelta(minutes=i * 10)
        ))
    
    return states


# =============================================================================
# SPICE MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_spiceypy():
    """Mock SpiceyPy module for testing without SPICE installation."""
    mock_spice = MagicMock()
    
    # Mock furnsh (load kernel)
    mock_spice.furnsh = MagicMock()
    
    # Mock unload
    mock_spice.unload = MagicMock()
    
    # Mock kclear (clear all kernels)
    mock_spice.kclear = MagicMock()
    
    # Mock str2et (string to ephemeris time)
    mock_spice.str2et = MagicMock(return_value=0.0)
    
    # Mock spkez (get state vector)
    def mock_spkez(target, et, frame, abcorr, observer):
        # Return mock state vector [x, y, z, vx, vy, vz]
        return [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0], 0.0
    mock_spice.spkez = MagicMock(side_effect=mock_spkez)
    
    # Mock spkpos (get position)
    def mock_spkpos(target, et, frame, abcorr, observer):
        return [7000.0, 0.0, 0.0], 0.0
    mock_spice.spkpos = MagicMock(side_effect=mock_spkpos)
    
    # Mock spkcov (get coverage window)
    mock_window = MagicMock()
    mock_window.card = MagicMock(return_value=2)
    mock_spice.spkcov = MagicMock(return_value=mock_window)
    mock_spice.wnfetd = MagicMock(return_value=(0.0, 86400.0))
    
    return mock_spice


@pytest.fixture
def mock_spice_kernels(tmp_path):
    """Create mock SPICE kernel files."""
    kernels_dir = tmp_path / "kernels"
    kernels_dir.mkdir()
    
    # Create mock kernel files (just empty files for path testing)
    (kernels_dir / "naif0012.tls").touch()
    (kernels_dir / "de440.bsp").touch()
    (kernels_dir / "test_constellation.bsp").touch()
    
    return kernels_dir


@pytest.fixture
def spice_config_file(tmp_path, mock_spice_kernels):
    """Create a sample SPICE configuration file."""
    import json
    
    config = {
        "name": "TestConstellation",
        "epoch": "2025-01-01T00:00:00Z",
        "leapseconds": "naif0012.tls",
        "planetary": ["de440.bsp"],
        "spacecraft_kernels": ["test_constellation.bsp"],
        "satellites": {
            "TEST-SAT-001": -100001,
            "TEST-SAT-002": -100002,
            "TEST-SAT-003": -100003,
        },
        "reference_frame": "J2000",
        "observer": "EARTH"
    }
    
    config_file = tmp_path / "spice_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f)
    
    return config_file


# =============================================================================
# NS-3 MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_ns3_subprocess():
    """Mock subprocess for NS-3 file mode testing."""
    import json
    
    def create_mock_result(transfers=None):
        if transfers is None:
            transfers = [
                {
                    "source": "SAT-001",
                    "destination": "SAT-002",
                    "packet_id": 0,
                    "timestamp": 0.023,
                    "success": True,
                    "latency_ms": 23.4
                }
            ]
        
        output = {
            "status": "success",
            "simulation_time": 60.0,
            "transfers": transfers,
            "statistics": {
                "total_packets_sent": len(transfers),
                "total_packets_received": sum(1 for t in transfers if t["success"]),
                "average_latency_ms": 23.4
            }
        }
        
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(output)
        mock_result.stderr = ""
        
        return mock_result
    
    return create_mock_result


@pytest.fixture
def mock_ns3_socket():
    """Mock socket for NS-3 socket mode testing."""
    import json
    
    class MockSocket:
        def __init__(self):
            self.sent_data = []
            self.response_queue = []
            self.connected = False
        
        def connect(self, address):
            self.connected = True
        
        def sendall(self, data):
            self.sent_data.append(data)
            response = {
                "status": "success",
                "transfers": [
                    {
                        "source": "SAT-001",
                        "destination": "SAT-002",
                        "packet_id": 0,
                        "timestamp": 0.023,
                        "success": True,
                        "latency_ms": 23.4
                    }
                ]
            }
            self.response_queue.append(json.dumps(response) + "\n")
        
        def recv(self, bufsize):
            if self.response_queue:
                return self.response_queue.pop(0).encode()
            return b""
        
        def close(self):
            self.connected = False
    
    return MockSocket()


@pytest.fixture
def mock_ns3_bindings():
    """Mock NS-3 Python bindings for testing."""
    mock_core = MagicMock()
    mock_network = MagicMock()
    mock_internet = MagicMock()
    mock_p2p = MagicMock()
    mock_mobility = MagicMock()
    mock_apps = MagicMock()
    
    mock_core.Simulator = MagicMock()
    mock_core.Simulator.Stop = MagicMock()
    mock_core.Simulator.Run = MagicMock()
    mock_core.Simulator.Destroy = MagicMock()
    mock_core.Seconds = lambda x: x
    mock_core.Vector = lambda x, y, z: (x, y, z)
    
    mock_network.NodeContainer = MagicMock
    
    mock_internet.InternetStackHelper = MagicMock
    mock_p2p.PointToPointHelper = MagicMock
    mock_mobility.MobilityHelper = MagicMock
    mock_mobility.ListPositionAllocator = MagicMock
    
    return {
        "core": mock_core,
        "network": mock_network,
        "internet": mock_internet,
        "point_to_point": mock_p2p,
        "mobility": mock_mobility,
        "applications": mock_apps
    }


# =============================================================================
# NETWORK BACKEND FIXTURES
# =============================================================================

@pytest.fixture
def sample_topology():
    """Sample network topology for testing."""
    return {
        "nodes": [
            {"id": "SAT-001", "type": "satellite", "position": [7000, 0, 0]},
            {"id": "SAT-002", "type": "satellite", "position": [0, 7000, 0]},
            {"id": "SAT-003", "type": "satellite", "position": [-7000, 0, 0]},
            {"id": "BASE-1", "type": "ground", "position": [6371, 0, 0]},
        ],
        "links": [
            ("SAT-001", "SAT-002"),
            ("SAT-002", "SAT-003"),
            ("BASE-1", "SAT-001"),
        ],
        "config": {
            "data_rate": "10Mbps",
            "propagation_model": "constant_speed"
        }
    }


@pytest.fixture
def active_links_set():
    """Sample set of active links."""
    return {
        ("SAT-001", "SAT-002"),
        ("SAT-002", "SAT-003"),
    }


# =============================================================================
# FILE SYSTEM FIXTURES
# =============================================================================

@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for output files."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def temp_work_dir(tmp_path):
    """Temporary working directory."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    return work_dir


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def assert_positions_close(pos1: np.ndarray, pos2: np.ndarray, rtol: float = 1e-5):
    """Assert two position vectors are close."""
    np.testing.assert_allclose(pos1, pos2, rtol=rtol)


def assert_simulation_unchanged(sim_before, sim_after):
    """Assert simulation state hasn't changed unexpectedly."""
    assert sim_before.num_satellites == sim_after.num_satellites
    assert sim_before.num_orbits == sim_after.num_orbits
    
    for i, (sat_b, sat_a) in enumerate(zip(
        sim_before.satellites, sim_after.satellites
    )):
        assert sat_b.satellite_id == sat_a.satellite_id
        assert_positions_close(
            sat_b.get_position_eci(),
            sat_a.get_position_eci()
        )


def run_simulation_steps(sim, num_steps: int, timestep: float = 60.0):
    """Helper to run simulation for several steps."""
    states = []
    for _ in range(num_steps):
        state = sim.step(timestep)
        states.append(state)
    return states


# =============================================================================
# FIXTURE CLEANUP
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_spice():
    """Ensure SPICE kernels are unloaded after each test."""
    yield
    try:
        import spiceypy as spice
        spice.kclear()
    except ImportError:
        pass