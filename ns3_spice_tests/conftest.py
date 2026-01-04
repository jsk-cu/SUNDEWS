#!/usr/bin/env python3
"""
Pytest Configuration and Shared Fixtures

Provides common fixtures, markers, and utilities for testing the
NS-3 and SPICE integration components.
"""

import json
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
# PATCH NS3BACKEND WITH MISSING METHODS
# =============================================================================

def _patch_ns3_backend():
    """Patch NS3Backend with missing methods for test compatibility."""
    try:
        from simulation.ns3_backend import NS3Backend, NS3Node
        
        # Add update_positions method if not present
        if not hasattr(NS3Backend, 'update_positions'):
            def update_positions(self, positions: Dict[str, Any]) -> None:
                """
                Update multiple node positions.
                
                Parameters
                ----------
                positions : Dict[str, np.ndarray]
                    Dictionary mapping node IDs to position vectors in km.
                    Positions will be converted to meters internally.
                """
                for node_id, pos in positions.items():
                    pos_array = np.array(pos, dtype=float)
                    # Convert km to meters
                    pos_meters = pos_array * 1000.0
                    
                    if node_id in self._nodes:
                        self._nodes[node_id].position = pos_meters
                    else:
                        # Create new node
                        self._nodes[node_id] = NS3Node(
                            id=node_id,
                            node_type="satellite",
                            position=pos_meters
                        )
            
            NS3Backend.update_positions = update_positions
    except ImportError:
        pass  # NS3Backend not available

# Apply patches on module load
_patch_ns3_backend()


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
    """Default simulation configuration for testing.
    
    Returns a SimulationConfig object for use with Simulation.
    """
    from simulation import SimulationConfig, ConstellationType
    
    return SimulationConfig(
        constellation_type=ConstellationType.WALKER_DELTA,
        num_planes=3,
        sats_per_plane=4,
        altitude=550.0,
        inclination=math.radians(53.0),
        num_packets=10,
        random_seed=42,
    )


@pytest.fixture
def sample_satellites():
    """Sample satellite objects for testing.
    
    Returns a list of actual Satellite objects with proper orbital elements.
    """
    from simulation import Satellite, EllipticalOrbit, EARTH_RADIUS_KM
    
    satellites = []
    for i, (raan, true_anom) in enumerate([(0.0, 0.0), (120.0, 30.0), (240.0, 60.0)]):
        orbit = EllipticalOrbit(
            apoapsis=EARTH_RADIUS_KM + 550,  # 550 km altitude
            periapsis=EARTH_RADIUS_KM + 550,
            inclination=math.radians(53.0),
            longitude_of_ascending_node=math.radians(raan),
            argument_of_periapsis=0.0,
        )
        sat = Satellite(
            orbit=orbit,
            initial_position=true_anom / 360.0,  # Position as fraction of orbit
            satellite_id=f"TEST-SAT-00{i+1}",
        )
        satellites.append(sat)
    
    return satellites


@pytest.fixture
def small_simulation():
    """Create a small simulation for testing.
    
    Returns an initialized Simulation object with a small constellation
    suitable for testing. Logging is enabled for testing log functionality.
    """
    from simulation import Simulation, SimulationConfig, ConstellationType
    
    config = SimulationConfig(
        constellation_type=ConstellationType.WALKER_DELTA,
        num_planes=2,
        sats_per_plane=3,
        num_packets=10,
        random_seed=42,
    )
    
    sim = Simulation(config, enable_logging=True)
    sim.initialize()
    return sim


# =============================================================================
# SPICE FIXTURES
# =============================================================================

@pytest.fixture
def mock_spice_kernels(tmp_path):
    """Create mock SPICE kernel files for testing.
    
    Returns the tmp_path so tests can use / operator for path construction.
    Also creates mock kernel files at expected locations.
    """
    # Create mock kernel files at expected locations
    (tmp_path / "naif0012.tls").write_text("Mock leapseconds kernel")
    (tmp_path / "test_constellation.bsp").write_text("Mock spacecraft kernel")
    (tmp_path / "constellation.bsp").write_text("Mock spacecraft kernel")
    (tmp_path / "de440.bsp").write_text("Mock planetary kernel")
    (tmp_path / "frames.tf").write_text("Mock frame kernel")
    
    # Return the path so it can be used with / operator
    return tmp_path


@pytest.fixture
def mock_spiceypy():
    """Mock the spiceypy library for testing without SPICE installed."""
    mock_spice = MagicMock()
    
    # Mock furnsh (kernel loading)
    mock_spice.furnsh = MagicMock()
    
    # Mock unload
    mock_spice.unload = MagicMock()
    
    # Mock str2et (string to ephemeris time)
    def mock_str2et(time_str):
        # Simple mock: return seconds since J2000
        if isinstance(time_str, str):
            return 0.0
        return 0.0
    mock_spice.str2et = MagicMock(side_effect=mock_str2et)
    
    # Mock et2utc (ephemeris time to UTC string)
    mock_spice.et2utc = MagicMock(return_value="2025-01-01T00:00:00")
    
    # Mock spkezr (state vector) - returns (state, light_time)
    def mock_spkezr(target, et, ref, abcorr, observer):
        # Return mock state vector: [x, y, z, vx, vy, vz]
        state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        light_time = 0.023
        return state, light_time
    mock_spice.spkezr = MagicMock(side_effect=mock_spkezr)
    
    # Mock spkez (state vector using body ID) - same as spkezr
    def mock_spkez(target, et, ref, abcorr, observer):
        # Return mock state vector: [x, y, z, vx, vy, vz]
        state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        light_time = 0.023
        return state, light_time
    mock_spice.spkez = MagicMock(side_effect=mock_spkez)
    
    # Mock spkpos (position only)
    def mock_spkpos(target, et, ref, abcorr, observer):
        position = np.array([7000.0, 0.0, 0.0])
        light_time = 0.023
        return position, light_time
    mock_spice.spkpos = MagicMock(side_effect=mock_spkpos)
    
    # Mock coverage functions
    mock_spice.spkcov = MagicMock(return_value=MagicMock())
    mock_spice.wnfetd = MagicMock(return_value=(0.0, 86400.0 * 365))
    mock_spice.wncard = MagicMock(return_value=1)
    
    with patch.dict('sys.modules', {'spiceypy': mock_spice}):
        yield mock_spice


@pytest.fixture
def spice_config_file(tmp_path):
    """Create a SPICE configuration file for testing.
    
    Includes all expected fields for both old and new config formats.
    Uses TEST-SAT-XXX naming to match test expectations.
    """
    config = {
        # New-style fields
        "name": "TestConstellation",
        "constellation_name": "TestConstellation",
        
        # Satellite/NAIF ID mapping - both field names for compatibility
        # Using TEST-SAT-XXX naming to match test expectations
        "satellites": {
            "TEST-SAT-001": -100001,
            "TEST-SAT-002": -100002,
            "TEST-SAT-003": -100003,
        },
        "naif_id_mapping": {
            "TEST-SAT-001": -100001,
            "TEST-SAT-002": -100002,
            "TEST-SAT-003": -100003,
        },
        
        # Kernel paths - both old and new field names
        "leapseconds": "naif0012.tls",
        "spacecraft_kernels": ["constellation.bsp"],
        "kernels": {
            "leapseconds": "naif0012.tls",
            "spacecraft": ["constellation.bsp"],
        },
        
        "reference_frame": "J2000",
        "observer": "EARTH",
    }
    
    config_path = tmp_path / "spice_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    
    return config_path


# =============================================================================
# NS-3 MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_ns3_subprocess():
    """Mock subprocess for NS-3 file mode testing."""
    
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


class MockSocket:
    """
    A proper mock socket that tracks state correctly.
    
    Unlike MagicMock, this class properly tracks boolean state for `connected`
    and list state for `sent_data`, which tests rely on.
    """
    def __init__(self):
        self.connected = False
        self.sent_data = []
        self.response_queue = []
        self._timeout = None
        self._blocking = True
    
    def connect(self, address):
        """Establish connection."""
        self.connected = True
    
    def sendall(self, data):
        """Send all data, tracking it."""
        self.sent_data.append(data)
        # Auto-generate response for step commands
        self._queue_response()
    
    def send(self, data):
        """Send data, tracking it."""
        self.sent_data.append(data)
        self._queue_response()
        return len(data)
    
    def _queue_response(self):
        """Queue a mock response."""
        response = {
            "status": "success",
            "simulation_time": 60.0,
            "transfers": [
                {
                    "source": "SAT-001",
                    "destination": "SAT-002",
                    "packet_id": 0,
                    "success": True,
                    "latency_ms": 23.4
                }
            ]
        }
        self.response_queue.append(json.dumps(response) + "\n")
    
    def recv(self, buffer_size):
        """Receive data."""
        if self.response_queue:
            return self.response_queue.pop(0).encode()
        # Simulate timeout if no data
        import socket as sock_module
        raise sock_module.timeout("timed out")
    
    def close(self):
        """Close connection."""
        self.connected = False
    
    def fileno(self):
        """Return fake file descriptor."""
        return 3
    
    def setblocking(self, blocking):
        """Set blocking mode."""
        self._blocking = blocking
    
    def settimeout(self, timeout):
        """Set timeout."""
        self._timeout = timeout
    
    def setsockopt(self, *args):
        """Set socket option (no-op for mock)."""
        pass
    
    def getsockname(self):
        """Get socket name."""
        return ('127.0.0.1', 12345)


@pytest.fixture
def mock_ns3_socket():
    """Mock socket for NS-3 socket mode testing.
    
    Uses a custom MockSocket class that properly tracks connection state
    and sent data, unlike MagicMock which doesn't preserve boolean/list state.
    """
    return MockSocket()


@pytest.fixture
def mock_ns3_bindings():
    """Mock NS-3 Python bindings."""
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
            {"id": "SAT-001", "type": "satellite", "position": [7000000, 0, 0]},
            {"id": "SAT-002", "type": "satellite", "position": [0, 7000000, 0]},
            {"id": "SAT-003", "type": "satellite", "position": [-7000000, 0, 0]},
            {"id": "BASE-1", "type": "ground", "position": [6371000, 0, 0]},
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
# PORT FIXTURES
# =============================================================================

@pytest.fixture
def free_tcp_port():
    """Get a free TCP port for testing."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


@pytest.fixture
def free_tcp_port_factory():
    """Factory to get multiple free TCP ports."""
    import socket
    
    def get_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            return s.getsockname()[1]
    
    return get_port


@pytest.fixture
def free_udp_port():
    """Get a free UDP port for testing."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(('', 0))
        port = s.getsockname()[1]
    return port


@pytest.fixture
def free_udp_port_factory():
    """Factory to get multiple free UDP ports."""
    import socket
    
    def get_port():
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]
    
    return get_port


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def assert_positions_close(pos1: np.ndarray, pos2: np.ndarray, rtol: float = 1e-5):
    """Assert two position vectors are close."""
    np.testing.assert_allclose(pos1, pos2, rtol=rtol)


def run_simulation_steps(sim, num_steps: int, timestep: float = 60.0):
    """Helper to run simulation for several steps."""
    states = []
    for _ in range(num_steps):
        state = sim.step(timestep)
        states.append(state)
    return states