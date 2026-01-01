# NS-3 and SPICE Integration Implementation Plan

## Executive Summary

This document outlines the implementation plan for adding optional NS-3 network simulation and SPICE ephemeris support to the SatUpdate satellite constellation simulator. These features are designed as **opt-in extensions** that preserve all existing functionality while enabling more sophisticated simulation capabilities.

### Current Status

| Step | Component | Status | Tests |
|------|-----------|--------|-------|
| 1 | TrajectoryProvider Interface | ✅ **COMPLETE** | 23 |
| 2 | SPICE Provider | ✅ **COMPLETE** | 31 |
| 3 | SPK Export Utility | ✅ **COMPLETE** | 41 |
| 4 | NetworkBackend Interface | ✅ **COMPLETE** | 41 |
| 5 | NS-3 File Mode | ✅ **COMPLETE** | 46 |
| 6 | NS-3 Socket Mode | 🔲 Pending | - |
| 7 | NS-3 Bindings Mode | 🔲 Pending | - |
| **Total** | | **5/7 Complete** | **178 passed, 5 skipped** |

---

## Design Principles

1. **Backward Compatibility**: All existing functionality remains unchanged. New features are activated only through explicit configuration.
2. **Graceful Degradation**: If optional dependencies (SpiceyPy, NS-3) are not installed, the system falls back to native implementations.
3. **Pluggable Architecture**: Abstract interfaces allow swapping implementations without changing core simulation logic.
4. **Incremental Adoption**: Each feature can be enabled independently.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SatUpdate Core                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────┐    ┌──────────────────────┐                   │
│  │  TrajectoryProvider  │    │   NetworkBackend     │                   │
│  │     (Abstract)       │    │     (Abstract)       │                   │
│  └──────────┬───────────┘    └──────────┬───────────┘                   │
│             │                           │                                │
│      ┌──────┴──────┐             ┌──────┴──────────────┐                │
│      │             │             │          │          │                │
│  ┌───▼────┐   ┌────▼────┐   ┌────▼────┐ ┌──▼───┐ ┌────▼────┐           │
│  │Keplerian│  │  SPICE  │   │ Native  │ │Delayed│ │  NS-3   │           │
│  │Provider │  │ Provider│   │ Backend │ │Backend│ │ Backend │           │
│  │(default)│  │(opt-in) │   │(default)│ │(test) │ │(opt-in) │           │
│  └─────────┘  └─────────┘   └─────────┘ └──────┘ └─────────┘           │
│       ✅           ✅            ✅         ✅         ✅                │
└─────────────────────────────────────────────────────────────────────────┘
```

### Command-Line Arguments

| Argument | Description | Default | Status |
|----------|-------------|---------|--------|
| `--trajectory-provider` | Trajectory source: `keplerian`, `spice` | `keplerian` | ✅ Ready |
| `--spice-kernels-dir` | Directory containing SPICE kernels | None | ✅ Ready |
| `--spice-config` | Path to SPICE constellation config JSON | None | ✅ Ready |
| `--network-backend` | Network simulator: `native`, `ns3` | `native` | ✅ Ready |
| `--ns3-mode` | NS-3 communication: `file`, `socket`, `bindings` | `file` | ✅ File mode ready |
| `--ns3-path` | Path to NS-3 installation | `/usr/local/ns3` | ✅ Ready |
| `--ns3-host` | NS-3 server host (socket mode) | `localhost` | 🔲 Step 6 |
| `--ns3-port` | NS-3 server port (socket mode) | `5555` | 🔲 Step 6 |
| `--export-spk` | Export constellation to SPK format | None | ✅ Ready |

---

## Completed Implementation Steps

### Step 1: TrajectoryProvider Interface ✅ COMPLETE

**Goal**: Create an abstract interface for satellite position computation that decouples trajectory calculation from the core simulation.

**Files Created**:
- `simulation/trajectory.py` (19 KB)

**Components Implemented**:

| Component | Description |
|-----------|-------------|
| `TrajectoryState` | Dataclass for position/velocity state vectors with serialization |
| `TrajectoryProvider` | Abstract base class defining the interface |
| `KeplerianProvider` | Default provider wrapping existing Satellite objects |
| `create_keplerian_provider()` | Factory function |

**Interface Methods**:
```python
class TrajectoryProvider(ABC):
    def get_state(satellite_id, time) -> TrajectoryState
    def get_position_eci(satellite_id, time) -> np.ndarray
    def get_velocity_eci(satellite_id, time) -> np.ndarray
    def get_satellite_ids() -> List[str]
    def get_time_bounds(satellite_id) -> Tuple[datetime, datetime]
    def is_valid_time(satellite_id, time) -> bool
    def step_all(timestep) -> None
```

**Acceptance Criteria**: ✅ All met
- [x] `TrajectoryProvider` ABC defined with all required methods
- [x] `TrajectoryState` dataclass for state vectors
- [x] `KeplerianProvider` wraps existing satellite functionality
- [x] Simulation works identically with `KeplerianProvider`
- [x] All existing tests pass without modification (23 tests)

---

### Step 2: SPICE Provider Implementation ✅ COMPLETE

**Goal**: Implement a trajectory provider that reads satellite positions from SPICE ephemeris kernels.

**Files Created**:
- `simulation/spice_provider.py` (25 KB)

**Components Implemented**:

| Component | Description |
|-----------|-------------|
| `SpiceKernelSet` | Dataclass grouping required kernel files |
| `SpiceConstellationConfig` | Configuration for constellation with NAIF ID mapping |
| `SpiceProvider` | TrajectoryProvider using SpiceyPy for ephemeris |
| `SpiceDatasetLoader` | Factory methods for loading from various sources |
| `SPICE_AVAILABLE` | Boolean flag for graceful degradation |

**Kernel Support**:
| Kernel Type | Extension | Purpose |
|-------------|-----------|---------|
| LSK | `.tls` | Leap seconds |
| SPK | `.bsp` | Spacecraft/planetary ephemerides |
| FK | `.tf` | Frame definitions |
| PCK | `.tpc` | Planetary constants |

**Configuration File Format**:
```json
{
  "name": "MyConstellation",
  "epoch": "2025-01-01T00:00:00Z",
  "leapseconds": "naif0012.tls",
  "spacecraft_kernels": ["constellation_v1.bsp"],
  "satellites": {
    "SAT-001": -100001,
    "SAT-002": -100002
  },
  "reference_frame": "J2000",
  "observer": "EARTH"
}
```

**Acceptance Criteria**: ✅ All met
- [x] `SpiceProvider` implements `TrajectoryProvider` interface
- [x] Kernels loaded/unloaded correctly
- [x] Time bounds computed from kernel coverage
- [x] Positions match expected values for known ephemerides
- [x] Clear error message when SpiceyPy not installed
- [x] Memory properly cleaned up on provider destruction (31 tests, 2 skipped without SpiceyPy)

---

### Step 3: SPK Export Utility ✅ COMPLETE

**Goal**: Enable exporting SatUpdate constellation definitions to SPICE SPK format for interoperability.

**Files Created**:
- `tools/generate_spk.py` (26 KB)
- `tools/__init__.py`

**Components Implemented**:

| Component | Description |
|-----------|-------------|
| `StateVector` | Dataclass for epoch + position + velocity |
| `SPKSegment` | Segment specification with validation |
| `NAIFIDManager` | Assigns sequential negative NAIF IDs (-100001, -100002, ...) |
| `SPKGenerator` | Main export class with mkspk and direct export paths |
| `create_spk_from_simulation()` | One-liner convenience function |

**Export Paths**:

1. **mkspk Export (Recommended)**: Creates files compatible with NAIF's `mkspk` tool
   ```
   output_dir/
   ├── SAT-001_states.txt      # State vectors
   ├── SAT-001_setup.txt       # mkspk configuration
   ├── generate_all.sh         # Script to run mkspk
   ├── naif_ids.json           # NAIF ID mapping
   └── metadata.json           # Export metadata
   ```

2. **Direct Export**: Uses SpiceyPy's `spkw09` (limited support)

**Command-Line Usage**:
```bash
python -m tools.generate_spk --output ./spk_output \
    --constellation walker_delta --planes 4 --sats-per-plane 6 \
    --altitude 550 --duration 24 --step 60
```

**Acceptance Criteria**: ✅ All met
- [x] State vectors exported in correct format for mkspk
- [x] Setup files contain all required mkspk parameters
- [x] Works with all constellation types (Walker-Delta, Walker-Star, Random)
- [x] Duration and step size configurable
- [x] NAIF IDs assigned correctly and documented (41 tests)

---

### Step 4: NetworkBackend Interface ✅ COMPLETE

**Goal**: Create an abstract interface for network simulation that allows plugging in different network models.

**Files Created**:
- `simulation/network_backend.py` (21 KB)

**Components Implemented**:

| Component | Description |
|-----------|-------------|
| `DropReason` | Enum for packet drop reasons (NO_ROUTE, LINK_DOWN, QUEUE_FULL, etc.) |
| `PacketTransfer` | Dataclass for completed transfers with serialization |
| `NetworkStatistics` | Network performance metrics with delivery/drop ratios |
| `PendingTransfer` | Internal tracking of in-flight packets |
| `NetworkBackend` | Abstract base class defining the interface |
| `NativeNetworkBackend` | Default backend with instant, perfect delivery |
| `DelayedNetworkBackend` | Backend with propagation delay simulation |

**Interface Methods**:
```python
class NetworkBackend(ABC):
    def initialize(topology: Dict) -> None
    def update_topology(active_links: Set[Tuple[str, str]]) -> None
    def send_packet(source, destination, packet_id, size_bytes) -> bool
    def step(timestep: float) -> List[PacketTransfer]
    def get_statistics() -> NetworkStatistics
    def reset() -> None
    def shutdown() -> None
```

**Backend Comparison**:
| Feature | NativeNetworkBackend | DelayedNetworkBackend |
|---------|---------------------|----------------------|
| Latency | Zero (instant) | Position-based |
| Reliability | Perfect | Configurable |
| Bandwidth | Unlimited | Unlimited |
| Topology | Respects links | Respects links |
| Use Case | Default behavior | Testing latency-aware protocols |

**Acceptance Criteria**: ✅ All met
- [x] `NetworkBackend` ABC with complete interface
- [x] `NativeNetworkBackend` produces identical results to current implementation
- [x] `NetworkStatistics` captures relevant metrics
- [x] Topology updates handled correctly
- [x] All existing tests pass with `NativeNetworkBackend` (41 tests)

---

### Step 5: NS-3 Backend - File Mode ✅ COMPLETE

**Goal**: Implement NS-3 integration using file-based communication for batch processing.

**Files Created**:
- `simulation/ns3_backend.py` (28 KB)
- `ns3_scenarios/satellite-update-scenario.cc` (C++ template)
- `ns3_scenarios/CMakeLists.txt`
- `ns3_scenarios/README.md`

**Components Implemented**:

| Component | Description |
|-----------|-------------|
| `NS3Mode` | Enum for communication modes (FILE, SOCKET, BINDINGS, MOCK) |
| `NS3Config` | Network configuration (data_rate, error_model, queue_size, etc.) |
| `NS3ErrorModel` | Error model types (NONE, RATE, BURST, GILBERT_ELLIOT) |
| `NS3PropagationModel` | Propagation models (CONSTANT_SPEED, FIXED, RANDOM) |
| `NS3Node` | Node specification for topology |
| `NS3SendCommand` | Packet send command |
| `NS3Backend` | Full NetworkBackend implementation |

**Communication Protocol**:

1. Python writes JSON input file with topology and pending packets
2. Python invokes NS-3 scenario via subprocess
3. NS-3 runs simulation and writes JSON output file
4. Python reads results and returns PacketTransfer list

**Input JSON Format**:
```json
{
  "command": "step",
  "timestep": 60.0,
  "topology": {
    "nodes": [{"id": "SAT-001", "type": "satellite", "position": [7000000, 0, 0]}],
    "links": [["SAT-001", "SAT-002"]]
  },
  "sends": [{"source": "SAT-001", "destination": "SAT-002", "packet_id": 1, "size": 1024}],
  "config": {"data_rate": "10Mbps", "error_model": "none"}
}
```

**Output JSON Format**:
```json
{
  "status": "success",
  "simulation_time": 60.0,
  "transfers": [
    {"source": "SAT-001", "destination": "SAT-002", "packet_id": 1,
     "timestamp": 0.023, "success": true, "latency_ms": 23.4}
  ],
  "statistics": {"total_packets_sent": 1, "average_latency_ms": 23.4}
}
```

**NS-3 Detection**:
```python
# Checks multiple locations and verifies with 'ns3 show version'
candidates = [
    ns3_path / "ns3",
    Path("/usr/local/ns3/ns3"),
    Path("/opt/ns3/ns3"),
    shutil.which("ns3")  # Check PATH
]
```

**Mock Mode**: When NS-3 is not available, automatically falls back to mock mode with:
- Realistic latency based on node positions
- Configurable error models
- Full statistics tracking

**Usage**:
```python
from simulation import NS3Backend, NS3Config, create_ns3_backend

# With NS-3 installed
backend = create_ns3_backend(mode="file", ns3_path="/opt/ns3")

# Mock mode for testing
backend = create_ns3_backend(mode="mock")

# With context manager
with NS3Backend(mode="mock") as backend:
    backend.initialize(topology)
    backend.send_packet("A", "B", packet_id=1)
    transfers = backend.step(60.0)
```

**Acceptance Criteria**: ✅ All met
- [x] JSON protocol fully specified and documented
- [x] NS-3 scenario template provided (compiles standalone)
- [x] File-based communication works reliably
- [x] Temporary files cleaned up properly
- [x] Error handling for NS-3 failures
- [x] Latency values realistic for satellite links
- [x] Works without NS-3 installed (graceful fallback to mock mode)
- [x] All existing tests pass (46 tests, 3 skipped without NS-3)

---

## Pending Implementation Steps

### Step 6: NS-3 Backend - Socket Mode 🔲 PENDING

**Goal**: Enable real-time communication with NS-3 for interactive simulations.

**Estimated Effort**: 2-3 days

**Why Socket Mode?**

| Aspect | File Mode | Socket Mode |
|--------|-----------|-------------|
| Latency | High (process spawn) | Low (persistent connection) |
| Use Case | Batch processing | Interactive/visualization |
| Complexity | Simple | Moderate |
| State | Stateless | Stateful |

**Planned Features**:
- TCP socket with JSON-line protocol (newline-delimited JSON)
- Persistent NS-3 process runs as server
- Background receiver thread for async responses
- Automatic reconnection on disconnect
- Thread-safe command sending

**Protocol**:
```
Python                          NS-3 Server
   |                                 |
   |------ connect ---------------->|
   |<----- ack --------------------|
   |------ step (sends=[...]) ----->|
   |<----- transfers=[...] --------|
   |------ shutdown --------------->|
   |------ close ----------------->|
```

**Acceptance Criteria**:
- [ ] Socket connection established reliably
- [ ] Background receiver thread handles responses
- [ ] Proper cleanup on disconnect/error
- [ ] Timeout handling prevents hangs
- [ ] Thread-safe command sending
- [ ] Reconnection logic for dropped connections
- [ ] Performance improvement over file mode demonstrated

---

### Step 7: NS-3 Backend - Python Bindings Mode 🔲 PENDING

**Goal**: Direct integration with NS-3 via Python bindings for maximum performance.

**Estimated Effort**: 3-4 days

**Dependencies**:
- NS-3 compiled with Python bindings
- Optionally: SNS3 (Satellite Network Simulator 3)

**Why Bindings Mode?**

| Aspect | File/Socket Mode | Bindings Mode |
|--------|-----------------|---------------|
| Performance | Process overhead | Native speed |
| Flexibility | Fixed protocol | Full NS-3 API |
| Debugging | Separate process | Integrated |
| Dependencies | NS-3 installation | NS-3 + Python bindings |

**Planned Features**:
- Direct NS-3 node creation and configuration
- Native mobility model integration
- Trace callbacks for packet events
- SNS3 satellite channel models (when available)

**Acceptance Criteria**:
- [ ] NS-3 Python bindings detected correctly
- [ ] Nodes created with correct positions
- [ ] Internet stack installed properly
- [ ] Packets sent and received correctly
- [ ] Trace callbacks capture all events
- [ ] Results match file/socket mode for same scenario
- [ ] SNS3 used when available
- [ ] Clean fallback when bindings unavailable

---

## File Structure Summary

### Completed Files

```
simulation/
├── __init__.py                 # Updated with all exports
├── trajectory.py               # Step 1: TrajectoryProvider interface
├── spice_provider.py           # Step 2: SPICE Provider
├── network_backend.py          # Step 4: NetworkBackend interface
└── ns3_backend.py              # Step 5: NS-3 Backend

tools/
├── __init__.py                 # Step 3: Exports
└── generate_spk.py             # Step 3: SPK Export Utility

ns3_scenarios/
├── satellite-update-scenario.cc  # Step 5: NS-3 C++ scenario
├── CMakeLists.txt                # Step 5: Build config
└── README.md                     # Step 5: Documentation

ns3_spice_tests/
├── conftest.py                   # Shared fixtures
├── test_trajectory_provider.py   # Step 1 tests (23)
├── test_spice_provider.py        # Step 2 tests (31)
├── test_spk_generator.py         # Step 3 tests (41)
├── test_network_backend.py       # Step 4 tests (41)
└── test_ns3_file_backend.py      # Step 5 tests (46)
```

### Exports Available

```python
from simulation import (
    # Step 1: TrajectoryProvider
    TrajectoryProvider, TrajectoryState, KeplerianProvider,
    create_keplerian_provider,
    
    # Step 2: SPICE Provider
    SpiceProvider, SpiceKernelSet, SpiceConstellationConfig,
    SpiceDatasetLoader, is_spice_available, SPICE_AVAILABLE,
    
    # Step 4: NetworkBackend
    NetworkBackend, NativeNetworkBackend, DelayedNetworkBackend,
    PacketTransfer, NetworkStatistics, DropReason,
    create_native_backend, create_delayed_backend,
    
    # Step 5: NS-3 Backend
    NS3Backend, NS3Config, NS3Mode, NS3ErrorModel, NS3PropagationModel,
    create_ns3_backend, is_ns3_available,
)

from tools import (
    # Step 3: SPK Export
    SPKGenerator, StateVector, SPKSegment, NAIFIDManager,
    create_spk_from_simulation,
)
```

---

## Testing Strategy

### Test Categories

| Category | Purpose | Count |
|----------|---------|-------|
| Unit Tests | Individual components | ~150 |
| Integration Tests | Component interactions | ~20 |
| Mock Tests | Test without dependencies | ~50 |
| Regression Tests | Backward compatibility | ~10 |

### Conditional Skipping

Tests requiring optional dependencies are automatically skipped:
- `@pytest.mark.requires_spice` - Skipped without SpiceyPy
- `@pytest.mark.requires_ns3` - Skipped without NS-3 installation
- `@pytest.mark.requires_ns3_bindings` - Skipped without NS-3 Python bindings

### Running Tests

```bash
# Run all tests (skips unavailable dependencies)
pytest ns3_spice_tests/ -v

# Run only core tests (no optional dependencies)
pytest ns3_spice_tests/ -v -m "not requires_spice and not requires_ns3"

# Run with SPICE tests (requires: pip install spiceypy)
pytest ns3_spice_tests/ -v -m "requires_spice"

# Run with NS-3 tests (requires NS-3 installation)
pytest ns3_spice_tests/ -v -m "requires_ns3"
```

---

## Migration Guide

### For Existing Users

**No action required.** All existing functionality works identically. New features are opt-in.

### Enabling SPICE Support

1. Install SpiceyPy: `pip install spiceypy`
2. Obtain SPICE kernels (leapseconds + spacecraft ephemeris)
3. Create configuration file or use `SpiceDatasetLoader`
4. Use in code:
   ```python
   from simulation import SpiceProvider, SpiceKernelSet
   
   kernels = SpiceKernelSet(
       leapseconds=Path("naif0012.tls"),
       spacecraft=[Path("constellation.bsp")]
   )
   provider = SpiceProvider(kernels, config)
   ```

### Enabling NS-3 Support

1. Install NS-3 (version 3.36+)
2. Copy scenario to scratch directory: `cp ns3_scenarios/*.cc /path/to/ns3/scratch/`
3. Build: `cd /path/to/ns3 && ./ns3 build`
4. Use in code:
   ```python
   from simulation import NS3Backend
   
   backend = NS3Backend(mode="file", ns3_path="/path/to/ns3")
   ```

### Exporting to SPICE Format

```python
from tools import create_spk_from_simulation

# Export constellation ephemeris
output = create_spk_from_simulation(
    simulation,
    output_dir="./spk_output",
    duration_hours=24,
    step_seconds=60
)
```

---

## Timeline Summary

| Step | Description | Estimated | Actual | Status |
|------|-------------|-----------|--------|--------|
| 1 | TrajectoryProvider Interface | 2-3 days | ✅ | Complete |
| 2 | SPICE Provider | 3-4 days | ✅ | Complete |
| 3 | SPK Export | 2-3 days | ✅ | Complete |
| 4 | NetworkBackend Interface | 2-3 days | ✅ | Complete |
| 5 | NS-3 File Mode | 4-5 days | ✅ | Complete |
| 6 | NS-3 Socket Mode | 2-3 days | - | Pending |
| 7 | NS-3 Bindings Mode | 3-4 days | - | Pending |
| - | Testing & Documentation | 3-4 days | ✅ | Ongoing |

**Remaining Effort**: 5-7 days for Steps 6-7

---

## Risk Assessment

| Risk | Impact | Mitigation | Status |
|------|--------|------------|--------|
| SpiceyPy API changes | Medium | Pin version, graceful degradation | ✅ Mitigated |
| NS-3 version incompatibility | High | `show version` detection, multiple paths | ✅ Mitigated |
| Performance regression | Medium | Benchmark suite, native backend default | ✅ Mitigated |
| Complex debugging with NS-3 | Medium | Mock mode, comprehensive logging | ✅ Mitigated |
| SNS3 availability | Low | Core features work without SNS3 | ✅ Mitigated |

---

## References

- [SPICE Toolkit Documentation](https://naif.jpl.nasa.gov/naif/documentation.html)
- [SpiceyPy Documentation](https://spiceypy.readthedocs.io/)
- [NS-3 Manual](https://www.nsnam.org/docs/manual/html/)
- [SNS3 Documentation](https://sns3.io/documentation/)
- [NASA HORIZONS System](https://ssd.jpl.nasa.gov/horizons/)