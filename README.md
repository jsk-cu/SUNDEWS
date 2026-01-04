# SatUpdate - Satellite Constellation Simulator

A high-fidelity satellite constellation simulator with support for multi-packet update dissemination algorithms, optional NS-3 network simulation, and SPICE ephemeris integration.

## Overview

SatUpdate simulates homogeneous satellite constellations with the goal of experimenting with algorithms for disseminating multi-packet updates to the entire constellation. The simulator provides:

- **Trajectory Simulation**: Keplerian orbit propagation or NASA SPICE ephemeris
- **Network Simulation**: Native instant delivery, delayed propagation, or NS-3 high-fidelity simulation
- **Pluggable Architecture**: Abstract interfaces allow swapping implementations without code changes
- **Graceful Degradation**: Falls back to built-in implementations when optional dependencies are unavailable

## Installation

### Basic Installation

```bash
git clone <repository>
cd satupdate
pip install -r requirements.txt
```

### Optional Dependencies

```bash
# For SPICE ephemeris support
pip install spiceypy

# For NS-3 network simulation
# See NS-3 Installation section below
```

## Quick Start

### Command Line

```bash
# Default Walker-Delta constellation (3×4 satellites)
python main.py

# Walker-Star polar constellation
python main.py --type walker_star -p 4 -s 6

# Random constellation with 15 satellites
python main.py --type random --num 15

# Headless simulation for 2 hours
python main.py --headless --duration 7200

# With logging enabled
python main.py --log-loc output.json
```

### Python API

```python
from simulation import Simulation, SimulationConfig, ConstellationType

config = SimulationConfig(
    constellation_type=ConstellationType.WALKER_DELTA,
    num_planes=4,
    sats_per_plane=6,
    altitude=550,
    num_packets=100,
)

sim = Simulation(config)
sim.initialize()

# Run simulation steps
for _ in range(100):
    sim.step(60.0)  # 60 second timestep
    
    if sim.is_update_complete():
        print("All satellites updated!")
        break
```

## Command-Line Reference

### Basic Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--type` | `-t` | `walker_delta` | Constellation type: `walker_delta`, `walker_star`, `random`, `spice` |
| `--planes` | `-p` | `3` | Number of orbital planes |
| `--sats-per-plane` | `-s` | `4` | Satellites per plane |
| `--num` | `-n` | `10` | Number of satellites (random constellation) |
| `--altitude` | `-a` | `550` | Orbital altitude in km |
| `--inclination` | `-i` | `53` | Orbital inclination in degrees |
| `--phasing` | `-f` | `1` | Walker phasing parameter F |

### Communication Options

| Option | Default | Description |
|--------|---------|-------------|
| `--comm-range` | unlimited | Inter-satellite communication range in km |
| `--num-packets` | `100` | Number of packets in the software update |
| `--agent-controller` | `min` | Agent type: `base`, `min`, `random`, `rarity`, `demand`, `rank` |

### Base Station Options

| Option | Default | Description |
|--------|---------|-------------|
| `--bs-latitude` | `0.0` | Base station latitude in degrees |
| `--bs-longitude` | `0.0` | Base station longitude in degrees |
| `--bs-altitude` | `0.0` | Base station altitude in km |
| `--bs-range` | `10000` | Base station communication range in km |

### Simulation Control

| Option | Default | Description |
|--------|---------|-------------|
| `--time-scale` | `60` | Simulation seconds per real second |
| `--seed` | None | Random seed for reproducibility |
| `--paused` | false | Start with simulation paused |
| `--headless` | false | Run without visualization |
| `--duration` | `3600` | Simulation duration in seconds (headless) |
| `--timestep` | `60` | Simulation timestep in seconds (headless) |
| `--log-loc` | None | Path to save simulation log |

### SPICE Ephemeris Options

| Option | Default | Description |
|--------|---------|-------------|
| `--trajectory-provider` | `keplerian` | Trajectory source: `keplerian` or `spice` |
| `--spice-config` | None | Path to SPICE configuration JSON file |
| `--spice-kernels-dir` | None | Directory containing SPICE kernels |
| `--spice-bsp` | None | Path to spacecraft ephemeris kernel (.bsp) |
| `--spice-tls` | None | Path to leapseconds kernel (.tls) |
| `--spice-planetary-bsp` | None | Path to planetary ephemeris kernel (.bsp) |

### NS-3 Network Options

| Option | Default | Description |
|--------|---------|-------------|
| `--network-backend` | `native` | Backend: `native`, `delayed`, or `ns3` |
| `--ns3-mode` | `file` | NS-3 mode: `file`, `socket`, `bindings`, `mock` |
| `--ns3-path` | `/usr/local/ns3` | Path to NS-3 installation |
| `--ns3-host` | `localhost` | NS-3 server host (socket mode) |
| `--ns3-port` | `5555` | NS-3 server port (socket mode) |
| `--ns3-data-rate` | `10Mbps` | Link data rate |
| `--ns3-propagation-model` | `constant_speed` | Propagation: `constant_speed`, `fixed`, `random` |
| `--ns3-error-model` | `none` | Error model: `none`, `rate`, `burst`, `gilbert_elliot` |
| `--ns3-error-rate` | `0.0` | Error rate (for rate-based model) |

### SPK Export Options

| Option | Default | Description |
|--------|---------|-------------|
| `--export-spk` | None | Output directory for SPK export |
| `--export-spk-duration` | `24.0` | Export duration in hours |
| `--export-spk-step` | `60.0` | Export time step in seconds |

## SPICE Ephemeris Support

For high-fidelity orbital mechanics using NASA's SPICE toolkit:

### Installation

```bash
pip install spiceypy
```

### Obtaining SPICE Kernels

1. **Leapseconds Kernel (.tls)**: Download from [NAIF](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/)
2. **Planetary Ephemeris (.bsp)**: Download from [NAIF](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/)
3. **Spacecraft Ephemeris**: Generate using `--export-spk` or obtain from mission archives

### Command-Line Usage

```bash
# Using configuration file
python main.py --trajectory-provider spice --spice-config config.json

# Using individual kernel files
python main.py --trajectory-provider spice \
    --spice-bsp constellation.bsp \
    --spice-tls naif0012.tls \
    --spice-planetary-bsp de440.bsp

# Export constellation to SPK format
python main.py --export-spk ./spk_output --export-spk-duration 48
```

### SPICE Configuration File Format

```json
{
    "leapseconds": "naif0012.tls",
    "spacecraft_kernels": ["constellation.bsp"],
    "planetary": ["de440.bsp"],
    "satellites": {
        "SAT-001": -100001,
        "SAT-002": -100002,
        "SAT-003": -100003
    },
    "reference_frame": "J2000",
    "observer": "EARTH"
}
```

### Python API

```python
from simulation import SpiceProvider, SpiceKernelSet, SpiceDatasetLoader
from pathlib import Path

# From configuration file
provider = SpiceDatasetLoader.from_config_file("config.json")

# From individual kernels
kernels = SpiceKernelSet(
    leapseconds=Path("naif0012.tls"),
    spacecraft=[Path("constellation.bsp")],
    planetary=[Path("de440.bsp")]
)
provider = SpiceProvider(kernels, naif_mapping)

# Get satellite state
state = provider.get_state("SAT-001", epoch)
print(f"Position: {state.position_eci}")
print(f"Velocity: {state.velocity_eci}")
```

## NS-3 Network Simulation

For high-fidelity network simulation with realistic latency, queuing, and error models:

### Network Backend Options

| Backend | Description | Use Case |
|---------|-------------|----------|
| `NativeNetworkBackend` | Instant, perfect delivery | Fast prototyping |
| `DelayedNetworkBackend` | Propagation delay simulation | Latency testing |
| `NS3Backend (mock)` | Mock NS-3 with realistic latency | Testing without NS-3 |
| `NS3Backend (file)` | Full NS-3 via file I/O | Batch processing |
| `NS3Backend (socket)` | Full NS-3 via TCP socket | Interactive simulation |
| `NS3Backend (bindings)` | Direct NS-3 Python API | Maximum performance |

### Command-Line Usage

```bash
# Mock mode for testing
python main.py --network-backend ns3 --ns3-mode mock

# File mode with NS-3 installation
python main.py --network-backend ns3 --ns3-mode file --ns3-path /opt/ns3

# Socket mode for interactive simulation
python main.py --network-backend ns3 --ns3-mode socket \
    --ns3-host localhost --ns3-port 5555

# Bindings mode for maximum performance
python main.py --network-backend ns3 --ns3-mode bindings

# With custom network parameters
python main.py --network-backend ns3 --ns3-mode mock \
    --ns3-data-rate 50Mbps \
    --ns3-propagation-model constant_speed \
    --ns3-error-model rate --ns3-error-rate 0.01
```

### Python API

```python
from simulation import NS3Backend, NS3Config, check_ns3_bindings, create_ns3_backend

# Auto-detect best available mode
if check_ns3_bindings():
    backend = NS3Backend(mode="bindings")
elif is_ns3_available():
    backend = NS3Backend(mode="file", ns3_path="/opt/ns3")
else:
    backend = NS3Backend(mode="mock")

# Configure network parameters
config = NS3Config(
    data_rate="10Mbps",
    propagation_model="constant_speed",
    error_model="none",
)

backend = NS3Backend(mode="mock", config=config)

# Initialize and use
backend.initialize(topology)
backend.send_packet("SAT-001", "SAT-002", packet_id=1)
transfers = backend.step(60.0)

# Check results
for transfer in transfers:
    print(f"{transfer.source_id} -> {transfer.destination_id}: "
          f"{'OK' if transfer.success else 'FAILED'}")
```

### NS-3 Installation

#### Ubuntu/Debian

```bash
# Install dependencies
sudo apt install g++ python3 cmake ninja-build git

# Clone NS-3
git clone https://gitlab.com/nsnam/ns-3-dev.git /opt/ns3
cd /opt/ns3

# Configure and build
./ns3 configure --enable-examples --enable-tests
./ns3 build

# Optional: Enable Python bindings
./ns3 configure --enable-python-bindings
./ns3 build
```

#### Installing the SatUpdate Scenario

```bash
cd /opt/ns3
cp /path/to/satupdate/ns3_scenarios/satellite-update-scenario.cc scratch/
./ns3 build
```

#### Starting NS-3 Socket Server

```bash
cd /opt/ns3
./ns3 run "satellite-update-scenario --server --port=5555"
```

## Agent Controllers

The simulator includes several agent controllers for packet distribution:

| Agent | Description |
|-------|-------------|
| `base` | Dummy agent, no requests |
| `min` | Orders by completion, requests lowest packet numbers |
| `random` | Orders by completion, requests random packets |
| `rarity` | Requests rarest packets first |
| `demand` | Prioritizes most in-demand packets |
| `rank` | Prioritizes packets with fewest alternate sources |

```bash
# Use different agents
python main.py --agent-controller rarity
python main.py --agent-controller demand
```

## Visualization Controls

When running with visualization (default, without `--headless`):

| Key | Action |
|-----|--------|
| Arrow keys | Rotate camera |
| `+` / `-` | Zoom in/out |
| `[` / `]` | Decrease/increase time scale |
| `SPACE` | Pause/Resume |
| `R` | Regenerate constellation |
| `ESC` | Quit |

Satellite colors indicate update progress: Red (0%) → Yellow (50%) → Green (100%)

## Examples

### Basic Simulations

```bash
# Default Walker-Delta constellation
python main.py

# Large Starlink-like constellation
python main.py --type walker_delta -p 72 -s 22 --altitude 550 --inclination 53

# Polar constellation for global coverage
python main.py --type walker_star -p 6 -s 11 --altitude 780 --inclination 86

# Quick test with small constellation
python main.py --type random --num 5 --headless --duration 300
```

### With SPICE Ephemeris

```bash
# Load from configuration
python main.py --trajectory-provider spice --spice-config my_constellation.json

# Load individual kernels and visualize
python main.py --trajectory-provider spice \
    --spice-bsp satellites.bsp \
    --spice-tls naif0012.tls

# Export Walker-Delta to SPK, then reload with SPICE
python main.py --export-spk ./output --export-spk-duration 24
# After running mkspk to create binary SPK...
python main.py --trajectory-provider spice --spice-bsp ./output/constellation.bsp --spice-tls naif0012.tls
```

### With NS-3 Network Simulation

```bash
# Mock mode for testing
python main.py --network-backend ns3 --ns3-mode mock --headless

# File mode batch simulation
python main.py --network-backend ns3 --ns3-mode file \
    --ns3-path /opt/ns3 \
    --headless --duration 7200

# Socket mode for interactive visualization
python main.py --network-backend ns3 --ns3-mode socket \
    --ns3-host localhost --ns3-port 5555

# With error modeling
python main.py --network-backend ns3 --ns3-mode mock \
    --ns3-error-model rate --ns3-error-rate 0.05
```

### Combined SPICE + NS-3

```bash
# High-fidelity simulation with SPICE orbits and NS-3 networking
python main.py \
    --trajectory-provider spice \
    --spice-config constellation.json \
    --network-backend ns3 \
    --ns3-mode socket \
    --ns3-host localhost \
    --ns3-port 5555 \
    --headless \
    --duration 86400 \
    --log-loc results.json
```

## Project Structure

```
satupdate/
├── main.py                   # Command-line entry point
├── simulation/
│   ├── __init__.py           # Module exports
│   ├── simulation.py         # Core simulation logic
│   ├── orbit.py              # Orbital mechanics
│   ├── satellite.py          # Satellite model
│   ├── constellation.py      # Constellation generators
│   ├── base_station.py       # Ground station model
│   ├── trajectory.py         # TrajectoryProvider interface
│   ├── spice_provider.py     # SPICE ephemeris provider
│   ├── network_backend.py    # NetworkBackend interface
│   ├── ns3_backend.py        # NS-3 integration (all modes)
│   └── logging.py            # Simulation logging
│
├── agents/
│   ├── __init__.py           # Agent exports
│   └── *.py                  # Agent implementations
│
├── visualization/
│   ├── __init__.py           # Visualization exports
│   ├── visualizer.py         # Main visualizer
│   ├── camera.py             # Camera controls
│   └── renderer.py           # Rendering logic
│
├── tools/
│   ├── __init__.py           # Tool exports
│   └── generate_spk.py       # SPK export utility
│
├── ns3_scenarios/
│   ├── satellite-update-scenario.cc  # NS-3 C++ scenario
│   ├── CMakeLists.txt               # Build configuration
│   └── README.md                    # NS-3 scenario documentation
│
├── examples/
│   ├── run_simulation.py     # Programmatic simulation example
│   └── run_logging.py        # Logging example
│
└── ns3_spice_tests/
    ├── conftest.py           # Test fixtures
    └── test_*.py             # Test modules
```

## Running Tests

```bash
# Run all tests
pytest ns3_spice_tests/ -v

# Run without optional dependencies
pytest ns3_spice_tests/ -v -m "not requires_spice and not requires_ns3"

# Run specific test modules
pytest ns3_spice_tests/test_network_backend.py -v
pytest ns3_spice_tests/test_spice_provider.py -v
pytest ns3_spice_tests/test_ns3_socket_mode.py -v

# Run with coverage
pytest ns3_spice_tests/ --cov=simulation --cov-report=html
```

## API Reference

### Simulation Exports

```python
from simulation import (
    # Core
    Simulation, SimulationConfig, SimulationState,
    ConstellationType, AgentStatistics,
    
    # Trajectory (Step 1)
    TrajectoryProvider, TrajectoryState, KeplerianProvider,
    
    # SPICE Provider (Step 2)
    SpiceProvider, SpiceKernelSet, SpiceConstellationConfig,
    SpiceDatasetLoader, is_spice_available, SPICE_AVAILABLE,
    
    # Network Backend (Step 4)
    NetworkBackend, NativeNetworkBackend, DelayedNetworkBackend,
    PacketTransfer, NetworkStatistics, DropReason,
    
    # NS-3 Backend (Steps 5-7)
    NS3Backend, NS3Config, NS3Mode, NS3ErrorModel, NS3PropagationModel,
    NS3SocketClient, NS3BindingsWrapper,
    check_ns3_bindings, check_sns3_bindings, create_ns3_backend,
)
```

### Tools Exports

```python
from tools import (
    SPKGenerator, StateVector, SPKSegment, NAIFIDManager,
    create_spk_from_simulation,
)
```