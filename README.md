# SatUpdate - Satellite Constellation Simulator

A Python-based satellite constellation simulator for experimenting with orbital mechanics and multi-satellite software update distribution algorithms.

## Overview

SatUpdate provides a complete framework for simulating satellite constellations and experimenting with distributed packet dissemination protocols. The simulator includes:

- **Keplerian orbital mechanics** for accurate satellite positioning
- **Multiple constellation patterns** (Walker-Delta, Walker-Star, random)
- **Agent-based packet distribution** protocol for software update simulation
- **Ground station** modeling with configurable position and range
- **Real-time 3D visualization** using Pygame
- **Headless mode** for batch simulations and analysis
- **JSON logging** for reproducibility and post-simulation analysis

## Project Structure

```
SatUpdate/
├── simulation/              # Core simulation engine
│   ├── __init__.py
│   ├── orbit.py             # Keplerian orbital mechanics (EllipticalOrbit)
│   ├── satellite.py         # Satellite class with position tracking
│   ├── constellation.py     # Constellation generation factories
│   ├── base_station.py      # Ground station modeling
│   ├── simulation.py        # Main Simulation class with agent protocol
│   └── logging.py           # JSON logging for analysis/reproducibility
│
├── visualization/           # Pygame-based 3D rendering
│   ├── __init__.py
│   ├── camera.py            # Spherical coordinate camera
│   ├── renderer.py          # Earth, orbits, satellites rendering
│   └── visualizer.py        # Main visualization loop
│
├── agents/                  # Packet distribution agents
│   ├── __init__.py          # Agent registry and utilities
│   ├── base_agent.py        # Dummy agent (no requests)
│   └── min_agent.py         # Minimum-first strategy
│
├── examples/                # Example scripts
│   ├── run_simulation.py    # Basic simulation examples
│   └── run_logging.py       # Logging and analysis examples
│
├── main.py                  # Command-line interface
└── README.md
```

## Installation

### Requirements

- Python 3.8+
- NumPy
- Pygame (for visualization only)

```bash
# Install dependencies
pip install numpy pygame

# Run from the SatUpdate directory
cd SatUpdate
python main.py --help
```

## Quick Start

### Command Line

```bash
# Default Walker-Delta constellation (3 planes × 4 satellites)
python main.py

# Walker-Star polar constellation
python main.py --type walker_star --planes 6 --sats-per-plane 6

# Random constellation with 15 satellites
python main.py --type random --num 15

# Custom communication ranges
python main.py --comm-range 3000 --bs-range 5000

# Use different agent controllers
python main.py --agent-controller min   # Orders by completion (default)
python main.py --agent-controller base  # Dummy agent (no distribution)

# Headless simulation (terminates early when update completes)
python main.py --headless --duration 7200 --timestep 60

# Enable logging (saves JSON log when update completes)
python main.py --headless --log-loc simulation_log.json

# Full help
python main.py --help
```

### Python API

```python
from simulation import Simulation, SimulationConfig, ConstellationType
import math

# Create configuration
config = SimulationConfig(
    constellation_type=ConstellationType.WALKER_DELTA,
    num_planes=4,
    sats_per_plane=6,
    altitude=550,                    # km
    inclination=math.radians(53),
    num_packets=100,                 # packets in software update
    communication_range=5000,        # inter-satellite range (km)
    base_station_latitude=40.7,      # New York
    base_station_longitude=-74.0,
    base_station_range=8000,         # ground station range (km)
)

# Create and initialize simulation (logging off by default)
sim = Simulation(config)
sim.initialize()

# Run simulation until update complete
while not sim.is_update_complete():
    sim.step(60)  # 60 second timestep
    stats = sim.state.agent_statistics
    print(f"Time: {sim.simulation_time/60:.0f} min, "
          f"Completion: {stats.average_completion:.1f}%")
```

### With Logging

```python
from simulation import Simulation, SimulationConfig, ConstellationType
import math

config = SimulationConfig(
    constellation_type=ConstellationType.WALKER_DELTA,
    num_planes=3,
    sats_per_plane=4,
    num_packets=50,
    random_seed=42,  # For reproducibility
)

# Enable logging by passing enable_logging=True
sim = Simulation(config, enable_logging=True)
sim.initialize(timestep=60.0)

# Run until complete
while not sim.is_update_complete():
    sim.step(60.0)

# Save log to file
sim.save_log("simulation_log.json")

# Or get log as dictionary
log = sim.get_log()
```

## Command-Line Arguments

### Constellation Type
| Argument | Description |
|----------|-------------|
| `--type`, `-t` | Constellation type: `walker_delta`, `walker_star`, `random` |

### Walker Constellation Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--planes`, `-p` | 3 | Number of orbital planes |
| `--sats-per-plane`, `-s` | 4 | Satellites per plane |
| `--phasing`, `-f` | 1 | Walker phasing parameter F |

### Random Constellation
| Argument | Default | Description |
|----------|---------|-------------|
| `--num`, `-n` | 10 | Number of satellites |

### Orbital Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--altitude`, `-a` | 550 | Orbital altitude (km) |
| `--inclination`, `-i` | 53 | Inclination (degrees) |

### Communication Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--comm-range` | unlimited | Inter-satellite communication range (km) |
| `--num-packets` | 100 | Packets in software update |

### Agent Controller
| Argument | Default | Description |
|----------|---------|-------------|
| `--agent-controller` | min | Agent type: `base` (dummy), `min` (completion-ordered) |

### Base Station Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--bs-latitude` | 0 | Base station latitude (degrees) |
| `--bs-longitude` | 0 | Base station longitude (degrees) |
| `--bs-altitude` | 0 | Base station altitude (km) |
| `--bs-range` | 10000 | Base station communication range (km) |

### Logging
| Argument | Default | Description |
|----------|---------|-------------|
| `--log-loc` | None | Path to save simulation log. **Enables logging when specified.** |

Logging is **disabled by default**. Specifying `--log-loc` both enables logging and sets the output file path.

### Simulation Control
| Argument | Default | Description |
|----------|---------|-------------|
| `--time-scale` | 60 | Simulation seconds per real second (visualization only) |
| `--seed` | random | Random seed for reproducibility |
| `--paused` | false | Start simulation paused (visualization only) |

### Headless Mode
| Argument | Default | Description |
|----------|---------|-------------|
| `--headless` | false | Run without visualization |
| `--duration` | 3600 | Maximum simulation duration (seconds) |
| `--timestep` | 60 | Simulation timestep (seconds) |

**Note:** The simulation terminates early when all satellites have received all packets.

### Window Settings
| Argument | Default | Description |
|----------|---------|-------------|
| `--width` | 1000 | Window width (pixels) |
| `--height` | 800 | Window height (pixels) |

## Visualization Controls

| Key | Action |
|-----|--------|
| Arrow keys | Rotate camera |
| `+` / `-` | Zoom in/out |
| `[` / `]` | Decrease/increase time scale |
| `SPACE` | Pause/Resume |
| `R` | Regenerate constellation |
| `ESC` | Quit |

### Satellite Colors

Satellites are colored based on their software update completion status:
- **Red**: 0% packets received
- **Yellow**: 50% packets received  
- **Green**: 100% packets received (fully updated)

## Constellation Types

### Walker-Delta

Standard Walker constellation with orbital planes evenly distributed over 360° of RAAN (Right Ascension of Ascending Node). Used by LEO constellations like Starlink.

```bash
python main.py --type walker_delta --planes 6 --sats-per-plane 8
```

### Walker-Star

Similar to Walker-Delta but planes are distributed over 180° of RAAN, creating a "star" pattern when viewed from above the pole. Used for polar orbit constellations like Iridium.

```bash
python main.py --type walker_star --planes 6 --sats-per-plane 6 --inclination 86
```

### Random

Randomized orbits for testing and experimentation. Each satellite gets independent orbital parameters.

```bash
python main.py --type random --num 15 --seed 42
```

## Software Update Distribution Protocol

The simulation models a software update being disseminated from a ground station to all satellites in the constellation.

### Protocol Overview

1. **Base Station** starts with all packets (complete software update)
2. **Satellites** start with no packets
3. Each timestep, a 4-phase protocol runs:
   - **Phase 1: Broadcast** - Agents announce what packets they have
   - **Phase 2: Request** - Agents request packets from neighbors
   - **Phase 3: Respond** - Agents decide which requests to fulfill
   - **Phase 4: Transfer** - Packets are transferred between agents

### Agent Types

| Agent | CLI Name | Description |
|-------|----------|-------------|
| `BaseAgent` | `base` | **Dummy agent**: Makes no requests. No packet distribution occurs. Useful as a control case. |
| `MinAgent` | `min` | **Minimum-first** (default): Orders neighbors by completion percentage (lowest first), then requests the lowest-indexed missing packets from each. |

### Communication Requirements

For two satellites to communicate:
1. **Line of sight** - Not blocked by Earth
2. **Within range** - Distance ≤ `communication_range` (if set)

For satellite-to-ground communication:
1. **Line of sight** - Satellite visible from ground station
2. **Within range** - Distance ≤ `base_station_range`
3. **Above horizon** - Satellite elevation ≥ minimum elevation angle

## Simulation Logging

The logging system captures simulation state and events in a structured JSON format for analysis and reproducibility.

### Enabling Logging

Logging is **off by default**. Enable it by:

1. **Command line:** Use the `--log-loc` argument
   ```bash
   python main.py --headless --log-loc simulation_log.json
   ```

2. **Python API:** Pass `enable_logging=True` to `Simulation()`
   ```python
   sim = Simulation(config, enable_logging=True)
   ```

### Log Behavior

- **Headless mode:** The simulation terminates when all satellites have all packets, and the log is saved at that point.
- **Visualization mode:** The log is saved automatically when the update completes. The info panel displays "Log: SAVED" status.

### Log Format

Each log is a JSON file with two top-level keys:

```json
{
  "header": { ... },
  "time_series": [ ... ]
}
```

#### Header

Contains all configuration needed to replicate the simulation:

| Field | Description |
|-------|-------------|
| `constellation_type` | Type of constellation (`walker_delta`, `walker_star`, `random`) |
| `num_planes` | Number of orbital planes |
| `sats_per_plane` | Satellites per plane |
| `num_satellites` | Total number of satellites |
| `altitude` | Orbital altitude in km |
| `inclination` | Orbital inclination in radians |
| `phasing_parameter` | Walker phasing parameter F |
| `random_seed` | Random seed (null if not set) |
| `communication_range` | Inter-satellite range in km (null = unlimited) |
| `num_packets` | Total packets in the update |
| `agent_type` | Agent controller name |
| `base_station_latitude` | Base station latitude in degrees |
| `base_station_longitude` | Base station longitude in degrees |
| `base_station_altitude` | Base station altitude in km |
| `base_station_range` | Base station communication range in km |
| `timestep` | Simulation timestep in seconds |
| `earth_radius` | Earth radius in km |
| `earth_mass` | Earth mass in kg |
| `created_at` | ISO timestamp when log was created |
| `version` | Log format version |

#### Time Series

A list of records, one per timestep (starting at step 0):

```json
{
  "step": 5,
  "time": 300.0,
  "packet_counts": {
    "WD-P1S1": 5,
    "WD-P1S2": 3,
    "WD-P2S1": 2,
    "WD-P2S2": 4
  },
  "communication_pairs": [
    ["WD-P1S1", "WD-P2S2"],
    ["WD-P1S3", "WD-P2S4"]
  ],
  "requests": [
    ["WD-P1S1", "BASE-1", 4, true],
    ["WD-P2S2", "WD-P1S1", 3, true],
    ["WD-P1S2", "WD-P2S1", 0, false]
  ]
}
```

| Field | Description |
|-------|-------------|
| `step` | Timestep number |
| `time` | Simulation time in seconds |
| `packet_counts` | Dictionary: `{satellite_id: packet_count}` |
| `communication_pairs` | List of active links: `[[sat_a, sat_b], ...]` |
| `requests` | List of requests: `[[requester, requestee, packet_idx, was_successful], ...]` |

### Analyzing Logs

```python
from simulation import load_simulation_log

# Load a saved log
log = load_simulation_log("simulation_log.json")

# Access configuration from header
print(f"Constellation: {log['header']['constellation_type']}")
print(f"Satellites: {log['header']['num_satellites']}")
print(f"Packets: {log['header']['num_packets']}")
print(f"Agent: {log['header']['agent_type']}")

# Analyze packet distribution over time
for record in log['time_series']:
    counts = record['packet_counts']
    avg = sum(counts.values()) / len(counts)
    print(f"Step {record['step']}: avg packets = {avg:.1f}")

# Calculate request success rate
total_requests = sum(len(r['requests']) for r in log['time_series'])
successful = sum(
    sum(1 for req in r['requests'] if req[3])  # req[3] is was_successful
    for r in log['time_series']
)
if total_requests > 0:
    print(f"Request success rate: {successful/total_requests*100:.1f}%")

# Find when each satellite completed
for sat_id in log['time_series'][0]['packet_counts'].keys():
    num_packets = log['header']['num_packets']
    for record in log['time_series']:
        if record['packet_counts'][sat_id] == num_packets:
            print(f"{sat_id} completed at step {record['step']}")
            break
```

## Extending the Agent Protocol

Agents use a class hierarchy where `BaseAgent` is the base class and custom
agents subclass it, overriding `make_requests()` to implement their strategy.

### Class Hierarchy

```
BaseAgent (base class)
├── Provides 4-phase protocol interface
├── Default make_requests() returns {} (no requests)
└── Useful as control case

MinAgent(BaseAgent)
├── Overrides make_requests()
└── Orders neighbors by completion, requests lowest packets
```

### Creating a Custom Agent

```python
from agents import BaseAgent, register_agent

class MyAgent(BaseAgent):
    name = "my_agent"
    description = "My custom distribution strategy"

    def make_requests(self, neighbor_broadcasts):
        """Override to implement custom request logic."""
        requests = {}
        missing = self.get_missing_packets()

        for neighbor_id, broadcast in neighbor_broadcasts.items():
            available = missing & broadcast.get("packets", set())
            if available:
                # Request the lowest-indexed available packet
                requests[neighbor_id] = min(available)
                missing.discard(requests[neighbor_id])

        return requests

# Register so it can be used via --agent-controller my_agent
register_agent("my_agent", MyAgent)
```

### Available Methods from BaseAgent

| Method/Property | Description |
|-----------------|-------------|
| `self.packets` | Set of packet indices this agent has |
| `self.get_missing_packets()` | Returns set of missing packet indices |
| `self.has_all_packets()` | True if agent has all packets |
| `self.get_completion_percentage()` | Returns 0-100 completion percentage |
| `self.get_packet_count()` | Returns number of packets held |
| `self.is_base_station` | True if this is the ground station |
| `self.num_packets` | Total packets in the update |
| `self.num_satellites` | Number of satellites in constellation |

### Protocol Methods

| Method | When Called | Default Behavior |
|--------|-------------|------------------|
| `broadcast_state()` | Phase 1 | Returns dict with packets, completion, etc. |
| `make_requests(broadcasts)` | Phase 2 | Returns `{}` — override this! |
| `receive_requests_and_update(requests)` | Phase 3 | Grants all valid requests |
| `receive_packets_and_update(received)` | Phase 4 | Adds received packets to inventory |

## API Reference

### SimulationConfig

```python
@dataclass
class SimulationConfig:
    constellation_type: ConstellationType  # WALKER_DELTA, WALKER_STAR, RANDOM
    num_planes: int = 3                    # Orbital planes (Walker)
    sats_per_plane: int = 4                # Satellites per plane (Walker)
    num_satellites: int = 12               # Total satellites (random)
    altitude: float = 550.0                # Orbital altitude (km)
    inclination: float = 0.925             # Inclination (radians)
    phasing_parameter: int = 1             # Walker phasing F
    random_seed: Optional[int] = None      # For reproducibility
    communication_range: Optional[float] = None  # km, None = unlimited
    num_packets: int = 100                 # Packets in update
    agent_class: Optional[Type] = None     # Custom agent class
    base_station_latitude: float = 0.0     # degrees
    base_station_longitude: float = 0.0    # degrees
    base_station_altitude: float = 0.0     # km
    base_station_range: float = 10000.0    # km
```

### Simulation Class

```python
class Simulation:
    def __init__(config: SimulationConfig, enable_logging: bool = False)
    def initialize(timestep: float = 60.0) -> None
    def step(timestep: float) -> SimulationState
    def run(duration: float, timestep: float) -> List[SimulationState]
    def reset() -> None
    def regenerate(new_seed: Optional[int] = None) -> None
    def is_update_complete() -> bool
    
    # Analysis methods
    def get_inter_satellite_distances() -> Dict[Tuple[str, str], float]
    def get_line_of_sight_matrix() -> Dict[Tuple[str, str], bool]
    def get_summary() -> Dict[str, Any]
    
    # Logging methods (only work if enable_logging=True)
    def save_log(filepath: str) -> None
    def get_log() -> Dict[str, Any]
    
    # Properties
    @property
    def num_satellites() -> int
    @property
    def num_orbits() -> int
    @property
    def simulation_time() -> float
```

### Satellite Class

```python
class Satellite:
    satellite_id: str
    orbit: EllipticalOrbit
    position: float  # 0-1, position in orbit
    
    def step(timestep: float) -> None
    def get_position_eci() -> np.ndarray      # [x, y, z] in km
    def get_velocity_eci() -> np.ndarray      # [vx, vy, vz] in km/s
    def get_geospatial_position(earth_rotation: float = 0) -> GeospatialPosition
    def get_radius() -> float                 # Distance from Earth center (km)
    def get_altitude() -> float               # Altitude above surface (km)
    def get_speed() -> float                  # Orbital velocity (km/s)
    def distance_to(other: Satellite) -> float
    def has_line_of_sight(other: Satellite) -> bool
```

### EllipticalOrbit Class

```python
class EllipticalOrbit:
    # Constructor parameters
    apoapsis: float              # km from Earth center
    periapsis: float             # km from Earth center
    inclination: float           # radians (0 to π)
    longitude_of_ascending_node: float  # RAAN, radians
    argument_of_periapsis: float        # radians
    
    # Derived properties
    semi_major_axis: float       # km
    semi_minor_axis: float       # km
    eccentricity: float          # 0 to <1
    period: float                # seconds
    apoapsis_altitude: float     # km above surface
    periapsis_altitude: float    # km above surface
    
    # Methods
    def position_eci(true_anomaly: float) -> np.ndarray
    def velocity_eci(true_anomaly: float) -> np.ndarray
    def radius_at_true_anomaly(nu: float) -> float
    def velocity_at_radius(r: float) -> float
```

### Logging Functions

```python
from simulation import load_simulation_log

# Load a saved log file
log = load_simulation_log("simulation_log.json")
# Returns: {"header": {...}, "time_series": [...]}
```

## Examples

### Batch Analysis with Logging

```python
from simulation import Simulation, SimulationConfig, ConstellationType, load_simulation_log
import math

results = []
for num_planes in [3, 4, 6, 8]:
    config = SimulationConfig(
        constellation_type=ConstellationType.WALKER_DELTA,
        num_planes=num_planes,
        sats_per_plane=6,
        altitude=550,
        inclination=math.radians(53),
        num_packets=50,
        random_seed=42,
    )
    
    sim = Simulation(config, enable_logging=True)
    sim.initialize(timestep=60.0)
    
    while not sim.is_update_complete():
        sim.step(60)
    
    log_path = f"sim_{num_planes}planes.json"
    sim.save_log(log_path)
    
    results.append({
        'planes': num_planes,
        'satellites': sim.num_satellites,
        'time_to_complete': sim.simulation_time,
        'log_path': log_path,
    })

# Print summary
for r in results:
    print(f"{r['planes']} planes, {r['satellites']} sats: "
          f"{r['time_to_complete']/60:.0f} min")

# Analyze request patterns
for r in results:
    log = load_simulation_log(r['log_path'])
    total_requests = sum(len(s['requests']) for s in log['time_series'])
    print(f"{r['planes']} planes: {total_requests} total requests")
```

### Agent Comparison

```python
from simulation import Simulation, SimulationConfig, ConstellationType
from agents import get_agent_class
import math

config_base = SimulationConfig(
    constellation_type=ConstellationType.WALKER_DELTA,
    num_planes=3,
    sats_per_plane=4,
    num_packets=50,
    random_seed=42,
)

for agent_name in ["base", "min"]:
    config = SimulationConfig(
        **{k: v for k, v in config_base.__dict__.items() if k != 'agent_class'},
        agent_class=get_agent_class(agent_name),
    )
    
    sim = Simulation(config)
    sim.initialize()
    
    for _ in range(50):
        sim.step(60)
    
    stats = sim.state.agent_statistics
    print(f"{agent_name}: {stats.average_completion:.1f}% avg completion")
```

## Units

| Quantity | Unit |
|----------|------|
| Distance | kilometers (km) |
| Angles | radians (internal), degrees (CLI) |
| Time | seconds |
| Velocity | km/s |
| Mass | kg |