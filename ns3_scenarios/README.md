# NS-3 Scenarios for SatUpdate

This directory contains NS-3 scenarios for high-fidelity network simulation
of satellite constellation communications.

## Overview

The `satellite-update-scenario` provides realistic network simulation including:
- Propagation delay based on satellite positions
- Queue management and packet drops
- Configurable data rates and error models
- Link utilization tracking

## Prerequisites

1. **NS-3 Installation** (version 3.36 or later recommended)
   ```bash
   # Download NS-3
   wget https://www.nsnam.org/release/ns-allinone-3.40.tar.bz2
   tar xjf ns-allinone-3.40.tar.bz2
   cd ns-allinone-3.40
   
   # Build NS-3
   ./build.py --enable-examples --enable-tests
   ```

2. **Optional: SNS3 (Satellite Network Simulator 3)**
   For more advanced satellite-specific features like antenna patterns
   and orbit propagation within NS-3.

## Installation

### Option 1: NS-3 Scratch Directory

Copy the scenario to NS-3's scratch directory:

```bash
cp satellite-update-scenario.cc /path/to/ns-3/scratch/
cd /path/to/ns-3
./ns3 build
```

### Option 2: External Build

Use the provided CMakeLists.txt:

```bash
mkdir build && cd build
cmake .. -DNS3_DIR=/path/to/ns-3
make
```

## Usage

### Command Line

```bash
./ns3 run "satellite-update-scenario --input=input.json --output=output.json"
```

### Input JSON Format

```json
{
  "command": "step",
  "timestep": 60.0,
  "simulation_time": 120.0,
  "topology": {
    "nodes": [
      {"id": "SAT-001", "type": "satellite", "position": [7000000, 0, 0]},
      {"id": "SAT-002", "type": "satellite", "position": [0, 7000000, 0]},
      {"id": "BASE-1", "type": "ground", "position": [6371000, 0, 0]}
    ],
    "links": [
      ["SAT-001", "SAT-002"],
      ["BASE-1", "SAT-001"]
    ]
  },
  "sends": [
    {"source": "BASE-1", "destination": "SAT-001", "packet_id": 0, "size": 1024},
    {"source": "SAT-001", "destination": "SAT-002", "packet_id": 1, "size": 1024}
  ],
  "config": {
    "data_rate": "10Mbps",
    "propagation_model": "constant_speed",
    "error_model": "none",
    "queue_size": 100,
    "mtu": 1500
  }
}
```

### Output JSON Format

```json
{
  "status": "success",
  "simulation_time": 60.0,
  "transfers": [
    {
      "source": "BASE-1",
      "destination": "SAT-001",
      "packet_id": 0,
      "timestamp": 0.023,
      "success": true,
      "latency_ms": 23.4
    },
    {
      "source": "SAT-001",
      "destination": "SAT-002",
      "packet_id": 1,
      "timestamp": 0.045,
      "success": false,
      "dropped_reason": "queue_overflow"
    }
  ],
  "statistics": {
    "total_packets_sent": 2,
    "total_packets_received": 1,
    "total_packets_dropped": 1,
    "average_latency_ms": 23.4,
    "throughput_bps": 445000,
    "link_utilization": {
      "SAT-001-SAT-002": 0.85
    }
  }
}
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `data_rate` | Link data rate | "10Mbps" |
| `propagation_model` | Delay model: "constant_speed", "fixed", "random" | "constant_speed" |
| `error_model` | Error model: "none", "rate", "burst" | "none" |
| `error_rate` | Error rate for "rate" model | 0.0 |
| `queue_size` | Queue size in packets | 100 |
| `mtu` | Maximum transmission unit | 1500 |
| `propagation_speed` | Signal speed in m/s | 299792458 |

## Python Integration

The NS-3 backend is integrated into SatUpdate via the `NS3Backend` class:

```python
from simulation import NS3Backend, NS3Config

# Create backend
config = NS3Config(data_rate="100Mbps", error_model=NS3ErrorModel.RATE, error_rate=0.01)
backend = NS3Backend(mode="file", ns3_path="/path/to/ns3", config=config)

# Initialize with topology
backend.initialize({
    "nodes": [...],
    "links": [...],
})

# Send packets
backend.send_packet("SAT-001", "SAT-002", packet_id=1)

# Execute simulation step
transfers = backend.step(60.0)

# Check results
for t in transfers:
    print(f"Packet {t.packet_id}: {'✓' if t.success else '✗'} ({t.latency_ms:.1f}ms)")
```

## Mock Mode

If NS-3 is not available, the backend automatically falls back to mock mode,
which provides approximate latency simulation based on node positions:

```python
# Explicitly use mock mode
backend = NS3Backend(mode="mock")
```

## Files

- `satellite-update-scenario.cc` - Main NS-3 scenario implementation
- `CMakeLists.txt` - Build configuration for external builds
- `README.md` - This documentation

## Extending

To add new features:

1. **New propagation models**: Add to `ScenarioConfig` and implement in channel setup
2. **New error models**: Use NS-3's `RateErrorModel` or create custom models
3. **Antenna patterns**: Integrate SNS3 or implement custom gain calculations
4. **Multi-path routing**: Add routing protocols and topology management

## Troubleshooting

### NS-3 not found
Ensure NS3_DIR environment variable or CMake variable points to your NS-3 installation.

### Build errors
Check NS-3 version compatibility. This scenario targets NS-3 3.36+.

### Slow execution
For large constellations, consider:
- Reducing simulation time granularity
- Using socket mode for persistent NS-3 process
- Using simplified propagation models

## References

- [NS-3 Documentation](https://www.nsnam.org/documentation/)
- [NS-3 Tutorial](https://www.nsnam.org/docs/tutorial/html/)
- [SNS3 Satellite Module](https://sns3.io/)