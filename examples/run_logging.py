#!/usr/bin/env python3
"""
Example: Simulation Logging

Demonstrates how to use the logging functionality to capture simulation
state and events for analysis.

The log format consists of:
- header: Configuration and metadata for simulation reproducibility
- time_series: List of timestep records with state and events

Each timestep record contains:
1. packet_counts: {satellite_id: num_packets} for all satellites
2. communication_pairs: [(sat_a, sat_b), ...] active links
3. requests: [(requester, requestee, packet_idx, was_successful), ...]
"""

import json
import math
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import (
    Simulation,
    SimulationConfig,
    ConstellationType,
    load_simulation_log,
)
from agents import get_agent_class


def example_basic_logging():
    """
    Basic example demonstrating simulation logging.
    """
    print("=" * 70)
    print("Example 1: Basic Simulation Logging")
    print("=" * 70)

    # Create simulation with logging enabled (must be explicit)
    config = SimulationConfig(
        constellation_type=ConstellationType.WALKER_DELTA,
        num_planes=2,
        sats_per_plane=3,
        altitude=550,
        inclination=math.radians(53),
        num_packets=20,
        random_seed=42,
    )

    # Logging is off by default, enable it explicitly
    sim = Simulation(config, enable_logging=True)
    sim.initialize(timestep=60.0)

    print(f"\nCreated constellation with {sim.num_satellites} satellites")
    print(f"Packets in update: {config.num_packets}")

    # Run simulation for a few steps
    print("\nRunning simulation for 5 steps...")
    for i in range(5):
        sim.step(60.0)
        stats = sim.state.agent_statistics
        print(f"  Step {i+1}: avg completion = {stats.average_completion:.1f}%")

    # Get the log
    log = sim.get_log()

    print(f"\n--- Log Structure ---")
    print(f"Header keys: {list(log['header'].keys())}")
    print(f"Number of timesteps recorded: {len(log['time_series'])}")

    # Show first timestep (initial state)
    print(f"\n--- Initial State (Step 0) ---")
    step0 = log['time_series'][0]
    print(f"Time: {step0['time']} seconds")
    print(f"Packet counts: {step0['packet_counts']}")
    print(f"Communication pairs: {len(step0['communication_pairs'])} links")
    print(f"Requests: {len(step0['requests'])} requests")

    # Show a later timestep
    print(f"\n--- After Step 3 ---")
    step3 = log['time_series'][3]
    print(f"Time: {step3['time']} seconds")
    print(f"Packet counts: {step3['packet_counts']}")
    print(f"Number of active links: {len(step3['communication_pairs'])}")
    print(f"Sample communication pairs: {step3['communication_pairs'][:3]}")
    print(f"Number of requests: {len(step3['requests'])}")
    if step3['requests']:
        print(f"Sample requests: {step3['requests'][:3]}")

    # Save to file
    output_path = "/tmp/basic_simulation_log.json"
    sim.save_log(output_path)
    print(f"\nLog saved to: {output_path}")


def example_analyze_log():
    """
    Example demonstrating log analysis.
    """
    print("\n" + "=" * 70)
    print("Example 2: Analyzing Simulation Logs")
    print("=" * 70)

    # Run a complete simulation
    config = SimulationConfig(
        constellation_type=ConstellationType.WALKER_DELTA,
        num_planes=2,
        sats_per_plane=4,
        altitude=550,
        inclination=math.radians(53),
        num_packets=30,
        random_seed=42,
        communication_range=3000,  # Limited range
    )

    sim = Simulation(config, enable_logging=True)
    sim.initialize(timestep=60.0)

    # Run until complete or max steps
    max_steps = 100
    while not sim.is_update_complete() and sim.state.step_count < max_steps:
        sim.step(60.0)

    log = sim.get_log()

    print(f"\n--- Simulation Summary ---")
    print(f"Constellation: {log['header']['constellation_type']}")
    print(f"Satellites: {log['header']['num_satellites']}")
    print(f"Packets: {log['header']['num_packets']}")
    print(f"Communication range: {log['header']['communication_range']} km")
    print(f"Agent type: {log['header']['agent_type']}")
    print(f"Total timesteps: {len(log['time_series'])}")

    # Analyze packet distribution over time
    print(f"\n--- Packet Distribution Analysis ---")
    for step_data in log['time_series'][::10]:  # Every 10th step
        step = step_data['step']
        counts = step_data['packet_counts']
        avg_packets = sum(counts.values()) / len(counts) if counts else 0
        min_packets = min(counts.values()) if counts else 0
        max_packets = max(counts.values()) if counts else 0
        print(f"  Step {step:3d}: avg={avg_packets:.1f}, min={min_packets}, max={max_packets}")

    # Analyze request success rate
    print(f"\n--- Request Success Analysis ---")
    total_requests = 0
    successful_requests = 0

    for step_data in log['time_series']:
        for req in step_data['requests']:
            total_requests += 1
            if req[3]:  # was_successful
                successful_requests += 1

    if total_requests > 0:
        success_rate = (successful_requests / total_requests) * 100
        print(f"Total requests: {total_requests}")
        print(f"Successful: {successful_requests}")
        print(f"Success rate: {success_rate:.1f}%")

    # Analyze connectivity over time
    print(f"\n--- Connectivity Analysis ---")
    link_counts = [len(step['communication_pairs']) for step in log['time_series']]
    avg_links = sum(link_counts) / len(link_counts) if link_counts else 0
    print(f"Average active links: {avg_links:.1f}")
    print(f"Min links: {min(link_counts)}")
    print(f"Max links: {max(link_counts)}")

    # Save log
    output_path = "/tmp/analyzed_simulation_log.json"
    sim.save_log(output_path)
    print(f"\nFull log saved to: {output_path}")


def example_compare_agents():
    """
    Example comparing different agents using logs.
    """
    print("\n" + "=" * 70)
    print("Example 3: Comparing Agent Performance via Logs")
    print("=" * 70)

    base_config = dict(
        constellation_type=ConstellationType.WALKER_DELTA,
        num_planes=3,
        sats_per_plane=4,
        altitude=550,
        inclination=math.radians(53),
        num_packets=30,
        random_seed=42,
    )

    results = {}

    for agent_name in ["base", "min"]:
        print(f"\nRunning with agent: {agent_name}")

        agent_class = get_agent_class(agent_name)
        config = SimulationConfig(**base_config, agent_class=agent_class)

        sim = Simulation(config, enable_logging=True)
        sim.initialize(timestep=60.0)

        max_steps = 50
        while not sim.is_update_complete() and sim.state.step_count < max_steps:
            sim.step(60.0)

        log = sim.get_log()

        # Calculate metrics
        final_step = log['time_series'][-1]
        final_counts = final_step['packet_counts']
        avg_completion = sum(final_counts.values()) / (len(final_counts) * config.num_packets) * 100

        total_requests = sum(len(step['requests']) for step in log['time_series'])
        successful_requests = sum(
            sum(1 for r in step['requests'] if r[3])
            for step in log['time_series']
        )

        results[agent_name] = {
            'steps': len(log['time_series']) - 1,  # Exclude initial state
            'avg_completion': avg_completion,
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'is_complete': sim.is_update_complete(),
        }

        # Save individual log
        sim.save_log(f"/tmp/agent_{agent_name}_log.json")

    # Compare results
    print(f"\n--- Agent Comparison ---")
    print(f"{'Agent':<10} {'Steps':>8} {'Completion':>12} {'Requests':>10} {'Success':>10} {'Complete':>10}")
    print("-" * 62)
    for agent_name, metrics in results.items():
        success_rate = (metrics['successful_requests'] / metrics['total_requests'] * 100
                       if metrics['total_requests'] > 0 else 0)
        print(f"{agent_name:<10} {metrics['steps']:>8} "
              f"{metrics['avg_completion']:>11.1f}% "
              f"{metrics['total_requests']:>10} "
              f"{success_rate:>9.1f}% "
              f"{str(metrics['is_complete']):>10}")


def example_load_and_replay():
    """
    Example showing how to load and analyze a saved log.
    """
    print("\n" + "=" * 70)
    print("Example 4: Loading and Replaying Logs")
    print("=" * 70)

    # First create and save a log
    config = SimulationConfig(
        constellation_type=ConstellationType.WALKER_STAR,
        num_planes=4,
        sats_per_plane=3,
        altitude=800,
        inclination=math.radians(86),
        num_packets=25,
        random_seed=123,
    )

    sim = Simulation(config, enable_logging=True)
    sim.initialize(timestep=60.0)

    for _ in range(20):
        sim.step(60.0)

    log_path = "/tmp/replay_test_log.json"
    sim.save_log(log_path)
    print(f"Saved log to: {log_path}")

    # Now load and analyze
    loaded_log = load_simulation_log(log_path)

    print(f"\n--- Loaded Log Header ---")
    header = loaded_log['header']
    print(f"Constellation type: {header['constellation_type']}")
    print(f"Satellites: {header['num_satellites']}")
    print(f"Altitude: {header['altitude']} km")
    print(f"Inclination: {math.degrees(header['inclination']):.1f}°")
    print(f"Random seed: {header['random_seed']}")
    print(f"Created at: {header['created_at']}")

    print(f"\n--- Replay Analysis ---")
    time_series = loaded_log['time_series']
    print(f"Total timesteps: {len(time_series)}")

    # Track specific satellite progress
    sat_ids = list(time_series[0]['packet_counts'].keys())[:3]
    print(f"\nTracking satellites: {sat_ids}")
    print(f"{'Step':>6} " + " ".join(f"{sid:>12}" for sid in sat_ids))
    print("-" * (8 + 13 * len(sat_ids)))

    for step_data in time_series[::5]:  # Every 5th step
        step = step_data['step']
        counts = step_data['packet_counts']
        row = f"{step:>6} " + " ".join(f"{counts.get(sid, 0):>12}" for sid in sat_ids)
        print(row)


def main():
    """Run all logging examples."""
    example_basic_logging()
    example_analyze_log()
    example_compare_agents()
    example_load_and_replay()

    print("\n" + "=" * 70)
    print("All logging examples complete!")
    print("=" * 70)
    print("\nGenerated log files:")
    print("  /tmp/basic_simulation_log.json")
    print("  /tmp/analyzed_simulation_log.json")
    print("  /tmp/agent_base_log.json")
    print("  /tmp/agent_min_log.json")
    print("  /tmp/replay_test_log.json")


if __name__ == "__main__":
    main()