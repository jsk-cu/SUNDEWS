#!/usr/bin/env python3
"""
SatUpdate - Satellite Constellation Simulator

Command-line entry point for running satellite constellation simulations
with optional visualization.

Usage:
    python main.py                                  # Default Walker-Delta
    python main.py --type walker_star               # Walker-Star (polar)
    python main.py --type random --num 10           # 10 random satellites
    python main.py --headless --duration 3600       # Headless simulation
    python main.py --log-loc sim.json               # Enable logging, save to sim.json
    python main.py --help                           # Show all options
"""

import argparse
import math
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Satellite Constellation Simulator and Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Default Walker-Delta (3×4 sats)
  %(prog)s --type walker_delta -p 6 -s 8      # 6 planes × 8 satellites
  %(prog)s --type walker_star -p 4 -s 6       # Polar constellation
  %(prog)s --type random --num 15             # 15 random satellites
  %(prog)s --headless --duration 7200         # 2-hour headless run
  %(prog)s --bs-range 5000 --comm-range 3000  # Custom communication ranges
  %(prog)s --log-loc output.json              # Enable logging, save to output.json

Controls (visualization mode):
  Arrow keys  : Rotate camera
  +/-         : Zoom in/out
  [ ]         : Decrease/increase time scale
  SPACE       : Pause/Resume
  R           : Regenerate constellation
  ESC         : Quit
        """,
    )

    # -------------------------------------------------------------------------
    # Constellation type
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--type",
        "-t",
        type=str,
        choices=["walker_delta", "walker_star", "random"],
        default="walker_delta",
        help="Constellation type (default: walker_delta)",
    )

    # -------------------------------------------------------------------------
    # Walker constellation parameters
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--planes",
        "-p",
        type=int,
        default=3,
        help="Number of orbital planes for Walker constellations (default: 3)",
    )
    parser.add_argument(
        "--sats-per-plane",
        "-s",
        type=int,
        default=4,
        help="Satellites per plane for Walker constellations (default: 4)",
    )
    parser.add_argument(
        "--phasing",
        "-f",
        type=int,
        default=1,
        help="Walker phasing parameter F (default: 1)",
    )

    # -------------------------------------------------------------------------
    # Random constellation parameters
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--num",
        "-n",
        type=int,
        default=10,
        help="Number of satellites for random constellation (default: 10)",
    )

    # -------------------------------------------------------------------------
    # Orbital parameters
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--altitude",
        "-a",
        type=float,
        default=550.0,
        help="Orbital altitude in km (default: 550)",
    )
    parser.add_argument(
        "--inclination",
        "-i",
        type=float,
        default=53.0,
        help="Orbital inclination in degrees (default: 53)",
    )

    # -------------------------------------------------------------------------
    # Communication parameters
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--comm-range",
        type=float,
        default=None,
        help="Inter-satellite communication range in km (default: unlimited)",
    )
    parser.add_argument(
        "--num-packets",
        type=int,
        default=100,
        help="Number of packets in the software update (default: 100)",
    )

    # -------------------------------------------------------------------------
    # Agent controller selection
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--agent-controller",
        type=str,
        choices=["base", "min"],
        default="min",
        help="Agent controller type: 'base' (dummy, no requests), "
             "'min' (orders by completion, requests lowest packets) (default: min)",
    )

    # -------------------------------------------------------------------------
    # Base station parameters
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--bs-latitude",
        type=float,
        default=0.0,
        help="Base station latitude in degrees (default: 0)",
    )
    parser.add_argument(
        "--bs-longitude",
        type=float,
        default=0.0,
        help="Base station longitude in degrees (default: 0)",
    )
    parser.add_argument(
        "--bs-altitude",
        type=float,
        default=0.0,
        help="Base station altitude above sea level in km (default: 0)",
    )
    parser.add_argument(
        "--bs-range",
        type=float,
        default=10000.0,
        help="Base station communication range in km (default: 10000)",
    )

    # -------------------------------------------------------------------------
    # Simulation control
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--time-scale",
        type=float,
        default=60.0,
        help="Time scale: sim seconds per real second (default: 60)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--paused",
        action="store_true",
        help="Start with simulation paused",
    )

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--log-loc",
        type=str,
        default=None,
        help="Path to save simulation log (enables logging when specified)",
    )

    # -------------------------------------------------------------------------
    # Headless mode
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run simulation without visualization",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3600.0,
        help="Simulation duration in seconds for headless mode (default: 3600)",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=60.0,
        help="Simulation timestep in seconds for headless mode (default: 60)",
    )

    # -------------------------------------------------------------------------
    # Window settings
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--width",
        type=int,
        default=1000,
        help="Window width in pixels (default: 1000)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=800,
        help="Window height in pixels (default: 800)",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Import simulation components
    # -------------------------------------------------------------------------
    from simulation import (
        Simulation,
        SimulationConfig,
        ConstellationType,
    )
    from agents import get_agent_class

    type_map = {
        "walker_delta": ConstellationType.WALKER_DELTA,
        "walker_star": ConstellationType.WALKER_STAR,
        "random": ConstellationType.RANDOM,
    }

    # Get the selected agent class
    agent_class = get_agent_class(args.agent_controller)

    # Determine if logging is enabled
    enable_logging = args.log_loc is not None

    # -------------------------------------------------------------------------
    # Create simulation configuration
    # -------------------------------------------------------------------------
    config = SimulationConfig(
        constellation_type=type_map[args.type],
        num_planes=args.planes,
        sats_per_plane=args.sats_per_plane,
        num_satellites=args.num,
        altitude=args.altitude,
        inclination=math.radians(args.inclination),
        phasing_parameter=args.phasing,
        random_seed=args.seed,
        communication_range=args.comm_range,
        num_packets=args.num_packets,
        base_station_latitude=args.bs_latitude,
        base_station_longitude=args.bs_longitude,
        base_station_altitude=args.bs_altitude,
        base_station_range=args.bs_range,
        agent_class=agent_class,
    )

    # -------------------------------------------------------------------------
    # Create and initialize simulation
    # -------------------------------------------------------------------------
    sim = Simulation(config, enable_logging=enable_logging)
    sim.initialize(timestep=args.timestep)

    # -------------------------------------------------------------------------
    # Print configuration summary
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("SatUpdate - Satellite Constellation Simulator")
    print("=" * 60)
    print(f"\nConstellation Type: {args.type}")
    print(f"Total Satellites: {sim.num_satellites}")
    print(f"Orbital Planes: {sim.num_orbits}")
    print(f"Altitude: {args.altitude} km")
    print(f"Inclination: {args.inclination}°")

    if args.type != "random":
        print(f"Phasing Parameter: {args.phasing}")

    if sim.satellites:
        period = sim.satellites[0].orbit.period
        print(f"Orbital Period: {period/60:.1f} min")

    print(f"\nPackets in Update: {args.num_packets}")
    print(f"Agent Controller: {args.agent_controller} ({agent_class.description})")
    if args.comm_range is not None:
        print(f"Inter-satellite Range: {args.comm_range} km")
    else:
        print("Inter-satellite Range: Unlimited")

    print(f"\nBase Station:")
    print(f"  Location: ({args.bs_latitude}°, {args.bs_longitude}°)")
    print(f"  Altitude: {args.bs_altitude} km")
    print(f"  Range: {args.bs_range} km")

    if enable_logging:
        print(f"\nLogging: Enabled (will save to {args.log_loc})")
    else:
        print(f"\nLogging: Disabled")

    # -------------------------------------------------------------------------
    # Run simulation
    # -------------------------------------------------------------------------
    if args.headless:
        print(f"\n{'=' * 60}")
        print(f"Running headless simulation for up to {args.duration:.0f} seconds...")
        print(f"Timestep: {args.timestep:.1f} seconds")
        print(f"(Will terminate early if all satellites receive all packets)")
        print(f"{'=' * 60}")

        elapsed = 0.0
        report_interval = max(600, args.duration / 10)
        next_report = report_interval

        while elapsed < args.duration:
            sim.step(args.timestep)
            elapsed += args.timestep

            if elapsed >= next_report:
                print(f"\nTime: {elapsed/60:.1f} minutes")

                # Sample satellite positions
                for sat_id, geo in list(sim.state.satellite_positions.items())[:3]:
                    print(
                        f"  {sat_id}: alt={geo.altitude:.0f} km, "
                        f"lat={geo.latitude_deg:+.1f}°, lon={geo.longitude_deg:+.1f}°"
                    )

                if sim.num_satellites > 3:
                    print(f"  ... and {sim.num_satellites - 3} more satellites")

                # Show update progress
                stats = sim.state.agent_statistics
                print(f"  Update progress: {stats.average_completion:.1f}% average")
                print(
                    f"  Fully updated: {stats.fully_updated_count}/{sim.num_satellites}"
                )

                next_report += report_interval

            # Check if update complete - terminate early
            if sim.is_update_complete():
                print("\n*** All satellites have received the complete update! ***")
                print(f"Completed at simulation time: {sim.simulation_time:.0f} seconds ({sim.simulation_time/60:.1f} minutes)")
                break

        # Final summary
        print(f"\n{'=' * 60}")
        print("Simulation Complete!")
        print(f"{'=' * 60}")
        print(f"Final simulation time: {sim.simulation_time:.0f} seconds")
        print(f"Steps executed: {sim.state.step_count}")

        stats = sim.state.agent_statistics
        print(f"\nUpdate Distribution:")
        print(f"  Average completion: {stats.average_completion:.1f}%")
        print(f"  Fully updated: {stats.fully_updated_count}/{sim.num_satellites}")

        # Inter-satellite statistics
        distances = sim.get_inter_satellite_distances()
        if distances:
            min_dist = min(distances.values())
            max_dist = max(distances.values())
            avg_dist = sum(distances.values()) / len(distances)
            print(f"\nInter-satellite distances:")
            print(f"  Min: {min_dist:.0f} km")
            print(f"  Max: {max_dist:.0f} km")
            print(f"  Avg: {avg_dist:.0f} km")

        los_matrix = sim.get_line_of_sight_matrix()
        if los_matrix:
            visible = sum(1 for v in los_matrix.values() if v)
            total = len(los_matrix)
            print(
                f"\nLine of sight: {visible}/{total} pairs "
                f"({visible/total*100:.1f}%)"
            )

        # Save log if enabled
        if enable_logging:
            sim.save_log(args.log_loc)
            print(f"\nSimulation log saved to: {args.log_loc}")

    else:
        # Run with visualization
        try:
            from visualization import Visualizer
        except ImportError as e:
            print(f"\nError: Could not import visualization module: {e}")
            print("Try running with --headless flag for simulation without graphics.")
            sys.exit(1)

        print(f"\n{'=' * 60}")
        print("Starting Visualization")
        print(f"{'=' * 60}")
        print("\nControls:")
        print("  Arrow keys  : Rotate camera")
        print("  +/-         : Zoom in/out")
        print("  [ ]         : Decrease/increase time scale")
        print("  SPACE       : Pause/Resume")
        print("  R           : Regenerate constellation")
        print("  ESC         : Quit")
        print("\nSatellite colors: Red (0%) -> Yellow (50%) -> Green (100%)")
        if enable_logging:
            print(f"\nLogging enabled: Log will be saved to {args.log_loc} when update completes")
        print()

        visualizer = Visualizer(
            width=args.width,
            height=args.height,
            time_scale=args.time_scale,
            paused=args.paused,
            log_location=args.log_loc,  # Pass log location to visualizer
        )

        visualizer.set_simulation(sim)
        visualizer.run()


if __name__ == "__main__":
    main()