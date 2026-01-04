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

Advanced usage with SPICE and NS-3:
    python main.py --trajectory-provider spice --spice-bsp constellation.bsp --spice-tls naif0012.tls
    python main.py --network-backend ns3 --ns3-mode socket --ns3-host localhost --ns3-port 5555
    python main.py --export-spk ./spk_output        # Export constellation to SPK format
"""

import argparse
import math
import sys
from pathlib import Path


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

SPICE Ephemeris Examples:
  %(prog)s --trajectory-provider spice --spice-config config.json
  %(prog)s --trajectory-provider spice --spice-bsp constellation.bsp --spice-tls naif0012.tls
  %(prog)s --export-spk ./spk_output          # Export to SPK format

NS-3 Network Simulation Examples:
  %(prog)s --network-backend ns3 --ns3-mode mock      # Mock NS-3 for testing
  %(prog)s --network-backend ns3 --ns3-mode file --ns3-path /opt/ns3
  %(prog)s --network-backend ns3 --ns3-mode socket --ns3-host localhost --ns3-port 5555
  %(prog)s --network-backend ns3 --ns3-mode bindings  # Direct Python bindings

Controls (visualization mode):
  Arrow keys  : Rotate camera
  +/-         : Zoom in/out
  [ ]         : Decrease/increase time scale
  SPACE       : Pause/Resume
  R           : Regenerate constellation
  ESC         : Quit
        """,
    )

    # =========================================================================
    # Constellation type
    # =========================================================================
    parser.add_argument(
        "--type",
        "-t",
        type=str,
        choices=["walker_delta", "walker_star", "random", "spice"],
        default="walker_delta",
        help="Constellation type: walker_delta, walker_star, random, or spice "
             "(spice requires --trajectory-provider spice) (default: walker_delta)",
    )

    # =========================================================================
    # Walker constellation parameters
    # =========================================================================
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

    # =========================================================================
    # Random constellation parameters
    # =========================================================================
    parser.add_argument(
        "--num",
        "-n",
        type=int,
        default=10,
        help="Number of satellites for random constellation (default: 10)",
    )

    # =========================================================================
    # Orbital parameters
    # =========================================================================
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

    # =========================================================================
    # Communication parameters
    # =========================================================================
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

    # =========================================================================
    # Agent controller selection
    # =========================================================================
    parser.add_argument(
        "--agent-controller",
        type=str,
        choices=["base", "min", "random", "rarity", "demand", "rank"],
        default="min",
        help="Agent controller type: 'base' (dummy, no requests), "
             "'min' (default, orders by completion, requests lowest packets), "
             "'random' (orders by completion, requests random packets), "
             "'rarity' (requests rarest packets), "
             "'demand' (requests made, most in-demand packets prioritized), "
             "'rank' (prioritize packets with fewest alternate sources)",
    )

    # =========================================================================
    # Base station parameters
    # =========================================================================
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

    # =========================================================================
    # Simulation control
    # =========================================================================
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

    # =========================================================================
    # Logging
    # =========================================================================
    parser.add_argument(
        "--log-loc",
        type=str,
        default=None,
        help="Path to save simulation log (enables logging when specified)",
    )

    # =========================================================================
    # Headless mode
    # =========================================================================
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

    # =========================================================================
    # Window settings
    # =========================================================================
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

    # =========================================================================
    # Trajectory Provider (SPICE Support)
    # =========================================================================
    parser.add_argument(
        "--trajectory-provider",
        type=str,
        choices=["keplerian", "spice"],
        default="keplerian",
        help="Trajectory computation method: 'keplerian' (default, analytical) "
             "or 'spice' (NASA SPICE ephemeris)",
    )
    parser.add_argument(
        "--spice-config",
        type=Path,
        default=None,
        help="Path to SPICE constellation configuration JSON file",
    )
    parser.add_argument(
        "--spice-kernels-dir",
        type=Path,
        default=None,
        help="Directory containing SPICE kernel files",
    )
    parser.add_argument(
        "--spice-bsp",
        type=Path,
        default=None,
        help="Path to spacecraft ephemeris kernel (.bsp file)",
    )
    parser.add_argument(
        "--spice-tls",
        type=Path,
        default=None,
        help="Path to leapseconds kernel (.tls file)",
    )
    parser.add_argument(
        "--spice-planetary-bsp",
        type=Path,
        default=None,
        help="Path to planetary ephemeris kernel (.bsp file, optional)",
    )

    # =========================================================================
    # Network Backend (NS-3 Support)
    # =========================================================================
    parser.add_argument(
        "--network-backend",
        type=str,
        choices=["native", "delayed", "ns3"],
        default="native",
        help="Network simulation backend: 'native' (instant delivery, default), "
             "'delayed' (propagation delay), or 'ns3' (high-fidelity NS-3)",
    )
    parser.add_argument(
        "--ns3-mode",
        type=str,
        choices=["file", "socket", "bindings", "mock"],
        default="file",
        help="NS-3 communication mode: 'file' (batch via JSON), "
             "'socket' (real-time TCP), 'bindings' (Python API), "
             "'mock' (testing without NS-3) (default: file)",
    )
    parser.add_argument(
        "--ns3-path",
        type=Path,
        default=None,
        help="Path to NS-3 installation directory (default: /usr/local/ns3)",
    )
    parser.add_argument(
        "--ns3-host",
        type=str,
        default="localhost",
        help="NS-3 server hostname for socket mode (default: localhost)",
    )
    parser.add_argument(
        "--ns3-port",
        type=int,
        default=5555,
        help="NS-3 server port for socket mode (default: 5555)",
    )
    parser.add_argument(
        "--ns3-data-rate",
        type=str,
        default="10Mbps",
        help="NS-3 link data rate (default: 10Mbps)",
    )
    parser.add_argument(
        "--ns3-propagation-model",
        type=str,
        choices=["constant_speed", "fixed", "random"],
        default="constant_speed",
        help="NS-3 propagation delay model (default: constant_speed)",
    )
    parser.add_argument(
        "--ns3-error-model",
        type=str,
        choices=["none", "rate", "burst", "gilbert_elliot"],
        default="none",
        help="NS-3 error model (default: none)",
    )
    parser.add_argument(
        "--ns3-error-rate",
        type=float,
        default=0.0,
        help="NS-3 error rate for rate-based error model (default: 0.0)",
    )

    # =========================================================================
    # SPK Export
    # =========================================================================
    parser.add_argument(
        "--export-spk",
        type=Path,
        default=None,
        help="Export constellation to SPICE SPK format in specified directory",
    )
    parser.add_argument(
        "--export-spk-duration",
        type=float,
        default=24.0,
        help="SPK export duration in hours (default: 24)",
    )
    parser.add_argument(
        "--export-spk-step",
        type=float,
        default=60.0,
        help="SPK export time step in seconds (default: 60)",
    )

    args = parser.parse_args()

    # =========================================================================
    # Import simulation components
    # =========================================================================
    from simulation import (
        Simulation,
        SimulationConfig,
        ConstellationType,
        is_spice_available,
        SPICE_AVAILABLE,
        is_ns3_available,
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

    # =========================================================================
    # Validate SPICE arguments
    # =========================================================================
    trajectory_provider = None
    spice_data = None
    
    if args.trajectory_provider == "spice":
        if not SPICE_AVAILABLE:
            print("\nError: SPICE provider requested but SpiceyPy is not installed.")
            print("Install with: pip install spiceypy")
            print("Or use --trajectory-provider keplerian (default)")
            sys.exit(1)

        from simulation import (
            SpiceDatasetLoader,
            load_spice_for_simulation,
        )

        # Check for configuration sources
        if args.spice_config:
            # Load from configuration file
            if not args.spice_config.exists():
                print(f"\nError: SPICE config file not found: {args.spice_config}")
                sys.exit(1)
            print(f"\nLoading SPICE configuration from: {args.spice_config}")
            trajectory_provider = SpiceDatasetLoader.from_config_file(
                args.spice_config,
                kernels_dir=args.spice_kernels_dir,
            )
            # For config file, we don't have spice_data - provider handles it
        elif args.spice_bsp and args.spice_tls:
            # Load from individual kernel files using the new module
            if not args.spice_bsp.exists():
                print(f"\nError: SPICE BSP file not found: {args.spice_bsp}")
                sys.exit(1)
            if not args.spice_tls.exists():
                print(f"\nError: SPICE TLS file not found: {args.spice_tls}")
                sys.exit(1)

            try:
                spice_data = load_spice_for_simulation(
                    spk_path=args.spice_bsp,
                    leapseconds_path=args.spice_tls,
                    planetary_path=args.spice_planetary_bsp,
                    verbose=True,
                )
                trajectory_provider = spice_data['provider']
            except Exception as e:
                print(f"\nError loading SPICE data: {e}")
                sys.exit(1)
        else:
            print("\nError: SPICE provider requires either:")
            print("  --spice-config <config.json>")
            print("  OR both --spice-bsp <file.bsp> and --spice-tls <file.tls>")
            sys.exit(1)

    # =========================================================================
    # Configure Network Backend
    # =========================================================================
    network_backend = None
    if args.network_backend == "ns3":
        from simulation import (
            NS3Backend,
            NS3Config,
            NS3Mode,
            NS3ErrorModel,
            NS3PropagationModel,
            check_ns3_bindings,
            create_ns3_backend,
        )

        # Map string arguments to enums
        propagation_map = {
            "constant_speed": NS3PropagationModel.CONSTANT_SPEED,
            "fixed": NS3PropagationModel.FIXED,
            "random": NS3PropagationModel.RANDOM,
        }
        error_map = {
            "none": NS3ErrorModel.NONE,
            "rate": NS3ErrorModel.RATE,
            "burst": NS3ErrorModel.BURST,
            "gilbert_elliot": NS3ErrorModel.GILBERT_ELLIOT,
        }

        ns3_config = NS3Config(
            data_rate=args.ns3_data_rate,
            propagation_model=propagation_map[args.ns3_propagation_model],
            error_model=error_map[args.ns3_error_model],
            error_rate=args.ns3_error_rate,
        )

        # Check for bindings mode availability
        if args.ns3_mode == "bindings" and not check_ns3_bindings():
            print("\nWarning: NS-3 Python bindings not available.")
            print("Falling back to file mode. To enable bindings:")
            print("  cd /path/to/ns3")
            print("  ./ns3 configure --enable-python-bindings")
            print("  ./ns3 build")
            args.ns3_mode = "file"

        print(f"\nConfiguring NS-3 backend:")
        print(f"  Mode: {args.ns3_mode}")
        print(f"  Data rate: {args.ns3_data_rate}")
        print(f"  Propagation model: {args.ns3_propagation_model}")
        print(f"  Error model: {args.ns3_error_model}")

        if args.ns3_mode == "socket":
            print(f"  Host: {args.ns3_host}")
            print(f"  Port: {args.ns3_port}")

        network_backend = create_ns3_backend(
            mode=args.ns3_mode,
            ns3_path=args.ns3_path,
            config=ns3_config,
            host=args.ns3_host,
            port=args.ns3_port,
        )

    elif args.network_backend == "delayed":
        from simulation import create_delayed_backend
        network_backend = create_delayed_backend()
        print("\nUsing delayed network backend (with propagation delay)")

    # =========================================================================
    # Build simulation configuration
    # =========================================================================
    # Handle SPICE constellation type
    use_spice_constellation = (args.type == "spice" and spice_data is not None)
    
    if use_spice_constellation:
        # For SPICE type, we'll use satellites from the loaded SPICE data
        constellation_type = ConstellationType.WALKER_DELTA
        num_spice_sats = len(spice_data['satellites'])
        args.planes = 1
        args.sats_per_plane = num_spice_sats
    else:
        constellation_type = type_map.get(args.type, ConstellationType.WALKER_DELTA)

    config = SimulationConfig(
        constellation_type=constellation_type,
        num_planes=args.planes,
        sats_per_plane=args.sats_per_plane,
        phasing_parameter=args.phasing,
        num_satellites=args.num,
        altitude=args.altitude,
        inclination=math.radians(args.inclination),
        num_packets=args.num_packets,
        communication_range=args.comm_range,
        random_seed=args.seed,
        base_station_latitude=args.bs_latitude,
        base_station_longitude=args.bs_longitude,
        base_station_altitude=args.bs_altitude,
        base_station_range=args.bs_range,
        agent_class=agent_class,
    )

    # =========================================================================
    # Create simulation
    # =========================================================================
    sim = Simulation(config, enable_logging=enable_logging)
    
    if use_spice_constellation:
        # Use the satellites and orbits from the SPICE data
        sim.set_custom_constellation(
            spice_data['orbits'],
            spice_data['satellites']
        )
        print(f"\nUsing {len(spice_data['satellites'])} satellites from SPICE ephemeris")
    else:
        sim.initialize()

    # =========================================================================
    # Print configuration summary
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Satellite Constellation Simulator")
    print(f"{'=' * 60}")

    if use_spice_constellation:
        print(f"\nConstellation: SPICE-defined")
        print(f"  Source: {args.spice_config or args.spice_bsp}")
        print(f"  Satellites: {len(sim.satellites)}")
        for sat in sim.satellites[:5]:
            pos = sat.get_geospatial_position()
            print(f"    {sat.satellite_id}: alt={pos.altitude:.0f} km")
        if len(sim.satellites) > 5:
            print(f"    ... and {len(sim.satellites) - 5} more")
    elif args.type == "random":
        print(f"\nConstellation: Random ({args.num} satellites)")
    else:
        print(f"\nConstellation: {args.type.replace('_', ' ').title()}")
        print(f"  Orbital planes: {args.planes}")
        print(f"  Sats per plane: {args.sats_per_plane}")
        print(f"  Total satellites: {args.planes * args.sats_per_plane}")
        print(f"  Phasing (F): {args.phasing}")

    if not use_spice_constellation:
        print(f"\nOrbital Parameters:")
        print(f"  Altitude: {args.altitude} km")
        print(f"  Inclination: {args.inclination}°")

    print(f"\nTrajectory Provider: {args.trajectory_provider}")
    if args.trajectory_provider == "spice":
        print(f"  SPICE kernels loaded successfully")

    print(f"\nNetwork Backend: {args.network_backend}")
    if args.network_backend == "ns3":
        print(f"  NS-3 mode: {args.ns3_mode}")

    print(f"\nUpdate Distribution:")
    print(f"  Packets: {args.num_packets}")
    print(f"  Agent: {args.agent_controller}")
    if args.comm_range:
        print(f"  Inter-sat comm range: {args.comm_range} km")
    else:
        print(f"  Inter-sat comm range: unlimited")

    print(f"\nBase Station:")
    print(f"  Location: ({args.bs_latitude}°, {args.bs_longitude}°)")
    print(f"  Altitude: {args.bs_altitude} km")
    print(f"  Range: {args.bs_range} km")

    if enable_logging:
        print(f"\nLogging: Enabled (will save to {args.log_loc})")
    else:
        print(f"\nLogging: Disabled")

    # =========================================================================
    # Handle SPK Export
    # =========================================================================
    if args.export_spk:
        try:
            from tools import SPKGenerator, create_spk_from_simulation
        except ImportError:
            print("\nError: SPK export tools not available.")
            print("Ensure the tools package is properly installed.")
            sys.exit(1)

        print(f"\n{'=' * 60}")
        print(f"Exporting constellation to SPK format...")
        print(f"{'=' * 60}")
        print(f"  Output directory: {args.export_spk}")
        print(f"  Duration: {args.export_spk_duration} hours")
        print(f"  Time step: {args.export_spk_step} seconds")

        output_path = create_spk_from_simulation(
            sim,
            args.export_spk,
            duration_hours=args.export_spk_duration,
            step_seconds=args.export_spk_step,
        )

        print(f"\nSPK export complete!")
        print(f"  Output: {output_path}")
        print(f"\nTo create binary SPK file, run:")
        print(f"  mkspk -setup {output_path}/mkspk_setup.txt -input {output_path}/ephemeris_data.txt -output constellation.bsp")

        # If only exporting, exit here
        if args.headless and args.duration == 0:
            return

    # =========================================================================
    # Run simulation
    # =========================================================================
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

        # Network backend statistics
        if network_backend:
            net_stats = network_backend.get_statistics()
            print(f"\nNetwork Statistics:")
            print(f"  Packets sent: {net_stats.packets_sent}")
            print(f"  Packets received: {net_stats.packets_received}")
            print(f"  Packets dropped: {net_stats.packets_dropped}")
            if net_stats.average_latency_ms > 0:
                print(f"  Average latency: {net_stats.average_latency_ms:.2f} ms")
            network_backend.shutdown()

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
            log_location=args.log_loc,
        )

        visualizer.set_simulation(sim)
        visualizer.run()

        # Cleanup network backend if used
        if network_backend:
            network_backend.shutdown()


if __name__ == "__main__":
    main()