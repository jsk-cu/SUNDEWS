#!/usr/bin/env python3
"""
Tests for Step 3: SPK Export Utility

These tests verify:
1. StateVector and SPKSegment dataclasses
2. SPKGenerator class functionality
3. State vectors exported in correct format for mkspk
4. Setup files contain all required mkspk parameters
5. Export works with all constellation types
6. Duration and step size configurable
7. NAIF IDs assigned correctly and documented
"""

import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestStateVector:
    """Tests for StateVector dataclass."""
    
    def test_state_vector_creation(self):
        """Test StateVector can be created."""
        from tools import StateVector
        
        state = StateVector(
            epoch=datetime(2025, 1, 1, 0, 0, 0),
            position=np.array([7000.0, 0.0, 0.0]),
            velocity=np.array([0.0, 7.5, 0.0])
        )
        
        assert state.position.shape == (3,)
        assert state.velocity.shape == (3,)
        assert state.epoch == datetime(2025, 1, 1, 0, 0, 0)
    
    def test_state_vector_list_conversion(self):
        """Test StateVector converts lists to numpy arrays."""
        from tools import StateVector
        
        state = StateVector(
            epoch=datetime(2025, 1, 1),
            position=[7000.0, 100.0, 200.0],
            velocity=[0.1, 7.5, 0.2]
        )
        
        assert isinstance(state.position, np.ndarray)
        assert isinstance(state.velocity, np.ndarray)
    
    def test_state_vector_serialization(self):
        """Test StateVector to_dict and from_dict."""
        from tools import StateVector
        
        state = StateVector(
            epoch=datetime(2025, 1, 1, 12, 30, 0),
            position=np.array([7000.0, 100.0, 200.0]),
            velocity=np.array([0.1, 7.5, 0.2])
        )
        
        data = state.to_dict()
        assert "epoch" in data
        assert "position" in data
        assert "velocity" in data
        
        restored = StateVector.from_dict(data)
        np.testing.assert_allclose(restored.position, state.position)
        np.testing.assert_allclose(restored.velocity, state.velocity)


class TestSPKSegment:
    """Tests for SPKSegment dataclass."""
    
    def test_segment_creation(self):
        """Test SPKSegment can be created."""
        from tools import SPKSegment, StateVector
        
        states = [
            StateVector(
                epoch=datetime(2025, 1, 1, 0, 0, 0),
                position=[7000.0, 0.0, 0.0],
                velocity=[0.0, 7.5, 0.0]
            ),
            StateVector(
                epoch=datetime(2025, 1, 1, 0, 1, 0),
                position=[6999.0, 50.0, 0.0],
                velocity=[-0.1, 7.5, 0.0]
            )
        ]
        
        segment = SPKSegment(
            satellite_id="SAT-001",
            naif_id=-100001,
            states=states
        )
        
        assert segment.satellite_id == "SAT-001"
        assert segment.naif_id == -100001
        assert len(segment.states) == 2
        assert segment.center_body == 399  # Earth default
        assert segment.reference_frame == "J2000"
    
    def test_segment_time_bounds(self):
        """Test SPKSegment time bounds properties."""
        from tools import SPKSegment, StateVector
        
        states = [
            StateVector(datetime(2025, 1, 1, 0, 0, 0), [7000, 0, 0], [0, 7.5, 0]),
            StateVector(datetime(2025, 1, 1, 1, 0, 0), [6000, 100, 0], [0, 7.5, 0]),
        ]
        
        segment = SPKSegment("SAT-001", -100001, states)
        
        assert segment.start_time == datetime(2025, 1, 1, 0, 0, 0)
        assert segment.end_time == datetime(2025, 1, 1, 1, 0, 0)
        assert segment.duration_seconds == 3600.0
    
    def test_segment_validation_valid(self):
        """Test SPKSegment validation passes for valid segment."""
        from tools import SPKSegment, StateVector
        
        states = [
            StateVector(datetime(2025, 1, 1, 0, 0, 0), [7000, 0, 0], [0, 7.5, 0]),
            StateVector(datetime(2025, 1, 1, 0, 1, 0), [6999, 50, 0], [0, 7.5, 0]),
        ]
        
        segment = SPKSegment("SAT-001", -100001, states)
        errors = segment.validate()
        
        assert len(errors) == 0
    
    def test_segment_validation_positive_naif_id(self):
        """Test SPKSegment validation fails for positive NAIF ID."""
        from tools import SPKSegment, StateVector
        
        states = [
            StateVector(datetime(2025, 1, 1, 0, 0, 0), [7000, 0, 0], [0, 7.5, 0]),
            StateVector(datetime(2025, 1, 1, 0, 1, 0), [6999, 50, 0], [0, 7.5, 0]),
        ]
        
        segment = SPKSegment("SAT-001", 100001, states)  # Positive!
        errors = segment.validate()
        
        assert len(errors) > 0
        assert any("negative" in e for e in errors)
    
    def test_segment_validation_insufficient_states(self):
        """Test SPKSegment validation fails for single state."""
        from tools import SPKSegment, StateVector
        
        states = [
            StateVector(datetime(2025, 1, 1, 0, 0, 0), [7000, 0, 0], [0, 7.5, 0]),
        ]
        
        segment = SPKSegment("SAT-001", -100001, states)
        errors = segment.validate()
        
        assert len(errors) > 0
        assert any("2 states" in e for e in errors)


class TestNAIFIDManager:
    """Tests for NAIFIDManager class."""
    
    def test_manager_creation(self):
        """Test NAIFIDManager can be created."""
        from tools import NAIFIDManager
        
        manager = NAIFIDManager()
        assert manager is not None
    
    def test_get_id_assigns_sequential(self):
        """Test get_id assigns sequential NAIF IDs."""
        from tools import NAIFIDManager
        
        manager = NAIFIDManager()
        
        id1 = manager.get_id("SAT-001")
        id2 = manager.get_id("SAT-002")
        id3 = manager.get_id("SAT-003")
        
        assert id1 == -100001
        assert id2 == -100002
        assert id3 == -100003
    
    def test_get_id_returns_same_for_same_satellite(self):
        """Test get_id returns same ID for same satellite."""
        from tools import NAIFIDManager
        
        manager = NAIFIDManager()
        
        id1 = manager.get_id("SAT-001")
        id2 = manager.get_id("SAT-001")
        
        assert id1 == id2
    
    def test_naif_ids_are_negative(self):
        """Test all assigned NAIF IDs are negative."""
        from tools import NAIFIDManager
        
        manager = NAIFIDManager()
        
        for i in range(10):
            naif_id = manager.get_id(f"SAT-{i:03d}")
            assert naif_id < 0, f"NAIF ID {naif_id} is not negative"
    
    def test_get_mapping(self):
        """Test get_mapping returns complete mapping."""
        from tools import NAIFIDManager
        
        manager = NAIFIDManager()
        manager.get_id("SAT-001")
        manager.get_id("SAT-002")
        
        mapping = manager.get_mapping()
        
        assert len(mapping) == 2
        assert "SAT-001" in mapping
        assert "SAT-002" in mapping
    
    def test_set_explicit_mapping(self):
        """Test set_mapping allows explicit NAIF IDs."""
        from tools import NAIFIDManager
        
        manager = NAIFIDManager()
        manager.set_mapping({
            "CUSTOM-001": -200001,
            "CUSTOM-002": -200002
        })
        
        assert manager.get_id("CUSTOM-001") == -200001
        assert manager.get_id("CUSTOM-002") == -200002


class TestSPKGeneratorBasic:
    """Basic tests for SPKGenerator class."""
    
    def test_generator_creation(self, temp_output_dir):
        """Test SPKGenerator can be created."""
        from tools import SPKGenerator
        
        gen = SPKGenerator(temp_output_dir)
        
        assert gen.output_dir == temp_output_dir
        assert gen.segments == []
        assert gen.producer_id == "SatUpdate"
    
    def test_generator_custom_producer(self, temp_output_dir):
        """Test SPKGenerator with custom producer ID."""
        from tools import SPKGenerator
        
        gen = SPKGenerator(temp_output_dir, producer_id="MyProject")
        
        assert gen.producer_id == "MyProject"
    
    def test_add_satellite_trajectory(self, temp_output_dir):
        """Test adding satellite trajectory."""
        from tools import SPKGenerator
        
        gen = SPKGenerator(temp_output_dir)
        
        states = [
            {"time": datetime(2025, 1, 1, 0, 0, 0), "pos": [7000.0, 0.0, 0.0], "vel": [0.0, 7.5, 0.0]},
            {"time": datetime(2025, 1, 1, 0, 1, 0), "pos": [6999.0, 50.0, 0.0], "vel": [-0.1, 7.5, 0.0]}
        ]
        
        segment = gen.add_satellite_trajectory("SAT-001", -100001, states)
        
        assert len(gen.segments) == 1
        assert segment.satellite_id == "SAT-001"
        assert segment.naif_id == -100001
        assert len(segment.states) == 2
    
    def test_add_trajectory_auto_naif_id(self, temp_output_dir):
        """Test automatic NAIF ID assignment."""
        from tools import SPKGenerator
        
        gen = SPKGenerator(temp_output_dir)
        
        segment = gen.add_satellite_trajectory("SAT-001", states=[])
        
        assert segment.naif_id == -100001  # Auto-assigned
    
    def test_get_summary(self, temp_output_dir):
        """Test get_summary returns correct information."""
        from tools import SPKGenerator
        
        gen = SPKGenerator(temp_output_dir)
        gen.add_satellite_trajectory(
            "SAT-001", -100001,
            [
                {"time": datetime(2025, 1, 1, 0, 0, 0), "pos": [7000, 0, 0], "vel": [0, 7.5, 0]},
                {"time": datetime(2025, 1, 1, 0, 1, 0), "pos": [6999, 50, 0], "vel": [0, 7.5, 0]},
            ]
        )
        
        summary = gen.get_summary()
        
        assert summary["num_segments"] == 1
        assert len(summary["segments"]) == 1
        assert summary["segments"][0]["satellite_id"] == "SAT-001"
        assert summary["segments"][0]["num_states"] == 2


class TestSPKExportForMkspk:
    """Tests for mkspk-compatible export."""
    
    def test_export_creates_output_dir(self, tmp_path):
        """Test export creates output directory."""
        from tools import SPKGenerator
        
        output_dir = tmp_path / "spk_output"
        gen = SPKGenerator(output_dir)
        gen.add_satellite_trajectory(
            "SAT-001", -100001,
            [
                {"time": datetime(2025, 1, 1, 0, 0, 0), "pos": [7000, 0, 0], "vel": [0, 7.5, 0]},
                {"time": datetime(2025, 1, 1, 0, 1, 0), "pos": [6999, 50, 0], "vel": [0, 7.5, 0]},
            ]
        )
        
        gen.export_for_mkspk()
        
        assert output_dir.exists()
    
    def test_export_creates_state_file(self, temp_output_dir):
        """Test state vector file is created."""
        from tools import SPKGenerator
        
        gen = SPKGenerator(temp_output_dir)
        gen.add_satellite_trajectory(
            "SAT-001", -100001,
            [
                {"time": datetime(2025, 1, 1, 0, 0, 0), "pos": [7000, 0, 0], "vel": [0, 7.5, 0]},
                {"time": datetime(2025, 1, 1, 0, 1, 0), "pos": [6999, 50, 0], "vel": [0, 7.5, 0]},
            ]
        )
        gen.export_for_mkspk()
        
        state_file = temp_output_dir / "SAT-001_states.txt"
        assert state_file.exists()
    
    def test_state_file_format(self, temp_output_dir):
        """Test state file has correct format."""
        from tools import SPKGenerator
        
        gen = SPKGenerator(temp_output_dir)
        gen.add_satellite_trajectory(
            "SAT-001", -100001,
            [
                {"time": datetime(2025, 1, 1, 0, 0, 0), "pos": [7000.0, 0.0, 0.0], "vel": [0.0, 7.5, 0.0]},
                {"time": datetime(2025, 1, 1, 0, 1, 0), "pos": [6999.0, 50.0, 0.0], "vel": [-0.1, 7.5, 0.0]},
            ]
        )
        gen.export_for_mkspk()
        
        state_file = temp_output_dir / "SAT-001_states.txt"
        content = state_file.read_text()
        
        # Skip comment lines
        data_lines = [l for l in content.strip().split("\n") if not l.startswith("#")]
        
        assert len(data_lines) == 2
        
        # Parse first data line
        parts = data_lines[0].split()
        assert len(parts) == 7  # timestamp + 3 pos + 3 vel
        assert "2025-01-01" in parts[0]
    
    def test_export_creates_setup_file(self, temp_output_dir):
        """Test mkspk setup file is created."""
        from tools import SPKGenerator
        
        gen = SPKGenerator(temp_output_dir)
        gen.add_satellite_trajectory(
            "SAT-001", -100001,
            [
                {"time": datetime(2025, 1, 1, 0, 0, 0), "pos": [7000, 0, 0], "vel": [0, 7.5, 0]},
                {"time": datetime(2025, 1, 1, 0, 1, 0), "pos": [6999, 50, 0], "vel": [0, 7.5, 0]},
            ]
        )
        gen.export_for_mkspk()
        
        setup_file = temp_output_dir / "SAT-001_setup.txt"
        assert setup_file.exists()
    
    def test_setup_file_required_fields(self, temp_output_dir):
        """Test setup file contains all required mkspk parameters."""
        from tools import SPKGenerator
        
        gen = SPKGenerator(temp_output_dir)
        gen.add_satellite_trajectory(
            "SAT-001", -100001,
            [
                {"time": datetime(2025, 1, 1, 0, 0, 0), "pos": [7000, 0, 0], "vel": [0, 7.5, 0]},
                {"time": datetime(2025, 1, 1, 0, 1, 0), "pos": [6999, 50, 0], "vel": [0, 7.5, 0]},
            ]
        )
        gen.export_for_mkspk()
        
        setup_file = temp_output_dir / "SAT-001_setup.txt"
        content = setup_file.read_text()
        
        required_fields = [
            "INPUT_DATA_TYPE",
            "OUTPUT_SPK_TYPE",
            "OBJECT_ID",
            "OBJECT_NAME",
            "CENTER_ID",
            "REF_FRAME_NAME",
            "DATA_ORDER",
        ]
        
        for field in required_fields:
            assert field in content, f"Missing required field: {field}"
    
    def test_export_creates_generate_script(self, temp_output_dir):
        """Test generate_all.sh script is created."""
        from tools import SPKGenerator
        
        gen = SPKGenerator(temp_output_dir)
        gen.add_satellite_trajectory(
            "SAT-001", -100001,
            [
                {"time": datetime(2025, 1, 1, 0, 0, 0), "pos": [7000, 0, 0], "vel": [0, 7.5, 0]},
                {"time": datetime(2025, 1, 1, 0, 1, 0), "pos": [6999, 50, 0], "vel": [0, 7.5, 0]},
            ]
        )
        gen.export_for_mkspk()
        
        script_file = temp_output_dir / "generate_all.sh"
        assert script_file.exists()
        
        # Check executable
        assert script_file.stat().st_mode & 0o111
    
    def test_export_creates_naif_mapping(self, temp_output_dir):
        """Test naif_ids.json file is created."""
        from tools import SPKGenerator
        
        gen = SPKGenerator(temp_output_dir)
        gen.add_satellite_trajectory(
            "SAT-001", -100001,
            [
                {"time": datetime(2025, 1, 1, 0, 0, 0), "pos": [7000, 0, 0], "vel": [0, 7.5, 0]},
                {"time": datetime(2025, 1, 1, 0, 1, 0), "pos": [6999, 50, 0], "vel": [0, 7.5, 0]},
            ]
        )
        gen.export_for_mkspk()
        
        mapping_file = temp_output_dir / "naif_ids.json"
        assert mapping_file.exists()
        
        with open(mapping_file) as f:
            mapping = json.load(f)
        
        assert "satellites" in mapping
        assert "SAT-001" in mapping["satellites"]
        assert mapping["satellites"]["SAT-001"]["naif_id"] == -100001
    
    def test_export_multiple_satellites(self, temp_output_dir):
        """Test exporting multiple satellites."""
        from tools import SPKGenerator
        
        gen = SPKGenerator(temp_output_dir)
        
        for i in range(3):
            gen.add_satellite_trajectory(
                f"SAT-{i+1:03d}",
                states=[
                    {"time": datetime(2025, 1, 1, 0, 0, 0), "pos": [7000 + i*100, 0, 0], "vel": [0, 7.5, 0]},
                    {"time": datetime(2025, 1, 1, 0, 1, 0), "pos": [6999 + i*100, 50, 0], "vel": [0, 7.5, 0]},
                ]
            )
        
        gen.export_for_mkspk()
        
        # Check all files created
        for i in range(3):
            sat_id = f"SAT-{i+1:03d}"
            assert (temp_output_dir / f"{sat_id}_states.txt").exists()
            assert (temp_output_dir / f"{sat_id}_setup.txt").exists()


class TestSPKExportFromSimulation:
    """Tests for exporting simulation to SPK."""
    
    def test_add_from_simulation(self, small_simulation, temp_output_dir):
        """Test adding trajectories from simulation."""
        from tools import SPKGenerator
        
        gen = SPKGenerator(temp_output_dir)
        
        segments = gen.add_from_simulation(
            small_simulation,
            duration_hours=0.5,  # 30 minutes
            step_seconds=60.0
        )
        
        assert len(segments) == small_simulation.num_satellites
        for segment in segments:
            assert len(segment.states) == 31  # 30 minutes / 60s + initial
    
    def test_export_from_simulation(self, small_simulation, temp_output_dir):
        """Test full export from simulation."""
        from tools import SPKGenerator
        
        gen = SPKGenerator(temp_output_dir)
        gen.add_from_simulation(
            small_simulation,
            duration_hours=0.5,
            step_seconds=60.0
        )
        gen.export_for_mkspk()
        
        # Should have files for all satellites
        for sat in small_simulation.satellites:
            assert (temp_output_dir / f"{sat.satellite_id}_states.txt").exists()
    
    def test_create_spk_convenience_function(self, small_simulation, temp_output_dir):
        """Test create_spk_from_simulation convenience function."""
        from tools import create_spk_from_simulation
        
        output = create_spk_from_simulation(
            small_simulation,
            temp_output_dir,
            duration_hours=0.5,
            step_seconds=60.0
        )
        
        assert output == temp_output_dir
        assert (temp_output_dir / "generate_all.sh").exists()
        assert (temp_output_dir / "naif_ids.json").exists()


class TestSPKExportConstellationTypes:
    """Tests for exporting different constellation types."""
    
    def test_walker_delta_constellation(self, temp_output_dir):
        """Test SPK export with Walker-Delta constellation."""
        from simulation import Simulation, SimulationConfig, ConstellationType
        from tools import SPKGenerator
        
        config = SimulationConfig(
            constellation_type=ConstellationType.WALKER_DELTA,
            num_planes=2,
            sats_per_plane=2,
            random_seed=42,
        )
        sim = Simulation(config)
        sim.initialize()
        
        gen = SPKGenerator(temp_output_dir)
        segments = gen.add_from_simulation(sim, duration_hours=0.5, step_seconds=60.0)
        gen.export_for_mkspk()
        
        assert len(segments) == 4  # 2 planes * 2 sats
    
    def test_walker_star_constellation(self, temp_output_dir):
        """Test SPK export with Walker-Star constellation."""
        from simulation import Simulation, SimulationConfig, ConstellationType
        from tools import SPKGenerator
        
        config = SimulationConfig(
            constellation_type=ConstellationType.WALKER_STAR,
            num_planes=2,
            sats_per_plane=2,
            random_seed=42,
        )
        sim = Simulation(config)
        sim.initialize()
        
        gen = SPKGenerator(temp_output_dir)
        segments = gen.add_from_simulation(sim, duration_hours=0.5, step_seconds=60.0)
        gen.export_for_mkspk()
        
        assert len(segments) == 4
    
    def test_random_constellation(self, temp_output_dir):
        """Test SPK export with random constellation."""
        from simulation import Simulation, SimulationConfig, ConstellationType
        from tools import SPKGenerator
        
        config = SimulationConfig(
            constellation_type=ConstellationType.RANDOM,
            num_satellites=5,
            random_seed=42,
        )
        sim = Simulation(config)
        sim.initialize()
        
        gen = SPKGenerator(temp_output_dir)
        segments = gen.add_from_simulation(sim, duration_hours=0.5, step_seconds=60.0)
        gen.export_for_mkspk()
        
        assert len(segments) == 5


class TestSPKExportConfiguration:
    """Tests for export configuration options."""
    
    def test_custom_duration(self, small_simulation, temp_output_dir):
        """Test custom duration export."""
        from tools import SPKGenerator
        
        gen = SPKGenerator(temp_output_dir)
        segments = gen.add_from_simulation(
            small_simulation,
            duration_hours=1.0,  # 1 hour
            step_seconds=120.0  # 2 minute steps
        )
        
        # 1 hour / 2 minutes = 30 steps + 1 initial
        assert all(len(s.states) == 31 for s in segments)
    
    def test_custom_step_size(self, small_simulation, temp_output_dir):
        """Test custom step size export."""
        from tools import SPKGenerator
        
        gen = SPKGenerator(temp_output_dir)
        segments = gen.add_from_simulation(
            small_simulation,
            duration_hours=0.5,  # 30 minutes
            step_seconds=30.0   # 30 second steps
        )
        
        # 30 minutes / 30 seconds = 60 steps + 1 initial
        assert all(len(s.states) == 61 for s in segments)


class TestSPKCommandLineExport:
    """Tests for command-line export arguments."""
    
    def test_export_argument_parsing(self, temp_output_dir):
        """Test command-line argument parsing pattern."""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--output", type=Path)
        parser.add_argument("--duration", type=float, default=24.0)
        parser.add_argument("--step", type=float, default=60.0)
        
        args = parser.parse_args([
            "--output", str(temp_output_dir),
            "--duration", "12.0",
            "--step", "30.0"
        ])
        
        assert args.output == temp_output_dir
        assert args.duration == 12.0
        assert args.step == 30.0
    
    def test_default_export_parameters(self):
        """Test default export parameters."""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--duration", type=float, default=24.0)
        parser.add_argument("--step", type=float, default=60.0)
        
        args = parser.parse_args([])
        
        assert args.duration == 24.0  # 24 hours
        assert args.step == 60.0      # 60 seconds


class TestBackwardCompatibilityStep3:
    """Ensure Step 3 changes don't break existing functionality."""
    
    def test_simulation_unchanged(self, simulation_config):
        """Test Simulation API is unchanged."""
        from simulation import Simulation
        
        sim = Simulation(simulation_config)
        
        # All existing methods exist
        assert hasattr(sim, 'initialize')
        assert hasattr(sim, 'step')
        assert hasattr(sim, 'satellites')
    
    def test_satellite_methods_unchanged(self, small_simulation):
        """Test Satellite methods used by SPK export."""
        for sat in small_simulation.satellites:
            # Methods used by SPK export
            assert hasattr(sat, 'satellite_id')
            assert hasattr(sat, 'get_position_eci')
            assert hasattr(sat, 'get_velocity_eci')
            
            pos = sat.get_position_eci()
            vel = sat.get_velocity_eci()
            
            assert pos.shape == (3,)
            assert vel.shape == (3,)
    
    def test_trajectory_provider_unchanged(self, sample_satellites):
        """Test TrajectoryProvider from Step 1 unchanged."""
        from simulation import KeplerianProvider
        
        epoch = datetime(2025, 1, 1, 0, 0, 0)
        provider = KeplerianProvider(sample_satellites, epoch)
        
        assert len(provider.get_satellite_ids()) == 3
    
    def test_spice_provider_unchanged(self):
        """Test SpiceProvider from Step 2 unchanged."""
        from simulation import SpiceProvider, SpiceKernelSet, is_spice_available
        
        # All exports still available
        assert SpiceProvider is not None
        assert SpiceKernelSet is not None
        assert callable(is_spice_available)