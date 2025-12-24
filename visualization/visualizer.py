#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizer Module for Satellite Constellation Simulator

Provides a Pygame-based interactive visualization of satellite constellations.
The visualizer can create and step a simulation, displaying satellites and their
orbits in real-time.
"""

import sys
import math
from typing import Optional, Dict

import pygame

# Add parent directory to path for imports when running as script
if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulation import (
    Simulation,
    SimulationConfig,
    ConstellationType,
)
from .camera import Camera
from .renderer import Renderer, Colors


class Visualizer:
    """
    Interactive visualization of satellite constellation simulations.
    
    The visualizer creates a Pygame window and renders a 3D view of Earth
    with satellites and their orbits. Users can control the camera with
    keyboard inputs and control simulation playback.
    
    Parameters
    ----------
    width : int
        Window width in pixels (default 1000)
    height : int
        Window height in pixels (default 800)
    title : str
        Window title
    time_scale : float
        Initial simulation time scale (simulation seconds per real second)
    paused : bool
        Start paused (default False)
    
    Attributes
    ----------
    screen : pygame.Surface
        The Pygame display surface
    camera : Camera
        The 3D camera
    renderer : Renderer
        The rendering engine
    simulation : Simulation
        The satellite constellation simulation
    time_scale : float
        Current time scale
    paused : bool
        Whether simulation is paused
    running : bool
        Whether the visualizer is running
    """
    
    DEFAULT_TIME_SCALE = 60.0  # 1 minute per second
    MIN_TIME_SCALE = 1.0
    MAX_TIME_SCALE = 3600.0
    
    def __init__(
        self,
        width: int = 1000,
        height: int = 800,
        title: str = "Satellite Constellation Visualizer",
        time_scale: float = DEFAULT_TIME_SCALE,
        paused: bool = False
    ):
        # Initialize Pygame
        pygame.init()
        
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        
        # Create camera and renderer
        self.camera = Camera()
        self.renderer = Renderer(self.screen)
        
        # Simulation state
        self.simulation: Optional[Simulation] = None
        self.time_scale = time_scale
        self.paused = paused
        self.running = False
        
        # Pygame resources
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 16)
    
    def set_simulation(self, simulation: Simulation) -> None:
        """
        Set the simulation to visualize.
        
        Parameters
        ----------
        simulation : Simulation
            An initialized simulation
        """
        self.simulation = simulation
    
    def create_simulation(
        self,
        constellation_type: str = "walker_delta",
        **kwargs
    ) -> Simulation:
        """
        Create and set a new simulation.
        
        Parameters
        ----------
        constellation_type : str
            Type of constellation: "random", "walker_delta", "walker_star"
        **kwargs
            Additional simulation configuration parameters
        
        Returns
        -------
        Simulation
            The created simulation
        """
        type_map = {
            "random": ConstellationType.RANDOM,
            "walker_delta": ConstellationType.WALKER_DELTA,
            "walker_star": ConstellationType.WALKER_STAR,
        }
        
        if constellation_type not in type_map:
            raise ValueError(f"Unknown constellation type: {constellation_type}")
        
        config = SimulationConfig(constellation_type=type_map[constellation_type])
        
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        self.simulation = Simulation(config)
        self.simulation.initialize()
        
        return self.simulation
    
    def _handle_events(self) -> None:
        """Handle Pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            if event.type == pygame.KEYDOWN:
                self._handle_keydown(event.key)
    
    def _handle_keydown(self, key: int) -> None:
        """Handle key press events."""
        if key == pygame.K_ESCAPE:
            self.running = False
        
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
        
        elif key == pygame.K_r:
            # Regenerate constellation
            if self.simulation is not None:
                self.simulation.regenerate()
                print("Regenerated constellation")
        
        elif key == pygame.K_LEFTBRACKET:
            # Decrease time scale
            self.time_scale = max(self.MIN_TIME_SCALE, self.time_scale / 2)
            print(f"Time scale: {self.time_scale}x")
        
        elif key == pygame.K_RIGHTBRACKET:
            # Increase time scale
            self.time_scale = min(self.MAX_TIME_SCALE, self.time_scale * 2)
            print(f"Time scale: {self.time_scale}x")
    
    def _handle_continuous_keys(self) -> None:
        """Handle continuous key presses for camera control."""
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_LEFT]:
            self.camera.rotate_left()
        if keys[pygame.K_RIGHT]:
            self.camera.rotate_right()
        if keys[pygame.K_UP]:
            self.camera.rotate_up()
        if keys[pygame.K_DOWN]:
            self.camera.rotate_down()
        
        if keys[pygame.K_PLUS] or keys[pygame.K_EQUALS] or keys[pygame.K_KP_PLUS]:
            self.camera.zoom_in()
        if keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]:
            self.camera.zoom_out()
    
    def _update(self, dt: float) -> None:
        """
        Update simulation state.
        
        Parameters
        ----------
        dt : float
            Real time delta in seconds
        """
        if self.simulation is None:
            return
        
        if not self.paused:
            sim_dt = dt * self.time_scale
            self.simulation.step(sim_dt)
    
    def _get_satellite_completion_percentages(self) -> Dict[str, float]:
        """
        Get completion percentages for all satellites.
        
        Returns
        -------
        Dict[str, float]
            Dictionary mapping satellite IDs to completion percentages (0-100)
        """
        if self.simulation is None:
            return {}
        
        completion_percentages = {}
        stats = self.simulation.state.agent_statistics
        
        for satellite in self.simulation.satellites:
            agent_id = self.simulation.satellite_id_to_agent_id.get(satellite.satellite_id)
            if agent_id is not None:
                completion = stats.completion_percentage.get(agent_id, 0.0)
            else:
                completion = 0.0
            completion_percentages[satellite.satellite_id] = completion
        
        return completion_percentages
    
    def _render(self) -> None:
        """Render the current frame."""
        # Clear screen
        self.renderer.clear()
        
        if self.simulation is None:
            # Draw message if no simulation
            self.renderer.draw_text(
                "No simulation loaded. Call create_simulation() first.",
                (self.width // 2 - 200, self.height // 2),
                self.font
            )
            pygame.display.flip()
            return
        
        # Calculate Earth rotation angle
        earth_rotation = self.simulation.state.time * self.simulation.earth_rotation_rate
        
        # Draw Earth
        self.renderer.draw_earth(self.camera)
        
        # Draw Earth grid
        self.renderer.draw_earth_grid(self.camera)
        
        # Draw orbits (unique orbits only, in light gray)
        self.renderer.draw_orbits(self.camera, self.simulation.orbits)
        
        # Draw communication links (before satellites so they appear underneath)
        self.renderer.draw_communication_links(
            self.camera,
            self.simulation.satellites,
            self.simulation.state.active_links
        )
        
        # Draw base station links
        self.renderer.draw_base_station_links(
            self.camera,
            self.simulation.base_stations,
            self.simulation.satellites,
            self.simulation.state.base_station_links,
            earth_rotation
        )
        
        # Get satellite completion percentages for coloring
        completion_percentages = self._get_satellite_completion_percentages()
        
        # Draw satellites (colored by packet completion: red -> yellow -> green)
        self.renderer.draw_satellites(
            self.camera,
            self.simulation.satellites,
            completion_percentages
        )
        
        # Draw base stations
        self.renderer.draw_base_stations(
            self.camera,
            self.simulation.base_stations,
            earth_rotation
        )
        
        # Draw info panel
        self.renderer.draw_info_panel(
            self.camera,
            self.simulation,
            self.font,
            self.time_scale,
            self.paused
        )
        
        # Update display
        pygame.display.flip()
    
    def run(self) -> None:
        """
        Run the visualization main loop.
        
        This blocks until the user closes the window or presses ESC.
        """
        self.running = True
        
        while self.running:
            # Get time delta
            dt = self.clock.tick(60) / 1000.0  # Convert to seconds
            
            # Handle events
            self._handle_events()
            
            # Handle continuous key presses
            self._handle_continuous_keys()
            
            # Update simulation
            self._update(dt)
            
            # Render
            self._render()
        
        pygame.quit()
    
    def step(self) -> bool:
        """
        Perform a single visualization step.
        
        This is useful for external control of the visualization loop.
        
        Returns
        -------
        bool
            False if the visualizer should stop, True otherwise
        """
        dt = self.clock.tick(60) / 1000.0
        
        self._handle_events()
        
        if not self.running:
            return False
        
        self._handle_continuous_keys()
        self._update(dt)
        self._render()
        
        return True
    
    def close(self) -> None:
        """Close the visualizer and clean up resources."""
        pygame.quit()


def run_visualizer(
    constellation_type: str = "walker_delta",
    num_planes: int = 3,
    sats_per_plane: int = 4,
    altitude: float = 550,
    inclination_deg: float = 53,
    num_satellites: int = 5,  # For random constellations
    random_seed: Optional[int] = 42,
    num_packets: int = 100,  # Number of packets in the software update
    time_scale: float = 60.0,
    paused: bool = False,
    width: int = 1000,
    height: int = 800
) -> None:
    """
    Convenience function to launch the visualizer with a constellation.
    
    Parameters
    ----------
    constellation_type : str
        Type: "random", "walker_delta", "walker_star"
    num_planes : int
        Number of orbital planes (Walker constellations)
    sats_per_plane : int
        Satellites per plane (Walker constellations)
    altitude : float
        Orbital altitude in km
    inclination_deg : float
        Inclination in degrees
    num_satellites : int
        Total satellites (random constellations)
    random_seed : Optional[int]
        Random seed for reproducibility
    num_packets : int
        Number of packets in the software update
    time_scale : float
        Initial time scale
    paused : bool
        Start paused
    width : int
        Window width
    height : int
        Window height
    """
    visualizer = Visualizer(
        width=width,
        height=height,
        time_scale=time_scale,
        paused=paused
    )
    
    visualizer.create_simulation(
        constellation_type=constellation_type,
        num_planes=num_planes,
        sats_per_plane=sats_per_plane,
        num_satellites=num_satellites,
        altitude=altitude,
        inclination=math.radians(inclination_deg),
        random_seed=random_seed,
        num_packets=num_packets
    )
    
    print(f"Created {constellation_type} constellation:")
    print(f"  Satellites: {visualizer.simulation.num_satellites}")
    print(f"  Orbits: {visualizer.simulation.num_orbits}")
    print(f"  Base Stations: {len(visualizer.simulation.base_stations)}")
    print(f"  Packets: {num_packets}")
    print(f"\nSatellite colors: Red (0%) -> Yellow (50%) -> Green (100%)")
    
    visualizer.run()


if __name__ == "__main__":
    # Default demo: Walker-Delta constellation
    print("Starting Satellite Constellation Visualizer")
    print("=" * 50)
    print("\nControls:")
    print("  Arrow keys: Rotate camera")
    print("  +/- : Zoom in/out")
    print("  [ ] : Decrease/increase time scale")
    print("  SPACE : Pause/Resume")
    print("  R : Regenerate constellation")
    print("  ESC : Quit")
    print("\nSatellite colors indicate packet completion:")
    print("  Red = 0%, Yellow = 50%, Green = 100%")
    print()
    
    run_visualizer(
        constellation_type="walker_delta",
        num_planes=3,
        sats_per_plane=4,
        altitude=550,
        inclination_deg=53,
        random_seed=42,
        num_packets=100
    )