#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Renderer Module for Satellite Visualization

Provides rendering functions for drawing Earth, satellites, orbits,
and other visualization elements using Pygame.
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
import pygame

from .camera import Camera


# Default color scheme
class Colors:
    """Default color palette for visualization."""
    BACKGROUND = (10, 10, 25)
    SPHERE = (40, 90, 160)
    SPHERE_HIGHLIGHT = (80, 140, 200)
    GRID_LINE = (20, 50, 100)
    GRID_LINE_BACK = (15, 35, 70)
    TEXT = (220, 220, 220)
    TEXT_DIM = (200, 200, 200)
    
    # Orbit colors - now light gray for all orbits
    ORBIT_FRONT = (180, 180, 180)
    ORBIT_BACK = (100, 100, 100)
    
    # Communication link colors
    LINK_COLOR = (50, 255, 100)        # Bright green for visible links
    LINK_COLOR_BACK = (30, 150, 60)    # Dimmer green for links behind Earth
    
    # Base station colors
    BASE_STATION_COLOR = (50, 255, 100)       # Bright green
    BASE_STATION_COLOR_BACK = (30, 150, 60)   # Dimmer green when behind Earth
    BASE_STATION_LINK_COLOR = (50, 255, 100)  # Green for base station links
    BASE_STATION_LINK_COLOR_BACK = (30, 150, 60)
    
    # Satellite packet status colors
    SATELLITE_NO_PACKETS = (255, 50, 50)      # Red - 0% completion
    SATELLITE_HALF_PACKETS = (255, 255, 50)   # Yellow - 50% completion
    SATELLITE_ALL_PACKETS = (50, 255, 50)     # Green - 100% completion


def interpolate_color(
    color1: Tuple[int, int, int],
    color2: Tuple[int, int, int],
    t: float
) -> Tuple[int, int, int]:
    """
    Linearly interpolate between two colors.
    
    Parameters
    ----------
    color1 : Tuple[int, int, int]
        Start color (RGB)
    color2 : Tuple[int, int, int]
        End color (RGB)
    t : float
        Interpolation factor (0.0 = color1, 1.0 = color2)
    
    Returns
    -------
    Tuple[int, int, int]
        Interpolated color
    """
    t = max(0.0, min(1.0, t))
    return (
        int(color1[0] + (color2[0] - color1[0]) * t),
        int(color1[1] + (color2[1] - color1[1]) * t),
        int(color1[2] + (color2[2] - color1[2]) * t)
    )


def get_packet_completion_color(completion_percentage: float) -> Tuple[int, int, int]:
    """
    Get satellite color based on packet completion percentage.
    
    Red (0%) -> Yellow (50%) -> Green (100%)
    
    Parameters
    ----------
    completion_percentage : float
        Percentage of packets received (0.0 to 100.0)
    
    Returns
    -------
    Tuple[int, int, int]
        RGB color tuple
    """
    # Normalize to 0-1 range
    t = completion_percentage / 100.0
    t = max(0.0, min(1.0, t))
    
    if t <= 0.5:
        # Red to Yellow (0% to 50%)
        return interpolate_color(
            Colors.SATELLITE_NO_PACKETS,
            Colors.SATELLITE_HALF_PACKETS,
            t * 2.0  # Scale 0-0.5 to 0-1
        )
    else:
        # Yellow to Green (50% to 100%)
        return interpolate_color(
            Colors.SATELLITE_HALF_PACKETS,
            Colors.SATELLITE_ALL_PACKETS,
            (t - 0.5) * 2.0  # Scale 0.5-1 to 0-1
        )


# Import EARTH_RADIUS_KM - handle both direct and package imports
try:
    from ..simulation import EARTH_RADIUS_KM
except ImportError:
    try:
        from simulation import EARTH_RADIUS_KM
    except ImportError:
        EARTH_RADIUS_KM = 6371.0


class Renderer:
    """
    Handles all rendering operations for the satellite visualizer.
    
    Parameters
    ----------
    screen : pygame.Surface
        The Pygame surface to render to
    earth_visual_radius : float
        Visual radius of Earth in rendering units (default 1.0)
    num_latitude_lines : int
        Number of latitude grid lines (default 8)
    num_longitude_lines : int
        Number of longitude grid lines (default 12)
    orbit_points : int
        Number of points to sample when drawing orbits (default 360)
    satellite_size : int
        Base size of satellite markers in pixels (default 12)
    """
    
    def __init__(
        self,
        screen: pygame.Surface,
        earth_visual_radius: float = 1.0,
        num_latitude_lines: int = 8,
        num_longitude_lines: int = 12,
        orbit_points: int = 360,
        satellite_size: int = 12
    ):
        self.screen = screen
        self.screen_width = screen.get_width()
        self.screen_height = screen.get_height()
        
        self.earth_visual_radius = earth_visual_radius
        self.num_latitude_lines = num_latitude_lines
        self.num_longitude_lines = num_longitude_lines
        self.orbit_points = orbit_points
        self.satellite_size = satellite_size
        
        # Scale factor for converting km to visual units
        self.scale_factor = earth_visual_radius / EARTH_RADIUS_KM
    
    def clear(self, color: Tuple[int, int, int] = Colors.BACKGROUND) -> None:
        """Clear the screen with the given background color."""
        self.screen.fill(color)
    
    def project_point(
        self,
        point_3d: np.ndarray,
        camera: Camera
    ) -> Tuple[Optional[Tuple[float, float]], float]:
        """
        Project a 3D point to 2D screen coordinates.
        
        Parameters
        ----------
        point_3d : np.ndarray
            3D point [x, y, z] in visualization coordinates
        camera : Camera
            The camera for projection
        
        Returns
        -------
        Tuple[Optional[Tuple[float, float]], float]
            ((screen_x, screen_y), depth) or (None, depth) if behind camera
        """
        cam_pos = camera.get_position()
        forward, up, right = camera.get_view_matrix()
        
        to_point = point_3d - cam_pos
        depth = np.dot(to_point, forward)
        
        if depth <= 0.1:
            return None, depth
        
        fov_scale = self.screen_height / 2
        x_proj = np.dot(to_point, right) / depth * fov_scale
        y_proj = -np.dot(to_point, up) / depth * fov_scale
        
        screen_x = self.screen_width / 2 + x_proj
        screen_y = self.screen_height / 2 + y_proj
        
        return (screen_x, screen_y), depth
    
    def get_projected_sphere_radius(self, camera: Camera) -> float:
        """Calculate the apparent radius of Earth on screen."""
        fov_scale = self.screen_height / 2
        return self.earth_visual_radius / camera.distance * fov_scale
    
    def is_point_visible_on_sphere(
        self,
        point_3d: np.ndarray,
        camera: Camera
    ) -> bool:
        """Check if a point on Earth's surface is facing the camera."""
        cam_pos = camera.get_position()
        to_camera = cam_pos - point_3d
        normal = point_3d / np.linalg.norm(point_3d)
        return np.dot(normal, to_camera) > 0
    
    def is_point_in_front_of_earth(
        self,
        point_3d: np.ndarray,
        camera: Camera
    ) -> bool:
        """
        Check if a point in space is visible (not occluded by Earth).
        
        Parameters
        ----------
        point_3d : np.ndarray
            3D point in visualization coordinates
        camera : Camera
            The camera
        
        Returns
        -------
        bool
            True if the point is visible (in front of Earth from camera)
        """
        cam_pos = camera.get_position()
        point_dist = np.linalg.norm(point_3d)
        
        # If point is inside Earth, it's behind
        if point_dist < self.earth_visual_radius:
            return False
        
        to_point = point_3d - cam_pos
        to_point_dist = np.linalg.norm(to_point)
        to_point_normalized = to_point / to_point_dist
        
        # Find closest approach to Earth center along the ray
        closest_approach_t = -np.dot(cam_pos, to_point_normalized)
        
        if closest_approach_t > to_point_dist:
            return True
        if closest_approach_t < 0:
            return True
        
        closest_point = cam_pos + closest_approach_t * to_point_normalized
        closest_dist_to_center = np.linalg.norm(closest_point)
        
        if closest_dist_to_center > self.earth_visual_radius:
            return True
        
        # Ray intersects Earth - check if point is before intersection
        half_chord = math.sqrt(self.earth_visual_radius**2 - closest_dist_to_center**2)
        entry_t = closest_approach_t - half_chord
        
        return to_point_dist < entry_t
    
    def draw_earth(self, camera: Camera) -> None:
        """Draw the semi-transparent Earth sphere with gradient."""
        center_3d = np.array([0, 0, 0])
        center_2d, _ = self.project_point(center_3d, camera)
        
        if center_2d is None:
            return
        
        radius = self.get_projected_sphere_radius(camera)
        
        # Create surface with alpha for transparency
        sphere_surface = pygame.Surface(
            (int(radius * 2) + 4, int(radius * 2) + 4),
            pygame.SRCALPHA
        )
        center_on_surface = (int(radius) + 2, int(radius) + 2)
        
        # Draw sphere with gradient and transparency
        for i in range(int(radius), 0, -2):
            factor = i / radius
            alpha = int(200 * factor + 55)
            r = int(Colors.SPHERE[0] + 
                   (Colors.SPHERE_HIGHLIGHT[0] - Colors.SPHERE[0]) * (1 - factor))
            g = int(Colors.SPHERE[1] + 
                   (Colors.SPHERE_HIGHLIGHT[1] - Colors.SPHERE[1]) * (1 - factor))
            b = int(Colors.SPHERE[2] + 
                   (Colors.SPHERE_HIGHLIGHT[2] - Colors.SPHERE[2]) * (1 - factor))
            color = (min(255, r), min(255, g), min(255, b), alpha)
            pygame.draw.circle(sphere_surface, color, center_on_surface, int(i))
        
        self.screen.blit(
            sphere_surface,
            (int(center_2d[0] - radius - 2), int(center_2d[1] - radius - 2))
        )
    
    def _generate_circle_points(
        self,
        center: np.ndarray,
        axis: np.ndarray,
        radius: float,
        num_points: int = 72
    ) -> List[np.ndarray]:
        """Generate points along a circle in 3D space."""
        axis = axis / np.linalg.norm(axis)
        
        if abs(axis[0]) < 0.9:
            perp1 = np.cross(axis, np.array([1, 0, 0]))
        else:
            perp1 = np.cross(axis, np.array([0, 1, 0]))
        perp1 = perp1 / np.linalg.norm(perp1)
        
        perp2 = np.cross(axis, perp1)
        perp2 = perp2 / np.linalg.norm(perp2)
        
        points = []
        for i in range(num_points + 1):
            angle = 2 * math.pi * i / num_points
            point = center + radius * (math.cos(angle) * perp1 + math.sin(angle) * perp2)
            points.append(point)
        
        return points
    
    def _draw_line_segments(
        self,
        points: List[Tuple[float, float]],
        color: Tuple[int, int, int],
        width: int
    ) -> None:
        """Draw connected line segments, handling discontinuities."""
        if len(points) < 2:
            return
        
        segments = []
        current_segment = [points[0]]
        
        for i in range(1, len(points)):
            prev = points[i-1]
            curr = points[i]
            dist = math.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
            
            if dist > 50:  # Discontinuity threshold
                if len(current_segment) > 1:
                    segments.append(current_segment)
                current_segment = [curr]
            else:
                current_segment.append(curr)
        
        if len(current_segment) > 1:
            segments.append(current_segment)
        
        for segment in segments:
            if len(segment) > 1:
                pygame.draw.lines(
                    self.screen, color, False,
                    [(int(p[0]), int(p[1])) for p in segment], width
                )
    
    def draw_earth_grid(self, camera: Camera) -> None:
        """Draw latitude and longitude grid lines on Earth."""
        # Draw back lines first
        for i in range(1, self.num_latitude_lines):
            lat = -math.pi/2 + math.pi * i / self.num_latitude_lines
            self._draw_latitude_line(camera, lat, is_back=True)
        
        for i in range(self.num_longitude_lines):
            lon = 2 * math.pi * i / self.num_longitude_lines
            self._draw_longitude_line(camera, lon, is_back=True)
        
        # Draw front lines
        for i in range(1, self.num_latitude_lines):
            lat = -math.pi/2 + math.pi * i / self.num_latitude_lines
            self._draw_latitude_line(camera, lat, is_back=False)
        
        for i in range(self.num_longitude_lines):
            lon = 2 * math.pi * i / self.num_longitude_lines
            self._draw_longitude_line(camera, lon, is_back=False)
    
    def _draw_latitude_line(
        self,
        camera: Camera,
        latitude_angle: float,
        is_back: bool = False
    ) -> None:
        """Draw a latitude line (parallel to equator)."""
        z = self.earth_visual_radius * math.sin(latitude_angle)
        circle_radius = self.earth_visual_radius * math.cos(latitude_angle)
        center = np.array([0, 0, z])
        axis = np.array([0, 0, 1])
        
        points = self._generate_circle_points(center, axis, circle_radius, num_points=72)
        
        screen_points_front = []
        screen_points_back = []
        
        for point in points:
            proj, _ = self.project_point(point, camera)
            if proj is not None:
                visible = self.is_point_visible_on_sphere(point, camera)
                if visible:
                    screen_points_front.append(proj)
                else:
                    screen_points_back.append(proj)
        
        if is_back and len(screen_points_back) > 1:
            self._draw_line_segments(screen_points_back, Colors.GRID_LINE_BACK, 1)
        
        if not is_back and len(screen_points_front) > 1:
            base_width = max(1, int(3 / camera.distance * 2))
            self._draw_line_segments(screen_points_front, Colors.GRID_LINE, base_width)
    
    def _draw_longitude_line(
        self,
        camera: Camera,
        longitude_angle: float,
        is_back: bool = False
    ) -> None:
        """Draw a longitude line (meridian)."""
        axis = np.array([
            math.cos(longitude_angle + math.pi/2),
            math.sin(longitude_angle + math.pi/2),
            0
        ])
        center = np.array([0, 0, 0])
        
        points = self._generate_circle_points(
            center, axis, self.earth_visual_radius, num_points=72
        )
        
        screen_points_front = []
        screen_points_back = []
        
        for point in points:
            proj, _ = self.project_point(point, camera)
            if proj is not None:
                visible = self.is_point_visible_on_sphere(point, camera)
                if visible:
                    screen_points_front.append(proj)
                else:
                    screen_points_back.append(proj)
        
        if is_back and len(screen_points_back) > 1:
            self._draw_line_segments(screen_points_back, Colors.GRID_LINE_BACK, 1)
        
        if not is_back and len(screen_points_front) > 1:
            base_width = max(1, int(3 / camera.distance * 2))
            self._draw_line_segments(screen_points_front, Colors.GRID_LINE, base_width)
    
    def draw_orbit(
        self,
        camera: Camera,
        orbit,  # EllipticalOrbit
        color_front: Tuple[int, int, int] = None,
        color_back: Tuple[int, int, int] = None
    ) -> None:
        """
        Draw an elliptical orbit around Earth.
        
        Parameters
        ----------
        camera : Camera
            The camera for projection
        orbit : EllipticalOrbit
            The orbit to draw
        color_front : Tuple[int, int, int], optional
            Color for visible portions (default: light gray)
        color_back : Tuple[int, int, int], optional
            Color for occluded portions (default: darker gray)
        """
        if color_front is None:
            color_front = Colors.ORBIT_FRONT
        if color_back is None:
            color_back = Colors.ORBIT_BACK
            
        screen_points_front = []
        screen_points_back = []
        
        for i in range(self.orbit_points + 1):
            nu = 2 * math.pi * i / self.orbit_points
            pos_km = orbit.position_eci(nu)
            pos_visual = pos_km * self.scale_factor
            
            proj, _ = self.project_point(pos_visual, camera)
            if proj is not None:
                in_front = self.is_point_in_front_of_earth(pos_visual, camera)
                if in_front:
                    screen_points_front.append(proj)
                else:
                    screen_points_back.append(proj)
        
        # Draw back portions first
        if len(screen_points_back) > 1:
            self._draw_line_segments(screen_points_back, color_back, 1)
        
        # Draw front portions
        if len(screen_points_front) > 1:
            line_width = max(2, int(4 / camera.distance * 2))
            self._draw_line_segments(screen_points_front, color_front, line_width)
    
    def draw_orbits(self, camera: Camera, orbits: List) -> None:
        """
        Draw all unique orbits in light gray.
        
        If multiple satellites share the same orbit, only draw it once.
        
        Parameters
        ----------
        camera : Camera
            The camera for projection
        orbits : List[EllipticalOrbit]
            List of orbits to draw
        """
        # Track which orbits we've already drawn using their key parameters
        drawn_orbits: Set[Tuple] = set()
        
        for orbit in orbits:
            # Create a hashable key from orbit parameters
            # Using rounded values to handle floating point comparison
            orbit_key = (
                round(orbit.semi_major_axis, 2),
                round(orbit.eccentricity, 6),
                round(orbit.inclination, 6),
                round(orbit.longitude_of_ascending_node, 6),
                round(orbit.argument_of_periapsis, 6)
            )
            
            if orbit_key not in drawn_orbits:
                drawn_orbits.add(orbit_key)
                self.draw_orbit(
                    camera, orbit,
                    color_front=Colors.ORBIT_FRONT,
                    color_back=Colors.ORBIT_BACK
                )
    
    def draw_satellite(
        self,
        camera: Camera,
        satellite,  # Satellite
        color: Tuple[int, int, int],
        size: Optional[int] = None
    ) -> None:
        """Draw a single satellite as a triangle."""
        if size is None:
            size = self.satellite_size
        
        pos_km = satellite.get_position_eci()
        pos_visual = pos_km * self.scale_factor
        
        in_front = self.is_point_in_front_of_earth(pos_visual, camera)
        proj, depth = self.project_point(pos_visual, camera)
        
        if proj is None:
            return
        
        # Perspective-adjusted size
        perspective_size = size * (3.0 / depth) if depth > 0 else size
        perspective_size = max(4, min(20, perspective_size))
        
        # Get velocity for orientation
        vel_km = satellite.get_velocity_eci()
        vel_visual = vel_km * self.scale_factor
        
        pos_future = pos_visual + vel_visual * 100
        proj_future, _ = self.project_point(pos_future, camera)
        
        if proj_future is not None:
            direction = (proj_future[0] - proj[0], proj_future[1] - proj[1])
        else:
            direction = None
        
        # Generate triangle points
        triangle_points = self._get_satellite_triangle_points(
            proj, perspective_size, direction
        )
        
        # Determine colors based on visibility
        if in_front:
            draw_color = color
            outline_color = (255, 255, 255)
            outline_width = 2
        else:
            draw_color = tuple(int(c * 0.4) for c in color)
            outline_color = tuple(int(c * 0.5) for c in color)
            outline_width = 1
        
        # Draw triangle
        int_points = [(int(p[0]), int(p[1])) for p in triangle_points]
        pygame.draw.polygon(self.screen, draw_color, int_points)
        pygame.draw.polygon(self.screen, outline_color, int_points, outline_width)
    
    def _get_satellite_triangle_points(
        self,
        center_2d: Tuple[float, float],
        size: float,
        direction: Optional[Tuple[float, float]] = None
    ) -> List[Tuple[float, float]]:
        """Generate triangle points for a satellite marker."""
        if direction is not None and (direction[0] != 0 or direction[1] != 0):
            dx, dy = direction
            mag = math.sqrt(dx*dx + dy*dy)
            dx, dy = dx/mag, dy/mag
            px, py = -dy, dx
        else:
            dx, dy = 0, -1
            px, py = 1, 0
        
        cx, cy = center_2d
        
        tip = (cx + dx * size, cy + dy * size)
        left = (cx - dx * size * 0.5 + px * size * 0.6,
                cy - dy * size * 0.5 + py * size * 0.6)
        right = (cx - dx * size * 0.5 - px * size * 0.6,
                 cy - dy * size * 0.5 - py * size * 0.6)
        
        return [tip, left, right]
    
    def draw_satellites(
        self,
        camera: Camera,
        satellites: List,
        completion_percentages: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Draw all satellites colored by their packet completion status.
        
        Parameters
        ----------
        camera : Camera
            The camera for projection
        satellites : List[Satellite]
            List of satellites to draw
        completion_percentages : Optional[Dict[str, float]]
            Dictionary mapping satellite IDs to completion percentages (0-100).
            If None, all satellites are drawn red (0% completion).
        """
        for satellite in satellites:
            # Get completion percentage for this satellite
            if completion_percentages is not None:
                completion = completion_percentages.get(satellite.satellite_id, 0.0)
            else:
                completion = 0.0
            
            # Get color based on completion
            color = get_packet_completion_color(completion)
            
            self.draw_satellite(camera, satellite, color)
    
    def draw_base_station(
        self,
        camera: Camera,
        base_station,  # BaseStation
        earth_rotation_angle: float,
        size: int = 10
    ) -> None:
        """
        Draw a base station as a green square on Earth's surface.
        
        Parameters
        ----------
        camera : Camera
            The current camera
        base_station : BaseStation
            The base station to draw
        earth_rotation_angle : float
            Current Earth rotation angle in radians
        size : int
            Size of the square in pixels
        """
        # Get position in ECI coordinates
        pos_km = base_station.get_position_eci(earth_rotation_angle)
        pos_visual = pos_km * self.scale_factor
        
        # Check if visible (on front of Earth)
        in_front = self.is_point_visible_on_sphere(pos_visual, camera)
        proj, depth = self.project_point(pos_visual, camera)
        
        if proj is None:
            return
        
        # Perspective-adjusted size
        perspective_size = size * (3.0 / depth) if depth > 0 else size
        perspective_size = max(4, min(16, perspective_size))
        
        # Choose color based on visibility
        if in_front:
            color = Colors.BASE_STATION_COLOR
            outline_color = (255, 255, 255)
            outline_width = 2
        else:
            color = Colors.BASE_STATION_COLOR_BACK
            outline_color = tuple(int(c * 0.5) for c in Colors.BASE_STATION_COLOR)
            outline_width = 1
        
        # Draw square centered at projection point
        half_size = perspective_size / 2
        rect = pygame.Rect(
            int(proj[0] - half_size),
            int(proj[1] - half_size),
            int(perspective_size),
            int(perspective_size)
        )
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, outline_color, rect, outline_width)
    
    def draw_base_stations(
        self,
        camera: Camera,
        base_stations: List,
        earth_rotation_angle: float
    ) -> None:
        """Draw all base stations."""
        for base_station in base_stations:
            self.draw_base_station(camera, base_station, earth_rotation_angle)
    
    def draw_base_station_links(
        self,
        camera: Camera,
        base_stations: List,
        satellites: List,
        base_station_links: set,  # Set[Tuple[str, str]]
        earth_rotation_angle: float
    ) -> None:
        """
        Draw communication links between base stations and satellites.
        
        Parameters
        ----------
        camera : Camera
            The current camera
        base_stations : List
            List of BaseStation objects
        satellites : List
            List of Satellite objects
        base_station_links : set
            Set of (base_station_name, satellite_id) tuples for active links
        earth_rotation_angle : float
            Current Earth rotation angle in radians
        """
        if not base_station_links:
            return
        
        # Build lookup dicts
        bs_by_name = {bs.name: bs for bs in base_stations}
        sat_by_id = {sat.satellite_id: sat for sat in satellites}
        
        for bs_name, sat_id in base_station_links:
            base_station = bs_by_name.get(bs_name)
            satellite = sat_by_id.get(sat_id)
            
            if base_station is None or satellite is None:
                continue
            
            # Get 3D positions in visual coordinates
            pos_bs_km = base_station.get_position_eci(earth_rotation_angle)
            pos_sat_km = satellite.get_position_eci()
            pos_bs_visual = pos_bs_km * self.scale_factor
            pos_sat_visual = pos_sat_km * self.scale_factor
            
            # Project to screen coordinates
            proj_bs, depth_bs = self.project_point(pos_bs_visual, camera)
            proj_sat, depth_sat = self.project_point(pos_sat_visual, camera)
            
            if proj_bs is None or proj_sat is None:
                continue
            
            # Determine if link is visible
            # Base station is visible if on front of Earth
            bs_visible = self.is_point_visible_on_sphere(pos_bs_visual, camera)
            sat_visible = self.is_point_in_front_of_earth(pos_sat_visual, camera)
            
            if bs_visible and sat_visible:
                color = Colors.BASE_STATION_LINK_COLOR
                width = max(1, int(2 / camera.distance * 2))
            else:
                color = Colors.BASE_STATION_LINK_COLOR_BACK
                width = 1
            
            # Draw the line
            pygame.draw.line(
                self.screen,
                color,
                (int(proj_bs[0]), int(proj_bs[1])),
                (int(proj_sat[0]), int(proj_sat[1])),
                width
            )
    
    def draw_communication_links(
        self,
        camera: Camera,
        satellites: List,
        active_links: set  # Set[Tuple[str, str]]
    ) -> None:
        """
        Draw communication links between satellites.
        
        Parameters
        ----------
        camera : Camera
            The current camera
        satellites : List
            List of Satellite objects
        active_links : set
            Set of (sat1_id, sat2_id) tuples for active links
        """
        if not active_links:
            return
        
        # Build a lookup dict for satellites by ID
        sat_by_id = {sat.satellite_id: sat for sat in satellites}
        
        for sat1_id, sat2_id in active_links:
            sat1 = sat_by_id.get(sat1_id)
            sat2 = sat_by_id.get(sat2_id)
            
            if sat1 is None or sat2 is None:
                continue
            
            # Get 3D positions in visual coordinates
            pos1_km = sat1.get_position_eci()
            pos2_km = sat2.get_position_eci()
            pos1_visual = pos1_km * self.scale_factor
            pos2_visual = pos2_km * self.scale_factor
            
            # Project to screen coordinates
            proj1, depth1 = self.project_point(pos1_visual, camera)
            proj2, depth2 = self.project_point(pos2_visual, camera)
            
            if proj1 is None or proj2 is None:
                continue
            
            # Determine if link is in front or behind Earth
            # Link is "in front" if both satellites are in front of Earth
            in_front1 = self.is_point_in_front_of_earth(pos1_visual, camera)
            in_front2 = self.is_point_in_front_of_earth(pos2_visual, camera)
            
            if in_front1 and in_front2:
                # Both satellites visible - draw bright link
                color = Colors.LINK_COLOR
                width = max(1, int(2 / camera.distance * 2))
            else:
                # At least one satellite behind Earth - draw dim link
                color = Colors.LINK_COLOR_BACK
                width = 1
            
            # Draw the line
            pygame.draw.line(
                self.screen,
                color,
                (int(proj1[0]), int(proj1[1])),
                (int(proj2[0]), int(proj2[1])),
                width
            )
    
    def draw_text(
        self,
        text: str,
        position: Tuple[int, int],
        font: pygame.font.Font,
        color: Tuple[int, int, int] = Colors.TEXT
    ) -> int:
        """
        Draw text at the given position.
        
        Returns the height of the rendered text.
        """
        surface = font.render(text, True, color)
        self.screen.blit(surface, position)
        return surface.get_height()
    
    def draw_info_panel(
        self,
        camera: Camera,
        simulation,  # Simulation
        font: pygame.font.Font,
        time_scale: float,
        paused: bool
    ) -> None:
        """Draw the information panel with camera and simulation info."""
        # Format simulation time
        sim_time = simulation.state.time
        hours = int(sim_time // 3600)
        minutes = int((sim_time % 3600) // 60)
        seconds = int(sim_time % 60)
        
        # Count total possible links
        n = len(simulation.satellites)
        total_pairs = n * (n - 1) // 2
        active_count = len(simulation.state.active_links)
        bs_link_count = len(simulation.state.base_station_links)
        
        # Get agent statistics
        stats = simulation.state.agent_statistics
        
        info_lines = [
            f"Camera Longitude: {camera.theta_degrees:.1f}°",
            f"Camera Latitude: {camera.phi_degrees:.1f}°",
            f"Zoom: {camera.distance:.2f}",
            "",
            f"Sim Time: {hours:02d}:{minutes:02d}:{seconds:02d}",
            f"Time Scale: {time_scale:.0f}x" + (" [PAUSED]" if paused else ""),
            f"Active Links: {active_count}/{total_pairs}",
            f"Base Station Links: {bs_link_count}",
            "",
            f"Packets: {stats.total_packets}",
            f"Avg Completion: {stats.average_completion:.1f}%",
            f"Fully Updated: {stats.fully_updated_count}/{n}",
            "",
            "Controls:",
            "← → : Rotate longitude",
            "↑ ↓ : Rotate latitude",
            "+/- : Zoom in/out",
            "[ ] : Time scale",
            "SPACE : Pause/Resume",
            "R : Regenerate",
            "ESC : Quit",
        ]
        
        y = 10
        for line in info_lines:
            y += self.draw_text(line, (10, y), font) + 2
        
        # Draw color legend
        y += 10
        self.draw_text("Satellite Colors:", (10, y), font)
        y += 20
        
        # Red square - 0%
        pygame.draw.rect(self.screen, Colors.SATELLITE_NO_PACKETS, (10, y, 15, 15))
        self.draw_text("0% packets", (30, y), font, Colors.TEXT_DIM)
        y += 20
        
        # Yellow square - 50%
        pygame.draw.rect(self.screen, Colors.SATELLITE_HALF_PACKETS, (10, y, 15, 15))
        self.draw_text("50% packets", (30, y), font, Colors.TEXT_DIM)
        y += 20
        
        # Green square - 100%
        pygame.draw.rect(self.screen, Colors.SATELLITE_ALL_PACKETS, (10, y, 15, 15))
        self.draw_text("100% packets", (30, y), font, Colors.TEXT_DIM)
        
        # Satellite details on the right side
        y = 10
        for i, satellite in enumerate(simulation.satellites):
            geo = satellite.get_geospatial_position()
            orbit = satellite.orbit
            
            # Get agent completion for this satellite
            agent_id = simulation.satellite_id_to_agent_id.get(satellite.satellite_id)
            if agent_id is not None:
                completion = stats.completion_percentage.get(agent_id, 0.0)
            else:
                completion = 0.0
            
            sat_info = [
                f"Satellite {i+1}: {satellite.satellite_id}",
                f"  Alt: {geo.altitude:.0f} km",
                f"  Lat: {geo.latitude_deg:+.1f}°",
                f"  Lon: {geo.longitude_deg:+.1f}°",
                f"  Packets: {completion:.0f}%",
            ]
            
            # Color the satellite name based on completion
            sat_color = get_packet_completion_color(completion)
            
            for j, line in enumerate(sat_info):
                line_color = sat_color if j == 0 else Colors.TEXT_DIM
                y += self.draw_text(line, (self.screen_width - 200, y), font, line_color) + 1
            y += 8
            
            # Limit display to avoid overflow
            if y > self.screen_height - 100:
                self.draw_text(
                    f"... and {len(simulation.satellites) - i - 1} more",
                    (self.screen_width - 200, y), font, Colors.TEXT_DIM
                )
                break