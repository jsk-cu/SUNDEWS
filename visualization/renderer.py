#!/usr/bin/env python3
"""
Renderer Module

Pygame-based rendering for satellite constellation visualization.
Draws Earth, satellites, orbits, communication links, and grid lines.
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
import pygame

from .camera import Camera


class Colors:
    """Default color palette."""

    BACKGROUND = (10, 10, 25)
    SPHERE = (40, 90, 160)
    SPHERE_HIGHLIGHT = (80, 140, 200)
    GRID_LINE = (60, 120, 180)
    GRID_LINE_BACK = (25, 55, 95)
    TEXT = (220, 220, 220)
    TEXT_DIM = (200, 200, 200)
    TEXT_SUCCESS = (50, 255, 100)

    # Orbit colors
    ORBIT_FRONT = (180, 180, 180)
    ORBIT_BACK = (100, 100, 100)

    # Communication links
    LINK_COLOR = (50, 255, 100)
    LINK_COLOR_BACK = (30, 150, 60)

    # Base station
    BASE_STATION_COLOR = (50, 255, 100)
    BASE_STATION_COLOR_BACK = (30, 150, 60)
    BASE_STATION_LINK_COLOR = (50, 255, 100)
    BASE_STATION_LINK_COLOR_BACK = (30, 150, 60)

    # Satellite status colors
    SATELLITE_NO_PACKETS = (255, 50, 50)
    SATELLITE_HALF_PACKETS = (255, 255, 50)
    SATELLITE_ALL_PACKETS = (50, 255, 50)


def interpolate_color(
    color1: Tuple[int, int, int], color2: Tuple[int, int, int], t: float
) -> Tuple[int, int, int]:
    """Linearly interpolate between two colors."""
    t = max(0.0, min(1.0, t))
    return (
        int(color1[0] + (color2[0] - color1[0]) * t),
        int(color1[1] + (color2[1] - color1[1]) * t),
        int(color1[2] + (color2[2] - color1[2]) * t),
    )


def get_packet_completion_color(completion_percentage: float) -> Tuple[int, int, int]:
    """Get satellite color based on packet completion."""
    t = max(0.0, min(1.0, completion_percentage / 100.0))

    if t <= 0.5:
        return interpolate_color(
            Colors.SATELLITE_NO_PACKETS, Colors.SATELLITE_HALF_PACKETS, t * 2.0
        )
    else:
        return interpolate_color(
            Colors.SATELLITE_HALF_PACKETS, Colors.SATELLITE_ALL_PACKETS, (t - 0.5) * 2.0
        )


# Import Earth radius
try:
    from ..simulation import EARTH_RADIUS_KM
except ImportError:
    try:
        from simulation import EARTH_RADIUS_KM
    except ImportError:
        EARTH_RADIUS_KM = 6371.0


class Renderer:
    """Handles all rendering operations."""

    def __init__(
        self,
        screen: pygame.Surface,
        earth_visual_radius: float = 1.0,
        num_latitude_lines: int = 8,
        num_longitude_lines: int = 12,
        orbit_points: int = 360,
        satellite_size: int = 12,
    ):
        self.screen = screen
        self.screen_width = screen.get_width()
        self.screen_height = screen.get_height()

        self.earth_visual_radius = earth_visual_radius
        self.num_latitude_lines = num_latitude_lines
        self.num_longitude_lines = num_longitude_lines
        self.orbit_points = orbit_points
        self.satellite_size = satellite_size

        self.scale_factor = earth_visual_radius / EARTH_RADIUS_KM

    def clear(self, color: Tuple[int, int, int] = Colors.BACKGROUND) -> None:
        self.screen.fill(color)

    def project_point(
        self, point_3d: np.ndarray, camera: Camera
    ) -> Tuple[Optional[Tuple[float, float]], float]:
        """Project 3D point to 2D screen coordinates."""
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
        fov_scale = self.screen_height / 2
        return self.earth_visual_radius / camera.distance * fov_scale

    def is_point_visible_on_sphere(self, point_3d: np.ndarray, camera: Camera) -> bool:
        cam_pos = camera.get_position()
        to_camera = cam_pos - point_3d
        normal = point_3d / np.linalg.norm(point_3d)
        return np.dot(normal, to_camera) > 0

    def is_point_in_front_of_earth(self, point_3d: np.ndarray, camera: Camera) -> bool:
        cam_pos = camera.get_position()
        point_dist = np.linalg.norm(point_3d)

        if point_dist < self.earth_visual_radius:
            return False

        to_point = point_3d - cam_pos
        to_point_dist = np.linalg.norm(to_point)
        to_point_normalized = to_point / to_point_dist

        closest_approach_t = -np.dot(cam_pos, to_point_normalized)

        if closest_approach_t > to_point_dist:
            return True
        if closest_approach_t < 0:
            return True

        closest_point = cam_pos + closest_approach_t * to_point_normalized
        closest_dist_to_center = np.linalg.norm(closest_point)

        if closest_dist_to_center > self.earth_visual_radius:
            return True

        half_chord = math.sqrt(
            self.earth_visual_radius**2 - closest_dist_to_center**2
        )
        entry_t = closest_approach_t - half_chord

        return to_point_dist < entry_t

    def draw_earth(self, camera: Camera) -> None:
        """Draw Earth sphere with gradient."""
        center_3d = np.array([0, 0, 0])
        center_2d, _ = self.project_point(center_3d, camera)

        if center_2d is None:
            return

        radius = self.get_projected_sphere_radius(camera)

        sphere_surface = pygame.Surface(
            (int(radius * 2) + 4, int(radius * 2) + 4), pygame.SRCALPHA
        )
        center_on_surface = (int(radius) + 2, int(radius) + 2)

        for i in range(int(radius), 0, -2):
            factor = i / radius
            alpha = int(200 * factor + 55)
            r = int(
                Colors.SPHERE[0]
                + (Colors.SPHERE_HIGHLIGHT[0] - Colors.SPHERE[0]) * (1 - factor)
            )
            g = int(
                Colors.SPHERE[1]
                + (Colors.SPHERE_HIGHLIGHT[1] - Colors.SPHERE[1]) * (1 - factor)
            )
            b = int(
                Colors.SPHERE[2]
                + (Colors.SPHERE_HIGHLIGHT[2] - Colors.SPHERE[2]) * (1 - factor)
            )
            color = (min(255, r), min(255, g), min(255, b), alpha)
            pygame.draw.circle(sphere_surface, color, center_on_surface, int(i))

        self.screen.blit(
            sphere_surface,
            (int(center_2d[0] - radius - 2), int(center_2d[1] - radius - 2)),
        )

    def draw_earth_grid(self, camera: Camera) -> None:
        """Draw latitude and longitude grid lines on Earth's surface."""
        for i in range(1, self.num_latitude_lines):
            lat = -math.pi / 2 + math.pi * i / self.num_latitude_lines
            self._draw_latitude_line(camera, lat, is_back=True)

        for i in range(self.num_longitude_lines):
            lon = 2 * math.pi * i / self.num_longitude_lines
            self._draw_longitude_line(camera, lon, is_back=True)

        for i in range(1, self.num_latitude_lines):
            lat = -math.pi / 2 + math.pi * i / self.num_latitude_lines
            self._draw_latitude_line(camera, lat, is_back=False)

        for i in range(self.num_longitude_lines):
            lon = 2 * math.pi * i / self.num_longitude_lines
            self._draw_longitude_line(camera, lon, is_back=False)

    def _draw_latitude_line(self, camera: Camera, latitude: float, is_back: bool = False) -> None:
        num_points = 120
        z = self.earth_visual_radius * math.sin(latitude)
        circle_radius = self.earth_visual_radius * math.cos(latitude)

        if circle_radius < 0.001:
            return

        points = []
        for i in range(num_points + 1):
            theta = 2 * math.pi * i / num_points
            x = circle_radius * math.cos(theta)
            y = circle_radius * math.sin(theta)
            points.append(np.array([x, y, z]))

        self._draw_sphere_line(camera, points, is_back)

    def _draw_longitude_line(self, camera: Camera, longitude: float, is_back: bool = False) -> None:
        num_points = 120
        points = []

        for i in range(num_points + 1):
            lat = -math.pi / 2 + math.pi * i / num_points
            x = self.earth_visual_radius * math.cos(lat) * math.cos(longitude)
            y = self.earth_visual_radius * math.cos(lat) * math.sin(longitude)
            z = self.earth_visual_radius * math.sin(lat)
            points.append(np.array([x, y, z]))

        self._draw_sphere_line(camera, points, is_back)

    def _draw_sphere_line(self, camera: Camera, points: List[np.ndarray], draw_back: bool) -> None:
        if len(points) < 2:
            return

        projected = []
        visible = []

        for point in points:
            proj, depth = self.project_point(point, camera)
            is_visible = self.is_point_visible_on_sphere(point, camera)
            projected.append(proj)
            visible.append(is_visible)

        for i in range(len(points) - 1):
            proj1, proj2 = projected[i], projected[i + 1]
            vis1, vis2 = visible[i], visible[i + 1]

            if proj1 is None or proj2 is None:
                continue

            segment_visible = vis1 and vis2
            segment_back = not vis1 and not vis2

            if draw_back and not segment_back:
                continue
            if not draw_back and not segment_visible:
                continue

            dx = proj2[0] - proj1[0]
            dy = proj2[1] - proj1[1]
            if dx * dx + dy * dy > 2500:
                continue

            color = Colors.GRID_LINE_BACK if draw_back else Colors.GRID_LINE
            width = 1 if draw_back else max(1, int(2 / camera.distance * 2))

            pygame.draw.line(
                self.screen, color,
                (int(proj1[0]), int(proj1[1])),
                (int(proj2[0]), int(proj2[1])),
                width,
            )

    def draw_orbit(self, camera: Camera, orbit, color_front=None, color_back=None) -> None:
        if color_front is None:
            color_front = Colors.ORBIT_FRONT
        if color_back is None:
            color_back = Colors.ORBIT_BACK

        points = []
        for i in range(self.orbit_points + 1):
            nu = 2 * math.pi * i / self.orbit_points
            pos_km = orbit.position_eci(nu)
            pos_visual = pos_km * self.scale_factor
            points.append(pos_visual)

        self._draw_space_line(camera, points, color_back, 1, draw_back=True)
        line_width = max(2, int(4 / camera.distance * 2))
        self._draw_space_line(camera, points, color_front, line_width, draw_back=False)

    def _draw_space_line(self, camera: Camera, points: List[np.ndarray], color, width: int, draw_back: bool) -> None:
        if len(points) < 2:
            return

        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]

            proj1, _ = self.project_point(p1, camera)
            proj2, _ = self.project_point(p2, camera)

            if proj1 is None or proj2 is None:
                continue

            front1 = self.is_point_in_front_of_earth(p1, camera)
            front2 = self.is_point_in_front_of_earth(p2, camera)

            segment_front = front1 and front2
            segment_back = not front1 and not front2

            if draw_back and not segment_back:
                continue
            if not draw_back and not segment_front:
                continue

            dx = proj2[0] - proj1[0]
            dy = proj2[1] - proj1[1]
            if dx * dx + dy * dy > 2500:
                continue

            pygame.draw.line(
                self.screen, color,
                (int(proj1[0]), int(proj1[1])),
                (int(proj2[0]), int(proj2[1])),
                width,
            )

    def draw_orbits(self, camera: Camera, orbits: List) -> None:
        drawn: Set[Tuple] = set()

        for orbit in orbits:
            key = (
                round(orbit.semi_major_axis, 2),
                round(orbit.eccentricity, 6),
                round(orbit.inclination, 6),
                round(orbit.longitude_of_ascending_node, 6),
                round(orbit.argument_of_periapsis, 6),
            )

            if key not in drawn:
                drawn.add(key)
                self.draw_orbit(camera, orbit)

    def draw_satellite(self, camera: Camera, satellite, color, size=None) -> None:
        if size is None:
            size = self.satellite_size

        pos_km = satellite.get_position_eci()
        pos_visual = pos_km * self.scale_factor

        in_front = self.is_point_in_front_of_earth(pos_visual, camera)
        proj, depth = self.project_point(pos_visual, camera)

        if proj is None:
            return

        perspective_size = size * (3.0 / depth) if depth > 0 else size
        perspective_size = max(4, min(20, perspective_size))

        vel_km = satellite.get_velocity_eci()
        vel_visual = vel_km * self.scale_factor
        pos_future = pos_visual + vel_visual * 100
        proj_future, _ = self.project_point(pos_future, camera)

        if proj_future is not None:
            direction = (proj_future[0] - proj[0], proj_future[1] - proj[1])
        else:
            direction = None

        triangle_points = self._get_triangle_points(proj, perspective_size, direction)

        if in_front:
            draw_color = color
            outline_color = (255, 255, 255)
            outline_width = 2
        else:
            draw_color = tuple(int(c * 0.4) for c in color)
            outline_color = tuple(int(c * 0.5) for c in color)
            outline_width = 1

        int_points = [(int(p[0]), int(p[1])) for p in triangle_points]
        pygame.draw.polygon(self.screen, draw_color, int_points)
        pygame.draw.polygon(self.screen, outline_color, int_points, outline_width)

    def _get_triangle_points(self, center, size, direction=None):
        if direction is not None and (direction[0] != 0 or direction[1] != 0):
            dx, dy = direction
            mag = math.sqrt(dx * dx + dy * dy)
            dx, dy = dx / mag, dy / mag
            px, py = -dy, dx
        else:
            dx, dy = 0, -1
            px, py = 1, 0

        cx, cy = center
        tip = (cx + dx * size, cy + dy * size)
        left = (cx - dx * size * 0.5 + px * size * 0.6, cy - dy * size * 0.5 + py * size * 0.6)
        right = (cx - dx * size * 0.5 - px * size * 0.6, cy - dy * size * 0.5 - py * size * 0.6)

        return [tip, left, right]

    def draw_satellites(self, camera: Camera, satellites: List, completion_percentages=None) -> None:
        for satellite in satellites:
            if completion_percentages is not None:
                completion = completion_percentages.get(satellite.satellite_id, 0.0)
            else:
                completion = 0.0

            color = get_packet_completion_color(completion)
            self.draw_satellite(camera, satellite, color)

    def draw_base_station(self, camera: Camera, base_station, earth_rotation_angle: float, size: int = 10) -> None:
        pos_km = base_station.get_position_eci(earth_rotation_angle)
        pos_visual = pos_km * self.scale_factor

        in_front = self.is_point_visible_on_sphere(pos_visual, camera)
        proj, depth = self.project_point(pos_visual, camera)

        if proj is None:
            return

        perspective_size = size * (3.0 / depth) if depth > 0 else size
        perspective_size = max(4, min(16, perspective_size))

        if in_front:
            color = Colors.BASE_STATION_COLOR
            outline_color = (255, 255, 255)
            outline_width = 2
        else:
            color = Colors.BASE_STATION_COLOR_BACK
            outline_color = tuple(int(c * 0.5) for c in Colors.BASE_STATION_COLOR)
            outline_width = 1

        half_size = perspective_size / 2
        rect = pygame.Rect(
            int(proj[0] - half_size),
            int(proj[1] - half_size),
            int(perspective_size),
            int(perspective_size),
        )
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, outline_color, rect, outline_width)

    def draw_base_stations(self, camera: Camera, base_stations: List, earth_rotation_angle: float) -> None:
        for base_station in base_stations:
            self.draw_base_station(camera, base_station, earth_rotation_angle)

    def draw_communication_links(self, camera: Camera, satellites: List, active_links: set) -> None:
        if not active_links:
            return

        sat_by_id = {sat.satellite_id: sat for sat in satellites}

        for sat1_id, sat2_id in active_links:
            sat1 = sat_by_id.get(sat1_id)
            sat2 = sat_by_id.get(sat2_id)

            if sat1 is None or sat2 is None:
                continue

            pos1 = sat1.get_position_eci() * self.scale_factor
            pos2 = sat2.get_position_eci() * self.scale_factor

            proj1, _ = self.project_point(pos1, camera)
            proj2, _ = self.project_point(pos2, camera)

            if proj1 is None or proj2 is None:
                continue

            front1 = self.is_point_in_front_of_earth(pos1, camera)
            front2 = self.is_point_in_front_of_earth(pos2, camera)

            if front1 and front2:
                color = Colors.LINK_COLOR
                width = max(1, int(2 / camera.distance * 2))
            else:
                color = Colors.LINK_COLOR_BACK
                width = 1

            pygame.draw.line(
                self.screen, color,
                (int(proj1[0]), int(proj1[1])),
                (int(proj2[0]), int(proj2[1])),
                width,
            )

    def draw_base_station_links(self, camera: Camera, base_stations: List, satellites: List, base_station_links: set, earth_rotation_angle: float) -> None:
        if not base_station_links:
            return

        bs_by_name = {bs.name: bs for bs in base_stations}
        sat_by_id = {sat.satellite_id: sat for sat in satellites}

        for bs_name, sat_id in base_station_links:
            bs = bs_by_name.get(bs_name)
            sat = sat_by_id.get(sat_id)

            if bs is None or sat is None:
                continue

            pos_bs = bs.get_position_eci(earth_rotation_angle) * self.scale_factor
            pos_sat = sat.get_position_eci() * self.scale_factor

            proj_bs, _ = self.project_point(pos_bs, camera)
            proj_sat, _ = self.project_point(pos_sat, camera)

            if proj_bs is None or proj_sat is None:
                continue

            bs_visible = self.is_point_visible_on_sphere(pos_bs, camera)
            sat_visible = self.is_point_in_front_of_earth(pos_sat, camera)

            if bs_visible and sat_visible:
                color = Colors.BASE_STATION_LINK_COLOR
                width = max(1, int(2 / camera.distance * 2))
            else:
                color = Colors.BASE_STATION_LINK_COLOR_BACK
                width = 1

            pygame.draw.line(
                self.screen, color,
                (int(proj_bs[0]), int(proj_bs[1])),
                (int(proj_sat[0]), int(proj_sat[1])),
                width,
            )

    def draw_text(self, text: str, position, font, color=Colors.TEXT) -> int:
        surface = font.render(text, True, color)
        self.screen.blit(surface, position)
        return surface.get_height()

    def draw_info_panel(
        self,
        camera: Camera,
        simulation,
        font,
        time_scale: float,
        paused: bool,
        log_saved: bool = False,
        log_location: Optional[str] = None
    ) -> None:
        """Draw information panel with log status."""
        sim_time = simulation.state.time
        hours = int(sim_time // 3600)
        minutes = int((sim_time % 3600) // 60)
        seconds = int(sim_time % 60)

        n = len(simulation.satellites)
        total_pairs = n * (n - 1) // 2
        active_count = len(simulation.state.active_links)
        bs_link_count = len(simulation.state.base_station_links)

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
        ]

        # Add log status if logging is enabled
        if log_location is not None:
            info_lines.append("")
            if log_saved:
                info_lines.append(f"Log: SAVED")
            else:
                info_lines.append(f"Log: Recording...")

        info_lines.extend([
            "",
            "Controls:",
            "← → : Rotate longitude",
            "↑ ↓ : Rotate latitude",
            "+/- : Zoom in/out",
            "[ ] : Time scale",
            "SPACE : Pause/Resume",
            "R : Regenerate",
            "ESC : Quit",
        ])

        y = 10
        for line in info_lines:
            # Highlight log saved status
            if line == "Log: SAVED":
                y += self.draw_text(line, (10, y), font, Colors.TEXT_SUCCESS) + 2
            else:
                y += self.draw_text(line, (10, y), font) + 2

        # Color legend
        y += 10
        self.draw_text("Satellite Colors:", (10, y), font)
        y += 20

        pygame.draw.rect(self.screen, Colors.SATELLITE_NO_PACKETS, (10, y, 15, 15))
        self.draw_text("0% packets", (30, y), font, Colors.TEXT_DIM)
        y += 20

        pygame.draw.rect(self.screen, Colors.SATELLITE_HALF_PACKETS, (10, y, 15, 15))
        self.draw_text("50% packets", (30, y), font, Colors.TEXT_DIM)
        y += 20

        pygame.draw.rect(self.screen, Colors.SATELLITE_ALL_PACKETS, (10, y, 15, 15))
        self.draw_text("100% packets", (30, y), font, Colors.TEXT_DIM)

        # Satellite details on right side
        y = 10
        for i, satellite in enumerate(simulation.satellites):
            geo = satellite.get_geospatial_position()

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

            sat_color = get_packet_completion_color(completion)

            for j, line in enumerate(sat_info):
                line_color = sat_color if j == 0 else Colors.TEXT_DIM
                y += self.draw_text(
                    line, (self.screen_width - 200, y), font, line_color
                ) + 1
            y += 8

            if y > self.screen_height - 100:
                self.draw_text(
                    f"... and {len(simulation.satellites) - i - 1} more",
                    (self.screen_width - 200, y),
                    font,
                    Colors.TEXT_DIM,
                )
                break