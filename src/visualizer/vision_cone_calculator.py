#!/usr/bin/env python3
"""
Unified Vision Cone Calculator for Valorant Track Visualizer
Ensures perfect synchronization between visual display and map control calculations
"""

import math
from typing import List, Tuple, Optional
from shapely.geometry import Polygon
from .map_collision_system import MapCollisionSystem, CollisionShape
from functools import lru_cache

class VisionConeCalculator:
    """Unified vision cone calculation for both display and map control"""
    
    def __init__(self, collision_system: MapCollisionSystem, 
                 vision_range: float = 400.0, cone_angle: float = 75.0):
        """
        Initialize vision cone calculator
        
        Args:
            collision_system: Collision system for wall/height detection
            vision_range: Maximum vision distance in pixels
            cone_angle: Vision cone angle in degrees
        """
        self.collision_system = collision_system
        self.vision_range = vision_range
        self.cone_angle = math.radians(cone_angle)
        self.half_angle = self.cone_angle / 2
        
        # Number of ray samples for smooth cone edges (reduced for performance)
        self.num_arc_points = 10  # Reduced from 40 for better performance
        
        # Direct collision detection - no spatial partitioning needed
        # This ensures all walls are always checked and none are missed
    
    def calculate_vision_cone(self, agent_id: str, center_x: float, center_y: float, 
                            orientation: float, utilities: List = None) -> List[Tuple[float, float]]:
        """
        Generate cone points using ray-casting with collision detection
        
        Args:
            agent_id: Agent identifier for height-based vision
            center_x: Agent X position in map coordinates
            center_y: Agent Y position in map coordinates
            orientation: Agent orientation in degrees
            utilities: List of utility objects that can block vision (smokes, walls, etc.)
            
        Returns:
            List of (x, y) coordinates defining cone polygon in map coordinates
        """
        # Convert orientation to radians
        angle_rad = math.radians(orientation)
        
        # Calculate cone boundaries
        left_angle = angle_rad - self.half_angle
        right_angle = angle_rad + self.half_angle
        
        # Update agent height for collision detection
        self.collision_system.update_agent_height_from_position(agent_id, center_x, center_y)
        
        # Create vision cone points
        cone_points = [(center_x, center_y)]  # Start from agent center
        
        # Cast rays along the arc
        for i in range(self.num_arc_points + 1):
            angle = left_angle + (right_angle - left_angle) * i / self.num_arc_points
            intersection = self._cast_vision_ray(agent_id, center_x, center_y, angle, utilities)
            cone_points.append(intersection)
        
        return cone_points
    
    def _cast_vision_ray(self, agent_id: str, start_x: float, start_y: float, 
                        angle: float, utilities: List = None) -> Tuple[float, float]:
        """
        Cast a single vision ray and find intersection with obstacles
        
        Args:
            agent_id: Agent identifier for height-based vision
            start_x: Ray start X position
            start_y: Ray start Y position
            angle: Ray angle in radians
            utilities: List of utility objects that can block vision
            
        Returns:
            (x, y) intersection point
        """
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        closest_intersection = None
        min_distance = float('inf')
        
        # Maximum ray distance
        max_distance = self.vision_range
        default_end_x = start_x + max_distance * dx
        default_end_y = start_y + max_distance * dy
        
        # Check utility intersections first (smokes, walls, etc.)
        if utilities:
            for utility in utilities:
                if self._utility_blocks_vision(utility):
                    intersection = self._find_utility_intersection(
                        start_x, start_y, angle, utility
                    )
                    if intersection:
                        distance = math.sqrt((intersection[0] - start_x)**2 + (intersection[1] - start_y)**2)
                        if distance < min_distance and distance > 1:  # Avoid zero-distance
                            min_distance = distance
                            closest_intersection = intersection
        
        # Check wall intersections directly (walls always block vision)
        if hasattr(self.collision_system, 'walls'):
            for wall in self.collision_system.walls:
                intersection = self._find_shape_intersection(
                    start_x, start_y, default_end_x, default_end_y, wall
                )
                if intersection:
                    distance = math.sqrt((intersection[0] - start_x)**2 + (intersection[1] - start_y)**2)
                    if distance < min_distance and distance > 1:  # Avoid zero-distance
                        min_distance = distance
                        closest_intersection = intersection
        
        # Check height area intersections directly
        if hasattr(self.collision_system, 'height_areas'):
            agent_height = self.collision_system.get_agent_height(agent_id)
            agent_height_value = agent_height.value
            
            # Track which height areas we've entered
            areas_entered = set()
            all_intersections = []
            
            # Find all intersections with all height areas directly
            for height_area in self.collision_system.height_areas:
                area_height_value = height_area.height_level.value
                height_diff = area_height_value - agent_height_value
                
                # Only check areas above agent
                if height_diff > 0:
                    intersections = self._find_all_shape_intersections(
                        start_x, start_y, default_end_x, default_end_y, 
                        height_area.shape
                    )
                    
                    for intersection in intersections:
                        distance = math.sqrt((intersection[0] - start_x)**2 + (intersection[1] - start_y)**2)
                        if distance > 1:  # Avoid zero-distance
                            all_intersections.append({
                                'distance': distance,
                                'point': intersection,
                                'height_diff': height_diff,
                                'area_id': id(height_area)
                            })
            
            # Sort intersections by distance
            all_intersections.sort(key=lambda x: x['distance'])
            
            # Process intersections to find vision blocking
            for intersection in all_intersections:
                height_diff = intersection['height_diff']
                area_id = intersection['area_id']
                
                # Areas more than 2 levels above block vision immediately
                if height_diff > 2:
                    if intersection['distance'] < min_distance:
                        min_distance = intersection['distance']
                        closest_intersection = intersection['point']
                    break
                
                # Areas exactly 2 levels above - can see but not past
                elif height_diff == 2:
                    if area_id not in areas_entered:
                        # Entering the area - we can see it
                        areas_entered.add(area_id)
                    else:
                        # Exiting the area - vision stops here
                        if intersection['distance'] < min_distance:
                            min_distance = intersection['distance']
                            closest_intersection = intersection['point']
                        break
                
                # Areas 1 level above - we can see through
                else:
                    if area_id not in areas_entered:
                        areas_entered.add(area_id)
        
        # If no intersection found, use maximum vision range
        if closest_intersection is None:
            closest_intersection = (default_end_x, default_end_y)
        
        return closest_intersection
    
    def clear_spatial_cache(self):
        """No spatial cache to clear in direct collision detection mode"""
        pass
    
    def _find_shape_intersection(self, start_x: float, start_y: float, 
                                end_x: float, end_y: float, 
                                shape: CollisionShape) -> Optional[Tuple[float, float]]:
        """Find first intersection between ray and shape"""
        if shape.shape_type == "rectangle":
            return self._find_rectangle_intersection(start_x, start_y, end_x, end_y, shape)
        elif shape.shape_type == "polygon":
            return self._find_polygon_intersection(start_x, start_y, end_x, end_y, shape)
        elif shape.shape_type == "circle":
            return self._find_circle_intersection(start_x, start_y, end_x, end_y, shape)
        elif shape.shape_type == "line":
            return self._find_line_intersection(start_x, start_y, end_x, end_y, shape)
        return None
    
    @lru_cache(1028)
    def _find_all_shape_intersections(self, start_x: float, start_y: float, 
                                     end_x: float, end_y: float, 
                                     shape: CollisionShape) -> List[Tuple[float, float]]:
        """Find all intersections between ray and shape (for entry/exit detection)"""
        intersections = []
        
        if shape.shape_type in ['polygon', 'rectangle']:
            # Get shape edges
            if shape.shape_type == 'polygon':
                points = shape.points
            else:  # rectangle
                if len(shape.points) == 2:
                    x1, y1 = shape.points[0]
                    x2, y2 = shape.points[1]
                else:
                    x1, y1, x2, y2 = shape.points
                points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            
            # Check each edge
            for i in range(len(points)):
                p1 = points[i]
                p2 = points[(i + 1) % len(points)]
                
                intersection = self._line_segment_intersection(
                    start_x, start_y, end_x, end_y,
                    p1[0], p1[1], p2[0], p2[1]
                )
                if intersection:
                    intersections.append(intersection)
        
        elif shape.shape_type == 'circle':
            # Circle can have up to 2 intersections
            circle_intersections = self._find_circle_intersections(
                start_x, start_y, end_x, end_y, shape
            )
            intersections.extend(circle_intersections)
        
        return intersections
    
    def _find_rectangle_intersection(self, start_x: float, start_y: float, 
                                   end_x: float, end_y: float, 
                                   shape: CollisionShape) -> Optional[Tuple[float, float]]:
        """Find intersection with rectangle"""
        if len(shape.points) == 2:
            rx1, ry1 = shape.points[0]
            rx2, ry2 = shape.points[1]
        else:
            rx1, ry1, rx2, ry2 = shape.points
        
        left = min(rx1, rx2)
        right = max(rx1, rx2)
        top = min(ry1, ry2)
        bottom = max(ry1, ry2)
        
        intersections = []
        
        # Check each edge
        edges = [
            (left, top, right, top),      # Top
            (right, top, right, bottom),   # Right
            (right, bottom, left, bottom), # Bottom
            (left, bottom, left, top)      # Left
        ]
        
        for edge in edges:
            intersection = self._line_segment_intersection(
                start_x, start_y, end_x, end_y,
                edge[0], edge[1], edge[2], edge[3]
            )
            if intersection:
                intersections.append(intersection)
        
        # Return closest intersection
        if intersections:
            return min(intersections, key=lambda p: (p[0] - start_x)**2 + (p[1] - start_y)**2)
        
        return None
    
    @lru_cache(1028)
    def _find_polygon_intersection(self, start_x: float, start_y: float, 
                                 end_x: float, end_y: float, 
                                 shape: CollisionShape) -> Optional[Tuple[float, float]]:
        """Find intersection with polygon"""
        intersections = []
        
        for i in range(len(shape.points)):
            px1, py1 = shape.points[i]
            px2, py2 = shape.points[(i + 1) % len(shape.points)]
            
            intersection = self._line_segment_intersection(
                start_x, start_y, end_x, end_y, px1, py1, px2, py2
            )
            if intersection:
                intersections.append(intersection)
        
        # Return closest intersection
        if intersections:
            return min(intersections, key=lambda p: (p[0] - start_x)**2 + (p[1] - start_y)**2)
        
        return None
    
    def _find_circle_intersection(self, start_x: float, start_y: float, 
                                end_x: float, end_y: float, 
                                shape: CollisionShape) -> Optional[Tuple[float, float]]:
        """Find first intersection with circle"""
        intersections = self._find_circle_intersections(start_x, start_y, end_x, end_y, shape)
        if intersections:
            # Return closest intersection
            return min(intersections, key=lambda p: (p[0] - start_x)**2 + (p[1] - start_y)**2)
        return None
    
    def _find_circle_intersections(self, start_x: float, start_y: float, 
                                 end_x: float, end_y: float, 
                                 shape: CollisionShape) -> List[Tuple[float, float]]:
        """Find all intersections with circle (entry and exit points)"""
        intersections = []
        
        # Ray direction
        dx = end_x - start_x
        dy = end_y - start_y
        
        # Vector from ray start to circle center
        cx = shape.center_x - start_x
        cy = shape.center_y - start_y
        
        # Quadratic equation coefficients
        a = dx * dx + dy * dy
        if a == 0:
            return intersections
        
        b = -2 * (cx * dx + cy * dy)
        c = cx * cx + cy * cy - shape.radius * shape.radius
        
        discriminant = b * b - 4 * a * c
        if discriminant >= 0:
            sqrt_discriminant = math.sqrt(discriminant)
            t1 = (-b - sqrt_discriminant) / (2 * a)
            t2 = (-b + sqrt_discriminant) / (2 * a)
            
            # Add valid intersections
            for t in [t1, t2]:
                if 0 <= t <= 1:
                    intersection_x = start_x + t * dx
                    intersection_y = start_y + t * dy
                    intersections.append((intersection_x, intersection_y))
        
        return intersections
    
    @lru_cache(1028)
    def _find_line_intersection(self, start_x: float, start_y: float, 
                              end_x: float, end_y: float, 
                              shape: CollisionShape) -> Optional[Tuple[float, float]]:
        """Find intersection with line shape"""
        if len(shape.points) >= 2:
            lx1, ly1 = shape.points[0]
            lx2, ly2 = shape.points[1]
            return self._line_segment_intersection(
                start_x, start_y, end_x, end_y, lx1, ly1, lx2, ly2
            )
        return None
    
    @lru_cache(2048)
    def _line_segment_intersection(self, x1: float, y1: float, x2: float, y2: float,
                                 x3: float, y3: float, x4: float, y4: float) -> Optional[Tuple[float, float]]:
        """Find intersection point between two line segments"""
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None  # Lines are parallel
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        # Check if intersection is within both line segments
        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection_x = x1 + t * (x2 - x1)
            intersection_y = y1 + t * (y2 - y1)
            return (intersection_x, intersection_y)
        
        return None
    
    def get_cone_for_display(self, cone_points: List[Tuple[float, float]], 
                           map_dimensions: Tuple[int, int], 
                           display_dimensions: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Convert cone points from map coordinates to display coordinates
        
        Args:
            cone_points: List of (x, y) in map coordinates
            map_dimensions: (width, height) of map
            display_dimensions: (width, height) of display
            
        Returns:
            List of (x, y) in display coordinates
        """
        if not map_dimensions or not display_dimensions:
            # No scaling needed
            return [(int(x), int(y)) for x, y in cone_points]
        
        map_w, map_h = map_dimensions
        disp_w, disp_h = display_dimensions
        
        # Scale factors
        scale_x = disp_w / map_w
        scale_y = disp_h / map_h
        
        # Convert each point
        display_points = []
        for x, y in cone_points:
            disp_x = int(x * scale_x)
            disp_y = int(y * scale_y)
            display_points.append((disp_x, disp_y))
        
        return display_points
    
    def get_cone_polygon(self, cone_points: List[Tuple[float, float]]) -> Polygon:
        """
        Convert cone points to Shapely polygon for area calculations
        
        Args:
            cone_points: List of (x, y) coordinates
            
        Returns:
            Shapely Polygon object
        """
        if len(cone_points) < 3:
            return Polygon()  # Empty polygon
        
        try:
            # Create polygon from points
            poly = Polygon(cone_points)
            
            # Validate and fix if needed
            if not poly.is_valid:
                poly = poly.buffer(0)  # Fix invalid geometry
            
            return poly
        except Exception as e:
            print(f"Error creating vision cone polygon: {e}")
            return Polygon()  # Return empty polygon on error
    
    def _utility_blocks_vision(self, utility) -> bool:
        """Check if a utility object blocks vision (e.g., smokes, walls)"""
        utility_type = utility.get('type', '').lower()
        
        # Define which utility types block vision
        vision_blocking_types = {
            # All smoke types (both circle and line)
            'smoke-circle', 'smoke-line',  # Re-enabled smoke-line
            # Wall types
            'wall-rect', 'wall-circle', 'wall-line', 'wall',
        }
        
        # Also check if the utility type contains key blocking words
        blocking_keywords = ['smoke', 'wall']
        for keyword in blocking_keywords:
            if keyword in utility_type:
                return True
        
        return utility_type in vision_blocking_types
    
    def _find_utility_intersection(self, start_x: float, start_y: float, angle: float, 
                                 utility) -> Optional[Tuple[float, float]]:
        """Find intersection point between vision ray and utility object"""
        try:
            # Get utility position and size
            util_pos = utility.get('position', {})
            util_x = util_pos.get('x', 0)
            util_y = util_pos.get('y', 0)
            
            # Utility coordinates are already in map coordinate system
            map_util_x = float(util_x)
            map_util_y = float(util_y)
            
            # Get utility radius - calculate from bbox if not provided
            util_radius = utility.get('radius')
            if util_radius is None:
                bbox = utility.get('bbox', [0, 0, 0, 0])
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox
                    # Use half the average of width and height as radius
                    width = abs(x2 - x1)
                    height = abs(y2 - y1)
                    util_radius = (width + height) / 4.0
                else:
                    util_radius = 30.0  # Default radius
            
            # Handle different utility types
            utility_type = utility.get('type', '').lower()
            
            if utility_type in ['smoke-circle', 'omen_smoke', 'brimstone_smoke', 'astra_smoke']:
                # Handle circular utilities
                return self._find_circle_utility_intersection(
                    start_x, start_y, angle, map_util_x, map_util_y, util_radius
                )
            elif utility_type == 'smoke-line':
                # Handle line-based smokes
                line_endpoints = utility.get('line_endpoints', [])
                if len(line_endpoints) >= 2:
                    return self._find_line_utility_intersection(
                        start_x, start_y, angle, line_endpoints
                    )
            else:
                # Default to circle intersection for other types
                return self._find_circle_utility_intersection(
                    start_x, start_y, angle, map_util_x, map_util_y, util_radius
                )
            
            return None
            
        except (KeyError, TypeError, ValueError):
            # If utility data is malformed, skip intersection
            return None
    
    def _find_circle_utility_intersection(self, start_x: float, start_y: float, angle: float,
                                        util_x: float, util_y: float, radius: float) -> Optional[Tuple[float, float]]:
        """Find intersection between ray and circular utility"""
        # Ray direction
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        # Vector from ray start to utility center
        cx = util_x - start_x
        cy = util_y - start_y
        
        # Project utility center onto ray direction
        projection = cx * dx + cy * dy
        
        # If utility is behind the ray start, no intersection
        if projection < 0:
            return None
        
        # Find closest point on ray to utility center
        closest_x = start_x + projection * dx
        closest_y = start_y + projection * dy
        
        # Distance from utility center to ray
        distance_to_ray = math.sqrt((closest_x - util_x)**2 + (closest_y - util_y)**2)
        
        # Check if ray intersects utility circle
        if distance_to_ray <= radius:
            # Calculate exact intersection point using Pythagorean theorem
            # Distance from closest point to actual intersection
            intersection_offset = math.sqrt(radius**2 - distance_to_ray**2)
            
            # Find intersection point (closer to ray start)
            intersection_distance = projection - intersection_offset
            
            if intersection_distance >= 0:
                intersection_x = start_x + intersection_distance * dx
                intersection_y = start_y + intersection_distance * dy
                return (intersection_x, intersection_y)
        
        return None
    
    def _find_line_utility_intersection(self, start_x: float, start_y: float, angle: float,
                                      line_endpoints: List) -> Optional[Tuple[float, float]]:
        """Find intersection between ray and line utility (smoke-line)"""
        if len(line_endpoints) < 2:
            return None
        
        # Ray direction
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        # Ray end point (use max vision range)
        end_x = start_x + self.vision_range * dx
        end_y = start_y + self.vision_range * dy
        
        # Find closest intersection with any line segment
        closest_intersection = None
        min_distance = float('inf')
        
        for i in range(len(line_endpoints) - 1):
            x1, y1 = line_endpoints[i]
            x2, y2 = line_endpoints[i + 1]
            
            intersection = self._line_segment_intersection(
                start_x, start_y, end_x, end_y,
                float(x1), float(y1), float(x2), float(y2)
            )
            
            if intersection:
                distance = math.sqrt((intersection[0] - start_x)**2 + (intersection[1] - start_y)**2)
                if distance < min_distance and distance > 1:  # Avoid zero-distance
                    min_distance = distance
                    closest_intersection = intersection
        
        return closest_intersection
