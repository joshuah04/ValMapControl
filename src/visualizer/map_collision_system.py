#!/usr/bin/env python3
"""
Map Collision System for Valorant Track Visualizer
Handles map boundaries, walls, and playable areas for vision cone clipping
"""

import json
import math
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from .collision.height_level import HeightLevel
from .collision.collision_shape import CollisionShape
from .collision.height_area import HeightArea
from .collision.trigger_zone import TriggerZone
from .collision.player_history_tracker import PlayerHistoryTracker

class MapCollisionSystem:
    """Manages collision detection and map boundaries with height-based areas"""
    
    def __init__(self, map_name: str, coordinate_system_size: int = 700):
        """Initialize map collision system
        
        Args:
            map_name: Name of the map
            coordinate_system_size: Size of the coordinate system
        """
        self.map_name = map_name
        self.coordinate_system_size = coordinate_system_size
        self.height_areas: List[HeightArea] = []  # Height-based areas (define playable map)
        self.walls: List[CollisionShape] = []  # Walls block vision regardless of height
        self.trigger_zones: List[TriggerZone] = []  # Trigger zones for area transitions
        self.agent_heights: Dict[str, HeightLevel] = {}  # Track agent height levels
        self.position_tracker = PlayerHistoryTracker()  # Track position history for layered areas
        
        # Legacy support removed - only using height areas now
        
        # Load collision data
        self.load_collision_data()
    
    def _process_coordinates(self, points: List) -> List:
        """Process coordinates - all coordinates are already in map pixel space"""
        # No scaling needed - coordinates are stored in actual map pixel coordinates
        return points if points else []
    
    def load_collision_data(self):
        """Load collision data from JSON file"""
        config_path = Path(f"maps/collision_configs/{self.map_name}_collision.json")
        
        if not config_path.exists():
            print(f"No collision config found for {self.map_name}")
            return
        
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            # Load height areas (new height-based system)
            if 'height_areas' in data:
                for area_data in data['height_areas']:
                    points = self._process_coordinates(area_data['shape']['points'])
                    shape = CollisionShape(
                        area_data['shape']['type'],
                        points,
                        area_data['shape'].get('name', 'height_shape')
                    )
                    height_level = HeightLevel(area_data['height_level'])
                    height_area = HeightArea(shape, height_level, area_data.get('name', ''))
                    self.height_areas.append(height_area)
            
            # Legacy boundary support removed - only using height areas now
            
            # Load walls
            for wall_data in data.get('walls', []):
                points = self._process_coordinates(wall_data['points'])
                wall = CollisionShape(
                    wall_data['type'],
                    points,
                    wall_data.get('name', 'wall')
                )
                self.walls.append(wall)
            
            # Playable areas removed - height areas define playable map
            
            # Load trigger zones
            for trigger_data in data.get('trigger_zones', []):
                points = self._process_coordinates(trigger_data['shape']['points'])
                trigger_shape = CollisionShape(
                    trigger_data['shape']['type'],
                    points,
                    trigger_data['shape'].get('name', 'trigger_shape')
                )
                trigger_zone = TriggerZone(
                    trigger_shape,
                    trigger_data.get('target_area', 'unknown'),
                    trigger_data.get('transition_type', 'elevation'),
                    trigger_data.get('name', 'trigger')
                )
                self.trigger_zones.append(trigger_zone)
            
            print(f"Loaded collision data for {self.map_name}")
            print(f"  Height areas: {len(self.height_areas)}")
            print(f"  Walls: {len(self.walls)}")
            print(f"  Trigger zones: {len(self.trigger_zones)}")
        
        except Exception as e:
            print(f"Error loading collision data for {self.map_name}: {e}")
    
    def get_height_at_position(self, x: float, y: float) -> HeightLevel:
        """Get the height level at a specific position"""
        # Check height areas (highest priority takes precedence)
        highest_level = HeightLevel.FLOOR
        for height_area in self.height_areas:
            if height_area.contains_point(x, y):
                if height_area.height_level > highest_level:
                    highest_level = height_area.height_level
        return highest_level
    
    def get_overlapping_height_areas(self, x: float, y: float) -> List[HeightArea]:
        """Get all height areas that contain a specific position"""
        overlapping_areas = []
        for height_area in self.height_areas:
            if height_area.contains_point(x, y):
                overlapping_areas.append(height_area)
        return overlapping_areas
    
    def is_layered_position(self, x: float, y: float) -> bool:
        """Check if a position has multiple overlapping height areas"""
        overlapping_areas = self.get_overlapping_height_areas(x, y)
        return len(overlapping_areas) > 1
    
    def get_height_with_history(self, agent_id: str, x: float, y: float) -> HeightLevel:
        """Get height level using position history for layered area transitions"""
        overlapping_areas = self.get_overlapping_height_areas(x, y)
        
        # Single height area - use standard logic
        if len(overlapping_areas) <= 1:
            return self.get_height_at_position(x, y)
        
        # Multiple height areas (layered position)
        # Use history to determine appropriate height
        if self.position_tracker.has_recent_height_history(agent_id):
            previous_height = self.position_tracker.get_previous_height(agent_id)
            dominant_height = self.position_tracker.get_dominant_recent_height(agent_id)
            
            # Check if previous height matches any overlapping area
            available_heights = [area.height_level for area in overlapping_areas]
            
            # Prefer previous height if available in this layered area
            if previous_height in available_heights:
                return previous_height
            
            # Fall back to dominant recent height if available
            if dominant_height in available_heights:
                return dominant_height
        
        # Default to lowest available height in layered area
        available_heights = [area.height_level for area in overlapping_areas]
        return min(available_heights)
    
    def set_agent_height(self, agent_id: str, height: HeightLevel):
        """Set the height level for an agent"""
        self.agent_heights[agent_id] = height
    
    def get_agent_height(self, agent_id: str) -> HeightLevel:
        """Get the height level for an agent"""
        return self.agent_heights.get(agent_id, HeightLevel.FLOOR)
    
    def update_agent_height_from_position(self, agent_id: str, x: float, y: float):
        """Update an agent's height based on their position with history tracking"""
        # Get height using history for layered areas
        height = self.get_height_with_history(agent_id, x, y)
        
        # Update agent height
        self.set_agent_height(agent_id, height)
        
        # Update position history
        self.position_tracker.update_position(agent_id, x, y, height)
    
    def can_agents_see_each_other(self, viewer_id: str, target_id: str, 
                                  viewer_pos: Tuple[float, float], target_pos: Tuple[float, float]) -> bool:
        """Check if two agents can see each other based on height and walls with layered area support"""
        # Update positions to get current heights with history
        self.update_agent_height_from_position(viewer_id, viewer_pos[0], viewer_pos[1])
        self.update_agent_height_from_position(target_id, target_pos[0], target_pos[1])
        
        # Get agent heights
        viewer_height = self.get_agent_height(viewer_id)
        target_height = self.get_agent_height(target_id)
        
        # Check for layered area visibility rules
        if self._are_in_layered_areas(viewer_pos, target_pos):
            # In layered areas, can only see agents at exact same height
            if viewer_height != target_height:
                return False
        else:
            # Normal height-based visibility rules
            if not HeightLevel.can_see(viewer_height, target_height):
                return False
        
        # Check if walls block the line of sight (walls always block regardless of height)
        return not self._line_intersects_walls(viewer_pos[0], viewer_pos[1], target_pos[0], target_pos[1])
    
    def _are_in_layered_areas(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> bool:
        """Check if either position is in a layered area"""
        return (self.is_layered_position(pos1[0], pos1[1]) or 
                self.is_layered_position(pos2[0], pos2[1]))
    
    def _line_intersects_walls(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """Check if a line intersects any walls"""
        for wall in self.walls:
            if wall.intersects_line(x1, y1, x2, y2):
                return True
        return False

    def is_position_valid(self, x: float, y: float) -> bool:
        """Check if a position is valid (inside playable area, not in walls)"""
        # Position validation using height areas only
        # Height areas define the valid playable space
        
        # Check if inside a wall
        for wall in self.walls:
            if wall.contains_point(x, y):
                return False
        
        # Check if inside any height area (these define the playable map)
        for height_area in self.height_areas:
            if height_area.contains_point(x, y):
                return True
        
        # Outside all height areas - not playable
        return False
    
    def update_agent_position(self, agent_id: str, x: float, y: float) -> Dict[str, str]:
        """Update agent position, height, and check for trigger zone transitions with layered area support"""
        transitions = {}
        
        # Update agent height based on position with history tracking
        self.update_agent_height_from_position(agent_id, x, y)
        
        # Check for layered area transitions
        if self.is_layered_position(x, y):
            current_height = self.get_agent_height(agent_id)
            overlapping_areas = self.get_overlapping_height_areas(x, y)
            available_heights = [area.height_level.name for area in overlapping_areas]
            transitions['layered_area'] = f"entered_layered_area_at_{current_height.name}_level"
            print(f"Agent {agent_id} entered layered area at {current_height.name} level (available: {available_heights})")
        
        # Check all trigger zones
        for trigger in self.trigger_zones:
            if trigger.check_agent_transition(agent_id, x, y):
                # Agent entered trigger zone
                if not hasattr(self, 'area_access_state'):
                    self.area_access_state = {}
                if agent_id not in self.area_access_state:
                    self.area_access_state[agent_id] = set()
                
                # Grant access to target area
                self.area_access_state[agent_id].add(trigger.target_area)
                transitions[trigger.name] = f"entered_{trigger.target_area}"
                
                print(f"Agent {agent_id} crossed trigger '{trigger.name}' -> access to '{trigger.target_area}'")
        
        return transitions
    
    def can_agent_access_area(self, agent_id: str, area_name: str) -> bool:
        """Check if agent has access to a specific area"""
        if agent_id not in self.area_access_state:
            return True  # Default access if no restrictions
        
        return area_name in self.area_access_state[agent_id]
    
    def get_agent_accessible_areas(self, agent_id: str) -> set:
        """Get all areas accessible to an agent"""
        return self.area_access_state.get(agent_id, set())
    
    def is_position_valid_with_triggers(self, agent_id: str, x: float, y: float) -> bool:
        """Enhanced position validation that considers trigger zones and area access"""
        # First check basic position validity
        if not self.is_position_valid(x, y):
            return False
        
        # If no trigger zones defined, use standard validation
        if not self.trigger_zones:
            return True
        
        # Area access validation using height areas only
        
        return True
    
    def get_vision_blocking_shapes(self, agent_id: str = None) -> List[CollisionShape]:
        """Get all shapes that block vision for a specific agent"""
        # Walls always block vision regardless of height
        blocking_shapes = list(self.walls)
        
        # Vision blocking uses walls and height areas only
        
        return blocking_shapes
    
    def clip_vision_line(self, start_x: float, start_y: float, end_x: float, end_y: float, 
                        agent_id: str = None) -> Tuple[float, float]:
        """Clip a vision line to the nearest wall or height area intersection"""
        closest_intersection = (end_x, end_y)
        min_distance = float('inf')
        
        # Get vision-blocking shapes for this agent (walls always block)
        blocking_shapes = self.get_vision_blocking_shapes(agent_id)
        
        for shape in blocking_shapes:
            if shape.intersects_line(start_x, start_y, end_x, end_y):
                # For now, return the wall intersection point
                # This is a simplified implementation
                intersection = self._find_line_intersection(start_x, start_y, end_x, end_y, shape)
                if intersection:
                    distance = math.sqrt((intersection[0] - start_x)**2 + (intersection[1] - start_y)**2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_intersection = intersection
        
        # Check height area intersections for height-aware vision blocking
        if agent_id:
            viewer_height = self.get_agent_height(agent_id)
            
            for height_area in self.height_areas:
                # Only check areas that are higher than viewer
                if height_area.height_level > viewer_height:
                    if height_area.shape.intersects_line(start_x, start_y, end_x, end_y):
                        # Find intersection point with this height area
                        intersection = self._find_height_area_intersection(start_x, start_y, end_x, end_y, height_area)
                        if intersection:
                            # Check if anything behind this area would be blocked
                            distance = math.sqrt((intersection[0] - start_x)**2 + (intersection[1] - start_y)**2)
                            target_height = self.get_height_at_position(end_x, end_y)
                            
                            # Block vision if target is at same height as or behind this blocking area
                            if target_height >= height_area.height_level and distance < min_distance:
                                min_distance = distance
                                closest_intersection = intersection
        
        return closest_intersection
    
    def _find_height_area_intersection(self, start_x: float, start_y: float, end_x: float, end_y: float, 
                                     height_area: HeightArea) -> Optional[Tuple[float, float]]:
        """Find intersection point between line and height area"""
        # Use the same intersection logic as walls
        return self._find_line_intersection(start_x, start_y, end_x, end_y, height_area.shape)
    
    def can_see_position(self, viewer_id: str, viewer_pos: Tuple[float, float], 
                        target_pos: Tuple[float, float]) -> bool:
        """Check if an agent can see a specific position based on height and walls with layered area support"""
        # Update viewer position to get current height with history
        self.update_agent_height_from_position(viewer_id, viewer_pos[0], viewer_pos[1])
        
        # Get viewer and target heights
        viewer_height = self.get_agent_height(viewer_id)
        target_height = self.get_height_at_position(target_pos[0], target_pos[1])
        
        # Check for layered area visibility rules
        if self._are_in_layered_areas(viewer_pos, target_pos):
            # In layered areas, can only see positions at exact same height
            if viewer_height != target_height:
                return False
        else:
            # Normal height-based visibility rules
            if not HeightLevel.can_see(viewer_height, target_height):
                return False
            
            # Check if areas behind higher levels block vision
            if self._higher_area_blocks_vision(viewer_pos, target_pos, viewer_height):
                return False
        
        # Check if walls block the line of sight
        return not self._line_intersects_walls(viewer_pos[0], viewer_pos[1], target_pos[0], target_pos[1])
    
    def _higher_area_blocks_vision(self, viewer_pos: Tuple[float, float], 
                                  target_pos: Tuple[float, float], viewer_height: HeightLevel) -> bool:
        """Check if higher level areas block vision to the target"""
        # For each height area that is higher than viewer
        for height_area in self.height_areas:
            if height_area.height_level > viewer_height:
                # Check if the line of sight passes through this higher area
                if self._line_intersects_height_area(viewer_pos[0], viewer_pos[1], 
                                                   target_pos[0], target_pos[1], height_area):
                    # If target is at the same height as this blocking area, vision is blocked
                    target_height = self.get_height_at_position(target_pos[0], target_pos[1])
                    if target_height >= height_area.height_level:
                        return True
        return False
    
    def _line_intersects_height_area(self, x1: float, y1: float, x2: float, y2: float, 
                                    height_area: HeightArea) -> bool:
        """Check if a line intersects a height area"""
        return height_area.shape.intersects_line(x1, y1, x2, y2)
    
    def _find_line_intersection(self, start_x: float, start_y: float, end_x: float, end_y: float, wall: CollisionShape) -> Optional[Tuple[float, float]]:
        """Find intersection point between line and wall (simplified)"""
        # This is a simplified implementation
        # For a more accurate implementation, you'd need to calculate exact intersection points
        # For now, return the midpoint of the line segment
        return ((start_x + end_x) / 2, (start_y + end_y) / 2)
    
    def save_collision_data(self):
        """Save collision data to JSON file"""
        config_path = Path(f"maps/collision_configs/{self.map_name}_collision.json")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'map_name': self.map_name,
            'height_areas': [],  # New height-based system
            # Legacy map_boundaries removed  
            'walls': [],
            'trigger_zones': []
        }
        
        # Save height areas
        for height_area in self.height_areas:
            data['height_areas'].append({
                'name': height_area.name,
                'height_level': int(height_area.height_level),
                'shape': {
                    'type': height_area.shape.shape_type,
                    'points': height_area.shape.points,
                    'name': height_area.shape.name
                }
            })
        
        # Legacy boundaries removed - using height areas only
        
        for wall in self.walls:
            data['walls'].append({
                'type': wall.shape_type,
                'points': wall.points,
                'name': wall.name
            })
        
        # Playable areas removed - height areas define playable map
        
        # Save trigger zones
        for trigger in self.trigger_zones:
            data['trigger_zones'].append({
                'name': trigger.name,
                'target_area': trigger.target_area,
                'transition_type': trigger.transition_type,
                'shape': {
                    'type': trigger.trigger_shape.shape_type,
                    'points': trigger.trigger_shape.points,
                    'name': trigger.trigger_shape.name
                }
            })
        
        with open(config_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved collision data for {self.map_name}")

def create_collision_config_tool():
    """Interactive tool for creating collision configurations"""
    print("Map Collision Configuration Tool")
    print("This tool will be implemented in the interactive map editor script")
    print("Use map_perimeter_editor.py to create collision configurations")
