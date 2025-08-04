#!/usr/bin/env python3
"""
Interactive Map Perimeter Editor for Valorant Track Visualizer
Allows you to define map boundaries, walls, and playable areas by clicking on the map
"""

import pygame
import json
import math
import sys
from pathlib import Path
from typing import List, Tuple
import argparse
from .map_collision_system import MapCollisionSystem, CollisionShape, TriggerZone, HeightArea, HeightLevel

class MapPerimeterEditor:
    """Interactive editor for defining map boundaries and collision areas"""
    
    def __init__(self, minimap_path: str, map_name: str, display_size: Tuple[int, int] = None):
        """Initialize map perimeter editor
        
        Args:
            minimap_path: Path to minimap image file
            map_name: Name of the map
            display_size: Optional display window size, None for auto-detect
        """
        pygame.init()
        
        self.map_name = map_name
        self.minimap_path = minimap_path
        self.map_dimensions = None  # Will be set when loading minimap
        self.background = None
        
        # Load minimap first to determine actual map dimensions
        self.load_minimap()
        
        # Use actual map dimensions as display size (same as enhanced track visualizer)
        if self.map_dimensions:
            self.display_w, self.display_h = self.map_dimensions
            print(f"Using map dimensions as display size: {self.display_w}x{self.display_h}")
        elif display_size:
            self.display_w, self.display_h = display_size
        else:
            self.display_w, self.display_h = (880, 780)  # Default fallback
        
        self.screen = pygame.display.set_mode((self.display_w, self.display_h))
        pygame.display.set_caption(f"Map Perimeter Editor - {map_name}")
        
        # Drawing state
        self.current_tool = "height_area"  # "height_area", "wall", "trigger"
        self.current_shape = "polygon"  # "rectangle", "polygon", "circle", "line"
        self.current_points = []
        self.is_drawing = False
        
        # Height area state
        self.current_height_level = HeightLevel.FLOOR
        
        # Trigger zone state
        self.trigger_target_area = ""
        self.trigger_transition_type = "elevation"  # "elevation", "teleport", "area_change"
        
        # Colors
        self.colors = {
            "boundary": (255, 255, 0),      # Yellow (legacy)
            "wall": (255, 0, 0),            # Red
            "trigger": (255, 0, 255),       # Magenta
            "current": (255, 255, 255),     # White
            "grid": (50, 50, 50),           # Dark gray
            # Height level colors
            "height_floor": (139, 69, 19),     # Brown
            "height_low": (255, 165, 0),       # Orange
            "height_mid": (255, 255, 0),       # Yellow
            "height_high": (0, 255, 255),      # Cyan
            "height_ceiling": (128, 0, 128),   # Purple
        }
        
        # Storage
        self.height_areas: List[HeightArea] = []  # Height-based collision system (defines playable map)
        self.walls: List[CollisionShape] = []
        self.trigger_zones: List[TriggerZone] = []
        
        # Load existing data if available
        self.load_existing_data()
        
        # Fonts
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # UI state
        self.show_help = True
        self.show_grid = True
        self.grid_size = 50
        
        # Text input state
        self.text_input_active = False
        self.text_input_prompt = ""
        self.text_input_value = ""
        self.text_input_callback = None
        
        # Constants
        self.ALPHA_TRANSPARENT = 50
        self.ALPHA_SEMI_TRANSPARENT = 60
        self.ALPHA_MEDIUM = 80
        self.ALPHA_OPAQUE = 100
        self.TEXT_BG_PADDING = (4, 2)
        self.BACKGROUND_COLOR = (20, 20, 20)
        self.POINT_RADIUS = 4
        self.LINE_WIDTH = 3
        
        # Height level to color mapping
        self.HEIGHT_LEVEL_COLORS = {
            HeightLevel.FLOOR: self.colors["height_floor"],
            HeightLevel.LOW: self.colors["height_low"],
            HeightLevel.MID: self.colors["height_mid"],
            HeightLevel.HIGH: self.colors["height_high"],
            HeightLevel.CEILING: self.colors["height_ceiling"]
        }
        
        # Key mappings for event handling
        self.HEIGHT_LEVEL_KEYS = {
            pygame.K_0: HeightLevel.FLOOR,
            pygame.K_1: HeightLevel.LOW,
            pygame.K_2: HeightLevel.MID,
            pygame.K_3: HeightLevel.HIGH,
            pygame.K_4: HeightLevel.CEILING
        }
        
        self.TOOL_KEYS = {
            pygame.K_5: "wall",
            pygame.K_6: "trigger"
        }
        
        self.SHAPE_KEYS = {
            pygame.K_q: "rectangle",
            pygame.K_w: "polygon",
            pygame.K_e: "circle",
            pygame.K_r: "line"
        }
    
    def load_minimap(self):
        """Load minimap at original size (same as enhanced track visualizer)"""
        try:
            # Load the minimap image
            img = pygame.image.load(self.minimap_path)
            
            # Get original minimap size and store it
            self.map_dimensions = img.get_size()
            original_w, original_h = self.map_dimensions
            print(f"Loaded minimap: {self.minimap_path}")
            print(f"  Map dimensions: {original_w}x{original_h}")
            
            # Use original minimap size without scaling (1:1 mapping)
            self.background = pygame.Surface((original_w, original_h))
            self.background.fill((20, 20, 20))  # Dark background
            
            # Blit the original minimap directly (no scaling)
            self.background.blit(img, (0, 0))
            
            print(f"  Using original size (no scaling): {original_w}x{original_h}")
            
        except Exception as e:
            print(f"Error loading minimap: {e}")
            self.create_default_background()
    
    def create_default_background(self):
        """Create default background if minimap fails to load"""
        # Use default dimensions if map dimensions aren't available
        if not hasattr(self, 'display_w') or not hasattr(self, 'display_h'):
            self.display_w, self.display_h = (880, 780)
        
        self.background = pygame.Surface((self.display_w, self.display_h))
        self.background.fill(self.BACKGROUND_COLOR)
        
        # Set default map dimensions
        self.map_dimensions = (self.display_w, self.display_h)
    
    def _draw_transparent_surface(self, surface_size: Tuple[int, int], color: Tuple[int, int, int], 
                                alpha: int, position: Tuple[int, int]):
        """Helper method to draw transparent surfaces"""
        if alpha < 255:
            s = pygame.Surface(surface_size, pygame.SRCALPHA)
            s.fill((*color, alpha))
            self.screen.blit(s, position)
    
    def _calculate_polygon_center(self, points: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Calculate the center point of a polygon"""
        center_x = sum(p[0] for p in points) / len(points)
        center_y = sum(p[1] for p in points) / len(points)
        return center_x, center_y
    
    def _draw_text_with_background(self, text: str, font, color: Tuple[int, int, int], 
                                  position: Tuple[int, int], bg_alpha: int = 150):
        """Draw text with semi-transparent background"""
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=position)
        
        # Background for text
        bg_w = text_rect.width + self.TEXT_BG_PADDING[0]
        bg_h = text_rect.height + self.TEXT_BG_PADDING[1]
        bg = pygame.Surface((bg_w, bg_h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, bg_alpha))
        self.screen.blit(bg, (text_rect.x - self.TEXT_BG_PADDING[0]//2, 
                             text_rect.y - self.TEXT_BG_PADDING[1]//2))
        self.screen.blit(text_surface, text_rect)
        
        return text_rect
    
    def _validate_polygon_points(self, points: List[Tuple[float, float]]) -> bool:
        """Validate if polygon has enough points"""
        return len(points) >= 3
    
    def _start_text_input(self, prompt: str, callback):
        """Start text input mode"""
        self.text_input_active = True
        self.text_input_prompt = prompt
        self.text_input_value = ""
        self.text_input_callback = callback
    
    def _handle_text_input(self, event):
        """Handle text input events"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                if self.text_input_callback:
                    self.text_input_callback(self.text_input_value)
                self._end_text_input()
            elif event.key == pygame.K_ESCAPE:
                self._end_text_input()
            elif event.key == pygame.K_BACKSPACE:
                self.text_input_value = self.text_input_value[:-1]
            else:
                self.text_input_value += event.unicode
    
    def _end_text_input(self):
        """End text input mode"""
        self.text_input_active = False
        self.text_input_prompt = ""
        self.text_input_value = ""
        self.text_input_callback = None
    
    def _set_trigger_target(self, target: str):
        """Callback for setting trigger target area"""
        if target.strip():
            self.trigger_target_area = target.strip()
            print(f"Set trigger target area: {self.trigger_target_area}")
        else:
            print("Invalid target area name")
    
    # Direct coordinate mapping - no conversion needed
    # Map coordinates are the same as collision coordinates

    def load_existing_data(self):
        """Load existing collision data"""
        config_path = Path(f"maps/collision_configs/{self.map_name}_collision.json")
        
        if config_path.exists():
            try:
                # Load collision data using actual map dimensions as coordinate system size
                if self.map_dimensions:
                    map_w, map_h = self.map_dimensions
                    coordinate_system_size = map_w  # Use actual map width as coordinate system size
                    
                    collision_system = MapCollisionSystem(self.map_name, coordinate_system_size=coordinate_system_size)
                    
                    # Load data directly - already in map coordinates
                    self.walls = collision_system.walls
                    self.trigger_zones = collision_system.trigger_zones
                    self.height_areas = collision_system.height_areas
                    
                    print(f"Loaded existing collision data for {self.map_name}")
                    print(f"  Map dimensions: {map_w}x{map_h}")
                    print(f"  Coordinate system size: {coordinate_system_size}")
                    print(f"  Loaded {len(self.walls)} walls")
                    print(f"  Loaded {len(self.trigger_zones)} trigger zones")
                    print(f"  Loaded {len(self.height_areas)} height areas")
                else:
                    print(f"Warning: Map dimensions not available, cannot load collision data")
            except Exception as e:
                print(f"Error loading existing data: {e}")
    
    
    def save_data(self):
        """Save collision data to JSON file"""
        config_path = Path(f"maps/collision_configs/{self.map_name}_collision.json")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'map_name': self.map_name,
            'height_areas': [],  # Height-based collision system (defines playable map)
            'walls': [],
            'trigger_zones': []
        }
        
        # Save height areas (new system) - already in map coordinates
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
        
        # Save trigger zones - already in map coordinates
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
        
        print(f"Saved collision data to {config_path}")
    
    def draw_grid(self):
        """Draw coordinate grid"""
        if not self.show_grid:
            return
        
        color = self.colors["grid"]
        
        # Draw vertical lines
        for x in range(0, self.display_w, self.grid_size):
            pygame.draw.line(self.screen, color, (x, 0), (x, self.display_h), 1)
        
        # Draw horizontal lines
        for y in range(0, self.display_h, self.grid_size):
            pygame.draw.line(self.screen, color, (0, y), (self.display_w, y), 1)
    
    def draw_shape(self, shape: CollisionShape, color: Tuple[int, int, int], alpha: int = 100):
        """Draw a collision shape"""
        if shape.shape_type == "rectangle":
            # Convert points to display coordinates
            if len(shape.points) == 2:
                p1 = (int(shape.points[0][0]), int(shape.points[0][1]))
                p2 = (int(shape.points[1][0]), int(shape.points[1][1]))
            else:
                p1 = (int(shape.points[0]), int(shape.points[1]))
                p2 = (int(shape.points[2]), int(shape.points[3]))
            
            x1, y1 = p1
            x2, y2 = p2
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            
            # Draw filled rectangle with transparency
            self._draw_transparent_surface((width, height), color, alpha, (min(x1, x2), min(y1, y2)))
            
            # Draw outline
            pygame.draw.rect(self.screen, color, (min(x1, x2), min(y1, y2), width, height), 2)
        
        elif shape.shape_type == "polygon":
            # Convert points to display coordinates
            display_points = [(int(point[0]), int(point[1])) for point in shape.points]
            
            if len(display_points) >= 3:
                # Draw filled polygon with transparency
                if alpha < 255:
                    s = pygame.Surface((self.display_w, self.display_h), pygame.SRCALPHA)
                    pygame.draw.polygon(s, (*color, alpha), display_points)
                    self.screen.blit(s, (0, 0))
                
                # Draw outline
                pygame.draw.polygon(self.screen, color, display_points, 2)
        
        elif shape.shape_type == "circle":
            # Convert center to display coordinates
            center_x, center_y = int(shape.center_x), int(shape.center_y)
            # Use radius directly since we have 1:1 mapping now
            radius = int(shape.radius)
            
            # Draw filled circle with transparency
            if alpha < 255:
                s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*color, alpha), (radius, radius), radius)
                self.screen.blit(s, (center_x - radius, center_y - radius))
            
            # Draw outline
            pygame.draw.circle(self.screen, color, (center_x, center_y), radius, 2)
        
        elif shape.shape_type == "line":
            # Convert line endpoints to display coordinates
            if len(shape.points) >= 2:
                start = (int(shape.points[0][0]), int(shape.points[0][1]))
                end = (int(shape.points[1][0]), int(shape.points[1][1]))
                
                # Draw line
                pygame.draw.line(self.screen, color, start, end, self.LINE_WIDTH)
                
                # Draw endpoints
                pygame.draw.circle(self.screen, color, start, self.POINT_RADIUS)
                pygame.draw.circle(self.screen, color, end, self.POINT_RADIUS)
    
    def draw_current_shape(self):
        """Draw the shape currently being drawn"""
        if len(self.current_points) < 1:
            return
        
        color = self.colors["current"]
        
        if self.current_shape == "rectangle" and len(self.current_points) == 2:
            p1 = (int(self.current_points[0][0]), int(self.current_points[0][1]))
            p2 = (int(self.current_points[1][0]), int(self.current_points[1][1]))
            x1, y1 = p1
            x2, y2 = p2
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            pygame.draw.rect(self.screen, color, (min(x1, x2), min(y1, y2), width, height), 2)
        
        elif self.current_shape == "polygon" and len(self.current_points) >= 2:
            display_points = [(int(point[0]), int(point[1])) for point in self.current_points]
            
            if self._validate_polygon_points(self.current_points):
                pygame.draw.polygon(self.screen, color, display_points, 2)
            else:
                pygame.draw.lines(self.screen, color, False, display_points, 2)
        
        elif self.current_shape == "circle" and len(self.current_points) == 2:
            center = (int(self.current_points[0][0]), int(self.current_points[0][1]))
            edge = (int(self.current_points[1][0]), int(self.current_points[1][1]))
            radius = int(math.sqrt((edge[0] - center[0])**2 + (edge[1] - center[1])**2))
            pygame.draw.circle(self.screen, color, center, radius, 2)
        
        elif self.current_shape == "line" and len(self.current_points) >= 1:
            if len(self.current_points) == 1:
                # Draw point for line start
                point = (int(self.current_points[0][0]), int(self.current_points[0][1]))
                pygame.draw.circle(self.screen, color, point, 4)
            elif len(self.current_points) == 2:
                # Draw line
                start = (int(self.current_points[0][0]), int(self.current_points[0][1]))
                end = (int(self.current_points[1][0]), int(self.current_points[1][1]))
                pygame.draw.line(self.screen, color, start, end, 3)
                pygame.draw.circle(self.screen, color, start, 4)
                pygame.draw.circle(self.screen, color, end, 4)
        
        # Draw points
        for point in self.current_points:
            display_point = (int(point[0]), int(point[1]))
            pygame.draw.circle(self.screen, color, display_point, self.POINT_RADIUS)
    
    def draw_ui(self):
        """Draw user interface"""
        # Tool info
        tool_text = f"Tool: {self.current_tool.upper()} | Shape: {self.current_shape.upper()}"
        if self.current_tool == "height_area":
            tool_text += f" | Height: {self.current_height_level.name}"
        elif self.current_tool == "trigger":
            tool_text += f" | Target: {self.trigger_target_area or 'NONE'}"
        text_surface = self.font.render(tool_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 10))
        
        # Instructions
        if self.show_help:
            help_lines = [
                "Controls:",
                "0-4: Set height level (0=floor, 1=low, 2=mid, 3=high, 4=ceiling)",
                "5-6: Select tool (5=wall, 6=trigger)", 
                "Q-R: Select shape (Q=rectangle, W=polygon, E=circle, R=line)",
                "Left Click: Add point/finish shape",
                "Right Click: Finish polygon/cancel",
                "T: Set trigger target area (for trigger tool)",
                "X: Delete last height area",
                "Z: Delete last wall",
                "V: Delete last trigger",
                "S: Save data",
                "G: Toggle grid",
                "H: Toggle help",
                "ESC: Quit"
            ]
            
            y_offset = 50
            for line in help_lines:
                text = self.small_font.render(line, True, (200, 200, 200))
                self.screen.blit(text, (10, y_offset))
                y_offset += 20
        
        # Current shape info
        if self.current_points:
            info_text = f"Points: {len(self.current_points)}"
            info_surface = self.small_font.render(info_text, True, (255, 255, 255))
            self.screen.blit(info_surface, (10, self.display_h - 30))
        
        # Text input overlay
        if self.text_input_active:
            # Draw text input overlay
            overlay_height = 60
            overlay_y = self.display_h // 2 - overlay_height // 2
            overlay_surface = pygame.Surface((self.display_w - 40, overlay_height), pygame.SRCALPHA)
            overlay_surface.fill((0, 0, 0, 200))
            self.screen.blit(overlay_surface, (20, overlay_y))
            
            # Draw prompt
            prompt_text = self.font.render(self.text_input_prompt, True, (255, 255, 255))
            self.screen.blit(prompt_text, (30, overlay_y + 10))
            
            # Draw input value with cursor
            input_display = self.text_input_value + "_"
            input_text = self.font.render(input_display, True, (255, 255, 255))
            self.screen.blit(input_text, (30, overlay_y + 30))
            
            # Draw help text
            help_text = self.small_font.render("Press ENTER to confirm, ESC to cancel", True, (200, 200, 200))
            self.screen.blit(help_text, (self.display_w - 250, overlay_y + 5))
        
        # Status
        status_lines = [
            f"Height Areas: {len(self.height_areas)}",
            f"Walls: {len(self.walls)}",
            f"Triggers: {len(self.trigger_zones)}"
        ]
        
        y_offset = self.display_h - 80
        for line in status_lines:
            text = self.small_font.render(line, True, (200, 200, 200))
            self.screen.blit(text, (10, y_offset))
            y_offset += 16
    
    def handle_click(self, pos: Tuple[int, int], button: int):
        """Handle mouse clicks"""
        if button == 1:  # Left click
            map_pos = (float(pos[0]), float(pos[1]))
            
            if self.current_shape == "rectangle":
                self.current_points.append(map_pos)
                if len(self.current_points) == 2:
                    self.finish_shape()
            
            elif self.current_shape == "polygon":
                self.current_points.append(map_pos)
                # Polygon continues until right click or double click
            
            elif self.current_shape == "circle":
                self.current_points.append(map_pos)
                if len(self.current_points) == 2:
                    self.finish_shape()
            
            elif self.current_shape == "line":
                self.current_points.append(map_pos)
                if len(self.current_points) == 2:
                    self.finish_shape()
        
        elif button == 3:  # Right click
            if self.current_shape == "polygon" and self._validate_polygon_points(self.current_points):
                self.finish_shape()
            else:
                self.cancel_shape()
    
    def finish_shape(self):
        """Finish creating the current shape"""
        if not self.current_points:
            return
        
        # Create shape based on type
        if self.current_shape == "rectangle" and len(self.current_points) == 2:
            shape = CollisionShape("rectangle", self.current_points, f"{self.current_tool}_rect")
        
        elif self.current_shape == "polygon" and self._validate_polygon_points(self.current_points):
            shape = CollisionShape("polygon", self.current_points, f"{self.current_tool}_poly")
        
        elif self.current_shape == "circle" and len(self.current_points) == 2:
            center = self.current_points[0]
            edge = self.current_points[1]
            radius = math.sqrt((edge[0] - center[0])**2 + (edge[1] - center[1])**2)
            circle_data = [center[0], center[1], radius]
            shape = CollisionShape("circle", circle_data, f"{self.current_tool}_circle")
        
        elif self.current_shape == "line" and len(self.current_points) == 2:
            shape = CollisionShape("line", self.current_points, f"{self.current_tool}_line")
        
        else:
            print(f"Cannot finish shape - insufficient points")
            return
        
        # Add to appropriate collection
        if self.current_tool == "height_area":
            height_area_num = len(self.height_areas) + 1
            shape.name = f"height_{self.current_height_level.name.lower()}_{height_area_num}"
            height_area = HeightArea(shape, self.current_height_level, shape.name)
            self.height_areas.append(height_area)
            print(f"Added height area {height_area_num}: {shape.shape_type} at {self.current_height_level.name}")
        # Legacy boundary tool removed - use height areas instead
        elif self.current_tool == "wall":
            self.walls.append(shape)
            print(f"Added wall: {shape.shape_type}")
        # Playable areas removed - height areas define playable map
        elif self.current_tool == "trigger":
            if not self.trigger_target_area:
                print("Error: No target area set for trigger. Press T to set target area.")
                return
            
            trigger_num = len(self.trigger_zones) + 1
            trigger_name = f"trigger_{trigger_num}"
            trigger_zone = TriggerZone(shape, self.trigger_target_area, self.trigger_transition_type, trigger_name)
            self.trigger_zones.append(trigger_zone)
            print(f"Added trigger {trigger_num}: {shape.shape_type} -> {self.trigger_target_area}")
        
        self.current_points = []
    
    def cancel_shape(self):
        """Cancel current shape creation"""
        self.current_points = []
        print("Cancelled shape creation")
    
    def run(self):
        """Main editor loop"""
        clock = pygame.time.Clock()
        running = True
        
        print(f"Map Perimeter Editor - {self.map_name}")
        print("Use mouse to define map boundaries and collision areas")
        print("Press H to toggle help")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if self.text_input_active:
                        self._handle_text_input(event)
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_h:
                        self.show_help = not self.show_help
                    elif event.key == pygame.K_g:
                        self.show_grid = not self.show_grid
                    elif event.key == pygame.K_s:
                        self.save_data()
                    # Height level hotkeys (0-4)
                    elif event.key in self.HEIGHT_LEVEL_KEYS:
                        self.current_height_level = self.HEIGHT_LEVEL_KEYS[event.key]
                        self.current_tool = "height_area"
                        self.cancel_shape()
                        print(f"Set height level: {self.current_height_level.name}")
                    # Tool selection hotkeys (5-8)
                    elif event.key in self.TOOL_KEYS:
                        self.current_tool = self.TOOL_KEYS[event.key]
                        self.cancel_shape()
                    # Shape selection hotkeys (Q, W, E, R)
                    elif event.key in self.SHAPE_KEYS:
                        self.current_shape = self.SHAPE_KEYS[event.key]
                        self.cancel_shape()
                    elif event.key == pygame.K_x:
                        # Delete last height area
                        if self.current_tool == "height_area" and self.height_areas:
                            deleted = self.height_areas.pop()
                            print(f"Deleted height area: {deleted.name}")
                        else:
                            print("No height areas to delete")
                    elif event.key == pygame.K_z:
                        # Delete last wall
                        if self.walls:
                            deleted = self.walls.pop()
                            print(f"Deleted wall: {deleted.name}")
                        else:
                            print("No walls to delete")
                    # Playable areas removed - only height areas, walls, and triggers remain
                    elif event.key == pygame.K_v:
                        # Delete last trigger zone
                        if self.trigger_zones:
                            deleted = self.trigger_zones.pop()
                            print(f"Deleted trigger zone: {deleted.name}")
                        else:
                            print("No trigger zones to delete")
                    elif event.key == pygame.K_t:
                        # Set trigger target area
                        if self.current_tool == "trigger":
                            self._start_text_input("Enter target area name: ", self._set_trigger_target)
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos, event.button)
            
            # Draw everything
            if self.background:
                self.screen.blit(self.background, (0, 0))
            else:
                self.screen.fill((20, 20, 20))
            
            self.draw_grid()
            
            # Draw height areas (behind everything else)
            for height_area in self.height_areas:
                height_color = {
                    HeightLevel.FLOOR: self.colors["height_floor"],
                    HeightLevel.LOW: self.colors["height_low"],
                    HeightLevel.MID: self.colors["height_mid"],
                    HeightLevel.HIGH: self.colors["height_high"],
                    HeightLevel.CEILING: self.colors["height_ceiling"]
                }.get(height_area.height_level, (128, 128, 128))
                self.draw_shape(height_area.shape, height_color, 60)
                
                # Draw height level label
                if height_area.shape.shape_type == "polygon" and len(height_area.shape.points) > 0:
                    center_x, center_y = self._calculate_polygon_center(height_area.shape.points)
                    display_center = (int(center_x), int(center_y))
                    label_text = f"H{height_area.height_level}"
                    text = self.small_font.render(label_text, True, height_color)
                    text_rect = text.get_rect(center=display_center)
                    # Background for text
                    bg = pygame.Surface((text_rect.width + 4, text_rect.height + 2), pygame.SRCALPHA)
                    bg.fill((0, 0, 0, 150))
                    self.screen.blit(bg, (text_rect.x - 2, text_rect.y - 1))
                    self.screen.blit(text, text_rect)
            
            # Legacy boundaries removed - only height areas, walls, playable areas, and triggers
            
            for wall in self.walls:
                self.draw_shape(wall, self.colors["wall"], self.ALPHA_MEDIUM)
            
            # Playable areas removed - height areas define playable map
            
            # Draw trigger zones
            for trigger in self.trigger_zones:
                self.draw_shape(trigger.trigger_shape, self.colors["trigger"], 70)
                # Draw trigger label
                if trigger.trigger_shape.shape_type == "polygon" and len(trigger.trigger_shape.points) > 0:
                    center_x, center_y = self._calculate_polygon_center(trigger.trigger_shape.points)
                    display_center = (int(center_x), int(center_y))
                    label_text = f"→{trigger.target_area}"
                    text = self.small_font.render(label_text, True, self.colors["trigger"])
                    text_rect = text.get_rect(center=display_center)
                    self.screen.blit(text, text_rect)
            
            # Draw current shape being created
            self.draw_current_shape()
            
            # Draw UI
            self.draw_ui()
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()

def main():
    parser = argparse.ArgumentParser(description="Interactive Map Perimeter Editor")
    parser.add_argument("map_name", help="Name of the map (e.g., icebox, ascent)")
    parser.add_argument("--minimap", help="Path to minimap image (auto-detected if not provided)")
    parser.add_argument("--display-size", type=int, nargs=2, default=None,
                        help="Display window size [width height] (defaults to map dimensions)")
    
    args = parser.parse_args()
    
    # Auto-detect minimap path if not provided
    minimap_path = args.minimap
    if not minimap_path:
        minimap_path = f"maps/{args.map_name}.png"
        if not Path(minimap_path).exists():
            print(f"Minimap not found: {minimap_path}")
            print("Available maps:")
            maps_dir = Path("maps")
            if maps_dir.exists():
                for map_file in maps_dir.glob("*.png"):
                    print(f"  {map_file.name}")
            return 1
    
    # Create and run editor
    display_size = tuple(args.display_size) if args.display_size else None
    editor = MapPerimeterEditor(minimap_path, args.map_name, display_size)
    editor.run()
    
    return 0

if __name__ == "__main__":
    exit(main())
