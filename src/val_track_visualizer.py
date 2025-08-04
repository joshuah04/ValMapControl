#!/usr/bin/env python3
"""
Track Visualizer for Valorant Video Tracker
"""

import json
from pathlib import Path
import pygame
import pygame.gfxdraw
from collections import defaultdict, deque
import math
import argparse
import time
import numpy as np
from typing import Optional, Tuple, Dict
from visualizer.map_collision_system import MapCollisionSystem, CollisionShape, create_collision_config_tool
from visualizer.vision_cone_calculator import VisionConeCalculator
from visualizer.collision_aware_grid_control import CollisionAwareGridControl
from post_tracking_stats import analyze_tracker_output

class TrackVisualizer:
    """Visualizer output"""
    
    def __init__(self, display_size=None, minimap_path=None, fullscreen=False, map_name=None, 
                 decay_time=5.0, permanent_control=False, takeover_threshold=0.6, 
                 contest_threshold=0.3, min_control_strength=0.4):
        """Initialize track visualizer
        
        Args:
            display_size: Tuple of (width, height) for display size, None for auto-detect
            minimap_path: Path to minimap image file
            fullscreen: Whether to run in fullscreen mode
            map_name: Name of the map for collision system
            decay_time: Time for control areas to decay
            permanent_control: Whether control areas are permanent
            takeover_threshold: Threshold for team takeover
            contest_threshold: Threshold for contested areas
            min_control_strength: Minimum control strength to display
        """
        pygame.init()
        
        # Set map name first (needed for loading minimap to determine dimensions)
        self.map_name = map_name
        self.decay_time = decay_time
        self.permanent_control = permanent_control
        self.takeover_threshold = takeover_threshold
        self.contest_threshold = contest_threshold
        self.min_control_strength = min_control_strength
        self.map_dimensions = None  # Will be set when loading minimap
        
        # Load background first to determine actual map dimensions
        self.background = None
        self.load_minimap(minimap_path)
        
        # Use actual map dimensions as display size (no scaling)
        if self.map_dimensions:
            self.display_w, self.display_h = self.map_dimensions
            print(f"Using map dimensions as display size: {self.display_w}x{self.display_h}")
        elif display_size:
            self.display_w, self.display_h = display_size
        else:
            self.display_w, self.display_h = (1000, 1000)  # Default fallback (not good)
        
        if fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.display_w, self.display_h = self.screen.get_size()
        else:
            self.screen = pygame.display.set_mode((self.display_w, self.display_h))
        
        pygame.display.set_caption("Valorant Track Visualizer")
        
        # Color scheme
        self.colors = {
            # Team utilities
            "team1-turret": (200, 255, 200),        # Dark green
            "team2-turret": (255, 200, 200),        # Dark red
            "team1-trip-circle": (100, 255, 100),   # Darkish green
            "team1-trip-rect": (100, 255, 100),
            "team1-trip-line": (100, 255, 100),
            "team2-trip-circle": (255, 100, 100),   # Darkish red
            "team2-trip-rect": (255, 100, 100),
            "team2-trip-line": (255, 100, 100),
            
            # Neutral utilities
            "smoke-circle": (200, 200, 200),    # Gray
            "smoke-line": (200, 200, 200), 
            "wall-rect": (255, 255, 0),         # Yellow
            "wall-circle": (255, 255, 0),
            "wall-line": (255, 255, 0),
            
            # Agents
            "team1": (0, 255, 0),       # Green
            "team2": (255, 0, 0),       # Red
        }
        
        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        # Map dimensions will be used as display dimensions (no scaling)
        self.minimap_size = self.map_dimensions if self.map_dimensions else (880, 780)
        
        # Animation and tracking
        self.agent_trails = defaultdict(lambda: deque(maxlen=15))
        self.agent_positions = {}
        self.smoothing_factor = 0.4
        self.current_frame_data = {}  # Store current frame data for vision cone calculations
        
        # Statistics
        self.frame_stats = {
            'total_frames': 0,
            'agents_detected': 0,
            'utilities_detected': 0,
            'confidence_avg': 0.0
        }
        
        # Collision system with layered height support - initialized in load_minimap if map_name is provided
        # self.collision_system is set in load_minimap()
        if not hasattr(self, 'collision_system'):
            self.collision_system = None
        self.show_collision_overlay = False
        
        self.vision_cone_calculator = None
        if self.collision_system:
            self.vision_cone_calculator = VisionConeCalculator(
                self.collision_system,
                vision_range=400.0,
                cone_angle=75.0
            )

        # Map control system - initialized in load_minimap, don't reset here
        if not hasattr(self, 'map_control_system'):
            self.map_control_system = None
        self.show_map_control = True
        
        # Performance optimization: Frame skipping for map control
        self.map_control_frame_skip = 0  # Update every 3 frames
        self.frame_counter = 0
        self.cached_control_data = None
        self.cached_control_percentages = {}
        
        # Performance optimization: Vision cone caching
        self.vision_cone_cache = {}  # Cache vision cones per agent
        self.vision_cone_threshold = 5.0  # Recalculate if position changes > 5 pixels or orientation > 5 degrees
        
        # Performance optimization: Pre-allocated surfaces for reuse
        self.surface_cache = {
            'control': pygame.Surface((self.display_w, self.display_h), pygame.SRCALPHA),
            'collision': pygame.Surface((self.display_w, self.display_h), pygame.SRCALPHA),
            'cone': pygame.Surface((self.display_w, self.display_h), pygame.SRCALPHA),
            'utility_small': pygame.Surface((200, 200), pygame.SRCALPHA),
            'utility_medium': pygame.Surface((400, 400), pygame.SRCALPHA),
            'text_background': pygame.Surface((300, 50), pygame.SRCALPHA),
            'hud_panel': pygame.Surface((280, 150), pygame.SRCALPHA),
            'control_hud': pygame.Surface((180, 130), pygame.SRCALPHA),
            'legend': pygame.Surface((190, 100), pygame.SRCALPHA)
        }
        
        # Convert surfaces for better performance
        for key, surface in self.surface_cache.items():
            self.surface_cache[key] = surface.convert_alpha()
        
        # Legacy surface references for compatibility
        self.control_surface = self.surface_cache['control']
        self.collision_surface = self.surface_cache['collision']
        self.cone_surface = self.surface_cache['cone']
        
        # Performance optimization: Cache performance stats
        self.cached_perf_stats = {}
        self.perf_stats_update_interval = 10  # Update every 10 frames
        
        # Performance optimization: Pre-calculated coordinate conversion factors
        self.coord_scale_x = None
        self.coord_scale_y = None
        self.scale_factors = None
        
        # Performance profiling
        self.perf_stats = {}
        self.profile_enabled = False

        self.start_time = time.time()  # Track start time for consistent timing
        self.current_time = self.start_time
    
    def _setup_coordinate_conversion(self, tracking_dimensions):
        """Pre-calculate coordinate conversion matrices - call once per video"""
        if self.map_dimensions and tracking_dimensions:
            tracking_w, tracking_h = tracking_dimensions
            map_w, map_h = self.map_dimensions
            
            # Store scale factors
            self.coord_scale_x = map_w / tracking_w
            self.coord_scale_y = map_h / tracking_h
            
            # Store as numpy array for batch operations
            self.scale_factors = np.array([self.coord_scale_x, self.coord_scale_y])
    
    def tracking_to_map_coords(self, x, y, tracking_dimensions=None):
        """Convert tracking coordinates to map coordinates - optimized"""
        # Use pre-calculated factors if available
        if self.coord_scale_x is not None and self.coord_scale_y is not None:
            return x * self.coord_scale_x, y * self.coord_scale_y
        
        # Fallback to original calculation
        if not self.map_dimensions or not tracking_dimensions:
            return x, y
        
        # Calculate and cache for future use
        self._setup_coordinate_conversion(tracking_dimensions)
        return x * self.coord_scale_x, y * self.coord_scale_y
    
    def tracking_to_map_coords_batch(self, coords_list):
        """Convert multiple coordinates at once using numpy"""
        if not coords_list or self.scale_factors is None:
            return coords_list
        
        # Convert list of (x,y) tuples to numpy array
        coords_array = np.array(coords_list, dtype=np.float32)
        # Multiply by scale factors
        converted = coords_array * self.scale_factors
        # Return as list of tuples
        return [(x, y) for x, y in converted]
    
    def _to_int_coords(self, x, y):
        """Convert float coordinates to integers for drawing"""
        return int(x + 0.5), int(y + 0.5)  # Round to nearest pixel
    
    def _should_recalculate_vision_cone(self, agent_id, map_x, map_y, orientation):
        """Caching with spatial hashing and dead zones"""
        if agent_id not in self.vision_cone_cache:
            return True
        
        cached_data = self.vision_cone_cache[agent_id]
        
        # Use grid-based position check (less sensitive to small movements)
        grid_size = 15  # 15-pixel grid cells
        grid_x = int(map_x / grid_size)
        grid_y = int(map_y / grid_size)
        cached_grid_x = int(cached_data['map_x'] / grid_size)
        cached_grid_y = int(cached_data['map_y'] / grid_size)
        
        if grid_x != cached_grid_x or grid_y != cached_grid_y:
            return True
        
        # Orientation check with larger dead zone
        orientation_diff = abs(orientation - cached_data['orientation'])
        if orientation_diff > 180:
            orientation_diff = 360 - orientation_diff
        
        return orientation_diff > 15  # Increased from 5 to 15 degrees
    
    def map_to_display_coords(self, map_x, map_y):
        """Convert map coordinates to display coordinates"""
        # Direct mapping - map coordinates are now display coordinates
        return int(map_x), int(map_y)
    
    def _detect_tracking_dimensions(self, json_data):
        """Auto-detect tracking dimensions from JSON data"""
        
        # Prioritize reading the 'video_dimensions' key from the JSON file
        if 'video_dimensions' in json_data:
            try:
                dims_str = json_data['video_dimensions']
                w_str, h_str = dims_str.lower().split('x')
                detected_w, detected_h = int(w_str), int(h_str)
                print(f"Detected tracking dimensions from JSON video_dimensions: {detected_w}x{detected_h}")
                return (detected_w, detected_h)
            except (ValueError, KeyError, TypeError, IndexError) as e:
                print(f"Warning: Could not parse 'video_dimensions'. Falling back to frame scan. Error: {e}")

        # Fallback to scanning frames if the key is missing or invalid
        print("Falling back to scanning frames to detect dimensions...")
        max_x, max_y = 0, 0
        
        # Handle both new ({'frames': ...}) and old data structures
        frame_source = json_data.get('frames', json_data)
        
        if not frame_source or not isinstance(frame_source, dict):
             print("Warning: No frame data found to scan for dimensions. Using default 650x580.")
             return (650, 580)

        # Sample frames to find the maximum coordinate values
        sample_keys = list(frame_source.keys())[:50] # Scan first 50 frames for better accuracy
        for frame_key in sample_keys:
            frame_data = frame_source.get(frame_key, {})
            
            for agent in frame_data.get('agents', []):
                if 'bbox' in agent and len(agent['bbox']) >= 4:
                    max_x = max(max_x, agent['bbox'][2]) # x2
                    max_y = max(max_y, agent['bbox'][3]) # y2
            for util in frame_data.get('utilities', []):
                 if 'bbox' in util and len(util['bbox']) >= 4:
                    max_x = max(max_x, util['bbox'][2])
                    max_y = max(max_y, util['bbox'][3])
        
        if max_x == 0 or max_y == 0:
            print("Warning: Could not determine dimensions from frame scan. Using default 650x580.")
            return (650, 580)
        
        detected_w = int(max_x * 1.05) # Add a 5% padding
        detected_h = int(max_y * 1.05) # Add a 5% padding
        
        print(f"Auto-detected tracking dimensions via frame scan: {detected_w}x{detected_h}")
        return (detected_w, detected_h)
    
    def load_minimap(self, minimap_path):
        """Load and prepare minimap background"""
        if minimap_path and Path(minimap_path).exists():
            try:
                # Load the minimap image
                img = pygame.image.load(minimap_path)
                
                # Get original minimap size and store it
                self.map_dimensions = img.get_size()
                original_w, original_h = self.map_dimensions
                print(f"Loaded minimap: {minimap_path}")
                print(f"  Map dimensions: {original_w}x{original_h}")
                
                # Initialize collision system with actual map dimensions
                if self.map_name:
                    try:
                        # Use actual map width as coordinate system size since collision coordinates are in map pixel space
                        coordinate_system_size = original_w  # Use actual map width instead of max dimension
                        self.collision_system = MapCollisionSystem(self.map_name, coordinate_system_size=coordinate_system_size)
                        print(f"  Loaded collision system with coordinate size: {coordinate_system_size} (map: {original_w}x{original_h})")
                        
                        
                        # Initialize collision-aware grid map control system
                        try:
                            self.map_control_system = CollisionAwareGridControl(
                                collision_system=self.collision_system,
                                grid_size=24,  # 24px grid cells for optimal performance
                                vision_range=400.0,
                                player_control_radius=48.0,
                                decay_time=self.decay_time,
                                cone_angle=75.0,
                                permanent_control=self.permanent_control,
                                takeover_threshold=self.takeover_threshold,
                                contest_threshold=self.contest_threshold,
                                min_control_strength=self.min_control_strength
                            )
                            control_mode = "PERMANENT" if self.permanent_control else "TIME-DECAY"
                            print(f"Collision-aware grid system loaded ({control_mode} mode, optimized for map layout!)")
                            if self.permanent_control:
                                print(f"Takeover threshold: {self.takeover_threshold}")
                                print(f"Contest threshold: {self.contest_threshold}")
                                print(f"Min control strength: {self.min_control_strength}")
                                
                        except Exception as control_error:
                            print(f"  Error: Could not load collision-aware map control system: {control_error}")
                            import traceback
                            traceback.print_exc()
                            self.map_control_system = None
                            
                    except Exception as collision_error:
                        print(f"  Warning: Could not load collision system: {collision_error}")
                
                # Use original minimap size without scaling (1:1 mapping)
                self.background = pygame.Surface((original_w, original_h))
                self.background.fill((20, 20, 20))  # Dark background
                
                # Blit the original minimap directly (no scaling)
                self.background.blit(img, (0, 0))
                
                print(f"  Using original size (no scaling): {original_w}x{original_h}")
                
            except Exception as e:
                print(f"Error loading minimap: {e}")
                self.create_default_background()
        else:
            print(f"Minimap not found at: {minimap_path}")
            self.create_default_background()
    
    def create_default_background(self):
        """Create default grid background"""
        # Ensure display dimensions are set
        if not hasattr(self, 'display_w') or not hasattr(self, 'display_h'):
            self.display_w, self.display_h = (880, 780)  # Default fallback
        
        self.background = pygame.Surface((self.display_w, self.display_h))
        self.background.fill((20, 20, 20))
        
        # Draw grid
        grid_color = (40, 40, 40)
        for x in range(0, self.display_w, 50):
            pygame.draw.line(self.background, grid_color, (x, 0), (x, self.display_h))
        for y in range(0, self.display_h, 50):
            pygame.draw.line(self.background, grid_color, (0, y), (self.display_w, y))
    
    def draw_utility(self, utility_data, tracking_dimensions):
        """Draw utility with visualization"""
        try:
            pos_x = utility_data['position']['x']
            pos_y = utility_data['position']['y']
            # Convert tracking coordinates to map coordinates, then to display coordinates
            map_x, map_y = self.tracking_to_map_coords(pos_x, pos_y)
            x, y = self.map_to_display_coords(map_x, map_y)
        except (KeyError, TypeError) as e:
            print(f"Error processing utility position: {e}")
            return
        
        util_type = utility_data.get('type', 'unknown')
        confidence = utility_data.get('confidence', 1.0)
        
        # Validate confidence (inline for performance)
        confidence = confidence if isinstance(confidence, (int, float)) else 1.0
        
        color = self.colors.get(util_type, (255, 255, 255))
        
        # Adjust color intensity based on confidence
        if confidence < 1.0:
            color = tuple(int(c * (0.5 + 0.5 * confidence)) for c in color)
        
        bbox = utility_data.get('bbox')
        if bbox:
            # Convert bbox from tracking to display coordinates (optimized)
            map_x1, map_y1 = self.tracking_to_map_coords(bbox[0], bbox[1])
            map_x2, map_y2 = self.tracking_to_map_coords(bbox[2], bbox[3])
            x1, y1 = self.map_to_display_coords(map_x1, map_y1)
            x2, y2 = self.map_to_display_coords(map_x2, map_y2)
            width = x2 - x1
            height = y2 - y1
            
            if "line" in util_type:
                # Line drawing with confidence visualization
                line_width = max(2, int(4 * confidence))
                if confidence > 0.7:
                    # High confidence - solid line with glow
                    pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), line_width + 2)
                    pygame.draw.line(self.screen, (255, 255, 255), (x1, y1), (x2, y2), line_width)
                else:
                    # Lower confidence - dashed line
                    self.draw_dashed_line((x1, y1), (x2, y2), color, line_width)
            
            elif "circle" in util_type or ("smoke" in util_type and "line" not in util_type):
                # Draw circle with transparency (optimized)
                radius = max(width, height) // 2
                if confidence > 0.7:  # Only use transparency for high confidence
                    s = self.surface_cache['utility_small']
                    s.fill((0, 0, 0, 0))  # Clear with transparent
                    # Resize if needed for this specific utility
                    if radius*2 > 200:
                        s = pygame.transform.scale(s, (radius*2, radius*2))
                    else:
                        s = self.surface_cache['utility_small']
                    alpha = int(120 * confidence)
                    pygame.draw.circle(s, (*color, alpha), (radius, radius), radius)
                    self.screen.blit(s, (x - radius, y - radius))
                else:
                    # Low confidence - just draw outline
                    pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)
            
            else:
                # Rectangle utilities
                if width <= 200 and height <= 200:
                    s = self.surface_cache['utility_small']
                    s.fill((0, 0, 0, 0))  # Clear with transparent
                else:
                    s = self.surface_cache['utility_medium']
                    s.fill((0, 0, 0, 0))  # Clear with transparent
                    if width > 400 or height > 400:
                        s = pygame.transform.scale(s, (width, height))
                
                alpha = int(80 * confidence)
                temp_surface = pygame.Surface((width, height), pygame.SRCALPHA)
                temp_surface.fill((*color, alpha))
                self.screen.blit(temp_surface, (x1, y1))
                pygame.draw.rect(self.screen, color, (x1, y1, width, height), 2)
        else:
            # Default point visualization
            radius = 8
            pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)
            pygame.gfxdraw.aacircle(self.screen, x, y, radius, (255, 255, 255))
        
        # Draw confidence indicator
        if confidence < 1.0:
            conf_text = f"{confidence:.2f}"
            text = self.font_small.render(conf_text, True, (255, 255, 255))
            self.screen.blit(text, (x + 10, y - 10))
    
    def draw_dashed_line(self, start, end, color, width):
        """Draw dashed line for low confidence utilities"""
        x1, y1 = start
        x2, y2 = end
        
        dx = x2 - x1
        dy = y2 - y1
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance > 0:
            dx /= distance
            dy /= distance
            
            dash_length = 8
            gap_length = 4
            pos = 0
            
            while pos < distance:
                start_x = x1 + dx * pos
                start_y = y1 + dy * pos
                end_pos = min(pos + dash_length, distance)
                end_x = x1 + dx * end_pos
                end_y = y1 + dy * end_pos
                
                pygame.draw.line(self.screen, color, 
                               (int(start_x), int(start_y)), 
                               (int(end_x), int(end_y)), width)
                pos += dash_length + gap_length
    
    def draw_map_control_overlay(self, tracking_dimensions):
        """Draw map control visualization overlay with strength-based transparency using cached data"""
        if not self.map_control_system or not self.show_map_control or not self.cached_control_data:
            return
        
        # Use cached control data for performance
        pixel_control = self.cached_control_data['pixel_control']
        control_colors = self.map_control_system.get_control_colors()
        grid_size = self.cached_control_data.get('grid_size', 8)
        
        # Clear the pre-allocated surface
        self.control_surface.fill((0, 0, 0, 0))  # Transparent
        
        # Draw control pixels with strength-based alpha
        for (px, py), control_info in pixel_control.items():
            status = control_info['status']
            strength = control_info['strength']
            
            # Convert map coordinates to display coordinates
            display_x, display_y = self.map_to_display_coords(px, py)
            
            # Get base color
            base_color = control_colors[status]
            
            # Adjust alpha based on strength with better visibility
            # Strength 1.0 = full alpha, strength 0.0 = minimum alpha
            min_alpha = 20   # Minimum visibility
            max_alpha = 100  # Maximum visibility (not too opaque)
            alpha = int(min_alpha + (max_alpha - min_alpha) * strength)
            
            # Create color with adjusted alpha
            color = (base_color[0], base_color[1], base_color[2], alpha)
            
            # Draw a filled square for each grid point
            # Use filled_rect for better performance with alpha
            rect = pygame.Rect(display_x - grid_size//2, display_y - grid_size//2, 
                            grid_size, grid_size)
            pygame.draw.rect(self.control_surface, color, rect)
            
            # Optional: Add slight glow effect for active control
            if strength > 0.8:
                # Draw a slightly larger, more transparent square for glow
                glow_alpha = int(alpha * 0.3)
                glow_color = (base_color[0], base_color[1], base_color[2], glow_alpha)
                glow_rect = rect.inflate(4, 4)  # Make it 4 pixels larger on each side
                pygame.draw.rect(self.control_surface, glow_color, glow_rect)
        
        # Blit the control surface to main screen
        self.screen.blit(self.control_surface, (0, 0))
    
    def draw_control_hud(self, control_percentages: Dict[str, float]):
        """Draw map control HUD showing team percentages and persistence info"""
        if not self.map_control_system or not self.show_map_control:
            return
        
        # Control HUD panel - bottom left corner (smaller)
        hud_x = 10
        hud_y = self.display_h - 140  # Smaller height
        hud_w, hud_h = 180, 130
        
        # Create HUD background using smaller cached surface
        hud_surface = self.surface_cache['control_hud']
        hud_surface.fill((0, 0, 0, 200))
        self.screen.blit(hud_surface, (hud_x, hud_y))
        
        # Title (smaller)
        title = self.font_small.render("Control", True, (255, 255, 255))
        self.screen.blit(title, (hud_x + 8, hud_y + 8))
        
        # Control percentages (compact)
        y_offset = 25
        
        # Team 1 (Green) with bar
        team1_text = f"T1:{control_percentages['team1']:.0f}%"
        team1_surface = self.font_small.render(team1_text, True, (0, 255, 0))
        self.screen.blit(team1_surface, (hud_x + 8, hud_y + y_offset))
        # Draw percentage bar (smaller)
        bar_width = int((hud_w - 30) * control_percentages['team1'] / 100)
        pygame.draw.rect(self.screen, (0, 255, 0, 80), 
                        (hud_x + 8, hud_y + y_offset + 12, bar_width, 2))
        
        # Team 2 (Red) with bar
        y_offset += 20
        team2_text = f"T2:{control_percentages['team2']:.0f}%"
        team2_surface = self.font_small.render(team2_text, True, (255, 0, 0))
        self.screen.blit(team2_surface, (hud_x + 8, hud_y + y_offset))
        # Draw percentage bar (smaller)
        bar_width = int((hud_w - 30) * control_percentages['team2'] / 100)
        pygame.draw.rect(self.screen, (255, 0, 0, 80), 
                        (hud_x + 8, hud_y + y_offset + 12, bar_width, 2))
        
        # Contested (Yellow)
        y_offset += 20
        contested_text = f"Contest:{control_percentages['contested']:.0f}%"
        contested_surface = self.font_small.render(contested_text, True, (255, 255, 0))
        self.screen.blit(contested_surface, (hud_x + 8, hud_y + y_offset))
        
        # Neutral/Open (Gray)
        y_offset += 16
        neutral_text = f"Open:{control_percentages['neutral']:.0f}%"
        neutral_surface = self.font_small.render(neutral_text, True, (128, 128, 128))
        self.screen.blit(neutral_surface, (hud_x + 8, hud_y + y_offset))
        
        # Divider line (smaller)
        y_offset += 18
        pygame.draw.line(self.screen, (100, 100, 100), 
                        (hud_x + 8, hud_y + y_offset), 
                        (hud_x + hud_w - 8, hud_y + y_offset), 1)
        
        # Performance info using cached stats (compact)
        y_offset += 8
        
        # Update cached performance stats less frequently
        if self.frame_counter % self.perf_stats_update_interval == 0:
            self.cached_perf_stats = self.map_control_system.get_performance_stats()
        
        # Show compact performance info using cached data
        info_lines = [
            f"Decay:{self.cached_perf_stats.get('decay_time', 5.0)}s R:{self.cached_perf_stats.get('player_control_radius', 50)}px",
            f"Poly T1:{self.cached_perf_stats.get('persistent_polygons', {}).get('team1', 0)} T2:{self.cached_perf_stats.get('persistent_polygons', {}).get('team2', 0)}"
        ]
        
        for line in info_lines:
            info_surface = self.font_small.render(line, True, (200, 200, 200))
            self.screen.blit(info_surface, (hud_x + 8, hud_y + y_offset))
            y_offset += 12

    def draw_collision_overlay(self, tracking_dimensions):
        """Draw collision boundaries and walls with vision blocking indicators"""
        if not self.collision_system or not self.show_collision_overlay:
            return
        
        # Get current agent positions and heights for context
        current_agents = self.current_frame_data.get('agents', []) if hasattr(self, 'current_frame_data') else []
        
        # Draw height areas with different colors and opacity based on vision blocking
        height_colors = {
            0: (139, 69, 19),   # FLOOR - brown
            1: (255, 165, 0),   # LOW - orange
            2: (255, 255, 0),   # MID - yellow
            3: (0, 255, 255),   # HIGH - cyan
            4: (128, 0, 128),   # CEILING - purple
        }
        
        # For each height area, determine if it blocks vision for any agent and check for layered areas
        processed_positions = set()
        
        for height_area in self.collision_system.height_areas:
            base_color = height_colors.get(int(height_area.height_level), (128, 128, 128))
            area_height_value = height_area.height_level.value
            
            # Check if this area blocks vision for any current agent
            blocks_vision_for_any = False
            partial_vision_for_any = False
            is_layered = False
            
            # Check for layered areas using center point of height area
            if hasattr(height_area.shape, 'points') and height_area.shape.points:
                if height_area.shape.shape_type == 'rectangle' and len(height_area.shape.points) >= 2:
                    # Calculate center of rectangle
                    if len(height_area.shape.points[0]) == 2:  # Points format: [[x1,y1], [x2,y2]]
                        x1, y1 = height_area.shape.points[0]
                        x2, y2 = height_area.shape.points[1]
                    else:  # Points format: [x1, y1, x2, y2]
                        x1, y1, x2, y2 = height_area.shape.points
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                elif height_area.shape.shape_type == 'polygon':
                    # Calculate centroid of polygon
                    center_x = sum(p[0] for p in height_area.shape.points) / len(height_area.shape.points)
                    center_y = sum(p[1] for p in height_area.shape.points) / len(height_area.shape.points)
                else:
                    center_x = center_y = 0
                
                # Check if this position has multiple height areas (layered)
                overlapping_areas = self.collision_system.get_overlapping_height_areas(center_x, center_y)
                is_layered = len(overlapping_areas) > 1
                
                # Add position to processed set to avoid duplicate layered indicators
                pos_key = (round(center_x/10)*10, round(center_y/10)*10)  # Grid-based to group nearby areas
                position_already_processed = pos_key in processed_positions
                processed_positions.add(pos_key)
            
            for agent in current_agents:
                agent_id = f"{agent.get('team', 'team1')}_{agent.get('track_id', 0)}"
                agent_height = self.collision_system.get_agent_height(agent_id)
                height_diff = area_height_value - agent_height.value
                
                if height_diff > 2:
                    blocks_vision_for_any = True
                    break
                elif height_diff == 2:
                    partial_vision_for_any = True
            
            # Adjust visualization based on vision blocking and layered areas
            if is_layered and not position_already_processed:
                # Purple tint for layered areas with special indicator
                color = (min(255, base_color[0] + 40), base_color[1], min(255, base_color[2] + 60))
                alpha = 100
                overlapping_heights = [area.height_level.name for area in overlapping_areas]
                label = f"{height_area.height_level.name.lower()} [LAYERED: {','.join(overlapping_heights)}]"
            elif blocks_vision_for_any:
                # Areas that block vision - darker and more opaque
                color = tuple(int(c * 0.6) for c in base_color)
                alpha = 120
                label = f"{height_area.height_level.name.lower()} [BLOCKS]"
            elif partial_vision_for_any:
                # Areas at max visible height - normal color but translucent
                color = base_color
                alpha = 90
                label = f"{height_area.height_level.name.lower()} [EDGE]"
            else:
                # Areas below or at agent height - very translucent
                color = base_color
                alpha = 60
                label = f"{height_area.height_level.name.lower()}"
            
            self.draw_collision_shape(height_area.shape, tracking_dimensions, color, label, alpha=alpha)
        
        # Legacy boundaries removed - only using height areas now
        
        # Draw walls (always block vision regardless of height)
        for wall in self.collision_system.walls:
            self.draw_collision_shape(wall, tracking_dimensions, (255, 0, 0), "wall", alpha=150)
        
        # Playable areas removed - height areas define playable map
        
        # Draw trigger zones
        for trigger in self.collision_system.trigger_zones:
            self.draw_collision_shape(trigger.trigger_shape, tracking_dimensions, (255, 0, 255), "trigger", alpha=60)
            
            # Draw trigger label
            if trigger.trigger_shape.shape_type == "polygon" and len(trigger.trigger_shape.points) > 0:
                center_x = sum(p[0] for p in trigger.trigger_shape.points) / len(trigger.trigger_shape.points)
                center_y = sum(p[1] for p in trigger.trigger_shape.points) / len(trigger.trigger_shape.points)
                display_center = self.map_to_display_coords(center_x, center_y)
                label_text = f"→{trigger.target_area}"
                text = self.font_small.render(label_text, True, (255, 0, 255))
                text_rect = text.get_rect(center=display_center)
                # Background for text using cached surface
                bg_width = text_rect.width + 4
                bg_height = text_rect.height + 2
                if bg_width <= 300 and bg_height <= 50:
                    bg = self.surface_cache['text_background']
                    bg.fill((0, 0, 0, 0))  # Clear with transparent
                    bg_surface = pygame.Surface((bg_width, bg_height), pygame.SRCALPHA)
                    bg_surface.fill((0, 0, 0, 150))
                    bg.blit(bg_surface, (0, 0))
                    self.screen.blit(bg, (text_rect.x - 2, text_rect.y - 1))
                else:
                    bg = pygame.Surface((bg_width, bg_height), pygame.SRCALPHA)
                    bg.fill((0, 0, 0, 150))
                    self.screen.blit(bg, (text_rect.x - 2, text_rect.y - 1))
                self.screen.blit(text, text_rect)
        
        # Draw vision range legend
        if current_agents and self.collision_system.height_areas:
            # Create a small legend showing vision range
            legend_x = self.display_w - 200
            legend_y = 10
            legend_surface = self.surface_cache['legend']
            legend_surface.fill((0, 0, 0, 180))
            self.screen.blit(legend_surface, (legend_x, legend_y))
            
            # Legend title
            title = self.font_small.render("Vision Range", True, (255, 255, 255))
            self.screen.blit(title, (legend_x + 10, legend_y + 5))
            
            # Vision range info
            info_lines = [
                "Can see: +2 levels",
                "Blocked by: >2 levels",
                "Edge vision: at +2"
            ]
            
            y_offset = 25
            for line in info_lines:
                text = self.font_small.render(line, True, (200, 200, 200))
                self.screen.blit(text, (legend_x + 10, legend_y + y_offset))
                y_offset += 18
    
    def draw_collision_shape(self, shape: CollisionShape, tracking_dimensions, color: tuple, 
                           shape_type: str, alpha: int = 100):
        """Draw a collision shape"""
        if shape.shape_type == 'rectangle':
            # Convert rectangle points to display coordinates
            if len(shape.points) == 2:
                x1, y1 = self.map_to_display_coords(*shape.points[0])
                x2, y2 = self.map_to_display_coords(*shape.points[1])
            else:
                x1, y1 = self.map_to_display_coords(shape.points[0], shape.points[1])
                x2, y2 = self.map_to_display_coords(shape.points[2], shape.points[3])
            
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            
            # Draw filled rectangle with transparency
            if alpha < 255:
                s = pygame.Surface((width, height), pygame.SRCALPHA)
                s.fill((*color, alpha))
                self.screen.blit(s, (min(x1, x2), min(y1, y2)))
            
            # Draw outline
            pygame.draw.rect(self.screen, color, 
                           (min(x1, x2), min(y1, y2), width, height), 2)
            
        elif shape.shape_type == 'polygon':
            # Convert polygon points to display coordinates
            display_points = []
            for point in shape.points:
                display_points.append(self.map_to_display_coords(*point))
            
            if len(display_points) >= 3:
                # Draw filled polygon with transparency
                if alpha < 255:
                    s = pygame.Surface((self.display_w, self.display_h), pygame.SRCALPHA)
                    pygame.draw.polygon(s, (*color, alpha), display_points)
                    self.screen.blit(s, (0, 0))
                
                # Draw outline
                pygame.draw.polygon(self.screen, color, display_points, 2)
        
        elif shape.shape_type == 'circle':
            # Convert circle to display coordinates
            center_x, center_y = self.map_to_display_coords(shape.center_x, shape.center_y)
            # Use radius directly since we have 1:1 mapping now
            scaled_radius = int(shape.radius)
            
            # Draw filled circle with transparency
            if alpha < 255:
                s = pygame.Surface((scaled_radius*2, scaled_radius*2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*color, alpha), (scaled_radius, scaled_radius), scaled_radius)
                self.screen.blit(s, (center_x - scaled_radius, center_y - scaled_radius))
            
            # Draw outline
            pygame.draw.circle(self.screen, color, (center_x, center_y), scaled_radius, 2)
        
        elif shape.shape_type == 'line':
            # Convert line endpoints to display coordinates
            if len(shape.points) >= 2:
                start_x, start_y = self.map_to_display_coords(*shape.points[0])
                end_x, end_y = self.map_to_display_coords(*shape.points[1])
                
                # Draw line with specified width
                line_width = 3 if alpha < 255 else 2
                pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, end_y), line_width)
                
                # Draw endpoints
                pygame.draw.circle(self.screen, color, (start_x, start_y), 3)
                pygame.draw.circle(self.screen, color, (end_x, end_y), 3)
        
        # Draw label
        if shape.name:
            if shape.shape_type == 'rectangle':
                label_x, label_y = (x1 + x2) // 2, (y1 + y2) // 2
            elif shape.shape_type == 'circle':
                label_x, label_y = center_x, center_y
            elif shape.shape_type == 'line' and len(shape.points) >= 2:
                # Label at midpoint of line
                start_x, start_y = self.map_to_display_coords(*shape.points[0])
                end_x, end_y = self.map_to_display_coords(*shape.points[1])
                label_x, label_y = (start_x + end_x) // 2, (start_y + end_y) // 2
            else:
                return  # Skip label for other shapes
            
            text = self.font_small.render(shape.name, True, color)
            text_rect = text.get_rect(center=(label_x, label_y))
            
            # Background for text using cached surface
            bg_width = text_rect.width + 4
            bg_height = text_rect.height + 2
            if bg_width <= 300 and bg_height <= 50:
                bg = self.surface_cache['text_background']
                bg.fill((0, 0, 0, 0))  # Clear with transparent
                bg_surface = pygame.Surface((bg_width, bg_height), pygame.SRCALPHA)
                bg_surface.fill((0, 0, 0, 150))
                bg.blit(bg_surface, (0, 0))
                self.screen.blit(bg, (text_rect.x - 2, text_rect.y - 1))
            else:
                bg = pygame.Surface((bg_width, bg_height), pygame.SRCALPHA)
                bg.fill((0, 0, 0, 150))
                self.screen.blit(bg, (text_rect.x - 2, text_rect.y - 1))
            self.screen.blit(text, text_rect)
    
    def _find_wall_intersection_point(self, start_x: float, start_y: float, end_x: float, end_y: float, wall: CollisionShape) -> Optional[Tuple[float, float]]:
        """Find precise intersection point between ray and wall"""
        if wall.shape_type == "rectangle":
            return self._find_rectangle_intersection(start_x, start_y, end_x, end_y, wall)
        elif wall.shape_type == "polygon":
            return self._find_polygon_intersection(start_x, start_y, end_x, end_y, wall)
        elif wall.shape_type == "circle":
            return self._find_circle_intersection(start_x, start_y, end_x, end_y, wall)
        elif wall.shape_type == "line":
            return self._find_line_intersection(start_x, start_y, end_x, end_y, wall)
        
        return None
    
    def _find_rectangle_intersection(self, start_x: float, start_y: float, end_x: float, end_y: float, wall: CollisionShape) -> Optional[Tuple[float, float]]:
        """Find intersection with rectangle wall edges"""
        if len(wall.points) == 2:
            rx1, ry1 = wall.points[0]
            rx2, ry2 = wall.points[1]
        else:
            rx1, ry1, rx2, ry2 = wall.points
        
        # Ensure rectangle bounds are correct
        left = min(rx1, rx2)
        right = max(rx1, rx2)
        top = min(ry1, ry2)
        bottom = max(ry1, ry2)
        
        # Check intersection with each edge of rectangle
        intersections = []
        
        # Top edge
        intersection = self._line_segment_intersection(start_x, start_y, end_x, end_y, left, top, right, top)
        if intersection:
            intersections.append(intersection)
        
        # Right edge
        intersection = self._line_segment_intersection(start_x, start_y, end_x, end_y, right, top, right, bottom)
        if intersection:
            intersections.append(intersection)
        
        # Bottom edge
        intersection = self._line_segment_intersection(start_x, start_y, end_x, end_y, right, bottom, left, bottom)
        if intersection:
            intersections.append(intersection)
        
        # Left edge
        intersection = self._line_segment_intersection(start_x, start_y, end_x, end_y, left, bottom, left, top)
        if intersection:
            intersections.append(intersection)
        
        # Return closest intersection to start point
        if intersections:
            closest = min(intersections, key=lambda p: (p[0] - start_x)**2 + (p[1] - start_y)**2)
            return closest
        
        return None
    
    def _find_polygon_intersection(self, start_x: float, start_y: float, end_x: float, end_y: float, wall: CollisionShape) -> Optional[Tuple[float, float]]:
        """Find intersection with polygon wall edges"""
        intersections = []
        
        for i in range(len(wall.points)):
            px1, py1 = wall.points[i]
            px2, py2 = wall.points[(i + 1) % len(wall.points)]
            
            intersection = self._line_segment_intersection(start_x, start_y, end_x, end_y, px1, py1, px2, py2)
            if intersection:
                intersections.append(intersection)
        
        # Return closest intersection to start point
        if intersections:
            closest = min(intersections, key=lambda p: (p[0] - start_x)**2 + (p[1] - start_y)**2)
            return closest
        
        return None
    
    def _find_circle_intersection(self, start_x: float, start_y: float, end_x: float, end_y: float, wall: CollisionShape) -> Optional[Tuple[float, float]]:
        """Find intersection with circular wall"""
        # Ray direction
        dx = end_x - start_x
        dy = end_y - start_y
        
        # Vector from ray start to circle center
        cx = wall.center_x - start_x
        cy = wall.center_y - start_y
        
        # Quadratic equation coefficients for ray-circle intersection
        a = dx * dx + dy * dy
        b = -2 * (cx * dx + cy * dy)
        c = cx * cx + cy * cy - wall.radius * wall.radius
        
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return None  # No intersection
        
        # Find the closest intersection point
        sqrt_discriminant = math.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2 * a)
        t2 = (-b + sqrt_discriminant) / (2 * a)
        
        # Choose the closest positive t (intersection in ray direction)
        valid_t = None
        if 0 <= t1 <= 1:
            valid_t = t1
        elif 0 <= t2 <= 1:
            valid_t = t2
        
        if valid_t is not None:
            intersection_x = start_x + valid_t * dx
            intersection_y = start_y + valid_t * dy
            return (intersection_x, intersection_y)
        
        return None
    
    def _find_circle_entry_exit(self, start_x: float, start_y: float, end_x: float, end_y: float, circle_shape: CollisionShape) -> list:
        """Find both entry and exit points for a ray passing through a circle"""
        intersections = []
        
        # Ray direction
        dx = end_x - start_x
        dy = end_y - start_y
        
        # Vector from ray start to circle center
        cx = circle_shape.center_x - start_x
        cy = circle_shape.center_y - start_y
        
        # Quadratic equation coefficients for ray-circle intersection
        a = dx * dx + dy * dy
        b = -2 * (cx * dx + cy * dy)
        c = cx * cx + cy * cy - circle_shape.radius * circle_shape.radius
        
        discriminant = b * b - 4 * a * c
        
        if discriminant >= 0:
            sqrt_discriminant = math.sqrt(discriminant)
            t1 = (-b - sqrt_discriminant) / (2 * a)
            t2 = (-b + sqrt_discriminant) / (2 * a)
            
            # Add both intersection points if they're in the ray direction
            for t in [t1, t2]:
                if 0 <= t <= 1:
                    intersection_x = start_x + t * dx
                    intersection_y = start_y + t * dy
                    intersections.append((intersection_x, intersection_y))
        
        return intersections

    def _find_line_intersection(self, start_x: float, start_y: float, end_x: float, end_y: float, wall: CollisionShape) -> Optional[Tuple[float, float]]:
        """Find intersection with line wall"""
        if len(wall.points) >= 2:
            lx1, ly1 = wall.points[0]
            lx2, ly2 = wall.points[1]
            
            return self._line_segment_intersection(start_x, start_y, end_x, end_y, lx1, ly1, lx2, ly2)
        
        return None
    
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
    
    def _utility_blocks_vision(self, utility) -> bool:
        """Check if a utility object blocks vision (e.g., smokes, walls)"""
        utility_type = utility.get('type', '').lower()
        
        # Define which utility types block vision
        vision_blocking_types = {
            # All smoke types (both circle and line)
            'smoke-circle', # 'smoke-line', 
            # 'smoke',
            # Wall types
            'wall-rect', 'wall-circle', 'wall-line', 'wall',
            # Agent-specific abilities that block vision
            'sage_wall', 'viper_wall', 'omen_smoke', 'brimstone_smoke', 'astra_smoke',
            'cypher_cage', 'breach_stun', 'viper_poison_cloud', 'phoenix_wall',
            'harbor_cove', 'harbor_high_tide', 'harbor_cascade'
        }
        
        # Also check if the utility type contains key blocking words
        blocking_keywords = [# 'smoke', 
                           'wall', 'cage', 'cloud', 'barrier']
        for keyword in blocking_keywords:
            if keyword in utility_type:
                return True
        
        return utility_type in vision_blocking_types
    
    def _find_utility_intersection(self, start_x: float, start_y: float, angle: float, 
                                 utility, tracking_dimensions) -> Optional[Tuple[float, float]]:
        """Find intersection point between vision ray and utility object"""
        try:
            # Get utility position and size
            util_pos = utility.get('position', {})
            util_x = util_pos.get('x', 0)
            util_y = util_pos.get('y', 0)
            
            # Utility coordinates are already in map space when passed to vision cone calculator
            # The vision cone calculator receives utilities converted to map coordinates
            util_map_x = util_x  # Already in map coordinates
            util_map_y = util_y  # Already in map coordinates
            
            # Calculate radius in map coordinates (same logic as drawing but in map space)
            bbox = utility.get('bbox')
            if bbox:
                # bbox is already in map coordinates when passed to vision cone calculator
                map_x1, map_y1 = bbox[0], bbox[1]
                map_x2, map_y2 = bbox[2], bbox[3]
                width = abs(map_x2 - map_x1)
                height = abs(map_y2 - map_y1)
                # Use same radius calculation as drawing: max(width, height) // 2
                map_radius = max(width, height) // 2
            else:
                # Fallback to default radius if no bbox
                map_radius = utility.get('radius', 30)  # Already in map coordinates
            
            # Ray direction
            dx = math.cos(angle)
            dy = math.sin(angle)
            
            # Vector from ray start to utility center (all in map coordinates)
            cx = util_map_x - start_x
            cy = util_map_y - start_y
            
            # Use precise circle intersection calculation
            # Quadratic equation coefficients for ray-circle intersection
            ray_length = math.sqrt(dx*dx + dy*dy)
            if ray_length == 0:
                return None
            
            # Normalize ray direction
            dx_norm = dx / ray_length
            dy_norm = dy / ray_length
            
            # Project circle center onto ray direction
            projection = cx * dx_norm + cy * dy_norm
            
            # If utility is behind the ray start, no intersection
            if projection < 0:
                return None
            
            # Find closest point on ray to utility center
            closest_x = start_x + projection * dx_norm
            closest_y = start_y + projection * dy_norm
            
            # Distance from utility center to ray
            distance_to_ray = math.sqrt((closest_x - util_map_x)**2 + (closest_y - util_map_y)**2)
            
            # Check if ray intersects utility circle
            if distance_to_ray <= map_radius:
                # Calculate exact intersection point using pythagorean theorem
                # Distance from closest point to actual intersection
                intersection_offset = math.sqrt(map_radius**2 - distance_to_ray**2)
                
                # Find intersection point (closer to ray start)
                intersection_distance = projection - intersection_offset
                
                if intersection_distance >= 0:
                    intersection_x = start_x + intersection_distance * dx_norm
                    intersection_y = start_y + intersection_distance * dy_norm
                    return (intersection_x, intersection_y)
            
            return None
            
        except (KeyError, TypeError, ValueError):
            # If utility data is malformed, skip intersection
            return None
    
    def draw_agent(self, agent_data, tracking_dimensions, utilities=None):
        """Draw agent with visualization"""
        try:
            pos_x = agent_data['position']['x']
            pos_y = agent_data['position']['y']
            # Convert tracking coordinates to map coordinates, then to display coordinates
            map_x, map_y = self.tracking_to_map_coords(pos_x, pos_y)
            x, y = self.map_to_display_coords(map_x, map_y)
        except (KeyError, TypeError) as e:
            print(f"Error processing agent position: {e}")
            return
        
        agent_name = agent_data.get('agent', 'Unknown')
        team = agent_data.get('team', 'team1')
        orientation = agent_data.get('orientation', 0)
        track_id = agent_data.get('track_id', 0)
        confidence = agent_data.get('confidence', 1.0)
        
        # Validate position against collision system and update triggers
        position_valid = True
        trigger_transitions = {}
        if self.collision_system:
            # Use map coordinates directly (collision system now uses actual map dimensions)
            
            # Update agent position and check for trigger transitions
            agent_id = f"{team}_{track_id}"
            trigger_transitions = self.collision_system.update_agent_position(agent_id, map_x, map_y)
            
            # Use trigger-aware position validation
            position_valid = self.collision_system.is_position_valid_with_triggers(agent_id, map_x, map_y)
        
        # Validate confidence (inline for performance)
        confidence = confidence if isinstance(confidence, (int, float)) else 1.0
        
        color = self.colors.get(team, (255, 255, 255))
        
        # Modify color if position is invalid
        if not position_valid:
            # Make invalid agents more transparent and add red tint
            color = tuple(int(c * 0.6) for c in color)
            color = (min(255, color[0] + 50), color[1], color[2])  # Add red tint
        
        # Smooth movement
        smooth_key = f"{team}_{track_id}"
        if smooth_key in self.agent_positions:
            old_x, old_y = self.agent_positions[smooth_key]
            x = int(old_x + (x - old_x) * self.smoothing_factor)
            y = int(old_y + (y - old_y) * self.smoothing_factor)
        self.agent_positions[smooth_key] = (x, y)
        
        # Add to trail
        self.agent_trails[smooth_key].append((x, y))
        
        # Draw trail (simplified)
        trail_points = self.agent_trails[smooth_key]
        if len(trail_points) > 1:
            # Draw only the last few segments for performance
            recent_points = list(trail_points)[-5:]  # Last 5 points only
            for i in range(1, len(recent_points)):
                alpha = i / len(recent_points) * 0.4
                trail_color = tuple(int(c * alpha) for c in color)
                pygame.draw.line(self.screen, trail_color, recent_points[i-1], recent_points[i], 2)
        
        # Draw agent
        radius = int(16 * confidence)
        
        # Simple glow effect for high confidence (optimized)
        if confidence > 0.8:
            # Single outer circle for glow instead of multiple surfaces
            pygame.gfxdraw.aacircle(self.screen, x, y, radius + 2, tuple(int(c * 0.5) for c in color))
        
        # Main circle
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)
        pygame.gfxdraw.aacircle(self.screen, x, y, radius, (255, 255, 255))
        
        # Draw agent name above the player
        if agent_name and agent_name != 'Unknown':
            font = pygame.font.Font(None, 20)  # Small font for agent name
            text_surface = font.render(agent_name, True, (255, 255, 255))
            text_rect = text_surface.get_rect()
            text_rect.centerx = x
            text_rect.bottom = y - radius - 5  # Position above the agent circle
            
            # Draw black background for better readability
            bg_rect = text_rect.inflate(4, 2)
            pygame.draw.rect(self.screen, (0, 0, 0, 180), bg_rect)
            
            # Draw the text
            self.screen.blit(text_surface, text_rect)
        
        # Draw 75-degree vision cone using unified calculator with caching
        if orientation != 0 and self.vision_cone_calculator:
            agent_id = f"{team}_{track_id}"
            
            # Check if we need to recalculate the vision cone
            if self._should_recalculate_vision_cone(agent_id, map_x, map_y, orientation):
                # Convert utilities to map coordinates only once per frame
                if not hasattr(self, '_cached_map_utilities') or self._cached_map_utilities is None:
                    self._cached_map_utilities = []
                    if utilities:
                        for utility in utilities:
                            converted_utility = utility.copy()
                            
                            # Convert position to map coordinates
                            if 'position' in utility:
                                util_pos = utility['position']
                                util_map_x, util_map_y = self.tracking_to_map_coords(
                                    util_pos['x'], util_pos['y']
                                )
                                converted_utility['position'] = {'x': util_map_x, 'y': util_map_y}
                            
                            # Convert bbox to map coordinates too
                            if 'bbox' in utility:
                                bbox = utility['bbox']
                                map_x1, map_y1 = self.tracking_to_map_coords(bbox[0], bbox[1])
                                map_x2, map_y2 = self.tracking_to_map_coords(bbox[2], bbox[3])
                                converted_utility['bbox'] = [map_x1, map_y1, map_x2, map_y2]
                            
                            self._cached_map_utilities.append(converted_utility)
                
                # Calculate new vision cone
                cone_points_map = self.vision_cone_calculator.calculate_vision_cone(
                    agent_id, map_x, map_y, orientation, self._cached_map_utilities
                )
                
                # Cache the results
                self.vision_cone_cache[agent_id] = {
                    'map_x': map_x,
                    'map_y': map_y,
                    'orientation': orientation,
                    'cone_points_map': cone_points_map
                }
            else:
                # Use cached vision cone
                cone_points_map = self.vision_cone_cache[agent_id]['cone_points_map']
            
            # Convert to display coordinates
            if self.map_dimensions:
                cone_points_display = self.vision_cone_calculator.get_cone_for_display(
                    cone_points_map, self.map_dimensions, (self.display_w, self.display_h)
                )
            else:
                # Direct mapping if no scaling needed
                cone_points_display = [(int(x), int(y)) for x, y in cone_points_map]
            
            # Draw semi-transparent cone
            if len(cone_points_display) > 2:
                # REUSE pre-allocated surface instead of creating new one
                self.cone_surface.fill((0, 0, 0, 0))  # Clear with transparent
                
                # Draw filled polygon with transparency
                cone_color = (*color, 50)
                pygame.draw.polygon(self.cone_surface, cone_color, cone_points_display)
                
                # Draw cone boundary lines
                boundary_color = (*color, 100)
                pygame.draw.lines(self.cone_surface, boundary_color, True, cone_points_display[1:], 1)
                
                # Blit the cone surface
                self.screen.blit(self.cone_surface, (0, 0))
            
            # Draw orientation arrow (always visible)
            arrow_length = radius + 12
            angle_rad = math.radians(orientation)
            end_x = int(x + arrow_length * math.cos(angle_rad))
            end_y = int(y + arrow_length * math.sin(angle_rad))
            
            pygame.draw.line(self.screen, (255, 255, 255), (x, y), (end_x, end_y), 3)
            
            # Arrowhead
            arrow_size = 6
            left_x = int(end_x + arrow_size * math.cos(angle_rad + 2.6))
            left_y = int(end_y + arrow_size * math.sin(angle_rad + 2.6))
            right_x = int(end_x + arrow_size * math.cos(angle_rad - 2.6))
            right_y = int(end_y + arrow_size * math.sin(angle_rad - 2.6))
            
            pygame.draw.polygon(self.screen, (255, 255, 255), 
                              [(end_x, end_y), (left_x, left_y), (right_x, right_y)])
        elif orientation != 0:
            # Fallback if no vision calculator available
            # Just draw the orientation arrow
            angle_rad = math.radians(orientation)
            arrow_length = radius + 12
            end_x = int(x + arrow_length * math.cos(angle_rad))
            end_y = int(y + arrow_length * math.sin(angle_rad))
            
            pygame.draw.line(self.screen, (255, 255, 255), (x, y), (end_x, end_y), 3)
    
    def draw_hud(self, frame_number, timestamp, stats):
        """Draw HUD with cached surface"""
        # Use pre-allocated surface
        hud_surface = self.surface_cache['hud_panel']
        hud_surface.fill((0, 0, 0, 200))
        self.screen.blit(hud_surface, (10, 10))
        
        # Title (smaller)
        title = self.font_medium.render("Tracker", True, (255, 255, 255))
        self.screen.blit(title, (15, 15))
        
        # Frame info (compact)
        info_lines = [
            f"F:{frame_number} T:{timestamp:.1f}s",
            f"Agents:{stats.get('agents', 0)} Utils:{stats.get('utilities', 0)}",
            f"Conf:{stats.get('avg_confidence', 0):.2f}"
        ]
        
        y_pos = 40
        for line in info_lines:
            text = self.font_small.render(line, True, (200, 200, 200))
            self.screen.blit(text, (15, y_pos))
            y_pos += 18
        
        # Utility breakdown (compact)
        util_types = stats.get('utility_types', {})
        if util_types:
            y_pos += 5
            breakdown_title = self.font_small.render("Utils:", True, (255, 255, 255))
            self.screen.blit(breakdown_title, (15, y_pos))
            y_pos += 15
            
            for util_type, count in list(util_types.items())[:3]:  # Show top 3
                text = self.font_small.render(f" {util_type}:{count}", True, (180, 180, 180))
                self.screen.blit(text, (15, y_pos))
                y_pos += 14
    
    def render_frame(self, frame_data, frame_number, tracking_dimensions):
        """Render a complete frame with performance profiling"""
        frame_start = time.perf_counter()
        
        # Store current frame data for vision cone calculations
        self.current_frame_data = frame_data
        
        # Reset utility cache for this frame
        self._cached_map_utilities = None
        
        # Clean up vision cone cache for agents no longer present
        current_agent_ids = set()
        for agent in frame_data.get('agents', []):
            agent_id = f"{agent.get('team', 'team1')}_{agent.get('track_id', 0)}"
            current_agent_ids.add(agent_id)
        
        # Remove cached data for agents not in current frame
        if hasattr(self, 'vision_cone_cache'):
            cached_ids = set(self.vision_cone_cache.keys())
            stale_ids = cached_ids - current_agent_ids
            for stale_id in stale_ids:
                del self.vision_cone_cache[stale_id]
        
        # Clean up agent trails and positions for agents no longer present
        trail_ids = set(self.agent_trails.keys())
        position_ids = set(self.agent_positions.keys())
        stale_trail_ids = trail_ids - current_agent_ids
        stale_position_ids = position_ids - current_agent_ids
        
        for stale_id in stale_trail_ids:
            del self.agent_trails[stale_id]
        for stale_id in stale_position_ids:
            del self.agent_positions[stale_id]
        
        # Clear screen
        if self.background:
                self.screen.blit(self.background, (0, 0))
        else:
            self.screen.fill((20, 20, 20))            
        timestamp = frame_data.get('timestamp', 0)
        agents = frame_data.get('agents', [])
        utilities = frame_data.get('utilities', [])
        
        # Calculate current time based on frame timestamp
        # This ensures consistent timing when playing back recorded data
        self.current_time = self.start_time + timestamp
        
        # Pre-calculate coordinate conversion factors for this frame (cache if not set)
        if self.coord_scale_x is None or self.coord_scale_y is None:
            if self.map_dimensions and tracking_dimensions:
                tracking_w, tracking_h = tracking_dimensions
                map_w, map_h = self.map_dimensions
                self.coord_scale_x = map_w / tracking_w
                self.coord_scale_y = map_h / tracking_h
            else:
                self.coord_scale_x = 1.0
                self.coord_scale_y = 1.0
        
        # Update map control every frame (frame skipping removed)
        self.frame_counter += 1
        if self.map_control_system:
            # Convert agents to map coordinates for collision-aware system
            map_agents = []
            for agent in agents:
                if 'position' in agent:
                    pos_x, pos_y = agent['position']['x'], agent['position']['y']
                    map_x, map_y = self.tracking_to_map_coords(pos_x, pos_y)
                    map_agent = agent.copy()
                    map_agent['position'] = {'x': map_x, 'y': map_y}
                    map_agents.append(map_agent)
            
            # Convert utilities to map coordinates for control system
            map_utilities = []
            for utility in utilities:
                converted_utility = utility.copy()
                if 'position' in utility:
                    util_x, util_y = utility['position']['x'], utility['position']['y']
                    map_util_x, map_util_y = self.tracking_to_map_coords(util_x, util_y)
                    converted_utility['position'] = {'x': map_util_x, 'y': map_util_y}
                
                # Convert bbox if present
                if 'bbox' in utility and len(utility['bbox']) >= 4:
                    x1, y1, x2, y2 = utility['bbox']
                    map_x1, map_y1 = self.tracking_to_map_coords(x1, y1)
                    map_x2, map_y2 = self.tracking_to_map_coords(x2, y2)
                    converted_utility['bbox'] = [map_x1, map_y1, map_x2, map_y2]
                
                map_utilities.append(converted_utility)
            
            self.map_control_system.update_team_vision(map_agents, self.current_time, map_utilities)
            # Cache the results for use in skipped frames
            self.cached_control_data = self.map_control_system.get_control_visualization_data()
            self.cached_control_percentages = self.map_control_system.get_control_percentages()
        
        # Calculate basic statistics (optimized)
        agent_count = len(agents)
        utility_count = len(utilities)
        
        # Only calculate detailed stats every 10 frames for performance
        if True:
            all_confidences = [
                agent.get('confidence', 1.0) for agent in agents 
                if isinstance(agent.get('confidence'), (int, float))
            ] + [
                util.get('confidence', 1.0) for util in utilities 
                if isinstance(util.get('confidence'), (int, float))
            ]
            avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
            utility_types = defaultdict(int)
            for util in utilities:
                utility_types[util.get('type', 'unknown')] += 1
            
            # Cache for reuse in non-calculation frames
            self.cached_stats = {
                'avg_confidence': avg_confidence,
                'utility_types': utility_types
            }
        else:
            # Use cached values for non-calculation frames
            cached = getattr(self, 'cached_stats', {'avg_confidence': 0, 'utility_types': defaultdict(int)})
            avg_confidence = cached['avg_confidence']
            utility_types = cached['utility_types']
        
        stats = {
            'agents': agent_count,
            'utilities': utility_count,
            'avg_confidence': avg_confidence,
            'utility_types': utility_types
        }
        
        # Draw map control overlay (behind everything except background)
        self.draw_map_control_overlay(tracking_dimensions)
        
        # Draw collision overlay (behind everything)
        self.draw_collision_overlay(tracking_dimensions)
        
        # Draw utilities first (background layer)
        for utility in utilities:
            self.draw_utility(utility, tracking_dimensions)
        
        # Draw agents (foreground layer)
        for agent in agents:
            self.draw_agent(agent, tracking_dimensions, utilities)
        
        
        # Draw HUD
        self.draw_hud(frame_number, timestamp, stats)
        
        # Draw map control HUD using cached data
        if self.map_control_system and self.cached_control_percentages:
            self.draw_control_hud(self.cached_control_percentages)
        
        # Update display
        pygame.display.flip()
            
        # Track total frame time
        frame_end = time.perf_counter()
        frame_time = frame_end - frame_start
        self.perf_stats.setdefault("total_frame_time", deque(maxlen=50)).append(frame_time)
    
    def generate_post_tracking_stats(self, json_path: str):
        """Generate comprehensive statistics and readable reports after visualization completes"""
        print(f"\nGenerating post-tracking statistics from {json_path}:")
        try:
            # Generate CSV statistics
            team_stats, agent_stats = analyze_tracker_output(json_path)
            print(f"Statistics generated successfully!")
            print(f"CSV files exported to:")
            print(f"stats_output/agent_stats.csv")
            print(f"stats_output/team_stats.csv")
            
            # Note: Readable reports generation was removed
            # CSV files contain all necessary statistics data
            print(f"\nStatistics exported to CSV files in stats_output/")
            
            return team_stats, agent_stats
        except Exception as e:
            print(f"Error generating statistics: {e}")
            return None, None

    def play_tracking_data(self, tracking_data, fps=60):
        """Play tracking data with controls"""
        # Detect tracking dimensions from JSON data
        tracking_dimensions = self._detect_tracking_dimensions(tracking_data)
        print(f"Detected tracking dimensions: {tracking_dimensions[0]}x{tracking_dimensions[1]}")
        
        # Access the 'frames' dictionary, falling back to the top-level for old formats
        frame_source = tracking_data.get('frames', tracking_data)
        if not isinstance(frame_source, dict):
            print("Error: Frame data is not in a valid dictionary format.")
            return

        # Sort frames
        sorted_frames = sorted(frame_source.items(), key=lambda x: int(x[0]))
        
        clock = pygame.time.Clock()
        running = True
        paused = False
        frame_idx = 0
        
        # Frame rate limiter with consistent timing
        target_fps = fps
        frame_time = 5  # Fixed 5ms per frame (200 FPS equivalent)
        
        print("\nControls:")
        print("  SPACE: Pause/Resume")
        print("  LEFT/RIGHT: Navigate frames (when paused)")
        print("  R: Reset to beginning")
        print("  C: Toggle collision overlay")
        print("  M: Toggle map control overlay")
        print("  ESC: Quit")
        print(f"  Running at {fps} FPS")
        
        while running and frame_idx < len(sorted_frames):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        print(f"{'Paused' if paused else 'Resumed'}")
                    elif event.key == pygame.K_RIGHT and paused:
                        frame_idx = min(frame_idx + 1, len(sorted_frames) - 1)
                    elif event.key == pygame.K_LEFT and paused:
                        frame_idx = max(frame_idx - 1, 0)
                    elif event.key == pygame.K_r:
                        frame_idx = 0
                        print("Reset to beginning")
                    elif event.key == pygame.K_c:
                        self.show_collision_overlay = not self.show_collision_overlay
                        print(f"Collision overlay: {'ON' if self.show_collision_overlay else 'OFF'}")
                    elif event.key == pygame.K_m:
                        self.show_map_control = not self.show_map_control
                        print(f"Map control overlay: {'ON' if self.show_map_control else 'OFF'}")
            
            # Measure actual frame time
            frame_start = pygame.time.get_ticks()
            
            # Render current frame
            if frame_idx < len(sorted_frames):
                frame_number, frame_data = sorted_frames[frame_idx]
                self.render_frame(frame_data, int(frame_number), tracking_dimensions)
                            
            if not paused:
                # Simple frame advancement
                frame_idx += 1
                
                # Clear caches more frequently to prevent memory bloat
                if frame_idx % 100 == 0:
                    # Clear vision cone spatial cache
                    if self.vision_cone_calculator:
                        self.vision_cone_calculator.clear_spatial_cache()
                    
                    # Clear vision cone cache
                    if hasattr(self, 'vision_cone_cache'):
                        self.vision_cone_cache.clear()
                    
                    # Clear map control cached data
                    self.cached_control_data = None
                    self.cached_control_percentages = {}
                    
                    # Limit agent trails to prevent unbounded growth
                    max_agents = 10  # Keep trails for max 10 most recent agents
                    if len(self.agent_trails) > max_agents:
                        # Keep only the most recently updated trails
                        sorted_agents = sorted(self.agent_trails.items(), 
                                             key=lambda x: len(x[1]), reverse=True)
                        self.agent_trails = dict(sorted_agents[:max_agents])
                    
                    # Clear player statistics cache in map control system if it exists
                    if self.map_control_system and hasattr(self.map_control_system, 'player_stats'):
                        # Keep only recent statistics (last 500 data points per player)
                        for agent_id, stats in self.map_control_system.player_stats.items():
                            if len(stats.timestamps) > 500:
                                # Clear older half of the data
                                mid_point = len(stats.timestamps) // 2
                                for attr in ['timestamps', 'positions', 'heights', 'orientations',
                                           'control_area_history', 'vision_area_history', 'contested_area_history']:
                                    if hasattr(stats, attr):
                                        deque_obj = getattr(stats, attr)
                                        # Create new deque with recent data
                                        recent_data = list(deque_obj)[mid_point:]
                                        deque_obj.clear()
                                        deque_obj.extend(recent_data)
            
            # Maintain consistent frame rate
            frame_end = pygame.time.get_ticks()
            frame_duration = frame_end - frame_start
            if frame_duration < frame_time:
                pygame.time.wait(int(frame_time - frame_duration))
            else:
                # Use regular clock tick as fallback for heavy frames
                clock.tick(target_fps)
        
        pygame.quit()

def load_tracking_data(json_path):
    """Load and validate tracking data"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded tracking data from: {json_path}")
        
        # Handle both new and old data formats for logging
        frame_source = data.get('frames', data)
        if 'video_dimensions' in data:
            print(f"  Video Dimensions: {data['video_dimensions']}")

        print(f"  Total frames: {len(frame_source)}")
        
        # Sample first frame to show structure
        if frame_source and isinstance(frame_source, dict):
            first_frame_key = next(iter(frame_source))
            first_frame_data = frame_source[first_frame_key]
            print(f"  Agents in first frame: {len(first_frame_data.get('agents', []))}")
            print(f"  Utilities in first frame: {len(first_frame_data.get('utilities', []))}")
        
        return data
    
    except Exception as e:
        print(f"Error loading tracking data: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Valorant Track Visualizer")
    parser.add_argument("--json", type=str, default="track-output/out_data.json",
                        help="Path to tracking data JSON file")
    parser.add_argument("--minimap", type=str, default="maps/icebox.png",
                        help="Path to minimap image (auto-detect from maps/ if not provided)")
    parser.add_argument("--display-size", type=int, nargs=2, default=[1000, 1000],
                        help="Display window size [width height]")
    parser.add_argument("--fps", type=int, default=1000,
                        help="Playback frame rate (60 FPS for smooth playback)")
    parser.add_argument("--fullscreen", action="store_true",
                        help="Run in fullscreen mode")
    parser.add_argument("--map-name", type=str,
                        help="Map name for collision detection (e.g., icebox, ascent)")
    parser.add_argument("--create-collision-config", action="store_true",
                        help="Create collision configuration for the specified map")
    parser.add_argument("--decay-time", type=float, default=100.0,
                        help="Control decay time in seconds (default: 20.0, higher = slower decay)")
    parser.add_argument("--permanent", action="store_true",
                        help="Enable permanent control mode - control persists until overwritten by opposing team")
    parser.add_argument("--takeover-threshold", type=float, default=0.7,
                        help="Minimum strength needed to overwrite opponent control (default: 0.7)")
    parser.add_argument("--contest-threshold", type=float, default=0.2,
                        help="Strength difference threshold for contested state (default: 0.2)")
    parser.add_argument("--min-control-strength", type=float, default=0.4,
                        help="Minimum strength to establish initial control (default: 0.4)")
    
    args = parser.parse_args()
    
    # Handle collision config creation
    if args.create_collision_config:
        if not args.map_name:
            print("Error: --map-name required when creating collision config")
            return 1
        create_collision_config_tool()
        return 0
    
    # Load tracking data
    tracking_data = load_tracking_data(args.json)
    if not tracking_data:
        return 1
    
    # Auto-detect minimap if not provided
    minimap_path = args.minimap
    if not minimap_path:
        maps_dir = Path("maps")
        if maps_dir.exists():
            map_files = list(maps_dir.glob("*.png"))
            if map_files:
                minimap_path = str(map_files[0])  # Use first map found
                print(f"Auto-detected minimap: {minimap_path}")
    
    # Create and run visualizer
    visualizer = TrackVisualizer(
        display_size=tuple(args.display_size),
        minimap_path=minimap_path,
        fullscreen=args.fullscreen,
        map_name=args.map_name,
        decay_time=args.decay_time,
        permanent_control=args.permanent,
        takeover_threshold=args.takeover_threshold,
        contest_threshold=args.contest_threshold,
        min_control_strength=args.min_control_strength
    )
    
    visualizer.play_tracking_data(tracking_data, fps=args.fps)
    
    # Generate statistics after visualization completes
    visualizer.generate_post_tracking_stats(args.json)
    
    return 0

if __name__ == "__main__":
    exit(main())