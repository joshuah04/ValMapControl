#!/usr/bin/env python3
"""
Collision-Aware Grid Map Control System
Integrates map collision system with fast grid control for optimal performance
"""

import numpy as np
import math
from typing import Dict, List, Tuple
from .map_collision_system import MapCollisionSystem, HeightLevel
from .vision_cone_calculator import VisionConeCalculator
from .grid.control_status import ControlStatus
from .grid.player_control_stats import PlayerControlStats

class CollisionAwareGridControl:
    """Grid-based map control that respects collision boundaries and heights"""
    
    def __init__(self, collision_system: MapCollisionSystem, 
                 grid_size: int = 8, vision_range: float = 400.0, 
                 player_control_radius: float = 50.0, decay_time: float = 5.0,
                 cone_angle: float = 75.0, permanent_control: bool = False,
                 takeover_threshold: float = 0.6, contest_threshold: float = 0.3,
                 min_control_strength: float = 0.4):
        """
        Initialize collision-aware grid control
        
        Args:
            collision_system: The map collision system to integrate with
            grid_size: Grid cell size in pixels
            vision_range: Maximum vision distance
            player_control_radius: Control radius around players
            decay_time: Control decay time in seconds (only used if permanent_control=False)
            cone_angle: Vision cone angle in degrees
            permanent_control: If True, control persists until overwritten by opposing team
            takeover_threshold: Minimum strength needed to overwrite opponent control
            contest_threshold: Strength difference threshold for contested state
            min_control_strength: Minimum strength to establish initial control
        """
        self.collision_system = collision_system
        self.grid_size = grid_size
        self.vision_range = vision_range
        self.player_control_radius = player_control_radius
        self.decay_time = decay_time
        self.cone_angle = cone_angle
        
        # Competitive control parameters
        self.permanent_control = permanent_control
        self.takeover_threshold = takeover_threshold
        self.contest_threshold = contest_threshold
        self.min_control_strength = min_control_strength
        
        # Initialize vision cone calculator
        self.vision_cone_calculator = VisionConeCalculator(
            collision_system, vision_range, cone_angle
        )
        
        # Get map dimensions from collision system
        self.map_width = collision_system.coordinate_system_size
        self.map_height = collision_system.coordinate_system_size  # Assume square for now
        
        # Calculate grid dimensions
        self.grid_width = (self.map_width + grid_size - 1) // grid_size
        self.grid_height = (self.map_height + grid_size - 1) // grid_size
        
        # Pre-compute collision grid - which cells are playable
        self.playable_grid = self._compute_playable_grid()
        
        # Pre-compute height grid - what height level each cell is at
        self.height_grid = self._compute_height_grid()
        
        # Control grids (only for playable cells)
        self.control_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        self.team1_strength = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.team2_strength = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.timestamp_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float64)
        
        # Additional grids for competitive control
        if self.permanent_control:
            # Track current frame activity to distinguish permanent vs active control
            self.team1_current_activity = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
            self.team2_current_activity = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
            # Track permanent control ownership
            self.permanent_control_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
            self.permanent_strength_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        
        # Performance stats
        self.playable_cells = np.sum(self.playable_grid)
        
        # Individual player statistics tracking
        self.player_stats: Dict[str, PlayerControlStats] = {}
        self.enable_player_stats = True  # Can be disabled for performance
        
        print(f"CollisionAwareGridControl initialized:")
        print(f"  Grid: {self.grid_width}x{self.grid_height} cells")
        print(f"  Playable cells: {self.playable_cells}/{self.grid_width * self.grid_height}")
        print(f"  Playable ratio: {self.playable_cells / (self.grid_width * self.grid_height):.2%}")
        print(f"  Player stats tracking: {'ON' if self.enable_player_stats else 'OFF'}")
    
    def _compute_playable_grid(self) -> np.ndarray:
        """Pre-compute which grid cells are in height areas (playable map areas)"""
        playable_grid = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                # Convert grid coordinates to map coordinates (center of cell)
                map_x = col * self.grid_size + self.grid_size // 2
                map_y = row * self.grid_size + self.grid_size // 2
                
                # Check if this position is in any height area (these define the playable map)
                is_playable = False
                for height_area in self.collision_system.height_areas:
                    if height_area.contains_point(map_x, map_y):
                        is_playable = True
                        break
                
                playable_grid[row, col] = is_playable
        
        return playable_grid
    
    def _compute_height_grid(self) -> np.ndarray:
        """Pre-compute height level for each grid cell"""
        height_grid = np.full((self.grid_height, self.grid_width), HeightLevel.FLOOR, dtype=np.uint8)
        
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                if not self.playable_grid[row, col]:
                    continue  # Skip non-playable cells
                
                # Convert grid coordinates to map coordinates
                map_x = col * self.grid_size + self.grid_size // 2
                map_y = row * self.grid_size + self.grid_size // 2
                
                # Find the height level at this position
                for height_area in self.collision_system.height_areas:
                    if height_area.contains_point(map_x, map_y):
                        height_grid[row, col] = height_area.height_level.value
                        break  # Use first match (assumes no overlapping height areas)
        
        return height_grid
    
    def update_team_vision(self, agents: List[Dict], current_time: float, utilities: List = None):
        """Update team vision and control based on agent positions"""
        if self.permanent_control:
            # Reset current frame activity
            self.team1_current_activity.fill(0.0)
            self.team2_current_activity.fill(0.0)
        else:
            # Apply decay to existing control (only in non-permanent mode)
            self._apply_decay(current_time)
            
            # Create temporary vision masks to track current frame vision
            self.team1_current_vision = np.zeros((self.grid_height, self.grid_width), dtype=bool)
            self.team2_current_vision = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        
        # Process each agent
        for agent in agents:
            try:
                # Get agent position and properties
                pos_x = agent['position']['x']
                pos_y = agent['position']['y']
                team = agent.get('team', 'team1')
                agent_id = f"{team}_{agent.get('track_id', 0)}"
                orientation = agent.get('orientation', 0)
                
                # Get agent height from collision system
                agent_height = self.collision_system.get_agent_height(agent_id)
                
                # Calculate vision cone first (includes utilities blocking)
                vision_polygon = None
                if self.vision_cone_calculator:
                    # Calculate vision cone points with utilities (or empty list if None)
                    utilities_list = utilities if utilities is not None else []
                    cone_points = self.vision_cone_calculator.calculate_vision_cone(
                        agent_id, pos_x, pos_y, orientation, utilities_list
                    )
                    
                    # Convert to Shapely polygon
                    if cone_points and len(cone_points) >= 3:
                        vision_polygon = self.vision_cone_calculator.get_cone_polygon(cone_points)
                
                # Track individual player statistics if enabled
                if self.enable_player_stats:
                    # Initialize player stats if first time seeing this agent
                    if agent_id not in self.player_stats:
                        self.player_stats[agent_id] = PlayerControlStats(agent_id, team)
                    
                    # Calculate player's current control and vision areas (before grid updates)
                    player_control_area, player_vision_area, player_contested_area = self._calculate_player_areas(
                        pos_x, pos_y, orientation, team, agent_height, agent_id, current_time
                    )
                    
                    # Update player statistics
                    self.player_stats[agent_id].update(
                        timestamp=current_time,
                        position=(pos_x, pos_y),
                        height=agent_height,
                        orientation=orientation,
                        control_area=player_control_area,
                        vision_area=player_vision_area,
                        contested_area=player_contested_area
                    )
                
                # Update control based on vision cone polygon
                self._update_agent_control(pos_x, pos_y, team, agent_height, current_time, vision_polygon)
                
            except (KeyError, TypeError) as e:
                continue  # Skip malformed agent data
        
        # Apply competitive control logic if in permanent mode
        if self.permanent_control:
            self._apply_competitive_control(current_time)
        else:
            # In non-permanent mode, resolve contested cells immediately
            self._resolve_contested_cells(current_time)
    
    def _update_agent_control(self, pos_x: float, pos_y: float, team: str, 
                            agent_height: HeightLevel, current_time: float, vision_polygon=None):
        """Update control based on both player radius AND vision cone (two separate systems)"""
        if vision_polygon is None or vision_polygon.is_empty:
            # Fallback: use basic collision-system line-of-sight if vision polygon unavailable
            self._update_agent_control_fallback(pos_x, pos_y, team, agent_height, current_time)
            return
        
        # Apply TWO separate control systems:
        # 1. Player radius control (circular area around player)
        # 2. Vision cone control (all visible areas regardless of distance)
        
        self._apply_player_radius_control(pos_x, pos_y, team, agent_height, current_time)
        self._apply_vision_cone_control(pos_x, pos_y, team, agent_height, current_time, vision_polygon)
    
    def _apply_player_radius_control(self, pos_x: float, pos_y: float, team: str, 
                                   agent_height: HeightLevel, current_time: float):
        """Apply control in circular radius around player (independent of vision)"""
        # Convert to grid coordinates
        grid_col = int(pos_x / self.grid_size)
        grid_row = int(pos_y / self.grid_size)
        
        # Calculate control radius in grid cells (larger search area)
        radius_cells = int(self.player_control_radius / self.grid_size) + 2
        
        # Check all cells in square area, but only grant control to those within circular radius
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                r = grid_row + dr
                c = grid_col + dc
                
                # Check bounds and playability
                if (0 <= r < self.grid_height and 0 <= c < self.grid_width and 
                    self.playable_grid[r, c]):
                    
                    # Check if agent can control this height level
                    cell_height = HeightLevel(self.height_grid[r, c])
                    if agent_height.can_see(agent_height, cell_height):
                        
                        # Calculate actual distance from agent to cell center
                        cell_center_x = c * self.grid_size + self.grid_size // 2
                        cell_center_y = r * self.grid_size + self.grid_size // 2
                        dist = math.sqrt((cell_center_x - pos_x)**2 + (cell_center_y - pos_y)**2)
                        
                        # Only grant control if within circular radius
                        if dist <= self.player_control_radius:
                            # Distance-based strength (closer = stronger)
                            strength = max(0.1, 1.0 - dist / self.player_control_radius)
                            
                            if self.permanent_control:
                                self._set_cell_activity(r, c, team, strength)
                            else:
                                self._set_cell_control(r, c, team, strength, current_time)
                                # Track current vision for contested cell resolution
                                if team == 'team1':
                                    self.team1_current_vision[r, c] = True
                                else:
                                    self.team2_current_vision[r, c] = True
    
    def _apply_vision_cone_control(self, pos_x: float, pos_y: float, team: str,
                                 agent_height: HeightLevel, current_time: float, vision_polygon):
        """Apply control to all cells within vision cone (independent of distance from player)"""
        from shapely.geometry import Point
        
        # Get bounding box of vision polygon to limit search area
        bounds = vision_polygon.bounds  # (minx, miny, maxx, maxy)
        min_col = max(0, int(bounds[0] / self.grid_size))
        max_col = min(self.grid_width - 1, int(bounds[2] / self.grid_size))
        min_row = max(0, int(bounds[1] / self.grid_size))
        max_row = min(self.grid_height - 1, int(bounds[3] / self.grid_size))
        
        # Check all cells within vision cone bounding box
        for r in range(min_row, max_row + 1):
            for c in range(min_col, max_col + 1):
                
                # Check bounds and playability
                if (0 <= r < self.grid_height and 0 <= c < self.grid_width and 
                    self.playable_grid[r, c]):
                    
                    # Check if agent can control this height level
                    cell_height = HeightLevel(self.height_grid[r, c])
                    if agent_height.can_see(agent_height, cell_height):
                        
                        # Check if cell center is within vision cone polygon
                        cell_center_x = c * self.grid_size + self.grid_size // 2
                        cell_center_y = r * self.grid_size + self.grid_size // 2
                        cell_point = Point(cell_center_x, cell_center_y)
                        
                        if vision_polygon.contains(cell_point):
                            # Vision control strength (could be different from radius control)
                            strength = 0.8  # Slightly less than direct radius control
                            
                            if self.permanent_control:
                                # In permanent mode, use max of existing and new strength
                                current_strength = self.team1_current_activity[r, c] if team == 'team1' else self.team2_current_activity[r, c]
                                final_strength = max(current_strength, strength)
                                self._set_cell_activity(r, c, team, final_strength)
                            else:
                                # In non-permanent mode, use max of existing and new strength
                                current_strength = self.team1_strength[r, c] if team == 'team1' else self.team2_strength[r, c]
                                final_strength = max(current_strength, strength)
                                self._set_cell_control(r, c, team, final_strength, current_time)
                                # Track current vision for contested cell resolution
                                if team == 'team1':
                                    self.team1_current_vision[r, c] = True
                                else:
                                    self.team2_current_vision[r, c] = True
    
    def _update_agent_control_fallback(self, pos_x: float, pos_y: float, team: str, 
                                     agent_height: HeightLevel, current_time: float):
        """Fallback control method using basic collision system line-of-sight"""
        # Convert to grid coordinates
        grid_col = int(pos_x / self.grid_size)
        grid_row = int(pos_y / self.grid_size)
        
        # Calculate control radius in grid cells
        radius_cells = int(self.player_control_radius / self.grid_size) + 1
        
        # Update control in radius around agent using collision system
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                r = grid_row + dr
                c = grid_col + dc
                
                # Check bounds and playability
                if (0 <= r < self.grid_height and 0 <= c < self.grid_width and 
                    self.playable_grid[r, c]):
                    
                    # Check if agent can control this height level
                    cell_height = HeightLevel(self.height_grid[r, c])
                    if agent_height.can_see(agent_height, cell_height):
                        # Calculate distance and strength
                        dist = math.sqrt(dr*dr + dc*dc) * self.grid_size
                        if dist <= self.player_control_radius:
                            # Check line-of-sight through collision system (walls only, no utilities)
                            cell_center_x = c * self.grid_size + self.grid_size // 2
                            cell_center_y = r * self.grid_size + self.grid_size // 2
                            
                            # Use collision system to check basic line-of-sight
                            if self.collision_system.can_agents_see_each_other(
                                f"agent_{team}", f"cell_{r}_{c}", 
                                (pos_x, pos_y), (cell_center_x, cell_center_y)):
                                
                                strength = 1.0
                                if self.permanent_control:
                                    self._set_cell_activity(r, c, team, strength)
                                else:
                                    self._set_cell_control(r, c, team, strength, current_time)
                                    # Track current vision for contested cell resolution
                                    if team == 'team1':
                                        self.team1_current_vision[r, c] = True
                                    else:
                                        self.team2_current_vision[r, c] = True
    
    
    def _set_cell_activity(self, row: int, col: int, team: str, strength: float):
        """Set current frame activity for a cell (used in permanent control mode)"""
        team_num = 1 if team == 'team1' else 2
        
        if team_num == 1:
            self.team1_current_activity[row, col] = max(self.team1_current_activity[row, col], strength)
        else:
            self.team2_current_activity[row, col] = max(self.team2_current_activity[row, col], strength)
    
    def _apply_competitive_control(self, current_time: float):
        """Apply competitive control logic - overwrite opponent control when strength is sufficient"""
        playable_mask = self.playable_grid
        
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                if not playable_mask[row, col]:
                    continue
                
                t1_activity = self.team1_current_activity[row, col]
                t2_activity = self.team2_current_activity[row, col]
                current_control = ControlStatus(self.permanent_control_grid[row, col])
                current_strength = self.permanent_strength_grid[row, col]
                
                # Determine new control based on competitive rules
                new_control, new_strength = self._determine_competitive_control(
                    t1_activity, t2_activity, current_control, current_strength
                )
                
                # Update permanent control
                self.permanent_control_grid[row, col] = new_control.value
                self.permanent_strength_grid[row, col] = new_strength
                
                # Update display grids
                self.team1_strength[row, col] = new_strength if new_control == ControlStatus.TEAM1 else 0.0
                self.team2_strength[row, col] = new_strength if new_control == ControlStatus.TEAM2 else 0.0
                self.timestamp_grid[row, col] = current_time
        
        # Update control grid based on permanent control
        self._update_control_grid_from_permanent()
    
    def _determine_competitive_control(self, t1_activity: float, t2_activity: float, 
                                     current_control: ControlStatus, current_strength: float) -> Tuple[ControlStatus, float]:
        """Determine new control status based on competitive rules"""
        
        # Check if either team has meaningful activity
        t1_has_control = t1_activity >= self.min_control_strength
        t2_has_control = t2_activity >= self.min_control_strength
        
        # No activity - maintain current control
        if not t1_has_control and not t2_has_control:
            return current_control, current_strength
        
        # Only one team has activity
        if t1_has_control and not t2_has_control:
            # Team 1 wants control
            if current_control == ControlStatus.TEAM2:
                # Need to overwrite Team 2 - check takeover threshold
                if t1_activity >= self.takeover_threshold:
                    return ControlStatus.TEAM1, t1_activity
                else:
                    # Not strong enough to overwrite, remain Team 2
                    return current_control, current_strength
            else:
                # No opposition or same team - establish/maintain Team 1 control
                return ControlStatus.TEAM1, max(t1_activity, current_strength)
        
        elif t2_has_control and not t1_has_control:
            # Team 2 wants control
            if current_control == ControlStatus.TEAM1:
                # Need to overwrite Team 1 - check takeover threshold
                if t2_activity >= self.takeover_threshold:
                    return ControlStatus.TEAM2, t2_activity
                else:
                    # Not strong enough to overwrite, remain Team 1
                    return current_control, current_strength
            else:
                # No opposition or same team - establish/maintain Team 2 control
                return ControlStatus.TEAM2, max(t2_activity, current_strength)
        
        else:
            # Both teams have activity - determine based on strength difference
            strength_diff = abs(t1_activity - t2_activity)
            
            if strength_diff <= self.contest_threshold:
                # Too close - contested
                return ControlStatus.CONTESTED, max(t1_activity, t2_activity)
            else:
                # Clear winner
                if t1_activity > t2_activity:
                    # Team 1 is stronger
                    if current_control == ControlStatus.TEAM2 and t1_activity < self.takeover_threshold:
                        # Not strong enough to overwrite Team 2
                        return current_control, current_strength
                    return ControlStatus.TEAM1, t1_activity
                else:
                    # Team 2 is stronger
                    if current_control == ControlStatus.TEAM1 and t2_activity < self.takeover_threshold:
                        # Not strong enough to overwrite Team 1
                        return current_control, current_strength
                    return ControlStatus.TEAM2, t2_activity
    
    def _update_control_grid_from_permanent(self):
        """Update display control grid based on permanent control"""
        playable_mask = self.playable_grid
        
        # Copy permanent control to display grid
        self.control_grid = np.where(playable_mask, self.permanent_control_grid, ControlStatus.NEUTRAL.value)
    
    def _resolve_contested_cells(self, current_time: float):
        """Resolve contested cells based on current vision (non-permanent mode)"""
        playable_mask = self.playable_grid
        
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                if not playable_mask[row, col]:
                    continue
                
                # Check if this cell is currently contested
                if self.control_grid[row, col] == ControlStatus.CONTESTED.value:
                    # Check current vision status
                    t1_has_vision = self.team1_current_vision[row, col]
                    t2_has_vision = self.team2_current_vision[row, col]
                    
                    # If only one team has vision now, give them control
                    if t1_has_vision and not t2_has_vision:
                        # Team 1 takes over the contested cell
                        self.control_grid[row, col] = ControlStatus.TEAM1.value
                        # Maintain existing team 1 strength, clear team 2
                        self.team2_strength[row, col] = 0.0
                        self.timestamp_grid[row, col] = current_time
                    elif t2_has_vision and not t1_has_vision:
                        # Team 2 takes over the contested cell
                        self.control_grid[row, col] = ControlStatus.TEAM2.value
                        # Maintain existing team 2 strength, clear team 1
                        self.team1_strength[row, col] = 0.0
                        self.timestamp_grid[row, col] = current_time
                    elif not t1_has_vision and not t2_has_vision:
                        # Neither team has vision - apply faster decay or immediate neutral
                        # Option 1: Immediate neutral
                        self.control_grid[row, col] = ControlStatus.NEUTRAL.value
                        self.team1_strength[row, col] = 0.0
                        self.team2_strength[row, col] = 0.0
                        
                    # If both teams still have vision, remain contested

    def _set_cell_control(self, row: int, col: int, team: str, strength: float, 
                         current_time: float):
        """Set control for a specific grid cell"""
        team_num = 1 if team == 'team1' else 2
        
        if team_num == 1:
            self.team1_strength[row, col] = max(self.team1_strength[row, col], strength)
        else:
            self.team2_strength[row, col] = max(self.team2_strength[row, col], strength)
        
        self.timestamp_grid[row, col] = current_time
        
        # Update control status
        t1_str = self.team1_strength[row, col]
        t2_str = self.team2_strength[row, col]
        strength_threshold = 0.1
        
        # Contested only when BOTH teams have meaningful control
        if t1_str > strength_threshold and t2_str > strength_threshold:
            self.control_grid[row, col] = ControlStatus.CONTESTED.value
        # Single team control when only ONE team has meaningful control
        elif t1_str > strength_threshold and t2_str <= strength_threshold:
            self.control_grid[row, col] = ControlStatus.TEAM1.value
        elif t2_str > strength_threshold and t1_str <= strength_threshold:
            self.control_grid[row, col] = ControlStatus.TEAM2.value
        else:
            self.control_grid[row, col] = ControlStatus.NEUTRAL.value
    
    def _apply_decay(self, current_time: float):
        """Apply decay to control strength over time"""
        # Only apply decay to playable cells
        playable_mask = self.playable_grid
        
        # Calculate time deltas
        time_delta = current_time - self.timestamp_grid
        time_delta = np.where(playable_mask, time_delta, 0)
        
        # Apply exponential decay
        decay_factor = np.exp(-time_delta / self.decay_time)
        decay_factor = np.where(playable_mask, decay_factor, 0)
        
        self.team1_strength *= decay_factor
        self.team2_strength *= decay_factor
        
        # Update control grid based on new strengths
        self._update_control_grid()
    
    def _update_control_grid(self):
        """Update control grid based on current team strengths"""
        # Only update playable cells
        playable_mask = self.playable_grid
        
        # Define strength threshold for meaningful control
        strength_threshold = 0.1
        
        # Determine which teams have meaningful control at each cell
        team1_has_control = (self.team1_strength > strength_threshold) & playable_mask
        team2_has_control = (self.team2_strength > strength_threshold) & playable_mask
        
        # Contested cells: BOTH teams have vision (meaningful control)
        contested = team1_has_control & team2_has_control
        
        # Single team control: Only ONE team has vision
        team1_only = team1_has_control & ~team2_has_control
        team2_only = team2_has_control & ~team1_has_control
        
        # Neutral: Neither team has meaningful control
        neutral = ~team1_has_control & ~team2_has_control & playable_mask
        
        # Update control grid
        self.control_grid[contested] = ControlStatus.CONTESTED.value
        self.control_grid[team1_only] = ControlStatus.TEAM1.value
        self.control_grid[team2_only] = ControlStatus.TEAM2.value
        self.control_grid[neutral] = ControlStatus.NEUTRAL.value
        self.control_grid[~playable_mask] = ControlStatus.NEUTRAL.value
    
    def get_control_visualization_data(self) -> Dict:
        """Get control data for visualization (only playable cells)"""
        pixel_control = {}
        
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                if not self.playable_grid[row, col]:
                    continue  # Skip non-playable cells
                
                # Convert grid coordinates to map pixel coordinates
                map_x = col * self.grid_size + self.grid_size // 2
                map_y = row * self.grid_size + self.grid_size // 2
                
                control_status = ControlStatus(self.control_grid[row, col])
                strength = max(self.team1_strength[row, col], self.team2_strength[row, col])
                
                if strength > 0.05:  # Only show cells with meaningful control
                    pixel_control[(map_x, map_y)] = {
                        'status': control_status,
                        'strength': strength
                    }
        
        return {
            'pixel_control': pixel_control,
            'grid_size': self.grid_size
        }
    
    def get_control_percentages(self) -> Dict[str, float]:
        """Calculate control percentages (only for playable areas)"""
        playable_mask = self.playable_grid
        total_playable = np.sum(playable_mask)
        
        if total_playable == 0:
            return {'team1': 0, 'team2': 0, 'contested': 0, 'neutral': 100}
        
        team1_cells = np.sum((self.control_grid == ControlStatus.TEAM1.value) & playable_mask)
        team2_cells = np.sum((self.control_grid == ControlStatus.TEAM2.value) & playable_mask)
        contested_cells = np.sum((self.control_grid == ControlStatus.CONTESTED.value) & playable_mask)
        neutral_cells = total_playable - team1_cells - team2_cells - contested_cells
        
        return {
            'team1': (team1_cells / total_playable) * 100,
            'team2': (team2_cells / total_playable) * 100,
            'contested': (contested_cells / total_playable) * 100,
            'neutral': (neutral_cells / total_playable) * 100
        }
    
    def get_control_colors(self) -> Dict[ControlStatus, Tuple[int, int, int]]:
        """Get colors for different control statuses"""
        return {
            ControlStatus.TEAM1: (0, 255, 0),      # Green
            ControlStatus.TEAM2: (255, 0, 0),      # Red  
            ControlStatus.CONTESTED: (255, 255, 0), # Yellow
            ControlStatus.NEUTRAL: (128, 128, 128)   # Gray
        }
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            'decay_time': self.decay_time,
            'player_control_radius': self.player_control_radius,
            'grid_size': self.grid_size,
            'playable_cells': int(self.playable_cells),
            'total_cells': int(self.grid_width * self.grid_height),
            'playable_ratio': float(self.playable_cells / (self.grid_width * self.grid_height))
        }
    
    def _calculate_player_areas(self, pos_x: float, pos_y: float, orientation: float, 
                               team: str, agent_height: HeightLevel, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        """Calculate control, vision, and contested areas for a specific player"""
        control_area = 0
        vision_area = 0
        contested_area = 0
        
        # Calculate control area around player position
        grid_x = int(pos_x / self.grid_size)
        grid_y = int(pos_y / self.grid_size)
        radius_cells = int(self.player_control_radius / self.grid_size) + 1
        
        # Check circular area around player
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                check_y = grid_y + dy
                check_x = grid_x + dx
                
                # Check grid bounds
                if not (0 <= check_y < self.grid_height and 0 <= check_x < self.grid_width):
                    continue
                
                # Skip non-playable cells
                if not self.playable_grid[check_y, check_x]:
                    continue
                
                # Calculate distance
                cell_center_x = check_x * self.grid_size + self.grid_size // 2
                cell_center_y = check_y * self.grid_size + self.grid_size // 2
                dist = math.sqrt((cell_center_x - pos_x)**2 + (cell_center_y - pos_y)**2)
                
                # Check if player can control this height level
                cell_height = self.height_grid[check_y, check_x]
                if not self._can_control_height(agent_height, cell_height):
                    continue
                
                # Within control radius
                if dist <= self.player_control_radius:
                    control_area += 1
                    
                    # Check if contested with other team
                    if team == 'team1':
                        if self.team2_strength[check_y, check_x] > 0:
                            contested_area += 1
                    else:
                        if self.team1_strength[check_y, check_x] > 0:
                            contested_area += 1
        
        # Calculate vision area from vision cone if orientation is available
        if orientation != 0:
            try:
                cone_points = self.vision_cone_calculator.calculate_vision_cone(
                    agent_id, pos_x, pos_y, orientation, utilities=[]
                )
                # Only count cells for statistics - don't apply to grid to avoid interference
                vision_area = self._count_cells_in_polygon(cone_points, agent_height)
            except Exception:
                vision_area = control_area  # Fallback to control area
        else:
            vision_area = control_area
        
        return control_area, vision_area, contested_area
    
    def _count_cells_in_polygon(self, polygon_points: List[Tuple[float, float]], 
                               agent_height: HeightLevel) -> int:
        """Count grid cells within a polygon (e.g., vision cone)"""
        if len(polygon_points) < 3:
            return 0
        
        count = 0
        
        # Use bounding box to limit search area
        min_x = min(p[0] for p in polygon_points)
        max_x = max(p[0] for p in polygon_points)
        min_y = min(p[1] for p in polygon_points)
        max_y = max(p[1] for p in polygon_points)
        
        start_grid_x = max(0, int(min_x / self.grid_size))
        end_grid_x = min(self.grid_width, int(max_x / self.grid_size) + 1)
        start_grid_y = max(0, int(min_y / self.grid_size))
        end_grid_y = min(self.grid_height, int(max_y / self.grid_size) + 1)
        
        for grid_y in range(start_grid_y, end_grid_y):
            for grid_x in range(start_grid_x, end_grid_x):
                # Skip non-playable cells
                if not self.playable_grid[grid_y, grid_x]:
                    continue
                
                # Check height compatibility
                cell_height = self.height_grid[grid_y, grid_x]
                if not self._can_control_height(agent_height, cell_height):
                    continue
                
                # Check if cell center is inside polygon
                cell_center_x = grid_x * self.grid_size + self.grid_size // 2
                cell_center_y = grid_y * self.grid_size + self.grid_size // 2
                
                if self._point_in_polygon(cell_center_x, cell_center_y, polygon_points):
                    count += 1
        
        return count
    
    def _point_in_polygon(self, x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
        """Check if a point is inside a polygon using ray casting algorithm"""
        if len(polygon) < 3:
            return False
        
        inside = False
        j = len(polygon) - 1
        
        for i in range(len(polygon)):
            if ((polygon[i][1] > y) != (polygon[j][1] > y)) and \
               (x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) / (polygon[j][1] - polygon[i][1]) + polygon[i][0]):
                inside = not inside
            j = i
        
        return inside
    
    def _can_control_height(self, agent_height: HeightLevel, cell_height: HeightLevel) -> bool:
        """Check if an agent can control a cell at a specific height"""
        return agent_height.can_see(agent_height, cell_height)
    
    # Individual Player Statistics API Methods
    
    def get_player_stats_at_time(self, agent_id: str, timestamp: float) -> Dict:
        """Get individual player statistics at a specific timestamp"""
        if agent_id in self.player_stats:
            return self.player_stats[agent_id].get_stats_at_time(timestamp)
        return {}
    
    def get_player_stats_in_interval(self, agent_id: str, start_time: float, end_time: float) -> Dict:
        """Get individual player statistics within a time interval"""
        if agent_id in self.player_stats:
            return self.player_stats[agent_id].get_stats_in_interval(start_time, end_time)
        return {}
    
    def get_all_players_stats_at_time(self, timestamp: float) -> Dict[str, Dict]:
        """Get all players' statistics at a specific timestamp"""
        return {agent_id: stats.get_stats_at_time(timestamp) 
                for agent_id, stats in self.player_stats.items()}
    
    def get_all_players_stats_in_interval(self, start_time: float, end_time: float) -> Dict[str, Dict]:
        """Get all players' statistics within a time interval"""
        return {agent_id: stats.get_stats_in_interval(start_time, end_time) 
                for agent_id, stats in self.player_stats.items()}
    
    def get_team_stats_at_time(self, team: str, timestamp: float) -> Dict[str, Dict]:
        """Get specific team's player statistics at a timestamp"""
        return {agent_id: stats.get_stats_at_time(timestamp) 
                for agent_id, stats in self.player_stats.items() 
                if stats.team == team}
    
    def get_team_stats_in_interval(self, team: str, start_time: float, end_time: float) -> Dict[str, Dict]:
        """Get specific team's player statistics within a time interval"""
        return {agent_id: stats.get_stats_in_interval(start_time, end_time) 
                for agent_id, stats in self.player_stats.items() 
                if stats.team == team}
    
    def export_player_stats_csv(self, filepath: str, start_time: float = None, end_time: float = None):
        """Export individual player statistics to CSV file"""
        import csv
        
        # Determine time range
        if start_time is None or end_time is None:
            all_timestamps = []
            for stats in self.player_stats.values():
                if stats.timestamps:
                    all_timestamps.extend(list(stats.timestamps))
            
            if not all_timestamps:
                print("No player statistics data to export")
                return
            
            if start_time is None:
                start_time = min(all_timestamps)
            if end_time is None:
                end_time = max(all_timestamps)
        
        # Collect all player data
        export_data = []
        for agent_id, stats in self.player_stats.items():
            interval_stats = stats.get_stats_in_interval(start_time, end_time)
            if interval_stats['frames_in_interval'] > 0:
                export_data.append({
                    'agent_id': agent_id,
                    'team': stats.team,
                    'avg_control_area': interval_stats['avg_control_area'],
                    'max_control_area': interval_stats['max_control_area'],
                    'avg_vision_area': interval_stats['avg_vision_area'],
                    'avg_contested_area': interval_stats['avg_contested_area'],
                    'total_control_cells': interval_stats['total_control_cells'],
                    'control_consistency': interval_stats['control_consistency'],
                    'avg_control_efficiency': interval_stats['avg_control_efficiency'],
                    'frames_tracked': interval_stats['frames_in_interval'],
                    'time_interval': f"{start_time:.2f}-{end_time:.2f}s"
                })
        
        # Write to CSV
        if export_data:
            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = export_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(export_data)
            print(f"Exported player statistics to {filepath}")
        else:
            print("No player statistics data to export in the specified time range")
    
    def get_player_stats_summary(self) -> Dict:
        """Get summary of all tracked players and their statistics"""
        summary = {
            'total_players_tracked': len(self.player_stats),
            'players_by_team': {'team1': 0, 'team2': 0},
            'total_frames_tracked': 0,
            'time_range': {'start': float('inf'), 'end': 0}
        }
        
        for agent_id, stats in self.player_stats.items():
            summary['players_by_team'][stats.team] += 1
            summary['total_frames_tracked'] += len(stats.timestamps)
            
            if stats.timestamps:
                summary['time_range']['start'] = min(summary['time_range']['start'], stats.timestamps[0])
                summary['time_range']['end'] = max(summary['time_range']['end'], stats.timestamps[-1])
        
        if summary['time_range']['start'] == float('inf'):
            summary['time_range'] = {'start': 0, 'end': 0}
        
        return summary