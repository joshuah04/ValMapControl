from typing import Dict, Tuple
from collections import deque
from ..map_collision_system import HeightLevel

class PlayerControlStats:
    """Individual player control statistics tracking"""
    
    def __init__(self, agent_id: str, team: str, max_history_length: int = 1000):
        """Initialize player control statistics
        
        Args:
            agent_id: Unique identifier for the agent
            team: Team name the agent belongs to
            max_history_length: Maximum number of history entries to store
        """
        self.agent_id = agent_id
        self.team = team
        self.max_history_length = max_history_length
        
        # Time-series data
        self.timestamps = deque(maxlen=max_history_length)
        self.positions = deque(maxlen=max_history_length)  # (x, y) tuples
        self.heights = deque(maxlen=max_history_length)    # HeightLevel values
        self.orientations = deque(maxlen=max_history_length)  # Orientation degrees
        
        # Control statistics over time
        self.control_area_history = deque(maxlen=max_history_length)      # Cells controlled per frame
        self.vision_area_history = deque(maxlen=max_history_length)       # Cells with vision per frame
        self.contested_area_history = deque(maxlen=max_history_length)    # Cells contested per frame
        
        # Aggregate statistics
        self.total_control_time = 0.0    # Total time controlling areas
        self.max_control_area = 0        # Maximum cells controlled at once
        self.total_cells_controlled = 0  # Cumulative cells controlled over all frames
        
        # Performance metrics
        self.avg_control_per_second = 0.0
        self.control_efficiency = 0.0    # Control area vs vision area ratio
        
        # Last frame data for calculations
        self.last_timestamp = 0.0
        self.last_control_area = 0
        self.last_vision_area = 0
        self.last_contested_area = 0
    
    def update(self, timestamp: float, position: Tuple[float, float], height: HeightLevel,
               orientation: float, control_area: int, vision_area: int, contested_area: int):
        """Update player statistics with new frame data"""
        # Store time-series data
        self.timestamps.append(timestamp)
        self.positions.append(position)
        self.heights.append(height)
        self.orientations.append(orientation)
        
        # Store control statistics
        self.control_area_history.append(control_area)
        self.vision_area_history.append(vision_area)
        self.contested_area_history.append(contested_area)
        
        # Update aggregate statistics
        if control_area > 0:
            if self.last_timestamp > 0:
                time_delta = timestamp - self.last_timestamp
                self.total_control_time += time_delta
            
        self.max_control_area = max(self.max_control_area, control_area)
        self.total_cells_controlled += control_area
        
        # Calculate performance metrics
        if len(self.timestamps) > 1:
            total_time = self.timestamps[-1] - self.timestamps[0]
            if total_time > 0:
                self.avg_control_per_second = sum(self.control_area_history) / total_time
        
        if vision_area > 0:
            self.control_efficiency = control_area / vision_area
        
        # Update last frame data
        self.last_timestamp = timestamp
        self.last_control_area = control_area
        self.last_vision_area = vision_area
        self.last_contested_area = contested_area
    
    def get_stats_at_time(self, timestamp: float) -> Dict:
        """Get player statistics at a specific timestamp"""
        if not self.timestamps:
            return self._empty_stats()
        
        # Find closest timestamp
        timestamps_list = list(self.timestamps)
        closest_idx = min(range(len(timestamps_list)), 
                         key=lambda i: abs(timestamps_list[i] - timestamp))
        
        if closest_idx < len(self.control_area_history):
            return {
                'agent_id': self.agent_id,
                'team': self.team,
                'timestamp': timestamps_list[closest_idx],
                'position': self.positions[closest_idx],
                'height': self.heights[closest_idx].name if hasattr(self.heights[closest_idx], 'name') else str(self.heights[closest_idx]),
                'orientation': self.orientations[closest_idx],
                'control_area': self.control_area_history[closest_idx],
                'vision_area': self.vision_area_history[closest_idx],
                'contested_area': self.contested_area_history[closest_idx],
                'control_efficiency': self.control_area_history[closest_idx] / max(1, self.vision_area_history[closest_idx])
            }
        
        return self._empty_stats()
    
    def get_stats_in_interval(self, start_time: float, end_time: float) -> Dict:
        """Get aggregated player statistics within a time interval"""
        if not self.timestamps:
            return self._empty_interval_stats()
        
        # Find indices within time range
        timestamps_list = list(self.timestamps)
        indices = [i for i, t in enumerate(timestamps_list) if start_time <= t <= end_time]
        
        if not indices:
            return self._empty_interval_stats()
        
        # Calculate interval statistics
        control_areas = [self.control_area_history[i] for i in indices]
        vision_areas = [self.vision_area_history[i] for i in indices]
        contested_areas = [self.contested_area_history[i] for i in indices]
        
        return {
            'agent_id': self.agent_id,
            'team': self.team,
            'interval': {'start': start_time, 'end': end_time},
            'frames_in_interval': len(indices),
            'avg_control_area': sum(control_areas) / len(control_areas),
            'max_control_area': max(control_areas),
            'avg_vision_area': sum(vision_areas) / len(vision_areas),
            'avg_contested_area': sum(contested_areas) / len(contested_areas),
            'total_control_cells': sum(control_areas),
            'control_consistency': len([c for c in control_areas if c > 0]) / len(control_areas),
            'avg_control_efficiency': sum(c / max(1, v) for c, v in zip(control_areas, vision_areas)) / len(control_areas),
            'positions': [self.positions[i] for i in indices],
            'heights': [self.heights[i].name if hasattr(self.heights[i], 'name') else str(self.heights[i]) for i in indices]
        }
    
    def _empty_stats(self) -> Dict:
        """Return empty stats structure"""
        return {
            'agent_id': self.agent_id,
            'team': self.team,
            'timestamp': 0.0,
            'position': (0, 0),
            'height': 'FLOOR',
            'orientation': 0.0,
            'control_area': 0,
            'vision_area': 0,
            'contested_area': 0,
            'control_efficiency': 0.0
        }
    
    def _empty_interval_stats(self) -> Dict:
        """Return empty interval stats structure"""
        return {
            'agent_id': self.agent_id,
            'team': self.team,
            'interval': {'start': 0, 'end': 0},
            'frames_in_interval': 0,
            'avg_control_area': 0.0,
            'max_control_area': 0,
            'avg_vision_area': 0.0,
            'avg_contested_area': 0.0,
            'total_control_cells': 0,
            'control_consistency': 0.0,
            'avg_control_efficiency': 0.0,
            'positions': [],
            'heights': []
        }
