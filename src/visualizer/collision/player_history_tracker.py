import time
from typing import List, Tuple, Dict, Optional
from collections import deque
from .height_level import HeightLevel

class PlayerHistoryTracker:
    """Tracks player position and height history for layered area transitions"""
    
    def __init__(self, max_history_length: int = 5):
        """Initialize player history tracker
        
        Args:
            max_history_length: Maximum position history entries per player
        """
        self.max_history_length = max_history_length
        # {agent_id: deque([(x, y, height, timestamp), ...])}
        self.position_history: Dict[str, deque] = {}
    
    def update_position(self, agent_id: str, x: float, y: float, height: HeightLevel):
        """Update position history for an agent"""
        timestamp = time.time()
        
        if agent_id not in self.position_history:
            self.position_history[agent_id] = deque(maxlen=self.max_history_length)
        
        # Add new position to history
        self.position_history[agent_id].append((x, y, height, timestamp))
    
    def get_recent_positions(self, agent_id: str, count: int = 3) -> List[Tuple[float, float, HeightLevel, float]]:
        """Get the most recent positions for an agent"""
        if agent_id not in self.position_history:
            return []
        
        history = list(self.position_history[agent_id])
        return history[-count:] if len(history) >= count else history
    
    def get_previous_height(self, agent_id: str) -> Optional[HeightLevel]:
        """Get the most recent height from history (excluding current position)"""
        if agent_id not in self.position_history or len(self.position_history[agent_id]) < 2:
            return None
        
        # Get second most recent position (previous position)
        history = list(self.position_history[agent_id])
        return history[-2][2]  # height is at index 2
    
    def get_dominant_recent_height(self, agent_id: str, frames_back: int = 3) -> Optional[HeightLevel]:
        """Get the most common height from recent positions"""
        recent_positions = self.get_recent_positions(agent_id, frames_back)
        if not recent_positions:
            return None
        
        # Count height occurrences
        height_counts = {}
        for _, _, height, _ in recent_positions:
            height_counts[height] = height_counts.get(height, 0) + 1
        
        # Return most common height
        return max(height_counts.keys(), key=lambda h: height_counts[h])
    
    def has_recent_height_history(self, agent_id: str) -> bool:
        """Check if agent has sufficient history for height determination"""
        return (agent_id in self.position_history and 
                len(self.position_history[agent_id]) >= 2)
    
    def clear_history(self, agent_id: str):
        """Clear history for a specific agent"""
        if agent_id in self.position_history:
            del self.position_history[agent_id]
    
    def get_movement_vector(self, agent_id: str) -> Optional[Tuple[float, float]]:
        """Get movement vector from previous to current position"""
        if agent_id not in self.position_history or len(self.position_history[agent_id]) < 2:
            return None
        
        history = list(self.position_history[agent_id])
        prev_x, prev_y, _, _ = history[-2]
        curr_x, curr_y, _, _ = history[-1]
        
        return (curr_x - prev_x, curr_y - prev_y)

