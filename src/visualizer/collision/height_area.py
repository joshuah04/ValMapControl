from typing import Optional

from .height_level import HeightLevel
from .collision_shape import CollisionShape

class HeightArea:
    """Represents an area at a specific height level"""
    
    def __init__(self, shape: 'CollisionShape', height_level: HeightLevel, name: str = ""):
        """Initialize height area
        
        Args:
            shape: Collision shape defining the area boundaries
            height_level: Height level of this area
            name: Optional name for the area
        """
        self.shape = shape
        self.height_level = height_level
        self.name = name or f"{height_level.name.lower()}_area"
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside this height area"""
        return self.shape.contains_point(x, y)
    
    def get_height_at_point(self, x: float, y: float) -> Optional[HeightLevel]:
        """Get the height level at a specific point"""
        if self.contains_point(x, y):
            return self.height_level
        return None
