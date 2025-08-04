from enum import IntEnum

class HeightLevel(IntEnum):
    """Height levels for map areas and agents"""
    FLOOR = 0      # Ground level
    LOW = 1        # Low platform/boxes  
    MID = 2        # Mid-level platforms
    HIGH = 3       # High platforms/second floor
    CEILING = 4    # Highest accessible areas
    
    @classmethod
    def can_see(cls, viewer_height: 'HeightLevel', target_height: 'HeightLevel') -> bool:
        """Check if a viewer at one height can see a target at another height"""
        # Can see targets up to 2 levels above current height, and all levels below
        height_difference = target_height - viewer_height
        return height_difference <= 2
