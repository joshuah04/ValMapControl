from enum import Enum

class ControlStatus(Enum):
    """Map control status for areas"""
    TEAM1 = 1       # Team 1 control (green)
    TEAM2 = 2       # Team 2 control (red)
    CONTESTED = 3   # Both teams have vision (yellow)
    NEUTRAL = 0     # No team has vision (neutral/open)