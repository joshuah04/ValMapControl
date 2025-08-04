from .collision_shape import CollisionShape

# Not currently used and not sure if functionality works
class TriggerZone:
    """Represents a trigger zone for area transitions"""
    
    def __init__(self, trigger_shape: 'CollisionShape', target_area: str, 
                 transition_type: str = "elevation", name: str = ""):
        """Initialize trigger zone
        
        Args:
            trigger_shape: Shape defining the trigger area
            target_area: Name of the area this trigger leads to
            transition_type: Type of transition (elevation, teleport, area_change)
            name: Optional name for the trigger zone
        """
        self.trigger_shape = trigger_shape
        self.target_area = target_area  # Name of the area this trigger leads to
        self.transition_type = transition_type  # "elevation", "teleport", "area_change"
        self.name = name
        self.active_agents = set()  # Track which agents are currently in this trigger
    
    def check_agent_transition(self, agent_id: str, x: float, y: float) -> bool:
        """Check if agent has entered/exited the trigger zone"""
        is_in_trigger = self.trigger_shape.contains_point(x, y)
        was_in_trigger = agent_id in self.active_agents
        
        if is_in_trigger and not was_in_trigger:
            # Agent entered trigger
            self.active_agents.add(agent_id)
            return True
        elif not is_in_trigger and was_in_trigger:
            # Agent exited trigger
            self.active_agents.discard(agent_id)
            return False
        
        return was_in_trigger  # No change in status
    
    def is_agent_in_trigger(self, agent_id: str) -> bool:
        """Check if agent is currently in the trigger zone"""
        return agent_id in self.active_agents