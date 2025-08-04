import math

class MotionPredictor:
    """Enhanced motion predictor with group-based prediction"""
    
    def __init__(self):
        """Initialize motion predictor with parameters"""
        self.max_velocity = 50  # Maximum pixels per frame
        self.acceleration_factor = 0.8  # How much velocity can change
        self.teammate_influence = 0.3  # How much teammates influence prediction
        self.group_influence = 0.7  # Strong influence for grouped agents
        self.min_history_for_prediction = 3
        
    def predict_position(self, track, teammates=None, group_members=None, fps=30):
        """
        Predict next position based on motion history, teammates, and group members
        
        Args:
            track: Track dictionary with position history
            teammates: List of teammate tracks for influence
            group_members: List of tracks in the same group (stronger influence)
            fps: Video FPS for time-based calculations
        
        Returns:
            Predicted (x, y) position
        """
        if len(track['position_history']) < 2:
            # Not enough history, return last known position
            return track['position_history'][-1]
        
        # Calculate velocity from recent positions
        velocity = self._calculate_velocity(track['position_history'])
        
        # Apply group influence if agent is grouped
        if group_members and len(group_members) > 0:
            group_velocity = self._calculate_group_velocity(group_members)
            # Strong influence from group members
            velocity = {
                'x': velocity['x'] * (1 - self.group_influence) + group_velocity['x'] * self.group_influence,
                'y': velocity['y'] * (1 - self.group_influence) + group_velocity['y'] * self.group_influence
            }
        elif teammates and len(teammates) >= 2:
            # Apply teammate influence if not grouped
            teammate_velocity = self._calculate_teammate_influence(track, teammates)
            velocity = {
                'x': velocity['x'] * (1 - self.teammate_influence) + teammate_velocity['x'] * self.teammate_influence,
                'y': velocity['y'] * (1 - self.teammate_influence) + teammate_velocity['y'] * self.teammate_influence
            }
        
        # Predict next position
        last_pos = track['position_history'][-1]
        predicted_pos = {
            'x': last_pos['x'] + velocity['x'],
            'y': last_pos['y'] + velocity['y']
        }
        
        return predicted_pos
    
    def _calculate_velocity(self, position_history):
        """Calculate velocity from position history"""
        if len(position_history) < 2:
            return {'x': 0, 'y': 0}
        
        # Use weighted average of recent velocities
        velocities = []
        weights = []
        
        for i in range(len(position_history) - 1, max(0, len(position_history) - 5), -1):
            if i > 0:
                dx = position_history[i]['x'] - position_history[i-1]['x']
                dy = position_history[i]['y'] - position_history[i-1]['y']
                
                # Limit velocity to reasonable bounds
                magnitude = math.sqrt(dx*dx + dy*dy)
                if magnitude > self.max_velocity:
                    scale = self.max_velocity / magnitude
                    dx *= scale
                    dy *= scale
                
                velocities.append({'x': dx, 'y': dy})
                # More recent velocities get higher weights
                weight = len(position_history) - i
                weights.append(weight)
        
        if not velocities:
            return {'x': 0, 'y': 0}
        
        # Calculate weighted average
        total_weight = sum(weights)
        avg_vx = sum(v['x'] * w for v, w in zip(velocities, weights)) / total_weight
        avg_vy = sum(v['y'] * w for v, w in zip(velocities, weights)) / total_weight
        
        return {'x': avg_vx, 'y': avg_vy}
    
    def _calculate_group_velocity(self, group_members):
        """Calculate average velocity of group members"""
        if not group_members:
            return {'x': 0, 'y': 0}
        
        velocities = []
        for member in group_members:
            if len(member['position_history']) >= 2:
                vel = self._calculate_velocity(member['position_history'])
                velocities.append(vel)
        
        if not velocities:
            return {'x': 0, 'y': 0}
        
        # Average velocity of all group members
        avg_vx = sum(v['x'] for v in velocities) / len(velocities)
        avg_vy = sum(v['y'] for v in velocities) / len(velocities)
        
        return {'x': avg_vx, 'y': avg_vy}
    
    def _calculate_teammate_influence(self, track, teammates):
        """Calculate velocity influence from nearby teammates"""
        if not teammates:
            return {'x': 0, 'y': 0}
        
        current_pos = track['position_history'][-1]
        teammate_velocities = []
        distances = []
        
        for teammate in teammates:
            if len(teammate['position_history']) < 2:
                continue
                
            # Calculate distance to teammate
            teammate_pos = teammate['position_history'][-1]
            distance = math.sqrt(
                (current_pos['x'] - teammate_pos['x'])**2 + 
                (current_pos['y'] - teammate_pos['y'])**2
            )
            
            # Only consider nearby teammates (within 100 pixels)
            if distance < 100:
                teammate_vel = self._calculate_velocity(teammate['position_history'])
                teammate_velocities.append(teammate_vel)
                # Closer teammates have more influence
                distances.append(1.0 / (distance + 1))
        
        if not teammate_velocities:
            return {'x': 0, 'y': 0}
        
        # Weight by inverse distance
        total_weight = sum(distances)
        avg_vx = sum(v['x'] * w for v, w in zip(teammate_velocities, distances)) / total_weight
        avg_vy = sum(v['y'] * w for v, w in zip(teammate_velocities, distances)) / total_weight
        
        return {'x': avg_vx, 'y': avg_vy}