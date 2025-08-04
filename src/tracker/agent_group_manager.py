from collections import defaultdict

# This is not currently being used. Set use_agent_grouping to True if you want to enable it.
# May not be functional.
class AgentGroupManager:
    """Manages agent grouping when they overlap"""

    def __init__(self, overlap_threshold=0.40):
        """Initialize agent group manager

        Args:
            overlap_threshold: IoU threshold for considering agents as overlapping
        """
        self.overlap_threshold = overlap_threshold
        self.agent_groups = {}  # group_id -> {track_ids, team, last_update, active}
        self.track_to_group = {}  # track_id -> group_id
        self.next_group_id = 0
        
    def calculate_bbox_overlap(self, bbox1, bbox2):
        """
        Calculate Intersection over Union (IoU) for two bounding boxes
        
        Args:
            bbox1, bbox2: Bounding boxes in format [x1, y1, x2, y2]
            
        Returns:
            float: IoU value between 0 and 1
        """
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # Check if there's no intersection
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        # Calculate intersection area
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update_groups(self, active_tracks, frame_number, detected_track_ids):
        """Update agent groups based on overlap and detection status"""
        # Clear outdated groups
        self._clean_old_groups(frame_number)
        
        # Update activation status of existing groups
        self._update_group_activation(detected_track_ids)
        
        # Group tracks by team
        team_tracks = defaultdict(list)
        for track in active_tracks:
            if track['state'] != 'dead':
                team_tracks[track['team']].append(track)
        
        # Check for overlaps within each team
        new_groups = []
        processed_tracks = set()
        
        for team, tracks in team_tracks.items():
            for i, track1 in enumerate(tracks):
                if track1['track_id'] in processed_tracks:
                    continue
                    
                overlapping_tracks = [track1['track_id']]
                processed_tracks.add(track1['track_id'])
                
                # Find all overlapping teammates
                for j, track2 in enumerate(tracks[i+1:], i+1):
                    if track2['track_id'] in processed_tracks:
                        continue
                        
                    overlap = self.calculate_bbox_overlap(track1['bbox'], track2['bbox'])
                    if overlap >= self.overlap_threshold:
                        overlapping_tracks.append(track2['track_id'])
                        processed_tracks.add(track2['track_id'])
                
                # Create group if multiple tracks overlap
                if len(overlapping_tracks) > 1:
                    new_groups.append({
                        'track_ids': set(overlapping_tracks),
                        'team': team,
                        'last_update': frame_number,
                        'active': False  # Groups start inactive
                    })
        
        # Merge with existing groups
        self._merge_groups(new_groups, frame_number)
    
    def _update_group_activation(self, detected_track_ids):
        """Activate groups when members start disappearing"""
        for group_id, group in self.agent_groups.items():
            if not group['active']:
                # Check if any group member is not detected
                missing_members = [tid for tid in group['track_ids'] 
                                 if tid not in detected_track_ids]
                
                if missing_members:
                    # Activate the group
                    group['active'] = True
                    print(f"Group {group_id} activated - missing members: {missing_members}")
    
    def _merge_groups(self, new_groups, frame_number):
        """Merge new groups with existing ones"""
        for new_group in new_groups:
            merged = False
            
            # Check if any track is already in a group
            for track_id in new_group['track_ids']:
                if track_id in self.track_to_group:
                    group_id = self.track_to_group[track_id]
                    # Add all tracks to existing group
                    self.agent_groups[group_id]['track_ids'].update(new_group['track_ids'])
                    self.agent_groups[group_id]['last_update'] = frame_number
                    # Keep activation status unchanged
                    
                    # Update track mappings
                    for tid in new_group['track_ids']:
                        self.track_to_group[tid] = group_id
                    
                    merged = True
                    break
            
            if not merged:
                # Create new group (inactive by default)
                group_id = self.next_group_id
                self.next_group_id += 1
                self.agent_groups[group_id] = new_group
                
                # Update track mappings
                for track_id in new_group['track_ids']:
                    self.track_to_group[track_id] = group_id
    
    def _clean_old_groups(self, current_frame, max_age=60):
        """Remove old groups that haven't been updated. """
        groups_to_remove = []
        
        for group_id, group in self.agent_groups.items():
            if current_frame - group['last_update'] > max_age:
                groups_to_remove.append(group_id)
        
        for group_id in groups_to_remove:
            # Remove track mappings
            for track_id in self.agent_groups[group_id]['track_ids']:
                if track_id in self.track_to_group:
                    del self.track_to_group[track_id]
            # Remove group
            del self.agent_groups[group_id]
    
    def get_group_members(self, track_id):
        """Get all track IDs in the same group as the given track"""
        if track_id not in self.track_to_group:
            return []
        
        group_id = self.track_to_group[track_id]
        return list(self.agent_groups[group_id]['track_ids'])
    
    def is_grouped(self, track_id):
        """Check if a track is part of a group"""
        return track_id in self.track_to_group
    
    def is_group_active(self, track_id):
        """Check if a track's group is active (should use group prediction)"""
        if track_id not in self.track_to_group:
            return False
        
        group_id = self.track_to_group[track_id]
        return self.agent_groups[group_id]['active']
    
    def get_group_info(self, track_id):
        """Get full group information for a track"""
        if track_id not in self.track_to_group:
            return None
        
        group_id = self.track_to_group[track_id]
        return {
            'group_id': group_id,
            'members': list(self.agent_groups[group_id]['track_ids']),
            'active': self.agent_groups[group_id]['active'],
            'team': self.agent_groups[group_id]['team']
        }
    
    def remove_from_group(self, track_id):
        """Remove a track from its group (e.g., when player dies)"""
        if track_id not in self.track_to_group:
            return
        
        group_id = self.track_to_group[track_id]
        
        # Remove track from group
        if group_id in self.agent_groups:
            self.agent_groups[group_id]['track_ids'].discard(track_id)
            
            # If group becomes empty or has only one member, dissolve it
            if len(self.agent_groups[group_id]['track_ids']) <= 1:
                # Remove remaining members from track mapping
                for remaining_track in list(self.agent_groups[group_id]['track_ids']):
                    if remaining_track in self.track_to_group:
                        del self.track_to_group[remaining_track]
                
                # Remove the group
                del self.agent_groups[group_id]
        
        # Remove track from mapping
        if track_id in self.track_to_group:
            del self.track_to_group[track_id]