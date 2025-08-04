#!/usr/bin/env python3
"""
Generates team and agent statistics after tracking is complete.
"""

import json
import csv
import os
from typing import Dict, List, Tuple
from collections import defaultdict
from dataclasses import dataclass
import math

@dataclass
class AgentStats:
    """Statistics for a single agent"""
    agent_id: str
    agent_name: str
    team: str
    track_id: int
    total_frames: int
    total_time: float
    
    # Movement statistics
    total_distance: float
    avg_speed: float
    
    # Map coverage (approximate)
    unique_positions: int
    map_coverage_score: float
    
    # Performance metrics
    avg_confidence: float
    orientation_changes: int
    consistency_score: float

@dataclass 
class TeamStats:
    """Statistics for a team"""
    team: str
    agents: List[str]
    total_frames: int
    
    # Team metrics
    team_spread: float        # How spread out the team is
    coordination_score: float # How well coordinated
    map_presence: float       # Overall map coverage
    
    # Individual stats summary
    agent_stats: Dict[str, AgentStats]
    best_performer: str
    most_active: str
    highest_coverage: str

class PostTrackingAnalyzer:
    """Simple analyzer that works with basic tracking data"""
    
    def __init__(self):
        """Initialize analyzer"""
        self.grid_size = 50  # Simplified grid for position tracking
    
    def _extract_confidence(self, confidence_data):
        """Extract confidence value from various formats"""
        if isinstance(confidence_data, dict):
            # Use combined confidence if available, otherwise detection confidence
            return confidence_data.get('combined', confidence_data.get('detection', confidence_data.get('classification', 1.0)))
        else:
            # Simple numeric confidence
            return float(confidence_data)
        
    def analyze_tracking_data(self, tracking_data: Dict) -> Tuple[Dict[str, TeamStats], Dict[str, AgentStats]]:
        """
        Analyze tracking data and generate statistics
        
        Args:
            tracking_data: Dictionary with 'frames' key containing tracking data
            
        Returns:
            Tuple of (team_stats, agent_stats)
        """
        print("Analyzing tracking data...")
        
        # Parse frames into agent data
        agent_frame_data = self._parse_frames(tracking_data)
        
        # Calculate individual agent statistics
        agent_stats = self._calculate_agent_stats(agent_frame_data)
        
        # Calculate team statistics
        team_stats = self._calculate_team_stats(agent_stats, agent_frame_data)
        
        print(f"Analysis complete: {len(team_stats)} teams, {len(agent_stats)} agents analyzed")
        
        return team_stats, agent_stats
    
    def _parse_frames(self, tracking_data: Dict) -> Dict[str, List[Dict]]:
        """Parse tracking data into agent frame lists"""
        agent_frames = defaultdict(list)
        
        frames = tracking_data.get('frames', {})
        
        for frame_num, frame_data in frames.items():
            timestamp = float(frame_num) / 30.0  # Assume 30 FPS
            
            agents = frame_data.get('agents', [])
            for agent in agents:
                agent_id = f"{agent['team']}_{agent['track_id']}"
                
                frame_info = {
                    'timestamp': timestamp,
                    'position': (agent['position']['x'], agent['position']['y']),
                    'orientation': agent.get('orientation', 0),
                    'confidence': self._extract_confidence(agent.get('confidence', 1.0)),
                    'team': agent['team'],
                    'track_id': agent['track_id'],
                    'agent_name': agent.get('agent', 'Unknown')  # Extract character name from 'agent' field
                }
                
                agent_frames[agent_id].append(frame_info)
        
        return dict(agent_frames)
    
    def _calculate_agent_stats(self, agent_frame_data: Dict[str, List[Dict]]) -> Dict[str, AgentStats]:
        """Calculate statistics for each agent"""
        agent_stats = {}
        
        for agent_id, frames in agent_frame_data.items():
            if not frames:
                continue
            
            # Basic info
            team = frames[0]['team']
            agent_name = frames[0]['agent_name']
            track_id = frames[0]['track_id']
            total_frames = len(frames)
            total_time = frames[-1]['timestamp'] - frames[0]['timestamp'] if len(frames) > 1 else 0
            
            # Calculate movement statistics
            total_distance = 0
            orientation_changes = 0
            confidences = []
            positions = []
            
            for i, frame in enumerate(frames):
                positions.append(frame['position'])
                confidences.append(frame['confidence'])
                
                if i > 0:
                    # Distance moved
                    prev_pos = frames[i-1]['position']
                    distance = math.sqrt(
                        (frame['position'][0] - prev_pos[0])**2 + 
                        (frame['position'][1] - prev_pos[1])**2
                    )
                    total_distance += distance
                    
                    # Orientation changes
                    prev_orientation = frames[i-1]['orientation']
                    orientation_diff = abs(frame['orientation'] - prev_orientation)
                    if orientation_diff > 180:
                        orientation_diff = 360 - orientation_diff
                    if orientation_diff > 30:  # Significant orientation change
                        orientation_changes += 1
            
            # Calculate unique positions (grid-based)
            unique_grid_positions = set()
            for pos in positions:
                grid_x = int(pos[0] / self.grid_size)
                grid_y = int(pos[1] / self.grid_size)
                unique_grid_positions.add((grid_x, grid_y))
            
            # Statistics
            avg_speed = total_distance / total_time if total_time > 0 else 0
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            map_coverage = len(unique_grid_positions) / max(1, total_frames) * 100
            consistency_score = len([c for c in confidences if c > 0.8]) / len(confidences) if confidences else 0
            
            agent_stats[agent_id] = AgentStats(
                agent_id=agent_id,
                agent_name=agent_name,
                team=team,
                track_id=track_id,
                total_frames=total_frames,
                total_time=total_time,
                total_distance=total_distance,
                avg_speed=avg_speed,
                unique_positions=len(unique_grid_positions),
                map_coverage_score=map_coverage,
                avg_confidence=avg_confidence,
                orientation_changes=orientation_changes,
                consistency_score=consistency_score
            )
        
        return agent_stats
    
    def _calculate_team_stats(self, agent_stats: Dict[str, AgentStats], 
                            agent_frame_data: Dict[str, List[Dict]]) -> Dict[str, TeamStats]:
        """Calculate team-level statistics"""
        team_stats = {}
        
        # Group agents by team
        teams = defaultdict(list)
        for agent_id, stats in agent_stats.items():
            teams[stats.team].append(agent_id)
        
        for team, agent_ids in teams.items():
            team_agent_stats = {aid: agent_stats[aid] for aid in agent_ids}
            
            # Calculate team metrics
            total_frames = sum(stats.total_frames for stats in team_agent_stats.values())
            
            # Team spread (how far apart team members are on average)
            team_spread = self._calculate_team_spread(team, agent_frame_data)
            
            # Coordination score (how well team moves together)
            coordination = self._calculate_coordination(team, agent_frame_data)
            
            # Map presence (combined coverage of all agents)
            map_presence = sum(stats.map_coverage_score for stats in team_agent_stats.values())
            
            # Find best performers
            best_performer = max(agent_ids, key=lambda aid: agent_stats[aid].consistency_score)
            most_active = max(agent_ids, key=lambda aid: agent_stats[aid].total_distance)
            highest_coverage = max(agent_ids, key=lambda aid: agent_stats[aid].unique_positions)
            
            team_stats[team] = TeamStats(
                team=team,
                agents=agent_ids,
                total_frames=total_frames,
                team_spread=team_spread,
                coordination_score=coordination,
                map_presence=map_presence,
                agent_stats=team_agent_stats,
                best_performer=best_performer,
                most_active=most_active,
                highest_coverage=highest_coverage
            )
        
        return team_stats
    
    def _calculate_team_spread(self, team: str, agent_frame_data: Dict[str, List[Dict]]) -> float:
        """Calculate average distance between team members (optimized)"""
        team_agents = [aid for aid, frames in agent_frame_data.items() 
                      if frames and frames[0]['team'] == team]
        
        if len(team_agents) < 2:
            return 0.0
        
        # Find minimum common frames to avoid timestamp matching
        min_frames = min(len(agent_frame_data[aid]) for aid in team_agents)
        if min_frames == 0:
            return 0.0
        
        total_distances = []
        
        # Sample every 20th frame for better performance, use same frame index for all agents
        sample_step = max(1, min_frames // 50)  # Max 50 samples
        
        for frame_idx in range(0, min_frames, sample_step):
            team_positions = []
            
            # Get positions at same frame index for all team agents
            for agent_id in team_agents:
                frames = agent_frame_data[agent_id]
                if frame_idx < len(frames):
                    team_positions.append(frames[frame_idx]['position'])
            
            # Calculate average distance between all pairs
            if len(team_positions) >= 2:
                pair_distances = []
                for i in range(len(team_positions)):
                    for j in range(i + 1, len(team_positions)):
                        pos1, pos2 = team_positions[i], team_positions[j]
                        distance = math.sqrt(
                            (pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2
                        )
                        pair_distances.append(distance)
                
                if pair_distances:
                    avg_distance = sum(pair_distances) / len(pair_distances)
                    total_distances.append(avg_distance)
        
        return sum(total_distances) / len(total_distances) if total_distances else 0.0
    
    def _calculate_coordination(self, team: str, agent_frame_data: Dict[str, List[Dict]]) -> float:
        """Calculate team coordination score (0-1) - optimized"""
        team_agents = [aid for aid, frames in agent_frame_data.items() 
                      if frames and frames[0]['team'] == team]
        
        if len(team_agents) < 2:
            return 0.0
        
        # Find minimum common frames
        min_frames = min(len(agent_frame_data[aid]) for aid in team_agents)
        if min_frames == 0:
            return 0.0
        
        coordination_scores = []
        
        # Sample much less frequently for performance - max 25 samples
        sample_step = max(1, min_frames // 25)
        
        for frame_idx in range(0, min_frames, sample_step):
            team_positions = []
            for agent_id in team_agents:
                frames = agent_frame_data[agent_id]
                if frame_idx < len(frames):
                    team_positions.append(frames[frame_idx]['position'])
            
            if len(team_positions) >= 2:
                # Calculate center of team
                center_x = sum(pos[0] for pos in team_positions) / len(team_positions)
                center_y = sum(pos[1] for pos in team_positions) / len(team_positions)
                
                # Calculate average distance from center (simplified)
                avg_distance = sum(
                    math.sqrt((pos[0] - center_x)**2 + (pos[1] - center_y)**2)
                    for pos in team_positions
                ) / len(team_positions)
                
                # Good coordination = moderate spread (not too tight, not too spread)
                optimal_spread = 150.0  # pixels
                coordination_score = max(0, 1.0 - abs(avg_distance - optimal_spread) / optimal_spread)
                coordination_scores.append(coordination_score)
        
        return sum(coordination_scores) / len(coordination_scores) if coordination_scores else 0.0
    
    def export_stats_csv(self, team_stats: Dict[str, TeamStats], agent_stats: Dict[str, AgentStats], 
                        output_dir: str = "stats_output"):
        """Export statistics to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Export agent statistics
        agent_csv_path = os.path.join(output_dir, "agent_stats.csv")
        with open(agent_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Agent_ID', 'Agent_Name', 'Team', 'Track_ID', 'Total_Frames', 'Total_Time_Seconds', 
                'Total_Distance', 'Avg_Speed', 'Unique_Positions', 'Map_Coverage_Score',
                'Avg_Confidence', 'Orientation_Changes', 'Consistency_Score'
            ])
            
            for agent_id, stats in agent_stats.items():
                writer.writerow([
                    stats.agent_id, stats.agent_name, stats.team, stats.track_id, stats.total_frames, f"{stats.total_time:.2f}",
                    f"{stats.total_distance:.1f}", f"{stats.avg_speed:.2f}", stats.unique_positions,
                    f"{stats.map_coverage_score:.1f}", f"{stats.avg_confidence:.3f}",
                    stats.orientation_changes, f"{stats.consistency_score:.3f}"
                ])
        
        # Export team statistics
        team_csv_path = os.path.join(output_dir, "team_stats.csv")
        with open(team_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Team', 'Agents', 'Total_Frames', 'Team_Spread', 'Coordination_Score',
                'Map_Presence', 'Best_Performer', 'Most_Active', 'Highest_Coverage'
            ])
            
            for team, stats in team_stats.items():
                writer.writerow([
                    stats.team, '|'.join(stats.agents), stats.total_frames,
                    f"{stats.team_spread:.1f}", f"{stats.coordination_score:.3f}",
                    f"{stats.map_presence:.1f}", stats.best_performer,
                    stats.most_active, stats.highest_coverage
                ])
        
        print(f"Statistics exported to:")
        print(f"  Agent stats: {agent_csv_path}")
        print(f"  Team stats: {team_csv_path}")
    
    def print_summary_report(self, team_stats: Dict[str, TeamStats], agent_stats: Dict[str, AgentStats]):
        """Print comprehensive summary report"""
        print(f"\n{'='*60}")
        print(f"POST-TRACKING ANALYSIS REPORT")
        print(f"{'='*60}")
        
        print(f"\nOVERVIEW:")
        print(f"  Teams: {len(team_stats)}")
        print(f"  Total Agents: {len(agent_stats)}")
        
        # Team comparison
        print(f"\nTEAM COMPARISON:")
        for team, stats in team_stats.items():
            print(f"  {team.upper()}:")
            print(f"    Agents: {len(stats.agents)}")
            print(f"    Map Presence: {stats.map_presence:.1f}")
            print(f"    Coordination: {stats.coordination_score:.3f}")
            print(f"    Team Spread: {stats.team_spread:.1f} pixels")
            print(f"    Best Performer: {stats.best_performer}")
        
        # Agent rankings
        print(f"\nTOP PERFORMERS:")
        
        # Consistency leaders
        consistency_ranking = sorted(agent_stats.items(), 
                                   key=lambda x: x[1].consistency_score, reverse=True)[:5]
        print(f"  Most Consistent:")
        for i, (agent_id, stats) in enumerate(consistency_ranking, 1):
            print(f"    {i}. {stats.agent_name} ({stats.team}): {stats.consistency_score:.3f}")
        
        # Most active
        distance_ranking = sorted(agent_stats.items(), 
                                key=lambda x: x[1].total_distance, reverse=True)[:5]
        print(f"  Most Active (Distance):")
        for i, (agent_id, stats) in enumerate(distance_ranking, 1):
            print(f"    {i}. {stats.agent_name} ({stats.team}): {stats.total_distance:.1f} pixels")
        
        # Best coverage
        coverage_ranking = sorted(agent_stats.items(), 
                                key=lambda x: x[1].unique_positions, reverse=True)[:5]
        print(f"  Best Map Coverage:")
        for i, (agent_id, stats) in enumerate(coverage_ranking, 1):
            print(f"    {i}. {stats.agent_name} ({stats.team}): {stats.unique_positions} positions")
        
        print(f"\n{'='*60}")
        print(f"Analysis complete!")

def analyze_tracker_output(json_file_path: str) -> Tuple[Dict, Dict]:
    """
    Analyze val_vid_tracker.py output and generate complete statistics
    
    Args:
        json_file_path: Path to tracking JSON output
    
    Returns:
        Tuple of (team_stats, agent_stats)
    """
    print(f"Loading tracking data from {json_file_path}...")
    
    with open(json_file_path, 'r') as f:
        tracking_data = json.load(f)
    
    # Initialize analyzer
    analyzer = PostTrackingAnalyzer()
    
    # Generate all statistics
    team_stats, agent_stats = analyzer.analyze_tracking_data(tracking_data)
    
    # Print summary report
    analyzer.print_summary_report(team_stats, agent_stats)
    
    # Export to CSV
    analyzer.export_stats_csv(team_stats, agent_stats)
    
    return team_stats, agent_stats

if __name__ == "__main__":
    print("Post-Tracking Statistics Analyzer")
    print("=" * 50)
    print("Features:")
    print("  Team and individual agent statistics")
    print("  Movement analysis and map coverage")
    print("  Performance rankings and comparisons")
    print("  CSV export for external analysis")
