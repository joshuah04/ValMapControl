#!/usr/bin/env python3
"""
Setup script for map perimeters
Helps you configure map boundaries for vision cone clipping
"""

import subprocess
import sys
from pathlib import Path

def setup_map_perimeters():
    """Interactive setup for map perimeters"""
    print("=== Map Perimeter Setup ===")
    print()
    
    # Check available maps
    maps_dir = Path("maps")
    if not maps_dir.exists():
        print("Error: maps/ directory not found")
        return 1
    
    # List available map images
    map_files = list(maps_dir.glob("*.png"))
    if not map_files:
        print("Error: No map images found in maps/ directory")
        return 1
    
    print("Available maps:")
    for i, map_file in enumerate(map_files, 1):
        map_name = map_file.stem
        config_path = maps_dir / "collision_configs" / f"{map_name}_collision.json"
        status = "Configured" if config_path.exists() else "Not configured"
        print(f"  {i}. {map_name} - {status}")
    
    print()
    
    # Get user choice
    while True:
        try:
            choice = input("Select a map to configure (number or name): ").strip()
            
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(map_files):
                    selected_map = map_files[idx].stem
                    break
            else:
                # Check if it's a valid map name
                if choice in [f.stem for f in map_files]:
                    selected_map = choice
                    break
            
            print("Invalid choice. Please try again.")
        except (ValueError, KeyboardInterrupt):
            print("\nExiting...")
            return 0
    
    print(f"\nSelected map: {selected_map}")
    
    # Check if already configured
    config_path = maps_dir / "collision_configs" / f"{selected_map}_collision.json"
    if config_path.exists():
        response = input("Map already has perimeter configuration. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return 0
    
    # Launch perimeter editor
    print(f"\nLaunching perimeter editor for {selected_map}...")
    
    try:
        # Run the perimeter editor
        result = subprocess.run([
            sys.executable, "core/map_perimeter_editor.py", selected_map
        ], check=True)
        
        print(f"\nPerimeter configuration saved for {selected_map}")
        print("You can now use the enhanced track visualizer with vision cone clipping!")
        
        # Show how to use it
        print(f"\nTo test with the visualizer:")
        print(f"python enhanced_track_visualizer.py --json track-output/out_data.json --minimap maps/{selected_map}.png --map-name {selected_map}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running perimeter editor: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 0
    
    return 0

def list_configured_maps():
    """List maps that have been configured"""
    maps_dir = Path("maps")
    collision_dir = maps_dir / "collision_configs"
    
    if not collision_dir.exists():
        print("No collision configurations found.")
        return
    
    config_files = list(collision_dir.glob("*_collision.json"))
    
    if not config_files:
        print("No collision configurations found.")
        return
    
    print("Configured maps:")
    for config_file in config_files:
        map_name = config_file.stem.replace("_collision", "")
        print(f"  - {map_name}")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        list_configured_maps()
        return 0
    
    return setup_map_perimeters()

if __name__ == "__main__":
    exit(main())