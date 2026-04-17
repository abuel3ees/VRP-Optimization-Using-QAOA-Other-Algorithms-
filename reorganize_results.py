#!/usr/bin/env python3
"""
Reorganize results directory structure.
Changes from: results/instance_name/plots|data|logs
To:           results/instance_name/K_vehicles/plots|data|logs
"""

import os
import shutil
from pathlib import Path

def reorganize_results():
    results_dir = Path("results")
    
    if not results_dir.exists():
        print("❌ No 'results' directory found")
        return
    
    # Process each instance
    for instance_dir in results_dir.iterdir():
        if not instance_dir.is_dir():
            continue
        
        instance_name = instance_dir.name
        print(f"\n📁 Processing instance: {instance_name}")
        
        # Find all files to determine vehicle counts
        vehicle_counts = set()
        for subdir in ["plots", "data", "logs"]:
            subdir_path = instance_dir / subdir
            if subdir_path.exists():
                for file in subdir_path.iterdir():
                    if file.is_file():
                        # Extract vehicle count from filename (e.g., "..._7veh_...")
                        import re
                        match = re.search(r'_(\d+)veh_', file.name)
                        if match:
                            k = int(match.group(1))
                            vehicle_counts.add(k)
        
        if not vehicle_counts:
            print(f"  ⚠️  No vehicle count found in filenames, skipping...")
            continue
        
        # Reorganize each vehicle count
        for k in sorted(vehicle_counts):
            vehicle_folder = instance_dir / f"{k}_vehicles"
            vehicle_folder.mkdir(parents=True, exist_ok=True)
            
            print(f"  📦 Organizing {k} vehicles...")
            
            # Move subdirectories
            for subdir_name in ["plots", "data", "logs"]:
                src_dir = instance_dir / subdir_name
                dest_dir = vehicle_folder / subdir_name
                
                if not src_dir.exists():
                    continue
                
                # Create destination
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                # Move files matching this K value
                for file in src_dir.iterdir():
                    if file.is_file() and f"_{k}veh_" in file.name:
                        dest_file = dest_dir / file.name
                        shutil.move(str(file), str(dest_file))
                        print(f"    ✓ Moved: {file.name}")
        
        # Clean up empty directories
        for subdir_name in ["plots", "data", "logs"]:
            subdir = instance_dir / subdir_name
            if subdir.exists() and not any(subdir.iterdir()):
                subdir.rmdir()
                print(f"  🗑️  Removed empty: {subdir_name}/")
    
    print("\n✅ Reorganization complete!")
    print("\nNew structure:")
    print("  results/")
    for instance_dir in sorted(results_dir.iterdir()):
        if instance_dir.is_dir():
            print(f"  ├─ {instance_dir.name}/")
            for vehicle_dir in sorted(instance_dir.iterdir()):
                if vehicle_dir.is_dir():
                    print(f"  │  ├─ {vehicle_dir.name}/")
                    for subdir in ["plots", "data", "logs"]:
                        if (vehicle_dir / subdir).exists():
                            file_count = len(list((vehicle_dir / subdir).iterdir()))
                            if file_count > 0:
                                print(f"  │  │  ├─ {subdir}/ ({file_count} files)")

if __name__ == "__main__":
    reorganize_results()
