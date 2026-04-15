import os
import sys
import glob
import subprocess
import yaml
from pathlib import Path

def main():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    
    # Load config to get standard output struct
    config_path = base_dir / 'configs' / 'stereo.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Standard locations
    left_dir = base_dir / 'outputs' / 'validation' / 'captured_frames' / 'left'
    right_dir = base_dir / 'outputs' / 'validation' / 'captured_frames' / 'right'

    if not left_dir.exists() or not right_dir.exists():
        print("Validation directories don't exist yet. Please run capture_validation_frames.py first.")
        return

    # Find all images
    left_images = sorted(list(left_dir.glob("*.png")) + list(left_dir.glob("*.jpg")))
    
    if not left_images:
        print("No captured images found in validation folder.")
        return

    print(f"Found {len(left_images)} left images to validate.")
    
    for left_img in left_images:
        filename = left_img.name
        
        # parse trial name assuming naming format 'left_TRIAL_NAME.png'
        if filename.startswith("left_"):
            trial_name = left_img.stem.replace("left_", "")
        else:
            trial_name = left_img.stem
            
        right_img = right_dir / f"right_{trial_name}{left_img.suffix}"
        
        if not right_img.exists():
            print(f"[!] Warning: No corresponding right image found for trial '{trial_name}' (expected {right_img.name}). Skipping.")
            continue
            
        print(f"\n{'='*50}")
        print(f" Validating Trial: {trial_name}")
        print(f"{'='*50}")
        
        # Ask user for physical distance dynamically
        actual_dist_input = input(f"Enter the actual physical distance in cm for '{trial_name}' (or type 'skip' to skip this pair): ")
        if actual_dist_input.lower().strip() == 'skip':
            print("Skipping...")
            continue
            
        try:
            dist_val = float(actual_dist_input)
        except ValueError:
            print("Invalid distance input. Skipping this pair.")
            continue
            
        # call the deep validation script automatically
        cmd = [
            sys.executable,
            str(script_dir / "validate_depth_click.py"),
            "--left-image", str(left_img),
            "--right-image", str(right_img),
            "--actual-distance-cm", str(dist_val),
            "--trial-name", trial_name
            # No need for --calib-dir or stereosgbm params since validate_depth_click now pulls from stereo.yaml
        ]
        
        print("\nOpening validation UI...")
        subprocess.run(cmd)

    print("\nAll batch validation finished!")

if __name__ == "__main__":
    main()
