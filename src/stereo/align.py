import numpy as np
import cv2
import time
import os
import yaml

def main():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent
    config_path = base_dir / 'configs' / 'stereo.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Camera settings
    camera_l_id = config['camera']['left_id']
    camera_r_id = config['camera']['right_id']
    width = config['camera']['width']
    height = config['camera']['height']

    img_l = cv2.VideoCapture(camera_l_id)
    img_r = cv2.VideoCapture(camera_r_id)

    if not (img_l.isOpened() and img_r.isOpened()):
        print("Error: Could not open one or both video streams.")
        return

    # Set camera resolution overriding default properties
    img_l.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    img_l.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    img_r.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    img_r.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    width_l = img_l.get(cv2.CAP_PROP_FRAME_WIDTH)
    height_l = img_l.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Resolution of left camera: ({int(width_l)} x {int(height_l)})")
    
    width_r = img_r.get(cv2.CAP_PROP_FRAME_WIDTH)
    height_r = img_r.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Resolution of right camera: ({int(width_r)} x {int(height_r)})")

    print("\nCameras initialized.")
    print("Running in headless mode (no GUI window to prevent XCB errors).")
    print("The blended view will be constantly overwritten to 'blended_alignment.jpg' at ~0.5 FPS.")
    print("Press Ctrl+C in this terminal to quit.\n")

    try:
        while True:
            # Grab frames (using consecutive grab ensures minimal time difference between left and right capture)
            img_r.grab()
            img_l.grab()
            
            # Retrieve actual frame data
            ret_l, frame_l = img_l.retrieve()
            ret_r, frame_r = img_r.retrieve()
            
            if not ret_l or not ret_r:
                print("Error: Failed to retrieve frame from one or both cameras.")
                break
                
            # Blend frames evenly (50% opacity left, 50% opacity right) for equal transparency
            blended = cv2.addWeighted(frame_l, 0.5, frame_r, 0.5, 0)
            
            # Add alignment crosshairs (center, structural quarters)
            w = int(width_l)
            h = int(height_l)
            
            # Draw horizontal alignment lines
            cv2.line(blended, (0, h // 2), (w, h // 2), (255, 0, 0), 1)
            cv2.line(blended, (0, h // 4), (w, h // 4), (255, 0, 0), 1)
            cv2.line(blended, (0, (h * 3) // 4), (w, (h * 3) // 4), (255, 0, 0), 1)
            
            # Draw vertical alignment lines
            cv2.line(blended, (w // 2, 0), (w // 2, h), (255, 0, 0), 1)
            cv2.line(blended, (w // 4, 0), (w // 4, h), (255, 0, 0), 1)
            cv2.line(blended, ((w * 3) // 4, 0), ((w * 3) // 4, h), (255, 0, 0), 1)
            
            # Save the image instead of imshow to avoid XCB/Qt headless errors
            scriptDir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.normpath(os.path.join(scriptDir, '..', '..'))
            out_dir = os.path.join(base_dir, 'outputs', 'calibration')
            os.makedirs(out_dir, exist_ok=True)
            output_file = os.path.join(out_dir, 'blended_alignment.jpg')
            cv2.imwrite(output_file, blended)
            print(f"[{time.strftime('%H:%M:%S')}] Saved updated {output_file}")
            
            # Wait 2 seconds for ~0.5 fps update rate
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping...")

    finally:
        img_l.release()
        img_r.release()
        print("Cameras released.")

if __name__ == "__main__":
    main()
