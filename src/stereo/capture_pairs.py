import os
import time
import threading
from datetime import datetime
import cv2

# Save frames robustly inside the stereo_cam_calibration/orig_frame/ folder
scriptDir = os.path.dirname(os.path.abspath(__file__))
origFrameDir = os.path.join(scriptDir, 'orig_frame')
dir_left_frame = os.path.join(origFrameDir, 'left')
dir_right_frame = os.path.join(origFrameDir, 'right')

# Ensure directories exist
os.makedirs(dir_left_frame, exist_ok=True)
os.makedirs(dir_right_frame, exist_ok=True)

# Camera settings
camera_l = 2
camera_r = 1
widthPixel = 640
heightPixel = 480

# State variables
capture_signal = False
auto_capture = False
running = True

def keyboard_listener():
    global capture_signal, auto_capture, running
    print("\n--- Controls ---")
    print("Type 'a' and press 'ENTER' to toggle AUTO-CAPTURE (Takes a photo every 2 seconds).")
    print("Press 'ENTER' to manually capture a SINGLE frame immediately (Override).")
    print("Type 'q' and press 'ENTER' to quit early.")
    print("----------------\n")
    while running:
        try:
            val = input()
            if val.strip().lower() == 'q':
                running = False
            elif val.strip().lower() == 'a':
                auto_capture = not auto_capture
                if auto_capture:
                    print("\n[>>> AUTO CAPTURE STARTED: Taking a photo every 2 seconds. Move the checkerboard! <<<]\n")
                else:
                    print("\n[||| AUTO CAPTURE PAUSED |||]\n")
            else:
                capture_signal = True
        except EOFError:
            pass

def main():
    global capture_signal, auto_capture, running
    img_l = cv2.VideoCapture(camera_l)
    img_r = cv2.VideoCapture(camera_r)
    
    if not (img_l.isOpened() and img_r.isOpened()):
        print("Error: Could not open one or both video streams.")
        return

    # Set camera resolution
    img_l.set(cv2.CAP_PROP_FRAME_WIDTH, widthPixel)
    img_l.set(cv2.CAP_PROP_FRAME_HEIGHT, heightPixel)
    img_r.set(cv2.CAP_PROP_FRAME_WIDTH, widthPixel)
    img_r.set(cv2.CAP_PROP_FRAME_HEIGHT, heightPixel)

    print(f"Resolution of left camera: ({int(img_l.get(cv2.CAP_PROP_FRAME_WIDTH))} x {int(img_l.get(cv2.CAP_PROP_FRAME_HEIGHT))})")
    print(f"Resolution of right camera: ({int(img_r.get(cv2.CAP_PROP_FRAME_WIDTH))} x {int(img_r.get(cv2.CAP_PROP_FRAME_HEIGHT))})")

    # Start keyboard listener thread
    listener_thread = threading.Thread(target=keyboard_listener, daemon=True)
    listener_thread.start()

    n = 1
    last_preview_time = time.time()
    last_auto_capture_time = time.time()

    print("Starting capture loop. Headless mode enabled.")
    try:
        while running:
            # Grab frames quickly
            img_r.grab()
            img_l.grab()

            ret_l, frame_l = img_l.retrieve()
            ret_r, frame_r = img_r.retrieve()
            
            if not ret_l or not ret_r:
                print("Error: Failed to retrieve frame from one or both cameras.")
                running = False
                break

            current_time = time.time()

            # Handle capture signal from user or auto interval
            if capture_signal or (auto_capture and current_time - last_auto_capture_time >= 2.0):
                frame_name_l = os.path.join(dir_left_frame, f'frame{n}_l.jpg')
                frame_name_r = os.path.join(dir_right_frame, f'frame{n}_r.jpg')
                cv2.imwrite(frame_name_l, frame_l)
                cv2.imwrite(frame_name_r, frame_r)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Captured synchronized pair #{n} of 32")
                
                n += 1
                capture_signal = False
                last_auto_capture_time = current_time
                
                if n > 32:
                    print("\nMaximum of 32 images reached. Automatically finishing capture session!")
                    running = False
                    break
                
            # Periodically write a preview file to disk (headless preview) - every 1 second
            if current_time - last_preview_time > 1.0:
                h, w = frame_l.shape[:2]
                resized_r = cv2.resize(frame_r, (w, h))
                preview = cv2.hconcat([frame_l, resized_r])
                previewPath = os.path.join(scriptDir, 'preview_calibration.jpg')
                cv2.imwrite(previewPath, preview)
                last_preview_time = current_time
                
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received...")
        running = False

    finally:
        img_l.release()
        img_r.release()
        print("Cameras released. Exiting.")

if __name__ == "__main__":
    main()
