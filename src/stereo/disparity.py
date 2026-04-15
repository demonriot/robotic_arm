import numpy as np
import time
import cv2
import pickle
import yaml
from pathlib import Path

def main():
    # Setup Paths
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent
    calibration_dir = base_dir / 'data' / 'calibration' / 'params'
    stereo_remap_path = calibration_dir / 'stereoRemap.p'
    
    config_path = base_dir / 'configs' / 'stereo.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load Remap Data
    try:
        with open(stereo_remap_path, 'rb') as f:
            stereoRemap = pickle.load(f)
    except Exception as e:
        print(f"Error loading {stereo_remap_path}: {e}")
        print("Please ensure you have generated the calibration parameters first!")
        return

    mapxL = stereoRemap['mapxL']
    mapyL = stereoRemap['mapyL']
    mapxR = stereoRemap['mapxR']
    mapyR = stereoRemap['mapyR']

    # Setup Cameras
    camera_l_id = config['camera']['left_id']
    camera_r_id = config['camera']['right_id']
    widthPixel = config['disparity_camera']['width']
    heightPixel = config['disparity_camera']['height']

    img_l = cv2.VideoCapture(camera_l_id)
    img_r = cv2.VideoCapture(camera_r_id)

    if not (img_l.isOpened() and img_r.isOpened()):
        print("Error: Could not open one or both video streams.")
        return

    img_l.set(cv2.CAP_PROP_FRAME_WIDTH, widthPixel)
    img_l.set(cv2.CAP_PROP_FRAME_HEIGHT, heightPixel)
    img_r.set(cv2.CAP_PROP_FRAME_WIDTH, widthPixel)
    img_r.set(cv2.CAP_PROP_FRAME_HEIGHT, heightPixel)

    # Disparity Param Setup for StereoSGBM (Much higher quality than basic StereoBM)
    min_disp = config['disparity']['min_disp']
    num_disp = config['disparity']['num_disp_multiplier']
    block_size = config['disparity']['block_size']

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=16 * num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    print("\nCameras initialized and calibration loaded.")
    print("Running live disparity map calculation headlessly.")
    print("Continuously updating 'disparity_preview.jpg' at maximum stable framerate.")
    print("Press Ctrl+C to quit.\n")

    try:
        while True:
            # Grab frames quickly to sync
            img_r.grab()
            img_l.grab()

            ret_l, frame_l = img_l.retrieve()
            ret_r, frame_r = img_r.retrieve()

            if not ret_l or not ret_r:
                print("Failed to capture frame from webcam.")
                break

            # 1. Image Rectification
            # Applies the math rules calculated earlier to artificially straighten the lenses
            dstL = cv2.remap(frame_l, mapxL, mapyL, cv2.INTER_LINEAR)
            dstR = cv2.remap(frame_r, mapxR, mapyR, cv2.INTER_LINEAR)

            # 2. Grayscale Conversion
            gray_l = cv2.cvtColor(dstL, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(dstR, cv2.COLOR_BGR2GRAY)

            # 3. Compute Disparity
            # Calculates the distance shifted across X-axis between left and right pixels
            disparity = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0

            # 4. Normalize for Visualization
            # Scale 0 to maximum disparity into a raw 0-255 image profile
            disp_vis = (disparity - min_disp) / (16 * num_disp) * 255.0
            disp_vis = np.clip(disp_vis, 0, 255).astype(np.uint8)

            # 5. Apply Jet Colormap (Blue = far away, Red = close)
            disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
            
            # Combine Rectified Original Left image with Disparity Map side-by-side
            preview = cv2.hconcat([dstL, disp_color])

            # Output Headlessly
            out_path = base_dir / 'outputs' / 'disparity' / 'disparity_preview.jpg'
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), preview)
            
            print(f"[{time.strftime('%H:%M:%S')}] Active High-FPS Disparity Tracking...")

            # Sleep slightly to prevent hard drive saturation from instantly writing high-res images
            time.sleep(0.06)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received... Stopping.")

    finally:
        img_l.release()
        img_r.release()
        print("Cameras released. Exiting.")

if __name__ == '__main__':
    main()
