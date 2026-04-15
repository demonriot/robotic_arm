import numpy as np
import cv2
import pickle
from pathlib import Path

# Set root directory to script location.
script_dir = Path(__file__).resolve().parent
print(f'\nscript_dir = {script_dir}\n')

# Paths setup
base_dir = script_dir.parent.parent
dir_stereoRemap = base_dir / 'data' / 'calibration' / 'params' / 'stereoRemap.p'
dir_original_L = base_dir / 'data' / 'calibration' / 'left'
dir_original_R = base_dir / 'data' / 'calibration' / 'right'

dir_compare_frame_pairs = base_dir / 'outputs' / 'calibration' / 'compare_rectified_frame_pairs'
dir_rectified_L = base_dir / 'outputs' / 'calibration' / 'rectified_frames' / 'left'
dir_rectified_R = base_dir / 'outputs' / 'calibration' / 'rectified_frames' / 'right'

# Make sure output directories exist
dir_compare_frame_pairs.mkdir(parents=True, exist_ok=True)
dir_rectified_L.mkdir(parents=True, exist_ok=True)
dir_rectified_R.mkdir(parents=True, exist_ok=True)

# Load the dictionary back from the pickle file.
try:
    with open(dir_stereoRemap, 'rb') as f:
        stereoRemap = pickle.load(f)
except Exception as e:
    print(f"Error loading {dir_stereoRemap}: {e}")
    exit()

mapxL = stereoRemap['mapxL']
mapyL = stereoRemap['mapyL']
mapxR = stereoRemap['mapxR']
mapyR = stereoRemap['mapyR']

# Load original frames (supports JPG and PNG)
imagesL = sorted(list(dir_original_L.glob('*.jpg')) + list(dir_original_L.glob('*.png')))
imagesR = sorted(list(dir_original_R.glob('*.jpg')) + list(dir_original_R.glob('*.png')))

if len(imagesL) != len(imagesR) or len(imagesL) == 0:
    print("Error: The number of left and right frames must be exactly the same and not empty.")
    exit()

# Rectify each frame.
for i in range(len(imagesL)):
    print(f'i = {i}')
    
    file_name_l = imagesL[i].name
    file_name_r = imagesR[i].name
    print(f'Rectifying {file_name_l} and {file_name_r}...')
    
    imgL = cv2.imread(str(imagesL[i]))
    imgR = cv2.imread(str(imagesR[i]))
    
    # Remap.
    dstL = cv2.remap(imgL, mapxL, mapyL, cv2.INTER_LINEAR)
    dstR = cv2.remap(imgR, mapxR, mapyR, cv2.INTER_LINEAR)
    
    # Combine frame pairs for visualization map.
    imgTwin = np.hstack((imgL, imgR))
    imgTwin_rect = np.hstack((dstL, dstR))
    
    # Draw horizontal guide lines on rectified twin to verify alignment
    h, w = imgTwin_rect.shape[:2]
    step = h // 10
    for y in range(step, h, step):
        cv2.line(imgTwin_rect, (0, y), (w, y), (0, 255, 0), 1)
        
    compareImg = np.vstack((imgTwin, imgTwin_rect))
    
    # We removed cv2.imshow to maintain headless operability.
    
    # Save comparing frame pairs.
    out_compare_path = dir_compare_frame_pairs / f"{file_name_l[:-6]}_compare.jpg"
    cv2.imwrite(str(out_compare_path), compareImg)
    
    # Save isolated rectified frames.
    out_rect_L_path = dir_rectified_L / f"{file_name_l[:-4]}_rectified.jpg"
    out_rect_R_path = dir_rectified_R / f"{file_name_r[:-4]}_rectified.jpg"
    
    cv2.imwrite(str(out_rect_L_path), dstL)
    cv2.imwrite(str(out_rect_R_path), dstR)

print("\nDone! All images remapped and saved.")
