import numpy as np
import cv2
import pickle
import yaml
from pathlib import Path

# Set root directory to script location.
script_dir = Path(__file__).resolve().parent
print(f'\nscript_dir = {script_dir}\n')

# Directories
base_dir = script_dir.parent.parent
dir_calib_parameter = base_dir / 'data' / 'calibration' / 'params'
dir_original_L = base_dir / 'data' / 'calibration' / 'left'
dir_original_R = base_dir / 'data' / 'calibration' / 'right'
dir_calib_process = base_dir / 'outputs' / 'calibration' / 'process_frames'

dir_calib_parameter.mkdir(parents=True, exist_ok=True)
dir_calib_process.mkdir(parents=True, exist_ok=True)

# Load config
config_path = base_dir / 'configs' / 'stereo.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Checkerboard properties
Nx = config['calibration']['checkerboard']['Nx']
Ny = config['calibration']['checkerboard']['Ny']
square_size_cm = config['calibration']['checkerboard']['square_size_cm']

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)

# prepare object points in REAL-WORLD UNITS (cm)
objp = np.zeros((Nx * Ny, 3), np.float32)
objp[:, :2] = np.mgrid[0:Ny, 0:Nx].T.reshape(-1, 2)
objp *= square_size_cm

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpointsL = []  # 2d points in image plane
imgpointsR = []  # 2d points in image plane

# Load original frames.
imagesL = sorted(list(dir_original_L.glob('*.jpg')) + list(dir_original_L.glob('*.png')))
imagesR = sorted(list(dir_original_R.glob('*.jpg')) + list(dir_original_R.glob('*.png')))

# Check if image counts match
if len(imagesL) != len(imagesR) or len(imagesL) == 0:
    print('Error: The image numbers of left and right cameras must be the same and not empty!')
    exit()

print(f"Found {len(imagesL)} image pairs. Processing...")

n = 0
widthPixel = 0
heightPixel = 0

for i in range(len(imagesL)):
    imgL = cv2.imread(str(imagesL[i]))
    imgR = cv2.imread(str(imagesR[i]))

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    if widthPixel == 0:
        heightPixel, widthPixel = grayL.shape

    # Find chessboard corners
    retL, cornersL = cv2.findChessboardCorners(grayL, (Ny, Nx), None)
    retR, cornersR = cv2.findChessboardCorners(grayR, (Ny, Nx), None)

    if retL and retR:
        n += 1
        print(f'Valid Pair Found: {n}')
        objpoints.append(objp)

        cornersL2 = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        imgpointsL.append(cornersL2)

        cornersR2 = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
        imgpointsR.append(cornersR2)

        imgL = cv2.drawChessboardCorners(imgL, (Ny, Nx), cornersL2, retL)
        imgR = cv2.drawChessboardCorners(imgR, (Ny, Nx), cornersR2, retR)
        imgTwin = np.hstack((imgL, imgR))

        base_name = imagesL[i].name
        cv2.imwrite(str(dir_calib_process / f"{base_name[:-4]}_corners.jpg"), imgTwin)
    else:
        print(f"Checkerboard not fully detected in pair {i+1}")

if len(objpoints) == 0:
    print("Error: Could not find any valid checkerboard patterns in the images provided.")
    exit()

print('\nCalculating the camera matrices and distortion coefficients independently...')
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, (widthPixel, heightPixel), None, None)
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, (widthPixel, heightPixel), None, None)
print('Done.\n')

print('Calculating reprojection errors...')
tot_errorL = 0
tot_errorR = 0

for i in range(len(objpoints)):
    imgpointsL2, _ = cv2.projectPoints(objpoints[i], rvecsL[i], tvecsL[i], mtxL, distL)
    imgpointsR2, _ = cv2.projectPoints(objpoints[i], rvecsR[i], tvecsR[i], mtxR, distR)
    errorL = cv2.norm(imgpointsL[i], imgpointsL2, cv2.NORM_L2) / len(imgpointsL2)
    errorR = cv2.norm(imgpointsR[i], imgpointsR2, cv2.NORM_L2) / len(imgpointsR2)
    tot_errorL += errorL
    tot_errorR += errorR

print(f'mean error L: {tot_errorL / len(objpoints)}')
print(f'mean error R: {tot_errorR / len(objpoints)}')

CamParasL = {'mtxL': mtxL, 'distL': distL, 'rvecsL': rvecsL, 'tvecsL': tvecsL}
CamParasR = {'mtxR': mtxR, 'distR': distR, 'rvecsR': rvecsR, 'tvecsR': tvecsR}

print('Saving calibration parameters...')
with open(dir_calib_parameter / 'CamParasL.p', 'wb') as f:
    pickle.dump(CamParasL, f)
with open(dir_calib_parameter / 'CamParasR.p', 'wb') as f:
    pickle.dump(CamParasR, f)
print('Done.\n')

print('Calculating the rectify parameters for stereo cameras...')
retval, cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, rotationMatrix, translationVector, essentialMatrix, fundamentalMatrix = cv2.stereoCalibrate(
    objpoints,
    imgpointsL,
    imgpointsR,
    mtxL,
    distL,
    mtxR,
    distR,
    (widthPixel, heightPixel),
    criteria,
    flags=cv2.CALIB_FIX_INTRINSIC
)

rotationMatrixL, rotationMatrixR, projectionMatrixL, projectionMatrixR, disp2depthMappingMatrix, validPixROI1, validPixROI2 = cv2.stereoRectify(
    cameraMatrixL,
    distCoeffsL,
    cameraMatrixR,
    distCoeffsR,
    (widthPixel, heightPixel),
    rotationMatrix,
    translationVector,
    alpha=1,
    newImageSize=(0, 0)
)

print('translationVector = \n', translationVector)
print('disp2depthMappingMatrix = \n', disp2depthMappingMatrix)

mapxL, mapyL = cv2.initUndistortRectifyMap(
    cameraMatrixL, distCoeffsL, rotationMatrixL, projectionMatrixL,
    (widthPixel, heightPixel), cv2.CV_32FC1
)
mapxR, mapyR = cv2.initUndistortRectifyMap(
    cameraMatrixR, distCoeffsR, rotationMatrixR, projectionMatrixR,
    (widthPixel, heightPixel), cv2.CV_32FC1
)

print("Map Shapes:", mapxL.shape, mapyR.shape)

stereoRemap = {'mapxL': mapxL, 'mapyL': mapyL, 'mapxR': mapxR, 'mapyR': mapyR}

print('Saving stereo remap matrices...')
with open(dir_calib_parameter / 'stereoRemap.p', 'wb') as f:
    pickle.dump(stereoRemap, f)
with open(dir_calib_parameter / 'disp2depth.p', 'wb') as f:
    pickle.dump(disp2depthMappingMatrix, f)
print('Done.\n')