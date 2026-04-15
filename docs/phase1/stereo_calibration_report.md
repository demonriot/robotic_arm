# Stereo Calibration and Validation Report

## 1. Hardware Setup

### Purpose
To establish a fixed and stable stereo camera system with a known geometric relationship between the two cameras.

### Setup Description
- Two identical cameras were mounted on a rigid stereo rig.
- The cameras were positioned with a fixed horizontal baseline (approximately 6–10 cm).
- Both cameras were aligned to be approximately:
  - parallel to each other
  - at the same height
- The stereo rig was mounted above the workspace in a top-down configuration.

### Alignment Process
- A visual alignment script was used to overlay both camera feeds.
- Cameras were adjusted until scene features approximately overlapped.
- Final alignment ensured minimal vertical disparity between corresponding points.

### Observations
- Perfect alignment is not required at this stage, as rectification compensates for minor misalignment.
- However, large misalignment increases calibration error and reduces disparity quality.

### Notes
- Rigid mounting is critical — any movement after calibration invalidates parameters.
- Consistent lighting between both cameras improves feature detection during calibration.

## 2. Data Capture

### Purpose
To collect a diverse set of stereo image pairs containing a calibration pattern, enabling accurate estimation of intrinsic and extrinsic camera parameters.

### Calibration Pattern
- A checkerboard pattern was used for calibration.
- The pattern consisted of a fixed grid of inner corners (e.g., 11 × 7).
- The checkerboard was printed on a flat surface to ensure geometric consistency.

### Image Capture Process
- Approximately 20–30 stereo image pairs were captured.
- Images were taken simultaneously from both cameras using a synchronized capture script.
- The checkerboard was moved across:
  - different positions within the frame
  - different orientations (tilt, rotation)
  - varying distances from the cameras

### Coverage Strategy
- Ensured that the checkerboard covered:
  - central regions
  - image corners
  - different depths within the scene
- This helps improve calibration accuracy across the entire field of view.

### Capture Conditions
- Adequate lighting was maintained to ensure clear corner detection.
- Motion blur was minimized by keeping the checkerboard stationary during capture.
- Both cameras maintained consistent exposure as much as possible.

### Output
- Raw stereo image pairs were saved for calibration processing.
- Images were organized into left and right directories for further processing.

### Observations
- Good calibration requires diversity in checkerboard pose, not just quantity.
- Poor coverage (e.g., only center images) leads to inaccurate distortion and depth estimation.

### Notes
- Synchronization between left and right images is important to avoid mismatch.
- Flatness of the checkerboard is critical — any bending introduces error.

## 3. Intrinsic Calibration (Per Camera)

### Purpose
To estimate the internal parameters of each camera, enabling accurate mapping between 3D world points and 2D image coordinates.

### Method
- Checkerboard corners were detected in each image using OpenCV.
- Known 3D coordinates of checkerboard corners (in the checkerboard frame) were paired with their detected 2D image coordinates.
- Calibration was performed separately for each camera using these correspondences.

### Estimated Parameters

For each camera, the following parameters were computed:

- **Camera Matrix (K)**:
  - Contains focal lengths (f_x, f_y) and principal point (c_x, c_y)
  - Defines how 3D points project onto the image plane

- **Distortion Coefficients (D)**:
  - Models lens distortion (radial and tangential)
  - Used to correct warped images

### Output
- Intrinsic parameters for both left and right cameras were saved for later use in:
  - stereo calibration
  - rectification
  - 3D reconstruction

### Observations
- Accurate corner detection is critical for reliable calibration.
- Image coverage across the frame improves estimation of distortion parameters.
- Noise in corner detection can introduce small errors in calibration.

### Notes
- Intrinsic calibration must be performed before stereo calibration.
- These parameters remain valid as long as the camera configuration does not change.

## 4. Stereo Calibration (Extrinsics)

### Purpose
To estimate the relative pose between the two cameras, enabling depth computation through triangulation.

### Method
- Corresponding checkerboard corners were detected in synchronized left and right image pairs.
- Using the intrinsic parameters of both cameras, stereo calibration was performed to estimate the geometric relationship between them.
- The calibration process minimized reprojection error across all image pairs.

### Estimated Parameters

The following stereo parameters were computed:

- **Rotation Matrix (R)**:
  - Describes the orientation of the right camera relative to the left camera
  - Encodes how the coordinate axes of one camera are rotated with respect to the other

- **Translation Vector (T)**:
  - Represents the position of the right camera relative to the left camera
  - The magnitude of T corresponds to the baseline distance between the cameras

### Output
- Stereo calibration parameters (R, T) were saved for use in:
  - stereo rectification
  - disparity computation
  - depth estimation
- Essential matrix (E) and Fundamental matrix (F) were also computed during stereo calibration.
### Observations
- Accurate intrinsic calibration is essential for reliable stereo calibration.
- Consistent and well-distributed checkerboard detections improve estimation stability.
- Errors in corner detection or synchronization can degrade calibration quality.

### Notes
- The baseline (distance between cameras) directly influences depth sensitivity:
  - larger baseline → better depth resolution at distance
  - smaller baseline → more stable matching for close objects
- These parameters remain valid as long as the relative camera positions do not change.

## 5. Stereo Rectification

### Purpose
To transform both camera images into a common image plane such that corresponding points lie on the same horizontal lines, simplifying disparity computation.

### Method
- Stereo calibration parameters (R, T, K, D) were used to compute rectification transforms.
- Rectification maps were generated for both left and right cameras.
- These maps were applied to raw images to produce rectified image pairs.

### Output
- Rectified left and right images with aligned epipolar geometry.
- Remapping functions used for real-time rectification.

### Validation
- Verified that corresponding features in left and right images lie on the same horizontal scanlines.
- Visual inspection using checkerboard patterns confirmed alignment across rows.
- Horizontal guide lines were used to confirm epipolar consistency.

### Observations
- Rectification significantly simplifies correspondence search for disparity computation.
- Proper calibration results in minimal vertical disparity after rectification.

### Notes
- Imperfect calibration can lead to slight residual vertical misalignment.
- Rectification must be applied consistently to all frames before disparity estimation.
- All downstream processing (detection, depth, 3D reconstruction) assumes rectified images.

## 6. Disparity and Depth Estimation

### Purpose
To compute pixel-wise depth information from rectified stereo image pairs using disparity.

### Method

#### Disparity Computation
- Rectified left and right images were used as input.
- Disparity was computed using OpenCV stereo matching algorithms:
  - StereoBM / StereoSGBM
- For each pixel in the left image, a corresponding pixel was searched along the same horizontal line in the right image.
- The horizontal shift between matching pixels defines the disparity value.

#### Depth Computation
- Depth was derived from disparity using stereo calibration parameters.
- The relationship between disparity and depth is inversely proportional:
  
  Greater disparity → closer object  
  Smaller disparity → farther object

- The reprojection matrix (Q) was used to convert disparity values into real-world depth measurements.

### Output
- Real-time disparity maps representing pixel-wise depth variation.
- Depth estimates for individual pixels or regions.
- Optional visualization of disparity as a heatmap for intuitive interpretation.

### Validation
- Observed that:
  - Objects closer to the camera produced higher disparity values.
  - Distant objects showed lower disparity values.
- Verified smooth depth gradients across continuous surfaces.
- Confirmed expected depth changes when objects were moved in the scene.

### Observations
- Disparity estimation performs well in textured regions.
- Low-texture or uniform surfaces result in poor matching and unreliable depth.
- Object boundaries exhibit noise due to ambiguity in correspondence.
- Some pixels produce invalid or missing disparity values.

### Notes
- Disparity quality depends heavily on lighting, texture, and calibration accuracy.
- Parameter tuning (block size, number of disparities, etc.) significantly affects results.
- Depth estimation becomes less reliable as disparity approaches zero (far objects).

## 7. Basic 3D Validation

### Purpose
To verify that the stereo pipeline can accurately recover 3D camera-frame coordinates (X, Y, Z)_cam from image points.

### Method
- An interactive validation script was used to select pixel locations (u, v) in the rectified image.
- For each selected point:
  - Depth (Z) was obtained from the disparity/depth map.
  - Intrinsic parameters were used to back-project the pixel into 3D space.

- The 3D coordinates were computed using:

  X = (u - c_x) * Z / f_x  
  Y = (v - c_y) * Z / f_y  
  Z = Z

### Depth Estimation Strategy
- Direct depth from a single pixel was found to be unstable.
- To improve robustness:
  - Depth values were sampled from a local region around (u, v)
  - Invalid values were filtered out
  - Robust statistics such as median and quartiles (Q1, Q3) were used

### Output
- Computed (u, v, Z) values for selected points
- Corresponding (X, Y, Z)_cam coordinates
- Visual annotations showing selected points and depth values

### Validation
- Verified that:
  - Z corresponds to physical distance from the camera
  - X increases/decreases with horizontal movement in the image
  - Y increases/decreases with vertical movement in the image
- Moving objects closer or farther resulted in consistent changes in Z
- Relative spatial relationships between points were preserved

### Observations
- Depth noise directly affects 3D coordinate accuracy
- Stable depth estimation requires filtering and aggregation
- Central regions of objects tend to produce more reliable depth than edges

### Notes
- This step confirms the correctness of the full pipeline:
  
  Rectification → Disparity → Depth → 3D Reprojection

- The resulting (X, Y, Z)_cam coordinates form the input for:
  - object localization (Phase 2)
  - camera-to-robot transformation (Phase 4)