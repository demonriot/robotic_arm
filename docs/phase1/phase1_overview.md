# Phase 1 Overview — Stereo Vision Setup and 3D Validation

## Goal

The goal of Phase 1 was to build a calibrated stereo vision pipeline capable of estimating depth and recovering 3D coordinates of points in the camera reference frame.

Given a pixel location (u, v) in the rectified image, the system should be able to compute its corresponding 3D position (X, Y, Z)_cam using stereo disparity and camera calibration parameters.

This forms the foundation for downstream tasks such as object localization, camera-to-robot transformation, and robotic manipulation in later phases.

## Scope Completed

The following components of the stereo vision pipeline were implemented and verified:

### 1. Hardware Setup and Alignment
- Mounted two cameras on a rigid stereo rig with a fixed baseline.
- Ensured approximate parallel alignment and consistent height.
- Verified both camera streams independently.

### 2. Synchronized Stereo Image Capture
- Implemented a script to capture synchronized image pairs from both cameras.
- Ensured consistent lighting and minimal motion blur during capture.

### 3. Intrinsic Calibration (Per Camera)
- Captured multiple checkerboard images across different orientations.
- Computed intrinsic parameters (K, D) for both cameras using detected corner points.

### 4. Stereo Calibration (Extrinsics)
- Estimated relative rotation (R) and translation (T) between the two cameras.
- Established geometric relationship required for depth computation.

### 5. Stereo Rectification
- Generated rectification maps to align image planes.
- Verified that corresponding points lie on horizontal epipolar lines.

### 6. Disparity and Depth Estimation
- Implemented real-time disparity computation using OpenCV (StereoBM / StereoSGBM).
- Converted disparity maps into depth estimates.
- Verified expected behavior: closer objects produce higher disparity.

### 7. Basic 3D Point Validation
- Implemented interactive validation to select image points.
- Computed corresponding (X, Y, Z)_cam using calibration and disparity.
- Verified approximate consistency of depth under object movement.

## System Outputs

The Phase 1 pipeline produces the following key outputs at different stages:

### 1. Calibration Parameters
- Intrinsic parameters for both cameras:
  - Camera matrices (K1, K2)
  - Distortion coefficients (D1, D2)
- Stereo extrinsics:
  - Rotation matrix (R)
  - Translation vector (T)
- Rectification and reprojection data:
  - Rectification maps
  - Q matrix for 3D reprojection

These parameters are stored in:
[data/calibration/params/](file:///d:/Faisal/Egg_Robot/data/calibration/params/)

These form the foundation for all subsequent depth and 3D computations.

---

### 2. Rectified Image Pairs
- Left and right images transformed into a common image plane
- Epipolar alignment ensures corresponding points lie on the same horizontal lines

These are optionally saved for verification and debugging in:
[outputs/calibration/rectified_frames/](file:///d:/Faisal/Egg_Robot/outputs/calibration/rectified_frames/)

---

### 3. Disparity Maps
- Pixel-wise disparity between left and right images
- Encodes relative depth information (inverse relationship with distance)

Generated in real-time and optionally stored in:
[outputs/disparity/](file:///d:/Faisal/Egg_Robot/outputs/disparity/)

---

### 4. Depth Estimates
- Depth values derived from disparity using calibration parameters
- Represent distance from camera to scene points

These are used internally for 3D computation and can be visualized or logged.

---

### 5. 3D Camera-Space Points
- Conversion of image points (u, v) into (X, Y, Z)_cam coordinates
- Uses depth and intrinsic parameters

Generated during validation using:
- interactive point selection
- robust depth estimation methods (e.g., median / quartiles)

---

### 6. Validation Outputs
- Annotated images showing selected points and computed depth
- Console logs of (u, v, Z) and (X, Y, Z)_cam values

Stored in:
[outputs/validation/](file:///d:/Faisal/Egg_Robot/outputs/validation/)

These outputs were used to verify correctness and consistency of the stereo pipeline.

## Validation and Observations

The stereo vision pipeline was validated both qualitatively and through interactive depth inspection.

### 1. Rectification Validation
- Verified that corresponding points in left and right images lie on the same horizontal scanlines.
- Checkerboard corners and scene features appeared aligned after rectification.
- Minor residual misalignment may exist due to limited calibration coverage.

---

### 2. Disparity Behavior
- Observed expected disparity-depth relationship:
  - Closer objects produced higher disparity values.
  - Distant objects produced lower disparity values.
- Depth gradients across the scene were visually consistent.

However:
- Disparity estimation degraded in low-texture regions.
- Object edges showed noise and discontinuities.
- Some regions produced invalid or missing disparity values.

---

### 3. Depth Consistency
- Depth values were extracted for selected pixels and compared across multiple frames.
- Moving an object closer/farther resulted in corresponding changes in estimated depth.
- Depth remained approximately consistent for static objects across frames.

However:
- Noise was observed in regions with weak disparity signals.
- Depth accuracy varied depending on surface texture and lighting.

---

### 4. 3D Point Validation
- Selected image points (u, v) were converted to (X, Y, Z)_cam coordinates.
- Verified that:
  - Z corresponded to physical distance from the camera.
  - X and Y varied consistently with horizontal and vertical movement in the image.
- Relative positioning of points in 3D space appeared coherent.

---

### 5. Effect of Depth Sampling Strategy
- Depth estimation from a single pixel was found to be unstable.
- More robust estimates were obtained using:
  - median depth within a local region
  - quartile-based filtering (Q1 / Q3)
- This motivated the use of ROI-based depth extraction in future phases.

---

### 6. Observed Failure Modes
- Textureless surfaces → unreliable disparity
- Reflective or uniform regions → noisy or missing depth
- Depth discontinuities at object boundaries
- Sensitivity to lighting conditions and exposure differences between cameras

---

### Summary

The stereo pipeline successfully produces consistent depth estimates and enables recovery of 3D camera-frame coordinates. 

While the system is sufficiently reliable for controlled environments, robustness improvements are required for real-world object localization, particularly in handling noise and invalid depth regions.

## Limitations and Next Steps

### Current Limitations

While the stereo pipeline successfully estimates depth and recovers 3D camera-frame coordinates, several limitations remain:

- **No Object Awareness**
  - The system operates on manually selected pixels.
  - It cannot automatically detect or localize objects in the scene.

- **Depth Noise and Instability**
  - Depth estimation is sensitive to:
    - low-texture regions
    - lighting variations
    - object boundaries
  - Single-pixel depth values are unreliable without filtering.

- **No Robust Target Localization**
  - There is no mechanism to associate depth with a meaningful target (e.g., an object to grasp).
  - Spatial understanding is limited to individual points rather than object-level reasoning.

- **No Camera-to-Robot Mapping**
  - The computed (X, Y, Z)_cam coordinates are not yet transformed into the robot's coordinate frame.
  - The system cannot yet guide robotic motion.

---

### Next Steps (Phase 2 — Object Detection and Localization)

The next phase focuses on extending the stereo pipeline to enable automatic object localization:

- **Object Detection**
  - Integrate a detection model (e.g., YOLOv8) to identify target objects in the image.
  - Extract bounding boxes or segmentation masks.

- **Depth Extraction from Regions**
  - Replace single-pixel depth estimation with region-based methods:
    - extract depth values from within detected object regions
    - apply robust statistics (median / quartiles)

- **3D Localization**
  - Compute (X, Y, Z)_cam for detected objects using:
    
    bounding box / mask → ROI → robust depth → 3D point

- **Validation**
  - Verify consistency of object position under movement.
  - Evaluate noise and stability of estimated coordinates.

---

### Long-Term Integration

These improvements will enable the transition toward:

- Camera-to-robot calibration (Phase 4)
- Inverse kinematics for motion planning (Phase 5)
- Full pick-and-place pipeline (Phase 6)