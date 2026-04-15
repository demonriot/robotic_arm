# 🧠 Robot Arm Pick-and-Place System (Stereo Vision + IK)

---

## 🟩 PHASE 1 — Stereo System Setup

### 🔧 1.1 Hardware Setup
- [x] Select two identical cameras (preferably same model)
- [x] Design and build rigid stereo mount
  - [x] Fix baseline (~6–10 cm)
  - [x] Ensure no relative movement between cameras
- [x] Align cameras approximately
  - [x] Same height
  - [x] Parallel orientation (roughly)
- [x] Mount cameras above workspace (prefer top-down view)
- [x] Verify both cameras work independently

---

### 📸 1.2 Stereo Data Capture
- [x] Write script to capture synchronized frames from both cameras
- [x] Save stereo image pairs
- [x] Ensure:
  - [x] Good lighting
  - [x] No motion blur
  - [x] Both images clearly visible

---

### 🧮 1.3 Camera Calibration (Intrinsics)
- [x] Print checkerboard pattern
- [x] Capture ~20–30 stereo image pairs with checkerboard
  - [x] Different angles
  - [x] Different positions
- [x] Detect checkerboard corners in both cameras
- [x] Calibrate each camera separately
  - [x] Compute: K1, D1 and K2, D2
- [x] Save intrinsic parameters

---

### 🔁 1.4 Stereo Calibration (Extrinsics between cameras)
- [x] Use stereo calibration function
- [x] Compute:
  - [x] R (rotation between cameras)
  - [x] T (translation between cameras)
- [x] Save stereo parameters

---

### 📏 1.5 Stereo Rectification
- [x] Rectify stereo images
- [x] Verify:
  - [x] Corresponding points lie on same horizontal lines
- [x] Save rectification maps

---

### 🧊 1.6 Disparity + Depth
- [x] Implement disparity computation
  - [x] Start with StereoBM or StereoSGBM
- [x] Convert disparity → depth map
- [x] Validate:
  - [x] Near objects = higher disparity
  - [x] Depth values are reasonable

---

### ✅ 1.7 Basic 3D Validation
- [x] Pick a known point in image
- [x] Compute (X, Y, Z)_camera
- [x] Move object and verify consistency
- [x] Measure approximate depth accuracy

---

## 🟩 PHASE 2 — Object Detection + Localization

### 🧠 2.1 Object Detection
- [ ] Setup detection model (YOLO or simple OpenCV)
- [ ] Train / fine-tune for:
  - [ ] egg OR surrogate object (ball, plastic egg)
- [ ] Verify detection works reliably

---

### 📍 2.2 Depth Extraction
- [ ] Extract bounding box from detector
- [ ] Compute center pixel (u, v)
- [ ] Extract ROI from depth map
- [ ] Compute robust depth:
  - [ ] median depth (ignore invalid values)

---

### 🌐 2.3 3D Point Computation
- [ ] Convert (u, v, Z) → (X, Y, Z)_camera
- [ ] Validate:
  - [ ] Move object → check consistency
  - [ ] Measure noise

---

## 🟩 PHASE 3 — Robot Arm Setup (ESP32 + Servos)

### 🔧 3.1 Hardware Setup
- [ ] Assemble robot arm
- [ ] Connect servos to ESP32
- [ ] Implement PWM control
- [ ] Test individual joint movement

---

### 📡 3.2 Communication
- [ ] Setup serial communication (Python ↔ ESP32)
- [ ] Send angle commands from Python
- [ ] Confirm execution on ESP32

---

### 🧭 3.3 Define Joint Configuration
- [ ] Define all joints:
  - [ ] Number of DOF
  - [ ] Joint types
- [ ] Define:
  - [ ] Zero position
  - [ ] Angle limits
  - [ ] Offsets

---

### 📐 3.4 Forward Kinematics (FK)
- [ ] Define link lengths
- [ ] Implement FK function:
  - [ ] input: joint angles
  - [ ] output: (X, Y, Z)_robot
- [ ] Validate:
  - [ ] Move arm → compare expected vs actual

---

## 🟩 PHASE 4 — Camera ↔ Robot Calibration (Extrinsics)

### 🎯 4.1 Calibration Setup
- [ ] Place calibration board or markers on table
- [ ] Ensure visible in stereo system

---

### 🤖 4.2 Data Collection
- [ ] For each calibration point:
  - [ ] Move robot tip to point
  - [ ] Record joint angles
  - [ ] Compute P_robot via FK
  - [ ] Detect same point → compute P_cam
- [ ] Collect ~15–30 pairs

---

### 🧮 4.3 Compute R and T
- [ ] Implement SVD-based rigid transform
- [ ] Solve:
  - [ ] P_robot = R * P_cam + T
- [ ] Save R and T

---

### 📏 4.4 Validate Calibration
- [ ] Transform test points:
  - [ ] P_cam → P_robot_pred
- [ ] Compare with actual robot positions
- [ ] Compute error (RMSE)
- [ ] Ensure acceptable accuracy (< ~2–3 cm)

---

## 🟩 PHASE 5 — Inverse Kinematics (IK)

### 🧠 5.1 IK Design
- [ ] Define IK problem:
  - [ ] input: (X, Y, Z)_robot
  - [ ] output: joint angles
- [ ] Start with simplified constraints:
  - [ ] fixed gripper orientation

---

### 📐 5.2 Implement IK
- [ ] Derive analytical IK (preferred)
  OR
- [ ] Implement numerical IK (fallback)
- [ ] Validate:
  - [ ] IK → FK → verify position

---

## 🟩 PHASE 6 — Integrated Pipeline

### 🔗 6.1 Full Pipeline
- [ ] Stereo → depth
- [ ] Detection → (u, v)
- [ ] Depth → Z
- [ ] Back-project → P_cam
- [ ] Transform → P_robot
- [ ] IK → joint angles
- [ ] Send to ESP32

---

### 🧪 6.2 Basic Pick Test
- [ ] Move to above object
- [ ] Lower to grasp
- [ ] Close gripper
- [ ] Lift
- [ ] Move to drop zone
- [ ] Release

---

### 🔁 6.3 Robustness
- [ ] Add retry if detection fails
- [ ] Add threshold on depth confidence
- [ ] Add safety offsets (approach height)

---

## 🟩 PHASE 7 — Refinement

### 📉 7.1 Error Handling
- [ ] Handle missing depth values
- [ ] Handle detection failure
- [ ] Handle unreachable IK targets

---

### 🧠 7.2 Improvements
- [ ] Add smoothing to depth
- [ ] Add filtering to detections
- [ ] Add logging for debugging

---

### 🥚 7.3 Final Target
- [ ] Replace object with egg
- [ ] Adjust grip strategy
- [ ] Test multiple scenarios

---

## 🚀 Optional (Advanced)

- [ ] Add uncertainty modeling
- [ ] Add feedback loop (re-detect after grasp)
- [ ] Add trajectory planning
- [ ] Add multi-object selection