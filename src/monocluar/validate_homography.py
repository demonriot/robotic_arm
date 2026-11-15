import csv
from pathlib import Path
import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / 'configs' / 'monocular.yaml', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
NX = int(cfg['calibration']['checkerboard']['Nx'])
NY = int(cfg['calibration']['checkerboard']['Ny'])
S = float(cfg['calibration']['checkerboard']['square_size_cm'])
assert NX >= 2 and NY >= 2 and S > 0
image_dir = ROOT / cfg['paths']['capture_dir']
out_dir = ROOT / 'outputs' / 'monocular' / 'validation'
out_dir.mkdir(parents=True, exist_ok=True)
params = np.load(ROOT / 'outputs' / 'monocular' / 'homography.npz')
H = params['H']
expected_size = tuple(map(int, params['image_size']))
world_grid = np.mgrid[0:NX, 0:NY].T.reshape(-1, 2).astype(np.float64) * S
rows = []
all_h, all_v, all_fit = [], [], []
images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
if not images:
    raise RuntimeError(f'No images found in {image_dir}')

for path in images:
    frame = cv2.imread(str(path))
    if frame is None:
        continue
    h, w = frame.shape[:2]
    if (w, h) != expected_size:
        print(f'SKIP {path.name}: {w}x{h} differs from calibration {expected_size}')
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(gray, (NX, NY))
    if not found:
        print(f'SKIP {path.name}: checkerboard not found')
        continue
    pts = corners.reshape(NY, NX, 2)
    projected = cv2.perspectiveTransform(corners.astype(np.float64), H).reshape(NY, NX, 2)
    horizontal = np.linalg.norm(projected[:, 1:, :] - projected[:, :-1, :], axis=2).ravel()
    vertical = np.linalg.norm(projected[1:, :, :] - projected[:-1, :, :], axis=2).ravel()
    # A symmetric 7x7 corner pattern can be indexed in four orientations.
    # Compare to the best of the four 180/90-degree-compatible corner orderings.
    candidates = [world_grid.reshape(NY, NX, 2)]
    if NX == NY:
        base = candidates[0]
        candidates = [np.rot90(base, k).copy() for k in range(4)]
    else:
        candidates.append(candidates[0][::-1, ::-1, :])
    # The corner-index ambiguity also includes reflections depending on board orientation.
    # Spacing validation is independent of that ambiguity.
    fit_rmse = min(float(np.sqrt(np.mean(np.sum((projected - c)**2, axis=2)))) for c in candidates)
    all_h.extend(horizontal.tolist())
    all_v.extend(vertical.tolist())
    all_fit.append(fit_rmse)
    rows.append([path.name, len(corners), horizontal.mean(), vertical.mean(),
                np.max(np.abs(horizontal-S)), np.max(np.abs(vertical-S)), fit_rmse])
    preview = frame.copy()
    cv2.drawChessboardCorners(preview, (NX, NY), corners, True)
    origin = tuple(np.round(pts[0, 0]).astype(int))
    cv2.circle(preview, origin, 9, (0, 0, 255), 3)
    cv2.putText(preview, 'First detected corner', (max(5, origin[0]-75), max(20, origin[1]-15)),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)
    cv2.imwrite(str(out_dir / f'{path.stem}_corners.jpg'), preview)
    print(f'{path.name}: horizontal={horizontal.mean():.3f} cm, '
        f'vertical={vertical.mean():.3f} cm, fit RMSE={fit_rmse:.3f} cm')

with open(out_dir / 'report.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['image', 'detected_corners', 'mean_horizontal_cm', 'mean_vertical_cm',
                    'max_horizontal_abs_error_cm', 'max_vertical_abs_error_cm', 'fit_rmse_cm'])
    writer.writerows(rows)
if not rows:
    raise RuntimeError('No valid checkerboard images found')
print('\n--- Overall results ---')
for name, values in [('Horizontal', all_h), ('Vertical', all_v)]:
    a = np.asarray(values)
    print(f'{name}: mean={a.mean():.3f} cm; mean abs error={np.mean(np.abs(a-S)):.3f} cm; '
        f'max abs error={np.max(np.abs(a-S)):.3f} cm')
print(f'Mean per-image grid fit RMSE: {np.mean(all_fit):.3f} cm')
print(f'Report saved: {out_dir / "report.csv"}')
print('\nNote: Checkerboard motion relative to the camera between images invalidates a single fixed-plane homography.')
