"""Empirical XYZ -> RAMPS motor target predictor.

Offline by default: does NOT open serial or move the robot.
Uses calibrated radial reach/height plus a separately fitted base azimuth.
Requires numpy: pip install numpy
"""
import numpy as np

# global motor X,Y,Z, measured gripper X,Y,Z (cm)
DATA = np.array([
 [0,0,0,0,6,5], [200,0,0,0,7.5,4.5], [400,0,0,0,9,3.2],
 [400,200,0,0,10,3.5], [400,400,0,0,11.5,4],
 [400,400,200,-2,11,4], [400,400,-200,2,11,4],
 [800,400,0,0,13,1.7], [400,800,0,0,14,5.1],
 [800,800,0,0,15.2,3], [800,800,350,-4,15,2.8],
 [200,490,-250,2.5,12,5.5], [-200,490,-650,5.7,8.7,7],
 [-200,790,550,-5.7,10.5,7.8], [300,1790,-250,2.5,17.5,10.2],
 [363,544,-46,0,12.5,5.4], [624,718,176,-2.7,14,4],
], dtype=float)

SESSION_OFFSET = np.array([624, 718, 176], dtype=float)
CURRENT_SESSION = np.array([0, 0, 0], dtype=float)
CURRENT_GLOBAL = CURRENT_SESSION + SESSION_OFFSET

# A small smooth local regression predicts (reach,height) from shoulder/elbow steps.
# Base azimuth is fitted separately from motor Z.
M = DATA[:,:3]
P = DATA[:,3:]
RH = np.column_stack((np.hypot(P[:,0],P[:,1]),P[:,2]))
AZ = np.arctan2(-P[:,0],P[:,1])
# Fit base angular response through the zero-reference origin.
BASE_GAIN = np.dot(M[:,2],AZ) / np.dot(M[:,2],M[:,2])

# Normalized motor coordinates for local weighted regression
SCALE = np.array([500.,700.])

def forward_rh(xy):
    """Local affine weighted fit. Returns radial reach and gripper height."""
    q = np.asarray(xy,dtype=float)
    d = np.linalg.norm((M[:,:2]-q)/SCALE,axis=1)
    idx = np.argsort(d)[:8]
    weights = 1/(0.2+d[idx]**2)
    A = np.column_stack((np.ones(len(idx)),(M[idx,:2]-q)/SCALE))
    reg = np.diag([1e-7,0.03,0.03])
    coef = np.linalg.solve(A.T@(weights[:,None]*A)+reg,A.T@(weights[:,None]*RH[idx]))
    return coef[0]

def forward(global_motor):
    x,y,z = global_motor
    r,h = forward_rh([x,y])
    angle = BASE_GAIN*z
    return np.array([-r*np.sin(angle),r*np.cos(angle),h])

def predict(target):
    target=np.asarray(target,dtype=float)
    if target.shape != (3,):
        raise ValueError('Target must have 3 coordinates')
    r=np.hypot(target[0],target[1]); h=target[2]
    az=np.arctan2(-target[0],target[1])
    base_z=az/BASE_GAIN
    # Exhaustive coarse search in the calibrated motor X/Y rectangle,
    # then fine search around best candidate; minimize predicted error.
    mins=M[:,:2].min(axis=0); maxs=M[:,:2].max(axis=0)
    best=None
    for dx,dy in [(100,100),(20,20),(5,5)]:
        if best is None:
            xs=np.arange(mins[0],maxs[0]+1,dx)
            ys=np.arange(mins[1],maxs[1]+1,dy)
        else:
            xs=np.arange(max(mins[0],best[0]-5*dx),min(maxs[0],best[0]+5*dx)+1,dx)
            ys=np.arange(max(mins[1],best[1]-5*dy),min(maxs[1],best[1]+5*dy)+1,dy)
        for x in xs:
            for y in ys:
                rh=forward_rh([x,y])
                err=np.sum((rh-np.array([r,h]))**2)
                # small preference for configurations close to current position
                err+=0.015*np.sum(((np.array([x,y])-CURRENT_GLOBAL[:2])/SCALE)**2)
                if best is None or err < best[2]:
                    best=(x,y,err)
    steps=np.rint([best[0],best[1],base_z]).astype(int)
    return steps,forward(steps)

def main():
    print('Empirical XYZ predictor — OFFLINE ONLY; no motors will move')
    print('Current session motor position:', CURRENT_SESSION.astype(int).tolist())
    print('Base azimuth gain: %.5f degrees/step' % np.degrees(BASE_GAIN))
    while True:
        try:
            raw=input('\nTarget XYZ cm (or q): ').strip()
        except EOFError:
            break
        if raw.lower() in ('q','quit'): break
        try:
            target=np.array([float(t) for t in raw.split()])
            if len(target)!=3: raise ValueError('Enter 3 numbers')
            nearest=np.linalg.norm(P-target,axis=1).min()
            if target[1] <= 0:
                print('REJECTED: only positive-Y workspace calibrated'); continue
            if nearest>10.0:
                print('REJECTED: target is %.2f cm from nearest calibration sample (>3 cm)' % nearest)
                continue
            steps,estimated=predict(target)
            session=steps-SESSION_OFFSET.astype(int)
            delta=session-CURRENT_SESSION.astype(int)
            print('Nearest measured XYZ distance: %.2f cm' % nearest)
            print('Predicted XYZ:', np.round(estimated,2).tolist())
            print('Global motor target:', steps.tolist())
            print('Session motor target:',session.tolist())
            print('Changes from current:',delta.tolist())
            print('Manual command: goto %d %d %d' % tuple(session))
            print('WARNING: Model predictions are unverified. Check joint limits, obstacles and path before executing.')
        except (ValueError,ZeroDivisionError,np.linalg.LinAlgError) as exc:
            print('Cannot predict:',exc)
if __name__=='__main__': main()
