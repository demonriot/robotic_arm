"""EGG_ROBOT cautious XYZ test. Run from project root: python xyz_control.py --port COM5"""
import argparse
import math
import time
import serial
from src.robot.kinematics import ArmGeometry, JointAngles, Point3D, forward_kinematics, inverse_kinematics, UnreachableTarget

G = ArmGeometry()
REF = JointAngles(base_deg=0, shoulder_deg=70, elbow_deg=-110)
SPD = G.joint_microsteps_per_rev / 360
LIMIT_STEPS = 110
SIGN = {'base': 1, 'shoulder': -1, 'elbow': 1}

def target_steps(a):
    return {
        'X': round((a.shoulder_deg-REF.shoulder_deg)*SPD*SIGN['shoulder']),
        'Y': round((a.elbow_deg-REF.elbow_deg)*SPD*SIGN['elbow']),
        'Z': round((a.base_deg-REF.base_deg)*SPD*SIGN['base']),
    }

def safe(a):
    st=target_steps(a)
    if any(abs(v)>LIMIT_STEPS for v in st.values()):
        return False, 'Outside initial +/-110 microstep reference envelope'
    if not (40 <= a.shoulder_deg <= 70):
        return False, 'Shoulder outside estimated 40..70 deg range'
    forearm=a.shoulder_deg+a.elbow_deg
    if not (-60 <= forearm <= -10):
        return False, 'Forearm outside estimated -60..-10 deg range'
    return True, ''

def request(ser, cmd):
    ser.write((cmd+'\n').encode('ascii'))
    end=time.monotonic()+3
    while time.monotonic()<end:
        s=ser.readline().decode('ascii','replace').strip()
        if s and s.startswith(('STATUS','OK','ERR')):
            print('Arduino:',s)
            return s
    raise TimeoutError('No response; do not assume movement was executed')

def status_fields(line):
    return {k:v for item in line.split() if '=' in item for k,v in [item.split('=',1)]}

def angle_from_steps(st):
    return JointAngles(
        REF.base_deg + st['Z']/SPD,
        REF.shoulder_deg - st['X']/SPD,
        REF.elbow_deg + st['Y']/SPD)

def preview(ser, point):
    state=status_fields(request(ser,'STATUS'))
    if state.get('armed')!='1' or state.get('busy')!='0':
        print('REJECTED: arm must be armed and idle');return
    current_steps={k:int(state[k]) for k in ('X','Y','Z')}
    current=angle_from_steps(current_steps)
    try:
        # Reference pose has negative relative elbow angle: select negative branch.
        desired=inverse_kinematics(point,G,elbow_branch='down')
    except UnreachableTarget as exc:
        print('REJECTED:',exc);return
    goal=target_steps(desired)
    for i in range(21):
        t=i/20
        a=JointAngles(*(c+(d-c)*t for c,d in zip(
            (current.base_deg,current.shoulder_deg,current.elbow_deg),
            (desired.base_deg,desired.shoulder_deg,desired.elbow_deg))))
        ok,reason=safe(a)
        if not ok:
            print('REJECTED: path sample',i,'-',reason);return
    deltas={k:goal[k]-current_steps[k] for k in goal}
    actual=forward_kinematics(desired,G)
    print('Target:',point,'\nIK:',desired,'\nFK check:',actual)
    print('Current steps:',current_steps,'Goal steps:',goal,'Delta steps:',deltas)
    print('WARNING: Only joint-space samples checked; no collision, torque, or physical limit sensing.')
    if input('Type MOVE to execute, anything else to cancel: ').strip()!='MOVE':
        print('Cancelled');return
    result=request(ser,f"MOVE3 {goal['X']} {goal['Y']} {goal['Z']}")
    if result.startswith('ERR'):print('Movement rejected by Arduino')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--port',required=True);args=ap.parse_args()
    print('Reference: base 0, shoulder 70, elbow -110. Arm MUST be physically at this pose before ARM.')
    print('Only tiny movements (+/-110 microsteps per axis) are allowed. STOP is not emergency stop.')
    with serial.Serial(args.port,115200,timeout=.2) as ser:
        time.sleep(2.5);ser.reset_input_buffer()
        print('Commands: status, arm, demo, goto X Y Z, stop, disarm, quit')
        while True:
            try:
                cmd=input('xyz> ').strip().lower().split()
                if not cmd:continue
                if cmd[0] in ('quit','exit'):break
                if len(cmd)==1 and cmd[0] in ('status','arm','stop','disarm'):
                    if cmd[0]=='arm' and input('Arm supported and at reference pose? Type YES: ')!='YES':continue
                    if cmd[0]=='disarm' and input('Gravity joints physically supported? Type YES: ')!='YES':continue
                    request(ser,cmd[0].upper());continue
                if cmd==['demo']:
                    # Small downward shoulder change, elbow relative angle constant.
                    preview(ser,forward_kinematics(JointAngles(0,68,-110),G));continue
                if len(cmd)==4 and cmd[0]=='goto':
                    preview(ser,Point3D(*map(float,cmd[1:])));continue
                print('Unknown command')
            except (ValueError,TimeoutError,KeyError) as exc:print('Error:',exc)
            except KeyboardInterrupt:print('\nInterrupted. No emergency stop issued.');break
if __name__=='__main__':main()
