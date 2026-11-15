"""Safe, small-step reference-pose joint test; no absolute homing or IK moves."""
import argparse
import time
import serial

STEPS_PER_DEGREE = 12800 / 360
JOINTS = {
    'base': {'axis': 'Z', 'sign': 1, 'reference': 0.0},
    'shoulder': {'axis': 'X', 'sign': -1, 'reference': 70.0},
    'elbow': {'axis': 'Y', 'sign': 1, 'reference': -110.0},
}

def exchange(ser, cmd):
    ser.write((cmd + '\n').encode('ascii'))
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        line = ser.readline().decode('ascii', errors='replace').strip()
        if line:
            print('Arduino:', line)
            if line.startswith(('OK', 'ERR', 'STATUS')):
                return line
    raise TimeoutError('No reply from Arduino')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', required=True, help='e.g. COM5')
    args = parser.parse_args()
    with serial.Serial(args.port, 115200, timeout=0.2) as ser:
        time.sleep(2.5)  # Mega resets on serial connection
        ser.reset_input_buffer()
        print('This is a small-range test only. Support the arm; check clearances.')
        print('Commands: status, arm, shoulder 68, shoulder 70, base 1, elbow -109, stop, disarm, quit')
        print('Reference pose: base 0, shoulder +70, relative elbow -110 degrees.')
        print('ARM only while physically at that reference pose. DISARM only if mechanically supported.')
        while True:
            try:
                command = input('robot> ').strip().lower()
                if command in ('quit', 'exit'):
                    break
                if command in ('status','arm','stop','disarm'):
                    exchange(ser, command.upper())
                    continue
                tokens=command.split()
                if len(tokens)!=2 or tokens[0] not in JOINTS:
                    print('Unknown command')
                    continue
                name, raw=tokens
                target=float(raw)
                cfg=JOINTS[name]
                # Small test envelope, not validated mechanical joint limits.
                if abs(target-cfg['reference'])>3.0:
                    print('REJECTED: target more than 3 degrees from approximate reference')
                    continue
                response=exchange(ser,'STATUS')
                if 'armed=1' not in response or 'busy=0' not in response:
                    print('Not armed or motor busy; wait and retry')
                    continue
                axis=cfg['axis']
                fields=dict(x.split('=',1) for x in response.split() if '=' in x)
                current=int(fields[axis])
                goal=round((target-cfg['reference'])*STEPS_PER_DEGREE*cfg['sign'])
                delta=goal-current
                if delta==0:
                    print('Already at requested commanded position')
                elif abs(delta)>110:
                    print('REJECTED: movement exceeds 110 microsteps')
                else:
                    exchange(ser,f'MOVE {axis} {delta}')
            except (ValueError, TimeoutError) as exc:
                print('Error:',exc)
            except KeyboardInterrupt:
                print('\nKeyboard interrupt. STOP is not an emergency stop.')
                break

if __name__=='__main__':
    main()
