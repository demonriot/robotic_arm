"""Conservative manual joint test for EGG_ROBOT.

Usage: python robot_control.py --port COM5
Requires: pip install pyserial
"""
import argparse
import time
import serial

STEPS_PER_DEGREE = 12800 / 360
MAX_DEGREES = 200 / STEPS_PER_DEGREE


def read_responses(port, wait=0.15):
    end = time.monotonic() + wait
    while time.monotonic() < end:
        line = port.readline().decode('utf-8', errors='replace').strip()
        if line:
            print('Arduino:', line)
            if line == 'DONE':
                return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', required=True, help='e.g. COM5')
    args = parser.parse_args()
    with serial.Serial(args.port, 115200, timeout=0.05) as board:
        time.sleep(2.2)  # Mega resets on serial connection
        read_responses(board, 0.3)
        print('Commands: x 2 | x -2 | y 1 | z -1 | status | stop | disable | quit')
        print(f'Individual moves limited to +/-{MAX_DEGREES:.3f} degrees.')
        print('WARNING: stop is controlled deceleration, NOT an emergency stop.')
        while True:
            try:
                command = input('robot> ').strip().lower()
            except (KeyboardInterrupt, EOFError):
                print('\nExiting. Motors remain in their previous enable state.')
                break
            if command in ('quit', 'exit'):
                break
            if command in ('status', 'stop', 'disable'):
                board.write((command.upper() + '\n').encode())
            else:
                parts = command.split()
                if len(parts) != 2 or parts[0] not in ('x', 'y', 'z'):
                    print('Enter axis and angle, e.g. x 2, or status/stop/disable/quit')
                    continue
                try:
                    degrees = float(parts[1])
                except ValueError:
                    print('Invalid angle')
                    continue
                if not (0 < abs(degrees) <= MAX_DEGREES):
                    print(f'Angle must be nonzero and <= {MAX_DEGREES:.3f} degrees')
                    continue
                steps = round(degrees * STEPS_PER_DEGREE)
                if steps == 0:
                    print('Angle rounds to zero microsteps')
                    continue
                board.write(f'MOVE {parts[0].upper()} {steps}\n'.encode())
                print(f'Sent {steps} steps (~{steps / STEPS_PER_DEGREE:.3f} degrees)')
            read_responses(board, 0.3)


if __name__ == '__main__':
    main()
