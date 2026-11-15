EGG_ROBOT: small XYZ integration test

Copy xyz_control.py and src/robot/kinematics.py (plus __init__.py files) into your existing Egg_Robot project.
Upload arduino/robot_xyz_serial.ino to Arduino Mega 2560, replacing the previous sketch.
Install Python dependency: pip install pyserial
From project root: python xyz_control.py --port COM5

With arm physically supported and at reference pose (base=0 chosen forward, shoulder~70 degrees above horizontal, forearm~40 degrees below horizontal), run:
  status
  arm
  demo
Type MOVE only after reviewing preview and verifying clearance.
Then run 'status' after the move.
To return to reference, enter 'goto' with the exact reference XYZ from FK, or restart after manually restoring and re-establishing reference. Do NOT simply re-arm while displaced.

SAFETY: Approximate reference and geometry, no hardware homing, no limit switches, no collision detection, no E-stop. 110 microstep software envelope only, not proven mechanical safety. Motors may miss steps. Do not disconnect powered steppers. Gravity-loaded joints must be supported. STOP decelerates and is not an emergency stop. DISARM removes holding torque. Coordinated run() updates all motors but does not guarantee synchronized arrival; use proper coordinated trajectory planning later.
