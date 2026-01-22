from helpers.lx16a import LX16A, ServoTimeoutError

LX16A.initialize("/dev/ttyUSB0")

try:
    servo1 = LX16A(0)
except ServoTimeoutError as e:
    print(f"Servo {e.id_} is not responding. Exiting...")
    quit()
