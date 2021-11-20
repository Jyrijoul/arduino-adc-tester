import time
import serial
from serial.serialwin32 import Serial
import serial.tools.list_ports
import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType

# Serial port constants
BAUD_RATE = 115200
TIMEOUT = None

# DAQ devices
system = nidaqmx.system.System.local()
system.driver_version

print("Available DAQ devices:")
for device in system.devices:
    print("\t", device)

# Serial devices (Arduinos)
serial_devices = serial.tools.list_ports.comports()
arduino_port = "COM1"  # Select COM1 as a starting point.

print("Available DAQ devices:")
for device in serial_devices:
    print("\t", device.device, "(Arduino Nano)" if "CH340" in device.description else "")
    if "CH340" in device.description:
        arduino_port = device.device

measurements = []

try:
    with serial.Serial(port=arduino_port, baudrate=BAUD_RATE, timeout=TIMEOUT) as ser:

        previous_time = time.perf_counter_ns()
        current_time = time.perf_counter_ns()
        while True:
            value = ser.readline().decode().strip()
            measurements.append(int(value))
            current_time = time.perf_counter_ns()
            print(f"Δt = {(current_time - previous_time) / 1000}, us, f = {1 / ((current_time - previous_time) / 1000000000)}")
            previous_time = current_time
            print(value)

except KeyboardInterrupt:
    print("Closing the program...")

    if len(measurements) > 0:
        # print(measurements)
        print(f"Measurements average = {np.mean(measurements)}±{np.std(measurements)}, based on {len(measurements)} measurements.")

# sample_rate = 1.0

# with nidaqmx.Task() as task:
#     task.ai_channels.add_ai_voltage_chan("Dev1/ai0")
#     task.timing.cfg_samp_clk_timing(sample_rate, samps_per_chan=3)

#     task.start()
#     for i in range(3):
#         output = task.read()
#         print(output)
    
#     task.stop()
