from os import write
import time
import serial
import serial.tools.list_ports
import numpy as np
import pandas as pd
import nidaqmx
from nidaqmx.stream_writers import AnalogSingleChannelWriter
from nidaqmx.constants import AcquisitionType

from serial_device import SerialDevice
from ni_usb_6211 import NiUsb6211

class Tester:
    def __init__(self, input_device, output_device) -> None:
        pass

# Serial port constants
BAUD_RATE = 115200
TIMEOUT = 10

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

n = 10000
data = np.linspace(0, 5, n)
measurements = np.zeros(n)
times = np.zeros(n)

# try:
#     with serial.Serial(port=arduino_port, baudrate=BAUD_RATE, timeout=TIMEOUT) as ser:

#         previous_time = time.perf_counter_ns()
#         current_time = time.perf_counter_ns()
#         while True:
#             value = ser.readline().decode().strip()
#             measurements.append(int(value))
#             current_time = time.perf_counter_ns()
#             print(f"Δt = {(current_time - previous_time) / 1000}, us, f = {1 / ((current_time - previous_time) / 1000000000)}")
#             previous_time = current_time
#             print(value)

# except KeyboardInterrupt:
#     print("Closing the program...")

#     if len(measurements) > 0:
#         # print(measurements)
#         print(f"Measurements average = {np.mean(measurements)}±{np.std(measurements)}, based on {len(measurements)} measurements.")

try:
    with nidaqmx.Task() as task:
        task.ao_channels.add_ao_voltage_chan("Dev1/ao1")
        # task.timing.cfg_samp_clk_timing(1000, samps_per_chan=10000)
        writer = AnalogSingleChannelWriter(task.out_stream)

        # task.start()
        # samples_written = writer.write_many_sample(data)
        # task.stop()

        with serial.Serial(port=arduino_port, baudrate=BAUD_RATE, timeout=TIMEOUT) as ser:

            writer.write_one_sample(0)
            ser.write("abcdef".encode())
            value = ser.readline().decode().strip()
                # measurements.append(int(value))
            print("Reading", value)
            
            start_time = time.perf_counter()
            
            previous_time = time.perf_counter_ns()
            current_time = time.perf_counter_ns()

            for i in range(len(data)):
                writer.write_one_sample(data[i])
                print(f"Wrote sample {data[i]}.")

                ser.write("0".encode())
                ser.flushOutput()
                # time.sleep(0.1)
                value = ser.readline().decode().strip()
                # measurements.append(int(value))
                print(value)
                measurements[i] = int(value)
                current_time = time.perf_counter_ns()
                times[i] = current_time
                # print(f"Δt = {(current_time - previous_time) / 1000}, us, f = {1 / ((current_time - previous_time) / 1000000000)}")
                previous_time = current_time

            stop_time = time.perf_counter()
        print("Total time:", stop_time - start_time, " s.")
        df = pd.DataFrame({"t": times, "vout": data, "code": measurements})
        # df.to_csv("measurements.csv", index=False)
        # print("Saved measurements to file.")
except KeyboardInterrupt:
    print("Closing the program...")

    if len(measurements) > 0:
        # print(measurements)
        print(f"Measurements average = {np.mean(measurements)}±{np.std(measurements)}, based on {len(measurements)} measurements.")


    # print(f"Samples written: {samples_written}.")

# sample_rate = 1.0

# with nidaqmx.Task() as task:
#     task.ai_channels.add_ai_voltage_chan("Dev1/ai0")
#     task.timing.cfg_samp_clk_timing(sample_rate, samps_per_chan=3)

#     task.start()
#     for i in range(3):
#         output = task.read()
#         print(output)
    
#     task.stop()
