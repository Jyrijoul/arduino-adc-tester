import time
import numpy as np
import pandas as pd
from datetime import datetime
from output_device import OutputDeviceInterface
from input_device import InputDeviceInterface
from ni_usb_6211 import NiUsb6211
from serial_device import SerialDevice
import cProfile


class Tester:
    def __init__(
        self, input_device, output_device, reference_voltage_estimate=5.0, profiling=False, verbose=True
    ) -> None:
        # print(issubclass(type(input_device), InputDeviceInterface))
        # print(issubclass(type(output_device), OutputDeviceInterface))
        assert issubclass(
            type(input_device), InputDeviceInterface
        ), f"The input device {input_device} must be a subclass of {InputDeviceInterface}."
        assert issubclass(
            type(output_device), OutputDeviceInterface
        ), f"The output device {output_device} must be a subclass of {OutputDeviceInterface}."

        self.input_device = input_device
        self.output_device = output_device
        self.reference_voltage_estimate = reference_voltage_estimate
        self.profiling = profiling
        self.verbose = verbose

        # Profile the code.
        if self.profiling:
            self.cp = cProfile.Profile()

    def init(self):
        self.input_device.init()
        self.output_device.init()

    def deinit(self):
        self.input_device.deinit()
        self.output_device.deinit()

    def print_loading_bar(idx, max_idx, bar_count=10):
        if idx % (int(max_idx / bar_count)) == 0:
            print("#", end="")

    def run_tests(self, output_file_prefix="measurements", output_file_extension="csv"):
        n = 10000
        output_data = np.linspace(0, 5, n)
        measurements = np.zeros(n)
        times = np.zeros(n)

        # If the output device has methods to measure
        # both the input device's reference voltage
        # and also its own output voltage,
        # then use them while performing measurements;

        # Create the required arrays regardless.
        # No modifying when the necessary methods do not exist.
        measured_references = np.full(n, self.reference_voltage_estimate)
        measured_outputs = np.copy(output_data)

        measure_output = False
        measure_reference = False

        if hasattr(self.output_device, "get_measured_output_voltage") and callable(
            self.output_device.get_measured_output_voltage
        ):
            measure_output = True

        if hasattr(self.output_device, "get_reference_voltage") and callable(
            self.output_device.get_reference_voltage
        ):
            measure_reference = True

        start_time = (
            time.perf_counter()
        )  # For total measurement time with lesser accuracy

        if not self.verbose:
            bar_count = 100
            print("_" * bar_count)

        # Start profiling.
        if self.profiling:
            self.cp.enable()

        for i in range(len(output_data)):
            self.output_device.write_sample(output_data[i])

            if measure_output:
                measured_outputs[i] = self.output_device.get_measured_output_voltage()
            if measure_reference:
                measured_references[i] = self.output_device.get_reference_voltage()

            measurements[i] = self.input_device.read_sample()
            if self.verbose:
                print(
                    f"Wrote sample {output_data[i]}; measured output = {measured_outputs[i]}; measured reference {measured_references[i]}."
                )
                print(f"Read code {measurements[i]}.")
            else:
                Tester.print_loading_bar(i, n, bar_count=bar_count)

            current_time = time.perf_counter_ns()
            times[i] = current_time
            # print(f"Δt = {(current_time - previous_time) / 1000}, us, f = {1 / ((current_time - previous_time) / 1000000000)}")
            previous_time = current_time

        # Stop profiling.
        if self.profiling:
            self.cp.disable()

        # Complete the loading bar
        if not self.verbose:
            print()
        stop_time = time.perf_counter()
        print("Total time:", stop_time - start_time, "s.")
        df = pd.DataFrame(
            {
                "t": times,
                "vout": output_data,
                "code": measurements,
                "vout_meas": measured_outputs,
                "vref": measured_references,
            }
        )
        df.to_csv(f'{output_file_prefix}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}_.{output_file_extension}', index=False)
        print("Saved measurements to file.")

        # Dump profiling statistics.
        if self.profiling:
            self.cp.dump_stats("profiling_2")


if __name__ == "__main__":
    tester = Tester(SerialDevice(), NiUsb6211(output_channel="ao1"), verbose=False)
    tester.init()
    tester.run_tests()
    tester.deinit()
