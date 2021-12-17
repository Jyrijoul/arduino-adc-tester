import time
import os
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
        self, input_device: InputDeviceInterface, output_device: OutputDeviceInterface, reference_voltage_estimate=5.0, profiling=False, output_file_extension="csv", add_date=False, verbose=True
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
        self.output_file_extension = output_file_extension
        self.add_date = add_date
        self.verbose = verbose

        # Profile the code.
        if self.profiling:
            self.cp = cProfile.Profile()

        # To not perform a double deinit.
        self.is_initialized = False

    def init(self, trigger_mode="software"):
        """Initializes all the devices with the correct triggering mode.

        Parameters
        ----------
        mode : str, optional
            Whether to use software or hardware triggering.
            Possible values are "software" (default) and "hardware".
        """

        self.input_device.init(mode=trigger_mode)
        self.output_device.init(mode=trigger_mode)
        self.is_initialized = True

    def deinit(self):
        if self.is_initialized:
            self.input_device.deinit()
            self.output_device.deinit()
            self.is_initialized = False

    def start_profiling(self):
        if self.profiling:
            self.cp.enable()

    def stop_profiling(self, output_file_prefix: str):
        if self.profiling:
            self.cp.disable()
            # Dump profiling statistics.
            self.cp.dump_stats("profiling_" + output_file_prefix)

    def print_loading_bar(idx, max_idx, bar_count=10):
        if idx % (int(max_idx / bar_count)) == 0:
            print("#", end="")

    def save_file(self, dataframe: pd.DataFrame, output_file_prefix: str):
        if self.add_date:
            dataframe.to_csv(f'{output_file_prefix}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}_.{self.output_file_extension}', index=False)
        else:
            files = os.listdir()
            # This is kind of an obscure way to just increment the suffix.
            n_existing_files = sum([1 if f"{output_file_prefix}_" in f else 0 for f in files])
            dataframe.to_csv(f'{output_file_prefix}_{n_existing_files}.{self.output_file_extension}', index=False)

        print("Saved measurements to file.")

    def test_linear_ramp_sw_trigger(self, n_samples: int, estimated_vref: float, output_file_prefix="measurements_lr_sw"):
        output_data = np.linspace(0, estimated_vref, n_samples)
        measurements = np.zeros(n_samples)
        times = np.zeros(n_samples)

        # If the output device has methods to measure
        # both the input device's reference voltage
        # and also its own output voltage,
        # then use them while performing measurements;

        # Create the required arrays regardless.
        # No modifying when the necessary methods do not exist.
        measured_references = np.full(n_samples, estimated_vref)
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
        self.start_profiling()

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
                Tester.print_loading_bar(i, n_samples, bar_count=bar_count)

            current_time = time.perf_counter_ns()
            times[i] = current_time
            # print(f"Δt = {(current_time - previous_time) / 1000}, us, f = {1 / ((current_time - previous_time) / 1000000000)}")
            previous_time = current_time

        # Stop profiling.
        self.stop_profiling(output_file_prefix)

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
        
        # Save the results.
        self.save_file(df, output_file_prefix)

    def test_linear_ramp_hw_trigger(self, n_samples: int, estimated_vref: float, sample_rate: float = 2800, output_file_prefix="measurements_lr_hw"):
        output_data = np.linspace(0, estimated_vref, n_samples)
        measurements = np.zeros(n_samples)
        times = np.zeros(n_samples)

        # Even though we cannot monitor the Vref and output, 
        # nevertheless create the required arrays
        # to comply with the data structure for analysis later on.
        measured_references = np.full(n_samples, estimated_vref)
        measured_outputs = np.copy(output_data)

        # Start profiling.
        self.start_profiling()

        start_time = (
            time.perf_counter()
        )  # For total measurement time with lesser accuracy

        # Start the writing task in background.
        self.output_device.write_samples_clocked(output_data, sample_rate)

        # Read all of the written samples.
        for i in range(len(output_data)):
            print(i)
            measurements[i] = self.input_device.read_sample_clocked()
            if self.verbose:
                print(
                    f"Wrote sample {output_data[i]}."
                )
                print(f"Read code {measurements[i]}.")

            current_time = time.perf_counter_ns()
            times[i] = current_time
            # print(f"Δt = {(current_time - previous_time) / 1000}, us, f = {1 / ((current_time - previous_time) / 1000000000)}")
            previous_time = current_time

        # Stop profiling.
        self.stop_profiling(output_file_prefix)

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
        
        # Save the results.
        self.save_file(df, output_file_prefix)

        # Dump profiling statistics.
        if self.profiling:
            self.cp.dump_stats("profiling_" + output_file_prefix)

    def test_linear_ramp(self, n_samples: int, estimated_vref: float, triggering="software", output_file_prefix="linear_ramp"):
        if triggering == "software":
            self.init()
            self.test_linear_ramp_sw_trigger(n_samples, estimated_vref, output_file_prefix = output_file_prefix + "_sw")
            self.deinit()
        else:
            self.init(trigger_mode="hardware")
            self.test_linear_ramp_hw_trigger(n_samples, estimated_vref, output_file_prefix=output_file_prefix + "_hw")
            self.deinit()

    def sine(n_samples: int, periods: float, min_value: float, max_value: float):
        x = np.linspace(0, 2 * np.pi * periods, n_samples, endpoint=False)
        # Create a sine wave and also map it to the specified range.
        return np.interp(np.sin(x), [-1, 1], [min_value, max_value])

    def test_signal_sine(self, n_samples: int, frequency: float, estimated_vref: float, sample_rate: float = 2800, output_file_prefix="sine"):
        self.init(trigger_mode="hardware")

        periods = n_samples / sample_rate * frequency
        output_data = Tester.sine(n_samples, periods, 0, estimated_vref)
        measurements = np.zeros(n_samples)
        times = np.zeros(n_samples)

        # Even though we cannot monitor the Vref and output, 
        # nevertheless create the required arrays
        # to comply with the data structure for analysis later on.
        measured_references = np.full(n_samples, estimated_vref)
        measured_outputs = np.copy(output_data)

        start_time = (
            time.perf_counter()
        )  # For total measurement time with lesser accuracy

        # Start profiling.
        self.start_profiling()

        # Start the writing task in background.
        self.output_device.write_samples_clocked(output_data, sample_rate)

        # Read all of the written samples.
        for i in range(len(output_data)):
            print(i)
            measurements[i] = self.input_device.read_sample_clocked()
            if self.verbose:
                print(
                    f"Wrote sample {output_data[i]}."
                )
                print(f"Read code {measurements[i]}.")

            current_time = time.perf_counter_ns()
            times[i] = current_time
            # print(f"Δt = {(current_time - previous_time) / 1000}, us, f = {1 / ((current_time - previous_time) / 1000000000)}")
            previous_time = current_time

        # Stop profiling.
        self.stop_profiling(output_file_prefix)

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
        
        # Save the results.
        self.save_file(df, output_file_prefix)
        self.deinit()

    def test_input_noise(self, n_samples: int, voltage: float, estimated_vref: float, sample_rate: float = 2800, output_file_prefix="input_noise"):
        self.init(trigger_mode="hardware")

        output_data = np.full(n_samples, voltage)
        measurements = np.zeros(n_samples)
        times = np.zeros(n_samples)

        # Even though we cannot monitor the Vref and output, 
        # nevertheless create the required arrays
        # to comply with the data structure for analysis later on.
        measured_references = np.full(n_samples, estimated_vref)
        measured_outputs = np.copy(output_data)

        start_time = (
            time.perf_counter()
        )  # For total measurement time with lesser accuracy

        # Start profiling.
        self.start_profiling()

        # Start the writing task in background.
        self.output_device.write_samples_clocked(output_data, sample_rate)

        # Read all of the written samples.
        for i in range(len(output_data)):
            print(i)
            measurements[i] = self.input_device.read_sample_clocked()
            if self.verbose:
                print(
                    f"Wrote sample {output_data[i]}."
                )
                print(f"Read code {measurements[i]}.")

            current_time = time.perf_counter_ns()
            times[i] = current_time
            # print(f"Δt = {(current_time - previous_time) / 1000}, us, f = {1 / ((current_time - previous_time) / 1000000000)}")
            previous_time = current_time

        # Stop profiling.
        self.stop_profiling(output_file_prefix)

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
        
        # Save the results.
        self.save_file(df, output_file_prefix)
        self.deinit()

    def test_vcc(self, n_samples: int) -> pd.DataFrame:
        measured_values = np.zeros(n_samples)
        codes = np.zeros(n_samples)
        if hasattr(self.output_device, "get_reference_voltage") and callable(
            self.output_device.get_reference_voltage
        ):
            for i in range(n_samples):
                measured_values[i] = self.output_device.get_reference_voltage()
                codes[i] = self.input_device.read_vcc()

        return pd.DataFrame({"code": codes, "vref": measured_values})


    def run_tests(self, n_samples: int = 8192, estimated_vref: float = 4.6, output_file_prefix="measurements", output_file_extension="csv", add_date=False):

        # First, run the linear ramp test (static testing).
        self.test_linear_ramp(n_samples, estimated_vref, triggering="hardware")

        # self.test_signal_sine(28000, 280, 4.6)

        # self.test_input_noise(10000, 2.5, 4.6)

if __name__ == "__main__":
    tester = Tester(SerialDevice(), NiUsb6211(output_channel="ao0", output_pulse_channel="ao1"), verbose=False)
    # tester.init()
    tester.run_tests()
    # tester.test_signal(n_samples=1000, estimated_vref=4.6, signal_function=np.sin, periods=1.0)
    tester.deinit()
