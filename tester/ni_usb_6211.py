import time
import numpy as np
import nidaqmx
from nidaqmx import constants
from output_device import OutputDeviceInterface
from nidaqmx.constants import TerminalConfiguration
from nidaqmx.stream_writers import AnalogSingleChannelWriter
from nidaqmx.stream_readers import AnalogMultiChannelReader


class NiUsb6211(OutputDeviceInterface):
    timeout = 0.0
    output_reading = 0.0

    def __init__(
        self,
        device_name="Dev1",
        output_channel="ao0",
        output_read_channel="ai0",
        vcc_read_channel="ai1",
        output_pulse_channel="ao1",
        reference_voltage_estimate=5.0,
        min_value=0.0,
        max_value=5.0,
        timeout=1.0,
        verbose=True,
    ) -> None:
        self.device_name = device_name
        self.output_channel = output_channel
        self.output_read_channel = output_read_channel
        self.output_pulse_channel = output_pulse_channel
        self.vcc_read_channel = vcc_read_channel
        self.vcc_reading = reference_voltage_estimate
        self.min_value = min_value
        self.max_value = max_value
        self.timeout = timeout
        self.verbose = verbose

    def find_devices() -> list[str]:
        """Finds the NI DAQ devices present in the system."""
        system = nidaqmx.system.System.local()
        return [device.name for device in system.devices]

    def error_handler(self, err):
        """A custom error handler in order to always properly close the opened tasks.
        
        Raises
        ------
        Exception
            Raises the same exception it receives from any other method.

        """
        self.write_task.close()
        self.read_task.close()
        raise err

    def init(self, mode="software") -> None:
        """Initializes all the required tasks.

        Parameters
        ----------
        mode : str, optional
            Whether to use software or hardware triggering.
            Possible values are "software" and "hardware" (default).
        """

        terminal_config = TerminalConfiguration.RSE

        try:
            self.write_task = nidaqmx.Task("write_task")
            self.read_task = nidaqmx.Task("read_task")

            if self.verbose:
                print("Created writing and reading tasks.")

            self.write_task.ao_channels.add_ao_voltage_chan(
                f"{self.device_name}/{self.output_channel}",
                min_val=self.min_value,
                max_val=self.max_value,
            )

            if mode == "hardware":
                self.write_task.ao_channels.add_ao_voltage_chan(
                    f"{self.device_name}/{self.output_pulse_channel}",
                    min_val=self.min_value,
                    max_val=self.max_value,
                )

            self.read_task.ai_channels.add_ai_voltage_chan(
                f"{self.device_name}/{self.output_read_channel}",
                min_val=self.min_value,
                max_val=self.max_value,
                terminal_config=terminal_config,
            )

            self.read_task.ai_channels.add_ai_voltage_chan(
                f"{self.device_name}/{self.vcc_read_channel}",
                min_val=self.min_value,
                max_val=self.max_value,
                terminal_config=terminal_config,
            )

            if self.verbose:
                print("Added required analog output and input channels.")

            if mode != "hardware":
                # task.timing.cfg_samp_clk_timing(1000, samps_per_chan=10000)
                self.writer = AnalogSingleChannelWriter(self.write_task.out_stream)
                self.reader = AnalogMultiChannelReader(self.read_task.in_stream)

            if self.verbose:
                print("Created writing and reading streams.")
        except Exception as e:
            self.error_handler(e)

    def read_samples(self, n: int, channels: list[int] = [0, 1]) -> np.ndarray:
        """Read N samples from the Arduino.

        Parameters
        ----------
        n : int
            The amount of samples to be read.
        channels: int or list[int], optional, default: [0, 1]
            For specifying which channels to read.
            0 corresponds to the current output, 1 to the reference voltage.
            Both a single integer or a list of one or two can be used.
            The default parameter reads both.

        Returns
        -------
        np.ndarray of floats
            With the shape of either 
            (2, n) when both channels are specified or
            (n,) when only one channel is specified.
        """

        try:
            # self.read_task.start()
            data = np.zeros((2, n))
            # self.output_reader.read_one_sample(data)
            self.reader.read_many_sample(data, n)
            return data[channels]
        except Exception as e:
            self.error_handler(e)

    def write_sample(self, value: float, read=True):
        """Outputs a desired voltage.

        The device must be initialized in the "software" triggering mode!
        Optionally also reads both the reference voltage
        as well as the current output voltage.
        
        Parameters
        ----------
        value : float
            The output voltage.
        read : bool, optional, default: True
            If True, also reads both the reference voltage
            as well as the current output voltage. Both of their values
            can be accessed by separate methods named "get_reference_voltage"
            and "get_measured_output_voltage", respectively.
            Also, pass "update = False" for both of them
            to directly get the values read and saved in this method.
        """

        # Enter the critical section, where an exception
        # indicates the need to deinitialize the tasks.
        try:
            # Try whether the value is a float or not.
            value = float(value)

            # Write the voltage.
            self.writer.write_one_sample(value)

            # Read the written voltage and VCC.
            if read:
                samples = self.read_samples(1)

        except Exception as e:
            self.error_handler(e)

        # Also read the VCC for reference, and the output voltage for validation.
        # These can be subsequently used by other methods.
        if read:
            self.output_reading = samples[0, 0]
            self.vcc_reading = samples[1, 0]
        # print(f"Output = {self.output_reading}, VCC = {self.vcc_reading}.")

    def write_samples_clocked(self, samples: np.ndarray, sample_rate: float) -> None:
        """Outputs desired voltages.

        The device must be initialized in the "hardware" triggering mode!
        Optionally also reads both the reference voltage
        as well as the current output voltage.
        
        Parameters
        ----------
        samples : np.ndarray of floats (or integers)
            The output voltages.
        sample_rate : float
            Specifies the sample rate for outputting samples and generating the hardware triggering signal (clock).
        """

        # Enter the critical section, where an exception
        # indicates the need to deinitialize the tasks.
        try:
            n_samples = len(samples)
            sample_rate *= 2

            # 2x the samples because for each real sample,
            # we want to output both 0 and 1 on the other channel to create a pulse train.
            output_samples = np.zeros((2, n_samples * 2))
            output_samples[0] = np.repeat(samples, 2)
            # 1 --> 0 == falling edge which the Arduino captures.
            output_samples[1] = np.tile([5, 0], n_samples)
            print(output_samples.shape)

            print(output_samples)

            # self.write_task.timing.cfg_samp_clk_timing(sample_rate, sample_mode=constants.AcquisitionType.CONTINUOUS)
            self.write_task.timing.cfg_samp_clk_timing(
                sample_rate,
                sample_mode=constants.AcquisitionType.FINITE,
                samps_per_chan=(n_samples * 2),
            )
            samples_written = self.write_task.write(output_samples, auto_start=True)

            print(f"{samples_written}/{n_samples} samples written.")
        except Exception as e:
            self.error_handler(e)

    def get_reference_voltage(self, update=True):
        """Reads and returns the Arduino's reference voltage.
        
        Parameters
        ----------
        update : bool, optional, default: True
            Whether to get a new reading or just to return the previously read value.
            This option works well in tandem with the 
            "read" parameter of the method "write_sample".
        """

        if update:
            # Read the written voltage and VCC.
            self.vcc_reading = self.read_samples(1, 1)

        return self.vcc_reading

    def get_measured_output_voltage(self, update=True):
        """Reads and returns the current output voltage.
        
        Parameters
        ----------
        update : bool, optional, default: True
            Whether to get a new reading or just to return the previously read value.
            This option works well in tandem with the 
            "read" parameter of the method "write_sample".
        """

        if update:
            # Read the written voltage and VCC.
            self.output_reading = self.read_samples(1, 0)

        return self.output_reading

    def deinit(self):
        """Properly closes all the opened tasks."""
        if self.verbose:
            print("Closing the tasks.")
        self.write_task.close()
        self.read_task.close()


if __name__ == "__main__":
    test = "hardware_trigger"

    if test == "software_trigger":
        niUsb6211 = NiUsb6211()
        print(NiUsb6211.find_devices())
        niUsb6211.init()

        niUsb6211.write_sample(3.3)
        samples = niUsb6211.read_samples(10, channels=[0, 1])
        print(samples, samples.shape)

        samples = niUsb6211.read_samples(1, channels=0)
        print(samples, samples.shape)
        print(
            niUsb6211.get_reference_voltage(), niUsb6211.get_measured_output_voltage()
        )
        niUsb6211.write_sample(0)
        niUsb6211.deinit()
    else:
        niUsb6211 = NiUsb6211(output_pulse_channel="ao1")
        print(NiUsb6211.find_devices())
        niUsb6211.init()

        n_samples = 8192
        output_data = np.linspace(0, 4.6, n_samples)

        sample_rate = 8192
        niUsb6211.write_samples_clocked(output_data, sample_rate)

        time.sleep(n_samples / sample_rate + 0.1)
        niUsb6211.deinit()
