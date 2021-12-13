from output_device import OutputDeviceInterface
import numpy as np
import nidaqmx
from nidaqmx.constants import TerminalConfiguration
from nidaqmx.stream_writers import AnalogSingleChannelWriter
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.stream_readers import AnalogSingleChannelReader
import time


class NiUsb6211(OutputDeviceInterface):
    timeout = 0.0
    output_reading = 0.0

    def __init__(
        self,
        device_name="Dev1",
        output_channel="ao0",
        output_read_channel="ai0",
        vcc_read_channel="ai1",
        reference_voltage_estimate=5.0,
        timeout=1.0,
        verbose=True,
    ) -> None:
        self.device_name = device_name
        self.output_channel = output_channel
        self.output_read_channel = output_read_channel
        self.vcc_read_channel = vcc_read_channel
        self.vcc_reading = reference_voltage_estimate
        self.timeout = timeout
        self.verbose = verbose

    def find_devices() -> list[str]:
        system = nidaqmx.system.System.local()
        return [device.name for device in system.devices]

    def error_handler(self, err):
        self.write_task.close()
        self.read_task.close()
        raise err

    def init(self):
        terminal_config = TerminalConfiguration.RSE
        min_val = -10
        max_val = 10

        try:
            self.write_task = nidaqmx.Task("write_task")
            self.read_task = nidaqmx.Task("read_task")

            if self.verbose:
                print("Created writing and reading tasks.")

            self.write_task.ao_channels.add_ao_voltage_chan(
                f"{self.device_name}/{self.output_channel}"
            )

            self.read_task.ai_channels.add_ai_voltage_chan(
                f"{self.device_name}/{self.output_read_channel}",
                min_val=min_val,
                max_val=max_val,
                terminal_config=terminal_config,
            )

            self.read_task.ai_channels.add_ai_voltage_chan(
                f"{self.device_name}/{self.vcc_read_channel}",
                min_val=min_val,
                max_val=max_val,
                terminal_config=terminal_config,
            )

            if self.verbose:
                print("Added required analog output and input channels.")

            # task.timing.cfg_samp_clk_timing(1000, samps_per_chan=10000)
            self.writer = AnalogSingleChannelWriter(self.write_task.out_stream)
            self.reader = AnalogMultiChannelReader(self.read_task.in_stream)

            if self.verbose:
                print("Created writing and reading streams.")
        except Exception as e:
            self.error_handler(e)

    def read_samples(self, n, channels: list[int] = [0, 1]) -> np.ndarray:
        try:
            # self.read_task.start()
            data = np.zeros((2, n))
            # self.output_reader.read_one_sample(data)
            self.reader.read_many_sample(data, n)
            return data[channels]
        except Exception as e:
            self.error_handler(e)

    def write_sample(self, value: float):
        # Enter the critical section, where an exception 
        # indicates the need to deinitialize the tasks.
        try:
            # Try whether the value is a float or not.
            value = float(value)

            # Write the voltage.
            self.writer.write_one_sample(value)

            # Read the written voltage and VCC.
            samples = self.read_samples(1)
            
        except Exception as e:
            self.error_handler(e)

        # Also read the VCC for reference, and the output voltage for validation.
        # These can be subsequently used by other methods.
        self.output_reading = samples[0, 0]
        self.vcc_reading = samples[1, 0]
        # print(f"Output = {self.output_reading}, VCC = {self.vcc_reading}.")
    
    def get_reference_voltage(self):
        return self.vcc_reading

    def get_measured_output_voltage(self):
        return self.output_reading

    def deinit(self):
        if self.verbose:
            print("Closing the tasks.")
        self.write_task.close()
        self.read_task.close()


if __name__ == "__main__":
    niUsb6211 = NiUsb6211()
    print(NiUsb6211.find_devices())
    niUsb6211.init()

    niUsb6211.write_sample(3.3)
    samples = niUsb6211.read_samples(10, channels=[0, 1])
    print(samples, samples.shape)

    samples = niUsb6211.read_samples(1, channels=0)
    print(samples, samples.shape)
    print(niUsb6211.get_reference_voltage(), niUsb6211.get_measured_output_voltage())
    niUsb6211.write_sample(0)
    niUsb6211.deinit()
