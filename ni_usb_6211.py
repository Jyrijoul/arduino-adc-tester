from output_device import OutputDeviceInterface
import numpy as np
import nidaqmx
from nidaqmx.constants import TerminalConfiguration
from nidaqmx.stream_writers import AnalogSingleChannelWriter
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.stream_readers import AnalogSingleChannelReader


class NiUsb6211(OutputDeviceInterface):
    timeout = 0

    def __init__(
        self,
        device_name="Dev1",
        output_channel="ao1",
        output_read_channel="ai0",
        vcc_read_channel="ai1",
        verbose=True,
    ) -> None:
        self.device_name = device_name
        self.output_channel = output_channel
        self.output_read_channel = output_read_channel
        self.vcc_read_channel = vcc_read_channel
        self.verbose = verbose

    def find_devices():
        # DAQ devices
        system = nidaqmx.system.System.local()
        # print(system.driver_version)

        print("Available DAQ devices:")
        for device in system.devices:
            print("\t", device)

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
            self.output_reader = AnalogMultiChannelReader(self.read_task.in_stream)

            if self.verbose:
                print("Created writing and reading streams.")
        except Exception as e:
            self.error_handler(e)

    def read_samples(self, n, channels=[0, 1]):
        try:
            self.read_task.start()
            data = np.zeros((2, n))
            # self.output_reader.read_one_sample(data)
            self.output_reader.read_many_sample(data, n)
            return data[channels]
        except Exception as e:
            self.error_handler(e)

    def write_sample(self, value):
        try:
            self.writer.write_one_sample(value)
        except Exception as e:
            self.error_handler(e)

    def deinit(self):
        if self.verbose:
                print("Closing the tasks.")
        self.write_task.close()
        self.read_task.close()


niUsb6211 = NiUsb6211()
NiUsb6211.find_devices()
niUsb6211.init()
print(niUsb6211.read_samples(10, channels=[0, 1]))
niUsb6211.deinit()
