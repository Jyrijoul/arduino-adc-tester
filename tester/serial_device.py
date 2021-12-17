import serial
from serial.serialutil import SerialException, SerialTimeoutException
import serial.tools.list_ports
from input_device import InputDeviceInterface


class SerialDevice(InputDeviceInterface):
    timeout = 0
    serial_port = ""
    baud_rate = 115200
    verbose = True

    def __init__(
        self, timeout=1, serial_port="", baud_rate: int = 2000000, verbose=True
    ) -> None:
        self.timeout = timeout
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.verbose = verbose
        # super().__init__()

    def init(self, mode="software", device_name="CH340") -> None:
        """Establishes the serial connection.

        Parameters
        ----------
        mode : str, optional
            Whether to use software or hardware triggering.
            Possible values are "software" (default) and "hardware".
        device_name : str, optional
            Provides the device's name to be searched for.
            "CH340" by default (Arduino Nano).

        Raises
        ------
        SerialException
            If the connection cannot be established.
        """

        self.mode = mode

        if self.serial_port == "":
            # Serial devices (Arduinos)
            serial_devices = serial.tools.list_ports.comports()

            if self.verbose:
                print("Available serial devices:")
            for device in serial_devices:
                if self.verbose:
                    print(
                        "\t",
                        device.device,
                        "(name matches)" if device_name in device.description else "",
                    )
                if device_name in device.description:
                    self.serial_port = device.device

        self.ser = serial.Serial(
            port=self.serial_port, baudrate=self.baud_rate, timeout=self.timeout
        )
        if self.verbose:
            print(f"Serial port {self.ser.port} opened.")

        try:
            # Try to receive a response from the serial port.
            if self.verbose:
                print("Trying to establish communication.", end="")
            conn_success = False

            if mode == "software":
                text_to_send = "s"
            elif mode == "hardware":
                text_to_send = "h"
            else:
                text_to_send = "s"  # Default

            # Start sending data.
            for i in range(10):
                self.ser.write(text_to_send.encode())
                self.ser.flushOutput()
                value = self.ser.readline()
                if self.verbose:
                    print(".", end="")
                if value != b"":
                    print(value.decode().strip())
                    conn_success = True
                    break

            if conn_success:
                if self.verbose:
                    print("\nCommunication successfully established.")
            else:
                if self.verbose:
                    print("Communication not established!")
                raise SerialException
        except Exception as e:
            self.ser.close()
            raise e

    def read_sample(self):
        """Requests one sample from the Arduino.
        
        Raises
        ------
        RuntimeError
            When called with hardware triggering.
        """

        return self.read()

    def read_vcc(self):
        """Requests the Arduino to read its own VCC.
        
        Raises
        ------
        RuntimeError
            When called with hardware triggering.
        """

        return self.read("v")

    def read(self, char_to_write: str = "a") -> int:
        """A general function to request a sample from the Arduino.
        
        Only to be used with software triggering.

        Raises
        ------
        RuntimeError
            When called with hardware triggering.
        """
        if self.mode == "software":
            try:
                self.ser.write(char_to_write.encode())
                self.ser.flushOutput()
                value = self.ser.readline().decode().strip()
                return int(value)
            except Exception as e:
                self.ser.close()
                raise e
        else:
            self.ser.close()
            raise RuntimeError(
                "The serial device needs to be initialized in software triggering mode."
            )

    def read_sample_clocked(self) -> int:
        """Waits for the Arduino to send a sample.
        
        Arduino reads and sends the sample when a falling edge 
        on the specified pin (D2 by default) is detected.

        When a specified timeout is received, -1 is returned.

        Raises
        ------
        RuntimeError
            When called with software triggering.
        """

        if self.mode == "hardware":
            try:
                value = self.ser.readline().decode().strip()
                if value == "":
                    # Return a non-valid result when timeout exceeded.
                    print("Serial timeout reached!")
                    return -1

                return int(value)
            except Exception as e:
                self.ser.close()
                raise e
        else:
            self.ser.close()
            raise RuntimeError(
                "The serial device needs to be initialized in hardware triggering mode."
            )

    # TODO: Figure out whether this function is needed at all.
    def read_samples_clocked(self, n_samples) -> int:
        """Reads the requested amount of samples from the Arduino.
        
        Arduino reads and sends a sample when a falling edge 
        on the specified pin (D2 by default) is detected.

        When a specified timeout is received, 
        -1 is used as that sample's value.

        Raises
        ------
        RuntimeError
            When called with software triggering.
        """
        
        if self.mode == "hardware":
            samples_read = 0
            samples = []
            try:
                while samples_read < n_samples:
                    value = self.ser.readline().decode().strip()
                    if value == "":
                        # Return a non-valid result when timeout exceeded.
                        print("Serial timeout reached!")
                        continue
                    else:
                        # samples.append(int(value))
                        print(value)
                        samples_read += 1

                return samples
            except Exception as e:
                self.ser.close()
                raise e
        else:
            self.ser.close()
            raise RuntimeError(
                "The serial device needs to be initialized in hardware triggering mode."
            )

    def deinit(self) -> None:
        """Closes the serial connection."""
        self.ser.close()
        if self.verbose:
            print("Serial connection closed.")


if __name__ == "__main__":
    serialDevice = SerialDevice(timeout=2)
    print(issubclass(SerialDevice, InputDeviceInterface))

    triggering = "hardware"

    if triggering == "software":
        serialDevice.init()
        for i in range(100):
            print(serialDevice.read_sample())
        serialDevice.deinit()
    else:
        serialDevice.init("hardware")
        for i in range(100):
            print(serialDevice.read_sample_clocked())
        serialDevice.deinit()
