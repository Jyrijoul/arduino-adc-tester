import serial
from serial.serialutil import SerialException
import serial.tools.list_ports
from input_device import InputDeviceInterface


class SerialDevice(InputDeviceInterface):
    timeout = 0
    serial_port = ""
    baud_rate = 115200
    verbose = True

    def __init__(self, timeout=1, serial_port="", baud_rate: int = 2000000, verbose=True) -> None:
        self.timeout = timeout
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.verbose = verbose
        # super().__init__()

    def init(self, device_name="CH340") -> None:
        if self.serial_port == "":
            # Serial devices (Arduinos)
            serial_devices = serial.tools.list_ports.comports()

            if self.verbose:
                print("Available serial devices:")
            for device in serial_devices:
                if self.verbose:print("\t", device.device,
                      "(name matches)" if device_name in device.description else "")
                if device_name in device.description:
                    self.serial_port = device.device

        self.ser = serial.Serial(
            port=self.serial_port, baudrate=self.baud_rate, timeout=self.timeout)
        if self.verbose:
            print(f"Serial port {self.ser.port} opened.")

        try:
            # Try to receive a response from the serial port.
            if self.verbose:
                print("Trying to establish communication.", end="")
            conn_success = False
            for i in range(10):
                self.ser.write("0".encode())
                self.ser.flushOutput()
                value = self.ser.readline()
                if self.verbose:
                    print(".", end="")
                if value != b"":
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

    def read_sample(self, char_to_write: str = "a") -> int:
        try:
            self.ser.write(char_to_write.encode())
            self.ser.flushOutput()
            value = self.ser.readline().decode().strip()
            return int(value)
        except Exception as e:
            self.ser.close()
            raise e

    def deinit(self) -> None:
        self.ser.close()
        if self.verbose:
            print("Serial connection closed.")


if __name__ == "__main__":
    serialDevice = SerialDevice()
    print(issubclass(SerialDevice, InputDeviceInterface))

    serialDevice.init()
    for i in range(100):
        print(serialDevice.read_sample())
    serialDevice.deinit()
