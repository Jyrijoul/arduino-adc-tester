from tester.serial_device import SerialDevice

def test_integration():
    serialDevice = SerialDevice()

    assert serialDevice, "No SerialDevice object returned!"

    serialDevice.init()
    assert serialDevice.ser, "No serial port connection created!"

    sample = serialDevice.read_sample()
    assert sample, "No reading returned!"
    assert type(sample) == int, "Reading is not an integer!"

    serialDevice.deinit()
    assert not serialDevice.ser.is_open, "Serial connection not closed!"
