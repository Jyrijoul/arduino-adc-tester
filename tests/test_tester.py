import numpy as np
from tester.tester import Tester


def test_sine():
    samples = 10000
    periods = 1
    min_value = 0
    max_value = 5
    sine = Tester.sine(samples, periods, min_value, max_value)
    assert np.allclose(np.min(sine), min_value), "Minimum value not observed!"
    assert np.allclose(np.max(sine), max_value), "Maximum value not observed!"


# def test_integration():
#     serialDevice = SerialDevice()

#     assert serialDevice, "No SerialDevice object returned!"

#     serialDevice.init()
#     assert serialDevice.ser, "No serial port connection created!"

#     sample = serialDevice.read_sample()
#     assert sample, "No reading returned!"
#     assert type(sample) == int, "Reading is not an integer!"

#     serialDevice.deinit()
#     assert not serialDevice.ser.is_open, "Serial connection not closed!"
