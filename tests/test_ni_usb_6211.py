from tester.ni_usb_6211 import NiUsb6211
import numpy as np

OUTPUT_READ_CHANNEL = "ai0"
VCC_READ_CHANNEL = "ai1"
TOLERANCE = 0.001

def test_find_devices():
    devices = NiUsb6211.find_devices()
    assert type(devices) == list, "Not a list!"

    if len(devices) > 0:
        assert type(devices[0]) == str, "An element is not a string!"

def test_integration():
    niUsb6211 = NiUsb6211(output_read_channel=OUTPUT_READ_CHANNEL, vcc_read_channel=VCC_READ_CHANNEL)
    assert niUsb6211, "The object does not exist!"
    
    assert niUsb6211.init() == None, "Initialization not successful!"

    # Read samples.

    # Read only one channel.
    samples = niUsb6211.read_samples(1, channels=0)
    assert samples, "No sample returned!"
    assert samples.shape == (1,), "Wrong shape!"

    # Read the other channel.
    samples = niUsb6211.read_samples(1, channels=1)
    assert samples, "No sample returned!"
    assert samples.shape == (1,), "Wrong shape!"

    # Read both channels, 1 sample.
    samples = niUsb6211.read_samples(1, channels=[0, 1])
    assert np.all(samples), "No samples returned!"
    assert samples.shape == (2, 1), "Wrong shape!"

    # Read both channels, many samples.
    samples = niUsb6211.read_samples(10, channels=[0, 1])
    assert np.all(samples), "No samples returned!"
    assert samples.shape == (2, 10), "Wrong shape!"

    # Writing
    sample = 1.0
    niUsb6211.write_sample(sample)
    assert abs(niUsb6211.get_measured_output_voltage() - sample) <= TOLERANCE, f"Output reading ({niUsb6211.get_measured_output_voltage()}) does not match the written value ({sample})."
    vref = niUsb6211.get_reference_voltage()
    
    def is_number(x):
        try:
            float(x)
            return True
        except:
            return False
    
    assert is_number(vref), f"Get_reference_voltage() output ({vref}) not a number!"

    # niUsb6211.read_samples(10, channels=[0, 1])
    niUsb6211.deinit()