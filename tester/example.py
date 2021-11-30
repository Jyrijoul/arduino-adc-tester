import nidaqmx
from nidaqmx.constants import TerminalConfiguration

with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan("Dev1/ai0", min_val=-10, max_val=10, terminal_config=TerminalConfiguration.RSE)
    print(task.read())

phys_chan = nidaqmx.system.PhysicalChannel('Dev1/ai0')
print(phys_chan.ai_term_cfgs)
