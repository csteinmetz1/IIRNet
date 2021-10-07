import numpy as np
import scipy.signal
import torch
import matplotlib.pyplot as plt
from iirnet.designer import Designer

# first load IIRNet with pre-trained weights
designer = Designer()

n = 32  # Desired filter order (4, 8, 16, 32, 64)
m = [0, -3, 0, 6, 0, -3, 0]  # Magnitude response specification
output = "sos"  # Output type ("sos", or "ba")

# now call the designer with parameters
sos = designer(n, m, output=output)

# measure and plot the response
w, h = scipy.signal.sosfreqz(sos.numpy(), fs=2)

# interpolate the target for plotting
m_int = torch.tensor(m).view(1, 1, -1).float()
m_int = torch.nn.functional.interpolate(m_int, 512)

plt.plot(w, 20 * np.log10(np.abs(h)))
plt.plot(w, m_int.view(-1))
plt.grid(c="lightgray")
plt.savefig("demo.png")
