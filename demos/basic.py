import torch
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from iirnet.designer import Designer

# first load IIRNet with pre-trained weights
designer = Designer()

n = 32  # Desired filter order (4, 8, 16, 32, 64)
m = [0, -3, 0, 12, 0, -6, 0]  # Magnitude response specification
mode = "linear"  # interpolation mode for specification
output = "sos"  # Output type ("sos", or "ba")

# now call the designer with parameters
sos = designer(n, m, mode=mode, output=output)

# measure and plot the response
w, h = scipy.signal.sosfreqz(sos.numpy(), fs=2)

# interpolate the target for plotting
m_int = torch.tensor(m).view(1, 1, -1).float()
m_int = torch.nn.functional.interpolate(m_int, 512, mode=mode)

fig, ax = plt.subplots(figsize=(6, 3))
plt.plot(w, 20 * np.log10(np.abs(h)), label="Estimation")
plt.plot(w, m_int.view(-1), label="Specification")
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.ylabel("dB")
plt.xlabel("Normalized frequency")
plt.grid(c="lightgray")
plt.legend()
plt.tight_layout()
plt.savefig("demo.svg")
