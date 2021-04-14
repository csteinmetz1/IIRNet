import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from iirnet.data import IIRFilterDataset

method = 'uniform_disk'
dataset = IIRFilterDataset(method=method, max_order=24)

mags = []
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

for idx, batch in enumerate(dataset):
    mag, phs, real, imag, sos = batch

    print(idx)
    w, h = scipy.signal.sosfreqz(sos, worN=1024, fs=44100)
    z,p,k = scipy.signal.sos2zpk(sos)

    #for pole in p:
    #    ax2.scatter(np.real(pole), np.imag(pole), c='tab:blue', s=1)

    mag_dB = 20 * np.log10(np.abs(h) + 1e-8)
    mags.append(mag_dB)
    ax1.plot(w, mag_dB)
    if idx > 1000: break

u = np.mean(mags, axis=0)
s = np.std(mags, axis=0)
ax1.plot(w, u, c='k')
ax1.plot(w, u + s, c='k', linestyle='--')
ax1.plot(w, u - s, c='k', linestyle='--')
#ax1.set_xscale('log')
ax1.set_ylim([-150, 150])
ax1.grid()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

# unit circle
x = np.linspace(-1.0, 1.0, 100)
y = np.linspace(-1.0, 1.0, 100)
X, Y = np.meshgrid(x,y)
F = X**2 + Y**2
ax2.contour(X,Y,F,[0], colors='k')
ax2.set_ylim([-1, 1])
ax2.set_xlim([-1, 1])

ax2.set_aspect('equal') 

fig1.savefig(f'{method}_mean+var.png')
fig2.savefig(f'{method}_pole+zero.png')
