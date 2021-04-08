import sys
import torch
import numpy as np
import scipy.signal
import iirnet.signal as signal
import matplotlib.pyplot as plt

bs = 4

p = torch.tensor([1.0, 2.5, -4.2])
p = p.view(1,-1).repeat(bs,1)
x = torch.tensor([5])

val = signal.polyval(p, x)
print(val.shape)

print(np.polyval([1.0, 2.5, -4.2], 5))


p = torch.rand(bs,3)
x = torch.rand(100)

signal.polyval(p, x)

input = torch.rand(bs,2,6)

w, h_batch = signal.sosfreqz(input)

#sys.exit(0)

for n in range(bs):

    input_sos = input[n:n+1,...]
    w, h_loop = signal.sosfreqz(input_sos)

    print(input_sos)
    # remove zero padded sos
    input_sos = input_sos[input_sos.sum(-1) != 0,:]
    print(input_sos)

    # normalize by a0
    a0 = input_sos[:,3].unsqueeze(-1)
    input_sos = input_sos/a0

    w, h_scipy = scipy.signal.sosfreqz(input_sos.squeeze())

    mag_pred = 20 * np.log10(np.abs(h_loop.squeeze()) + 1e-8)
    mag_target = 20 * np.log10(np.abs(h_batch[n,:]) + 1e-8)
    mag_scipy = 20 * np.log10(np.abs(h_scipy) + 1e-8)
    plt.plot(w, mag_target, color='b', label="target")
    plt.plot(w, mag_pred, color='r', label="pred")
    plt.plot(w, mag_scipy, color='g', label="pred")

    plt.savefig(f"{n}.png")
    plt.close('all')
