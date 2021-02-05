import time
import torch
import torchaudio
import matplotlib.pyplot as plt

from iirnet.signal import mag
from iirnet.loss import LogMagFrequencyLoss, LogMagTargetFrequencyLoss
from iirnet.filter import generate_characteristic_poly_filter

n_iters = 10000
magloss = LogMagFrequencyLoss()
magtarget = LogMagTargetFrequencyLoss()

def design_filter():
    mag, phs, _, _, sos = generate_characteristic_poly_filter(512, 10)
    sos = torch.tensor(sos)

    sos = sos.unsqueeze(0) # add batch dim

    # create the biquad coefficients we will optimize
    pred_sos = torch.rand(1,5,6, requires_grad=True)
    optimizer = torch.optim.SGD([pred_sos], lr=1e-1)

    tic = time.perf_counter()
    for n in range(n_iters):
        loss = magloss(pred_sos, sos)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    toc = time.perf_counter()

    print(f"{n+1}/{n_iters} MSE: {loss:0.3f} dB in {toc-tic:0.2f} sec")


rir, sr = torchaudio.load("small_room.wav")
rir = rir[0,:32764]
print(rir.shape)

target_h = torch.fft.rfft(rir)
target_mag = 20 * torch.log10(mag(target_h) + 1e-8).float()
plt.plot(target_mag)
plt.savefig("target.png")

target_h = target_h.unsqueeze(0)

# create the biquad coefficients we will optimize
pred_sos = torch.rand(1,10,6, requires_grad=True)
optimizer = torch.optim.Adam([pred_sos], lr=1e-2)

for n in range(n_iters):
    loss = magtarget(pred_sos, target_h)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"{n+1}/{n_iters} MSE: {loss:0.3f} dB")
