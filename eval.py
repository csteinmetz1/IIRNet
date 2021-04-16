import sys
import torch
import scipy.signal
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from data.hrtf import HRTFDataset
from iirnet.mlp import MLPModel
from iirnet.data import IIRFilterDataset
from iirnet.loss import LogMagTargetFrequencyLoss
import iirnet.signal as signal

pl.seed_everything(42)
eps = 1e-8
gpu = False

# prepare testing datasets
mag_loss = LogMagTargetFrequencyLoss()
rand_filters = IIRFilterDataset(max_order=100)
gaussain_filters = IIRFilterDataset(method="gaussian_peaks", max_order=100)
hrtfs = HRTFDataset('./data/HRTF')

# load models from disk
#model = MLPModel.load_from_checkpoint('lightning_logs/version_21/checkpoints/epoch=61-step=48483.ckpt')
#model = MLPModel.load_from_checkpoint('lightning_logs/version_16/checkpoints/epoch=39-step=31279.ckpt')
model = MLPModel.load_from_checkpoint('lightning_logs/version_78/checkpoints/epoch=39-step=31279.ckpt')
n_sections = 24
step = 4

model.eval()

if gpu:
    model.to("cuda")

errors = []

for idx, target_h in enumerate(hrtfs, 0):

    target_h_mag = signal.mag(target_h)
    target_h_ang = np.squeeze(np.unwrap(np.angle(target_h.numpy())))
    target_dB = 20 * torch.log10(target_h_mag + eps)
    target_dB = target_dB - torch.mean(target_dB)

    with torch.no_grad():
        if gpu: 
            target_dB = target_dB.to("cuda")
        pred_sos = model(target_dB)

    # here we can loop over each sub filter and measure response
    target_dB = target_dB.squeeze()

    subfilters = []
    for n in np.arange(n_sections, step=step):
        sos = pred_sos[:,0:n+2,:]
        w, input_h = signal.sosfreqz(sos, worN=target_h.shape[-1])
        input_dB  = 20 * torch.log10(signal.mag(input_h) + eps)
        input_dB = input_dB.squeeze()
        subfilters.append(input_dB.cpu().squeeze().numpy())

    error = torch.nn.functional.mse_loss(input_dB, target_dB)

    errors.append(error.item())
    print(f"{idx}/{len(hrtfs)}: MSE: {np.mean(errors):0.2f} dB")

    if True:
        mag_idx = 0
        phs_idx = 1
        plot_idx = 2

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))

        zeros,poles,k = scipy.signal.sos2zpk(pred_sos.squeeze())
        w_pred, h_pred = signal.sosfreqz(pred_sos, worN=target_h.shape[-1], fs=44100)
        mag_pred = 20 * np.log10(np.abs(h_pred.squeeze()) + 1e-8)

        axs[mag_idx].plot(w_pred, target_dB, color='tab:blue', label="target")
        axs[mag_idx].plot(w_pred, mag_pred, color='tab:red', label="pred")
        axs[mag_idx].set_xscale('log')
        axs[mag_idx].set_ylim([-60, 40])
        axs[mag_idx].grid()
        axs[mag_idx].spines['top'].set_visible(False)
        axs[mag_idx].spines['right'].set_visible(False)
        axs[mag_idx].spines['bottom'].set_visible(False)
        axs[mag_idx].spines['left'].set_visible(False)
        axs[mag_idx].set_ylabel('Amplitude (dB)')
        axs[mag_idx].set_xlabel('Frequency (Hz)')

        axs[phs_idx].plot(w_pred, np.squeeze(np.unwrap(np.angle(h_pred))), color='tab:red', label="pred")
        axs[phs_idx].plot(w_pred, target_h_ang, color='tab:blue', label="target")
        axs[phs_idx].set_xscale('log')
        #axs[phs_idx].set_ylim([-60, 40])
        axs[phs_idx].grid()
        axs[phs_idx].spines['top'].set_visible(False)
        axs[phs_idx].spines['right'].set_visible(False)
        axs[phs_idx].spines['bottom'].set_visible(False)
        axs[phs_idx].spines['left'].set_visible(False)
        axs[phs_idx].set_ylabel('Angle (radians)')

        # pole-zero plot
        for pole in poles:
            axs[plot_idx].scatter(
                            np.real(pole), 
                            np.imag(pole), 
                            c='tab:red', 
                            s=10, 
                            marker='x', 
                            facecolors='none')
        for zero in zeros:
            axs[plot_idx].scatter(
                            np.real(zero), 
                            np.imag(zero), 
                            s=10, 
                            marker='o', 
                            facecolors='none', 
                            edgecolors='tab:red')

        # unit circle
        unit_circle = circle1 = plt.Circle((0, 0), 1, color='k', fill=False)
        axs[plot_idx].add_patch(unit_circle)
        axs[plot_idx].set_ylim([-1.5, 1.5])
        axs[plot_idx].set_xlim([-1.5, 1.5])
        axs[plot_idx].grid()
        axs[plot_idx].spines['top'].set_visible(False)
        axs[plot_idx].spines['right'].set_visible(False)
        axs[plot_idx].spines['bottom'].set_visible(False)
        axs[plot_idx].spines['left'].set_visible(False)
        axs[plot_idx].set_aspect('equal') 
        axs[plot_idx].set_axisbelow(True)
        axs[plot_idx].set_ylabel("Im")
        axs[plot_idx].set_xlabel('Re')

        plt.tight_layout()
        plt.savefig(f'./data/plots/{idx}.png')
        plt.close('all')
