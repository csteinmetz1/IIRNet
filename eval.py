import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from data.hrtf import HRTFDataset
from iirnet.mlp import MLPModel
from iirnet.data import IIRFilterDataset
from iirnet.loss import LogMagTargetFrequencyLoss
import iirnet.signal as signal

pl.seed_everything(42)
eps = 1e-8

# prepare testing datasets
mag_loss = LogMagTargetFrequencyLoss()
rand_filters = IIRFilterDataset(max_order=100)
gaussain_filters = IIRFilterDataset(method="gaussian_peaks", max_order=100)
hrtfs = HRTFDataset('./data/HRTF')

# load models from disk
model = MLPModel.load_from_checkpoint('lightning_logs/version_17/checkpoints/epoch=10-step=8601.ckpt')
#model = MLPModel.load_from_checkpoint('lightning_logs/version_16/checkpoints/epoch=39-step=31279.ckpt')
model.eval()

if False:
    for idx, out in enumerate(gaussain_filters, 0):

        target_dB, phs, real, imag, sos = out

        target_dB = target_dB - torch.mean(target_dB)

        with torch.no_grad():
            pred_sos = model(target_dB)

        w, input_h = signal.sosfreqz(pred_sos, worN=target_dB.shape[-1])

        input_dB  = 20 * torch.log10(signal.mag(input_h) + eps)

        target_dB = target_dB.squeeze()
        input_dB = input_dB.squeeze()

        error = torch.nn.functional.mse_loss(input_dB, target_dB)

        print(idx, error)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
        ax.plot(w, target_dB.squeeze(), color='b', label="target")
        ax.plot(w, input_dB.squeeze(), color='r', label="pred")
        ax.set_xscale('log')
        ax.set_ylim([-60, 40])
        ax.set_ylabel('Amplitude [dB]')
        ax.set_xlabel('Frequency [Hz]')
        ax.legend()
        ax.grid()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'./data/plots/{idx}.png')
        plt.close('all')


for idx, target_h in enumerate(hrtfs, 0):

    target_h_mag = signal.mag(target_h)
    target_dB = 20 * torch.log10(target_h_mag + eps)
    target_dB = target_dB - torch.mean(target_dB)

    with torch.no_grad():
        pred_sos = model(target_dB)

    w, input_h = signal.sosfreqz(pred_sos, worN=target_h.shape[-1])

    input_dB  = 20 * torch.log10(signal.mag(input_h) + eps)

    target_dB = target_dB.squeeze()
    input_dB = input_dB.squeeze()

    error = torch.nn.functional.mse_loss(input_dB, target_dB)

    print(idx, error)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
    ax.plot(w, target_dB.squeeze(), color='b', label="target")
    ax.plot(w, input_dB.squeeze(), color='r', label="pred")
    ax.set_xscale('log')
    ax.set_ylim([-60, 40])
    ax.set_ylabel('Amplitude [dB]')
    ax.set_xlabel('Frequency [Hz]')
    ax.legend()
    ax.grid()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'./data/plots/{idx}.png')
    plt.close('all')



