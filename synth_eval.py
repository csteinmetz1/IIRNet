import sys
import torch
import scipy.signal
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from argparse import ArgumentParser


from data.hrtf import HRTFDataset
from data.gc import GuitarCabDataset
from iirnet.mlp import MLPModel
from iirnet.data import IIRFilterDataset
from iirnet.loss import LogMagTargetFrequencyLoss
import iirnet.signal as signal

import os
import glob 

pl.seed_everything(42)
eps = 1e-8
gpu = False

parser = ArgumentParser()

parser.add_argument('--shuffle', action="store_true")
parser.add_argument('--precompute', action="store_true")
parser.add_argument('--filter_method', type=str, default='char_poly')
parser.add_argument('--max_train_order', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--model_name', type=str, default='mlp', help='mlp or lstm')
parser.add_argument('--num_train_examples', type=int, default=100000)
parser.add_argument('--num_val_examples', type=int, default=10000)

temp_args, _ = parser.parse_known_args()

# let the model add what it wants
if temp_args.model_name == 'mlp':
    parser = MLPModel.add_model_specific_args(parser)
elif temp_args.model_name == 'lstm':
    parser = LSTMModel.add_model_specific_args(parser)

parser = pl.Trainer.add_argparse_args(parser)       # add all the available trainer options to argparse
args = parser.parse_args()   
# prepare testing datasets
mag_loss = LogMagTargetFrequencyLoss()

args.num_val_examples = 2 ##bash script not recognizing this variable?
val_datasetA = IIRFilterDataset(method="gaussian_peaks",
                               num_points=args.num_points, 
                               max_order=args.max_train_order, 
                               num_examples=args.num_val_examples,
                               precompute=args.precompute)

val_datasetB = IIRFilterDataset(method="pass",
                               num_points=args.num_points, 
                               max_order=args.max_train_order, 
                               num_examples=args.num_val_examples,
                               precompute=args.precompute)

val_datasetC = IIRFilterDataset(method="parametric",
                               num_points=args.num_points, 
                               max_order=args.max_train_order, 
                               num_examples=args.num_val_examples,
                               precompute=args.precompute)

# this one does not work at the moment
val_datasetD = IIRFilterDataset(method="uniform_parametric",
                               num_points=args.num_points, 
                               max_order=args.max_train_order, 
                               num_examples=args.num_val_examples,
                               precompute=args.precompute)

val_datasetE = IIRFilterDataset(method="char_poly",
                               num_points=args.num_points, 
                               max_order=args.max_train_order, 
                               min_order=args.max_train_order//2,
                               num_examples=args.num_val_examples,
                               precompute=args.precompute)

val_datasetF = IIRFilterDataset(method="uniform_disk",
                               num_points=args.num_points, 
                               max_order=args.max_train_order, 
                               num_examples=args.num_val_examples,
                               precompute=args.precompute)

val_datasetG = IIRFilterDataset(method="uniform_biquad",
                               num_points=args.num_points, 
                               max_order=args.max_train_order, 
                               num_examples=args.num_val_examples,
                               precompute=args.precompute)

val_datasetH = IIRFilterDataset(method="normal_poly",
                               num_points=args.num_points, 
                               max_order=args.max_train_order, 
                               num_examples=args.num_val_examples,
                               precompute=args.precompute)

val_datasetI = IIRFilterDataset(method="actual_uniform_disk",
                               num_points=args.num_points, 
                               max_order=args.max_train_order, 
                               num_examples=args.num_val_examples,
                               precompute=args.precompute)

val_dataset = torch.utils.data.ConcatDataset([
  val_datasetA, val_datasetB, val_datasetC,
  val_datasetD, val_datasetE, val_datasetF,
  val_datasetG, val_datasetH, val_datasetI
  ])

# load models from disk
version = str(2)
path = glob.glob(os.path.join('lightning_logs/version_'+version+'/checkpoints/', "*.ckpt"))
print(path)
model = MLPModel.load_from_checkpoint(path[0])
n_sections = 24
step = 4

model.eval()

if gpu:
    model.to("cuda")

errors = []

for idx, rand_filter in enumerate(val_dataset, 0):
    mag, phs, real, imag, sos = rand_filter
    target_h_mag = mag 
    target_h_ang = phs 
    target_dB = mag 

    with torch.no_grad():
        if gpu: 
            target_dB = target_dB.to("cuda")
        pred_sos = model(target_dB)

    # here we can loop over each sub filter and measure response
    target_dB = target_dB.squeeze()

    subfilters = []
    for n in np.arange(n_sections, step=step):
        sos = pred_sos[:,0:n+2,:]
        w, input_h = signal.sosfreqz(sos, worN=target_h_mag.shape[-1])
        input_dB  = 20 * torch.log10(signal.mag(input_h) + eps)
        input_dB = input_dB.squeeze()
        subfilters.append(input_dB.cpu().squeeze().numpy())

    error = torch.nn.functional.mse_loss(input_dB, target_dB)

    errors.append(error.item())
    print(f"{idx}/{len(val_dataset)}: MSE: {np.mean(errors):0.2f} dB")

    if True:
        mag_idx = 0
        phs_idx = 1
        plot_idx = 1

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 3))

        zeros,poles,k = scipy.signal.sos2zpk(pred_sos.squeeze())
        w_pred, h_pred = signal.sosfreqz(pred_sos, worN=target_h_mag.shape[-1], fs=44100)
        mag_pred = 20 * np.log10(np.abs(h_pred.squeeze()) + 1e-8)

        axs[mag_idx].plot(w_pred, target_dB, color='tab:blue', label="target")
        axs[mag_idx].plot(w_pred, mag_pred, color='tab:red', label="pred")
        # axs[mag_idx].plot(w_pred, mag_pred - target_dB, color='tab:green', label="error")

        # axs[mag_idx].set_xscale('log')
        axs[mag_idx].set_ylim([-60, 40])
        axs[mag_idx].grid()
        axs[mag_idx].spines['top'].set_visible(False)
        axs[mag_idx].spines['right'].set_visible(False)
        axs[mag_idx].spines['bottom'].set_visible(False)
        axs[mag_idx].spines['left'].set_visible(False)
        axs[mag_idx].set_ylabel('Amplitude (dB)')
        axs[mag_idx].set_xlabel('Frequency (Hz)')
        axs[mag_idx].legend()

        # axs[phs_idx].plot(w_pred, np.squeeze(np.unwrap(np.angle(h_pred))), color='tab:red', label="pred")
        # axs[phs_idx].plot(w_pred, np.unwrap(target_h_ang), color='tab:blue', label="target")
        # # axs[phs_idx].plot(w_pred, target_h_ang, color='tab:blue', label="target")
        # axs[phs_idx].set_xscale('log')
        # #axs[phs_idx].set_ylim([-60, 40])
        # axs[phs_idx].grid()
        # axs[phs_idx].spines['top'].set_visible(False)
        # axs[phs_idx].spines['right'].set_visible(False)
        # axs[phs_idx].spines['bottom'].set_visible(False)
        # axs[phs_idx].spines['left'].set_visible(False)
        # axs[phs_idx].set_ylabel('Angle (radians)')

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
        unit_circle = circle1 = plt.Circle((0, 0), 1, color='k', fill=False,zorder=1)
        axs[plot_idx].add_patch(unit_circle)
        axs[plot_idx].set_ylim([-1.1, 1.1])
        axs[plot_idx].set_xlim([-1.1, 1.1])
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
        plt.savefig(f'./data/plots/synth/{idx}.png')
        plt.close('all')
