import os
import io
import PIL.Image
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor

import iirnet.signal as signal

def plot_response_grid(pred_coefs, target_coefs, num_points=512, eps=1e-8, fs=44100):

    ncols = 5
    nrows = 5
    pred_coefs = pred_coefs[:ncols*nrows]
    target_coefs = target_coefs[:ncols*nrows]

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 12))
    axs = axs.reshape(-1)

    for idx, (p, t) in enumerate(zip(pred_coefs, target_coefs)):

        w_pred, h_pred = signal.sosfreqz(p, worN=num_points, fs=fs)
        w_target, h_target = signal.sosfreqz(t, worN=num_points, fs=fs)

        mag_pred = 20 * np.log10(np.abs(h_pred.squeeze()) + 1e-8)
        mag_target = 20 * np.log10(np.abs(h_target.squeeze()) + 1e-8)
        axs[idx].plot(w_target, mag_target, color='b', label="target")
        axs[idx].plot(w_pred, mag_pred, color='r', label="pred")
        axs[idx].set_xscale('log')
        axs[idx].set_ylim([-60, 40])
        #axs[0].legend()
        axs[idx].grid()
        axs[idx].spines['top'].set_visible(False)
        axs[idx].spines['right'].set_visible(False)
        axs[idx].spines['bottom'].set_visible(False)
        axs[idx].spines['left'].set_visible(False)

        if (idx) % ncols == 0:
            axs[idx].set_ylabel('Amplitude [dB]')

        if idx > (nrows * ncols) - ncols:
            axs[idx].set_xlabel('Frequency [Hz]')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)#.unsqueeze(0)
    plt.close('all')

    return image

def plot_compare_response(pred_coef, target_coef, num_points=512, eps=1e-8, fs=44100, ax=None):

    w_pred, h_pred = signal.sosfreqz(pred_coef, worN=num_points, fs=fs)
    w_target, h_target = signal.sosfreqz(target_coef, worN=num_points, fs=fs)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

    mag_pred = 20 * np.log10(np.abs(h_pred.squeeze()) + 1e-8)
    mag_target = 20 * np.log10(np.abs(h_target.squeeze()) + 1e-8)
    ax[0].plot(w_target, mag_target, color='b', label="target")
    ax[0].plot(w_pred, mag_pred, color='r', label="pred")
    ax[0].set_xscale('log')
    ax[0].set_ylim([-60, 40])
    ax[0].set_ylabel('Amplitude [dB]')
    ax[0].set_xlabel('Frequency [Hz]')
    ax[0].legend()
    ax[0].grid()
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['bottom'].set_visible(False)
    ax[0].spines['left'].set_visible(False)

    ang_pred = np.unwrap(np.angle(h_pred.squeeze()))
    ang_target = np.unwrap(np.angle(h_target.squeeze()))
    ax[1].plot(w_target, ang_target, color='b', label="target")
    ax[1].plot(w_pred, ang_pred, color='r', label="pred")
    ax[1].set_ylabel('Angle (radians)')
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_xscale('log')
    ax[1].grid()
    ax[1].axis('tight')
    ax[1].legend()
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['left'].set_visible(False)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)#.unsqueeze(0)

    plt.close("all")

    return image
