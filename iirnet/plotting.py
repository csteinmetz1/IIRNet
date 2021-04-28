import os
import io
import torch
import PIL.Image
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor

import iirnet.signal as signal

def plot_response_grid(
        pred_coefs, 
        target_coefs=None, 
        target_mags=None,
        num_points=512, 
        num_filters=5,
        eps=1e-8, 
        fs=44100
    ):

    ncols = 2
    nrows = num_filters
    pred_coefs = pred_coefs[:num_filters]

    if target_coefs is not None:
        target = target_coefs[:num_filters]
    elif target_mags is not None:
        target = target_mags[:num_filters]
    else:
        raise ValueError("Must pass either `target_coefs` or `target_mags`.")

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6, 12))
    axs = axs.reshape(-1)

    for idx, (p, t) in enumerate(zip(pred_coefs, target)):

        mag_idx = idx * 2
        plot_idx = mag_idx + 1

        try:
            zeros,poles,k = scipy.signal.sos2zpk(p.squeeze())
        except:
            zeros = []
            poles = []
            k = 0

        w_pred, h_pred = signal.sosfreqz(p, worN=num_points, fs=fs)
        mag_pred = 20 * np.log10(np.abs(h_pred.squeeze()) + 1e-8)

        if target_coefs is not None:
            w_target, h_target = signal.sosfreqz(t, worN=num_points, fs=fs)
            mag_target = 20 * np.log10(np.abs(h_target.squeeze()) + 1e-8)
        else:
            mag_target = t.squeeze()

        axs[mag_idx].plot(w_pred, mag_target, color='tab:blue', label="target")
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

def plot_responses(pred_sos, target_dB, filename=None):

    mag_idx = 0
    #phs_idx = 1
    plot_idx = 1

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))

    zeros,poles,k = scipy.signal.sos2zpk(pred_sos.squeeze())
    w_pred, h_pred = signal.sosfreqz(pred_sos, worN=target_dB.shape[-1], fs=44100)
    mag_pred = 20 * torch.log10(h_pred.abs() + 1e-8) 

    axs[mag_idx].plot(w_pred, target_dB, color='tab:blue', label="target")
    axs[mag_idx].plot(w_pred, mag_pred.squeeze(), color='tab:red', label="pred")
    #axs[mag_idx].plot(w_pred, mag_pred - target_dB, color='tab:green', label="error")

    axs[mag_idx].set_xscale('log')
    axs[mag_idx].set_ylim([-60, 40])
    axs[mag_idx].grid()
    axs[mag_idx].spines['top'].set_visible(False)
    axs[mag_idx].spines['right'].set_visible(False)
    axs[mag_idx].spines['bottom'].set_visible(False)
    axs[mag_idx].spines['left'].set_visible(False)
    axs[mag_idx].set_ylabel('Amplitude (dB)')
    axs[mag_idx].set_xlabel('Frequency (Hz)')
    axs[mag_idx].legend()

    #axs[phs_idx].plot(w_pred, np.squeeze(np.angle(h_pred)), color='tab:red', label="pred")
    #axs[phs_idx].plot(w_pred, target_h_ang, color='tab:blue', label="target")
    #axs[phs_idx].plot(w_pred, target_h_ang, color='tab:blue', label="target")
    #axs[phs_idx].set_xscale('log')
    #axs[phs_idx].set_ylim([-60, 40])
    #axs[phs_idx].grid()
    #axs[phs_idx].spines['top'].set_visible(False)
    #axs[phs_idx].spines['right'].set_visible(False)
    #axs[phs_idx].spines['bottom'].set_visible(False)
    #axs[phs_idx].spines['left'].set_visible(False)
    #axs[phs_idx].set_ylabel('Angle (radians)')

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
    if filename is not None:
        plt.savefig(f"{filename}")

    plt.close('all')
