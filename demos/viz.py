import sys
import torch
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from iirnet.designer import Designer

from tqdm import tqdm


def plot_response(sos, m, idx=0, dark=False):

    if dark:
        plt.style.use("dark_background")

    fig, axs = plt.subplots(figsize=(7, 3), ncols=2)

    w, h = scipy.signal.sosfreqz(sos.numpy(), fs=2)
    z, p, k = scipy.signal.sos2zpk(sos.squeeze().numpy())

    est = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    spec = plt.rcParams["axes.prop_cycle"].by_key()["color"][4]

    p_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][3]
    z_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][1]

    axs[0].plot(
        w,
        20 * np.log10(np.abs(h)),
        label="Estimation",
        c=est,
        linewidth=2,
    )
    axs[0].plot(
        w,
        m,
        c=spec,
        linewidth=1.2,
        linestyle="--",
        label="Specification",
    )
    # axs[0].spines["top"].set_visible(False)
    # axs[0].spines["bottom"].set_visible(False)
    # axs[0].spines["left"].set_visible(False)
    # axs[0].spines["right"].set_visible(False)

    axs[0].set_ylabel("dB")
    axs[0].set_xlabel("Normalized frequency")
    # axs[0].grid(c="lightgray")
    axs[0].set_ylim([-42, 42])
    # xs[0].set_xlim([-1.5, 1.5])
    # axs[0].legend()

    # pole-zero plot
    for pole in p:
        axs[1].scatter(
            np.real(pole),
            np.imag(pole),
            c=p_color,
            s=18,
            marker="x",
            facecolors="none",
        )
    for zero in z:
        axs[1].scatter(
            np.real(zero),
            np.imag(zero),
            s=18,
            marker="o",
            facecolors="none",
            edgecolors=z_color,
        )

    # unit circle
    circle_color = "w" if dark else "k"
    unit_circle = plt.Circle((0, 0), 1, color=circle_color, fill=False)
    axs[1].add_patch(unit_circle)
    axs[1].set_ylim([-1.1, 1.1])
    axs[1].set_xlim([-1.1, 1.1])
    # axs[1].grid()
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].spines["bottom"].set_visible(False)
    axs[1].spines["left"].set_visible(False)
    axs[1].set_aspect("equal")
    axs[1].axis("off")
    axs[1].set_axisbelow(True)
    # axs[1].set_ylabel("Im")
    # axs[1].set_xlabel("Re")

    plt.tight_layout()
    if dark:
        plt.savefig(f"frames_dark/{idx:03d}.png")
    else:
        plt.savefig(f"frames/{idx:03d}.png")
    # plt.savefig(f"frames/demo.png")

    plt.close("all")


def sine_sweep(w1, w2, T=1.0, num=50):
    t = np.linspace(0, 1, num=num)
    x = np.sin((w1 * T) / (np.log(w2 / w1)) * ((np.exp((t / T) * (w2 / w1))) - 1))
    return x


if __name__ == "__main__":

    # first load IIRNet with pre-trained weights
    designer = Designer()

    n = 32  # Desired filter order (4, 8, 16, 32, 64)

    num_frames = 1000

    func = np.linspace(-3.0, 3.0, num=num_frames)
    gain = (2 * np.sin(np.linspace(0, 2 * np.pi * 4, num=num_frames))) + 12
    dc = np.linspace(-2.0, 10.0, num=num_frames)
    # gain = 20 * sine_sweep(0.4, 2.0, num=num_frames)

    # plt.plot(gain)
    # plt.show()
    # sys.exit(0)

    for idx, (g, f, d) in tqdm(enumerate(zip(gain, func, dc))):
        x = np.linspace(0, 25, num=512)
        m = g * (np.sin(x) + np.sin(f * x)) + d
        mode = "linear"  # interpolation mode for specification
        output = "sos"  # Output type ("sos", or "ba")

        sos = designer(n, m, mode, output)

        plot_response(sos, m, idx=idx, dark=True)
        # sys.exit(0)
