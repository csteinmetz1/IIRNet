import sys
import torch
import numpy as np
import scipy.signal
import multiprocessing
from itertools import repeat

from iirnet.filter import *


class IIRFilterDataset(torch.utils.data.Dataset):
    """Dataset class to generate random IIR filters.

    method (str): Random filter sampling method. Default: 'char_poly'
        ['pass', 'parametric', 'char_poly', 'uniform_parametric']
    num_points (int): Number of points in the FFT for computing magnitude response. Default: 512
    num_examples (int): Number of filters per epoch. Default: 10,000.
    precompute (bool): Precompute `num_examples` filters before training. Default: False
    """

    def __init__(
        self,
        method="char_poly",
        num_points=512,
        max_order=10,
        min_order=None,
        num_examples=10000,
        precompute=False,
    ):
        super(IIRFilterDataset, self).__init__()
        self.num_points = num_points
        self.max_order = max_order
        self.min_order = min_order
        self.num_examples = num_examples
        self.precompute = precompute

        # assign the filter generation func
        if method == "pass":
            self.generate_filter = generate_pass_filter
        elif method == "parametric":
            self.generate_filter = generate_parametric_eq
        elif method == "char_poly":
            self.generate_filter = generate_characteristic_poly_filter
        elif method == "uniform_parametric":
            self.generate_filter = generate_uniform_parametric_eq
        elif method == "normal_biquad":
            self.generate_filter = generate_normal_biquad
        elif method == "uniform_mag_disk":
            self.generate_filter = generate_uniform_mag_disk_filter
        elif method == "gaussian_peaks":
            self.generate_filter = generate_gaussian_peaks
        elif method == "normal_poly":
            self.generate_filter = generate_normal_poly_filter
        elif method == "uniform_disk":
            self.generate_filter = generate_uniform_disk_filter
        else:
            raise ValueError(f"Invalid method: {method}")

        if self.precompute:
            self.examples = []
            params = list(repeat((num_points, max_order), times=self.num_examples))
            with multiprocessing.Pool(processes=24) as pool:
                self.examples = pool.starmap(self.generate_filter, params)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        if self.precompute:
            mag, phs, real, imag, sos = self.examples[idx]
        else:
            # generate random filter coeffiecents
            mag, phs, real, imag, sos = self.generate_filter(
                self.num_points,
                self.max_order,
            )

        # convert to float32 tensor
        mag = torch.tensor(mag.astype("float32"))
        phs = torch.tensor(phs.astype("float32"))
        real = torch.tensor(real.astype("float32"))
        imag = torch.tensor(imag.astype("float32"))
        sos = torch.tensor(sos.astype("float32"))

        return mag, phs, real, imag, sos
