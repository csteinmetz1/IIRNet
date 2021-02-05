import sys
import torch
import numpy as np
import scipy.signal
import multiprocessing
from itertools import repeat

from iirnet.filter import *

class IIRFilterDataset(torch.utils.data.Dataset):
    """

    method: ['pass', 'parametric', 'char_poly']

    """
    def __init__(self,
                method = "pass",
                num_points = 512,
                max_order = 10,
                eps = 1e-8,
                num_examples = 10000,
                standard_norm = False,
                precompute = True):
        super(IIRFilterDataset, self).__init__()
        self.num_points = num_points
        self.max_order = max_order
        self.eps = eps
        self.num_examples = num_examples
        self.standard_norm = standard_norm
        self.precompute = precompute

        if method == "pass":
            self.generate_filter = generate_pass_filter
        elif method == "parametric":
            self.generate_filter = generate_parametric_eq
            self.max_order = 10
        elif method == "char_poly":
            self.generate_filter = generate_characteristic_poly_filter
        else:
            raise ValueError(f"Invalid method: {method}")
        
        # normalizting coef
        self.sample_size = int(10e3)
        self.stats = {}

        if standard_norm:
            print("Computing normalization factors...")
            soss = np.zeros((self.sample_size, (self.max_order//2), 6))
            mags = np.zeros((self.sample_size, self.num_points))
            phss = np.zeros((self.sample_size, self.num_points))

            for n in range(self.sample_size):
                sys.stdout.write(f"* {n+1}/{self.sample_size}\r")
                sys.stdout.flush()
                mag, phs, real, imag, sos = self.generate_filter(num_points=num_points, max_order=max_order)
                soss[n,...] = sos
                mags[n,:] = mag
                phss[n,:] = phs

            # compute statistics
            self.stats["coef"] = {
                "mean" : np.mean(sos),
                "std" : np.std(sos)
            }
            self.stats["mag"] = {
                "mean" : np.mean(mags),
                "std" : np.std(mags)
            }
            self.stats["phs"] = {
                "mean" : np.mean(phss),
                "std" : np.std(phss)
            }
            print(self.stats)

        if self.precompute:
            self.examples = []
            params = list(repeat((num_points, max_order), times = self.num_examples))
            with multiprocessing.Pool(processes=24) as pool:
                self.examples = pool.starmap(self.generate_filter, params)
            print(len(self.examples))
        
    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        if self.precompute:
            mag, phs, real, imag, sos = self.examples[idx]
        else:
            # generate random filter coeffiecents
            mag, phs, real, imag, sos = self.generate_filter()
        
        # apply normalization
        if self.standard_norm:
            mag = (mag - self.stats["mag"]["mean"]) / self.stats["mag"]["std"] 
            phs = (phs - self.stats["phs"]["mean"]) / self.stats["phs"]["std"] 

        # convert to float32 tensor
        mag = torch.tensor(mag.astype('float32'))
        phs = torch.tensor(phs.astype('float32'))
        real = torch.tensor(real.astype('float32'))
        imag = torch.tensor(imag.astype('float32'))
        sos = torch.tensor(sos.astype('float32'))

        # fill empty sos with zeros 
        # we do this just to stack batches (make sure to ignore zero sos)
        #n_sections = sos.size(0)
        #empty_sections = (self.max_order//2) - n_sections
        #full_sos = torch.zeros(self.max_order//2,6)
        #full_sos[:n_sections,:] = sos
        #full_sos[n_sections:,:] = torch.tensor([1.,0.,0.,1.,0.,0.]).repeat(empty_sections,1) #torch.zeros(empty_sections,6)
        #sos = full_sos
    
        return mag, phs, real, imag, sos

