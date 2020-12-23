import sys
import torch
import numpy as np
import scipy.signal
from scipy.stats import loguniform

class IIRFilterDataset(torch.utils.data.Dataset):
  def __init__(self,
               num_points = 512,
               max_order = 10,
               eps = 1e-8,
               factor = 1,
               num_examples = 10000,
               standard_norm = False):
    super(IIRFilterDataset, self).__init__()
    self.num_points = num_points
    self.max_order = max_order
    self.eps = eps
    self.factor = factor
    self.num_examples = num_examples
    self.standard_norm = standard_norm

    # normalizting coef
    self.sample_size = int(10e3)
    self.stats = {}
  
    if standard_norm:
        print("Computing normalization factors...")
        coefs = np.zeros((self.sample_size, (self.max_order + 1) * 2))
        mags = np.zeros((self.sample_size, self.num_points))
        phss = np.zeros((self.sample_size, self.num_points))

        for n in range(self.sample_size):
            sys.stdout.write(f"* {n+1}/{self.sample_size}\r")
            sys.stdout.flush()
            mag, phs, sos = self.generate_filter()
            sos[n,:] = sos
            mags[n,:] = mag
            phss[n,:] = phs

        # compute statistics
        self.stats["coef"] = {
            "mean" : np.mean(sos, axis=0),
            "std" : np.std(sos, axis=0)
        }
        self.stats["mag"] = {
            "mean" : np.mean(mags, axis=0),
            "std" : np.std(mags, axis=0)
        }
        self.stats["phs"] = {
            "mean" : np.mean(phss, axis=0),
            "std" : np.std(phss, axis=0)
        }

  def __len__(self):
    return self.num_examples

  def __getitem__(self, idx):
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
    n_sections = sos.size(0)
    empty_sections = (self.max_order//2) - n_sections
    full_sos = torch.zeros(self.max_order//2,6)
    full_sos[:n_sections,:] = sos
    full_sos[n_sections:,:] = torch.tensor([1.,0.,0.,1.,0.,0.]).repeat(empty_sections,1) #torch.zeros(empty_sections,6)
    sos = full_sos
    
    return mag, phs, real, imag, sos

  def generate_filter(self):   
    """ Generate a random filter along with its magnitude and phase response.
    
    Returns:
        coef (ndarray): Recursive filter coeffients stored as [b0, b1, ..., bN, a0, a1, ..., aN].
        mag (ndarray): Magnitude response of the filter (linear) of `num_points`.
        phs (ndarray): Phase response of the filter (unwraped) of 'num_points`.

    """
    # first generate random coefs
    # we lay out coefs in an array [b0, b1, b2, a0, a1, a2]
    # in the future we will want to enforce some kind of normalization
    #coef = self.factor * (np.random.rand((self.filter_order + 1) * 2) * 2) - 1

    wn = float(loguniform.rvs(1e-3, 1e0))
    rp = np.random.rand() * 10
    N = np.random.randint(1,self.max_order)
    sos = scipy.signal.cheby1(N, rp, wn, output='sos')

    w, h = scipy.signal.sosfreqz(sos, worN=self.num_points)

    mag = np.abs(h)
    phs = np.unwrap(np.angle(h))
    real = np.real(h)
    imag = np.imag(h)

    return mag, phs, real, imag, sos
