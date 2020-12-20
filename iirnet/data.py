import sys
import torch
import numpy as np
import scipy.signal
from scipy.stats import loguniform

class IIRFilterDataset(torch.utils.data.Dataset):
  def __init__(self,
               num_points = 512,
               filter_order = 2,
               eps = 1e-8,
               factor = 1,
               num_examples = 10000,
               standard_norm = True):
    super(IIRFilterDataset, self).__init__()
    self.num_points = num_points
    self.filter_order = filter_order
    self.eps = eps
    self.factor = factor
    self.num_examples = num_examples
    self.standard_norm = standard_norm

    # normalizting coef
    self.sample_size = int(10e3)
    self.stats = {}
  
    if standard_norm:
      print("Computing normalization factors...")
      coefs = np.zeros((self.sample_size, (self.filter_order + 1) * 2))
      mags = np.zeros((self.sample_size, self.num_points))
      phss = np.zeros((self.sample_size, self.num_points))

      for n in range(self.sample_size):
        sys.stdout.write(f"* {n+1}/{self.sample_size}\r")
        sys.stdout.flush()
        coef, mag, phs = self.generate_filter()
        coefs[n,:] = coef
        mags[n,:] = mag
        phss[n,:] = phs

      # compute statistics
      self.stats["coef"] = {
          "mean" : np.mean(coefs, axis=0),
          "std" : np.std(coefs, axis=0)
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
    coef, mag, phs = self.generate_filter()
    
    # apply normalization
    if self.stats is not None:
      #coef = (coef - self.stats["coef"]["mean"]) / self.stats["coef"]["std"] 
      mag = (mag - self.stats["mag"]["mean"]) / self.stats["mag"]["std"] 
      phs = (phs - self.stats["phs"]["mean"]) / self.stats["phs"]["std"] 

    # convert to float32
    mag = mag.astype('float32')
    phs = phs.astype('float32')
    coef = coef.astype('float32')

    return mag, phs, coef

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
    b, a = scipy.signal.cheby1(self.filter_order, rp, wn)
    coef = np.concatenate((b, a), axis=-1)

    w, h = scipy.signal.freqz(b=coef[:3], a=coef[3:], worN=self.num_points)

    mag = np.abs(h)
    phs = np.unwrap(np.angle(h))

    return coef, mag, phs
