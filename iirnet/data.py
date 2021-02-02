import sys
import torch
import numpy as np
from numpy.random import default_rng
from numpy import linalg as LA
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
    mag, phs, real, imag, sos = self.generate_characteristic_filter()
    
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

    btype = np.random.choice(['lowpass', 'highpass'])

    wn = float(loguniform.rvs(1e-3, 1e0))
    rp = np.random.rand() * 20
    N = np.random.randint(1,self.max_order)

    sos = scipy.signal.cheby1(N, rp, wn, output='sos', btype=btype)

    w, h = scipy.signal.sosfreqz(sos, worN=self.num_points)

    mag = np.abs(h)
    phs = np.unwrap(np.angle(h))
    real = np.real(h)
    imag = np.imag(h)

    return mag, phs, real, imag, sos

  def generate_nercessian_filter(self):
    """ Generate a random filter according to the method specified in Nercissian's paper

    Returns: 
        coef
        mag
        phs
    """
    rng = default_rng()
    zeros = []
    poles = []
    sos_holder = []
    f_min = 20
    f_max = 20000
    f_s = 48000 #Sampling frequency used in paper
    g_min = -10
    g_max = 10
    q_min = 0.1
    q_max_shelf = 1
    q_max_peak = 3
    bern_shelf = 1.0 #Probability shelf filters has non-zero dB gain paper=0.5
    bern_peak = 1.0 #Probability shelf filters has non-zero dB gain paper=0.333
    num_peaks = 3 #Number of peaking filters to use paper=10

    ##Low Shelf Filter
    f_low = rng.beta(0.25,5)*(f_max-f_min)+f_min
    omega_low = 2*np.pi*f_low/f_s
    g = rng.binomial(1,bern_shelf)*(rng.beta(5,5)*(g_max-g_min)+g_min)
    q = rng.beta(1,5)*(q_max_shelf-q_min)+q_min
    A = np.power(10,g/40)
    alpha = np.sin(omega_low)*np.sqrt((A**2+1)*((1/q)-1)+2*A)

    b0 = A*((A+1)-(A-1)*np.cos(omega_low)+alpha)
    b1 = 2*A*((A-1)-(A+1)*np.cos(omega_low))
    b2 = A*((A+1)-(A-1)*np.cos(omega_low)-alpha)

    a0 = (A+1)+(A-1)*np.cos(omega_low)+alpha
    a1 = -2*A*((A-1)+(A+1)*np.cos(omega_low))
    a2 = (A+1)+(A-1)*np.cos(omega_low)-alpha

    sos_poly = np.asarray([b0,b1,b2,a0,a1,a2])
    sos_holder.append(sos_poly)
    num_poly = np.asarray([b0,b1,b2])
    zeros.append(num_poly)
    den_poly = np.asarray([a0,a1,a2])
    poles.append(den_poly)


    ##High Shelf Filter
    f_high = rng.beta(4,5)*(f_max-f_min)+f_min
    omega_high = 2*np.pi*f_high/f_s
    g = rng.binomial(1,bern_shelf)*(rng.beta(5,5)*(g_max-g_min)+g_min)
    q = rng.beta(1,5)*(q_max_shelf-q_min)+q_min
    A = np.power(10,g/40)
    alpha = np.sin(omega_high)*np.sqrt((A**2+1)*((1/q)-1)+2*A)

    b0 = A*((A+1)+(A-1)*np.cos(omega_high)+alpha)
    b1 = -2*A*((A-1)+(A+1)*np.cos(omega_high))
    b2 = A*((A+1)+(A-1)*np.cos(omega_high)-alpha)

    a0 = (A+1)-(A-1)*np.cos(omega_high)+alpha
    a1 = 2*A*((A-1)-(A+1)*np.cos(omega_high))
    a2 = (A+1)-(A-1)*np.cos(omega_high)-alpha

    sos_poly = np.asarray([b0,b1,b2,a0,a1,a2])
    sos_holder.append(sos_poly)
    num_poly = np.asarray([b0,b1,b2])
    zeros.append(num_poly)
    den_poly = np.asarray([a0,a1,a2])
    poles.append(den_poly)

    ##Peaking Filters
    for jj in range(num_peaks):
        f_peak = rng.uniform(low=f_low,high=f_high)
        omega = 2*np.pi*f_peak/f_s
        g = rng.binomial(1,bern_peak)*(rng.beta(5,5)*(g_max-g_min)+g_min)
        q = rng.beta(1,5)*(q_max_peak-q_min)+q_min

        alpha = np.sin(omega)/(2*q)
        A = np.power(10,g/40)

        b0 = 1+(alpha*A)
        b1 = -2*np.cos(omega)
        b2 = 1-(alpha*A)

        a0 = 1+(alpha/A)
        a1 = -2*np.cos(omega)
        a2 = 1-(alpha/A)

        sos_poly = np.asarray([b0,b1,b2,a0,a1,a2])
        sos_holder.append(sos_poly)
        num_poly = np.asarray([b0,b1,b2])
        zeros.append(num_poly)
        den_poly = np.asarray([a0,a1,a2])
        poles.append(den_poly)

    sos = np.vstack(sos_holder)
    my_norms = sos[:,3]
    sos = sos/my_norms[:,None] ##sosfreqz requires sos[:,3]=1
    w, h = scipy.signal.sosfreqz(sos, worN=self.num_points)
    mag = np.abs(h)
    phs = np.unwrap(np.angle(h))
    real = np.real(h)
    imag = np.imag(h)
    return mag, phs, real, imag, sos

  def generate_characteristic_filter(self):
    rng = default_rng()
    num_filters = 10

    max_ord = self.max_order
    n_fft = 4098
    trim_point = int(n_fft/2+1)

    sos = []
    num_ord = rng.integers(low=1,high=max_ord)
    den_ord = rng.integers(low=1,high=max_ord)
    num_ord = max_ord ##Comment these out when we can handle variable order filters
    den_ord = max_ord ##
    chosen_max = np.max((num_ord,den_ord))
    all_num = np.zeros(chosen_max,dtype=np.cdouble)
    all_den = np.zeros(chosen_max,dtype=np.cdouble)
    num_char_matrix = rng.normal(size=(num_ord,num_ord))
    den_char_matrix = rng.normal(size=(den_ord,den_ord))
    num_w, _ = LA.eig(num_char_matrix)
    den_w, _ = LA.eig(den_char_matrix)
    sort_num = np.argsort(-1*np.abs(np.imag(num_w)))
    sort_den = np.argsort(-1*np.abs(np.imag(den_w)))
    num_w = (1/np.sqrt(max_ord))*num_w[sort_num]
    all_num[:len(num_w)] = num_w
    den_w = (1/np.sqrt(max_ord))*den_w[sort_den]
    all_den[:len(den_w)] = den_w 

    for ii in range(chosen_max//2):
        num_poly = np.real(np.polymul([1,-1*all_num[2*ii]],[1,-1*all_num[2*ii+1]]))
        den_poly = np.real(np.polymul([1,-1*all_den[2*ii]],[1,-1*all_den[2*ii+1]]))
        sos.append(np.hstack((num_poly,den_poly)))
    if chosen_max%2==1:
        num_poly = np.real(np.polymul([1,0],[1,-1*all_num[-1]]))
        den_poly = np.real(np.polymul([1,0],[1,-1*all_den[-1]]))
        sos.append(np.hstack((num_poly,den_poly)))
    sos = np.asarray(sos)
    my_norms = sos[:,3]
    sos = sos/my_norms[:,None] ##sosfreqz requires sos[:,3]=1
    w, h = scipy.signal.sosfreqz(sos, worN=self.num_points)
    mag = np.abs(h)
    phs = np.unwrap(np.angle(h))
    real = np.real(h)
    imag = np.imag(h)
    return mag, phs, real, imag, sos
