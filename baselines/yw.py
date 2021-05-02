import sys
import torch
import scipy.linalg
from scipy import signal as scisiganl
import numpy as np

from iirnet.loss import LogMagTargetFrequencyLoss

def yulewalk(N, f, m, npt=512):
    """ Design an N-th order IIR filter using Yule-Walker.

    Args:
        N (int): Filter order. 
        f (ndarray): Array of frequency points.
        m (ndarray): Array of desired magnitude response.
        npt (int): FFT size.  must be power of 2.

    Returns:
        b (ndarray): Denominator coefficients. 
        a (ndarray): Numerator coefficients

    Based upon the MATLAB function `yulewalk.m`

    """

    lap = np.floor(npt/25)

    num_f = f.shape[0]
    num_m = m.shape[0]
    
    assert num_f == num_m # must be same size

    nbrk = num_f

    # Hold the interpolated target response here
    npt = npt + 1
    Ht = np.zeros((1,npt))

    # check that frequencies are laid out correctly
    df = np.diff(f)
    if not np.all(df >= 0):
        raise ValueError("Yule Walker requires monotonic frequency points.")

    # apply linear interpolation if needed to
    # increase the size of the frequency/mag target

    nb = 0
    Ht[0] = m[0]

    for i in np.arange(nbrk-1):
        
        if df[i] == 0:
            nb = int(nb - lap/2)
            ne = int(nb + lap)
        else:
            ne = int(np.floor(f[i+1]*npt))
        
        if nb < 0 and ne > npt:
            raise ValueError("Signal error.")

        j = np.arange(nb,ne)

        if ne == nb: 
            inc = 0
        else: 
            inc = (j-nb) / (ne-nb)

        Ht[:,nb:ne] = inc * m[i+1] + (1-inc)*m[i]

        nb = ne

    # stack negative frequencies
    Ht = np.concatenate((Ht, Ht[:,npt-2:0:-1]), axis=-1)

    n = Ht.shape[-1]
    n2 = int(np.floor((n+1)/2))
    nb = N
    nr = 4*N
    nt = np.arange(0,nr,1)

    # compute correlation function of magnitude squared response
    R = np.real(np.fft.ifft(Ht ** 2))

    # pick NR correlations
    R = R[:,:nr] * (0.54 + 0.46 * np.cos(np.pi * nt / (nr-1))) 

    # Form window to be used in the extracting the right "wing"
    # of two-sided covariance sequence.
    RWindow = np.concatenate(([1/2], np.ones((int(n2-1))), np.zeros((int(n-n2)))))

    # compute denominator (we will need to relfect poles still)
    a = denf(R,N) 

    # divide first term
    h = np.concatenate(([R[:,0]/2], R[:,1:nr]),axis=-1)

    # compute additive decomposition
    Qh = numf(h, a, N)
    
    # compute impulse response
    _, Ss = 2 * np.real(scipy.signal.freqz(Qh, a, n, whole=True))
    Ss = Ss.astype(complex) # required to mimic matlab operation
    hh = np.fft.ifft(np.exp(np.fft.fft(RWindow * np.fft.ifft(np.log(Ss)))))
    b = np.real(numf(hh[0:nr+1], a, N))

    return b, a

def numf(h,a,N):
    """ Compute numerator given impulse-response of B/A and denominator.

    Args:
        h (ndarray): Impulse response.
        a (ndarray): Denominator coefficients.
        N (int): Filter order

    """

    nh = h.shape[-1]

    # create impulse
    imp = np.zeros(nh)
    imp[0] = 1

    # compute impulse response
    b = np.array([1])
    impr = scipy.signal.lfilter(b,a.reshape(-1),imp) 

    # compute numerator
    b = np.zeros(N+1)
    b[0] = 1
    b = np.linalg.lstsq(
            scipy.linalg.toeplitz(b, impr).T, 
            h.reshape(-1,1), 
            rcond=None
        )

    return b[0]

def denf(R,N):
    """ Compute denominator from covariances. 
    
    Args:
        R (ndarray): Covariances.
        N (int): Filter order.
    
    Returns
        a (ndarray): Denomiantor coefficients

    """ 

    nr = R.shape[-1]
    Rm = scipy.linalg.toeplitz(R[:,N:nr-1], R[:,N:0:-1])
    Rhs = - R[:,N+1:nr]
    A = np.linalg.lstsq(
            Rm, 
            Rhs.reshape(-1,1), 
            rcond=None
    )
    a = np.concatenate(([[1]], A[0]))

    return a

class YuleWalkerFilterDesign(torch.nn.Module):
    """ Design a filter with modified Yule-Walker Equations.

    """
    def __init__(self, N=32, verbose=True):
        super(YuleWalkerFilterDesign, self).__init__()
        self.N = N
        self.verbose = verbose
        self.magtarget = LogMagTargetFrequencyLoss()

    def __call__(self, target_dB):

        f = np.linspace(0,1,num=target_dB.shape[-1])


        m = target_dB.clone().squeeze().numpy()
        m = 10 ** (m/20)
        m /= np.max(m)
        npt = m.shape[-1]

        b, a = yulewalk(
                self.N-1, 
                f, 
                m, 
                npt=npt
            )

        out_sos = scipy.signal.tf2sos(b.reshape(-1), a.reshape(-1))
        #out_sos = torch.ones(1,16,6, requires_grad=False)
        out_sos = torch.tensor(out_sos).unsqueeze(0)
        #print(out_sos.shape)

        return out_sos