import scipy.linalg
from scipy import signal as scisiganl
import numpy as np

def yulewalk(N, f, m, npt=512):
    """ Design an N-th order IIR filter using Yule-Walker.

    Args:
        N (int): Filter order. 
        f (ndarray): Array of frequency points.
        m (ndarray): Array of desired magnitude response.
        npt (int): FFT size. 

    Returns:
        b (ndarray): Denominator coefficients. 
        a (ndarray): Numerator coefficients

    Based upon the MATLAB function `yulewalk.m`

    """

    num_f = f.shape[0]
    num_m = m.shape[0]
    
    assert num_f == num_m # must be same size

    nbrk = num_f

    lap = np.floor(npt/25)

    # Hold the interpolated target response here
    npt = npt + 1
    Ht = np.zeros((1,npt))

    # check that frequencies are laid out correctly
    if not np.all(np.diff(f) >= 0):
        raise ValueError("Yule Walker requires monotonic frequency points.")

    # apply linear interpolation if needed to
    # increase the size of the frequency/mag target
    if num_f != npt:
       Ht = np.interp(np.linspace(0,1,num=npt), f, m)

    # stack negative frequencies
    Ht = np.concatenate((Ht, Ht[::-1]))
    n = len(Ht)
    n2 = np.floor((n+1)/2)
    nb = N
    nr = 4*N
    nt = np.arange(0,nr,1)

    # compute correlation function of magnitude squared response
    R = np.real(np.fft.ifft(Ht ** 2))

    # pick NR correlations
    R = R[:nr] * (0.54 + 0.46 * np.cos(np.pi * nt / (nr-1))) 

    # Form window to be used in the xtacting the right "wing"
    # of two-sided covariance sequence.
    RWindow = np.concatenate(([1/2], np.ones((int(n2-2))), np.zeros((int(n-n2-1)))))

    # compute denominator (we will need to relfect poles still)
    a = denf(R,N) 

    # divide first term
    h = np.concatenate(([R[0]/2], R[1:nr]))

    # Compute additive decomposition
    Qh = numf(h, a, N)

    print(Qh)


def numf(h,a,N):
    """ Compute numerator given impulse-response of B/A and denominator.

    Args:
        h (ndarray): Impulse response.
        a (ndarray): Denominator coefficients.
        N (int): Filter order

    """

    nh = h.shape[0]

    # create impulse
    imp = np.zeros(nh)
    imp[0] = 1

    # compute impulse response
    b = np.array([1])
    print(a, b, imp)
    impr = scipy.signal.lfilter(b,a.reshape(-1),imp) 

    # compute numerator
    b = np.zeros(N+1)
    b[0] = 1
    b = h / scipy.linalg.toeplitz(b, impr)

    return b

def denf(R,N):
    """ Compute denominator from covariances. 
    
    Args:
        R (ndarray): Covariances.
        N (int): Filter order.
    
    Returns
        a (ndarray): Denomiantor coefficients

    """ 

    nr = R.shape[0]
    Rm = scipy.linalg.toeplitz(R[N:nr-1], R[N+1:1:-1])
    Rhs = - R[N+1:nr]
    print(Rm.T.shape, Rhs.reshape(1,-1).shape)

    A = np.linalg.lstsq(Rm, Rhs.reshape(-1,1))
    a = np.concatenate(([[1]], A[0]))

    return a
