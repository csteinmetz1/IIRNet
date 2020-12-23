import torch
import numpy as np

def polyval(p, x):
    """ Evalute a polynomial at specific values. 
    
    Args:
        p (batch, coeffs)
        x (values)

    Returns:
        v (batch, values)

    """
    bs, N = p.size()
    val = 0
    for i in range(N-1):
        val = (val+p[:,i]) * x.view(-1,1)
    return (val + p[:,N-1]).T

def roots(p):
    n = p.size(-1)
    A = torch.diag(torch.ones(n-2),-1)
    A[0,:] = -p[1:] / p[0]
    #r = torch.symeig(A)
    r = torch.eig(A)

    r_eig = np.linalg.eigvals(A)
    r_np = np.roots(p.cpu().numpy())

    return r

def freqz(b, a=1, worN=512, whole=False, fs=2*np.pi, log=False, include_nyquist=False):
    """ Compute the frequency response of a digital filter. """

    h = None

    lastpoint = 2 * np.pi if whole else np.pi

    if log:
        w = np.logspace(0, lastpoint, worN, endpoint=include_nyquist and not whole)
    else:
        w = np.linspace(0, lastpoint, worN, endpoint=include_nyquist and not whole)

    w = torch.tensor(w)

    if a.size() == 1: 
        n_fft = worN if whole else worN * 2 
        h = torch.fft.rfft(b, n=n_fft)[:worN]
        h /= a

    if h is None:  
        zm1 = torch.exp(-1j * w)
        print(b.shape, zm1.shape)
        h = (polyval(b, zm1) /
            polyval(a, zm1))

    w = w*fs/(2*np.pi) 

    return w, h

def mag(x, eps=1e-8):
    real = (x.real ** 2)
    imag = (x.imag ** 2)
    mag = torch.sqrt(torch.clamp(real + imag, min=eps))

    return mag

def sosfreqz(sos, worN=512, whole=False, fs=2*np.pi, log=False):
    """ Compute the frequency response of a digital filter in SOS format. 
    
    Args:
        sos (Tensor): Array of second-order filter coefficients, with shape 
        ``(n_sections, 6)``.

    Returns: 

    
    """

    sos, bs, n_sections = _validate_sos(sos)

    if n_sections == 0:
        raise ValueError('Cannot compute frequencies with no sections')
    h = 1.

    # check for batches (if none add batch dim)
    if bs == 0:
        sos = sos.unsqueeze(0)

    for row in torch.chunk(sos, n_sections, dim=1):
        row = row.view(bs, 6)
        w, rowh = freqz(row[:,:3], row[:,3:], worN=worN, whole=whole, fs=fs, log=log)
        h *= rowh

    return w, h

def _validate_sos(sos, eps=1e-8):
    """ Helper to validate a SOS input. """

    if sos.ndim == 2:
        n_sections, m = sos.shape
        bs = 0
    elif sos.ndim == 3:
        bs, n_sections, m = sos.shape
        # flatten batch into sections dim
        sos = sos.view(-1,6)
    else:
        raise ValueError('sos array must be shape (batch, n_sections, 6) or (n_sections, 6)')

    if m != 6:
        raise ValueError('sos array must be shape (batch, n_sections, 6) or (n_sections, 6)')

    # remove zero padded sos
    sos = sos[sos.sum(-1) != 0,:]

    # normalize by a0
    a0 = sos[:,3].unsqueeze(-1)
    sos = sos/a0

    if not (sos[:, 3] == 1).all():
        raise ValueError('sos[:, 3] should be all ones')

    # fold sections back into batch dim
    if bs > 0:
        sos = sos.view(bs,-1,6)

    return sos, bs, n_sections