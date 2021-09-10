import torch
import numpy as np


def polyval(p, x):
    """Evalute a polynomial at specific values.

    Args:
        p (batch, coeffs)
        x (values)

    Returns:
        v (batch, values)

    """
    bs, N = p.size()
    val = torch.tensor([0], device=p.device)
    for i in range(N - 1):
        val = (val + p[:, i]) * x.view(-1, 1)
    return (val + p[:, N - 1]).T


def roots(p):
    n = p.size(-1)
    A = torch.diag(torch.ones(n - 2), -1)
    A[0, :] = -p[1:] / p[0]
    # r = torch.symeig(A)
    r = torch.eig(A)

    r_eig = np.linalg.eigvals(A)
    r_np = np.roots(p.cpu().numpy())

    return r


def freqz(
    b, a=1, worN=512, whole=False, fs=2 * np.pi, log=False, include_nyquist=False
):
    """Compute the frequency response of a digital filter."""

    h = None

    lastpoint = 2 * np.pi if whole else np.pi

    if log:
        w = np.logspace(0, lastpoint, worN, endpoint=include_nyquist and not whole)
    else:
        w = np.linspace(0, lastpoint, worN, endpoint=include_nyquist and not whole)

    w = torch.tensor(w, device=b.device)

    if a.size() == 1:
        n_fft = worN if whole else worN * 2
        h = torch.fft.rfft(b, n=n_fft)[:worN]
        h /= a

    if h is None:
        zm1 = torch.exp(-1j * w)
        h = polyval(b, zm1) / (polyval(a, zm1) + 1e-16)

    # need to catch NaNs here

    w = w * fs / (2 * np.pi)

    return w, h


def freqz_fft(
    b,
    a,
    worN=512,
    whole=False,
    fs=2 * np.pi,
    log=False,
    include_nyquist=False,
    eps=1e-8,
):

    lastpoint = 2 * np.pi if whole else np.pi
    w = np.linspace(0, lastpoint, worN, endpoint=include_nyquist and not whole)

    B = torch.fft.rfft(b, n=worN * 2)[..., :worN]
    A = torch.fft.rfft(a, n=worN * 2)[..., :worN]

    w = w * fs / (2 * np.pi)

    h = (B / (A + eps)) + eps

    return w, h


def mag(x, eps=1e-8):
    real = x.real ** 2
    imag = x.imag ** 2
    mag = torch.sqrt(torch.clamp(real + imag, min=eps))

    return mag


def sosfreqz(sos, worN=512, whole=False, fs=2 * np.pi, log=False, fast=False):
    """Compute the frequency response of a digital filter in SOS format.

    Args:
        sos (Tensor): Array of second-order filter coefficients, with shape
        (n_sections, 6) or (batch, n_sections, 6).

    Returns:


    """

    sos, bs, n_sections = _validate_sos(sos)

    if n_sections == 0:
        raise ValueError("Cannot compute frequencies with no sections")
    h = 1.0

    # check for batches (if none add batch dim)
    if bs == 0:
        sos = sos.unsqueeze(0)

    # this method of looping over SOS is somewhat slow

    if not fast:
        for row in torch.chunk(sos, n_sections, dim=1):
            # remove batch elements that are NaN
            row = torch.nan_to_num(row)
            row = row.reshape(-1, 6)  # shape: (batch_dim, 6)
            w, rowh = freqz_fft(
                row[:, :3], row[:, 3:], worN=worN, whole=whole, fs=fs, log=log
            )
            h *= rowh
    else:  # instead, move all SOS onto batch dim, compute response, then move back
        sos = sos.view(bs * n_sections, 6)
        w, sosh = freqz(sos[:, :3], sos[:, 3:], worN=worN, whole=whole, fs=fs, log=log)
        sosh = sosh.view(bs, n_sections, -1)
        for rowh in torch.chunk(sosh, n_sections, dim=1):
            rowh = rowh.view(bs, -1)
            h *= rowh

    return w, h


def _validate_sos(sos, eps=1e-8, normalize=False):
    """Helper to validate a SOS input."""

    if sos.ndim == 2:
        n_sections, m = sos.shape
        bs = 0
    elif sos.ndim == 3:
        bs, n_sections, m = sos.shape
        # flatten batch into sections dim
        sos = sos.reshape(bs * n_sections, 6)
    else:
        raise ValueError(
            "sos array must be shape (batch, n_sections, 6) or (n_sections, 6)"
        )

    if m != 6:
        raise ValueError(
            "sos array must be shape (batch, n_sections, 6) or (n_sections, 6)"
        )

    # remove zero padded sos
    # sos = sos[sos.sum(-1) != 0,:]

    # normalize by a0
    if normalize:
        a0 = sos[:, 3].unsqueeze(-1)
        sos = sos / a0

    # if not (sos[:, 3] == 1).all():
    #    raise ValueError('sos[:, 3] should be all ones')

    # fold sections back into batch dim
    if bs > 0:
        sos = sos.view(bs, -1, 6)

    return sos, bs, n_sections
