import sys
import torch
import numpy as np
import scipy.signal
import scipy.stats as stats

from numpy import linalg as LA
from scipy.stats import loguniform
from numpy.random import default_rng


def compute_dB_magnitude(mag_lin):
    mag_dB = 20 * np.log10(mag_lin + 1e-8)
    mag_dB = np.clip(mag_dB, a_min=-128, a_max=128)
    mag_dB /= 128  # scale between -1 and 1

    return mag_dB


def generate_pass_filter(num_points=512, max_order=2):
    """Generate a random highpass/lowpass filter along with its magnitude and phase response.

    Returns:
        coef (ndarray): Recursive filter coeffients stored as [b0, b1, ..., bN, a0, a1, ..., aN].
        mag (ndarray): Magnitude response of the filter (linear) of `num_points`.
        phs (ndarray): Phase response of the filter (unwraped) of 'num_points`.

    """
    # first generate random coefs
    # we lay out coefs in an array [b0, b1, b2, a0, a1, a2]
    # in the future we will want to enforce some kind of normalization
    # coef = self.factor * (np.random.rand((self.filter_order + 1) * 2) * 2) - 1

    btype = np.random.choice(["lowpass", "highpass"])

    wn = float(loguniform.rvs(1e-3, 1e0))
    # rp = np.random.rand() * 20
    N = max_order

    # sos = scipy.signal.cheby1(N, rp, wn, output='sos', btype=btype)
    sos = scipy.signal.butter(N, wn, output="sos", btype=btype)

    w, h = scipy.signal.sosfreqz(sos, worN=num_points)

    mag = np.abs(h)
    phs = np.unwrap(np.angle(h))
    real = np.real(h)
    imag = np.imag(h)

    mag = compute_dB_magnitude(mag)

    return mag, phs, real, imag, sos


def generate_parametric_eq(num_points, max_order, f_s=48000):
    """Generate a random parametric EQ cascase according to the method specified in
    [Nercessian 2020](https://dafx2020.mdw.ac.at/proceedings/papers/DAFx2020_paper_7.pdf).

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
    g_min = -10
    g_max = 10
    q_min = 0.1
    q_max_shelf = 1
    q_max_peak = 3
    bern_shelf = 0.5  # Probability shelf filters has non-zero dB gain paper=0.5
    bern_peak = 0.33  # Probability shelf filters has non-zero dB gain paper=0.333
    max_ord = max_order
    num_peaks = (max_order) // 2 - 2  # Number of peaking filters to use paper=10

    ##Low Shelf Filter
    f_low = rng.beta(0.25, 5) * (f_max - f_min) + f_min
    omega_low = 2 * np.pi * f_low / f_s
    g = rng.binomial(1, bern_shelf) * (rng.beta(5, 5) * (g_max - g_min) + g_min)
    q = rng.beta(1, 5) * (q_max_shelf - q_min) + q_min
    A = np.power(10, g / 40)
    alpha = np.sin(omega_low) * np.sqrt((A ** 2 + 1) * ((1 / q) - 1) + 2 * A)

    b0 = A * ((A + 1) - (A - 1) * np.cos(omega_low) + alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(omega_low))
    b2 = A * ((A + 1) - (A - 1) * np.cos(omega_low) - alpha)

    a0 = (A + 1) + (A - 1) * np.cos(omega_low) + alpha
    a1 = -2 * A * ((A - 1) + (A + 1) * np.cos(omega_low))
    a2 = (A + 1) + (A - 1) * np.cos(omega_low) - alpha

    sos_poly = np.asarray([b0, b1, b2, a0, a1, a2])
    sos_holder.append(sos_poly)
    num_poly = np.asarray([b0, b1, b2])
    zeros.append(num_poly)
    den_poly = np.asarray([a0, a1, a2])
    poles.append(den_poly)

    ##High Shelf Filter
    f_high = rng.beta(4, 5) * (f_max - f_min) + f_min
    omega_high = 2 * np.pi * f_high / f_s
    g = rng.binomial(1, bern_shelf) * (rng.beta(5, 5) * (g_max - g_min) + g_min)
    q = rng.beta(1, 5) * (q_max_shelf - q_min) + q_min
    A = np.power(10, g / 40)
    alpha = np.sin(omega_high) * np.sqrt((A ** 2 + 1) * ((1 / q) - 1) + 2 * A)

    b0 = A * ((A + 1) + (A - 1) * np.cos(omega_high) + alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(omega_high))
    b2 = A * ((A + 1) + (A - 1) * np.cos(omega_high) - alpha)

    a0 = (A + 1) - (A - 1) * np.cos(omega_high) + alpha
    a1 = 2 * A * ((A - 1) - (A + 1) * np.cos(omega_high))
    a2 = (A + 1) - (A - 1) * np.cos(omega_high) - alpha

    sos_poly = np.asarray([b0, b1, b2, a0, a1, a2])
    sos_holder.append(sos_poly)
    num_poly = np.asarray([b0, b1, b2])
    zeros.append(num_poly)
    den_poly = np.asarray([a0, a1, a2])
    poles.append(den_poly)

    ##Peaking Filters
    for jj in range(num_peaks):
        f_peak = rng.uniform(low=f_low, high=f_high)
        omega = 2 * np.pi * f_peak / f_s
        g = rng.binomial(1, bern_peak) * (rng.beta(5, 5) * (g_max - g_min) + g_min)
        q = rng.beta(1, 5) * (q_max_peak - q_min) + q_min

        alpha = np.sin(omega) / (2 * q)
        A = np.power(10, g / 40)

        b0 = 1 + (alpha * A)
        b1 = -2 * np.cos(omega)
        b2 = 1 - (alpha * A)

        a0 = 1 + (alpha / A)
        a1 = -2 * np.cos(omega)
        a2 = 1 - (alpha / A)

        sos_poly = np.asarray([b0, b1, b2, a0, a1, a2])
        sos_holder.append(sos_poly)
        num_poly = np.asarray([b0, b1, b2])
        zeros.append(num_poly)
        den_poly = np.asarray([a0, a1, a2])
        poles.append(den_poly)

    sos = np.vstack(sos_holder)
    my_norms = sos[:, 3]
    sos = sos / my_norms[:, None]  ##sosfreqz requires sos[:,3]=1
    w, h = scipy.signal.sosfreqz(sos, worN=num_points)
    mag = np.abs(h)
    phs = np.unwrap(np.angle(h))
    real = np.real(h)
    imag = np.imag(h)

    mag = compute_dB_magnitude(mag)

    return mag, phs, real, imag, sos


def generate_normal_biquad(num_points, max_order, min_order=None, norm=1.0):

    rng = default_rng()
    if min_order == None:
        chosen_ord = max_order
    else:
        chosen_ord = rng.randint(min_order, max_order)
    sos = rng.normal(scale=norm, size=(chosen_ord // 2, 6))

    a0 = sos[:, 3].reshape(-1, 1)
    sos = sos / a0

    w, h = scipy.signal.sosfreqz(sos, worN=num_points)

    mag = np.abs(h)
    phs = np.unwrap(np.angle(h))
    real = np.real(h)
    imag = np.imag(h)

    mag = compute_dB_magnitude(mag)

    return mag, phs, real, imag, sos


def generate_uniform_parametric_eq(num_points, max_order, f_s=48000):
    """Generate a random parametric EQ cascase according to the method specified in
    [Nercessian 2020](https://dafx2020.mdw.ac.at/proceedings/papers/DAFx2020_paper_7.pdf).

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
    g_min = -10
    g_max = 10
    q_min = 0.1
    q_max_shelf = 1
    q_max_peak = 3
    bern_shelf = 0.5  # Probability shelf filters has non-zero dB gain paper=0.5
    bern_peak = 0.33  # Probability shelf filters has non-zero dB gain paper=0.333
    max_ord = max_order
    num_peaks = (max_order) // 2 - 2  # Number of peaking filters to use paper=10

    ##Low Shelf Filter
    # f_low = rng.beta(0.25,5)*(f_max-f_min)+f_min
    # omega_low = 2*np.pi*f_low/f_s
    omega_low = rng.uniform(low=0.0, high=np.pi)
    # g = rng.binomial(1,bern_shelf)*(rng.beta(5,5)*(g_max-g_min)+g_min)
    g = rng.uniform(low=-10.0, high=10.0)
    # q = rng.beta(1,5)*(q_max_shelf-q_min)+q_min
    q = rng.uniform(low=0.1, high=1.0)
    A = np.power(10, g / 40)
    alpha = np.sin(omega_low) * np.sqrt((A ** 2 + 1) * ((1 / q) - 1) + 2 * A)

    b0 = A * ((A + 1) - (A - 1) * np.cos(omega_low) + alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(omega_low))
    b2 = A * ((A + 1) - (A - 1) * np.cos(omega_low) - alpha)

    a0 = (A + 1) + (A - 1) * np.cos(omega_low) + alpha
    a1 = -2 * A * ((A - 1) + (A + 1) * np.cos(omega_low))
    a2 = (A + 1) + (A - 1) * np.cos(omega_low) - alpha

    sos_poly = np.asarray([b0, b1, b2, a0, a1, a2])
    sos_holder.append(sos_poly)
    num_poly = np.asarray([b0, b1, b2])
    zeros.append(num_poly)
    den_poly = np.asarray([a0, a1, a2])
    poles.append(den_poly)

    ##High Shelf Filter
    # f_high = rng.beta(4,5)*(f_max-f_min)+f_min
    # omega_high = 2*np.pi*f_high/f_s
    omega_high = rng.uniform(low=0.0, high=np.pi)
    # g = rng.binomial(1,bern_shelf)*(rng.beta(5,5)*(g_max-g_min)+g_min)
    g = rng.uniform(low=-10.0, high=10.0)
    # q = rng.beta(1,5)*(q_max_shelf-q_min)+q_min
    q = rng.uniform(low=0.1, high=1.0)
    A = np.power(10, g / 40)
    alpha = np.sin(omega_high) * np.sqrt((A ** 2 + 1) * ((1 / q) - 1) + 2 * A)

    b0 = A * ((A + 1) + (A - 1) * np.cos(omega_high) + alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(omega_high))
    b2 = A * ((A + 1) + (A - 1) * np.cos(omega_high) - alpha)

    a0 = (A + 1) - (A - 1) * np.cos(omega_high) + alpha
    a1 = 2 * A * ((A - 1) - (A + 1) * np.cos(omega_high))
    a2 = (A + 1) - (A - 1) * np.cos(omega_high) - alpha

    sos_poly = np.asarray([b0, b1, b2, a0, a1, a2])
    sos_holder.append(sos_poly)
    num_poly = np.asarray([b0, b1, b2])
    zeros.append(num_poly)
    den_poly = np.asarray([a0, a1, a2])
    poles.append(den_poly)

    ##Peaking Filters
    for jj in range(num_peaks):
        # f_peak = rng.uniform(low=f_low,high=f_high)
        # omega = 2*np.pi*f_peak/f_s
        omega = rng.uniform(low=0.0, high=np.pi)
        # g = rng.binomial(1,bern_peak)*(rng.beta(5,5)*(g_max-g_min)+g_min)
        g = rng.uniform(low=-10, high=10)
        # q = rng.beta(1,5)*(q_max_peak-q_min)+q_min
        q = rng.uniform(low=0.1, high=3.0)

        alpha = np.sin(omega) / (2 * q)
        A = np.power(10, g / 40)

        b0 = 1 + (alpha * A)
        b1 = -2 * np.cos(omega)
        b2 = 1 - (alpha * A)

        a0 = 1 + (alpha / A)
        a1 = -2 * np.cos(omega)
        a2 = 1 - (alpha / A)

        sos_poly = np.asarray([b0, b1, b2, a0, a1, a2])
        sos_holder.append(sos_poly)
        num_poly = np.asarray([b0, b1, b2])
        zeros.append(num_poly)
        den_poly = np.asarray([a0, a1, a2])
        poles.append(den_poly)

    sos = np.vstack(sos_holder)
    my_norms = sos[:, 3]
    sos = sos / my_norms[:, None]  ##sosfreqz requires sos[:,3]=1
    w, h = scipy.signal.sosfreqz(sos, worN=num_points)
    mag = np.abs(h)
    phs = np.unwrap(np.angle(h))
    real = np.real(h)
    imag = np.imag(h)

    mag = compute_dB_magnitude(mag)

    return mag, phs, real, imag, sos


def generate_characteristic_poly_filter(
    num_points, max_order, min_order=None, eps=1e-8
):
    rng = default_rng()
    norm = 1.0  ##SHOULD BE HYPERPARAMETER
    sos = []

    if min_order == None:
        chosen_ord = max_order
    else:
        chosen_ord = rng.randint(low=min_order, high=max_order)

    num_ord = chosen_ord
    den_ord = chosen_ord
    chosen_max = chosen_ord

    all_num = np.zeros(chosen_max, dtype=np.cdouble)
    all_den = np.zeros(chosen_max, dtype=np.cdouble)
    num_char_matrix = rng.normal(size=(num_ord, num_ord))
    den_char_matrix = rng.normal(size=(den_ord, den_ord))
    num_w, _ = LA.eig(num_char_matrix)
    den_w, _ = LA.eig(den_char_matrix)
    sort_num = np.argsort(-1 * np.abs(np.imag(num_w)))
    sort_den = np.argsort(-1 * np.abs(np.imag(den_w)))
    num_w = norm * (1 / np.sqrt(chosen_ord)) * num_w[sort_num]
    all_num[: len(num_w)] = num_w
    den_w = norm * (1 / np.sqrt(chosen_ord)) * den_w[sort_den]
    all_den[: len(den_w)] = den_w

    for ii in range(chosen_max // 2):
        num_poly = np.real(
            np.polymul([1, -1 * all_num[2 * ii]], [1, -1 * all_num[2 * ii + 1]])
        )
        den_poly = np.real(
            np.polymul([1, -1 * all_den[2 * ii]], [1, -1 * all_den[2 * ii + 1]])
        )
        sos.append(np.hstack((num_poly, den_poly)))

    if chosen_max % 2 == 1:  # add an extra section to make even number of sections
        num_poly = np.real(np.polymul([1, 0], [1, -1 * all_num[-1]]))
        den_poly = np.real(np.polymul([1, 0], [1, -1 * all_den[-1]]))
        sos.append(np.hstack((num_poly, den_poly)))

    sos = np.asarray(sos)
    num_sos = sos.shape[0]
    sos_proto = np.tile(np.asarray([1.0, 0, 0, 1.0, 0, 0]), ((chosen_ord + 1) // 2, 1))
    sos_proto[:num_sos, :] = sos
    sos = sos_proto
    my_norms = sos[:, 3]
    sos = sos / my_norms[:, None]  ##sosfreqz requires sos[:,3]=1

    w, h = scipy.signal.sosfreqz(sos, worN=num_points)
    mag = np.abs(h)
    phs = np.unwrap(np.angle(h))
    real = np.real(h)
    imag = np.imag(h)

    mag = compute_dB_magnitude(mag)

    out = mag, phs, real, imag, sos

    return out


def generate_uniform_disk_filter(
    num_points, max_order, min_order=None, eps=1e-8, min_freq=20.0, log=False, fs=44100
):

    rng = default_rng()

    ##a and b are used for the loguniform sampling
    a = min_freq / (
        0.5 * fs * np.pi
    )  ##MIN CAN'T BE ZERO, CHOOSING 20HZ AS MINIMUM POLE/ZERO FREQUENCY
    b = np.pi
    norm = 0.9  ##SHOULD BE HYPERPARAMETER

    sos = []
    if min_order == None:
        num_ord = torch.tensor([max_order]).numpy()
        den_ord = torch.tensor([max_order]).numpy()
    else:
        num_ord = torch.randint(2, max_order, [1]).numpy()
        den_ord = num_ord
    chosen_max = np.max((num_ord, den_ord))
    all_num = np.zeros(chosen_max, dtype=np.cdouble)
    all_den = np.zeros(chosen_max, dtype=np.cdouble)
    zeros_mags = torch.distributions.uniform.Uniform(eps, norm).sample(num_ord // 2)
    poles_mags = torch.distributions.uniform.Uniform(eps, norm).sample(num_ord // 2)
    if not log:
        zeros_args = torch.distributions.uniform.Uniform(eps, np.pi).sample(
            num_ord // 2
        )
        poles_args = torch.distributions.uniform.Uniform(eps, np.pi).sample(
            num_ord // 2
        )
    else:
        zeros_args = loguniform.rvs(a, b, size=(num_ord) // 2)
        poles_args = loguniform.rvs(a, b, size=(num_ord) // 2)
    for z_mag, z_arg, p_mag, p_arg in zip(
        zeros_mags, zeros_args, poles_mags, poles_args
    ):
        num_poly = [1, -2 * z_mag * np.cos(z_arg), z_mag ** 2]
        den_poly = [1, -2 * p_mag * np.cos(p_arg), p_mag ** 2]
        sos.append(np.hstack((num_poly, den_poly)))
    if chosen_max % 2 == 1:  ##IF ODD, add an extra zero and pole
        zeros_mags = torch.distributions.uniform.Uniform(0.0, 1).sample()
        poles_mags = torch.distributions.uniform.Uniform(0.0, 1).sample()
        num_poly = [1, -1 * zeros_mags, 0]
        den_poly = [1, -1 * poles_mags, 0]
        sos.append(np.hstack((num_poly, den_poly)))

    sos = np.asarray(sos)
    num_sos = sos.shape[0]
    sos_proto = np.tile(np.asarray([1.0, 0, 0, 1.0, 0, 0]), ((max_order + 1) // 2, 1))
    sos_proto[:num_sos, :] = sos
    sos = sos_proto
    my_norms = sos[:, 3]
    sos = sos / my_norms[:, None]  ##sosfreqz requires sos[:,3]=1
    w, h = scipy.signal.sosfreqz(sos, worN=num_points)
    mag = np.abs(h)
    phs = np.unwrap(np.angle(h))
    real = np.real(h)
    imag = np.imag(h)

    mag = compute_dB_magnitude(mag)

    out = mag, phs, real, imag, sos

    return out


def generate_uniform_mag_disk_filter(
    num_points, max_order, min_order=None, eps=1e-8, min_freq=20.0, log=False, fs=44100
):

    rng = default_rng()

    ##a and b are used for the loguniform sampling
    a = min_freq / (
        0.5 * fs * np.pi
    )  ##MIN CAN'T BE ZERO, CHOOSING 20HZ AS MINIMUM POLE/ZERO FREQUENCY
    b = np.pi
    norm = 0.9  #

    sos = []
    if min_order == None:
        num_ord = torch.tensor([max_order]).numpy()
        den_ord = torch.tensor([max_order]).numpy()
    else:
        num_ord = torch.randint(2, max_order, [1]).numpy()
        den_ord = num_ord
    chosen_max = np.max((num_ord, den_ord))
    all_num = np.zeros(chosen_max, dtype=np.cdouble)
    all_den = np.zeros(chosen_max, dtype=np.cdouble)
    zeros_mags = torch.sqrt(
        torch.distributions.uniform.Uniform(eps, norm).sample(num_ord // 2)
    )
    poles_mags = torch.sqrt(
        torch.distributions.uniform.Uniform(eps, norm).sample(num_ord // 2)
    )
    if not log:
        zeros_args = torch.distributions.uniform.Uniform(eps, np.pi).sample(
            num_ord // 2
        )
        poles_args = torch.distributions.uniform.Uniform(eps, np.pi).sample(
            num_ord // 2
        )
    else:
        zeros_args = loguniform.rvs(a, b, size=(num_ord) // 2)
        poles_args = loguniform.rvs(a, b, size=(num_ord) // 2)
    for z_mag, z_arg, p_mag, p_arg in zip(
        zeros_mags, zeros_args, poles_mags, poles_args
    ):
        num_poly = [1, -2 * z_mag * np.cos(z_arg), z_mag ** 2]
        den_poly = [1, -2 * p_mag * np.cos(p_arg), p_mag ** 2]
        sos.append(np.hstack((num_poly, den_poly)))
    if chosen_max % 2 == 1:  ##IF ODD, add an extra zero and pole
        zeros_mags = torch.distributions.uniform.Uniform(0.0, 1).sample()
        poles_mags = torch.distributions.uniform.Uniform(0.0, 1).sample()
        num_poly = [1, -1 * zeros_mags, 0]
        den_poly = [1, -1 * poles_mags, 0]
        sos.append(np.hstack((num_poly, den_poly)))

    sos = np.asarray(sos)
    num_sos = sos.shape[0]
    sos_proto = np.tile(np.asarray([1.0, 0, 0, 1.0, 0, 0]), ((max_order + 1) // 2, 1))
    sos_proto[:num_sos, :] = sos
    sos = sos_proto
    my_norms = sos[:, 3]
    sos = sos / my_norms[:, None]  ##sosfreqz requires sos[:,3]=1
    w, h = scipy.signal.sosfreqz(sos, worN=num_points)
    mag = np.abs(h)
    phs = np.unwrap(np.angle(h))
    real = np.real(h)
    imag = np.imag(h)

    mag = compute_dB_magnitude(mag)

    out = mag, phs, real, imag, sos

    return out


def generate_gaussian_peaks(num_points, max_order, min_order=None):

    max_peaks = max_order
    mag = np.zeros(num_points)

    if min_order == None:
        num_peaks = max_order
    else:
        num_peaks = torch.randint(1, max_peaks, [1])

    for n in range(num_peaks):

        # sample parameters
        mu = torch.rand(1) * 20
        mu = -mu if torch.rand(1) > 0.5 else mu

        variance = ((torch.rand(1) * 1) + 0.1).squeeze().numpy()
        sigma = np.sqrt(variance)
        width = (torch.rand(1) * 100).squeeze().numpy()
        gain = torch.rand(1).numpy() * 40
        gain = -gain if torch.rand(1) > 0.5 else gain

        x = np.linspace(-width * sigma, width * sigma, num_points)
        mag += gain * stats.norm.pdf(x, mu, sigma)

    phs = np.zeros(num_points)
    real = np.zeros(num_points)
    imag = np.zeros(num_points)
    sos = np.tile(np.asarray([1.0, 0, 0, 1.0, 0, 0]), ((max_order + 1) // 2, 1))

    mag = compute_dB_magnitude(mag)

    out = mag, phs, real, imag, sos

    return out


def generate_normal_poly_filter(num_points, max_order, min_order=None, eps=1e-8):
    rng = default_rng()
    sos = []
    if min_order == None:
        chosen_ord = max_order
    else:
        chosen_ord = rng.randint(low=min_order, high=max_order)
    num_poly = rng.normal(size=chosen_ord + 1)
    den_poly = rng.normal(size=chosen_ord + 1)
    num_w = np.roots(num_poly)
    den_w = np.roots(den_poly)
    sort_num = np.argsort(-1 * np.abs(np.imag(num_w)))
    sort_den = np.argsort(-1 * np.abs(np.imag(den_w)))
    all_num = num_w[sort_num]
    all_den = den_w[sort_den]

    for ii in range(chosen_ord // 2):
        num_poly = np.real(
            np.polymul([1, -1 * all_num[2 * ii]], [1, -1 * all_num[2 * ii + 1]])
        )
        den_poly = np.real(
            np.polymul([1, -1 * all_den[2 * ii]], [1, -1 * all_den[2 * ii + 1]])
        )
        sos.append(np.hstack((num_poly, den_poly)))
    if chosen_ord % 2 == 1:
        num_poly = np.real(np.polymul([1, 0], [1, -1 * all_num[-1]]))
        den_poly = np.real(np.polymul([1, 0], [1, -1 * all_den[-1]]))
        sos.append(np.hstack((num_poly, den_poly)))

    sos = np.asarray(sos)
    num_sos = sos.shape[0]
    sos_proto = np.tile(np.asarray([1.0, 0, 0, 1.0, 0, 0]), ((chosen_ord + 1) // 2, 1))
    sos_proto[:num_sos, :] = sos
    sos = sos_proto
    my_norms = sos[:, 3]
    sos = sos / my_norms[:, None]  ##sosfreqz requires sos[:,3]=1

    w, h = scipy.signal.sosfreqz(sos, worN=num_points)
    mag = np.abs(h)
    phs = np.unwrap(np.angle(h))
    real = np.real(h)
    imag = np.imag(h)

    mag = compute_dB_magnitude(mag)

    out = mag, phs, real, imag, sos

    return out
