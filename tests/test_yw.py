import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

from iirnet.yulewalk import yulewalk

if __name__ == '__main__':

    N = 8
    f = np.array([0, 0.6, 0.6, 1])
    m = np.array([1,   1,   0, 0])

    b, a = yulewalk(N, f, m)

    w, h = scipy.signal.freqz(b, a, 128)
    plt.plot(w / np.pi, 20 * np.log10(np.abs(h)))
    plt.grid()
    plt.savefig('yw.png')