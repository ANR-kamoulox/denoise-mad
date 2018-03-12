import numpy as np
import scipy.signal as signal


def smooth(s, lengthscale, peaks_suppressor=False):
    """smoothes s vertically"""
    if peaks_suppressor:
        nChans = s.shape[1]
        lengthscale = int(2 * round(float(lengthscale) / 2))
        W = np.hamming(min(lengthscale, s.shape[0]))
        W /= np.sum(W)
        res = np.zeros(s.shape)
        for col in range(nChans):
            res[:, col] = signal.fftconvolve(s[:, col], W, mode='same')
        return res

    else:
        nChans = s.shape[1]
        lengthscale = int(2 * round(float(lengthscale) / 2))
        W = np.hamming(min(lengthscale, s.shape[0]))
        W /= np.sum(W)
        res = np.zeros(s.shape)
        for col in range(nChans):
            res[:, col] = signal.fftconvolve(s[:, col], W, mode='same')
        return res
