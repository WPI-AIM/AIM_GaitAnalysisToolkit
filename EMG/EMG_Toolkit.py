import numpy as np
import scipy as sp
from scipy import signal


def remove_mean(data):
    return data - np.mean(data)


def rectify(data):
    return abs(data)


def window_rms(a, window_size):
    a2 = np.power(a, 2)
    window = np.ones(window_size) / float(window_size)
    return np.sqrt(np.convolve(a2, window, 'valid'))


def butterworth(data, sfreq=1000.0, high_band=20.0, low_band=450.0):
    """
    time: Time data
    emg: EMG data
    high: high-pass cut off frequency
    low: low-pass cut off frequency
    sfreq: sampling frequency
    """

    # normalise cut-off frequencies to sampling frequency
    high_band = high_band / (sfreq / 2)
    low_band = low_band / (sfreq / 2)

    # create bandpass filter for EMG
    b1, a1 = sp.signal.butter(4, [high_band, low_band], btype='bandpass')

    return sp.signal.filtfilt(b1, a1, data)

def low_pass(data, low_pass=10.0, sfreq=1000.0):
    low_pass = low_pass / sfreq
    b2, a2 = sp.signal.butter(4, low_pass, btype='lowpass')
    return sp.signal.filtfilt(b2, a2, data)