import timeit
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import pickle
import os.path
# import mkl
from matplotlib import pyplot as plt
import pathlib

from pyedflib import highlevel
import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq
from os import listdir
from os.path import isfile, join
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm, colors, colorbar
import matplotlib.animation as animation

# Import MNE, as well as the MNE sample dataset
import mne
from mne import io
from mne.datasets import sample
from mne.viz import plot_topomap
from mne.time_frequency import psd_welch


# FOOOF imports
from fooof import FOOOFGroup
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fg
from fooof.plts.spectra import plot_spectrum

from pyedflib import highlevel
from mne.datasets import sample
from mne import read_evokeds
from joblib import Parallel, delayed

import multiprocessing as mp

if __name__ == '__main__':
    print("Number of processors: ", mp.cpu_count())

    ################################################################################################################################
    ww = 1000
    step = 1000
    Fs = 500
    ################################################################################################################################
    sigs, sig_headers, header = highlevel.read_edf('eegMentalArithmetic/Subject00_2.edf')
    # sigs = sigs[:,:sigs.shape[1]//20] ######################################################## Skracivanje snimka radi brze provere, obrisati kasnije
    n_channels = len(sig_headers)
    ch_names = [None]*n_channels
    for i in range(0,n_channels):
        if sig_headers[i]['label'][-2:] == 'p1' or sig_headers[i]['label'][-2:] == 'p2':
            ch_names[i] = sig_headers[i]['label'][-2:].upper()
        else:
            ch_names[i] = sig_headers[i]['label'][-2:]

    ch_types = ['eeg']*20 + ['ecg']
    sampling_freq = sig_headers[0]['sample_rate']

    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
    info.set_montage('standard_1020')

    rawAll = mne.io.RawArray(sigs, info, verbose=False)

    fg = FOOOFGroup(peak_width_limits=[2, 30], min_peak_height=0.15,
                        peak_threshold=0.1, max_n_peaks=10, verbose=False)
    spectra, freqs = psd_welch(rawAll, fmin=0, fmax=45, tmin=0,
                                   n_overlap=250, n_fft=1000, verbose=False)



    # Define the frequency range to fit
    freq_range = [1, 45]

    # fg.fit(freqs, spectra, freq_range)


    bands = Bands({'delta': [1, 7], # Ovo treba resiti na neki bolji nacin
                        'theta': [3, 7],
                        'alpha': [7, 14],
                        'beta': [14, 30],
                        'gamma': [30, 45],
                        'mu': [8, 13]})


    clustSig = [sigs[:, i:i + ww] for i in range(0, sigs.shape[1], step)]
    # clustFg = [None]*len(clustSig)


    def fooofGroupFunc(ind, clustSig, clustFg ):
        raw_tmp = mne.io.RawArray(clustSig[ind], info, verbose=False)
        spectra, freqs = psd_welch(raw_tmp, fmin=0, fmax=45, tmin=0, n_overlap=250, n_fft=1000, verbose=False)
        clustFg[ind] = FOOOFGroup(peak_width_limits=[2, 30], min_peak_height=0.15,peak_threshold=0.1, max_n_peaks=10, verbose=False)
        return clustFg[ind].fit(freqs, spectra, freq_range)
        print(ind)

    def fgfParallel(tmpSig):
        # raw_tmp = mne.io.Raw(tmpSig, info, verbose=False)
        # spectra, freqs = psd_welch(raw_tmp, fmin=0, fmax=45, tmin=0, n_overlap=250, n_fft=1000, verbose=False)
        # tmpFg = FOOOFGroup(peak_width_limits=[2, 30], min_peak_height=0.15, peak_threshold=0.1, max_n_peaks=10,
        #                           verbose=False)
        # tmpFg.fit(freqs, spectra, freq_range)
        return tmpSig

    startTime = timeit.default_timer()

    # Parallel(n_jobs=-1, backend='')(delayed(fooofGroupFunc)(ind=i, clustSig=clustSig, clustFg=clustFg) for i in range(len(clustFg)-1))

    pool = mp.Pool(mp.cpu_count())

    clustFg = [pool.apply(fgfParallel, args=(tmpSig)) for tmpSig in range(0,10)]

    pool.close()
    # for i in range(len(clustFg)):
    #     fooofGroupFunc(i, clustSig, clustFg)

    endTime = timeit.default_timer()
    # print('Whole time: ' + str(round(endTime - startTime, 2)))
    meanFgTime = round((endTime - startTime)/len(clustFg), 2)
    print('Mean time: ' + str(meanFgTime))
