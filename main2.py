import cProfile
import pstats

# Run your script with cProfile
profiler = cProfile.Profile()
profiler.enable()
###################################################################################################################

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
import numpy
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

# Import MNE, as well as the MNE sample dataset
import mne
from mne import io
from mne.datasets import sample
from mne.viz import plot_topomap
# from mne.time_frequency import psd_welch

# FOOOF imports
from fooof import FOOOFGroup
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fg
from fooof.plts.spectra import plot_spectrum

from pyedflib import highlevel
from mne.datasets import sample
from mne import read_evokeds

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def check_nans(data, nan_policy='zero'):
    """Check an array for nan values, and replace, based on policy."""

    # Find where there are nan values in the data
    nan_inds = np.where(np.isnan(data))

    # Apply desired nan policy to data
    if nan_policy == 'zero':
        data[nan_inds] = 0
    elif nan_policy == 'mean':
        data[nan_inds] = np.nanmean(data)
    else:
        raise ValueError('Nan policy not understood.')

    return data
def BrainMappingImage(sigs, sig_headers, header, title, save, subject_info, testType):


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

    raw = mne.io.RawArray(sigs,info)

    # spectra, freqs = psd_welch(raw, fmin=0, fmax=45, tmin=0,
    #                            n_overlap=250, n_fft=1000)
    spectrum = raw.compute_psd(method="welch", fmin=0, fmax=45, tmin=0, n_overlap=250, n_fft=1000)
    spectra, freqs = spectrum.get_data(return_freqs=True)
    fg = FOOOFGroup(peak_width_limits=[2, 30], min_peak_height=0.15,
                    peak_threshold=0.1, max_n_peaks=10, verbose=False)


    # Define the frequency range to fit
    freq_range = [1, 45]

    fg.fit(freqs, spectra, freq_range)


    bands = Bands({'delta': [1, 7], # Ovo treba resiti na neki bolji nacin
                    'theta': [3, 7],
                    'alpha': [7, 14],
                    'beta': [14, 30],
                    'gamma': [30, 45],
                    'mu': [8, 13]})

    fig, axes = plt.subplots(2, 3, figsize=(30, 10))

    vecPic = np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2]])
    # fig.tight_layout(pad=0)

    for ind, (label, band_def) in enumerate(bands):
        # Get the power values across channels for the current band
        band_power = check_nans(get_band_peak_fg(fg, band_def)[:, 1], 'mean')

        # Create a topomap for the current oscillation band
        im = mne.viz.plot_topomap(band_power, raw.info, cmap='Spectral_r', contours=2,
                             axes=axes[vecPic[ind][0],vecPic[ind][1]], show=False, ch_type='eeg', extrapolate='head')


        # Set the plot title
        axes[vecPic[ind][0],vecPic[ind][1]].set_title(label , {'fontsize': 20})
        norm = mpl.colors.Normalize(min(band_power), max(band_power))

        cbaxes = fig.add_axes([axes[vecPic[ind][0],vecPic[ind][1]].get_position().x1+0.01, axes[vecPic[ind][0],vecPic[ind][1]].get_position().y0, 0.008, 0.3])
        clb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='Spectral_r'),
                     cax=cbaxes, orientation='vertical')
        clb.ax.set_title('dB')

    subject_info_text = f"ID: {subject_info['ID']}" \
                        f"\nAge: {subject_info['Age']}" \
                        f"\nGender: {'Female' if subject_info['Gender'] == 'F' else 'Male'}" \
                        f"\nRecording Year: {subject_info['Recording Year']}" \
                        f"\nNumber of subtractions: {subject_info['Number of subtractions']}" \
                        f"\nCount Quality: {'Good' if subject_info['Count Quality'] == '1' else 'Bad'}" \
                        f"\nPhase: {'Resting' if testType == '1' else 'Counting'}"

    plt.figtext(0.1, 0.92, subject_info_text, fontsize=12, ha='left', va='top')

    print(title[0])
    if save:
        plt.savefig('BrainMapImage/' + title, orientation='landscape')
        plt.close()




with open('eegMentalArithmetic/subject-info.csv', mode='r') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        subjDict = {row[0]: {"ID":row[0], "Age": row[1],"Gender": row[2],"Recording Year": row[3],"Number of subtractions": row[4]
            , "Count Quality": row[5],} for row in csv_reader}

with open('Podaci/eegCoord.csv', mode='r') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        eegCD = {row[0].upper(): {"X": float(row[1]),"Y": float(row[2])} for row in csv_reader} #EEG coordinate Dictionary

mypath = pathlib.Path().resolve()
subPath = mypath / 'eegMentalArithmetic'
fileNames = [f for f in listdir(subPath) if isfile(join(subPath, f))]
subjectNames = numpy.unique([name[:9] for name in fileNames if name[:7] == 'Subject'])

fs =500
ww = 500
for subjectName in subjectNames:
    print(subjectName)

    for testType in ['1', '2']:
        sigs,sig_headers, header = highlevel.read_edf('eegMentalArithmetic/'+ subjectName + '_' + testType + '.edf')
        sID = subjectName[7:9]
        BrainMappingImage(sigs, sig_headers, header, subjectName + '_' + testType, save=True,
                          subject_info=subjDict[subjectName], testType=testType)

        # powerDict = {sig_headers[i]['label'].upper().strip('EEG '):
        #                  {'Delta': [], 'Theta': [], 'Alpha':[], 'Beta':[], 'Gamma':[], 'Mu':[]}
        #              for i in range(0, len(sig_headers)-2)}



        # clustSig =[sigs[:, i:i + ww] for i in range(0, sigs.shape[1], ww)]

        # for smallSig in clustSig:
        #
        #     N = len(smallSig[0])
        #     T = 1 / fs
        #     x = np.linspace(0.0, N * T, N, endpoint=False)
        #     y = np.delete(smallSig, 20, 0)
        #     yf = fft(y)
        #     xf = fftfreq(N, T)[:N // 5]
        #
        #     smallMat = 2.0/N*np.abs(yf[:, 0:N//5])
        #
        #     for i in range(0, len(sig_headers)-2):
        #         ann = sig_headers[i]['label'].upper().strip('EEG ')
        #
        #         powerDict[ann]['Delta'].append(np.mean(smallMat[i,:5]))
        #         powerDict[ann]['Theta'].append(np.mean(smallMat[i,4:7]))
        #         powerDict[ann]['Alpha'].append(np.mean(smallMat[i,8:15]))
        #         powerDict[ann]['Beta'].append(np.mean(smallMat[i,16:31]))
        #         powerDict[ann]['Gamma'].append(np.mean(smallMat[i,32:]))
        #         powerDict[ann]['Mu'].append(np.mean(smallMat[i,8:12]))

    print(subjectName)


###################################################################################################################
profiler.disable()

# Save and display the profiling results
with open("profile_stats.txt", "w") as stats_file:
    stats = pstats.Stats(profiler, stream=stats_file)
    stats.strip_dirs().sort_stats("cumulative").print_stats(20)
