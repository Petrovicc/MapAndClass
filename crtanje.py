# General imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm, colors, colorbar

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


sigs, sig_headers, header = highlevel.read_edf('eegMentalArithmetic/Subject01_1.edf')


def BrainMappingImage(sigs, sig_headers, header, title):
    len(sig_headers)
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




    spectra, freqs = psd_welch(raw, fmin=0, fmax=45, tmin=0,
                               n_overlap=250, n_fft=1000)

    fg = FOOOFGroup(peak_width_limits=[2, 30], min_peak_height=0.15,
                    peak_threshold=0.1, max_n_peaks=10, verbose=True)


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
    print(title[0])
    plt.savefig('BrainMapImage/' + title, orientation='landscape')

    plt.close()



