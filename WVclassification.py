import glob
import pywt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mne.io import read_raw_edf
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense


# load subject information
subject_info = pd.read_csv('eegMentalArithmetic/subject-info.csv')


# a function to load EEG data from an EDF file and perform wavelet transform
# 500 Hz sample rate
def load_and_transform_eeg(file_path, wavelet_name='db2', level=5):
    raw = read_raw_edf(file_path, preload=True, verbose=False)
    data = raw.get_data()
    transformed_data = []
    for channel in data:
        coeffs = pywt.wavedec(channel, wavelet_name, level=level)

        # Find the maximum length among the wavelet coefficients
        max_len = max([len(coeff) for coeff in coeffs])

        # Pad each coefficient array to the maximum length
        padded_coeffs = [np.pad(coeff, (0, max_len - len(coeff))) for coeff in coeffs]

        # Stack the padded coefficients
        stacked_coeffs = np.stack(padded_coeffs, axis=-1)

        transformed_data.append(stacked_coeffs)

    return np.array(transformed_data)

def segment_eeg(eeg_data, segment_length):
    """Segments the EEG data along the time axis."""
    num_channels = eeg_data.shape[0]
    num_samples = eeg_data.shape[1]
    num_coeffs = eeg_data.shape[2]

    segments = []
    for start in tqdm(range(0, num_samples - segment_length + 1), desc='Segmenting data', unit='segment'):
        end = start + segment_length
        segment = eeg_data[:, start:end, :, :]
        segments.append(segment)

    return np.array(segments)



# load and transform all EEG data
eeg_data = []

eeg_files = glob.glob('eegMentalArithmetic/*_2.edf')
for file in eeg_files:  # you need to define eeg_files
    transformed_eeg = load_and_transform_eeg(file)
    eeg_data.append(transformed_eeg)

#############################################################################################################
import pandas as pd
from sklearn.model_selection import train_test_split

# Load subject info
subject_info = pd.read_csv('eegMentalArithmetic/subject-info.csv')

# Get the 'Count quality' column as our labels
labels = subject_info['Count quality'].values


# Define the length of the segments in number of samples
segment_length = 500  # adjust this to your preferred segment length

# Segment the EEG data
eeg_data = np.stack(eeg_data, axis=-1)
# segmented_data = segment_eeg(eeg_data, segment_length)
# segmented_data = np.concatenate(segmented_data)
#

def data_generator():
    for i in range(eeg_data.shape[1] - segment_length + 1): # sliding window
        for j in range(eeg_data.shape[3]): # iterating over subjects
            segment = eeg_data[:, i:i+segment_length, :, j]  # get the segment
            label = labels[j]  # get the corresponding label
            yield segment, label

segment_length = 500  # your segment length
batch_size = 32  # adjust based on your memory capacity

# Create a tf.data.Dataset
dataset = tf.data.Dataset.from_generator(data_generator,
                                         output_signature=(tf.TensorSpec(shape=(21, segment_length, 6), dtype=tf.float32),
                                                           tf.TensorSpec(shape=(), dtype=tf.int32)))
dataset = dataset.batch(batch_size)


def train_generator():
    # use 70% of the data for training
    for i in range(int((eeg_data.shape[1] - segment_length + 1) * 0.7)):
        for j in range(eeg_data.shape[3]):  # iterating over subjects
            segment = eeg_data[:, i:i+segment_length, :, j]  # get the segment
            label = labels[j]  # get the corresponding label
            yield segment[..., np.newaxis], label  # add a new dimension to the segment for the channels

def val_generator():
    # use 30% of the data for validation
    for i in range(int((eeg_data.shape[1] - segment_length + 1) * 0.7), eeg_data.shape[1] - segment_length + 1):
        for j in range(eeg_data.shape[3]):  # iterating over subjects
            segment = eeg_data[:, i:i+segment_length, :, j]  # get the segment
            label = labels[j]  # get the corresponding label
            yield segment[..., np.newaxis], label  # add a new dimension to the segment for the channels

# Create tf.data.Datasets for training and validation
train_dataset = tf.data.Dataset.from_generator(train_generator,
                                               output_signature=(tf.TensorSpec(shape=(21, segment_length, 6, 1), dtype=tf.float32),
                                                                 tf.TensorSpec(shape=(), dtype=tf.int32)))
val_dataset = tf.data.Dataset.from_generator(val_generator,
                                             output_signature=(tf.TensorSpec(shape=(21, segment_length, 6, 1), dtype=tf.float32),
                                                               tf.TensorSpec(shape=(), dtype=tf.int32)))
train_dataset = train_dataset.batch(batch_size)
val_dataset = val_dataset.batch(batch_size)



# define the model
model = Sequential()
model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', input_shape=(21, segment_length, 6, 1)))
model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'))
model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
