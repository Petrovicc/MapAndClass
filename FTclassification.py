import mne
import numpy as np
import os
import pandas as pd
import tqdm
from scipy.signal import spectrogram, stft
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import glob
from mne.io import read_raw_edf


def get_band_mask(freqs, band):
    return (freqs >= band[0]) & (freqs <= band[1])


# Band definitions
bands = [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 45)]
band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']

# Prepare storage for all data and labels
all_data = []
all_labels = []

eeg_files = glob.glob(r'eegMentalArithmetic/*_2.edf')
subject_info = pd.read_csv(r'eegMentalArithmetic/subject-info.csv')
for file in tqdm.tqdm(eeg_files, "File processing"):
    # Load EEG file
    eeg = mne.io.read_raw_edf(file, preload=True, verbose=False)

    base_filename = os.path.basename(file).split('.')[0]

    subject_id = base_filename.split('_')[0]
    label = subject_info[subject_info['Subject'] == subject_id]['Count quality'].values[0]

    # Get the EEG data
    data = eeg.get_data()

    # Initialize the 3D array for this file
    file_data = np.zeros((len(band_names), data.shape[0], 251))

    # Calculate STFT for each channel
    for ch_idx in range(data.shape[0]):
        _, freqs, Sxx = stft(data[ch_idx], fs=500, nperseg=500, noverlap=250, detrend=False, return_onesided=True)
        Sxx = np.abs(Sxx) ** 2  # Take the absolute value and square it to get the power
        for band_idx, (band, band_name) in enumerate(zip(bands, band_names)):
            freq_mask = get_band_mask(freqs, band)
            if freq_mask.any():  # Check if any frequencies are within the band
                power = Sxx[:, freq_mask].mean(axis=1)  # Take the mean across frequencies in the band
                file_data[band_idx, ch_idx, :] = power

    all_data.append(file_data)
    all_labels += [label] * file_data.shape[2]

# Convert lists to NumPy arrays for easier manipulation
all_data = np.concatenate(all_data, axis=2)
all_labels = np.array(all_labels)

print(all_data.shape)
print(all_labels.shape)

# Normalize features
scaler = StandardScaler()
all_data = scaler.fit_transform(all_data.reshape(-1, all_data.shape[-1])).reshape(all_data.shape)

# Transpose data to have channels as the second dimension and time as the first
all_data = all_data.transpose(2, 1, 0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.2, random_state=42)
# X_train = X_train.reshape(-1, 21, 5)
# X_test = X_test.reshape(-1, 21, 5)

model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(21, 5)),
    MaxPooling1D(pool_size=2),
    Dropout(0.25),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from pandas import ExcelWriter

# Create an early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Fit the model and get the history
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=64, callbacks=[early_stopping])

# Convert the history to a pandas DataFrame
history_df = pd.DataFrame(history.history)

# Save the history to an Excel file
with ExcelWriter('Results/training_history.xlsx') as writer:
    history_df.to_excel(writer)


model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=32)




# Plotting Accuracy and Loss
history_df = pd.DataFrame(history.history)
history_df[['accuracy', 'val_accuracy']].plot()
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig('Results/Accuracy.png')

plt.figure()
history_df[['loss', 'val_loss']].plot()
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig('Results/Loss.png')

# Confusion Matrix
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype("int32")
conf_mat = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,7))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.savefig('Results/ConfusionMatrix.png')

from tensorflow.keras.utils import plot_model

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.save("mojaMreza.h5")


import netron

netron.start('mojaMreza.h5')