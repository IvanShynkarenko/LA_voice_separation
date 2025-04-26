'''
    Drawer module
'''

import numpy as np
import librosa
import matplotlib.pyplot as plt

def plot(reconstructed_sounds, sr):
    '''
        Plots in wave form format
    '''
    n = len(reconstructed_sounds)
    colors = ['b', 'y', 'g']
    _, ax = plt.subplots(nrows=n, ncols=1, sharex=True, figsize=(11, 3*n))
    for i in range(n):
        librosa.display.waveshow(reconstructed_sounds[i], sr=sr,
            color=colors[i], ax=ax[i], label=f'Source {i}', axis='time')
        ax[i].set(xlabel='Time [s]')
        ax[i].legend()

def mean_squared_error(audio1, audio2):
    '''
    Compute the Mean Squared Error (MSE) between two audio signals.
    
    Parameters:
    - audio1: numpy.ndarray, the first audio signal
    - audio2: numpy.ndarray, the second audio signal
    
    Returns:
    - mse: float, the mean squared error between the two signals
    '''
    min_length = min(len(audio1), len(audio2))
    audio1 = audio1[:min_length]
    audio2 = audio2[:min_length]
    
    mse = np.mean((audio1 - audio2) ** 2)
    
    return mse

def plot_spectogram(y, sr, FRAME1 = 1024, HOP = 512):
    '''
        Spectogram func
    '''
    D = librosa.stft(y, n_fft=FRAME1, hop_length=HOP)
    magnitude = np.abs(D)
    librosa.display.specshow(librosa.amplitude_to_db(magnitude), 
            sr=sr, n_fft=FRAME1, hop_length=HOP)

def RMSE(y1, y2):
    trim = min(len(y1), len(y2))

    y1 = y1[:trim]
    y2 = y2[:trim]
    rmse = np.sqrt(np.mean((y1 - y2) ** 2))
    return rmse

def spectral_difference(audio1, audio2, n_fft=2048, hop_length=512):
    '''
    Calculate the spectral difference between two audio signals.

    Parameters:
    - audio1 (np.ndarray): First audio signal.
    - audio2 (np.ndarray): Second audio signal.
    - sr (int): Sampling rate of the audio signals (default: 22050).
    - n_fft (int): Number of samples per frame for the STFT (default: 2048).
    - hop_length (int): Number of samples between frames for the STFT (default: 512).

    Returns:
    - float: Spectral difference between the two audio signals.
    '''
    trim = min(len(audio1), len(audio2))

    audio1 = audio1[:trim]
    audio2 = audio2[:trim]

    stft1 = librosa.stft(audio1, n_fft=n_fft, hop_length=hop_length)
    stft2 = librosa.stft(audio2, n_fft=n_fft, hop_length=hop_length)

    mag1 = np.abs(stft1)
    mag2 = np.abs(stft2)

    diff = np.mean(np.abs(mag1 - mag2))

    return diff