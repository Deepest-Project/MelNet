import numpy as np
import scipy, librosa

def normalize(wav):
    return wav / np.max( np.abs(wav) )

def trim(wav, threshold=0.01):
    cut = np.where((abs(wav)>threshold))[0]
    wav = wav[cut[0]:(cut[-1]+1)] # Trimming
    return wav

def wav_to_spec(wav, sr, n_fft, n_hop):
    _, _, Zxx = scipy.signal.stft(wav, fs=sr, nperseg=n_fft, noverlap=n_fft-n_hop)
    return 20 * np.log10(np.maximum(np.abs(Zxx), 1e-8))

def wav_to_melspec(wav, sr, n_fft, n_hop, n_mels, mel_basis=None):
    _, _, Zxx = scipy.signal.stft(wav, fs=sr, nperseg=n_fft, noverlap=n_fft-n_hop)
    if mel_basis is None:
        mel_basis = librosa.filters.mel(sr, n_fft, n_mels)
    D = np.matmul(mel_basis, np.abs(Zxx))
    return 20 * np.log10(np.maximum(D, 1e-8))

def spec_to_mel(spectrogram, sr, n_fft, n_mels, mel_basis=None):
    S = np.power(10.0, spectrogram * 0.05)
    if mel_basis is None:
        mel_basis = librosa.filters.mel(sr, n_fft, n_mels)
    D = np.matmul(mel_basis, S)
    return 20 * np.log10(np.maximum(D, 1e-8))