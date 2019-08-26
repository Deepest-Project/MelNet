import torch
import librosa
import numpy as np


class MelGen():
    def __init__(self, hp):
        self.hp = hp
        self.window = torch.hann_window(window_length=hp.audio.win_length).cuda()
        self.mel_basis = librosa.filters.mel(
            sr=hp.audio.sr, n_fft=hp.audio.n_fft, n_mels=hp.audio.n_mels)
        self.mel_basis = # [n_mels, n_fft//2+1]
            torch.from_numpy(self.mel_basis).cuda()

    def get_magnitude(self, x):
        x = torch.stft(x,
            n_fft=self.hp.audio.n_fft,
            hop_length=self.hp.audio.hop_length,
            win_length=self.hp.audio.win_length,
            window=self.window)
        mag = torch.norm(x, p=2, dim=-1)
        return mag # [B, n_fft//2+1, T]

    def get_logmel(self, x):
        mag = self.get_magnitude(x)
        mel = torch.log10(torch.matmul(self.mel_basis, mag) + 1e-6)
        return mel # [B, n_mels, T]
