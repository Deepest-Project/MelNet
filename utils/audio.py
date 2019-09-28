# based on https://github.com/keithito/tacotron/blob/master/util/audio.py

import torch
import librosa
import numpy as np


class MelGen():
    def __init__(self, hp):
        self.hp = hp
        self.window = torch.hann_window(window_length=hp.audio.win_length).cuda()
        self.mel_basis = librosa.filters.mel(
            sr=hp.audio.sr, n_fft=hp.audio.n_fft, n_mels=hp.audio.n_mels)
        self.mel_basis = \
            torch.from_numpy(self.mel_basis).cuda().float() # [n_mels, n_fft//2+1]

    def get_magnitude(self, x):
        x = torch.stft(x,
            n_fft=self.hp.audio.n_fft,
            hop_length=self.hp.audio.hop_length,
            win_length=self.hp.audio.win_length,
            window=self.window)
        mag = torch.norm(x, p=2, dim=-1)
        return mag # [B, n_fft//2+1, T]

    def get_mel(self, x):
        mag = self.get_magnitude(x)
        mel = torch.matmul(self.mel_basis, mag)
        return mel # [B, n_mels, T]

    def get_normalized_mel(self, x):
        x = self.get_mel(x)
        x = self.pre_spec(x)
        return x

    def pre_spec(self, x):
        return self.normalize(self.amp_to_db(x) - self.hp.audio.ref_level_db)

    def post_spec(self, x):
        return self.db_to_amp(self.denormalize(x) + self.hp.audio.ref_level_db)

    def amp_to_db(self, x):
        return 20.0 * torch.log10(torch.max(x, torch.tensor(1e-6).cuda()))

    def normalize(self, x):
        return torch.clamp(x / -self.hp.audio.min_level_db, -1.0, 0.0) + 1.0

    def db_to_amp(self, x):
        return torch.pow(10.0, 0.05*x)

    def denormalize(self, x):
        return (torch.clamp(x, 0.0, 1.0) - 1.0) * -self.hp.audio.min_level_db
