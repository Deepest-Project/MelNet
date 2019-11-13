import torch
import librosa
import numpy as np

from tqdm import tqdm

class Reconstruct():
    def __init__(self, hp):
        self.hp = hp
        self.window = torch.hann_window(window_length=hp.audio.win_length).cuda()
        self.mel_basis = librosa.filters.mel(
            sr=hp.audio.sr,
            n_fft=hp.audio.n_fft,
            n_mels=hp.audio.n_mels
        )
        self.mel_basis = torch.from_numpy(self.mel_basis).cuda() # [n_mels, n_fft//2+1]
        self.criterion = torch.nn.MSELoss()

    def get_mel(self, x):
        stft = torch.stft(
            input=x,
            n_fft=self.hp.audio.n_fft,
            hop_length=self.hp.audio.hop_length,
            win_length=self.hp.audio.win_length,
            window=self.window
        )
        mag = torch.norm(stft, p=2, dim=-1)
        melspectrogram = torch.matmul(self.mel_basis, mag)
        return melspectrogram

    def post_spec(self, x):
        x = (x - 1) * -self.hp.audio.min_level_db + self.hp.audio.ref_level_db
        x = torch.pow(10, x / 10)
        return x
    
    def pre_spec(self, x):
        x = torch.log10(x) * 10
        x = (x - self.hp.audio.ref_level_db) / -self.hp.audio.min_level_db + 1
        return x

    def inverse(self, melspectrogram, iters=1000):
        x = torch.normal(0, 1e-6, size=((melspectrogram.size(1) - 1) * self.hp.audio.hop_length, )).cuda().requires_grad_()
        optimizer = torch.optim.LBFGS([x], tolerance_change=1e-16)
        melspectrogram = self.post_spec(melspectrogram)

        def closure():
            optimizer.zero_grad()
            mel = self.get_mel(x)
            loss = self.criterion(mel, melspectrogram)
            loss.backward()
            return loss

        with tqdm(range(iters)) as pbar:
            for i in pbar:
                optimizer.step(closure=closure)
                pbar.set_postfix(loss=self.criterion(self.get_mel(x), melspectrogram).item())

        return x, self.pre_spec(self.get_mel(x))
