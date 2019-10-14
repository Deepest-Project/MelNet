# based on https://github.com/keithito/tacotron/blob/master/util/audio.py

import librosa
import numpy as np


class MelGen():
    def __init__(self, hp):
        self.hp = hp

    def get_normalized_mel(self, x):
        x = librosa.feature.melspectrogram(
            y=x,
            sr=self.hp.audio.sr,
            n_fft=self.hp.audio.n_fft,
            hop_length=self.hp.audio.hop_length,
            win_length=self.hp.audio.win_length,
            n_mels=self.hp.audio.n_mels
        )
        x = self.pre_spec(x)
        return x

    def pre_spec(self, x):
        return self.normalize(self.amp_to_db(x) - self.hp.audio.ref_level_db)

    def post_spec(self, x):
        return self.db_to_amp(self.denormalize(x) + self.hp.audio.ref_level_db)

    def amp_to_db(self, x):
        return 20.0 * np.log10(np.maximum(x, 1e-6))

    def normalize(self, x):
        return np.clip(x / -self.hp.audio.min_level_db, -1.0, 0.0) + 1.0

    def db_to_amp(self, x):
        return np.power(10.0, 0.05 * x)

    def denormalize(self, x):
        return (np.clip(x, 0.0, 1.0) - 1.0) * -self.hp.audio.min_level_db
