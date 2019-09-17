import math
import random
import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required
import librosa

def lr_scheduling(opt, step, init_lr=1e-3, warmup_steps=4000):
    opt.param_groups[0]['lr'] = init_lr * warmup_steps ** 0.5 * min(step * warmup_steps ** -1.5, step ** -0.5)
    return



def cut_wav(L, wav):
    samples = len(wav)
    if samples < L:
        start = random.randint(0, L - samples)
        # if shorter than desired segment length(3.0s), then let's pad with zero
        wav = np.pad(wav, (start, L - samples - start), 'constant', constant_values=0.0)
    else:
        start = random.randint(0, samples - L)
        wav = wav[start:start+L]

    return wav



def read_wav_np(wavpath):
    os.path.join(self.root_dir, wav_file)
    wav, sr = librosa.load(wavpath)
    
    if len(wav.shape) == 2:
        wav = wav[:, 0]
    
    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2147483648.0
    elif wav.dtype == np.uint8:
        wav = (wav - 128) / 128.0
    
    wav = wav.astype(np.float32)
    return wav
