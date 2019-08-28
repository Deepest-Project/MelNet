import numpy as np
import subprocess
import torch.nn.functional as F
from scipy.io.wavfile import read


def get_commit_hash():
	message = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
	return message.strip().decode('utf-8')

def read_wav_np(wavpath):
    sr, wav = read(wavpath)
    
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
