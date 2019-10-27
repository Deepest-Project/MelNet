import random
import numpy as np
import subprocess
import audiosegment

def get_commit_hash():
    message = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    return message.strip().decode('utf-8')

def read_wav_np(wavpath):
    file_format = wavpath.split('.')[-1]
    audio = audiosegment.from_file(wavpath).resample(sample_rate_Hz=22050)
    data = audio.raw_data
    wav = np.frombuffer(data, dtype=np.int16)
    
    if len(wav.shape) == 2:
        wav = wav[:, 0]
    
    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2147483648.0
    elif wav.dtype == np.uint8:
        wav = (wav - 128) / 128.0
    
    wav = wav.astype(np.float32)
    # wav= norm_wav(wav)
    # wav = trim_wav(wav)
    return wav


def cut_wav(L, wav):
    samples = len(wav)
    if samples < L:
        wav = np.pad(wav, (0, L - samples), 'constant', constant_values=0.0)
    else:
        start = random.randint(0, samples - L)
        wav = wav[start:start + L]

    return wav


def norm_wav(wav):
    assert isinstance(wav, np.ndarray) and len(wav.shape)==1, 'Wav file should be 1D numpy array'
    return wav / np.max( np.abs(wav) )


def trim_wav(wav, threshold=0.01):
    assert isinstance(wav, np.ndarray) and len(wav.shape)==1, 'Wav file should be 1D numpy array'
    cut = np.where((abs(wav)>threshold))[0]
    wav = wav[cut[0]:(cut[-1]+1)]
    return wav