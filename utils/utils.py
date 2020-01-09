import random
import numpy as np
import subprocess
import audiosegment

PAD = '_'
EOS = '~'
PUNC = '!\'(),-.:;?`'
SPACE = ' '
SYMBOLS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
en_symbols = SYMBOLS + PAD + EOS + PUNC + SPACE
_symbol_to_id = {s: i for i, s in enumerate(en_symbols)}

def get_length(wavpath, sample_rate):
    audio = audiosegment.from_file(wavpath).resample(sample_rate_Hz=sample_rate)
    return audio.duration_seconds

def process_blizzard(text: str):
    text = text.replace('@ ', '').replace('# ', '').replace('| ', '') + EOS
    seq = [_symbol_to_id[c] for c in text]
    return np.array(seq, dtype=np.int32)

def get_commit_hash():
    message = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    return message.strip().decode('utf-8')

def read_wav_np(wavpath, sample_rate):
    audio = audiosegment.from_file(wavpath).resample(sample_rate_Hz=sample_rate)
    wav = audio.to_numpy_array()
    
    if len(wav.shape) == 2:
        wav = wav.T.flatten()
    
    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2147483648.0
    elif wav.dtype == np.uint8:
        wav = (wav - 128) / 128.0
    
    wav = wav.astype(np.float32)
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