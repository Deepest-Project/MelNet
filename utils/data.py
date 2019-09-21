from torch.utils.data import Dataset, DataLoader
import scipy, librosa
import os
import torch
import random
from utils import hparams as hp
from .audio_processing import *


class Spliter():
    def __init__(self):
        self.split=0
    
    def reset(self):
        self.split=0
        
    def __call__(self, melspec):
        if self.split%2==0:
            self.split += 1
            return melspec[0::2, :], melspec[1::2, :]
        
        elif self.split%2==1:
            self.split += 1
            return melspec[:, 0::2], melspec[:, 1::2]


def pad_mel(melspecs):
    B, F, T = len(melspecs), melspecs[0].shape[0], max([x.shape[1] for x in melspecs])
    padded_mel =  np.zeros((B, F, T))
    for i, mel in enumerate(melspecs):
        padded_mel[i, :, :mel.shape[1]] = mel

    return torch.from_numpy(padded_mel).to(torch.float)


class MelData(Dataset):
    def __init__(self, hp):
        super(Dataset, self).__init__()
        self.root_dir = hp.root_dir
        self.n_tiers = hp.n_tiers
        self.sr = hp.sr
        self.n_fft = hp.n_fft
        self.n_mels = hp.n_mels
        self.n_hop = hp.n_hop
        self.n_overlap = hp.n_fft - hp.n_hop
        self.n_bucket = hp.n_bucket
        
        self.wav_files = list(filter(lambda f: f.endswith('.wav'), os.listdir(self.root_dir)))
        self.split = Spliter()
        self.mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)
        #self.bucket_by_sequence_length()
        
    def bucket_by_sequence_length(self):
        ##### Wav -> Melspectrogram #####
        self.wav_lengths = []
        for wav_file in self.wav_files:
            wav, _ = librosa.load( os.path.join(self.root_dir, wav_file) )
            self.wav_lengths.append(len(wav))
            
        self.wav_length = list(zip(self.wav_files, self.wav_lengths))
        self.wav_length.sort(key = lambda x: x[1])
        
        self.buckets = {}
        bucket_size = len(self.wav_length)//self.n_bucket
        for i in range(self.n_bucket):
            self.buckets[i] = self.wav_length[i*bucket_size : (i+1)*bucket_size]
            
    def shuffle(self):
        for i in range(self.n_bucket):
            random.shuffle(self.buckets[i])
        
    def __getitem__(self, i):
        ##### Wav -> Melspectrogram #####
        wav, sr = librosa.load( os.path.join(self.root_dir, self.wav_files[i]) )
        wav = normalize(wav)
        wav = trim(wav)
        melspec = wav_to_melspec(wav, self.sr, self.n_fft, self.n_hop, self.n_mels, self.mel_basis)
        
        ##### Melspectrogram Validation #####
        n_half_t = (self.n_tiers-1) // 2
        n_time = melspec.shape[1] - melspec.shape[1] % 2**n_half_t
        melspec = melspec[:, :n_time]
        
        ##### Build mel_tiers #####
        mel_tiers = [None] + [ 0 for _ in range(self.n_tiers) ] + [melspec]
        
        for t in range(self.n_tiers, 1, -1):
            tier, melspec = self.split(melspec)
            mel_tiers[t] = tier
        mel_tiers[1] = melspec
        self.split.reset()
        
        return mel_tiers
    
    def __len__(self):
        return len(self.wav_files)
    
    

class MelCollate():
    def __init__(self, hp):
        self.n_tiers = hp.n_tiers
        
    def __call__(self, batch):
        mel_tiers = [None] + [ [] for _ in range(self.n_tiers+1) ]
        
        for data in batch:
            for t in range(1, self.n_tiers+2):
                mel_tiers[t].append(data[t])
        
        for t in range(1, self.n_tiers+2):
            mel_tiers[t] = pad_mel(mel_tiers[t])
        
        return mel_tiers