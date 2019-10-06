import os
import glob
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils.utils import read_wav_np, cut_wav
from utils.audio import MelGen
from utils.tierutil import TierUtil

def create_dataloader(hp, args, train):
    if train:
        return DataLoader(dataset=AudioOnlyDataset(hp, args, True),
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=hp.train.num_workers,
                          pin_memory=True,
                          drop_last=True)
    else:
        return DataLoader(dataset=AudioOnlyDataset(hp, args, False),
                          batch_size=args.batch_size,
                          shuffle=False,
                          num_workers=1,
                          pin_memory=True,
                          drop_last=True)


class AudioOnlyDataset(Dataset):
    def __init__(self, hp, args, train):
        self.hp = hp
        self.args = args
        self.train = train
        self.data = hp.data.path
        self.melgen = MelGen(hp)
        self.tierutil = TierUtil(hp)

        self.wav_list = glob.glob(os.path.join(self.data, '**', '*.wav'), recursive=True)
        random.seed(123)
        random.shuffle(self.wav_list)
        if train:
            self.wav_list = self.wav_list[:int(0.95*len(self.wav_list))]
        else:
            self.wav_list = self.wav_list[int(0.95*len(self.wav_list)):]

        self.wavlen = int(hp.audio.sr * hp.audio.duration)
        self.tier = 0

        self.melgen = MelGen(hp)
        self.tierutil = TierUtil(hp)

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        wav = read_wav_np(self.wav_list[idx])
        wav = cut_wav(self.wavlen, wav).cuda()
        mel = self.melgen.get_logmel(wav)
        source, target = self.tierutil.cut_divide_tiers(mel, self.tier)

        return source, target


class AudioTextDataset(Dataset):
    def __init__(self, hp, args, train):
        self.hp = hp
        self.args = args
        self.train = train

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError
