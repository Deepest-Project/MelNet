import os
import glob
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils.utils import *
from utils.audio import MelGen
from utils.tierutil import TierUtil
# from text import text_to_sequence



def create_dataloader(hp, args, train):
    if train:
        return DataLoader(dataset=AudioOnlyDataset(hp, args, True),
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=hp.train.num_workers,
                          pin_memory=True,
                          drop_last=True)
                        #   collate_fn=TextCollate())
    else:
        return DataLoader(dataset=AudioOnlyDataset(hp, args, False),
                          batch_size=args.batch_size,
                          shuffle=False,
                          num_workers=1,
                          pin_memory=True,
                          drop_last=True)
                        #   collate_fn=TextCollate())



class AudioOnlyDataset(Dataset):
    def __init__(self, hp, args, train):
        self.hp = hp
        self.args = args
        self.train = train
        self.data = hp.data.path
        self.melgen = MelGen(hp)
        self.tierutil = TierUtil(hp)

        # this will search all files within hp.data.path
        self.file_list = []
        # for i, f in enumerate(glob.glob(os.path.join(hp.data.path, '**', hp.data.extension), recursive=True)):
        #     wav = read_wav_np(f)
        #     duraton = (len(wav)/hp.audio.sr)
        #     if duraton < hp.audio.duration:
        #         self.file_list.append(f)
        self.file_list = glob.glob(os.path.join(hp.data.path, '**', hp.data.extension), recursive=True)
        
        random.seed(123)
        random.shuffle(self.file_list)
        if train:
            self.file_list = self.file_list[:int(0.95 * len(self.file_list))]
        else:
            self.file_list = self.file_list[int(0.95 * len(self.file_list)):]

        self.wavlen = int(hp.audio.sr * hp.audio.duration)
        self.tier = self.args.tier

        self.melgen = MelGen(hp)
        self.tierutil = TierUtil(hp)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        wav = read_wav_np(self.file_list[idx])
        wav = cut_wav(self.wavlen, wav)
        mel = self.melgen.get_normalized_mel(wav)
        source, target = self.tierutil.cut_divide_tiers(mel, self.tier)

        return source, target



class AudioTextDataset(Dataset):
    def __init__(self, hp, args, train):
        self.hp = hp
        self.args = args
        self.train = train
        self.data = hp.data.path
        self.melgen = MelGen(hp)
        self.tierutil = TierUtil(hp)

        # this will search all files within hp.data.path
        self.root_dir = hp.data.path
        self.dataset = []
        with open(os.path.join(self.root_dir, 'transcript.v.1.2.txt'), 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                wav_name, _, _, text, _ = line.split('|')
                wav_name = wav_name[2:-4] + '.wav'
                
                wav_path = os.path.join(self.root_dir, 'wavs', wav_name)
                wav = read_wav_np(wav_path)
                duraton = (len(wav)/hp.audio.sr)
                if duraton < hp.audio.duration:
                    self.dataset.append((wav_path, text))
                
                #if len(self.dataset) > 100: break

        
        random.seed(123)
        random.shuffle(self.dataset)
        if train:
            self.dataset = self.dataset[:int(0.95 * len(self.dataset))]
        else:
            self.dataset = self.dataset[int(0.95 * len(self.dataset)):]

        self.wavlen = int(hp.audio.sr * hp.audio.duration)
        self.tier = self.args.tier

        self.melgen = MelGen(hp)
        self.tierutil = TierUtil(hp)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx][1]
        seq = text_to_sequence(text)
        
        wav = read_wav_np(self.dataset[idx][0])
        wav = cut_wav(self.wavlen, wav)
        mel = self.melgen.get_normalized_mel(wav)
        source, target = self.tierutil.cut_divide_tiers(mel, self.tier)

        return seq, source, target



class TextCollate():
    def __init__(self):
        return

    def __call__(self, batch):
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True)
        max_input_len = input_lengths[0]

        seq_padded = torch.zeros(len(batch), max_input_len, dtype=torch.long)
        for i in range(len(ids_sorted_decreasing)):
            seq = batch[ids_sorted_decreasing[i]][0]
            seq_padded[i, :len(seq)] = torch.from_numpy(seq).long()

        source_padded = torch.stack( [ torch.from_numpy(x[1]) for x in batch] )
        target_padded = torch.stack( [ torch.from_numpy(x[2]) for x in batch] )

        ### MASKING ###
        equal_check = target_padded - target_padded[:, 0:1]
        output_lengths = torch.sum(torch.any(equal_check!=0, dim=1), dim=-1).long()
        
        idx = torch.arange(1, target_padded.size(-1)+1).long() 
        mask = (output_lengths.unsqueeze(-1) < idx.unsqueeze(0)).to(torch.bool) # B, T
        source_padded.masked_fill_(mask.unsqueeze(1), 0)
        target_padded.masked_fill_(mask.unsqueeze(1), 0)
        
        return seq_padded, input_lengths, source_padded, target_padded, output_lengths





