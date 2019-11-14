import os
import glob
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils.utils import read_wav_np, cut_wav, get_length, process_blizzard
from utils.audio import MelGen
from utils.tierutil import TierUtil
from text import text_to_sequence

def create_dataloader(hp, args, train):
    if args.tts:
        dataset = AudioTextDataset(hp, args, train)
        return DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=train,
            num_workers=hp.train.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=TextCollate()
        )
    else:
        dataset = AudioOnlyDataset(hp, args, train)
        return DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=train,
            num_workers=hp.train.num_workers,
            pin_memory=True,
            drop_last=True
        )

class AudioOnlyDataset(Dataset):
    def __init__(self, hp, args, train):
        self.hp = hp
        self.args = args
        self.train = train
        self.data = hp.data.path
        self.melgen = MelGen(hp)
        self.tierutil = TierUtil(hp)

        # this will search all files within hp.data.path
        self.file_list = glob.glob(
            os.path.join(hp.data.path, '**', hp.data.extension),
            recursive=True
        )

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
        wav = read_wav_np(self.file_list[idx], sample_rate=self.hp.audio.sr)
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
        if hp.data.name == 'KSS':
            with open(os.path.join(self.root_dir, 'transcript.v.1.3.txt'), 'r') as f:
                lines = f.read().splitlines()
                for line in lines:
                    wav_name, _, _, text, length, _ = line.split('|')

                    wav_path = os.path.join(self.root_dir, 'kss', wav_name)
                    duraton = float(length)
                    if duraton < hp.audio.duration:
                        self.dataset.append((wav_path, text))

                # if len(self.dataset) > 100: break
        elif hp.data.name == 'Blizzard':
            with open(os.path.join(self.root_dir, 'prompts.gui'), 'r') as f:
                lines = f.read().splitlines()
                filenames = lines[::3]
                sentences = lines[1::3]
                for filename, sentence in zip(filenames, sentences):
                    wav_path = os.path.join(self.root_dir, 'wavn', filename + '.wav')
                    length = get_length(wav_path, hp.audio.sr)
                    if length < hp.audio.duration:
                        self.dataset.append((wav_path, sentence))
        else:
            raise NotImplementedError

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
        if self.hp.data.name == 'KSS':
            seq = text_to_sequence(text)
        elif self.hp.data.name == 'Blizzard':
            seq = process_blizzard(text)

        wav = read_wav_np(self.dataset[idx][0], sample_rate=self.hp.audio.sr)
        wav = cut_wav(self.wavlen, wav)
        mel = self.melgen.get_normalized_mel(wav)
        source, target = self.tierutil.cut_divide_tiers(mel, self.tier)

        return seq, source, target

class TextCollate():
    def __init__(self):
        return

    def __call__(self, batch):
        seq = [torch.from_numpy(x[0]).long() for x in batch]
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths = torch.LongTensor([x.shape[0] for x in seq])

        seq_padded = torch.nn.utils.rnn.pad_sequence(seq, batch_first=True)
        source_padded = torch.stack([torch.from_numpy(x[1]) for x in batch])
        target_padded = torch.stack([torch.from_numpy(x[2]) for x in batch])

        return seq_padded, input_lengths, source_padded, target_padded
