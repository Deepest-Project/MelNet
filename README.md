# MelNet

Implementation of [MelNet: A Generative Model for Audio in the Frequency Domain](<https://arxiv.org/abs/1906.01083>)

## Prerequisites

- Tested with Python 3.6.8 & 3.7.4, PyTorch 1.2.0 & 1.3.0.
- `pip install -r requirements.txt`

## How to train

### Datasets

- Blizzard, VoxCeleb2, and KSS have YAML files provided under `config/`. For other datasets, fill out your own YAML file according to the other provided ones.
- Unconditional training is possible for all kinds of datasets, provided that they have a consistent file extension specified by `data.extension` within the YAML file.
- Conditional training is currently only implemented for KSS and a subset of the Blizzard dataset.

### Running the code

- `python trainer.py -c [config YAML file path] -n [name of run] -t [tier number] -b [batch size] -s [TTS]`
  - Each tier can be trained separately. Since each tier is larger than the one before it (with the exception of tier 1), modify the batch size for each tier.
    - Tier 6 of the Blizzard dataset does not fit on a 16GB P100, even with a batch size of 1.
  - The `-s` flag is a boolean for determining whether to train a TTS tier. Since a TTS tier only differs at tier 1, this flag is ignored when `[tier number] != 0` . Warning: this flag is toggled `True` no matter what follows the flag. Ignore it if you're not planning to use it.

## How to sample

### Preparing the checkpoints

- The checkpoints must be stored under `chkpt/`.
- A YAML file named `inference.yaml` must be provided under `config/`.
- `inference.yaml` must specify the number of tiers, the names of the checkpoints, and whether or not it is a conditional generation.

### Running the code

- `python inference.py -c [config YAML file path] -p [inference YAML file path] -t [timestep of generated mel spectrogram] -n [name of sample] -i [input sentence for conditional generation]`
  - Timestep refers to the length of the mel spectrogram. The ratio of timestep to seconds is roughly `[sample rate] : [hop length of FFT]`.
  - The `-i` flag is optional, only needed for conditional generation. Surround the sentence with `""` and end with `.`.
  - Both unconditional generation and conditional generation currently does not support primed generation (extrapolating from provided data).

## To-do

- [x] Implement upsampling procedure
- [x] GMM sampling + loss function
- [x] Unconditional audio generation
- [x] TTS synthesis 
- [x] Tensorboard logging
- [x] Multi-GPU training
- [ ] Primed generation

## Implementation authors

- [Seungwon Park](<https://github.com/seungwonpark>), [June Young Yi](<https://github.com/Rick-McCoy>), [Yoonhyung Lee](<https://github.com/LEEYOONHYUNG>), [Joowhan Song](<https://github.com/Joovvhan>) @ Deepest Season 6

## License

MIT License
