import os
import glob
import argparse
import torch
import audiosegment
import matplotlib.pyplot as plt
import numpy as np

from utils.plotting import plot_spectrogram_to_numpy
from utils.reconstruct import Reconstruct
from utils.constant import t_div
from utils.hparams import HParam
from model.model import MelNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-p', '--infer_config', type=str, required=True,
                        help="yaml file for inference configuration")
    parser.add_argument('-t', '--timestep', type=int, default=240,
                        help="timestep of mel-spectrogram to generate")
    parser.add_argument('-n', '--name', type=str, default="result", required=False,
                        help="Name for sample")
    parser.add_argument('-i', '--input', type=str, default=None, required=False,
                        help="Input for conditional generation, leave empty for unconditional")
    args = parser.parse_args()

    hp = HParam(args.config)
    infer_hp = HParam(args.infer_config)

    assert args.timestep % t_div[hp.model.tier] == 0, \
        "timestep should be divisible by %d, got %d" % (t_div[hp.model.tier], args.timestep)

    model = MelNet(hp, args, infer_hp).cuda()
    model.load_tiers()
    model.eval()

    with torch.no_grad():
        generated = model.sample(args.input)

    os.makedirs('temp', exist_ok=True)
    torch.save(generated, os.path.join('temp', args.name + '.pt'))
    spectrogram = plot_spectrogram_to_numpy(generated[0].cpu().detach().numpy())
    plt.imsave(os.path.join('temp', args.name + '.png'), spectrogram.transpose((1, 2, 0)))

    waveform, wavespec = Reconstruct(hp).inverse(generated[0])
    wavespec = plot_spectrogram_to_numpy(wavespec.cpu().detach().numpy())
    plt.imsave(os.path.join('temp', 'Final ' + args.name + '.png'), wavespec.transpose((1, 2, 0)))

    waveform = waveform.unsqueeze(-1)
    waveform = waveform.cpu().detach().numpy()
    waveform *= 32768 / waveform.max()
    waveform = waveform.astype(np.int16)
    audio = audiosegment.from_numpy_array(
        waveform,
        framerate=hp.audio.sr
    )
    audio.export(os.path.join('temp', args.name + '.wav'), format='wav')
