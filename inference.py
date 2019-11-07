import os
import glob
import argparse
import torch

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
    args = parser.parse_args()

    hp = HParam(args.config)
    infer_hp = HParam(args.infer_config)

    assert args.timestep % t_div[hp.model.tier] == 0, \
        "timestep should be divisible by %d, got %d" % (t_div[hp.model.tier], args.timestep)

    model = MelNet(hp, args, infer_hp).cuda()
    model.load_tiers()
    model.eval()

    with torch.no_grad():
        x = model.unconditional_sample()

    print(x)
    print(x.shape)
    torch.save(x, 'result.pt')

