import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from .gmm import sample_gmm


def validate(args, model, melgen, tierutil, testloader, criterion, writer, step):
    model.eval()
    # torch.backends.cudnn.benchmark = False

    test_loss = []
    loader = tqdm(testloader, desc='Testing is in progress', dynamic_ncols=True)
    with torch.no_grad():
        for input_tuple in loader:
            if args.tts:
                seq, text_lengths, source, target, audio_lengths = input_tuple
                mu, std, pi, alignment = model(
                    source.cuda(non_blocking=True),
                    seq.cuda(non_blocking=True),
                    text_lengths.cuda(non_blocking=True),
                    audio_lengths.cuda(non_blocking=True)
                )
            else:
                source, target, audio_lengths = input_tuple
                mu, std, pi = model(
                    source.cuda(non_blocking=True),
                    audio_lengths.cuda(non_blocking=True)
                )
            loss = criterion(
                target.cuda(non_blocking=True),
                mu, std, pi,
                audio_lengths.cuda(non_blocking=True)
            )
            test_loss.append(loss)

        test_loss = sum(test_loss) / len(test_loss)
        audio_length = audio_lengths[0].item()
        source = source[0].cpu().detach().numpy()[:, :audio_length]
        target = target[0].cpu().detach().numpy()[:, :audio_length]
        result = sample_gmm(mu[0], std[0], pi[0]).cpu().detach().numpy()[:, :audio_length]
        if args.tts:
            alignment = alignment[0].cpu().detach().numpy()[:, :audio_length]
        else:
            alignment = None
        writer.log_validation(test_loss, source, target, result, alignment, step)

    model.train()
    # torch.backends.cudnn.benchmark = True
