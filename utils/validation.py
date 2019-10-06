import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


def validate(args, model, melgen, tierutil, testloader, criterion, writer, step):
    model.eval()
    torch.backends.cudnn.benchmark = False

    test_loss = 0.0
    loader = tqdm.tqdm(testloader, desc='Testing is in progress')
    with torch.no_grad():
        for source, target in loader:
            # audio = audio.cuda()
            # mel = melgen.get_normalized_mel(audio)
            # source, target = tierutil.cut_divide_tiers(mel, args.tier)
            result = model(source.cuda())
            loss = criterion(result, target.cuda())

            loss = loss.item()
            test_loss += loss

        test_loss /= len(testloader.dataset)
        source = source[0].cpu().detach().numpy()
        target = target[0].cpu().detach().numpy()
        result = result[0].cpu().detach().numpy()
        writer.log_validation(test_loss, source, target, result, step)

    model.train()
    torch.backends.cudnn.benchmark = True
