import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gmm import sample_gmm


def validate(args, model, melgen, tierutil, testloader, criterion, writer, step):
    model.eval()
    torch.backends.cudnn.benchmark = False

    test_loss = []
    loader = tqdm.tqdm(testloader, desc='Testing is in progress')
    with torch.no_grad():
        for source, target in loader:
            mu, std, pi = model(source.cuda())
            loss = criterion(target.cuda(), mu, std, pi)
            test_loss.append(loss.item())

        test_loss = sum(test_loss) / len(test_loss)
        source = source[0].cpu().detach().numpy()
        target = target[0].cpu().detach().numpy()
        result = sample_gmm(mu[0], std[0], pi[0]).cpu().detach().numpy()
        writer.log_validation(test_loss, source, target, result, step)

    model.train()
    torch.backends.cudnn.benchmark = True
