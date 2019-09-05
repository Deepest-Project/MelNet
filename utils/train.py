import os
import math
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import traceback

from model.model import MelNet
from model.loss import GMMLoss
from .utils import get_commit_hash
from .audio import MelGen
from .tierutil import TierUtil


def train(args, pt_dir, chkpt_path, trainloader, testloader, writer, logger, hp, hp_str):
    model = MelNet(hp).cuda()
    melgen = MelGen(hp)
    tierutil = TierUtil(hp)
    criterion = GMMLoss()

    if hp.train.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=hp.train.rmsprop.lr, momentum=hp.train.rmsprop.momentum)
    elif hp.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=hp.train.adam.lr)
    else:
        raise Exception("%s optimizer not supported yet" % hp.train.optimizer)

    githash = get_commit_hash()

    init_epoch = -1
    step = 0

    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['step']
        init_epoch = checkpoint['epoch']

        if hp_str != checkpoint['hp_str']:
            logger.warning("New hparams is different from checkpoint. Will use new.")

        if githash != checkpoint['githash']:
            logger.warning("Code might be different: git hash is different.")
            logger.warning("%s -> %s" % (checkpoint['githash'], githash))

        githash = checkpoint['githash']
    else:
        logger.info("Starting new training run.")

    # use this only if input size is always consistent.
    torch.backends.cudnn.benchmark = True
    try:
        model.train()
        for epoch in itertools.count(init_epoch+1):
            loader = tqdm.tqdm(trainloader, desc='Train data loader')
            for audio in loader:
                audio = audio.cuda()
                mel = melgen.get_logmel(audio)
                tiers = tierutil.cut_divide_tiers(mel)

                # for tier in range(1, hp.model.tier+1):
                for tierN in range(2, 3):
                    mu, std, pi = model(tiers[tierN], tierN)

                loss = criterion(tiers[tierN], mu, std, pi)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1

                loss = loss.item()
                if loss > 1e8 or math.isnan(loss):
                    logger.error("Loss exploded to %.02f at step %d!" % (loss, step))
                    raise Exception("Loss exploded")

                if step % hp.log.summary_interval == 0:
                    writer.log_training(loss, step)
                    loader.set_description("Loss %.02f at step %d" % (loss, step))

            save_path = os.path.join(pt_dir, '%s_%s_%03d.pt'
                % (args.name, githash, epoch))
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'epoch': epoch,
                'hp_str': hp_str,
                'githash': githash,
            }, save_path)
            logger.info("Saved checkpoint to: %s" % save_path)

            # validate(?, ?)
    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()
