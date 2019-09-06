import numpy as np
from tensorboardX import SummaryWriter

from . import plotting as plt 


class MyWriter(SummaryWriter):
    def __init__(self, hp, logdir):
        super(MyWriter, self).__init__(logdir)
        self.hp = hp

    def log_training(self, train_loss, step):
        self.add_scalar('train_loss', train_loss, step)

    def log_validation(self, test_loss, source, target, result, step):
        self.add_scalar('test_loss', test_loss, step)
        self.add_image('input', plt.plot_spectrogram_to_numpy(source), step)
        self.add_image('target', plt.plot_spectrogram_to_numpy(target), step)
        self.add_image('result', plt.plot_spectrogram_to_numpy(result), step)
        self.add_image('diff', plt.plot_spectrogram_to_numpy(np.abs(target - result)), step)

    def log_sample(self, step):
        raise NotImplementedError
