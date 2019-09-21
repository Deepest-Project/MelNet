root_dir = '../dataset/KSS/wavs'
n_tiers = 6
sr = 22050
n_fft = 6 * 256
n_hop = 256
n_mels = 256

K=3
layers=[6, 5, 4, 3, 2, 2]
batch_size = 4
hidden_dim = 512
epochs=5
use_central=True
n_bucket=10