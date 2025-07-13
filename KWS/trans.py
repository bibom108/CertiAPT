
import librosa
import numpy as np
import torch
import torch.nn as nn
import pcen
from MelScale import MelScale
import scipy.fftpack as fftpack

device = torch.device('cuda')

class Transformation(nn.Module):
    def __init__(self, tf, sigma):
        super(Transformation, self).__init__()
        self.n_mels = 40
        # self.dct_filters = torch.from_numpy(librosa.filters.dct(40, self.n_mels).astype(np.float32)).to(device)
        dct_filters_np = fftpack.dct(np.eye(self.n_mels), type=2, norm='ortho', n=40)
        self.dct_filters = torch.from_numpy(dct_filters_np.astype(np.float32)).to(device)
        self.sr = 16000
        self.f_max = 8000
        self.f_min = 20
        self.n_fft = 480
        self.hop_length = 16000 // 1000 * 10
        self.pcen_transform = pcen.StreamingPCENTransform(n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length, trainable=True)
        self.sigma = sigma
        self.melscale_transform = MelScale(sample_rate=16000, f_min=20, f_max=8000, n_mels =40, n_stft =241, norm ="slaney")
        self.tf = [torch.from_numpy(tf[i]).to(device) for i in range(len(tf))]
        self.alphas = nn.ParameterList([torch.nn.Parameter(torch.randn_like(self.tf[i][2])) for i in range(len(tf))]) 

    def count(self):
        return len(self.tf)
    
    def reset_alpha(self):
        self.alphas = nn.ParameterList([torch.nn.Parameter(torch.randn_like(self.tf[i])) for i in range(len(self.tf))])
    
    def forward(self, inputs, index=-1, use_noise=False):
        # inputs is tensor with batch
        if index != -1:
            noise = torch.rand_like(self.tf[index]) * self.sigma if use_noise else 0
            inputs = torch.exp(self.alphas[index] + noise) * self.tf[index] * inputs
        ## compute mfcc
        data = inputs**2
        data = self.melscale_transform(data)
        data[data > 0] = torch.log(data[data > 0])
        data = [torch.matmul(self.dct_filters, x) for x in torch.split(data, data.shape[1], dim=1)][0]
        ## adding z-score normalize
        mean = torch.mean(data)
        std = torch.std(data)
        if std != 0:
            data = (data - mean)/std
        data = torch.transpose(data, 1, 2)
        return data
