from math import ceil

import numpy as np
from scipy.stats import norm, binomtest
from statsmodels.stats.proportion import proportion_confint
import torch
import random
import pcen
from MelScale import MelScale
import scipy.fftpack as fftpack

class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, tf, alphas, base_classifier: torch.nn.Module, num_classes: int, sigma: float):
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
        self.tf = torch.from_numpy(tf).to("cuda")
        self.alphas = alphas
        self.n_mels = 40
        # self.dct_filters = torch.from_numpy(librosa.filters.dct(40, self.n_mels).astype(np.float32)).to(device)
        dct_filters_np = fftpack.dct(np.eye(self.n_mels), type=2, norm='ortho', n=40)
        self.dct_filters = torch.from_numpy(dct_filters_np.astype(np.float32)).to("cuda")
        self.sr = 16000
        self.f_max = 8000
        self.f_min = 20
        self.n_fft = 480
        self.hop_length = 16000 // 1000 * 10
        self.pcen_transform = pcen.StreamingPCENTransform(n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length, trainable=True)
        self.melscale_transform = MelScale(sample_rate=16000, f_min=20, f_max=8000, n_mels =40, n_stft =241, norm ="slaney")

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def predict(self, x: torch.tensor, n: int, batch_size: int) -> int:
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        
        return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1))
                sampled_alphas = random.choices(self.alphas, k=this_batch_size)
                sampled_alphas = torch.stack(sampled_alphas, 0).to("cuda")
                batched_tf = self.tf.repeat((this_batch_size, 1, 1))

                predictions = self.base_classifier(self.transform(batch, sampled_alphas, batched_tf, True)).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts
        
    def transform(self, inputs, alphas, tf, use_noise=False):
        noise = torch.rand_like(tf) * self.sigma if use_noise else 0
        inputs = torch.exp(alphas + noise) * tf * inputs

        ## compute mfcc
        data = (inputs**2).float()
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

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
    
