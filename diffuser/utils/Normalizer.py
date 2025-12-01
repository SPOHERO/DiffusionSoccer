import torch

class GaussianNormalizer:
    def __init__(self, X):
        # X: [N, F] torch tensor
        self.means = X.mean(dim=0, keepdim=True)
        self.stds  = X.std(dim=0, keepdim=True) + 1e-6

    def normalize(self, x):
        return (x - self.means) / self.stds

    def unnormalize(self, x):
        return x * self.stds + self.means
