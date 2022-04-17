import numpy as np
import random
from torch.utils.data.dataset import Dataset
import torch

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

class Mixup(Dataset):
    def __init__(self, dataset, num_class, beta=1., prob=1.0):
        self.dataset = dataset
        self.num_class = num_class
        self.beta = beta
        self.prob = prob

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        lb_onehot = onehot(self.num_class, lb)

        r = np.random.rand(1)
        if self.beta >= 0 and r <= self.prob:

            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))
            img2, lb2 = self.dataset[rand_index]
            lb2_onehot = onehot(self.num_class, lb2)

            img = lam * img + (1 - lam) * img2
            lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

        return img, lb_onehot

    def __len__(self):
        return len(self.dataset)