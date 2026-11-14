import numpy as np
import torch
from sklearn.decomposition import PCA

day_of = lambda w: (w / (2 * np.pi) * 7).astype(int) % 7        # weekly phase -> 0 = Mon ... 6 = Sun
hour_of = lambda d: (d / (2 * np.pi) * 24).astype(int) % 24      # daily phase -> local hour


def cell_means(A, daily, weekly):
    """Average token per (day, hour). A: [..., D], phases: [...]. Returns [7, 24, D]."""
    A = A.reshape(-1, A.shape[-1])
    dd, hh = day_of(weekly.ravel()), hour_of(daily.ravel())
    return np.stack([[A[(dd == d) & (hh == h)].mean(0) for h in range(24)] for d in range(7)])


def day_shift(mu, from_day, to_day, hour):
    """Mean-shift edit: move tokens from from_day to to_day at the same hour. All index arrays [n, 32]."""
    return torch.tensor(mu[to_day, hour] - mu[from_day, hour], dtype=torch.float32)
