import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


def wrap(a):
    return np.angle(np.exp(1j * a))


def fit_probe(X, phase):
    """Ridge probe from vectors [..., dim] to (cos, sin) of phase [...]. Returns a predict function."""
    X, y = X.reshape(-1, X.shape[-1]), phase.reshape(-1)
    scaler = StandardScaler().fit(X)
    reg = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(scaler.transform(X), np.c_[np.cos(y), np.sin(y)])
    return lambda B: reg.predict(scaler.transform(B.reshape(-1, B.shape[-1]))).reshape(*B.shape[:-1], 2)


def probe_score(predict, X, phase, period):
    """Mean cos(error) (1 perfect, 0 chance) and median |error| in hours, on held-out data."""
    pred = predict(X)
    err = wrap(np.arctan2(pred[..., 1], pred[..., 0]) - phase)
    return np.mean(np.cos(err)), np.median(np.abs(err)) * period / (2 * np.pi)
