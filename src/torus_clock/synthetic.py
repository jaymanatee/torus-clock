import numpy as np

P_DAY, P_WEEK = 24, 168

def make_series(n: int, length: int, noise: float=0.5, seed: int=0):
    rng = np.random.default_rng(seed)
    t = np.arange(length)[None, :]
    phi0 = rng.uniform(0, 2*np.pi, (n,1))
    psi0 = rng.uniform(0, 2*np.pi, (n,1))
    theta = 2*np.pi*t/P_DAY + phi0
    omega = 2*np.pi*t/P_WEEK + psi0

    w = rng.uniform(0.3, 1.0, (n,1))
    trend = rng.normal(0, 0.002, (n,1))
    amp = rng.lognormal(0, 0.5, (n,1))
    level = rng.normal(0, 3, (n,1))

    clean = level + amp * (np.cos(theta) + w*np.cos(omega) + trend*t)
    x = clean + amp * noise * rng.normal(size=(n, length))
    return {
        "x": x.astype(np.float32),
        "daily_phase": np.mod(theta, 2*np.pi),
        "weekly_phase": np.mod(omega, 2*np.pi),
    }
