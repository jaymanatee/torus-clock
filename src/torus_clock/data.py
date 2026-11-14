import numpy as np
import pandas as pd

TZ = "Europe/Paris"


def load_rte(*paths) -> pd.Series:
    """Hourly French consumption (MW) from éCO2mix annual files, on a gap-free UTC index (gaps = NaN)."""
    df = pd.concat(pd.read_csv(p, sep="\t", encoding="latin-1", index_col=False, low_memory=False) for p in paths)
    df["Consommation"] = pd.to_numeric(df["Consommation"], errors="coerce")   # "ND" etc. -> NaN
    df = df.dropna(subset=["Consommation"])
    local = pd.to_datetime(df["Date"] + " " + df["Heures"])
    # timestamps are local clock time: the skipped spring hour is padded with copies,
    # the repeated autumn hour appears once; drop both instead of guessing
    utc = local.dt.tz_localize(TZ, nonexistent="NaT", ambiguous="NaT").dt.tz_convert("UTC")
    s = pd.Series(df["Consommation"].astype(float).values, index=utc).loc[lambda x: x.index.notna()]
    s = s[~s.index.duplicated()].sort_index().resample("1h").mean()      # two half-hours -> one hour
    return s.rename("load_mw")


def calendar_phases(index: pd.DatetimeIndex):
    """Daily and weekly phase in local (wall-clock) time, at the middle of each hour."""
    loc = index.tz_convert(TZ)
    hour = loc.hour + 0.5
    return 2 * np.pi * hour / 24, 2 * np.pi * (loc.dayofweek * 24 + hour) / 168


def make_windows(load: pd.Series, start: str, end: str, T: int = 512, H: int = 64, stride: int = 17):
    """Windows of T context + H future hours whose first hour falls in [start, end).

    Returns x [n, T+H] (float32, NaN where missing), the UTC index of each window [n, T+H],
    and the daily / weekly phases [n, T+H] in local time.
    """
    s = load[start:end]
    first = np.arange(0, len(s) - (T + H) + 1, stride)
    idx = np.stack([s.index[i:i + T + H] for i in first])      # timestamps of every hour in every window
    x = np.stack([s.values[i:i + T + H] for i in first]).astype(np.float32)
    daily, weekly = calendar_phases(pd.DatetimeIndex(idx.ravel()))
    return x, idx, daily.to_numpy().reshape(x.shape), weekly.to_numpy().reshape(x.shape)
