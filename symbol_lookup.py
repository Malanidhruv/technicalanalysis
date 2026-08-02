"""Map AliceBlue tokens to trading symbols from local contract CSVs."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _load_maps():
    nse_map, bse_map = {}, {}

    nse_path = _ROOT / "NSE.csv"
    if nse_path.is_file():
        df = pd.read_csv(nse_path, usecols=["Symbol", "Token"], dtype=str)
        nse_map = dict(zip(df["Token"].str.strip(), df["Symbol"].str.strip()))

    # Filename has a space from AliceBlue export
    bse_path = _ROOT / "BSE (1).csv"
    if bse_path.is_file():
        df = pd.read_csv(bse_path, usecols=["Symbol", "Token"], dtype=str)
        bse_map = dict(zip(df["Token"].str.strip(), df["Symbol"].str.strip()))

    return nse_map, bse_map


def token_to_symbol(token, exchange="NSE"):
    """Return Symbol for a token, or the token string if unknown."""
    token_key = str(token).strip()
    nse_map, bse_map = _load_maps()
    exch = "BSE" if str(exchange).upper().startswith("BSE") else "NSE"
    lookup = bse_map if exch == "BSE" else nse_map
    return lookup.get(token_key, token_key)
