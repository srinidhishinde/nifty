from __future__ import annotations

import numpy as np
import pandas as pd

PATTERN_COLUMNS = (
    "doji", "hammer", "inverted_hammer", "shooting_star", "bullish_engulfing",
    "bearish_engulfing", "bullish_harami", "bearish_harami", "morning_star",
    "evening_star", "piercing_line", "dark_cloud_cover", "three_white_soldiers",
    "three_black_crows", "bullish_marubozu", "bearish_marubozu",
)


def add_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    out = df.copy()
    o, h, l, c = (pd.to_numeric(out[x], errors="coerce") for x in ("open", "high", "low", "close"))
    rng = (h - l).clip(lower=0)
    body = (c - o).abs()
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    body_safe = body.replace(0, np.nan)
    bullish, bearish = c > o, c < o
    prev_o, prev_c = o.shift(1), c.shift(1)
    prev_bull = bullish.shift(1).fillna(False).astype(bool)
    prev_bear = bearish.shift(1).fillna(False).astype(bool)
    prev_body = body.shift(1)
    prev2_o, prev2_c = o.shift(2), c.shift(2)
    bull2 = bullish.shift(2).fillna(False).astype(bool)
    bear2 = bearish.shift(2).fillna(False).astype(bool)

    out["doji"] = (body <= rng * 0.10).fillna(False).astype(int)
    out["hammer"] = ((lower >= body_safe * 2) & (upper <= body * 0.75) & ((c-l)/rng.replace(0,np.nan) >= 0.60)).fillna(False).astype(int)
    out["inverted_hammer"] = ((upper >= body_safe * 2) & (lower <= body * 0.75) & ((c-l)/rng.replace(0,np.nan) >= 0.60) & bullish).fillna(False).astype(int)
    out["shooting_star"] = ((upper >= body_safe * 2) & (lower <= body * 0.75) & ((h-c)/rng.replace(0,np.nan) <= 0.35)).fillna(False).astype(int)
    out["bullish_engulfing"] = (prev_bear & bullish & (c >= prev_o) & (o <= prev_c) & (body > prev_body)).fillna(False).astype(int)
    out["bearish_engulfing"] = (prev_bull & bearish & (c <= prev_o) & (o >= prev_c) & (body > prev_body)).fillna(False).astype(int)
    out.loc[prev_bull & bearish & (l <= prev_o) & (h >= prev_c) & (body >= prev_body), "bearish_engulfing"] = 1
    out.loc[prev_bear & bullish & (l <= prev_c) & (h >= prev_o) & (body >= prev_body), "bullish_engulfing"] = 1
    out["bullish_harami"] = (prev_bear & bullish & (o >= prev_c) & (c <= prev_o) & (body < prev_body)).fillna(False).astype(int)
    out["bearish_harami"] = (prev_bull & bearish & (o <= prev_c) & (c >= prev_o) & (body < prev_body)).fillna(False).astype(int)
    middle_small = prev_body <= (h.shift(1)-l.shift(1)).replace(0,np.nan)*0.35
    first_mid = (prev2_o+prev2_c)/2
    out["morning_star"] = (bear2 & middle_small & bullish & (c >= first_mid)).fillna(False).astype(int)
    out["evening_star"] = (bull2 & middle_small & bearish & (c <= first_mid)).fillna(False).astype(int)
    prev_mid = (prev_o+prev_c)/2
    out["piercing_line"] = (prev_bear & bullish & (c > prev_mid) & (c < prev_o) & (o < prev_c)).fillna(False).astype(int)
    out["dark_cloud_cover"] = (prev_bull & bearish & (c < prev_mid) & (c > prev_o) & (o > prev_c)).fillna(False).astype(int)
    out["three_white_soldiers"] = (bullish & bullish.shift(1).fillna(False).astype(bool) & bull2 & (c > c.shift(1)) & (c.shift(1) > c.shift(2)) & (o > o.shift(1)) & (o < c.shift(1)) & (o.shift(1) < c.shift(2))).fillna(False).astype(int)
    out["three_black_crows"] = (bearish & bearish.shift(1).fillna(False).astype(bool) & bear2 & (c < c.shift(1)) & (c.shift(1) < c.shift(2)) & (o < o.shift(1)) & (o > c.shift(1)) & (o.shift(1) > c.shift(2))).fillna(False).astype(int)
    out["bullish_marubozu"] = (bullish & body >= rng*0.85).fillna(False).astype(int)
    out["bearish_marubozu"] = (bearish & body >= rng*0.85).fillna(False).astype(int)

    weights = {"bullish_engulfing":3,"bearish_engulfing":-3,"morning_star":3,"evening_star":-3,
               "piercing_line":2,"dark_cloud_cover":-2,"three_white_soldiers":3,"three_black_crows":-3,
               "hammer":1,"inverted_hammer":1,"shooting_star":-2,"bullish_harami":1,"bearish_harami":-1,
               "bullish_marubozu":1,"bearish_marubozu":-1,"doji":0}
    out["candlestick_score"] = sum(out[name] * weight for name, weight in weights.items())
    return out
