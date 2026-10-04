from __future__ import annotations
from datetime import date, timedelta
import pandas as pd
from marketdata.yahoo_finance import fetch_yahoo_ohlcv, INTRADAY_MAX_DAYS

def fetch_yahoo_rolling_window(start: date, end: date, *, symbol="^NSEI", interval="5m"):
    if interval == "1d":
        return fetch_yahoo_ohlcv(start,end,symbol=symbol,interval=interval)
    chunks=[]; cursor=start
    while cursor <= end:
        chunk_end=min(end,cursor+timedelta(days=INTRADAY_MAX_DAYS-1))
        result=fetch_yahoo_ohlcv(cursor,chunk_end,symbol=symbol,interval=interval)
        if result.status not in {"OK","NO_DATA"}:
            return result
        if result.status=="OK" and not result.data.empty: chunks.append(result.data)
        cursor=chunk_end+timedelta(days=1)
    if not chunks:
        return fetch_yahoo_ohlcv(start,end,symbol=symbol,interval=interval)
    from marketdata.yahoo_finance import YahooOHLCVResult
    frame=pd.concat(chunks,ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return YahooOHLCVResult(frame,symbol,interval,start,end,frame.timestamp.min().date(),frame.timestamp.max().date(),"OK",
        f"Yahoo Finance {symbol} | interval {interval} | {len(frame):,} candles | chunked rolling window {frame.timestamp.min().date()} to {frame.timestamp.max().date()}")
