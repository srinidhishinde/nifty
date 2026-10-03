from datetime import time
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

def market_is_open(timestamp, instrument: str) -> bool:
    ts = timestamp
    if getattr(ts, "tzinfo", None) is None:
        ts = ts.replace(tzinfo=IST)
    else:
        ts = ts.astimezone(IST)
    current = ts.time()
    instrument = instrument.upper()
    if instrument == "NIFTY":
        return time(9, 15) <= current <= time(15, 30)
    if instrument == "MCX":
        return time(9, 0) <= current <= time(23, 30)
    return False
