from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote
import feedparser
import requests

POSITIVE = {"surge","rally","gain","gains","bullish","growth","strong","beat","recovery","optimism","up"}
NEGATIVE = {"crash","fall","falls","loss","losses","bearish","recession","war","risk","weak","drop","drops","inflation","tariff"}

@dataclass(frozen=True)
class GlobalNewsSnapshot:
    sentiment: float
    headline: str
    headlines: tuple[str, ...]

def score_text(text: str) -> float:
    words = {w.strip(".,:;!?()[]{}").lower() for w in text.split()}
    pos, neg = len(words & POSITIVE), len(words & NEGATIVE)
    total = pos + neg
    return 0.0 if total == 0 else max(-1.0, min(1.0, (pos - neg) / total))

def fetch_global_news(query: str = "global markets NIFTY India economy") -> GlobalNewsSnapshot:
    url = "https://news.google.com/rss/search?q=" + quote(query) + "&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        response = requests.get(url, timeout=5, headers={"User-Agent":"AI-Derivatives-Terminal/1.0"})
        response.raise_for_status()
        feed = feedparser.parse(response.content)
        headlines = tuple(str(e.get("title","")).strip() for e in feed.entries[:10] if str(e.get("title","")).strip())
    except Exception:
        headlines = ()
    if not headlines:
        return GlobalNewsSnapshot(0.0, "Global news unavailable", ())
    sentiment = sum(score_text(h) for h in headlines) / len(headlines)
    return GlobalNewsSnapshot(round(sentiment,3), headlines[0], headlines)
