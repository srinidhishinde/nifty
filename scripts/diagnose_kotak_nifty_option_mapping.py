"""Read-only diagnostic for Kotak NIFTY option-chain token mapping."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from broker.kotak_neo import KotakNeoBroker
from marketdata.providers.kotak_neo import KotakNeoProvider

def safe(v):
    s=str(v)
    return s if len(s)<=100 else s[:97]+"..."

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--totp",required=True)
    a=p.parse_args()
    if not a.totp.isdigit() or len(a.totp)!=6: p.error("--totp must be exactly six digits")
    broker=KotakNeoBroker()
    conn=broker.authenticate(a.totp)
    if not conn.connected:
        print("[FAIL] Authentication:",conn.message); return 1
    try:
        client=broker.client; provider=KotakNeoProvider(client)
        expiry=provider._nearest_expiry("NSE_FO","NIFTY")
        print("[INFO] expiry:",expiry)
        raw=client.option_chain(exchange="nse_fo",underlying="NIFTY",expiry=expiry,instrument_type="option",count=40)
        data=provider._response_data(raw)
        calls=data.get("call") or []
        if isinstance(calls,dict): calls=list(calls.values())
        if not calls:
            print("[FAIL] no calls"); print(provider.option_chain_diagnostics(raw)); return 1
        item=calls[0]; inst=item.get("instrument") or item.get("inst") or {}
        print("[INFO] option instrument keys:",sorted(inst.keys()))
        for k,v in inst.items(): print("  ",k,"=",safe(v))
        token_keys=("neoSymbol","pSymbol","exchangeToken","instrumentToken","instrument_token","token")
        tokens=[str(inst.get(k) or "").strip() for k in token_keys if inst.get(k)]
        print("[INFO] candidate tokens:",tokens)
        rows=client.search_scrip(exchange_segment="nse_fo",symbol="NIFTY",expiry="",option_type="",strike_price="")
        if not isinstance(rows,list): rows=[]
        print("[INFO] scrip rows:",len(rows))
        matched=[]
        for row in rows:
            ps=str(row.get("pSymbol") or "").strip()
            if any(t==ps or t.split("|")[-1]==ps for t in tokens): matched.append(row)
        print("[INFO] token matches:",len(matched))
        for row in matched[:10]:
            print("  MATCH", {k:safe(row.get(k)) for k in ("pSymbol","pTrdSymbol","pSymbolName","pExchSeg","pExpiryDate","pOptionType","dStrikePrice") if k in row})
        return 0
    except Exception as e:
        print("[FAIL] diagnostic:",e); return 1
    finally: broker.logout()
if __name__=="__main__": raise SystemExit(main())
