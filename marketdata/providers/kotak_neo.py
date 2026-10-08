from datetime import date, datetime
from typing import Any, List

from marketdata.models.market_data import Candle, OptionContract, Quote
from marketdata.providers.base import MarketDataProvider
from strategy.market_specs import validate_option_strike


class KotakNeoProvider(MarketDataProvider):
    """Adapter for the current Kotak Neo Python SDK market-data APIs."""

    EXCHANGE_ALIASES = {
        "NSE": "nse_fo",
        "NSE_FO": "nse_fo",
        "MCX": "mcx_fo",
        "MCX_FO": "mcx_fo",
        "BSE": "bse_fo",
        "BSE_FO": "bse_fo",
    }

    def __init__(self, client):
        self.client = client

    def get_account_state(self) -> dict:
        """Return the authoritative Neo limits/account snapshot.

        Account truth must come from the dedicated limits endpoint. We do not
        guess between positions, margin or other endpoints because those
        payloads have different semantics and can turn an incomplete response
        into a false equity/margin value.
        """
        fn = getattr(self.client, "limits", None)
        if not callable(fn):
            raise RuntimeError("Kotak Neo limits endpoint is unavailable.")
        raw = fn()
        if not isinstance(raw, dict):
            raise RuntimeError("Kotak Neo limits endpoint returned an invalid response.")
        error = self._response_error(raw)
        if error:
            raise RuntimeError(f"Kotak Neo account-state error: {error}")
        data = raw.get("data") if isinstance(raw.get("data"), dict) else raw
        if not isinstance(data, dict):
            raise RuntimeError("Kotak Neo account-state payload is invalid.")

        def required_number(keys: tuple[str, ...], name: str, *, allow_zero: bool) -> float:
            for key in keys:
                value = data.get(key)
                if value not in (None, ""):
                    parsed = self._float(value, float("nan"))
                    if parsed == parsed and parsed != float("inf") and parsed != float("-inf"):
                        if parsed > 0 or (allow_zero and parsed == 0):
                            return parsed
                    break
            raise RuntimeError(f"Kotak Neo account response is missing valid {name}.")

        timestamp = next(
            (data.get(k) for k in ("timestamp", "updatedAt", "updateTime", "lastUpdated") if data.get(k) not in (None, "")),
            None,
        )
        if timestamp is None:
            raise RuntimeError("Kotak Neo account response is missing authoritative timestamp.")

        equity = required_number(("equity", "netWorth", "net"), "equity", allow_zero=False)
        available_margin = required_number(
            ("availableMargin", "available_margin", "availableCash", "cash"),
            "available margin",
            allow_zero=True,
        )
        return {
            "equity": equity,
            "available_margin": available_margin,
            "timestamp": timestamp,
            "source": "KOTAK_NEO",
        }

    @classmethod
    def normalize_exchange(cls, exchange: str) -> str:
        value = str(exchange or "").strip()
        return cls.EXCHANGE_ALIASES.get(value.upper(), value.lower())

    @staticmethod
    def _float(value: Any, default: float = 0.0) -> float:
        try:
            if isinstance(value, str):
                value = value.strip().replace(',', '')
                if not value:
                    return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _quote_ltp(cls, quote: dict, fallback: float = 0.0) -> float:
        """Read Kotak LTP across current/legacy quote field names."""
        for key in (
            "ltp",
            "lastTradedPrice",
            "last_traded_price",
            "lastPrice",
            "last_price",
            "LTP",
        ):
            if key in quote and quote.get(key) not in (None, ""):
                value = cls._float(quote.get(key), fallback)
                if value > 0:
                    return value
        return fallback

    @staticmethod
    def _response_data(response: Any) -> dict:
        """Normalize both legacy and current Kotak market-data envelopes."""
        if not isinstance(response, dict):
            return {}
        data = response.get("data")
        if isinstance(data, dict):
            return data

        # Current Neo SDK option_chain() returns the payload directly:
        # {"call": [...], "put": [...], "common_data": {...}, ...}.
        payload_keys = {"call", "put", "common_data", "spot", "future", "future_contracts"}
        if payload_keys.intersection(response):
            return response
        return {}

    @staticmethod
    def _response_error(response: Any) -> str:
        if not isinstance(response, dict):
            return "Kotak Neo returned an invalid response."

        explicit_status = str(response.get("stat", "") or "").strip().lower()
        explicit_status_alt = str(response.get("status", "") or "").strip().lower()
        if explicit_status in {"not_ok", "error", "failed"} or explicit_status_alt in {"error", "failed"}:
            # Never treat a payload-bearing error response as successful market
            # data. Broker diagnostics take precedence over a non-empty data key.
            pass
        elif explicit_status == "ok" or explicit_status_alt in {"ok", "success"}:
            return ""
        else:
            # Some current Neo market-data success responses omit stat/status.
            # The current SDK also returns option-chain payload fields at top level.
            data = response.get("data")
            payload_keys = {"call", "put", "common_data", "spot", "future", "future_contracts"}
            if (isinstance(data, dict) and data) or payload_keys.intersection(response):
                return ""

        # Preserve the broker's actual diagnostic. The previous implementation
        # collapsed every no-data response into the same generic message, which
        # made it impossible to distinguish an expired/invalid expiry, a rate
        # limit, a session issue, or genuine absence of market data.
        code = response.get("stCode") or response.get("code") or response.get("statusCode")
        message = (
            response.get("errMsg")
            or response.get("emsg")
            or response.get("message")
            or response.get("msg")
            or response.get("desc")
            or response.get("errorMessage")
        )
        errors = response.get("error")
        if isinstance(errors, list) and errors:
            first = errors[0]
            if isinstance(first, dict):
                code = first.get("code") or code
                message = first.get("message") or message
            elif first:
                message = str(first)

        fault = response.get("fault")
        if isinstance(fault, dict):
            code = fault.get("code") or code
            message = fault.get("message") or message

        if code and message:
            return f"{message} (code {code})"
        if message:
            return str(message)
        if code:
            return f"Market-data request failed (code {code})"
        return "Kotak Neo returned no market-data records."

    def resolve_nifty_index_neosymbol(self, index_name: str = "Nifty 50") -> str:
        """Resolve the current NIFTY index Neo symbol from Kotak's scrip master."""
        queries = [index_name.strip(), "NIFTY", "Nifty 50"]
        rows = []
        seen = set()
        for query in queries:
            if not query or query.upper() in seen:
                continue
            seen.add(query.upper())
            result = self.client.search_scrip(
                exchange_segment="nse_cm",
                symbol=query,
                expiry="",
                option_type="",
                strike_price="",
            )
            if isinstance(result, list):
                rows.extend(result)

        if not rows:
            raise RuntimeError(
                f"Kotak Neo scrip master returned no NSE cash instrument for '{index_name}'."
            )

        target_names = {"NIFTY", "NIFTY 50", "NIFTY50"}
        candidates = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            token = str(row.get("pSymbol") or "").strip()
            segment = str(row.get("pExchSeg") or "nse_cm").strip().lower()
            name = str(row.get("pSymbolName") or "").strip()
            trading_symbol = str(row.get("pTrdSymbol") or "").strip()
            if not token or segment != "nse_cm":
                continue

            upper_name = name.upper()
            upper_trading_symbol = trading_symbol.upper()
            exact = 0 if upper_name in target_names or upper_trading_symbol in target_names else 1
            candidates.append((exact, token, name))

        if not candidates:
            raise RuntimeError(
                f"Kotak Neo scrip master returned no valid nse_cm NIFTY token for '{index_name}'."
            )

        candidates.sort(key=lambda item: (item[0], item[1]))
        return f"nse_cm|{candidates[0][1]}"

    def get_quote(self, symbol: str, exchange: str) -> Quote:
        raise NotImplementedError(
            "Resolve the Neo instrument token before requesting a quote."
        )

    def get_index_quote(self, index_name: str = "Nifty 50") -> Quote:
        """Fetch NIFTY using a current scrip-master token and broker timestamp."""
        neosymbol = self.resolve_nifty_index_neosymbol(index_name)
        segment, token = neosymbol.split("|", 1)
        response = self.client.quotes(
            instrument_tokens=[{"instrument_token": token, "exchange_segment": segment}],
            quote_type="all",
        )
        if isinstance(response, dict):
            data = self._response_data(response)
            response = data.get("quotes") or data.get("data") or []
        if not isinstance(response, list) or not response or not isinstance(response[0], dict):
            raise RuntimeError("Kotak Neo returned no NIFTY index quote.")
        row = response[0]
        ltp = self._float(row.get("ltp"))
        if ltp <= 0:
            raise RuntimeError("Kotak Neo returned an invalid NIFTY index price.")
        raw_ts = row.get("timestamp") or row.get("quoteTimestamp") or row.get("quote_timestamp") or row.get("lastTradeTime") or row.get("last_traded_time")
        quote_ts = None
        if raw_ts not in (None, ""):
            try:
                quote_ts = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
            except (TypeError, ValueError):
                try:
                    quote_ts = datetime.fromtimestamp(float(raw_ts))
                except (TypeError, ValueError, OSError):
                    raise RuntimeError("Kotak Neo NIFTY quote did not contain a parseable broker timestamp.")
        if quote_ts is None:
            raise RuntimeError("Kotak Neo NIFTY quote did not contain an authoritative broker timestamp.")
        return Quote(
            timestamp=quote_ts, symbol=index_name, exchange=segment,
            ltp=ltp, volume=self._float(row.get("last_volume") or row.get("volume")),
            open_interest=self._float(row.get("open_int")), bid=None, ask=None,
        )
    def _nearest_expiry(self, exchange: str, underlying: str) -> str:
        """Resolve the nearest available option expiry from Neo."""
        exchange_segment = self.normalize_exchange(exchange)
        response = self.client.expiries(
            exchange=exchange_segment,
            underlying=underlying.upper(),
            instrument_type="option",
        )
        if not isinstance(response, dict):
            raise RuntimeError("Kotak Neo expiry API returned an invalid response.")

        expiries = response.get("expiries") or []
        if not isinstance(expiries, list):
            data = response.get("data")
            if isinstance(data, dict):
                expiries = data.get("expiries") or []

        normalized = sorted(
            str(value).strip()
            for value in expiries
            if str(value).strip()
        )
        if not normalized:
            error = self._response_error(response)
            raise RuntimeError(
                f"Kotak Neo returned no available {underlying.upper()} option expiries"
                + (f": {error}" if error else ".")
            )

        # The API documents ISO YYYY-MM-DD expiries sorted ascending.
        # Choose the first expiry that is today or later.
        today = date.today().isoformat()
        future = [value for value in normalized if value >= today]
        if not future:
            raise RuntimeError(
                f"Kotak Neo returned only expired {underlying.upper()} option expiries."
            )
        return future[0]

    def resolve_option_underlying(self, underlying: str, exchange: str) -> str:
        """Resolve Kotak's canonical option-chain underlying name.

        Kotak option_chain() expects the underlying to match pSymbolName in
        the current scrip master. MCX display names can differ from that API
        identifier, so resolve the broker name instead of guessing.
        """
        requested = str(underlying or "").strip().upper()
        if not requested:
            raise ValueError("Option-chain underlying cannot be empty.")

        exchange_segment = self.normalize_exchange(exchange)
        aliases = [requested]
        if exchange_segment == "mcx_fo":
            aliases.extend({
                "CRUDEOIL": ["CRUDEOIL", "CRUDEOILM"],
                "CRUDEOILM": ["CRUDEOILM", "CRUDEOIL"],
            }.get(requested, []))

        seen = set()
        candidates = []
        for alias in aliases:
            alias = str(alias).strip().upper()
            if not alias or alias in seen:
                continue
            seen.add(alias)
            rows = self.client.search_scrip(
                exchange_segment=exchange_segment,
                symbol=alias,
                expiry="",
                option_type="",
                strike_price="",
            )
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                segment = str(row.get("pExchSeg") or exchange_segment).strip().lower()
                name = str(row.get("pSymbolName") or "").strip().upper()
                if segment == exchange_segment and name:
                    exact = 0 if name == requested else 1
                    candidates.append((exact, name))

        if not candidates:
            raise RuntimeError(
                f"Kotak Neo scrip master returned no {exchange_segment} option underlying for '{requested}'."
            )
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][1]

    @staticmethod
    def option_chain_diagnostics(response: Any) -> dict:
        """Return safe, non-credential diagnostics for a raw Neo option-chain response."""
        if not isinstance(response, dict):
            return {
                "response_type": type(response).__name__,
                "top_level_keys": [],
                "stat": "",
                "status": "",
                "stCode": "",
                "errMsg": "",
                "desc": "",
                "data_type": type(response).__name__,
                "data_keys": [],
                "call_count": 0,
                "put_count": 0,
            }
        data = response.get("data")
        if not isinstance(data, dict):
            data = {}
        calls = data.get("call") or data.get("calls") or []
        puts = data.get("put") or data.get("puts") or []
        return {
            "response_type": type(response).__name__,
            "top_level_keys": sorted(str(k) for k in response.keys()),
            "stat": str(response.get("stat") or ""),
            "status": str(response.get("status") or ""),
            "stCode": str(response.get("stCode") or response.get("code") or ""),
            "errMsg": str(response.get("errMsg") or response.get("emsg") or ""),
            "desc": str(response.get("desc") or ""),
            "data_type": type(data).__name__,
            "data_keys": sorted(str(k) for k in data.keys()),
            "call_count": len(calls) if isinstance(calls, list) else 0,
            "put_count": len(puts) if isinstance(puts, list) else 0,
        }

    def _scrip_master_option_symbols(
        self,
        underlying: str,
        exchange: str,
        expiry: str | None,
    ) -> dict[str, str]:
        """Map broker option tokens to canonical trading symbols.

        Some Neo option-chain responses have returned a malformed strike field.
        The scrip master is the broker's canonical instrument source, so use
        pTrdSymbol keyed by pSymbol before trusting a numeric strike payload.
        """
        # Neo's scrip-master endpoint is inconsistent about the expiry
        # format accepted by search_scrip(). The option-chain API may return
        # ISO expiry while search_scrip expects a different representation.
        # Query the underlying without an expiry first so token -> canonical
        # pTrdSymbol reconciliation cannot silently disappear.
        rows = []
        attempts = [
            (str(underlying).upper(), ""),
            (str(underlying).upper(), str(expiry or "")),
        ]
        seen_attempts = set()
        for symbol_query, expiry_query in attempts:
            key = (symbol_query, expiry_query)
            if key in seen_attempts:
                continue
            seen_attempts.add(key)
            try:
                result = self.client.search_scrip(
                    exchange_segment=exchange,
                    symbol=symbol_query,
                    expiry=expiry_query,
                    option_type="",
                    strike_price="",
                )
            except Exception:
                continue
            if isinstance(result, list):
                rows.extend(result)
            if rows:
                break
        if not rows:
            return {}
        mapping: dict[str, str] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            token = str(row.get("pSymbol") or "").strip()
            symbol = str(row.get("pTrdSymbol") or "").strip()
            if token and symbol:
                mapping[token] = symbol
            # Neo option-chain instruments may expose neoSymbol as
            # "nse_fo|<pSymbol>" while scrip-master pSymbol is bare. Keep both
            # forms so the canonical broker trading symbol is always usable.
            if "|" in token:
                mapping[token.split("|")[-1]] = symbol
            else:
                mapping[f"{exchange}|{token}"] = symbol
        return mapping

    def get_option_chain(
        self,
        underlying: str,
        exchange: str,
        expiry: str | None = None,
        count: int = 40,
        enrich_quotes: bool = False,
    ) -> List[OptionContract]:
        exchange_segment = self.normalize_exchange(exchange)
        if exchange_segment not in {"nse_fo", "bse_fo", "mcx_fo"}:
            raise ValueError(
                f"Unsupported option-chain exchange '{exchange}'. "
                "Use nse_fo, bse_fo or mcx_fo."
            )

        if count < 10 or count % 10 != 0:
            raise ValueError("Kotak Neo option-chain count must be a multiple of 10.")

        # Kotak's current SDK/API supports omitting expiry and will resolve
        # the nearest available expiry server-side. Prefer that path for live
        # polling because it avoids an unnecessary expiry API dependency and
        # avoids rejecting a valid chain when the expiry endpoint is temporarily
        # unavailable. If the caller supplied an expiry, preserve it exactly.
        request_expiry = str(expiry).strip() if expiry else None
        # Prefer an explicit broker-provided expiry for live requests. This
        # avoids relying on server-side nearest-expiry inference and gives us
        # a precise diagnostic when the selected expiry is unavailable.
        if request_expiry is None:
            try:
                request_expiry = self._nearest_expiry(exchange_segment, underlying)
            except Exception:
                # Keep the SDK-documented omitted-expiry path as a fallback
                # when the separate expiry endpoint is temporarily unavailable.
                request_expiry = None
        # MCX option-chain requests must use the broker's canonical
        # pSymbolName (for example CRUDEOIL/CRUDEOILM), not the UI label
        # blindly. This prevents valid MCX option chains from being rejected
        # because of an underlying-name mismatch.
        request_underlying = (
            self.resolve_option_underlying(underlying, exchange_segment)
            if exchange_segment == "mcx_fo"
            else underlying.upper()
        )
        response = self.client.option_chain(
            exchange=exchange_segment,
            underlying=request_underlying,
            expiry=request_expiry,
            instrument_type="option",
            count=count,
        )
        error = self._response_error(response)
        if error:
            raise RuntimeError(
                f"Kotak Neo option-chain error: {error}; "
                f"exchange={exchange_segment} underlying={request_underlying} "
                f"expiry={request_expiry or 'server-default'} count={count}"
            )

        data = self._response_data(response)
        calls = data.get("call") or []
        puts = data.get("put") or []
        if not calls and not puts:
            raise RuntimeError(
                "Kotak Neo returned an empty option chain. "
                "Check the underlying, exchange segment and expiry."
            )

        resolved_expiry = request_expiry
        if not resolved_expiry:
            common_data = data.get("common_data") or {}
            resolved_expiry = common_data.get("expiryDt") or common_data.get("expiry")

        # Current Kotak responses may return calls/puts as dicts keyed by
        # contract rather than lists. Normalize only the container shape here;
        # contract parsing remains unchanged.
        if isinstance(calls, dict):
            calls = list(calls.values())
        if isinstance(puts, dict):
            puts = list(puts.values())

        canonical_symbols = self._scrip_master_option_symbols(
            request_underlying, exchange_segment, resolved_expiry
        )

        contracts: list[OptionContract] = []
        for item in calls:
            contract = self._parse_option(item, underlying, exchange_segment, "CE", resolved_expiry, canonical_symbols)
            if contract:
                contracts.append(contract)
        for item in puts:
            contract = self._parse_option(item, underlying, exchange_segment, "PE", resolved_expiry, canonical_symbols)
            if contract:
                contracts.append(contract)

        # The option-chain endpoint already returns broker LTP/volume/OI and
        # option analytics. Quote enrichment is opt-in because Neo limits a
        # quotes request to 50 symbols and some SDK/backend combinations have
        # produced oversized quote requests despite provider-side batching.
        # Depth enrichment remains available explicitly through enrich_quotes=True.
        if enrich_quotes and contracts:
            self._enrich_quotes(contracts)

        return contracts

    def _parse_option(
        self,
        item: dict,
        underlying: str,
        exchange: str,
        option_type: str,
        requested_expiry: str | None,
        canonical_symbols: dict[str, str] | None = None,
    ) -> OptionContract | None:
        instrument = item.get("instrument") or item.get("inst") or {}
        quote = item.get("quote") or {}
        oi = item.get("openInterest") or item.get("oi") or {}

        symbol = instrument.get("symbol")
        neo_symbol = instrument.get("neoSymbol")
        canonical_symbol = (canonical_symbols or {}).get(str(neo_symbol).strip())
        if not canonical_symbol and "|" in str(neo_symbol):
            canonical_symbol = (canonical_symbols or {}).get(str(neo_symbol).split("|")[-1])
        identity_verified = bool(canonical_symbol)
        if canonical_symbol:
            symbol = canonical_symbol
        if not symbol or not neo_symbol:
            return None

        expiry = (
            instrument.get("expiryDt")
            or instrument.get("expiry")
            or item.get("expiryDt")
            or item.get("expiry")
            or requested_expiry
            or ""
        )
        payload_strike = self._float(
            instrument.get("strikePrice")
            or instrument.get("strike")
            or item.get("strikePrice")
            or item.get("strike")
        )

        # Kotak's compact trading symbol is not safely parseable by simply
        # taking all digits before CE/PE. For example:
        #   NIFTY26O1320250CE
        # contains expiry day 13 + strike 20250, so a suffix regex produces
        # the false strike 1320250. The instrument's strkPrc is the explicit
        # strike field and must win when supplied.
        import re
        symbol_match = re.search(r"(\d+(?:\.\d+)?)(CE|PE)$", str(symbol).upper())
        symbol_strike = (
            self._float(symbol_match.group(1))
            if symbol_match
            else 0.0
        )
        explicit_strike = payload_strike
        if explicit_strike <= 0:
            explicit_strike = self._float(instrument.get("strkPrc") or item.get("strkPrc"))

        # The canonical trading symbol is the final identity of an option
        # contract. If the payload strike disagrees with a valid symbol strike,
        # treat the payload value as stale/malformed rather than exposing a
        # potentially different tradable contract. Never silently rescale.
        symbol_valid = False
        if symbol_strike > 0:
            try:
                validate_option_strike(underlying, symbol_strike)
                symbol_valid = True
            except ValueError:
                pass

        if symbol_valid:
            strike = float(symbol_strike)
        elif explicit_strike > 0:
            strike = validate_option_strike(underlying, explicit_strike)
        else:
            raise ValueError(f"Invalid {underlying} option strike: {explicit_strike}")
        ltp = self._quote_ltp(quote)
        volume = self._float(quote.get("volume") or quote.get("vol"))
        current_oi = self._float(oi.get("current") or oi.get("cur"))
        oi_change = self._float(oi.get("change") or oi.get("chg"))
        implied_volatility = self._float(
            quote.get("iv") or quote.get("impliedVolatility") or item.get("iv")
        )
        delta = self._float(quote.get("delta") or item.get("delta"))
        theta = self._float(quote.get("theta") or item.get("theta"))
        vega = self._float(quote.get("vega") or item.get("vega"))
        gamma = self._float(quote.get("gamma") or item.get("gamma"))
        ltp_change_pct = self._float(
            quote.get("changePct") or quote.get("ltpChangePct") or item.get("changePct")
        )
        built_up = str(quote.get("builtUp") or item.get("builtUp") or "").strip()

        return OptionContract(
            symbol=str(symbol),
            exchange=exchange,
            underlying=underlying.upper(),
            expiry=str(expiry),
            strike=strike,
            option_type=option_type,
            instrument_token=str(neo_symbol),
            ltp=ltp,
            bid=None,
            ask=None,
            volume=volume,
            open_interest=current_oi,
            oi_change=oi_change,
            implied_volatility=implied_volatility,
            built_up=built_up,
            delta=delta,
            theta=theta,
            vega=vega,
            gamma=gamma,
            ltp_change_pct=ltp_change_pct,
            identity_verified=identity_verified,
            scrip_master_hash=__import__("hashlib").sha256(str(canonical_symbol or "").encode("utf-8")).hexdigest(),
        )

    def _enrich_quotes(self, contracts: list[OptionContract]) -> None:
        """Populate LTP/depth fields while respecting Neo quote symbol limits."""
        def fetch_batch(batch: list[OptionContract]) -> list[dict]:
            if not batch:
                return []
            tokens = []
            for contract in batch:
                token = contract.instrument_token or ""
                if "|" in token:
                    segment, instrument_token = token.split("|", 1)
                else:
                    segment, instrument_token = contract.exchange, token
                if instrument_token:
                    tokens.append({"instrument_token": instrument_token, "exchange_segment": segment})
            if not tokens:
                return []
            try:
                response = self.client.quotes(instrument_tokens=tokens[:50], quote_type="all")
            except Exception:
                if len(batch) <= 1:
                    return []
                midpoint = max(1, len(batch) // 2)
                return fetch_batch(batch[:midpoint]) + fetch_batch(batch[midpoint:])
            if not isinstance(response, list):
                data = self._response_data(response)
                response = data.get("quotes") or data.get("data") or []
            return response if isinstance(response, list) else []

        for start in range(0, len(contracts), 50):
            batch = contracts[start:start + 50]
            response = fetch_batch(batch)
            by_token = {}
            for quote in response:
                if not isinstance(quote, dict):
                    continue
                token = str(quote.get("exchange_token") or quote.get("instrument_token") or "")
                if token:
                    by_token[token] = quote
            for contract in batch:
                token = (contract.instrument_token or "").split("|")[-1]
                quote = by_token.get(token)
                if not quote:
                    continue
                ltp = self._quote_ltp(quote, contract.ltp or 0.0)
                quote_timestamp = None
                for _key in ("timestamp", "quoteTimestamp", "quote_timestamp", "lastTradeTime", "last_traded_time"):
                    _raw_ts = quote.get(_key)
                    if _raw_ts not in (None, ""):
                        try:
                            quote_timestamp = datetime.fromisoformat(str(_raw_ts).replace("Z", "+00:00"))
                        except (TypeError, ValueError):
                            try:
                                quote_timestamp = datetime.fromtimestamp(float(_raw_ts))
                            except (TypeError, ValueError, OSError):
                                quote_timestamp = None
                        if quote_timestamp is not None:
                            break
                depth = quote.get("depth") or {}
                buy = depth.get("buy") or depth.get("buyDepth") or []
                sell = depth.get("sell") or depth.get("sellDepth") or []
                bid = self._float(buy[0].get("price")) if buy and isinstance(buy[0], dict) else None
                ask = self._float(sell[0].get("price")) if sell and isinstance(sell[0], dict) else None
                index = contracts.index(contract)
                contracts[index] = OptionContract(
                    symbol=contract.symbol, exchange=contract.exchange, underlying=contract.underlying,
                    expiry=contract.expiry, strike=contract.strike, option_type=contract.option_type,
                    instrument_token=contract.instrument_token, ltp=ltp, bid=bid, ask=ask,
                    volume=contract.volume, open_interest=contract.open_interest, oi_change=contract.oi_change,
                    implied_volatility=contract.implied_volatility, built_up=contract.built_up,
                    delta=contract.delta, theta=contract.theta, vega=contract.vega, gamma=contract.gamma,
                    ltp_change_pct=contract.ltp_change_pct,
                    quote_timestamp=quote_timestamp,
                    quote_source="KOTAK_NEO",
                    identity_verified=contract.identity_verified,
                    scrip_master_hash=contract.scrip_master_hash,
                    lot_size=contract.lot_size,
                    tick_size=contract.tick_size,
                )
    def resolve_mcx_futures(self, symbol: str) -> dict:
        """Resolve the nearest tradable MCX futures contract from Neo scrip master.

        Tokens are never hardcoded because Kotak refreshes the scrip master daily.
        """
        requested = str(symbol or "").strip().upper()
        if not requested:
            raise ValueError("MCX symbol cannot be empty.")

        rows = self.client.search_scrip(
            exchange_segment="mcx_fo",
            symbol=requested,
            expiry="",
            option_type="FUT",
            strike_price="",
        )
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(f"No MCX futures contract found for '{requested}'.")

        from datetime import datetime
        candidates = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            token = str(row.get("pSymbol") or "").strip()
            segment = str(row.get("pExchSeg") or "mcx_fo").strip().lower()
            trading_symbol = str(row.get("pTrdSymbol") or "").strip()
            name = str(row.get("pSymbolName") or requested).strip().upper()
            option_type = str(row.get("pOptionType") or "").upper()
            expiry_raw = row.get("pExpiryDate")
            if not token or segment != "mcx_fo" or option_type not in {"", "XX", "FUT"}:
                continue
            expiry = None
            if expiry_raw:
                try:
                    expiry = datetime.strptime(str(expiry_raw), "%d%b%Y")
                except ValueError:
                    try:
                        expiry = datetime.fromisoformat(str(expiry_raw)).replace(tzinfo=None)
                    except ValueError:
                        expiry = None
            # Never select an already-expired futures contract. If Neo cannot
            # parse an expiry, keep it only as a last-resort candidate after
            # dated contracts so a valid current contract always wins.
            if expiry is not None and expiry.date() < datetime.now().date():
                continue
            candidates.append((expiry or datetime.max, token, trading_symbol, name, row))

        if not candidates:
            raise RuntimeError(
                f"MCX scrip master returned no current/future futures contract for '{requested}'."
            )
        candidates.sort(key=lambda item: item[0])
        expiry, token, trading_symbol, name, row = candidates[0]
        return {
            "symbol": name,
            "trading_symbol": trading_symbol,
            "neosymbol": f"mcx_fo|{token}",
            "instrument_token": token,
            "exchange_segment": "mcx_fo",
            "expiry": None if expiry == datetime.max else expiry.date().isoformat(),
            "lot_size": row.get("lLotSize") or row.get("iLotSize"),
        }

    def get_mcx_quote(self, contract: dict) -> dict:
        """Fetch one current MCX futures snapshot using a resolved Neo contract."""
        token = str(contract.get("instrument_token") or "").strip()
        if not token:
            raise ValueError("Resolved MCX contract has no instrument token.")
        response = self.client.quotes(
            instrument_tokens=[{"instrument_token": token, "exchange_segment": "mcx_fo"}],
            quote_type="all",
        )
        if not isinstance(response, list) or not response:
            data = self._response_data(response)
            response = data.get("quotes") or data.get("data") or []
        if not response or not isinstance(response[0], dict):
            raise RuntimeError(f"Kotak Neo returned no quote for {contract.get('trading_symbol') or token}.")
        row = response[0]
        ohlc = row.get("ohlc") or {}
        raw_ts = row.get("timestamp") or row.get("quoteTimestamp") or row.get("lastTradeTime") or row.get("last_traded_time")
        return {
            "open": self._float(ohlc.get("open")),
            "high": self._float(ohlc.get("high")),
            "low": self._float(ohlc.get("low")),
            "close": self._float(ohlc.get("close") or row.get("ltp")),
            "volume": self._float(row.get("last_volume") or row.get("volume")),
            "open_interest": self._float(row.get("open_int") or row.get("openInterest")),
            "ltp": self._float(row.get("ltp")),
            "timestamp": raw_ts,
            "source": "KOTAK_NEO",
            "instrument_token": token,
        }

    def get_historical_candles(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        start: date,
        end: date,
        neosymbol: str | None = None,
    ) -> List[Candle]:
        # Kotak's historical endpoint accepts exchange-segment|instrument-token
        # for tradable instruments such as NSE equities, but the live NIFTY index
        # instrument (nse_cm|26000) is rejected by the historical backend with
        # HTTP 422 "Invalid neosymbol". Do not rotate/substitute the index token,
        # fabricate candles, or silently fall back to Yahoo. NIFTY 5-minute
        # decision candles must come from the Kotak live market-data capture.
        is_nifty_index = symbol.lower().replace(" ", "") in {"nifty50", "nifty"}
        if is_nifty_index:
            raise RuntimeError(
                "Kotak Neo historical API does not support NIFTY index candles "
                "(nse_cm|26000). Use the Kotak live market-data capture for "
                "NIFTY decision candles; no synthetic/Yahoo fallback is allowed."
            )
        neosymbol = neosymbol or symbol

        if not neosymbol:
            raise RuntimeError(
                f"Kotak Neo historical-data instrument is not configured for '{symbol}'."
            )
        # The UI uses friendly timeframe names, while Neo's historical API
        # accepts only its documented interval tokens.
        interval_map = {
            "1m": "1min",
            "1min": "1min",
            "3m": "3min",
            "3min": "3min",
            "5m": "5min",
            "5min": "5min",
            "10m": "10min",
            "10min": "10min",
            "15m": "15min",
            "15min": "15min",
            "30m": "30min",
            "30min": "30min",
            "60m": "60min",
            "60min": "60min",
            "1h": "60min",
            "1d": "D",
            "D": "D",
            "day": "D",
            "1w": "W",
            "W": "W",
        }
        request_interval = interval_map.get(str(timeframe).strip(), str(timeframe).strip())
        response = self.client.historical_data(
            neosymbol=neosymbol,
            interval=request_interval,
            from_date=start.isoformat(),
            to_date=end.isoformat(),
        )
        error = self._response_error(response)
        if error:
            raise RuntimeError(f"Kotak Neo historical-data error: {error}")

        candles = []
        data = self._response_data(response)
        for row in data.get("candles", []) or []:
            if len(row) < 6:
                continue
            timestamp = datetime.fromisoformat(str(row[0]).replace("+0530", "+05:30"))
            candles.append(
                Candle(
                    timestamp=timestamp,
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    open=self._float(row[1]),
                    high=self._float(row[2]),
                    low=self._float(row[3]),
                    close=self._float(row[4]),
                    volume=self._float(row[5]),
                    open_interest=self._float(row[6]) if len(row) > 6 else None,
                )
            )
        return candles
