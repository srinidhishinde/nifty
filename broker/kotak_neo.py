from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from config.settings import settings


@dataclass(frozen=True)
class NeoConnection:
    connected: bool
    message: str
    greeting_name: str = ""
    ucc: str = ""
    base_url: str = ""


class KotakNeoBroker:
    """
    Thin adapter around the official Kotak Neo Python SDK.

    Authentication is deliberately separated from order submission.

    TOTP is supplied at runtime and is never persisted by this class.
    """

    def __init__(self) -> None:

        self.client: Any | None = None

        self.connected = False

        self.greeting_name = ""

        self.ucc = ""

        self.base_url = ""

        self.last_error = ""

    # --------------------------------------------------
    # Client creation
    # --------------------------------------------------

    def create_client(self) -> Any:

        if not settings.neo_consumer_key:
            raise ValueError(
                "NEO_CONSUMER_KEY is not configured"
            )

        from neo_api_client import NeoAPI

        # Retry only connection establishment failures (including transient
        # DNS resolution failures) inside httpx. This does not retry HTTP
        # responses, authentication errors, or read/write failures.
        # Supplying a custom transport means it owns both HTTP/2 negotiation
        # and connection-pool limits.
        limits = httpx.Limits(
            max_connections=20,
            max_keepalive_connections=10,
        )
        transport = httpx.HTTPTransport(
            retries=2,
            http2=True,
            limits=limits,
        )

        # IMPORTANT:
        # Use keyword arguments.
        #
        # The current SDK changed constructor ordering
        # compared with the legacy SDK.
        #
        # Current:
        # NeoAPI(
        #     consumer_key=...,
        #     environment="prod",
        # )
        #
        # See Kotak's current migration documentation.

        self.client = NeoAPI(
            consumer_key=settings.neo_consumer_key,
            environment="prod",
            transport=transport,
        )

        return self.client

    # --------------------------------------------------
    # Authentication
    # --------------------------------------------------

    def authenticate(
        self,
        totp: str,
    ) -> NeoConnection:

        totp = str(totp).strip()

        if not totp.isdigit():
            return self._failure(
                "TOTP must contain digits only."
            )

        if len(totp) != 6:
            return self._failure(
                "TOTP must be exactly 6 digits."
            )

        if not settings.neo_mobile:
            return self._failure(
                "NEO_MOBILE is not configured."
            )

        if not settings.neo_ucc:
            return self._failure(
                "NEO_UCC is not configured."
            )

        if not settings.neo_mpin:
            return self._failure(
                "NEO_MPIN is not configured."
            )

        try:

            client = (
                self.client
                if self.client is not None
                else self.create_client()
            )

            # --------------------------------------------------
            # Step 1: TOTP login
            # --------------------------------------------------

            login_response = client.totp_login(
                mobile_number=settings.neo_mobile,
                ucc=settings.neo_ucc,
                totp=totp,
            )

            if not isinstance(
                login_response,
                dict,
            ):
                login_response = {}

            login_errors = login_response.get("error") if isinstance(login_response, dict) else None
            if login_errors:
                message = "Kotak Neo TOTP login failed."
                if isinstance(login_errors, list) and login_errors:
                    first_error = login_errors[0]
                    if isinstance(first_error, dict):
                        code = str(first_error.get("code", "") or "").strip()
                        api_message = str(first_error.get("message", "") or "").strip()
                        if code == "10506" or "invalid totp" in api_message.lower():
                            message = "Kotak Neo rejected the TOTP (10506: Invalid TOTP). Generate a fresh 6-digit TOTP and submit it before it expires."
                        elif api_message:
                            message = f"Kotak Neo TOTP login failed: {api_message}"
                self.connected = False
                self.last_error = message
                return NeoConnection(connected=False, message=message)

            login_data = (
                login_response.get(
                    "data",
                    {},
                )
                or {}
            )

            if not login_data:
                return self._failure(
                    "Kotak Neo TOTP login did not return a valid authentication session."
                )

            # --------------------------------------------------
            # Step 2: MPIN validation
            # --------------------------------------------------

            validate_response = (
                client.totp_validate(
                    mpin=settings.neo_mpin,
                )
            )

            if not isinstance(
                validate_response,
                dict,
            ):
                validate_response = {}

            validate_data = (
                validate_response.get(
                    "data",
                    {},
                )
                or {}
            )

            status = str(
                validate_data.get(
                    "status",
                    login_data.get(
                        "status",
                        "",
                    ),
                )
            ).lower()

            # Some SDK/API responses expose success
            # through fields rather than only "status".
            #
            # Therefore we don't reject solely because
            # status is absent.

            self.greeting_name = str(
                validate_data.get(
                    "greetingName",
                    login_data.get(
                        "greetingName",
                        "",
                    ),
                )
                or ""
            )

            self.ucc = str(
                validate_data.get(
                    "ucc",
                    login_data.get(
                        "ucc",
                        settings.neo_ucc,
                    ),
                )
                or settings.neo_ucc
            )

            self.base_url = str(
                validate_data.get(
                    "baseUrl",
                    "",
                )
                or ""
            )

            if status == "failed":
                return self._failure(
                    "Kotak Neo authentication failed."
                )

            self.connected = True

            self.last_error = ""

            return NeoConnection(
                connected=True,
                message="Kotak Neo authentication successful.",
                greeting_name=self.greeting_name,
                ucc=self.ucc,
                base_url=self.base_url,
            )

        except Exception as exc:

            return self._failure(
                self._safe_error_message(exc)
            )

    # --------------------------------------------------
    # Connection status
    # --------------------------------------------------

    def connection_status(self) -> NeoConnection:

        if self.connected:

            return NeoConnection(
                connected=True,
                message="Kotak Neo is connected.",
                greeting_name=self.greeting_name,
                ucc=self.ucc,
                base_url=self.base_url,
            )

        return NeoConnection(
            connected=False,
            message=(
                self.last_error
                or "Kotak Neo is not connected."
            ),
        )

    # --------------------------------------------------
    # Logout
    # --------------------------------------------------

    def logout(self) -> None:

        if self.client is None:
            self.connected = False
            return

        try:

            logout = getattr(
                self.client,
                "logout",
                None,
            )

            if callable(logout):
                logout()

        except Exception:
            # Logout failure should never prevent
            # local session cleanup.
            pass

        finally:

            self.connected = False

            self.greeting_name = ""

            self.ucc = ""

            self.base_url = ""

    # --------------------------------------------------
    # Order safety
    # --------------------------------------------------

    def orders_allowed(self, health_report: Any | None = None) -> bool:
        """
        Live order execution is intentionally hard-disabled in this build.

        Live forecasting/data capture is independent of this guard. Even a
        passing health report or permissive environment configuration must not
        grant order permission until a separately reviewed execution release.
        """
        return False

    def place_order(self, *args: Any, **kwargs: Any) -> Any:
        """
        Live order submission is intentionally not enabled yet.

        The application must first complete market-data,
        option-selection, paper-execution and safety testing.
        """

        health_report = kwargs.pop("health_report", None)
        if not self.orders_allowed(health_report):

            raise RuntimeError(
                "Live order submission is hard-disabled in this build. "
                "Forecasting and market-data capture remain available independently."
            )

        raise NotImplementedError(
            "Live Kotak Neo order routing has not been enabled "
            "in this safety-first build."
        )

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    def _failure(
        self,
        message: str,
    ) -> NeoConnection:

        self.connected = False

        self.last_error = message

        return NeoConnection(
            connected=False,
            message=message,
        )

    @staticmethod
    def _safe_error_message(
        exc: Exception,
    ) -> str:

        message = str(exc)

        # Avoid accidentally displaying credential-like
        # values in the Streamlit UI.

        sensitive_names = (
            "consumer_key",
            "mobile_number",
            "mpin",
            "totp",
            "authorization",
            "token",
            "password",
        )

        lowered = message.lower()

        if any(
            name in lowered
            for name in sensitive_names
        ):
            return (
                "Kotak Neo authentication failed. "
                "Check credentials/TOTP and try again."
            )

        return (
            f"Kotak Neo authentication failed: {message}"
        )
