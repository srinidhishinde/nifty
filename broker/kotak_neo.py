from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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

            login_data = (
                login_response.get(
                    "data",
                    {},
                )
                or {}
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

    def orders_allowed(self) -> bool:
        """
        Deliberately refuses live order submission unless
        the project's explicit safety gate is enabled.
        """

        return settings.live_trading_allowed()

    def place_order(self, *args: Any, **kwargs: Any) -> Any:
        """
        Live order submission is intentionally not enabled yet.

        The application must first complete market-data,
        option-selection, paper-execution and safety testing.
        """

        if not self.orders_allowed():

            raise RuntimeError(
                "Live order submission is disabled. "
                "Enable production live-trading configuration "
                "only after completing validation."
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
