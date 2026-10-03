from config.settings import Settings


def test_live_trading_disabled_by_default():

    settings = Settings()

    assert settings.paper_trading is True
    assert settings.live_trading_enabled is False
    assert settings.allow_order_submission is False

    assert settings.live_trading_allowed() is False


def test_uat_environment_cannot_enable_live_trading():

    settings = Settings(
        environment="uat",
        paper_trading=False,
        live_trading_enabled=True,
        allow_order_submission=True,
    )

    assert settings.live_trading_allowed() is False


def test_production_requires_explicit_live_enablement():

    settings = Settings(
        environment="production",
        paper_trading=False,
        live_trading_enabled=True,
        allow_order_submission=True,
    )

    assert settings.live_trading_allowed() is True


def test_production_without_order_submission_is_blocked():

    settings = Settings(
        environment="production",
        paper_trading=False,
        live_trading_enabled=True,
        allow_order_submission=False,
    )

    assert settings.live_trading_allowed() is False


def test_paper_mode_cannot_submit_live_orders():

    settings = Settings(
        environment="production",
        paper_trading=True,
        live_trading_enabled=True,
        allow_order_submission=True,
        kotak_api_key="",
        kotak_api_secret="",
        kotak_access_token="",
    )

    assert settings.paper_trading is True
    assert settings.live_trading_allowed() is False


def test_live_trading_requires_all_gates():

    settings = Settings(
        environment="production",
        paper_trading=False,
        live_trading_enabled=False,
        allow_order_submission=True,
    )

    assert settings.live_trading_allowed() is False
