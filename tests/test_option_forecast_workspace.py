from app.option_forecast_workspace import _should_fetch_selected_expiry


def test_selected_expiry_fetch_is_allowed_after_hours():
    # Market-hours state is intentionally not a gate for retrieving the broker's
    # latest available chain; the workspace still labels after-hours data historical.
    assert _should_fetch_selected_expiry(
        expiry_chain_empty=True,
        connected=True,
        historical_date_selected=False,
        expiry="2026-10-13",
        broker_expiries=["2026-10-13", "2026-10-20"],
    )


def test_selected_expiry_fetch_is_blocked_for_explicit_prior_session():
    assert not _should_fetch_selected_expiry(
        expiry_chain_empty=True,
        connected=True,
        historical_date_selected=True,
        expiry="2026-10-13",
        broker_expiries=["2026-10-13"],
    )


def test_selected_expiry_fetch_requires_connected_broker():
    assert not _should_fetch_selected_expiry(
        expiry_chain_empty=True,
        connected=False,
        historical_date_selected=False,
        expiry="2026-10-13",
        broker_expiries=["2026-10-13"],
    )


def test_selected_expiry_fetch_requires_expiry_reported_by_broker():
    assert not _should_fetch_selected_expiry(
        expiry_chain_empty=True,
        connected=True,
        historical_date_selected=False,
        expiry="2026-10-13",
        broker_expiries=["2026-10-20"],
    )


def test_selected_expiry_fetch_is_skipped_when_snapshot_contains_expiry():
    assert not _should_fetch_selected_expiry(
        expiry_chain_empty=False,
        connected=True,
        historical_date_selected=False,
        expiry="2026-10-13",
        broker_expiries=["2026-10-13"],
    )
