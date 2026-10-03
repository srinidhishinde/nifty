from strategy.ce_pe_selector import select_direction


def test_conflicting_signal_returns_wait():

    result = select_direction(
        technical_ce_score=50,
        technical_pe_score=50,
        option_ce_score=50,
        option_pe_score=50,
        news_ce_score=50,
        news_pe_score=50,
        ml_ce_probability=0.50,
        ml_pe_probability=0.50,
    )

    assert result[0] == "WAIT"
