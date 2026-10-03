from datafiles.historical.sample.sample_chain import (
    create_sample_chain,
)

from backtest.contract_engine import (
    ContractReplayEngine,
)

from strategy.ce_pe_selector import (
    CEPESelector,
    MarketContext,
    OptionAnalysis,
)


def test_strategy_components_integrate():

    snapshots = create_sample_chain(
        rows=30,
    )

    first = snapshots[0]

    ce_contract = next(
        option
        for option in first.options
        if option.option_type == "CE"
    )

    pe_contract = next(
        option
        for option in first.options
        if option.option_type == "PE"
    )

    ce = OptionAnalysis(
        option_type="CE",
        strike=ce_contract.strike,
        ltp=ce_contract.option_ltp,
        volume=ce_contract.volume,
        open_interest=ce_contract.open_interest,
        oi_change=ce_contract.oi_change,
        implied_volatility=ce_contract.implied_volatility,
        score=65,
        reasons=("integration-test",),
    )

    pe = OptionAnalysis(
        option_type="PE",
        strike=pe_contract.strike,
        ltp=pe_contract.option_ltp,
        volume=pe_contract.volume,
        open_interest=pe_contract.open_interest,
        oi_change=pe_contract.oi_change,
        implied_volatility=pe_contract.implied_volatility,
        score=40,
        reasons=("integration-test",),
    )

    context = MarketContext(
        timeframe="5m",
        trend="BULLISH",
        momentum="POSITIVE",
        price_vs_vwap="ABOVE",
        volatility_regime="NORMAL",
    )

    selector = CEPESelector()

    signal = selector.generate(
        context=context,
        ce=ce,
        pe=pe,
    )

    assert signal.decision in {
        "CE",
        "PE",
        "WAIT",
    }

    engine = ContractReplayEngine(
        starting_capital=300000,
        max_loss_per_trade=15000,
    )

    result = engine.run(
        snapshots,
        hold_bars=3,
    )

    assert result.starting_capital == 300000

    assert result.max_drawdown >= 0
