"""Self-check against user's fundamental_score / combined_screener APIs."""
import pandas as pd

from combined_screener import build_watchlist, compute_entry_sl_target
from fundamental_score import CompanyFundamentals, fundamental_score, hard_filters


def main():
    empty = CompanyFundamentals(name="EMPTY")
    score, breakdown = fundamental_score(empty)
    assert 0 < score < 100
    assert set(breakdown) == {
        "growth_momentum", "cash_quality", "governance", "valuation", "catalyst_freshness"
    }

    # Hard filter: high pledge
    bad = CompanyFundamentals(name="BAD", promoter_pledge_pct=12)
    ok, reasons = hard_filters(bad)
    assert not ok and any("pledge" in r.lower() for r in reasons)

    strong = CompanyFundamentals(
        name="STRONG",
        revenue_growth_yoy=18,
        revenue_growth_qoq=22,
        eps_growth_yoy=15,
        eps_growth_qoq=25,
        operating_margin_curr=18,
        operating_margin_prev=15,
        roe=22,
        roce=20,
        debt_to_equity=0.2,
        interest_coverage=8,
        cfo=500,
        pat=400,
        receivable_days_curr=30,
        receivable_days_prev=35,
        inventory_days_curr=40,
        inventory_days_prev=45,
        promoter_pledge_pct=0,
        promoter_holding_curr=55,
        promoter_holding_prev=54,
        equity_dilution_yoy=1,
        pe_curr=18,
        pe_5y_avg=20,
        peer_avg_pe=22,
        days_since_last_result=3,
    )
    assert hard_filters(strong)[0]
    s2, _ = fundamental_score(strong)
    assert s2 > score

    # IDEA-style tiny-base EPS % must be excluded, not maxed out
    from fundamental_score import score_growth_momentum
    distorted = CompanyFundamentals(
        name="IDEAISH",
        revenue_growth_yoy=3.0,
        revenue_growth_qoq=0.1,
        eps_growth_yoy=183.0,
        eps_growth_qoq=1079.0,
    )
    _, gdetail = score_growth_momentum(distorted)
    assert "distorted" in gdetail["eps_growth"][1]
    assert gdetail["eps_growth"][0] == 6.0  # neutral half of 12 when both legs excluded

    df = pd.DataFrame({
        "open": [10 + i * 0.1 for i in range(80)],
        "high": [11 + i * 0.1 for i in range(80)],
        "low": [9 + i * 0.1 for i in range(80)],
        "close": [10 + i * 0.15 for i in range(80)],
        "volume": [1000] * 80,
    })
    plan = compute_entry_sl_target(df)
    assert plan is not None
    assert plan["rr"] in (2.0, 3.0)

    picks, dropped = build_watchlist(
        [
            {"Name": "STRONG", "Strength": 80, "Pattern": "Breakout"},
            {"Name": "MISSING", "Strength": 90, "Pattern": "X"},
            {"Name": "BAD", "Strength": 85, "Pattern": "Y"},
        ],
        {"STRONG": strong, "BAD": bad},
        {"STRONG": df, "BAD": df},
        top_n=2,
    )
    assert len(picks) == 1 and picks[0]["Name"] == "STRONG"
    drop_names = {d[0] for d in dropped}
    assert "MISSING" in drop_names and "BAD" in drop_names
    print("ok: fundamental_score + build_watchlist (user API)")


if __name__ == "__main__":
    main()
