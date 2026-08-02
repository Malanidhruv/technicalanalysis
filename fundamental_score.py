"""
Fundamental scoring engine for swing-trade candidates.

Purpose
-------
Produces ONE 0-100 fundamental score per company, meant to sit as a single
extra column next to your technical `Strength` score.

Design philosophy for SWING trading (days to weeks), not long-term investing:
  - Recent-quarter (QoQ) momentum matters MORE than trailing YoY/ROE, because
    it's what's actually driving price near-term.
  - Balance sheet leverage and cash-flow-vs-PAT mismatches are QUALITY GATES
    (hard_filters), not scored drivers - they change slowly and rarely
    explain why a stock is breaking out today. They still knock a stock out
    if genuinely bad, but they don't compete for weight against this
    quarter's numbers.
  - A "catalyst freshness" signal is included: if a company reported results
    recently, that's often the actual reason for the breakout, and is
    scored explicitly rather than left implicit.

Data source
-----------
This module does NOT fetch data itself - wire your own fetcher (e.g. the
BuildAlgos/screener-scraper stockScreener/ScreenerScrape classes, or the
Screener.in API) to populate a `CompanyFundamentals` object per symbol,
then call `fundamental_score(cf)`.
"""

from dataclasses import asdict, dataclass
from typing import Optional, Tuple, List, Dict


@dataclass
class CompanyFundamentals:
    name: str = ""

    # --- Growth: YoY (trailing context) and QoQ (near-term signal, weighted higher) ---
    revenue_growth_yoy: Optional[float] = None
    revenue_growth_qoq: Optional[float] = None
    eps_growth_yoy: Optional[float] = None
    eps_growth_qoq: Optional[float] = None

    # --- Margins & returns ---
    operating_margin_curr: Optional[float] = None
    operating_margin_prev: Optional[float] = None
    roe: Optional[float] = None
    roce: Optional[float] = None

    # --- Leverage (hard filters only - not scored) ---
    debt_to_equity: Optional[float] = None
    interest_coverage: Optional[float] = None

    # --- Cash quality (hard filter + light score - lagging/annual data) ---
    cfo: Optional[float] = None
    pat: Optional[float] = None
    receivable_days_curr: Optional[float] = None
    receivable_days_prev: Optional[float] = None
    inventory_days_curr: Optional[float] = None
    inventory_days_prev: Optional[float] = None

    # --- Governance ---
    promoter_pledge_pct: Optional[float] = None
    promoter_holding_curr: Optional[float] = None
    promoter_holding_prev: Optional[float] = None
    equity_dilution_yoy: Optional[float] = None

    # --- Valuation ---
    pe_curr: Optional[float] = None
    pe_5y_avg: Optional[float] = None
    peer_avg_pe: Optional[float] = None

    # --- Catalyst freshness: near-term relevance signal ---
    days_since_last_result: Optional[int] = None

    # Fetcher/cache metadata only (not scored)
    statement_type: str = "consolidated"

    def to_dict(self) -> Dict:
        return asdict(self)


def hard_filters(cf: CompanyFundamentals) -> Tuple[bool, List[str]]:
    """Returns (passed, reasons_for_failure)."""
    reasons = []

    if cf.promoter_pledge_pct is not None and cf.promoter_pledge_pct > 5:
        reasons.append(f"Promoter pledge {cf.promoter_pledge_pct:.1f}% > 5%")

    if cf.debt_to_equity is not None and cf.debt_to_equity > 2.0:
        reasons.append(f"D/E {cf.debt_to_equity:.2f} > 2.0")

    if cf.interest_coverage is not None and cf.interest_coverage < 1.5:
        reasons.append(f"Interest coverage {cf.interest_coverage:.2f} < 1.5")

    if cf.cfo is not None and cf.pat is not None and cf.pat > 0 and cf.cfo < 0:
        reasons.append("CFO negative while PAT positive")

    if (cf.promoter_holding_curr is not None and cf.promoter_holding_prev is not None
            and (cf.promoter_holding_prev - cf.promoter_holding_curr) > 5):
        reasons.append("Promoter holding dropped >5pp recently")

    return (len(reasons) == 0), reasons


def _bucket(value, thresholds_and_scores, max_score, label="value"):
    if value is None:
        return max_score / 2, f"{label}: missing -> neutral half-credit"
    for min_val, score in thresholds_and_scores:
        if value >= min_val:
            return score, f"{label}={value:.2f}"
    return 0, f"{label}={value:.2f} below lowest bracket"


#  QoQ growth is on a much smaller natural scale than YoY (a single quarter's
#  sequential move vs a full year's), so each needs its OWN threshold set
#  calibrated for its own scale. Blending the raw percentages against a single
#  YoY-sized threshold systematically under-scores strong QoQ momentum.
_YOY_GROWTH_THRESHOLDS = [(20, 1.0), (10, 0.67), (0, 0.33)]   # score as fraction of max
_QOQ_GROWTH_THRESHOLDS = [(6, 1.0), (3, 0.67), (0, 0.33)]     # ~4x smaller scale than YoY

_GROWTH_BLEND_WEIGHTS = (0.65, 0.35)  # (qoq_weight, yoy_weight)


# Growth % computed off a tiny/near-zero prior-period base (e.g. a loss
# narrowing from -50cr to -2cr) produces mathematically extreme values that
# don't reflect real momentum. Beyond this magnitude, treat the figure as
# unreliable rather than scoring it (usually maxing it out, which is worse).
_GROWTH_SANITY_CAP_PCT = 300


def _growth_component_score(value, thresholds):
    """Score a single growth figure (0.0-1.0) against thresholds sized for its own scale."""
    if value is None:
        return None
    if abs(value) > _GROWTH_SANITY_CAP_PCT:
        return None  # likely distorted by a tiny/negative base - don't trust it
    for min_val, frac in thresholds:
        if value >= min_val:
            return frac
    return 0.0


def _blended_growth_score(yoy, qoq, max_score):
    """
    Score YoY and QoQ SEPARATELY on scales calibrated to each, then blend the
    resulting 0-1 scores (not the raw percentages) using QoQ-weighted blend.
    This is what lets a genuinely strong quarter (e.g. 5%+ QoQ) score well
    even though 5% would look weak on a YoY scale.
    """
    def _reason(value, score):
        if value is None:
            return "missing"
        if score is None:
            return f"{value:.1f}% flagged as distorted (>{_GROWTH_SANITY_CAP_PCT}%, likely tiny base) - excluded"
        return None

    qoq_w, yoy_w = _GROWTH_BLEND_WEIGHTS
    qoq_score = _growth_component_score(qoq, _QOQ_GROWTH_THRESHOLDS)
    yoy_score = _growth_component_score(yoy, _YOY_GROWTH_THRESHOLDS)

    if qoq_score is not None and yoy_score is not None:
        frac = qoq_w * qoq_score + yoy_w * yoy_score
        note = f"QoQ={qoq:.1f}% (scored {qoq_score:.2f}), YoY={yoy:.1f}% (scored {yoy_score:.2f})"
    elif qoq_score is not None:
        frac = qoq_score
        note = f"QoQ={qoq:.1f}% only (scored {qoq_score:.2f}), YoY {_reason(yoy, yoy_score)}"
    elif yoy_score is not None:
        frac = yoy_score
        note = f"YoY={yoy:.1f}% only (scored {yoy_score:.2f}), QoQ {_reason(qoq, qoq_score)}"
    else:
        frac = 0.5
        qoq_r = _reason(qoq, qoq_score) or "missing"
        yoy_r = _reason(yoy, yoy_score) or "missing"
        note = f"QoQ {qoq_r}, YoY {yoy_r} -> neutral half-credit"

    return frac * max_score, note


def score_growth_momentum(cf: CompanyFundamentals) -> Tuple[float, Dict]:
    """0-45: blended (QoQ-weighted) revenue & EPS growth, sequential margin trend, ROE/ROCE."""
    detail = {}

    s, note = _blended_growth_score(cf.revenue_growth_yoy, cf.revenue_growth_qoq, 12)
    detail['revenue_growth'] = (round(s, 1), note)

    s2, note = _blended_growth_score(cf.eps_growth_yoy, cf.eps_growth_qoq, 12)
    detail['eps_growth'] = (round(s2, 1), note)

    margin_delta = None
    if cf.operating_margin_curr is not None and cf.operating_margin_prev is not None:
        margin_delta = cf.operating_margin_curr - cf.operating_margin_prev
    s3, note = _bucket(margin_delta, [(2, 11), (0, 7), (-2, 2)], 11, "sequential margin trend")
    detail['margin_trend'] = (round(s3, 1), note)

    roe_roce = None
    if cf.roe is not None or cf.roce is not None:
        vals = [v for v in [cf.roe, cf.roce] if v is not None]
        roe_roce = sum(vals) / len(vals)
    s4, note = _bucket(roe_roce, [(18, 10), (12, 6), (8, 3)], 10, "ROE/ROCE (trailing context)")
    detail['roe_roce'] = (round(s4, 1), note)

    return s + s2 + s3 + s4, detail


def score_cash_quality(cf: CompanyFundamentals) -> Tuple[float, Dict]:
    """0-15: CFO/PAT ratio, receivable & inventory day trends. Lower weight since
    this is annual/lagging data - mainly here to catch earnings-quality red
    flags the hard filter didn't already catch."""
    detail = {}

    cfo_pat_ratio = None
    if cf.cfo is not None and cf.pat not in (None, 0):
        cfo_pat_ratio = cf.cfo / cf.pat
    s, note = _bucket(cfo_pat_ratio, [(0.9, 6), (0.5, 3)], 6, "CFO/PAT")
    detail['cfo_pat_ratio'] = (round(s, 1), note)

    recv_delta = None
    if cf.receivable_days_curr is not None and cf.receivable_days_prev is not None:
        recv_delta = cf.receivable_days_prev - cf.receivable_days_curr
    s2, note = _bucket(recv_delta, [(0, 5), (-10, 2)], 5, "receivable days trend")
    detail['receivable_days'] = (round(s2, 1), note)

    inv_delta = None
    if cf.inventory_days_curr is not None and cf.inventory_days_prev is not None:
        inv_delta = cf.inventory_days_prev - cf.inventory_days_curr
    s3, note = _bucket(inv_delta, [(0, 4), (-10, 2)], 4, "inventory days trend")
    detail['inventory_days'] = (round(s3, 1), note)

    return s + s2 + s3, detail


def score_governance(cf: CompanyFundamentals) -> Tuple[float, Dict]:
    """0-15: pledge, promoter holding change, dilution. Mostly redundant with hard
    filters at the extremes - this scores the milder cases that don't fail outright."""
    detail = {}

    if cf.promoter_pledge_pct is None:
        pledge_score, note = 4, "pledge: missing -> neutral"
    elif cf.promoter_pledge_pct <= 0:
        pledge_score, note = 6, "pledge=0%"
    elif cf.promoter_pledge_pct <= 3:
        pledge_score, note = 3, f"pledge={cf.promoter_pledge_pct:.1f}%"
    else:
        pledge_score, note = 0, f"pledge={cf.promoter_pledge_pct:.1f}%"
    detail['pledge'] = (pledge_score, note)

    holding_delta = None
    if cf.promoter_holding_curr is not None and cf.promoter_holding_prev is not None:
        holding_delta = cf.promoter_holding_curr - cf.promoter_holding_prev
    s2, note = _bucket(holding_delta, [(0, 5), (-2, 2)], 5, "promoter holding change")
    detail['holding_change'] = (round(s2, 1), note)

    if cf.equity_dilution_yoy is None:
        s3, note = 2, "dilution: missing -> neutral"
    elif cf.equity_dilution_yoy < 5:
        s3, note = 4, f"dilution={cf.equity_dilution_yoy:.1f}%"
    elif cf.equity_dilution_yoy < 15:
        s3, note = 2, f"dilution={cf.equity_dilution_yoy:.1f}%"
    else:
        s3, note = 0, f"dilution={cf.equity_dilution_yoy:.1f}% (high)"
    detail['dilution'] = (s3, note)

    return pledge_score + s2 + s3, detail


def _scored_valuation_ratio(ratio, label, max_score):
    if ratio is None:
        return max_score / 2, f"{label}: missing -> neutral"
    if ratio <= 1.1:
        return max_score, f"{label}: {ratio:.2f}x (cheap/fair)"
    if ratio <= 1.3:
        return max_score * 0.6, f"{label}: {ratio:.2f}x (slightly rich)"
    return max_score * 0.2, f"{label}: {ratio:.2f}x (expensive)"


def score_valuation(cf: CompanyFundamentals) -> Tuple[float, Dict]:
    """0-15: PE vs own 5y average, PE vs peer average. Lower weight for a swing
    trade - you're not holding long enough for a rerating, this mainly guards
    against chasing an already-priced-for-perfection breakout."""
    detail = {}

    rel_hist = None
    if cf.pe_curr is not None and cf.pe_5y_avg not in (None, 0):
        rel_hist = cf.pe_curr / cf.pe_5y_avg
    s, note = _scored_valuation_ratio(rel_hist, "vs 5y avg PE", 7.5)
    detail['vs_history'] = (round(s, 1), note)

    rel_peer = None
    if cf.pe_curr is not None and cf.peer_avg_pe not in (None, 0):
        rel_peer = cf.pe_curr / cf.peer_avg_pe
    s2, note = _scored_valuation_ratio(rel_peer, "vs peer avg PE", 7.5)
    detail['vs_peers'] = (round(s2, 1), note)

    return s + s2, detail


def score_catalyst_freshness(cf: CompanyFundamentals) -> Tuple[float, Dict]:
    """0-10: was there a recent quarterly result that could explain today's breakout?
    A breakout with a fresh result behind it is a different trade than one with
    no fundamental trigger at all - purely technical breakouts aren't penalized,
    just not given this bonus."""
    d = cf.days_since_last_result
    if d is None:
        return 5, {'freshness': (5, "result date unknown -> neutral")}
    if d <= 5:
        s, note = 10, f"result {d} trading days ago - fresh catalyst"
    elif d <= 10:
        s, note = 6, f"result {d} trading days ago - still recent"
    elif d <= 20:
        s, note = 3, f"result {d} trading days ago - aging"
    else:
        s, note = 0, f"result {d} trading days ago - stale, breakout likely purely technical"
    return s, {'freshness': (s, note)}


def fundamental_score(cf: CompanyFundamentals) -> Tuple[float, Dict]:
    """
    Returns (total_score_0_to_100, breakdown_dict).
    Weights (swing-trade oriented): Growth Momentum 45 (QoQ-weighted), Cash
    Quality 15, Governance 15, Valuation 15, Catalyst Freshness 10.
    """
    g, g_detail = score_growth_momentum(cf)
    c, c_detail = score_cash_quality(cf)
    gov, gov_detail = score_governance(cf)
    v, v_detail = score_valuation(cf)
    f, f_detail = score_catalyst_freshness(cf)

    total = g + c + gov + v + f
    breakdown = {
        'growth_momentum': {'score': round(g, 1), 'max': 45, 'detail': g_detail},
        'cash_quality': {'score': round(c, 1), 'max': 15, 'detail': c_detail},
        'governance': {'score': round(gov, 1), 'max': 15, 'detail': gov_detail},
        'valuation': {'score': round(v, 1), 'max': 15, 'detail': v_detail},
        'catalyst_freshness': {'score': round(f, 1), 'max': 10, 'detail': f_detail},
    }
    return round(total, 1), breakdown
