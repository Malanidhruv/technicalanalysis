"""
Swing Setup tab: technical scan (Nifty500 + BSE500) → fundamentals → top 2 watchlist.

Strictly additive — reuses existing analyzers; does not modify their logic.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from advanced_analysis import analyze_stock_advanced
from alice_client import DEFAULT_WORKERS, get_cached_historical_data
from combined_screener import build_watchlist
from fundamentals_fetcher import fetch_fundamentals_many
from stock_analysis import analyze_stock
from stock_lists import STOCK_LISTS
from symbol_lookup import token_to_symbol


def _universe() -> List[Tuple[Any, str]]:
    """(token, exchange) pairs for Nifty 500 + BSE 500, de-duped by symbol (NSE wins)."""
    seen = set()
    out: List[Tuple[Any, str]] = []
    for token in STOCK_LISTS.get("NIFTY 500", []):
        sym = str(token_to_symbol(token, "NSE")).upper()
        if sym in seen:
            continue
        seen.add(sym)
        out.append((token, "NSE"))
    for token in STOCK_LISTS.get("BSE 500", []):
        sym = str(token_to_symbol(token, "BSE")).upper()
        if sym in seen:
            continue
        seen.add(sym)
        out.append((token, "BSE"))
    return out


def _run_technical_scan(alice, max_candidates: int = 10) -> List[Dict[str, Any]]:
    """
    Reuse existing per-stock analyzers (no strategy edits).
    Prefer Price Action Breakout; fall back to Consolidation Breakout per symbol.
    """
    universe = _universe()
    results: List[Dict[str, Any]] = []
    workers = min(DEFAULT_WORKERS, 16)

    def _one(token_ex):
        token, exchange = token_ex
        try:
            r = analyze_stock_advanced(alice, token, "Price Action Breakout", exchange)
            if r:
                r = dict(r)
                r["Name"] = str(r.get("Name", "")).upper()
                r["Exchange"] = exchange
                r["Token"] = token
                r["Strategy_Source"] = "Price Action Breakout"
                r["Lookback_Days"] = 365
                return r
        except Exception as exc:
            print(f"PA error {token}: {exc}")
        try:
            r = analyze_stock(alice, token, "Consolidation Breakout", exchange)
            if r:
                r = dict(r)
                r["Name"] = str(r.get("Name", "")).upper()
                r["Exchange"] = exchange
                r["Token"] = token
                r["Strategy_Source"] = "Consolidation Breakout"
                r["Lookback_Days"] = 420
                return r
        except Exception as exc:
            print(f"CB error {token}: {exc}")
        return None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, item) for item in universe]
        for fut in as_completed(futs):
            try:
                hit = fut.result()
                if hit:
                    results.append(hit)
            except Exception as exc:
                print(f"swing tech worker: {exc}")

    best: Dict[str, Dict[str, Any]] = {}
    for row in results:
        sym = str(row.get("Name", "")).upper()
        if not sym:
            continue
        prev = best.get(sym)
        if prev is None or float(row.get("Strength") or 0) > float(prev.get("Strength") or 0):
            best[sym] = row

    ranked = sorted(best.values(), key=lambda x: float(x.get("Strength") or 0), reverse=True)
    return ranked[:max_candidates]


def _load_price_lookup(alice, candidates: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """Reload OHLC for finalists — same cache key as the analyzer that produced them."""
    lookup: Dict[str, pd.DataFrame] = {}
    to_dt = datetime.now()
    for row in candidates:
        token = row.get("Token")
        exchange = row.get("Exchange") or "NSE"
        symbol = str(row.get("Name", "")).upper()
        days = int(row.get("Lookback_Days") or 365)
        if token is None or not symbol:
            continue
        try:
            _inst, df = get_cached_historical_data(
                alice, token, to_dt - timedelta(days=days), to_dt, "D", exchange
            )
            if df is not None and len(df) > 0:
                lookup[symbol] = df
        except Exception as exc:
            print(f"OHLC reload {symbol}: {exc}")
    return lookup


def _render_pick_card(pick: Dict[str, Any], rank: int) -> None:
    name = pick.get("Name") or pick.get("symbol") or "?"
    st.markdown(f"### #{rank} · {name}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Composite", f"{pick.get('Composite_Score', 0):.1f}")
    c2.metric("Technical", f"{pick.get('Technical_Score', 0):.1f}")
    c3.metric("Fundamental", f"{pick.get('Fundamental_Score', 0):.1f}")
    c4.metric("R:R", f"1:{pick['rr']}" if pick.get("rr") is not None else "—")

    st.caption(
        f"Pattern: {pick.get('Pattern') or '—'} · "
        f"Statement: {pick.get('statement_type') or '—'} · "
        f"Exchange: {pick.get('Exchange') or '—'}"
    )

    e1, e2, e3 = st.columns(3)
    e1.metric("Entry", f"₹{pick['entry']}")
    e2.metric("Stop Loss", f"₹{pick['stop_loss']}")
    e3.metric("Target", f"₹{pick['target']}")

    with st.expander("Fundamental score breakdown", expanded=False):
        buckets = pick.get("Fundamental_Breakdown") or {}
        for bname, info in buckets.items():
            score = info.get("score", 0)
            mx = info.get("max", 100)
            st.markdown(f"**{bname.replace('_', ' ').title()}** — {score}/{mx}")
            detail = info.get("detail") or {}
            for _k, pair in detail.items():
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    st.markdown(f"- {pair[1]} ({pair[0]})")
                else:
                    st.markdown(f"- {pair}")

    # TODO: auto order placement would plug in here later
    # place_swing_order(pick["Name"], pick["entry"], pick["stop_loss"], pick["target"])
    st.caption("Manual only — enter via your broker terminal. No orders are placed by this app.")


def render_swing_setup_tab(alice) -> None:
    st.markdown("### Swing Setup (Technical + Fundamental)")
    st.markdown(
        "Scans **Nifty 500 + BSE 500** with existing **Price Action Breakout** "
        "(fallback: **Consolidation Breakout**), scores fundamentals from Screener.in "
        "(24h disk cache, 2s delay when uncached), and returns **2 manual swing picks**."
    )

    if st.button("🔍 Run Swing Setup", key="swing_setup_run", type="primary", use_container_width=True):
        progress = st.progress(0, text="Starting technical scan…")
        status = st.empty()

        try:
            status.info("Step 1/3 — Technical scan across Nifty 500 + BSE 500…")
            candidates = _run_technical_scan(alice, max_candidates=10)
            progress.progress(35, text=f"Technical: {len(candidates)} candidates")
            st.session_state["swing_setup_candidates"] = candidates

            if not candidates:
                progress.progress(100, text="Done")
                st.warning(
                    "No technical candidates found. Try after market hours when "
                    "AliceBlue history is available."
                )
                return

            status.info(
                f"Step 2/3 — Fundamentals for {len(candidates)} symbols "
                "(cached hits are instant; uncached wait ~2s each)…"
            )
            symbols = [str(c["Name"]).upper() for c in candidates if c.get("Name")]
            prog_box = st.empty()

            def _cb(i, n, sym, state):
                prog_box.caption(f"Fundamentals {i}/{n}: {sym} ({state})")
                progress.progress(35 + int(50 * i / max(n, 1)), text=f"Fundamentals {i}/{n}")

            fundamentals = fetch_fundamentals_many(symbols, progress_cb=_cb)
            st.session_state["swing_setup_fundamentals"] = {
                k: v.to_dict() for k, v in fundamentals.items()
            }

            status.info("Step 3/3 — Cached OHLC + watchlist…")
            price_lookup = _load_price_lookup(alice, candidates)
            picks, dropped = build_watchlist(
                candidates, fundamentals, price_lookup, top_n=2
            )
            watch = {"picks": picks, "dropped": dropped}
            st.session_state["swing_setup_watchlist"] = watch
            progress.progress(100, text="Done")
            status.success(
                f"Watchlist ready — {len(picks)} picks "
                f"({len(dropped)} dropped / {len(candidates)} technical)."
            )
        except Exception as exc:
            st.error(f"Swing Setup failed: {exc}")
            progress.progress(100, text="Failed")
            return

    watch = st.session_state.get("swing_setup_watchlist")
    if not watch:
        st.info("Click **Run Swing Setup** to generate today's 2 manual swing ideas.")
        return

    picks = watch.get("picks") or []
    if picks:
        st.markdown("## Top 2 Swing Picks")
        for i, pick in enumerate(picks, 1):
            with st.container():
                _render_pick_card(pick, i)
                st.markdown("---")
    else:
        st.warning("No picks after fundamental filter — see dropped list below.")

    dropped = watch.get("dropped") or []
    with st.expander(f"Dropped candidates ({len(dropped)})", expanded=False):
        if not dropped:
            st.caption("None")
        else:
            for item in dropped:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    name, reason = item[0], item[1]
                elif isinstance(item, dict):
                    name, reason = item.get("symbol") or item.get("Name"), item.get("reason")
                else:
                    name, reason = str(item), ""
                st.markdown(
                    f"<small><b>{name}</b> — {reason}</small>",
                    unsafe_allow_html=True,
                )
