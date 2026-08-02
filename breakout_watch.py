"""
Breakout Watch tab: Stage 1 universe breakout scan + Stage 2 retest signals.

Additive UI only — does not modify Technical Screener or Swing Setup logic.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Tuple

import streamlit as st

from alice_client import get_cached_historical_data
from breakout_scanner import scan_universe_for_breakouts
from fundamental_score import fundamental_score
from fundamentals_fetcher import _cache_get
from retest_scanner import scan_watchlist_for_retests
from swing_setup import _universe
from symbol_lookup import token_to_symbol
from watchlist_store import (
    export_active_csv,
    get_active,
    get_connection,
    import_from_csv,
)


def get_cached_fundamental_score(symbol):
    """Cache-only lookup for Breakout Watch - never scrapes.

    Defined here (not only in fundamentals_fetcher) so a stale Cloud
    mount of fundamentals_fetcher.py can't break this tab's import.
    """
    cf = _cache_get(symbol)
    if cf is None:
        return None, None
    return fundamental_score(cf)

_UNIVERSE_LISTS = ["NIFTY 500", "BSE 500"]

_TIER_STYLE = {
    "ATH": ("#0d7a3e", "#e8f8ef"),
    "52W_HIGH": ("#1a5f9e", "#e8f1fb"),
    "VCP_BASE": ("#6b2d8b", "#f4ebf9"),  # higher-quality base
    "BASE": ("#5c5c5c", "#f0f0f0"),
}


def _tier_badge(tier: str) -> str:
    fg, bg = _TIER_STYLE.get(tier, ("#333", "#eee"))
    label = {"52W_HIGH": "52W", "VCP_BASE": "VCP", "BASE": "BASE", "ATH": "ATH"}.get(tier, tier)
    return (
        f'<span style="background:{bg};color:{fg};border:1px solid {fg};'
        f'padding:2px 8px;border-radius:4px;font-size:0.8rem;font-weight:600;">'
        f"{label}</span>"
    )


def _days_since(iso_date: str) -> int:
    try:
        return (date.today() - date.fromisoformat(iso_date[:10])).days
    except Exception:
        return 0


def _split_universe() -> Tuple[List[Tuple[Any, str]], List[Tuple[Any, str]], Dict[str, Any]]:
    """NSE/BSE (token, symbol) batches + symbol->token map for retest stage."""
    universe = _universe(_UNIVERSE_LISTS)
    nse: List[Tuple[Any, str]] = []
    bse: List[Tuple[Any, str]] = []
    exchange_map: Dict[str, Any] = {}
    for token, exchange in universe:
        sym = str(token_to_symbol(token, exchange)).upper()
        if not sym:
            continue
        exchange_map[sym] = token
        pair = (token, sym)
        if exchange == "BSE":
            bse.append(pair)
        else:
            nse.append(pair)
    return nse, bse, exchange_map


def _merge_stats(a: Dict, b: Dict) -> Dict:
    keys = ("scanned", "new_ath", "new_52w", "new_vcp", "new_base", "errors")
    return {k: int(a.get(k, 0)) + int(b.get(k, 0)) for k in keys}


def _render_signal_card(sig: Dict[str, Any]) -> None:
    name = sig.get("Name") or "?"
    tier = sig.get("Tier") or ""
    fund = sig.get("Fundamental_Score")

    head_l, head_r = st.columns([4, 1])
    with head_l:
        st.markdown(
            f"### {name} {_tier_badge(tier)}",
            unsafe_allow_html=True,
        )
    with head_r:
        if fund is not None:
            st.metric("Fund", f"{float(fund):.0f}")
        else:
            st.caption("Fund —")

    st.caption(
        f"Breakout ₹{sig.get('Breakout_Level')} · "
        f"{sig.get('Breakout_Date')} · "
        f"Exchange {sig.get('Exchange') or '—'}"
    )

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Entry", f"₹{sig.get('entry')}")
    e2.metric("Stop Loss", f"₹{sig.get('stop_loss')}")
    e3.metric("Target", f"₹{sig.get('target')}")
    rr = sig.get("rr")
    e4.metric("R:R", f"1:{rr}" if rr is not None else "—")

    if sig.get("stop_note"):
        st.caption(f"Stop: {sig['stop_note']}")
    if sig.get("target_note"):
        st.caption(f"Target: {sig['target_note']}")

    # TODO: place_breakout_retest_order(name, entry, stop_loss, target)
    st.caption("Manual only — enter via your broker terminal. No orders are placed by this app.")


def render_breakout_watch_tab(alice) -> None:
    st.markdown("### Breakout Watch")
    st.markdown(
        "Daily **breakout → retest** scanner over NIFTY 500 ∪ BSE 500. "
        "Stage 1 flags new breakouts onto a watchlist; Stage 2 checks active "
        "names for a held retest and sizes entry / SL / target."
    )
    st.info(
        "Run **once per day** — ideally around **3:15pm** before the close, "
        "or first thing the next morning. Not meant to be clicked repeatedly "
        "through the session."
    )
    st.warning(
        "Streamlit Cloud disk is ephemeral: `data/breakout_watchlist.db` is "
        "wiped on reboot/redeploy. **Export** after each scan and **Import** "
        "after a reboot so multi-day retest tracking is not lost."
    )

    conn = get_connection()

    # --- backup / restore ---
    bx1, bx2 = st.columns(2)
    with bx1:
        csv_text = export_active_csv(conn)
        st.download_button(
            "Export Watchlist (CSV)",
            data=csv_text,
            file_name=f"breakout_watchlist_{date.today().isoformat()}.csv",
            mime="text/csv",
            key="breakout_watch_export",
            use_container_width=True,
        )
    with bx2:
        uploaded = st.file_uploader(
            "Import Watchlist (CSV)",
            type=["csv"],
            key="breakout_watch_import_file",
            help="Restore active rows after a Cloud wipe.",
        )
        if uploaded is not None and st.button(
            "Restore from upload",
            key="breakout_watch_import_btn",
            use_container_width=True,
        ):
            n = import_from_csv(conn, uploaded)
            st.success(f"Restored {n} row(s) into the watchlist.")
            st.rerun()

    st.markdown("---")

    if st.button(
        "Run Daily Scan",
        type="primary",
        key="breakout_watch_run",
        use_container_width=True,
    ):
        nse_pairs, bse_pairs, exchange_map = _split_universe()
        total = len(nse_pairs) + len(bse_pairs)
        progress = st.progress(0.0)
        status = st.empty()

        def _on_progress(done: int, batch_total: int, offset: int):
            # ponytail: progress is approximate across two exchange batches
            overall = offset + done
            progress.progress(min(1.0, overall / total) if total else 1.0)
            status.caption(f"Universe scan: {overall}/{total}")

        empty = {
            "scanned": 0, "new_ath": 0, "new_52w": 0, "new_vcp": 0, "new_base": 0, "errors": 0,
        }
        stats_nse = empty
        stats_bse = empty

        with st.spinner("Stage 1 — scanning universe for breakouts…"):
            if nse_pairs:
                stats_nse = scan_universe_for_breakouts(
                    alice, nse_pairs, "NSE", conn, get_cached_historical_data,
                    on_progress=lambda d, t: _on_progress(d, t, 0),
                )
            if bse_pairs:
                offset = len(nse_pairs)
                stats_bse = scan_universe_for_breakouts(
                    alice, bse_pairs, "BSE", conn, get_cached_historical_data,
                    on_progress=lambda d, t: _on_progress(d, t, offset),
                )

        stats = _merge_stats(stats_nse, stats_bse)
        progress.progress(1.0)
        status.caption("Stage 2 — checking active watchlist for retests…")

        with st.spinner("Stage 2 — retest scan…"):
            signals = scan_watchlist_for_retests(
                alice, conn, exchange_map, get_cached_historical_data,
                fundamental_score_fn=get_cached_fundamental_score,
            )

        new_total = (
            stats["new_ath"] + stats["new_52w"] + stats["new_vcp"] + stats["new_base"]
        )
        summary = (
            f"Scanned {stats['scanned']} stocks. "
            f"{new_total} new breakouts added to watch "
            f"({stats['new_ath']} ATH, {stats['new_52w']} 52W, "
            f"{stats['new_vcp']} VCP base, {stats['new_base']} base). "
            f"{len(signals)} retest signals today."
        )
        st.session_state["breakout_watch_summary"] = summary
        st.session_state["breakout_watch_signals"] = signals
        st.session_state["breakout_watch_stats"] = stats
        st.session_state["breakout_watch_last_run"] = datetime.now().isoformat(timespec="seconds")
        status.empty()
        progress.empty()
        st.success(summary)

    summary = st.session_state.get("breakout_watch_summary")
    last_run = st.session_state.get("breakout_watch_last_run")
    if summary:
        st.markdown(f"**Last scan:** {last_run or '—'}")
        st.info(summary)

    signals = st.session_state.get("breakout_watch_signals") or []
    st.markdown("## Today's Retest Signals")
    if signals:
        for sig in signals:
            with st.container():
                _render_signal_card(sig)
                st.markdown("---")
    else:
        st.caption("No retest signals yet — run the daily scan, or none held today.")

    active = get_active(conn)
    with st.expander(f"Currently Watching ({len(active)})", expanded=False):
        if not active:
            st.caption("Watchlist empty.")
        else:
            for row in sorted(active, key=lambda r: r.get("breakout_date") or ""):
                tier = row.get("tier") or ""
                days = _days_since(row.get("breakout_date") or "")
                st.markdown(
                    f"**{row.get('symbol')}** {_tier_badge(tier)} · "
                    f"₹{row.get('breakout_level')} · "
                    f"{row.get('breakout_date')} ({days}d) · "
                    f"{row.get('exchange')}",
                    unsafe_allow_html=True,
                )
