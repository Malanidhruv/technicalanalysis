"""
Fetch CompanyFundamentals from Screener.in (anonymous HTML scrape) with 24h SQLite cache.

Approach inspired by BuildAlgos/screener-scraper (table section IDs), but kept as one
small self-contained module — no third-party scraper dependency.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from fundamental_score import CompanyFundamentals

_ROOT = Path(__file__).resolve().parent
_CACHE_DB = _ROOT / "data" / "fundamentals_cache.sqlite"
_CACHE_VERSION = 2  # bump when CompanyFundamentals schema changes
_TTL = timedelta(hours=24)
_DELAY_SEC = 2.0
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
_HEADERS = {
    "User-Agent": _UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_last_fetch_ts = 0.0


def _ensure_db() -> sqlite3.Connection:
    _CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_CACHE_DB))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fundamentals (
            symbol TEXT PRIMARY KEY,
            fetched_at TEXT NOT NULL,
            statement_type TEXT,
            cache_version INTEGER,
            payload TEXT NOT NULL
        )
        """
    )
    # Older DBs may lack cache_version — add if missing
    cols = {r[1] for r in conn.execute("PRAGMA table_info(fundamentals)")}
    if "cache_version" not in cols:
        conn.execute("ALTER TABLE fundamentals ADD COLUMN cache_version INTEGER")
    conn.commit()
    return conn


def _cache_get(symbol: str) -> Optional[CompanyFundamentals]:
    conn = _ensure_db()
    try:
        row = conn.execute(
            "SELECT fetched_at, payload, cache_version FROM fundamentals WHERE symbol = ?",
            (symbol.upper(),),
        ).fetchone()
        if not row:
            return None
        fetched_at = datetime.fromisoformat(row[0])
        if datetime.now() - fetched_at > _TTL:
            return None
        if row[2] != _CACHE_VERSION:
            return None
        data = json.loads(row[1])
        return CompanyFundamentals(**{
            k: v for k, v in data.items()
            if k in CompanyFundamentals.__dataclass_fields__
        })
    finally:
        conn.close()


def _cache_put(fund: CompanyFundamentals) -> None:
    conn = _ensure_db()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO fundamentals(
                symbol, fetched_at, statement_type, cache_version, payload
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                fund.name.upper(),
                datetime.now().isoformat(timespec="seconds"),
                fund.statement_type,
                _CACHE_VERSION,
                json.dumps(fund.to_dict()),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _throttle() -> None:
    global _last_fetch_ts
    elapsed = time.time() - _last_fetch_ts
    if elapsed < _DELAY_SEC:
        time.sleep(_DELAY_SEC - elapsed)
    _last_fetch_ts = time.time()


def _parse_number(text: str) -> Optional[float]:
    if text is None:
        return None
    t = str(text).strip().replace(",", "").replace("%", "")
    if t in ("", "-", "—", "NA", "N/A"):
        return None
    t = re.sub(r"[^\d.\-]", "", t)
    if t in ("", "-", "."):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _top_ratios(soup: BeautifulSoup) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for li in soup.select("#top-ratios li, li.flex.flex-space-between"):
        name_el = li.select_one(".name") or li.select_one("span.name")
        val_el = li.select_one(".value") or li.select_one("span.value") or li.select_one("span.number")
        if not name_el or not val_el:
            texts = [x.strip() for x in li.stripped_strings]
            if len(texts) < 2:
                continue
            key, raw = texts[0].lower(), texts[-1]
        else:
            key = name_el.get_text(" ", strip=True).lower()
            raw = val_el.get_text(" ", strip=True)
        num = _parse_number(raw)
        if num is None:
            continue
        out[key] = num
    return out


def _table_rows(soup: BeautifulSoup, section_id: str) -> Dict[str, List[Optional[float]]]:
    section = soup.find(id=section_id)
    if not section:
        return {}
    table = section.find("table")
    if not table:
        return {}
    rows: Dict[str, List[Optional[float]]] = {}
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 2:
            continue
        label = cells[0].get_text(" ", strip=True)
        label_key = re.sub(r"\s+", " ", label).strip().lower()
        if not label_key:
            continue
        if label_key.startswith("sep ") or re.match(r"^[a-z]{3} \d{4}$", label_key):
            continue
        vals = [_parse_number(c.get_text(" ", strip=True)) for c in cells[1:]]
        rows[label_key] = vals
    return rows


def _last_two(vals: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    nums = [v for v in vals if v is not None]
    if not nums:
        return None, None
    if len(nums) == 1:
        return nums[-1], None
    return nums[-1], nums[-2]


def _find_row(rows: Dict[str, List[Optional[float]]], *needles: str) -> List[Optional[float]]:
    for key, vals in rows.items():
        for n in needles:
            if n in key:
                return vals
    return []


def _growth_pct(curr: Optional[float], prev: Optional[float]) -> Optional[float]:
    if curr is None or prev is None or prev == 0:
        return None
    return ((curr - prev) / abs(prev)) * 100.0


def _parse_peers_avg_pe(soup: BeautifulSoup) -> Optional[float]:
    peers = soup.find(id="peers")
    if not peers:
        return None
    table = peers.find("table")
    if not table:
        return None
    header_cells = table.find_all("th")
    pe_idx = None
    for i, th in enumerate(header_cells):
        if "pe" in th.get_text(" ", strip=True).lower().replace(" ", ""):
            pe_idx = i
            break
    if pe_idx is None:
        return None
    pes = []
    for tr in table.find_all("tr")[1:]:
        tds = tr.find_all("td")
        if len(tds) <= pe_idx:
            continue
        v = _parse_number(tds[pe_idx].get_text(" ", strip=True))
        if v is not None and v > 0:
            pes.append(v)
    if not pes:
        return None
    return sum(pes) / len(pes)


def _parse_shareholding(soup: BeautifulSoup) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    for sid in ("quarterly-shp", "yearly-shp", "shareholding"):
        rows = _table_rows(soup, sid)
        if not rows:
            continue
        prom = _find_row(rows, "promoter")
        pledge = _find_row(rows, "pledge")
        curr, prev = _last_two(prom)
        pledge_curr, _ = _last_two(pledge)
        return curr, prev, pledge_curr
    return None, None, None


def _trading_days_between(start: datetime, end: datetime) -> int:
    """Approx trading days (Mon–Fri), excluding calendar weekends only."""
    if start > end:
        start, end = end, start
    days = 0
    cur = start.date()
    last = end.date()
    while cur < last:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            days += 1
    return days


def _days_since_last_result(soup: BeautifulSoup) -> Optional[int]:
    """
    Best-effort: latest quarterly column header (Mon YYYY) or 'Latest results' text.
    Leave None if not parseable — scorer treats that as neutral.
    """
    # 1) Newest non-TTM header in #quarters table
    section = soup.find(id="quarters")
    if section:
        table = section.find("table")
        if table:
            headers = [th.get_text(" ", strip=True) for th in table.find_all("th")]
            for h in reversed(headers):
                if not h or h.upper() == "TTM":
                    continue
                try:
                    dt = datetime.strptime(h, "%b %Y")
                    # Use month-end as proxy for result period end
                    if dt.month == 12:
                        end = datetime(dt.year, 12, 31)
                    else:
                        end = datetime(dt.year, dt.month + 1, 1) - timedelta(days=1)
                    return _trading_days_between(end, datetime.now())
                except ValueError:
                    continue

    # 2) Visible "result" / "results" date somewhere on the page
    text = soup.get_text(" ", strip=True)
    m = re.search(
        r"(?:Latest results?|Results?)\s*(?:in|:)?\s*([A-Za-z]{3}\s+\d{4}|\d{1,2}\s+[A-Za-z]{3}\s+\d{4})",
        text,
        re.I,
    )
    if m:
        raw = m.group(1)
        for fmt in ("%b %Y", "%d %b %Y"):
            try:
                dt = datetime.strptime(raw, fmt)
                return _trading_days_between(dt, datetime.now())
            except ValueError:
                continue
    return None


def _soup_from_url(url: str) -> Optional[BeautifulSoup]:
    _throttle()
    resp = requests.get(url, headers=_HEADERS, timeout=25)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def _build_fundamentals(symbol: str, soup: BeautifulSoup, statement_type: str) -> CompanyFundamentals:
    top = _top_ratios(soup)
    pnl = _table_rows(soup, "profit-loss")       # annual
    quarters = _table_rows(soup, "quarters")     # quarterly
    ratios = _table_rows(soup, "ratios")
    cash = _table_rows(soup, "cash-flow")
    bal = _table_rows(soup, "balance-sheet")

    # YoY from annual P&L (latest two years)
    sales_a = _find_row(pnl, "sales", "revenue")
    sales_yoy_c, sales_yoy_p = _last_two(sales_a)
    eps_a = _find_row(pnl, "eps in rs", "eps")
    eps_yoy_c, eps_yoy_p = _last_two(eps_a)

    # QoQ from quarterly table (latest two quarters) — leave None if unavailable
    sales_q = _find_row(quarters, "sales", "revenue")
    sales_q_c, sales_q_p = _last_two(sales_q)
    eps_q = _find_row(quarters, "eps in rs", "eps")
    eps_q_c, eps_q_p = _last_two(eps_q)

    # Prefer quarterly OPM for sequential margin; fall back to annual
    opm_q = _find_row(quarters, "opm %", "operating profit margin")
    opm_a = _find_row(pnl, "opm %", "operating profit margin")
    if opm_q:
        opm_c, opm_p = _last_two(opm_q)
    elif opm_a:
        opm_c, opm_p = _last_two(opm_a)
    else:
        op = _find_row(pnl, "operating profit")
        op_c, op_p = _last_two(op)
        opm_c = (op_c / sales_yoy_c * 100.0) if op_c is not None and sales_yoy_c else None
        opm_p = (op_p / sales_yoy_p * 100.0) if op_p is not None and sales_yoy_p else None

    np_row = _find_row(pnl, "net profit", "pat")
    pat_curr, _ = _last_two(np_row)

    cfo_row = _find_row(cash, "cash from operating", "operating activity")
    cfo_curr, _ = _last_two(cfo_row)

    recv = _find_row(ratios, "debtors", "receivable", "trade receivables")
    inv = _find_row(ratios, "inventory days", "inventory")
    recv_c, recv_p = _last_two(recv)
    inv_c, inv_p = _last_two(inv)

    equity = _find_row(bal, "equity capital", "equity")
    eq_c, eq_p = _last_two(equity)

    prom_c, prom_p, pledge = _parse_shareholding(soup)

    def top_get(*keys: str) -> Optional[float]:
        for k, v in top.items():
            for needle in keys:
                if needle in k:
                    return v
        return None

    roe = top_get("roe") or _last_two(_find_row(ratios, "roe"))[0]
    roce = top_get("roce") or _last_two(_find_row(ratios, "roce"))[0]
    de = top_get("debt to equity", "d/e") or _last_two(
        _find_row(ratios, "debt to equity", "debt/equity")
    )[0]
    pe = top_get("stock p/e", "p/e")
    pe_5y = top_get("5 years", "median p/e", "5 year")

    return CompanyFundamentals(
        name=symbol.upper(),
        revenue_growth_yoy=_growth_pct(sales_yoy_c, sales_yoy_p),
        revenue_growth_qoq=_growth_pct(sales_q_c, sales_q_p),
        eps_growth_yoy=_growth_pct(eps_yoy_c, eps_yoy_p),
        eps_growth_qoq=_growth_pct(eps_q_c, eps_q_p),
        operating_margin_curr=opm_c,
        operating_margin_prev=opm_p,
        roe=roe,
        roce=roce,
        debt_to_equity=de,
        interest_coverage=_last_two(
            _find_row(ratios, "interest coverage", "interest covered")
        )[0],
        cfo=cfo_curr,
        pat=pat_curr,
        receivable_days_curr=recv_c,
        receivable_days_prev=recv_p,
        inventory_days_curr=inv_c,
        inventory_days_prev=inv_p,
        promoter_pledge_pct=pledge,
        promoter_holding_curr=prom_c,
        promoter_holding_prev=prom_p,
        equity_dilution_yoy=_growth_pct(eq_c, eq_p),
        pe_curr=pe,
        pe_5y_avg=pe_5y,
        peer_avg_pe=_parse_peers_avg_pe(soup),
        days_since_last_result=_days_since_last_result(soup),
        statement_type=statement_type,
    )


def fetch_fundamentals_for_symbol(symbol: str, use_cache: bool = True) -> Optional[CompanyFundamentals]:
    """
    Scrape Screener.in for symbol. Prefer consolidated; fall back to standalone.
    Returns None if the symbol cannot be fetched at all.
    Incomplete fields stay as None (scored neutrally upstream).
    """
    sym = symbol.strip().upper()
    if not sym:
        return None

    if use_cache:
        cached = _cache_get(sym)
        if cached is not None:
            return cached

    urls = [
        (f"https://www.screener.in/company/{sym}/consolidated/", "consolidated"),
        (f"https://www.screener.in/company/{sym}/", "standalone"),
    ]

    last_err: Optional[Exception] = None
    for url, stype in urls:
        try:
            soup = _soup_from_url(url)
            if soup is None:
                continue
            title = (soup.title.get_text(" ", strip=True) if soup.title else "").lower()
            if "page not found" in title or "404" in title:
                continue
            if not soup.find(id="profit-loss") and not soup.select("#top-ratios li"):
                continue
            fund = _build_fundamentals(sym, soup, stype)
            _cache_put(fund)
            return fund
        except Exception as exc:
            last_err = exc
            continue

    if last_err:
        print(f"Fundamentals fetch failed for {sym}: {last_err}")
    return None


def fetch_fundamentals_many(
    symbols: List[str],
    progress_cb=None,
) -> Dict[str, CompanyFundamentals]:
    """Fetch many symbols; skip failures. progress_cb(i, n, symbol, status)."""
    out: Dict[str, CompanyFundamentals] = {}
    n = len(symbols)
    for i, sym in enumerate(symbols, 1):
        try:
            cached = _cache_get(sym)
            if cached is not None:
                out[sym.upper()] = cached
                if progress_cb:
                    progress_cb(i, n, sym, "cache")
                continue
            fund = fetch_fundamentals_for_symbol(sym, use_cache=False)
            if fund is not None:
                out[sym.upper()] = fund
                if progress_cb:
                    progress_cb(i, n, sym, fund.statement_type)
            elif progress_cb:
                progress_cb(i, n, sym, "failed")
        except Exception as exc:
            print(f"Skip {sym}: {exc}")
            if progress_cb:
                progress_cb(i, n, sym, "error")
    return out
