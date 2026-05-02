from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yfinance as yf


START = "2014-01-01"
END = "2024-12-31"
USER_AGENT = "FinTTA academic open-data prototype; contact: local research repo"

SP500_HISTORY_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%2801-17-2026%29.csv"
)
SP500_CURRENT_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500.csv"
SP500_TICKER_DATES_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
CBOE_VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

FRENCH_URLS = {
    "ff5_daily": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "momentum_daily": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip",
}

FRED_SERIES = {
    "FEDFUNDS": "Effective Federal Funds Rate",
    "DGS3MO": "3-month Treasury constant maturity",
    "DGS2": "2-year Treasury constant maturity",
    "DGS10": "10-year Treasury constant maturity",
    "T10Y2Y": "10-year minus 2-year Treasury",
    "T10Y3M": "10-year minus 3-month Treasury",
    "BAA10Y": "Moody's Baa corporate bond minus 10-year Treasury",
    "AAA10Y": "Moody's Aaa corporate bond minus 10-year Treasury",
    "VIXCLS": "Cboe VIX close from FRED",
    "NFCI": "Chicago Fed National Financial Conditions Index",
    "UNRATE": "Unemployment rate",
    "CPIAUCSL": "CPI all urban consumers",
    "INDPRO": "Industrial production",
    "DCOILWTICO": "WTI crude oil",
    "DTWEXBGS": "Trade weighted U.S. dollar index",
}


@dataclass(slots=True)
class SourceRecord:
    name: str
    status: str
    path: str | None
    rows: int | None
    note: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Download no-key public/open data for the FinTTA prototype.")
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--raw-dir", default="data/raw/open")
    parser.add_argument("--normalized-raw-dir", default="data/raw")
    parser.add_argument("--intermediate-dir", default="data/intermediate")
    parser.add_argument("--max-tickers", type=int, default=0, help="Debug limiter; 0 means all PIT S&P tickers.")
    parser.add_argument("--skip-prices", action="store_true")
    parser.add_argument("--skip-finra", action="store_true")
    parser.add_argument("--finra-start", default="2018-08-01", help="FINRA consolidated NMS begins in 2018.")
    parser.add_argument("--finra-end", default=END)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    normalized_dir = Path(args.normalized_raw_dir)
    intermediate_dir = Path(args.intermediate_dir)
    for path in [raw_dir, normalized_dir, intermediate_dir]:
        path.mkdir(parents=True, exist_ok=True)

    records: list[SourceRecord] = []
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    history = download_sp500(session, raw_dir, records)
    universe = build_universe(history, raw_dir, normalized_dir, args.start, args.end, records)
    tickers = sorted(universe["ticker"].dropna().unique())
    if args.max_tickers > 0:
        tickers = tickers[: args.max_tickers]

    if not args.skip_prices:
        download_prices(tickers, normalized_dir, args.start, args.end, records)
    download_vix(session, raw_dir, intermediate_dir, records, args.start, args.end)
    download_fred(session, raw_dir, normalized_dir, records, args.start, args.end)
    download_french(session, raw_dir, normalized_dir, records, args.start, args.end)
    download_sec_ticker_map(session, raw_dir, records)
    if not args.skip_finra:
        download_finra_short_volume(session, raw_dir, normalized_dir, tickers, args.finra_start, args.finra_end, records)

    manifest = {
        "dataset_grade": "open-data-prototype",
        "start": args.start,
        "end": args.end,
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "important_caveats": [
            "This is not CRSP-grade and is not fully survivor-bias-free.",
            "Yahoo/yfinance prices may omit or truncate delisted ticker history.",
            "S&P 500 sector/industry fields come from current public metadata unless a paid PIT source is supplied.",
            "FINRA CNMS short-volume files begin in 2018 and are short-sale volume, not short interest or borrow cost.",
            "SEC ticker mapping is downloaded; full EDGAR company facts are not bulk-downloaded by default because they are large.",
        ],
        "sources": [asdict(r) for r in records],
    }
    manifest_path = raw_dir / "open_data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def download_sp500(session: requests.Session, raw_dir: Path, records: list[SourceRecord]) -> pd.DataFrame:
    history_path = raw_dir / "sp500_history_fja05680.csv"
    current_path = raw_dir / "sp500_current_wikipedia_fja05680.csv"
    ticker_dates_path = raw_dir / "sp500_ticker_start_end_fja05680.csv"
    history = fetch_csv(session, SP500_HISTORY_URL, history_path)
    current = fetch_csv(session, SP500_CURRENT_URL, current_path)
    ticker_dates = fetch_csv(session, SP500_TICKER_DATES_URL, ticker_dates_path)
    records.append(SourceRecord("sp500_history", "downloaded", str(history_path), len(history), "PIT S&P 500 components from fja05680/sp500."))
    records.append(SourceRecord("sp500_current", "downloaded", str(current_path), len(current), "Current S&P 500 metadata from fja05680/sp500."))
    records.append(SourceRecord("sp500_ticker_start_end", "downloaded", str(ticker_dates_path), len(ticker_dates), "Ticker membership intervals from fja05680/sp500."))
    return history


def build_universe(
    history: pd.DataFrame,
    raw_dir: Path,
    normalized_dir: Path,
    start: str,
    end: str,
    records: list[SourceRecord],
) -> pd.DataFrame:
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"])
    history = history[(history["date"] <= pd.Timestamp(end))].sort_values("date")
    days = pd.bdate_range(start, end)
    rows = []
    latest_idx = 0
    latest_tickers: list[str] = []
    hist_dates = history["date"].to_numpy()
    hist_tickers = history["tickers"].astype(str).tolist()
    for day in days:
        while latest_idx < len(history) and pd.Timestamp(hist_dates[latest_idx]) <= day:
            latest_tickers = [t.strip() for t in hist_tickers[latest_idx].split(",") if t.strip()]
            latest_idx += 1
        if not latest_tickers:
            continue
        for ticker in latest_tickers:
            rows.append(
                {
                    "date": day,
                    "asset_id": ticker,
                    "ticker": ticker,
                    "permno": "",
                    "permco": "",
                    "cusip": "",
                    "figi": "",
                    "exchange": "",
                    "share_code": "",
                    "is_active": True,
                    "is_tradable": True,
                    "sector": "unknown",
                    "industry": "unknown",
                    "country": "US",
                    "currency": "USD",
                    "index_membership_sp500": True,
                    "index_membership_russell1000": False,
                    "universe_inclusion_reason": "fja05680_sp500_pit_component",
                }
            )
    universe = pd.DataFrame(rows)
    current_path = raw_dir / "sp500_current_wikipedia_fja05680.csv"
    if current_path.exists():
        current = pd.read_csv(current_path)
        if {"Symbol", "GICS Sector", "GICS Sub-Industry"}.issubset(current.columns):
            mapping = current.rename(
                columns={"Symbol": "ticker", "GICS Sector": "sector_current", "GICS Sub-Industry": "industry_current"}
            )[["ticker", "sector_current", "industry_current"]]
            universe = universe.merge(mapping, on="ticker", how="left")
            universe["sector"] = universe["sector_current"].fillna(universe["sector"])
            universe["industry"] = universe["industry_current"].fillna(universe["industry"])
            universe = universe.drop(columns=["sector_current", "industry_current"])
    path = normalized_dir / "open_universe_daily.parquet"
    universe.to_parquet(path, index=False)
    records.append(SourceRecord("open_universe_daily", "created", str(path), len(universe), "Daily PIT S&P 500 universe, business-day expanded."))
    return universe


def download_prices(
    tickers: list[str],
    normalized_dir: Path,
    start: str,
    end: str,
    records: list[SourceRecord],
) -> None:
    yahoo_symbols = [to_yahoo_symbol(t) for t in tickers]
    print(f"downloading yfinance OHLCV for {len(yahoo_symbols)} tickers...")
    data = yf.download(
        yahoo_symbols,
        start=start,
        end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=False,
        actions=True,
        threads=True,
        progress=False,
    )
    rows = []
    if isinstance(data.columns, pd.MultiIndex):
        for original, yahoo_symbol in zip(tickers, yahoo_symbols):
            if yahoo_symbol not in data.columns.get_level_values(0):
                continue
            sub = data[yahoo_symbol].dropna(how="all").reset_index()
            rows.extend(normalize_yfinance_rows(sub, original))
    else:
        rows.extend(normalize_yfinance_rows(data.reset_index(), tickers[0] if tickers else ""))
    prices = pd.DataFrame(rows)
    if prices.empty:
        records.append(SourceRecord("open_prices_daily", "failed", None, 0, "yfinance returned no rows."))
        return
    prices = prices.sort_values(["date", "asset_id"]).reset_index(drop=True)
    path = normalized_dir / "open_prices_daily.parquet"
    prices.to_parquet(path, index=False)
    records.append(SourceRecord("open_prices_daily", "created", str(path), len(prices), "Yahoo/yfinance daily OHLCV/actions, open-data prototype only."))


def normalize_yfinance_rows(frame: pd.DataFrame, ticker: str) -> list[dict]:
    if "Date" not in frame.columns:
        return []
    frame = frame.rename(columns={c: c.lower().replace(" ", "_") for c in frame.columns})
    rows = []
    for _, row in frame.iterrows():
        close = row.get("close", np.nan)
        adj_close = row.get("adj_close", close)
        volume = row.get("volume", np.nan)
        if pd.isna(close) or pd.isna(volume):
            continue
        rows.append(
            {
                "date": pd.to_datetime(row["date"]),
                "asset_id": ticker,
                "ticker": ticker,
                "open": row.get("open", np.nan),
                "high": row.get("high", np.nan),
                "low": row.get("low", np.nan),
                "close": close,
                "adjusted_close": adj_close,
                "volume": volume,
                "dividend_amount": row.get("dividends", 0.0),
                "split_factor": row.get("stock_splits", 0.0),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return []
    out["ret_1d_ex_delist"] = out.groupby("asset_id")["adjusted_close"].pct_change()
    out["delisting_return"] = 0.0
    out["ret_1d"] = out["ret_1d_ex_delist"].fillna(0.0)
    out["price"] = out["adjusted_close"]
    out["dollar_volume"] = out["price"].abs() * out["volume"]
    out["spread_proxy"] = (out["high"] - out["low"]).abs() / out["price"].abs().replace(0, np.nan)
    out["liquidity_score"] = (1.0 - 20.0 * out["spread_proxy"].fillna(0.0)).clip(0.0, 1.0)
    out["market_cap"] = np.nan
    return out.to_dict("records")


def download_vix(
    session: requests.Session,
    raw_dir: Path,
    intermediate_dir: Path,
    records: list[SourceRecord],
    start: str,
    end: str,
) -> None:
    raw_path = raw_dir / "cboe_vix_history.csv"
    vix = fetch_csv(session, CBOE_VIX_URL, raw_path)
    vix = vix.rename(columns={c: c.strip().lower() for c in vix.columns})
    vix["date"] = pd.to_datetime(vix["date"])
    vix = vix[(vix["date"] >= start) & (vix["date"] <= end)].rename(columns={"close": "vix"})
    path = intermediate_dir / "vix_daily.parquet"
    vix[["date", "open", "high", "low", "vix"]].to_parquet(path, index=False)
    records.append(SourceRecord("cboe_vix", "downloaded", str(path), len(vix), "Cboe VIX daily OHLC filtered to requested period."))


def download_fred(
    session: requests.Session,
    raw_dir: Path,
    normalized_dir: Path,
    records: list[SourceRecord],
    start: str,
    end: str,
) -> None:
    frames = []
    for series in FRED_SERIES:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        path = raw_dir / f"fred_{series}.csv"
        try:
            df = fetch_csv(session, url, path, timeout=60)
            value_col = [c for c in df.columns if c != "observation_date"][0]
            df = df.rename(columns={"observation_date": "date", value_col: series.lower()})
            df["date"] = pd.to_datetime(df["date"])
            frames.append(df[["date", series.lower()]])
            records.append(SourceRecord(f"fred_{series}", "downloaded", str(path), len(df), FRED_SERIES[series]))
            time.sleep(0.2)
        except Exception as exc:
            records.append(SourceRecord(f"fred_{series}", "failed", str(path), None, str(exc)))
    if not frames:
        return
    macro = frames[0]
    for frame in frames[1:]:
        macro = macro.merge(frame, on="date", how="outer")
    macro = macro.sort_values("date")
    macro = macro[(macro["date"] >= start) & (macro["date"] <= end)]
    macro["vintage_date"] = macro["date"]
    path = normalized_dir / "fred_macro_asof_or_latest.parquet"
    macro.to_parquet(path, index=False)
    records.append(SourceRecord("fred_macro_asof_or_latest", "created", str(path), len(macro), "FRED latest/revised values; use ALFRED for true vintage macro."))


def download_french(
    session: requests.Session,
    raw_dir: Path,
    normalized_dir: Path,
    records: list[SourceRecord],
    start: str,
    end: str,
) -> None:
    parsed = []
    for name, url in FRENCH_URLS.items():
        zip_path = raw_dir / f"kenneth_french_{name}.zip"
        content = fetch_bytes(session, url, zip_path)
        df = parse_french_zip(content)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        parsed.append((name, df))
        records.append(SourceRecord(f"kenneth_french_{name}", "downloaded", str(zip_path), len(df), "Kenneth French Data Library daily factor file."))
    if not parsed:
        return
    factors = parsed[0][1]
    for _, df in parsed[1:]:
        factors = factors.merge(df, on="date", how="outer")
    factors = factors.sort_values("date")
    path = normalized_dir / "kenneth_french_daily_factors.parquet"
    factors.to_parquet(path, index=False)
    records.append(SourceRecord("kenneth_french_daily_factors", "created", str(path), len(factors), "Merged FF5 daily and momentum daily factors."))


def download_sec_ticker_map(session: requests.Session, raw_dir: Path, records: list[SourceRecord]) -> None:
    path = raw_dir / "sec_company_tickers.json"
    content = fetch_bytes(session, SEC_COMPANY_TICKERS_URL, path)
    obj = json.loads(content.decode("utf-8"))
    records.append(SourceRecord("sec_company_tickers", "downloaded", str(path), len(obj), "SEC ticker to CIK mapping; full company facts are intentionally not bulk-downloaded."))


def download_finra_short_volume(
    session: requests.Session,
    raw_dir: Path,
    normalized_dir: Path,
    tickers: list[str],
    start: str,
    end: str,
    records: list[SourceRecord],
) -> None:
    ticker_set = set(tickers)
    days = pd.bdate_range(start, end)
    out_frames = []
    saved_dir = raw_dir / "finra_cnms_daily"
    saved_dir.mkdir(parents=True, exist_ok=True)
    for idx, day in enumerate(days):
        url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{day:%Y%m%d}.txt"
        path = saved_dir / f"CNMSshvol{day:%Y%m%d}.txt"
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore")
        else:
            try:
                r = session.get(url, timeout=20)
                if r.status_code != 200 or not r.text.startswith("Date|Symbol|"):
                    continue
                text = r.text
                path.write_text(text, encoding="utf-8")
                time.sleep(0.05)
            except Exception:
                continue
        df = pd.read_csv(io.StringIO(text), sep="|")
        if "Symbol" not in df:
            continue
        df = df[df["Symbol"].isin(ticker_set)]
        if df.empty:
            continue
        df = df.rename(
            columns={
                "Date": "date",
                "Symbol": "asset_id",
                "ShortVolume": "short_sale_volume",
                "ShortExemptVolume": "short_exempt_volume",
                "TotalVolume": "short_total_volume",
            }
        )
        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
        df["short_sale_volume_ratio"] = df["short_sale_volume"] / df["short_total_volume"].replace(0, np.nan)
        out_frames.append(df[["date", "asset_id", "short_sale_volume", "short_exempt_volume", "short_total_volume", "short_sale_volume_ratio"]])
        if (idx + 1) % 250 == 0:
            print(f"FINRA short-volume days scanned: {idx + 1}/{len(days)}")
    if not out_frames:
        records.append(SourceRecord("finra_cnms_short_volume", "failed", None, 0, "No FINRA CNMS rows were downloaded."))
        return
    out = pd.concat(out_frames, ignore_index=True).sort_values(["date", "asset_id"])
    path = normalized_dir / "finra_cnms_short_volume.parquet"
    out.to_parquet(path, index=False)
    records.append(SourceRecord("finra_cnms_short_volume", "created", str(path), len(out), "FINRA consolidated NMS daily short-sale volume, available from 2018."))


def fetch_csv(session: requests.Session, url: str, path: Path, timeout: int = 30) -> pd.DataFrame:
    content = fetch_bytes(session, url, path, timeout=timeout)
    return pd.read_csv(io.BytesIO(content))


def fetch_bytes(session: requests.Session, url: str, path: Path, timeout: int = 30) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    path.write_bytes(response.content)
    return response.content


def parse_french_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        names = zf.namelist()
        csv_name = names[0]
        text = zf.read(csv_name).decode("utf-8", errors="ignore")
    lines = text.splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith(","))
    data_lines = []
    header = "date" + lines[header_idx]
    for line in lines[header_idx + 1 :]:
        if not line.strip() or not line[:8].strip().isdigit():
            break
        data_lines.append(line)
    df = pd.read_csv(io.StringIO("\n".join([header] + data_lines)))
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0
    return df


def to_yahoo_symbol(ticker: str) -> str:
    return ticker.replace(".", "-")


if __name__ == "__main__":
    main()
