#!/usr/bin/env python3
import os
import time
from time import perf_counter
from typing import Tuple

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from alpaca_trade_api.rest import REST, TimeFrame, APIError

# ── CONFIG ───────────────────────────────────────────────────────────────
TICKERS = ["AAPL", "JPM", "AMZN", "TSLA", "MSFT"]

# Must match training:
WINDOW = int(os.environ.get("RL_WINDOW", 20))   # price window length W
MODEL_PATH = os.environ.get("MODEL_PATH", "ppo_ceemd_cnnlstm_rl.zip")

INITIAL_CASH = float(os.environ.get("INITIAL_CASH", 10_000))
SLEEP_INTERVAL = int(os.environ.get("SLEEP_INTERVAL", 60))  # seconds

# Alpaca
ALPACA_API_KEY    = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
ALPACA_BASE_URL   = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Indicators used in training:
# RSI(14), MACD(12,26,9), BB width(20,2), Ret (1), Vol20 (20), VolZ60 (60)
RSI_WIN = 14
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9
BB_WIN, BB_DEV = 20, 2
RET_WIN = 1
VOL_WIN = 20
VOLZ_WIN = 60

# Grab plenty of history to avoid NaNs from warmups
HIST_N = max(300, WINDOW + MACD_SLOW + 50, VOLZ_WIN + 50, BB_WIN + 50)

# ── RL AGENT ─────────────────────────────────────────────────────────────
def _const(v):
    return lambda _progress: v

# Strip a trailing ".zip" to avoid accidental ".zip.zip" handling in some paths
_model_path_stem = MODEL_PATH[:-4] if MODEL_PATH.endswith(".zip") else MODEL_PATH
agent = PPO.load(
    _model_path_stem,
    custom_objects={
        # set these to your training values if different
        "lr_schedule": _const(2.5e-4),
        "clip_range": _const(0.2),
    },
)

# ── ALPACA CLIENT ────────────────────────────────────────────────────────
api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version="v2")

# ── PORTFOLIO STATE (local tracker of shares) ────────────────────────────
portfolio = {sym: {"shares": 0} for sym in TICKERS}

# ── HELPERS: Indicators (pandas-only, mirrors training) ─────────────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns: 'close', 'volume'. Returns df with columns:
      close, volume, ret, rsi14, macd, macd_signal, bb_width, vol20, volz60
    """
    df = df.copy()
    c = df["close"].astype(float)
    v = df["volume"].astype(float)

    # Return on chosen timeframe
    df["ret"] = c.pct_change(periods=RET_WIN)

    # RSI(14)
    delta = c.diff()
    gain = delta.clip(lower=0.0).rolling(RSI_WIN, min_periods=RSI_WIN).mean()
    loss = -delta.clip(upper=0.0).rolling(RSI_WIN, min_periods=RSI_WIN).mean()
    rs = gain / (loss + 1e-12)
    df["rsi14"] = 100.0 - (100.0 / (1.0 + rs))

    # MACD(12,26,9)
    ema_fast = c.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = c.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=MACD_SIG, adjust=False).mean()
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal

    # Bollinger width (20,2) normalized by MA20 magnitude
    ma = c.rolling(BB_WIN, min_periods=BB_WIN).mean()
    sd = c.rolling(BB_WIN, min_periods=BB_WIN).std()
    upper = ma + BB_DEV * sd
    lower = ma - BB_DEV * sd
    df["bb_width"] = (upper - lower) / (ma.abs() + 1e-12)

    # Rolling volatility of returns (20)
    df["vol20"] = df["ret"].rolling(VOL_WIN, min_periods=VOL_WIN).std()

    # Volume z-score (60)
    mu = v.rolling(VOLZ_WIN, min_periods=VOLZ_WIN).mean()
    sdv = v.rolling(VOLZ_WIN, min_periods=VOLZ_WIN).std()
    df["volz60"] = (v - mu) / (sdv + 1e-12)

    # forward/back fill just in case, then drop rows still NaN on key features
    df = df.ffill().bfill()
    needed = ["rsi14", "macd", "macd_signal", "bb_width", "ret", "vol20", "volz60"]
    df = df.dropna(subset=needed)
    return df

def get_recent_bars(sym: str, limit: int) -> pd.DataFrame:
    # Pull latest minute bars; adjust timeframe if you trained on another frame
    bars = api.get_bars(symbol=sym, timeframe=TimeFrame.Minute, limit=limit).df
    if bars.empty:
        return pd.DataFrame()
    # For single symbol, Alpaca returns single-index DF; ensure columns lower-case
    bars = bars.reset_index()
    bars.columns = [str(c).lower() for c in bars.columns]
    cols = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in bars.columns]
    return bars[cols].copy()

def build_asset_block(sym: str) -> Tuple[np.ndarray, float]:
    """
    Returns:
      - feature block for one asset: [WINDOW normalized closes, 7 scalars] as 1-D numpy
      - latest price (float)
    """
    df = get_recent_bars(sym, limit=HIST_N)
    if df.empty or len(df) < WINDOW:
        raise ValueError(f"Not enough bars for {sym} (have {len(df)}, need >= {WINDOW})")

    df_ind = compute_indicators(df)
    if len(df_ind) < WINDOW:
        raise ValueError(f"Not enough post-indicator rows for {sym} (have {len(df_ind)}, need >= {WINDOW})")

    # WINDOW closes ending at the last bar (exclusive of the next step, mirroring env)
    closes_win = df_ind["close"].iloc[-WINDOW:].to_numpy(dtype=float)
    base = float(max(closes_win[0], 1e-12))
    norm_window = (closes_win / base).astype(np.float64)  # length = WINDOW

    # last scalars
    needed_cols = ["rsi14", "macd", "macd_signal", "bb_width", "ret", "vol20", "volz60"]
    missing = [c for c in needed_cols if c not in df_ind.columns]
    if missing:
        raise KeyError(f"{sym} missing columns after indicator calc: {missing}")

    last = df_ind.iloc[-1]
    scalars = np.array(
        [float(last[c]) for c in needed_cols],
        dtype=np.float64
    )

    latest_price = float(last["close"])
    block = np.concatenate([norm_window, scalars], axis=0)  # length = WINDOW + 7
    return block, latest_price

def softmax(weights: np.ndarray) -> np.ndarray:
    w = weights.astype(np.float64).ravel()
    w = w - np.max(w)
    e = np.exp(w)
    s = e / (np.sum(e) + 1e-12)
    return s

# ── MAIN LOOP ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("▶️  Starting RL Alpaca trading loop. Ctrl+C to exit.", flush=True)
    try:
        while True:
            loop_start = perf_counter()
            now = pd.Timestamp.now(tz="UTC").tz_convert("US/Eastern")
            print(f"\n🔄 Loop start: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}", flush=True)

            # 1) Market check
            clock = api.get_clock()
            if not clock.is_open:
                print(f"  ❌ Market closed, next open at {clock.next_open}")
                time.sleep(SLEEP_INTERVAL)
                continue
            print(f"  ⏱ Market open, next close at {clock.next_close}")

            # 2) Refresh account & cash
            account = api.get_account()
            cash = float(account.cash)

            # 3) Build observation (exactly like training layout)
            blocks = []
            latest_prices = []
            failed = []
            for sym in TICKERS:
                try:
                    block, px = build_asset_block(sym)
                    blocks.append(block)
                    latest_prices.append(px)
                except Exception as e:
                    failed.append((sym, str(e)))
                    print(f"  ⚠️  {sym}: {e}")

            if len(blocks) != len(TICKERS):
                print("  ⚠️  Missing features for one or more tickers, skipping loop.")
                time.sleep(SLEEP_INTERVAL)
                continue

            per_asset = np.concatenate(blocks, axis=0)  # K*(W+7)
            latest_prices = np.array(latest_prices, dtype=np.float64)

            # Local portfolio net worth
            shares_vec = np.array([portfolio[s]["shares"] for s in TICKERS], dtype=np.float64)
            pos_val = float(np.sum(shares_vec * latest_prices))
            net_worth = cash + pos_val
            if net_worth <= 0:
                print("  ❌ Net worth non-positive, skipping.")
                time.sleep(SLEEP_INTERVAL)
                continue

            cash_ratio = np.array([cash / net_worth], dtype=np.float64)  # (1,)
            obs = np.concatenate([per_asset, cash_ratio, shares_vec], axis=0).astype(np.float32).reshape(1, -1)

            # 4) Policy -> target weights (long-only, sum ≤ 1)
            raw_action, _ = agent.predict(obs, deterministic=True)
            weights = np.clip(raw_action.reshape(-1), 0.0, 1.0)
            total_w = float(weights.sum())
            if total_w > 1.0:
                weights /= total_w  # keep ≤1
            # OPTIONAL: always allocate all via softmax (comment out above two lines if you use this)
            # weights = softmax(raw_action) * 0.95

            # 5) Target allocations
            investable_value = net_worth  # policy can leave cash via sum(weights) <= 1
            target_values = weights * investable_value
            current_values = shares_vec * latest_prices
            deltas = target_values - current_values  # $ to add/remove per asset

            # 6) Rebalance with market orders (minimal sanity checks)
            for i, sym in enumerate(TICKERS):
                px = latest_prices[i]
                target_shares = int(target_values[i] // max(px, 1e-12))
                cur_sh = portfolio[sym]["shares"]
                to_trade = target_shares - cur_sh
                if to_trade == 0:
                    continue

                if to_trade > 0:
                    est_cost = to_trade * px
                    if est_cost > cash * 0.99:   # leave a little cash headroom
                        to_trade = int((cash * 0.99) // max(px, 1e-12))
                    if to_trade <= 0:
                        continue
                    try:
                        api.submit_order(symbol=sym, qty=to_trade, side="buy", type="market", time_in_force="day")
                        portfolio[sym]["shares"] += to_trade
                        cash -= to_trade * px
                        print(f"  ✅ BUY  {to_trade:4d} {sym} @ {px:.2f}")
                    except APIError as e:
                        print(f"  ❌ BUY {sym}: {e}")
                else:
                    sell_qty = min(-to_trade, cur_sh)
                    if sell_qty <= 0:
                        continue
                    try:
                        api.submit_order(symbol=sym, qty=sell_qty, side="sell", type="market", time_in_force="day")
                        portfolio[sym]["shares"] -= sell_qty
                        cash += sell_qty * px
                        print(f"  ✅ SELL {sell_qty:4d} {sym} @ {px:.2f}")
                    except APIError as e:
                        print(f"  ❌ SELL {sym}: {e}")

            # 7) Loop summary & sleep
            loop_time = perf_counter() - loop_start
            next_run = now + pd.Timedelta(seconds=SLEEP_INTERVAL)
            print(f"✅ Loop done in {loop_time:.2f}s. Next run ~ {next_run.strftime('%H:%M:%S %Z')}", flush=True)
            time.sleep(SLEEP_INTERVAL)

    except KeyboardInterrupt:
        print("🛑  Stopped by user", flush=True)
    except Exception as e:
        print(f"⚠️  Error in loop: {e}", flush=True)
        time.sleep(5)
