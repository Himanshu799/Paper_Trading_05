#!/usr/bin/env python3
import os
import time
from time import perf_counter
from typing import Tuple
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from alpaca_trade_api.rest import REST, TimeFrame, APIError

# ── CONFIG ───────────────────────────────────────────────────────────────
TICKERS = ["AAPL", "JPM", "AMZN", "TSLA", "MSFT"]

# Must match training (51-dim obs = K*WINDOW + 1 + K; with K=5 → WINDOW=9)
WINDOW = int(os.environ.get("RL_WINDOW", 9))   # price window length W
MODEL_PATH = os.environ.get("MODEL_PATH", "ppo_ceemd_cnnlstm_rl.zip")

INITIAL_CASH = float(os.environ.get("INITIAL_CASH", 10_000))
SLEEP_INTERVAL = int(os.environ.get("SLEEP_INTERVAL", 60))  # seconds between loops

# Alpaca creds
ALPACA_API_KEY    = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
ALPACA_BASE_URL   = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Timeframe + feed (match training)
# ALPACA_TIMEFRAME: "minute" or "day"
ALPACA_TIMEFRAME = os.environ.get("ALPACA_TIMEFRAME", "minute").lower()
ALPACA_FEED      = os.environ.get("ALPACA_FEED", "iex")  # "iex" (free) or "sip" (paid)

# Lookback to ensure enough bars for WINDOW even at market open
HIST_LOOKBACK_MINUTES = int(os.environ.get("HIST_LOOKBACK_MINUTES", 1200))  # ~20 hours minutes
HIST_LOOKBACK_DAYS    = int(os.environ.get("HIST_LOOKBACK_DAYS", 120))      # ~6 months days

# Safety buffer for history slicing
HIST_N = max(3 * WINDOW, 300)

# ── RL AGENT ─────────────────────────────────────────────────────────────
def _const(v):
    return lambda _progress: v

_model_path_stem = MODEL_PATH[:-4] if MODEL_PATH.endswith(".zip") else MODEL_PATH
agent = PPO.load(
    _model_path_stem,
    custom_objects={
        "lr_schedule": _const(2.5e-4),
        "clip_range": _const(0.2),
    },
)

# ── ALPACA CLIENT ────────────────────────────────────────────────────────
api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version="v2")

# ── PORTFOLIO STATE (local tracker of shares) ────────────────────────────
portfolio = {sym: {"shares": 0} for sym in TICKERS}

# ── DATA FETCH ───────────────────────────────────────────────────────────
def get_recent_bars(sym: str, limit: int) -> pd.DataFrame:
    """
    Fetch enough bars across days so WINDOW is always available.
    Returns columns: timestamp, open, high, low, close, volume (lowercase).
    """
    if ALPACA_TIMEFRAME == "day":
        tf = TimeFrame.Day
        start_dt = datetime.now(timezone.utc) - timedelta(days=HIST_LOOKBACK_DAYS)
    else:
        tf = TimeFrame.Minute
        start_dt = datetime.now(timezone.utc) - timedelta(minutes=HIST_LOOKBACK_MINUTES)

    try:
        bars = api.get_bars(
            symbol=sym,
            timeframe=tf,
            start=start_dt.isoformat(),
            feed=ALPACA_FEED,
            limit=None
        ).df
    except APIError as e:
        print(f"  ❌ get_bars error for {sym}: {e}")
        return pd.DataFrame()

    if bars is None or bars.empty:
        return pd.DataFrame()

    bars = bars.reset_index()
    bars.columns = [str(c).lower() for c in bars.columns]
    cols = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in bars.columns]
    bars = bars[cols].sort_values("timestamp")

    # Keep a tail big enough to be safe
    if len(bars) > max(limit, HIST_N):
        bars = bars.tail(max(limit, HIST_N)).copy()
    return bars

# ── FEATURES: price windows ONLY (matches training obs) ──────────────────
def build_asset_block(sym: str) -> Tuple[np.ndarray, float]:
    """
    Returns:
      - normalized close window of length = WINDOW (1-D float64)
      - latest close price (float)
    """
    df = get_recent_bars(sym, limit=HIST_N)
    if df.empty or len(df) < WINDOW:
        raise ValueError(f"Not enough bars for {sym} (have {len(df)}, need >= {WINDOW})")

    closes = df["close"].to_numpy(dtype=float)
    window = closes[-WINDOW:]
    base = float(max(window[0], 1e-12))
    norm_window = (window / base).astype(np.float64)  # shape = (WINDOW,)
    latest_price = float(window[-1])
    return norm_window, latest_price

def softmax(weights: np.ndarray) -> np.ndarray:
    w = weights.astype(np.float64).ravel()
    w = w - np.max(w)
    e = np.exp(w)
    return e / (np.sum(e) + 1e-12)

# ── MAIN LOOP ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("▶️  Starting RL Alpaca trading loop. Ctrl+C to exit.", flush=True)
    try:
        while True:
            loop_start = perf_counter()
            now = pd.Timestamp.now(tz="UTC").tz_convert("US/Eastern")
            print(f"\n🔄 Loop start: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}", flush=True)

            # 1) Market check
            try:
                clock = api.get_clock()
            except Exception as e:
                print(f"  ❌ get_clock error: {e}")
                time.sleep(SLEEP_INTERVAL); continue

            if not clock.is_open:
                print(f"  ❌ Market closed, next open at {clock.next_open}")
                time.sleep(SLEEP_INTERVAL); continue
            print(f"  ⏱ Market open, next close at {clock.next_close}")

            # 2) Account & cash
            try:
                account = api.get_account()
                cash = float(account.cash)
            except Exception as e:
                print(f"  ❌ get_account error: {e}")
                time.sleep(SLEEP_INTERVAL); continue

            # 3) Build observation exactly like training: [K*WINDOW] + [1 cash_ratio] + [K shares]
            blocks = []
            latest_prices = []
            for sym in TICKERS:
                try:
                    block, px = build_asset_block(sym)   # (WINDOW,), latest price
                    blocks.append(block)
                    latest_prices.append(px)
                except Exception as e:
                    print(f"  ⚠️  {sym}: {e}")

            if len(blocks) != len(TICKERS):
                print("  ⚠️  Missing features for one or more tickers, skipping loop.")
                time.sleep(SLEEP_INTERVAL); continue

            per_asset = np.concatenate(blocks, axis=0)  # shape = K*WINDOW
            latest_prices = np.array(latest_prices, dtype=np.float64)

            shares_vec = np.array([portfolio[s]["shares"] for s in TICKERS], dtype=np.float64)
            pos_val = float(np.sum(shares_vec * latest_prices))
            net_worth = cash + pos_val
            if net_worth <= 0:
                print("  ❌ Net worth non-positive, skipping.")
                time.sleep(SLEEP_INTERVAL); continue

            cash_ratio = np.array([cash / net_worth], dtype=np.float64)  # (1,)
            obs_vec = np.concatenate([per_asset, cash_ratio, shares_vec], axis=0).astype(np.float32)
            # Sanity: expect (K*WINDOW + 1 + K,)
            # print("obs shape:", obs_vec.shape)  # uncomment to verify once
            obs = obs_vec.reshape(1, -1)

            # 4) Policy -> target weights (long-only, sum ≤ 1)
            raw_action, _ = agent.predict(obs, deterministic=True)
            weights = np.clip(raw_action.reshape(-1), 0.0, 1.0)
            total_w = float(weights.sum())
            if total_w > 1.0:
                weights /= total_w
            # Or force full allocation:
            # weights = softmax(raw_action) * 0.95

            # 5) Target allocations
            investable_value = net_worth
            target_values = weights * investable_value
            current_values = shares_vec * latest_prices
            deltas = target_values - current_values  # $ delta per asset

            # 6) Rebalance with market orders
            for i, sym in enumerate(TICKERS):
                px = latest_prices[i]
                target_shares = int(target_values[i] // max(px, 1e-12))
                cur_sh = portfolio[sym]["shares"]
                to_trade = target_shares - cur_sh
                if to_trade == 0:
                    continue

                if to_trade > 0:
                    est_cost = to_trade * px
                    if est_cost > cash * 0.99:
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
