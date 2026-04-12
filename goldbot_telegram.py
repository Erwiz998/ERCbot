#!/usr/bin/env python3
"""
XAUUSD Gold Signal Bot — Telegram
Data : Twelve Data API (XAUUSD realtime, delay ~1 menit)
Sinyal: 7 Kondisi Teknikal
⚠️  Bot ini HANYA kirim SINYAL — tidak auto trade
"""

import asyncio
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

from telegram import Bot
from telegram.constants import ParseMode

# ══════════════════════════════════════════════════════════════
# ⚙️  KONFIGURASI — isi di Railway Environment Variables
# ══════════════════════════════════════════════════════════════
TELEGRAM_TOKEN    = os.environ.get("TELEGRAM_TOKEN",    "ISI_TOKEN_TELEGRAM")
TELEGRAM_CHAT_ID  = os.environ.get("TELEGRAM_CHAT_ID",  "ISI_CHAT_ID_TELEGRAM")
TWELVE_DATA_KEY   = os.environ.get("TWELVE_DATA_KEY",   "ISI_API_KEY_TWELVEDATA")

SCAN_INTERVAL = 900       # Scan tiap 15 menit (900 detik)
SYMBOL        = "XAU/USD" # Format Twelve Data
TIMEFRAME     = "15min"   # M15
CANDLE_LIMIT  = 250       # Jumlah candle

# Parameter indikator
EMA200   = 200
MA99     = 99
MA20     = 20
BB_P     = 20
BB_DEV   = 2.0
ATR_P    = 14
RSI_P    = 14
VOL_MULT = 1.2
RSI_OB   = 65.0
RSI_OS   = 35.0
MIN_ATR  = 0.5
SR_LOOK  = 100
RR       = 2.0
MARGIN   = 3
LEVERAGE = 50

# ══════════════════════════════════════════════════════════════
# 📥  AMBIL DATA DARI TWELVE DATA
# ══════════════════════════════════════════════════════════════
def get_candles():
    """Ambil candle XAUUSD M15 dari Twelve Data"""
    url    = "https://api.twelvedata.com/time_series"
    params = {
        "symbol":     SYMBOL,
        "interval":   TIMEFRAME,
        "outputsize": CANDLE_LIMIT,
        "apikey":     TWELVE_DATA_KEY,
        "order":      "ASC",  # Dari lama ke baru
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    # Cek error dari API
    if data.get("status") == "error":
        raise Exception(f"Twelve Data error: {data.get('message')}")

    values = data.get("values", [])
    if not values:
        raise Exception("Tidak ada data dari Twelve Data")

    rows = []
    for v in values:
        rows.append({
            "time":   v["datetime"],
            "open":   float(v["open"]),
            "high":   float(v["high"]),
            "low":    float(v["low"]),
            "close":  float(v["close"]),
            "volume": float(v.get("volume", 1)),
        })

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df

# ══════════════════════════════════════════════════════════════
# 📐  HITUNG SEMUA INDIKATOR
# ══════════════════════════════════════════════════════════════
def compute_indicators(df):
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    df["ema200"]   = c.ewm(span=EMA200, adjust=False).mean()
    df["ma99"]     = c.rolling(MA99).mean()
    df["vol_ma"]   = v.rolling(MA20).mean()

    df["bb_mid"]   = c.rolling(BB_P).mean()
    bb_std         = c.rolling(BB_P).std()
    df["bb_upper"] = df["bb_mid"] + BB_DEV * bb_std
    df["bb_lower"] = df["bb_mid"] - BB_DEV * bb_std

    tr             = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"]      = tr.rolling(ATR_P).mean()

    delta          = c.diff()
    gain           = delta.clip(lower=0).rolling(RSI_P).mean()
    loss           = (-delta.clip(upper=0)).rolling(RSI_P).mean()
    rs             = gain / loss.replace(0, np.nan)
    df["rsi"]      = 100 - (100 / (1 + rs))
    return df

# ══════════════════════════════════════════════════════════════
# 🔍  CEK 7 KONDISI TEKNIKAL
# ══════════════════════════════════════════════════════════════
def check_signal(df):
    df    = compute_indicators(df)
    i     = -2  # Candle closed, bukan candle aktif
    row   = df.iloc[i]
    close = row["close"]
    open_ = row["open"]
    high  = row["high"]
    low   = row["low"]
    atr   = row["atr"]
    r     = {}

    # [1] Trend Alignment — EMA200
    r["trend_up"]   = close > row["ema200"]
    r["trend_down"] = close < row["ema200"]

    # [2] Volume Anomaly — lonjakan 1.2x MA20
    r["vol_spike"]  = (row["vol_ma"] > 0) and (row["volume"] >= row["vol_ma"] * VOL_MULT)

    # [3] Pinbar — ekor >= 2x body
    body          = abs(close - open_)
    upper_sh      = high - max(close, open_)
    lower_sh      = min(close, open_) - low
    min_body      = atr * 0.05
    r["bull_pin"] = (body < min_body * 4) and (lower_sh >= body * 2) and (lower_sh > upper_sh)
    r["bear_pin"] = (body < min_body * 4) and (upper_sh >= body * 2) and (upper_sh > lower_sh)

    # [4] Dynamic Wall — sentuh MA99 atau outer BB
    zone          = atr * 0.3
    r["dyn_wall"] = (abs(low  - row["ma99"])     < zone or
                     abs(high - row["ma99"])     < zone or
                     abs(high - row["bb_upper"]) < zone or
                     abs(low  - row["bb_lower"]) < zone)

    # [5] Static S/R — swing high/low 100 candle terakhir
    lookback     = df.iloc[i - SR_LOOK : i]
    sr_high      = lookback["high"].max()
    sr_low       = lookback["low"].min()
    r["near_sr"] = (abs(close - sr_high) < atr * 0.5 or
                    abs(close - sr_low)  < atr * 0.5)

    # [6] RSI Filter
    r["rsi_bull_ok"] = row["rsi"] < RSI_OB
    r["rsi_bear_ok"] = row["rsi"] > RSI_OS

    # [7] Session + ATR Filter (WIB)
    h_wib           = datetime.now(pytz.timezone("Asia/Jakarta")).hour
    r["session_ok"] = (14 <= h_wib < 23) or (h_wib >= 20) or (h_wib < 5)
    r["atr_ok"]     = atr >= MIN_ATR

    # ── KEPUTUSAN ─────────────────────────────────────────────
    long_ok  = (r["trend_up"]   and r["vol_spike"] and r["bull_pin"] and
                r["dyn_wall"]   and r["near_sr"]   and r["rsi_bull_ok"] and
                r["session_ok"] and r["atr_ok"])

    short_ok = (r["trend_down"] and r["vol_spike"] and r["bear_pin"] and
                r["dyn_wall"]   and r["near_sr"]   and r["rsi_bear_ok"] and
                r["session_ok"] and r["atr_ok"])

    score = sum([
        r["trend_up"] or r["trend_down"],
        r["vol_spike"],
        r["bull_pin"] or r["bear_pin"],
        r["dyn_wall"],
        r["near_sr"],
        r["rsi_bull_ok"] or r["rsi_bear_ok"],
        r["session_ok"] and r["atr_ok"],
    ])

    return {
        "long":  long_ok, "short": short_ok,
        "score": score,   "price": close,
        "atr":   atr,     "rsi":   row["rsi"],
        "r":     r,       "time":  df.iloc[i]["time"],
    }

# ══════════════════════════════════════════════════════════════
# 📨  FORMAT PESAN TELEGRAM
# ══════════════════════════════════════════════════════════════
def format_message(sig):
    price   = sig["price"]
    atr     = sig["atr"]
    sl_dist = atr * 1.5
    tp_dist = sl_dist * RR
    r       = sig["r"]

    if sig["long"]:
        direction = "LONG 🟢"
        sl = price - sl_dist
        tp = price + tp_dist
    else:
        direction = "SHORT 🔴"
        sl = price + sl_dist
        tp = price - tp_dist

    score = sig["score"]
    grade = "A+" if score == 7 else "A" if score == 6 else "B+"
    emoji = "🏆" if score == 7 else "✅" if score >= 6 else "⚠️"

    cond = (
        f"  • Trend EMA200 : {'✅' if r['trend_up'] or r['trend_down'] else '❌'}\n"
        f"  • Volume Spike : {'✅' if r['vol_spike'] else '❌'}\n"
        f"  • Pinbar       : {'✅' if r['bull_pin'] or r['bear_pin'] else '❌'}\n"
        f"  • Dynamic Wall : {'✅' if r['dyn_wall'] else '❌'}\n"
        f"  • Static S/R   : {'✅' if r['near_sr'] else '❌'}\n"
        f"  • RSI Filter   : {'✅' if r['rsi_bull_ok'] or r['rsi_bear_ok'] else '❌'} ({sig['rsi']:.1f})\n"
        f"  • Session+ATR  : {'✅' if r['session_ok'] and r['atr_ok'] else '❌'}"
    )

    return (
        f"{emoji} *XAUUSD SIGNAL — {grade}*\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🕐 {sig['time'].strftime('%d/%m %H:%M')} | Score: *{score}/7*\n"
        f"📡 Data: Twelve Data (delay ~1 menit)\n"
        f"📍 Type: *{direction}*\n\n"
        f"💰 Entry  : `{price:.2f}`\n"
        f"🛑 SL     : `{sl:.2f}`\n"
        f"🎯 TP     : `{tp:.2f}`\n"
        f"📊 R:R    : 1:{RR}\n"
        f"💼 Margin : ${MARGIN} × {LEVERAGE}x\n\n"
        f"📋 *Kondisi:*\n{cond}\n\n"
        f"⚠️ _Bukan financial advice. Eksekusi manual di FBS._"
    )

# ══════════════════════════════════════════════════════════════
# 🚀  MAIN LOOP — jalan terus setiap 15 menit
# ══════════════════════════════════════════════════════════════
async def run_bot():
    bot = Bot(token=TELEGRAM_TOKEN)
    print(f"✅ GoldBot Twelve Data aktif | Scan tiap {SCAN_INTERVAL//60} menit")

    await bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text=(
            "🤖 *XAUUSD GoldBot v2.7 — AKTIF*\n"
            "📡 Data: Twelve Data XAU/USD\n"
            "Scanning M15 · 7 Kondisi Teknikal\n"
            "Delay data ~1 menit ✅"
        ),
        parse_mode=ParseMode.MARKDOWN
    )

    while True:
        try:
            now = datetime.now().strftime("%H:%M:%S")
            print(f"[{now}] Fetching Twelve Data...")

            df  = get_candles()
            sig = check_signal(df)

            if sig["long"] or sig["short"]:
                await bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=format_message(sig),
                    parse_mode=ParseMode.MARKDOWN
                )
                direction = "LONG" if sig["long"] else "SHORT"
                print(f"✅ Sinyal: {direction} | Score {sig['score']}/7 | {sig['price']:.2f}")
            else:
                print(f"⏳ No signal | Score {sig['score']}/7 | RSI {sig['rsi']:.1f} | {sig['price']:.2f}")

        except Exception as e:
            print(f"❌ Error: {e}")
            try:
                await bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=f"⚠️ Bot error: {str(e)[:150]}"
                )
            except Exception:
                pass

        await asyncio.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    asyncio.run(run_bot())
