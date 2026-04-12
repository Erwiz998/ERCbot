#!/usr/bin/env python3
"""
XAUUSD Gold Signal Bot — Telegram
Data : Binance (XAUUSDT)
Sinyal: 7 Kondisi Teknikal
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
# ⚙️  KONFIGURASI — baca dari environment variable (Railway)
#     Kalau mau test lokal, ganti langsung di sini
# ══════════════════════════════════════════════════════════════
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN",   "ISI_TOKEN_KAMU_DI_SINI")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "ISI_CHAT_ID_KAMU_DI_SINI")

SCAN_INTERVAL = 900       # Scan tiap 15 menit
SYMBOL        = "BTCUSDT" # Pair Binance
TIMEFRAME     = "15m"     # M15

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
MIN_ATR  = 0.8
SR_LOOK  = 100
RR       = 2.0
MARGIN   = 3
LEVERAGE = 50

# ══════════════════════════════════════════════════════════════
# 📥  AMBIL DATA HARGA dari Binance (gratis, tanpa API key)
# ══════════════════════════════════════════════════════════════
def get_candles():
    url    = "https://api.binance.com/api/v3/klines"
    params = {"symbol": SYMBOL, "interval": TIMEFRAME, "limit": 250}
    r      = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data   = r.json()
    df     = pd.DataFrame(data, columns=[
        "time","open","high","low","close","volume",
        "close_time","qav","trades","tbav","tbqv","ignore"
    ])
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="ms")
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
    i     = -2  # Pakai candle closed, bukan candle aktif

    row   = df.iloc[i]
    close = row["close"]
    open_ = row["open"]
    high  = row["high"]
    low   = row["low"]
    atr   = row["atr"]
    r     = {}

    # [1] Trend Alignment
    r["trend_up"]   = close > row["ema200"]
    r["trend_down"] = close < row["ema200"]

    # [2] Volume Anomaly
    r["vol_spike"]  = (row["vol_ma"] > 0) and (row["volume"] >= row["vol_ma"] * VOL_MULT)

    # [3] Pinbar
    body     = abs(close - open_)
    upper_sh = high - max(close, open_)
    lower_sh = min(close, open_) - low
    min_body = atr * 0.05
    r["bull_pin"] = (body < min_body * 4) and (lower_sh >= body * 2) and (lower_sh > upper_sh)
    r["bear_pin"] = (body < min_body * 4) and (upper_sh >= body * 2) and (upper_sh > lower_sh)

    # [4] Dynamic Wall — MA99 atau outer Bollinger
    zone          = atr * 0.3
    r["dyn_wall"] = (abs(low  - row["ma99"])     < zone or
                     abs(high - row["ma99"])      < zone or
                     abs(high - row["bb_upper"])  < zone or
                     abs(low  - row["bb_lower"])  < zone)

    # [5] Static S/R — swing high/low 100 candle terakhir
    lookback   = df.iloc[i - SR_LOOK : i]
    sr_high    = lookback["high"].max()
    sr_low     = lookback["low"].min()
    sr_zone    = atr * 0.5
    r["near_sr"] = (abs(close - sr_high) < sr_zone or
                    abs(close - sr_low)  < sr_zone)

    # [6] RSI Filter
    r["rsi_bull_ok"] = row["rsi"] < RSI_OB
    r["rsi_bear_ok"] = row["rsi"] > RSI_OS

    # [7] Session + ATR Filter (WIB / WITA)
    now_wib    = datetime.now(pytz.timezone("Asia/Jakarta"))
    h_wib      = now_wib.hour
    london     = 14 <= h_wib < 23
    ny         = h_wib >= 20 or h_wib < 5
    r["session_ok"] = london or ny
    r["atr_ok"]     = atr >= MIN_ATR

    # ── KEPUTUSAN ENTRY ───────────────────────────────────────
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
        "long":  long_ok,
        "short": short_ok,
        "score": score,
        "price": close,
        "atr":   atr,
        "rsi":   row["rsi"],
        "r":     r,
        "time":  df.iloc[i]["time"],
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
        sl        = price - sl_dist
        tp        = price + tp_dist
    else:
        direction = "SHORT 🔴"
        sl        = price + sl_dist
        tp        = price - tp_dist

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

    msg = (
        f"{emoji} *XAUUSD SIGNAL — {grade}*\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🕐 {sig['time'].strftime('%H:%M')} WIB | Score: *{score}/7*\n"
        f"📍 Type: *{direction}*\n\n"
        f"💰 Entry  : `{price:.2f}`\n"
        f"🛑 SL     : `{sl:.2f}`\n"
        f"🎯 TP     : `{tp:.2f}`\n"
        f"📊 R:R    : 1:{RR}\n"
        f"💼 Margin : ${MARGIN} × {LEVERAGE}x\n\n"
        f"📋 *Kondisi:*\n{cond}\n\n"
        f"⚠️ _Bukan financial advice. Selalu pakai manajemen risiko\\._"
    )
    return msg

# ══════════════════════════════════════════════════════════════
# 🚀  MAIN LOOP — jalan terus setiap 15 menit
# ══════════════════════════════════════════════════════════════
async def run_bot():
    bot = Bot(token=TELEGRAM_TOKEN)
    print(f"✅ GoldBot Telegram aktif | Scan tiap {SCAN_INTERVAL//60} menit")

    await bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text=(
            "🤖 *XAUUSD GoldBot v2\\.7 — AKTIF*\n"
            "Scanning M15 · 7 Kondisi Teknikal\n"
            "Data: Binance XAUUSDT 📡"
        ),
        parse_mode=ParseMode.MARKDOWN_V2
    )

    while True:
        try:
            now = datetime.now().strftime("%H:%M:%S")
            print(f"[{now}] Scanning market...")

            df  = get_candles()
            sig = check_signal(df)

            if sig["long"] or sig["short"]:
                msg = format_message(sig)
                await bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=msg,
                    parse_mode=ParseMode.MARKDOWN
                )
                direction = "LONG" if sig["long"] else "SHORT"
                print(f"✅ Sinyal dikirim: {direction} | Score {sig['score']}/7 | Price {sig['price']:.2f}")
            else:
                print(f"⏳ No signal | Score {sig['score']}/7 | RSI {sig['rsi']:.1f} | Price {sig['price']:.2f}")

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
