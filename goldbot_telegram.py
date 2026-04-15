#!/usr/bin/env python3
"""
XAUUSD Gold Signal Bot — Telegram v4.0
Rules    : 5 Syarat Asli (Trend, Volume Anomaly, Pinbar, Dynamic Wall, Static S/R)
Scan     : Tiap 5 menit (diam kalau tidak ada sinyal)
Sinyal   : Hanya kirim kalau semua 5 syarat terpenuhi
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
# ⚙️  KONFIGURASI
# ══════════════════════════════════════════════════════════════
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN",   "ISI_TOKEN_TELEGRAM")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "ISI_CHAT_ID_TELEGRAM")
TWELVE_DATA_KEY  = os.environ.get("TWELVE_DATA_KEY",  "ISI_API_KEY_TWELVEDATA")

SCAN_INTERVAL = 300       # Scan tiap 5 menit
SYMBOL        = "XAU/USD"
TIMEFRAME     = "15min"
CANDLE_LIMIT  = 250

# ── Parameter Indikator ───────────────────────────────────────
EMA200    = 200   # [1] Trend Alignment
MA99      = 99    # [4] Dynamic Wall
BB_P      = 20    # [4] Dynamic Wall (Bollinger)
BB_DEV    = 2.0
ATR_P     = 14
RSI_P     = 14
VOL_MULT  = 1.2   # [2] Volume Anomaly — lonjakan 1.2x MA20
MA20_VOL  = 20
SR_LOOK   = 100   # [5] Static S/R — 100 candle terakhir

RR        = 2.0
MARGIN    = 3
LEVERAGE  = 50

# Cooldown — hindari spam sinyal sama
last_signal = {"direction": None, "price": 0.0}

# ══════════════════════════════════════════════════════════════
# 📥  AMBIL DATA
# ══════════════════════════════════════════════════════════════
def get_candles():
    url    = "https://api.twelvedata.com/time_series"
    params = {
        "symbol":     SYMBOL,
        "interval":   TIMEFRAME,
        "outputsize": CANDLE_LIMIT,
        "apikey":     TWELVE_DATA_KEY,
        "order":      "ASC",
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("status") == "error":
        raise Exception(f"Twelve Data error: {data.get('message')}")
    values = data.get("values", [])
    if not values:
        raise Exception("Tidak ada data")
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
    return df.sort_values("time").reset_index(drop=True)

# ══════════════════════════════════════════════════════════════
# 📐  INDIKATOR
# ══════════════════════════════════════════════════════════════
def compute_indicators(df):
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # EMA200 — Trend
    df["ema200"]   = c.ewm(span=EMA200, adjust=False).mean()

    # MA99 — Dynamic Wall
    df["ma99"]     = c.rolling(MA99).mean()

    # Bollinger Bands — Dynamic Wall
    df["bb_mid"]   = c.rolling(BB_P).mean()
    bb_std         = c.rolling(BB_P).std()
    df["bb_upper"] = df["bb_mid"] + BB_DEV * bb_std
    df["bb_lower"] = df["bb_mid"] - BB_DEV * bb_std

    # ATR
    tr             = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"]      = tr.rolling(ATR_P).mean()

    # Volume MA20 — Volume Anomaly
    df["vol_ma"]   = v.rolling(MA20_VOL).mean()

    return df

# ══════════════════════════════════════════════════════════════
# 🔍  CEK 5 SYARAT ASLI
# ══════════════════════════════════════════════════════════════
def check_signal(df):
    df    = compute_indicators(df)
    i     = -2  # Candle closed
    row   = df.iloc[i]
    close = row["close"]
    open_ = row["open"]
    high  = row["high"]
    low   = row["low"]
    atr   = row["atr"]
    r     = {}

    # ─────────────────────────────────────────────────────────
    # [1] TREND ALIGNMENT — Harga searah EMA200
    # ─────────────────────────────────────────────────────────
    r["trend_up"]   = close > row["ema200"]
    r["trend_down"] = close < row["ema200"]

    # ─────────────────────────────────────────────────────────
    # [2] VOLUME ANOMALY — Lonjakan volume >= 1.2x MA20
    # ─────────────────────────────────────────────────────────
    r["vol_spike"]  = (row["vol_ma"] > 0) and (row["volume"] >= row["vol_ma"] * VOL_MULT)

    # ─────────────────────────────────────────────────────────
    # [3] PRICE REJECTION (PINBAR) — Ekor >= 2x body
    # ─────────────────────────────────────────────────────────
    body          = abs(close - open_)
    upper_sh      = high - max(close, open_)
    lower_sh      = min(close, open_) - low
    min_body      = atr * 0.05
    r["bull_pin"] = (body < min_body * 4) and (lower_sh >= body * 2) and (lower_sh > upper_sh)
    r["bear_pin"] = (body < min_body * 4) and (upper_sh >= body * 2) and (upper_sh > lower_sh)

    # ─────────────────────────────────────────────────────────
    # [4] DYNAMIC WALLS — Sentuh MA99 atau outer Bollinger
    # ─────────────────────────────────────────────────────────
    zone          = atr * 0.3
    r["dyn_wall"] = (abs(low  - row["ma99"])     < zone or
                     abs(high - row["ma99"])     < zone or
                     abs(high - row["bb_upper"]) < zone or
                     abs(low  - row["bb_lower"]) < zone)

    # ─────────────────────────────────────────────────────────
    # [5] STATIC S/R — Swing high/low 100 candle terakhir
    # ─────────────────────────────────────────────────────────
    lookback     = df.iloc[i - SR_LOOK : i]
    sr_high      = lookback["high"].max()
    sr_low       = lookback["low"].min()
    r["near_sr"] = (abs(close - sr_high) < atr * 0.5 or
                    abs(close - sr_low)  < atr * 0.5)

    # ── KEPUTUSAN ─────────────────────────────────────────────
    long_ok  = (r["trend_up"]   and r["vol_spike"] and
                r["bull_pin"]   and r["dyn_wall"]  and r["near_sr"])

    short_ok = (r["trend_down"] and r["vol_spike"] and
                r["bear_pin"]   and r["dyn_wall"]  and r["near_sr"])

    score = sum([
        r["trend_up"] or r["trend_down"],
        r["vol_spike"],
        r["bull_pin"] or r["bear_pin"],
        r["dyn_wall"],
        r["near_sr"],
    ])

    return {
        "long":  long_ok, "short": short_ok,
        "score": score,   "price": close,
        "atr":   atr,     "r":     r,
        "time":  df.iloc[i]["time"],
    }

# ══════════════════════════════════════════════════════════════
# 📨  FORMAT PESAN SINYAL
# ══════════════════════════════════════════════════════════════
def format_signal(sig):
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

    cond = (
        f"  • Trend EMA200  : {'✅' if r['trend_up'] or r['trend_down'] else '❌'}\n"
        f"  • Volume Anomaly: {'✅' if r['vol_spike'] else '❌'}\n"
        f"  • Pinbar        : {'✅' if r['bull_pin'] or r['bear_pin'] else '❌'}\n"
        f"  • Dynamic Wall  : {'✅' if r['dyn_wall'] else '❌'}\n"
        f"  • Static S/R    : {'✅' if r['near_sr'] else '❌'}"
    )

    return (
        f"🔥 *GAS TRADE SEKARANG!*\n"
        f"🏆 *XAUUSD SIGNAL — 5/5*\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🕐 {sig['time'].strftime('%d/%m %H:%M')} WIB\n"
        f"⏱ TF: M15 | Data: Twelve Data\n"
        f"📍 Type: *{direction}*\n\n"
        f"💰 Entry : `{price:.2f}`\n"
        f"🛑 SL    : `{sl:.2f}`\n"
        f"🎯 TP    : `{tp:.2f}`\n"
        f"📊 R:R   : 1:{RR}\n"
        f"💼 Margin: ${MARGIN} × {LEVERAGE}x\n\n"
        f"📋 *Konfirmasi:*\n{cond}\n\n"
        f"⚠️ _Bukan financial advice. Eksekusi manual di FBS._"
    )

# ══════════════════════════════════════════════════════════════
# 🚀  MAIN LOOP — scan tiap 5 menit, diam kalau tidak ada sinyal
# ══════════════════════════════════════════════════════════════
async def run_bot():
    global last_signal
    bot = Bot(token=TELEGRAM_TOKEN)
    print(f"✅ GoldBot v4.0 aktif | 5 Syarat Asli | Scan tiap {SCAN_INTERVAL//60} menit")

    await bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text=(
            "🤖 *XAUUSD GoldBot v4.0 — AKTIF*\n"
            "📡 Data: Twelve Data XAU/USD M15\n"
            "📊 5 Syarat: Trend · Volume · Pinbar · DynWall · S/R\n"
            "🔕 Bot hanya kirim pesan saat sinyal muncul"
        ),
        parse_mode=ParseMode.MARKDOWN
    )

    while True:
        try:
            now = datetime.now().strftime("%H:%M:%S")
            df  = get_candles()
            sig = check_signal(df)

            if sig["long"] or sig["short"]:
                direction  = "LONG" if sig["long"] else "SHORT"
                price_diff = abs(sig["price"] - last_signal["price"])
                is_same    = (last_signal["direction"] == direction and price_diff < 2.0)

                if not is_same:
                    await bot.send_message(
                        chat_id=TELEGRAM_CHAT_ID,
                        text=format_signal(sig),
                        parse_mode=ParseMode.MARKDOWN
                    )
                    last_signal = {"direction": direction, "price": sig["price"]}
                    print(f"[{now}] ✅ SINYAL: {direction} | Score 5/5 | {sig['price']:.2f}")
                else:
                    print(f"[{now}] ⏭ Skip duplikat | {direction} | {sig['price']:.2f}")
            else:
                print(f"[{now}] ⏳ No signal | Score {sig['score']}/5 | {sig['price']:.2f}")

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
