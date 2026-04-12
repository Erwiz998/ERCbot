#!/usr/bin/env python3
"""
XAUUSD Gold Signal Bot — Telegram
Data     : Twelve Data API (XAU/USD)
Timeframe: M5 (5 menit)
Scan     : Setiap 5 menit
Sinyal   : 7 Kondisi Teknikal
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
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN",   "ISI_TOKEN_TELEGRAM")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "ISI_CHAT_ID_TELEGRAM")
TWELVE_DATA_KEY  = os.environ.get("TWELVE_DATA_KEY",  "ISI_API_KEY_TWELVEDATA")

SCAN_INTERVAL = 300       # Scan tiap 5 menit
SYMBOL        = "XAU/USD"
TIMEFRAME     = "15min"    # ✅ M5 timeframe
CANDLE_LIMIT  = 250       # 250 candle M5 = ~20 jam data

# Parameter indikator
EMA200    = 200
MA99      = 99
BB_P      = 20
BB_DEV    = 2.0
ATR_P     = 14
RSI_P     = 14
BODY_MULT = 1.5   # Momentum: body > 1.5x rata-rata
ATR_MULT  = 1.2   # ATR Breakout: range > 1.2x ATR
RSI_OB    = 65.0
RSI_OS    = 35.0
MIN_ATR   = 0.5   # ✅ Lebih kecil karena M5 (volatilitas per candle lebih kecil)
SR_LOOK   = 100
RR        = 2.0
MARGIN    = 3
LEVERAGE  = 50

# Cooldown — hindari spam sinyal sama
last_signal = {"direction": None, "price": 0.0}

# ══════════════════════════════════════════════════════════════
# 📥  AMBIL DATA DARI TWELVE DATA
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
        raise Exception("Tidak ada data dari Twelve Data")

    rows = []
    for v in values:
        rows.append({
            "time":  v["datetime"],
            "open":  float(v["open"]),
            "high":  float(v["high"]),
            "low":   float(v["low"]),
            "close": float(v["close"]),
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
    o = df["open"]

    df["ema200"]   = c.ewm(span=EMA200, adjust=False).mean()
    df["ma99"]     = c.rolling(MA99).mean()

    df["bb_mid"]   = c.rolling(BB_P).mean()
    bb_std         = c.rolling(BB_P).std()
    df["bb_upper"] = df["bb_mid"] + BB_DEV * bb_std
    df["bb_lower"] = df["bb_mid"] - BB_DEV * bb_std

    # ATR
    tr           = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"]    = tr.rolling(ATR_P).mean()

    # RSI
    delta        = c.diff()
    gain         = delta.clip(lower=0).rolling(RSI_P).mean()
    loss         = (-delta.clip(upper=0)).rolling(RSI_P).mean()
    rs           = gain / loss.replace(0, np.nan)
    df["rsi"]    = 100 - (100 / (1 + rs))

    # Momentum — rata-rata body 20 candle
    df["body"]   = (c - o).abs()
    df["body_ma"] = df["body"].rolling(20).mean()

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

    # [2] Momentum Candle — body > 1.5x rata-rata body
    body          = abs(close - open_)
    r["momentum"] = (row["body_ma"] > 0) and (body >= row["body_ma"] * BODY_MULT)

    # [3] Pinbar — ekor >= 2x body
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

    # [7] ATR Breakout + Session Filter
    candle_range    = high - low
    r["atr_break"]  = candle_range >= atr * ATR_MULT
    r["atr_ok"]     = atr >= MIN_ATR
    h_wib           = datetime.now(pytz.timezone("Asia/Jakarta")).hour
    r["session_ok"] = (14 <= h_wib < 23) or (h_wib >= 20) or (h_wib < 5)

    # ── KEPUTUSAN ─────────────────────────────────────────────
    long_ok  = (r["trend_up"]   and r["momentum"]  and r["bull_pin"] and
                r["dyn_wall"]   and r["near_sr"]   and r["rsi_bull_ok"] and
                r["atr_break"]  and r["atr_ok"]    and r["session_ok"])

    short_ok = (r["trend_down"] and r["momentum"]  and r["bear_pin"] and
                r["dyn_wall"]   and r["near_sr"]   and r["rsi_bear_ok"] and
                r["atr_break"]  and r["atr_ok"]    and r["session_ok"])

    score = sum([
        r["trend_up"] or r["trend_down"],
        r["momentum"],
        r["bull_pin"] or r["bear_pin"],
        r["dyn_wall"],
        r["near_sr"],
        r["rsi_bull_ok"] or r["rsi_bear_ok"],
        r["atr_break"] and r["atr_ok"] and r["session_ok"],
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
        f"  • Trend EMA200   : {'✅' if r['trend_up'] or r['trend_down'] else '❌'}\n"
        f"  • Momentum Candle: {'✅' if r['momentum'] else '❌'}\n"
        f"  • Pinbar         : {'✅' if r['bull_pin'] or r['bear_pin'] else '❌'}\n"
        f"  • Dynamic Wall   : {'✅' if r['dyn_wall'] else '❌'}\n"
        f"  • Static S/R     : {'✅' if r['near_sr'] else '❌'}\n"
        f"  • RSI Filter     : {'✅' if r['rsi_bull_ok'] or r['rsi_bear_ok'] else '❌'} ({sig['rsi']:.1f})\n"
        f"  • ATR+Session    : {'✅' if r['atr_break'] and r['session_ok'] else '❌'}"
    )

    return (
        f"{emoji} *XAUUSD SIGNAL — {grade}*\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🕐 {sig['time'].strftime('%d/%m %H:%M')} | Score: *{score}/7*\n"
        f"⏱ TF: M5 | Data: Twelve Data\n"
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
# 🚀  MAIN LOOP — scan tiap 5 menit
# ══════════════════════════════════════════════════════════════
async def run_bot():
    global last_signal
    bot = Bot(token=TELEGRAM_TOKEN)
    print(f"✅ GoldBot M5 aktif | Scan tiap {SCAN_INTERVAL//60} menit")

    await bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text=(
            "🤖 *XAUUSD GoldBot v3.1 — AKTIF*\n"
            "📡 Data: Twelve Data XAU/USD\n"
            "⏱ Timeframe: M5 | Scan: 5 menit\n"
            "📊 7 Kondisi Teknikal"
        ),
        parse_mode=ParseMode.MARKDOWN
    )

    while True:
        try:
            now = datetime.now().strftime("%H:%M:%S")
            print(f"[{now}] Scanning M5...")

            df  = get_candles()
            sig = check_signal(df)

            if sig["long"] or sig["short"]:
                direction  = "LONG" if sig["long"] else "SHORT"
                price_diff = abs(sig["price"] - last_signal["price"])
                is_same    = (last_signal["direction"] == direction and price_diff < 1.5)

                if not is_same:
                    await bot.send_message(
                        chat_id=TELEGRAM_CHAT_ID,
                        text=format_message(sig),
                        parse_mode=ParseMode.MARKDOWN
                    )
                    last_signal = {"direction": direction, "price": sig["price"]}
                    print(f"✅ Sinyal: {direction} | Score {sig['score']}/7 | {sig['price']:.2f}")
                else:
                    print(f"⏭ Skip duplikat | {direction} | {sig['price']:.2f}")
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
