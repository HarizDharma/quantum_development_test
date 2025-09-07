import ccxt
import openai
import os
import asyncio
import time
import pandas as pd
import talib
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

bitget = ccxt.bitget()
bitget.options["defaultType"] = "swap"

def get_ohlcv(symbol, timeframe="5m", limit=100000):
    data = bitget.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

def apply_technical_indicators(df):
    df["rsi"] = talib.RSI(df["close"], timeperiod=14)
    df["ema20"] = talib.EMA(df["close"], timeperiod=20)
    df["ema50"] = talib.EMA(df["close"], timeperiod=50)
    macd, macdsignal, _ = talib.MACD(df["close"])
    df["macd"] = macd
    df["macdsignal"] = macdsignal
    upper, middle, lower = talib.BBANDS(df["close"], timeperiod=20)
    df["bb_upper"] = upper
    df["bb_lower"] = lower
    return df

def ask_gpt_for_trade(symbol, last_row):
    prompt = f'''
    Ticker: {symbol}
    Harga: {last_row["close"]}
    RSI: {last_row["rsi"]}
    EMA20: {last_row["ema20"]}
    EMA50: {last_row["ema50"]}
    MACD: {last_row["macd"]}, MACD Signal: {last_row["macdsignal"]}
    BB Upper: {last_row["bb_upper"]}, BB Lower: {last_row["bb_lower"]}
    Volume: {last_row["volume"]}

    Berdasarkan data teknikal di atas, apakah ini saat yang tepat untuk entry LONG atau SHORT?
    Jika ya, berikan:
    - Arah: LONG / SHORT
    - Alasan singkat
    - TP (Take Profit)
    - SL (Stop Loss)

    Jika belum saatnya entry, katakan 'WAIT'
    '''

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200
    )
    return response.choices[0].message.content

def main():
    print("=== AI Trading Bot — Bitget + ChatGPT ===")
    symbol = input("Masukkan simbol (misalnya: BTC/USDT:USDT): ").strip()

    while True:
        try:
            df = get_ohlcv(symbol)
            df = apply_technical_indicators(df)
            signal = ask_gpt_for_trade(symbol, df.iloc[-1])
            print(f"\n[AI GPT SIGNAL] {symbol}:\n{signal}")
            os.makedirs("logs", exist_ok=True)
            with open("logs/ai_signals.csv", "a") as f:
                f.write(f"{pd.Timestamp.now()},{symbol},{signal.replace(',', ' ')}\n")
        except Exception as e:
            print(f"Error: {e}")

        time.sleep(10)

if __name__ == "__main__":
    main()
