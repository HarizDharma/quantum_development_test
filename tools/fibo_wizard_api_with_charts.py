
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fibonacci AI Futures — API Wizard (Bitget) — WITH CHARTS
========================================================
- Semua fitur versi FIXED + output CHART saat BACKTEST.
- Chart yang disimpan:
  1) equity_curve.png
  2) drawdown_curve.png
  3) price_trades.png (Close + EMA + titik entry/exit)

Catatan:
- Menggunakan matplotlib standar (tanpa seaborn). Satu plot per figure.
- Jika tersedia, akan coba candlestick via `mplfinance`; jika tidak, fallback ke line chart.
"""

import os, sys, json, time, math, warnings
import numpy as np
import pandas as pd

from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

RESET="\033[0m"; BOLD="\033[1m"; CYAN="\033[36m"; GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"

def say(x): print(x)
def header(x): print(f"\n{BOLD}{CYAN}=== {x} ==={RESET}")
def ask(q, default="", example=""):
    qtxt = f"{BOLD}{q}{RESET}"
    if example: qtxt += f"\n{YELLOW}Contoh:{RESET} {example}"
    if default != "": qtxt += f"\nTekan Enter untuk default [{default}]"
    print(qtxt)
    ans = input("> ").strip()
    return ans if ans else default

# ---------- Helpers ----------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def normalize_timeframe(tf: str) -> str:
    tf = tf.strip().lower()
    if tf.isdigit():
        return f"{tf}m"
    allowed_units = ("m","h","d","w")
    if tf and tf[-1] in allowed_units and tf[:-1].isdigit():
        return tf
    num = "".join([c for c in tf if c.isdigit()])
    if num:
        return f"{num}m"
    raise ValueError(f"Timeframe tidak valid: {tf}. Gunakan 5m/15m/1h/4h/1d.")

def to_df(ohlcv):
    if not ohlcv:
        raise RuntimeError("API mengembalikan 0 bar OHLCV.")
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df[["datetime","open","high","low","close","volume"]]

# ---------- Indicators & Fib ----------
def ema(s, span): return s.ewm(span=span, adjust=False).mean()

def rsi(series, window=14):
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)
    ru = pd.Series(up, index=series.index).rolling(window).mean()
    rd = pd.Series(dn, index=series.index).rolling(window).mean()
    rs = ru / (rd + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    m = ema(series, fast) - ema(series, slow)
    s = ema(m, signal)
    return m, s, m - s

def atr(df, window=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def detect_swings(df, left=3, right=3):
    H, L, n = df["high"].values, df["low"].values, len(df)
    hi = np.zeros(n, dtype=bool); lo = np.zeros(n, dtype=bool)
    for i in range(left, n - right):
        win_h = H[i-left:i+right+1]
        win_l = L[i-left:i+right+1]
        if H[i] == win_h.max() and (win_h.argmax() == left):
            hi[i] = True
        if L[i] == win_l.min() and (win_l.argmin() == left):
            lo[i] = True
    return pd.Series(hi, index=df.index), pd.Series(lo, index=df.index)

def last_idx(mask: pd.Series) -> pd.Series:
    mask_bool = mask.astype(bool).values
    idx = np.arange(len(mask_bool))
    last = np.where(mask_bool, idx, np.nan).astype(float)
    last = pd.Series(last).ffill().fillna(-1).astype(int)
    last.index = mask.index
    return last

def fib_levels(df, hi_idx, lo_idx):
    if hi_idx < 0 or lo_idx < 0: return {}
    hi = float(df.loc[hi_idx, "high"]); lo = float(df.loc[lo_idx, "low"])
    diff = hi - lo
    up_leg = lo_idx > hi_idx
    if up_leg:
        return {"trend": 1, "f382": hi - 0.382 * diff, "f500": hi - 0.500 * diff, "f618": hi - 0.618 * diff}
    else:
        return {"trend": -1, "f382": lo + 0.382 * diff, "f500": lo + 0.500 * diff, "f618": lo + 0.618 * diff}

def fib_feats(df, left=3, right=3):
    hi_m, lo_m = detect_swings(df, left, right)
    hi_i, lo_i = last_idx(hi_m), last_idx(lo_m)
    out = pd.DataFrame(index=df.index)
    price = df["close"].values
    d382 = np.full(len(df), np.nan); d500 = np.full(len(df), np.nan); d618 = np.full(len(df), np.nan); tr = np.zeros(len(df))
    for i in range(len(df)):
        f = fib_levels(df, hi_i.iloc[i], lo_i.iloc[i])
        if not f: continue
        tr[i] = f["trend"]
        d382[i] = abs(price[i] - f["f382"]) / max(1e-12, price[i])
        d500[i] = abs(price[i] - f["f500"]) / max(1e-12, price[i])
        d618[i] = abs(price[i] - f["f618"]) / max(1e-12, price[i])
    out["fib_dist_382"] = d382; out["fib_dist_500"] = d500; out["fib_dist_618"] = d618; out["fib_trend"] = tr
    out["fib_score"] = (1/(1+out["fib_dist_500"])) + (1/(1+out["fib_dist_618"])) + 0.5*(1/(1+out["fib_dist_382"]))
    return out

# ---------- Dataset & Labels ----------
def build_features(df: pd.DataFrame):
    out = df.copy()
    out["ema_fast"] = ema(out["close"], 9)
    out["ema_slow"] = ema(out["close"], 21)
    out["rsi"] = rsi(out["close"], 14)
    m, s, h = macd(out["close"], 12, 26, 9)
    out["macd"] = m; out["macd_signal"] = s; out["macd_hist"] = h
    out["ret_1"] = out["close"].pct_change(1)
    out["vol_10"] = out["ret_1"].rolling(10).std()
    f = fib_feats(out, 3, 3)
    out = pd.concat([out, f], axis=1)
    out = out.dropna().reset_index(drop=True)
    return out, ["ema_fast","ema_slow","rsi","macd","macd_signal","macd_hist","ret_1","vol_10",
                 "fib_dist_382","fib_dist_500","fib_dist_618","fib_trend","fib_score"]

def make_labels(df: pd.DataFrame, horizon=6, threshold=0.001, three_class=False):
    fwd = df["close"].shift(-horizon)/df["close"] - 1.0
    if three_class:
        y = pd.Series(0, index=df.index, dtype=int)
        y[fwd > threshold] = 1
        y[fwd < -threshold] = -1
        return y
    else:
        y = pd.Series(np.nan, index=df.index, dtype=float)
        y[fwd > threshold] = 1
        y[fwd < -threshold] = -1
        return y

# ---------- API fetch ----------
def fetch_bitget_ohlcv(symbol="BTC/USDT:USDT", timeframe="15m", total=3000):
    import ccxt
    tf = normalize_timeframe(timeframe)
    if tf != timeframe:
        print(f"{YELLOW}Timeframe '{timeframe}' dinormalisasi ke '{tf}'.{RESET}")
    ex = ccxt.bitget({"options": {"defaultType": "swap"}, "enableRateLimit": True})
    ex.load_markets()
    left = total; since = None; rows = []; per = min(1000, total)
    while left > 0:
        chunk = ex.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=min(per, left))
        if not chunk: break
        rows += chunk; since = chunk[-1][0] + 1; left -= len(chunk)
        time.sleep(ex.rateLimit / 1000.0)
    return to_df(rows)

# ---------- ML core ----------
def train_model_api(symbol, timeframe, candles, model_path, horizon=6, threshold=0.001, three_class=False):
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report
        import joblib
    except Exception as e:
        print(f"{RED}Butuh scikit-learn & joblib. Install: pip install scikit-learn joblib{RESET}")
        raise

    df = fetch_bitget_ohlcv(symbol, timeframe, candles)
    Xdf, feat_cols = build_features(df)
    y_all = make_labels(Xdf, horizon, threshold, three_class=three_class)

    if not three_class:
        mask = y_all.notna()
        Xdf = Xdf.loc[mask].copy()
        y = y_all.loc[mask].astype(int)
    else:
        y = y_all

    if len(Xdf) < 300:
        raise RuntimeError("Data terlalu sedikit setelah feature engineering. Tambah candles atau ubah timeframe.")

    X = Xdf[feat_cols].values
    cut = int(len(X) * 0.8)
    Xtr, ytr = X[:cut], y.values[:cut]
    Xte, yte = X[cut:], y.values[cut:]

    clf = RandomForestClassifier(
        n_estimators=400, max_depth=7, min_samples_leaf=20, random_state=1337,
        class_weight="balanced"
    )
    clf.fit(Xtr, ytr)

    rep = classification_report(yte, clf.predict(Xte), digits=3, zero_division=0)
    print(rep)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({
        "model": clf,
        "feat_cols": feat_cols,
        "horizon": horizon,
        "threshold": threshold,
        "three_class": three_class
    }, model_path)
    print(f"{GREEN}Model tersimpan:{RESET} {model_path}")

# ---------- Charts helpers ----------
def save_equity_curve(eq: pd.Series, out_dir: str, fname="equity_curve.png"):
    import matplotlib.pyplot as plt
    ensure_dir(out_dir)
    plt.figure(figsize=(10,4))
    plt.plot(eq.index, eq.values)
    plt.title("Equity Curve (1 = start)")
    plt.xlabel("Index")
    plt.ylabel("Equity")
    path = os.path.join(out_dir, fname)
    plt.tight_layout(); plt.savefig(path); plt.close()
    return path

def save_drawdown_curve(eq: pd.Series, out_dir: str, fname="drawdown_curve.png"):
    import matplotlib.pyplot as plt
    peak = eq.cummax()
    dd = eq/peak - 1.0
    ensure_dir(out_dir)
    plt.figure(figsize=(10,4))
    plt.plot(dd.index, dd.values)
    plt.title("Drawdown")
    plt.xlabel("Index")
    plt.ylabel("DD")
    path = os.path.join(out_dir, fname)
    plt.tight_layout(); plt.savefig(path); plt.close()
    return path

def save_price_trades_chart(Xdf: pd.DataFrame, trades: pd.DataFrame, out_dir: str, fname="price_trades.png"):
    """
    Jika mplfinance tersedia => candlestick; else => line (close) + EMA + marker entry/exit.
    """
    ensure_dir(out_dir)
    path = os.path.join(out_dir, fname)
    try:
        import mplfinance as mpf
        dfa = Xdf.copy()
        dfa = dfa.set_index(pd.to_datetime(dfa["datetime"])).copy()
        dfa = dfa[["open","high","low","close","volume"]]
        aps = []
        # EMA addplots
        aps.append(mpf.make_addplot(Xdf["ema_fast"].values, panel=0))
        aps.append(mpf.make_addplot(Xdf["ema_slow"].values, panel=0))
        # Markers entry/exit via scatter on close
        ent_idx = trades["entry_idx"].values
        ex_idx  = trades["exit_idx"].values
        closes = Xdf["close"].values
        ent_y = closes[ent_idx]
        ex_y  = closes[ex_idx]
        ent_x = Xdf.index[ent_idx]
        ex_x  = Xdf.index[ex_idx]
        aps.append(mpf.make_addplot(ent_y, scatter=True, markersize=20, panel=0))
        aps.append(mpf.make_addplot(ex_y, scatter=True, markersize=20, panel=0))
        mpf.plot(dfa, type="candle", addplot=aps, style="classic", savefig=path)
        return path
    except Exception:
        import matplotlib.pyplot as plt
        # Line fallback
        t = pd.to_datetime(Xdf["datetime"])
        plt.figure(figsize=(12,5))
        plt.plot(t, Xdf["close"].values, label="Close")
        plt.plot(t, Xdf["ema_fast"].values, label="EMA 9")
        plt.plot(t, Xdf["ema_slow"].values, label="EMA 21")
        # Markers
        ent_idx = trades["entry_idx"].values
        ex_idx  = trades["exit_idx"].values
        plt.scatter(t.iloc[ent_idx], Xdf["close"].iloc[ent_idx].values, marker="^", s=50, label="Entry")
        plt.scatter(t.iloc[ex_idx],  Xdf["close"].iloc[ex_idx].values,  marker="v", s=50, label="Exit")
        plt.legend()
        plt.title("Price with Entries/Exits")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.tight_layout(); plt.savefig(path); plt.close()
        return path

# ---------- Backtest ----------
def backtest_api(symbol, timeframe, candles, model_path, fees=0.0004, slippage=0.5,
                 horizon=12, tp_atr=0.7, sl_atr=1.2, fib_eps=0.004, proba=0.65,
                 save_charts=False, charts_dir="charts"):
    try:
        import joblib
    except Exception:
        print(f"{RED}Butuh joblib. Install: pip install joblib{RESET}")
        raise

    bundle = joblib.load(model_path)
    df = fetch_bitget_ohlcv(symbol, timeframe, candles)
    Xdf, feat_cols = build_features(df)
    Xdf["ATR"] = atr(Xdf, 14)
    Xdf = Xdf.dropna().reset_index(drop=True)

    clf = bundle["model"]; cols = bundle["feat_cols"]
    proba_all = clf.predict_proba(Xdf[cols])
    classes = list(clf.classes_)
    i_minus = classes.index(-1) if -1 in classes else None
    i_plus  = classes.index(1)  if  1 in classes else None

    price = Xdf["close"].values; high = Xdf["high"].values; low = Xdf["low"].values
    ema_f = Xdf["ema_fast"].values; ema_s = Xdf["ema_slow"].values
    d500 = Xdf["fib_dist_500"].values; d618 = Xdf["fib_dist_618"].values; ftr = Xdf["fib_trend"].values; A = Xdf["ATR"].values

    N = len(Xdf); trades = []; i = 0
    while i < N - 1:
        up = (ema_f[i] > ema_s[i]) and (ftr[i] > 0)
        dn = (ema_f[i] < ema_s[i]) and (ftr[i] < 0)
        pL = proba_all[i, i_plus] if i_plus is not None else 0.0
        pS = proba_all[i, i_minus] if i_minus is not None else 0.0
        goL = up and (pL >= proba) and (min(d500[i], d618[i]) <= fib_eps)
        goS = dn and (pS >= proba) and (min(d500[i], d618[i]) <= fib_eps)
        if not (goL or goS):
            i += 1; continue

        direction = 1 if goL else -1
        entry_idx = i + 1
        if entry_idx >= N: break
        ep = price[entry_idx] + (slippage * (1 if direction == 1 else -1))
        tp = ep + direction * tp_atr * A[entry_idx]
        sl = ep - direction * sl_atr * A[entry_idx]

        exit_idx = None; exit_price = None; reason = None
        for j in range(entry_idx + 1, min(N, entry_idx + horizon + 1)):
            hit_tp = (high[j] >= tp) if direction == 1 else (low[j] <= tp)
            hit_sl = (low[j] <= sl) if direction == 1 else (high[j] >= sl)
            if hit_tp and hit_sl: hit = "SL"     # konservatif
            elif hit_tp:          hit = "TP"
            elif hit_sl:          hit = "SL"
            else:                 hit = None
            if hit:
                exit_idx = j; exit_price = tp if hit == "TP" else sl; reason = hit; break
        if exit_idx is None:
            exit_idx = min(N - 1, entry_idx + horizon); exit_price = price[exit_idx]; reason = "TIME"
        ret = (exit_price - ep) / ep * direction
        ret -= fees * 2
        trades.append({
            "entry_idx": entry_idx, "exit_idx": exit_idx, "dir": direction, "ret": float(ret),
            "reason": reason, "entry_price": float(ep), "exit_price": float(exit_price),
            "entry_time": str(Xdf["datetime"].iloc[entry_idx]), "exit_time": str(Xdf["datetime"].iloc[exit_idx])
        })
        i = exit_idx

    if not trades:
        print(f"{YELLOW}Tidak ada trade dengan filter saat ini. Longgarkan proba/fib_eps/horizon.{RESET}")
        return

    td = pd.DataFrame(trades)
    eq = (1.0 + td["ret"]).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    stats = {
        "trades": int(len(td)),
        "win_rate": float((td["ret"] > 0).mean()),
        "gross_return": float(eq.iloc[-1] - 1.0),
        "max_drawdown": float(dd.min()),
        "tp_rate": float((td["reason"] == "TP").mean()),
        "sl_rate": float((td["reason"] == "SL").mean()),
        "time_exits": float((td["reason"] == "TIME").mean())
    }
    print(json.dumps(stats, indent=2))

    if save_charts:
        out_dir = ensure_dir(charts_dir)
        eq_path = save_equity_curve(eq, out_dir, "equity_curve.png")
        dd_path = save_drawdown_curve(eq, out_dir, "drawdown_curve.png")
        price_path = save_price_trades_chart(Xdf, td, out_dir, "price_trades.png")
        say(f"{GREEN}Charts saved:{RESET}")
        say(f" - {eq_path}")
        say(f" - {dd_path}")
        say(f" - {price_path}")

# ---------- Private endpoints (STATUS / PAPER / LIVE) ----------
def bitget_auth_instance(kind="swap"):
    try:
        import ccxt
    except Exception:
        print(f"{RED}Butuh ccxt. Install: pip install ccxt{RESET}")
        raise
    ex = ccxt.bitget({
        "apiKey": os.getenv("BITGET_KEY"),
        "secret": os.getenv("BITGET_SECRET"),
        "password": os.getenv("BITGET_PASSWORD"),
        "enableRateLimit": True,
    })
    ex.options["defaultType"] = kind
    ex.load_markets()
    return ex

def show_spot_balances():
    ex = bitget_auth_instance("spot")
    bal = ex.fetch_balance()
    say(f"{GREEN}Spot balances (ringkas):{RESET}")
    for sym in ("USDT","BTC","ETH"):
        tot = bal.get("total", {}).get(sym)
        if tot is not None:
            say(f" - {sym}: total={bal['total'][sym]} free={bal['free'][sym]} used={bal['used'][sym]}")
    return bal

def show_futures_position(symbol="BTC/USDT:USDT"):
    ex = bitget_auth_instance("swap")
    try:
        poss = ex.fetch_positions([symbol])
    except Exception:
        poss = ex.fetch_positions()
    if not poss:
        say(f"{YELLOW}Tidak ada posisi futures aktif.{RESET}")
        return None
    pos = None
    for p in poss:
        if p.get("symbol") == symbol and abs(p.get("contracts", 0) or 0) > 0:
            pos = p; break
    if not pos:
        say(f"{YELLOW}Tidak ada posisi aktif untuk {symbol}.{RESET}")
        return None
    say(f"{GREEN}Posisi {symbol}:{RESET} side={pos.get('side')} contracts={pos.get('contracts')} entry={pos.get('entryPrice')} PnL={pos.get('unrealizedPnl')}%={pos.get('percentage')}")
    return pos

def close_futures_position(symbol="BTC/USDT:USDT"):
    ex = bitget_auth_instance("swap")
    pos = show_futures_position(symbol)
    if not pos: return
    side = pos.get("side"); contracts = pos.get("contracts") or 0
    if contracts == 0:
        say(f"{YELLOW}Contracts=0, tidak ada yang ditutup.{RESET}")
        return
    px = ex.fetch_ticker(symbol)["last"]
    mkt = ex.market(symbol); csize = mkt.get("contractSize") or 1.0
    amount = abs(contracts) if contracts else max(1e-6, abs(pos.get("notional", 0))/px/csize)
    opp = "sell" if side == "long" else "buy"
    try:
        ex.create_order(symbol, "market", opp, amount, None, {"reduceOnly": True})
        say(f"{GREEN}Close posisi terkirim: {opp} {amount} @~{px}{RESET}")
    except Exception as e:
        say(f"{YELLOW}Gagal close posisi: {e}{RESET}")

def paper_loop(symbol, timeframe, model_path, poll, size_usd):
    try:
        import ccxt, joblib
    except Exception:
        print(f"{RED}Butuh ccxt & joblib. Install: pip install ccxt joblib{RESET}")
        return
    ex = ccxt.bitget({"options": {"defaultType": "swap"}, "enableRateLimit": True})
    ex.load_markets()
    bundle = joblib.load(model_path); clf = bundle["model"]; cols = bundle["feat_cols"]
    say(f"{GREEN}Paper mode dimulai. Ctrl+C untuk stop.{RESET}")
    while True:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=normalize_timeframe(timeframe), limit=400)
        df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        Xf, _ = build_features(df[["datetime","open","high","low","close","volume"]])
        prob = clf.predict_proba(Xf[cols])
        classes = list(clf.classes_)
        ip = classes.index(1) if 1 in classes else None
        im = classes.index(-1) if -1 in classes else None
        pL = prob[-1, ip] if ip is not None else 0.0
        pS = prob[-1, im] if im is not None else 0.0
        last_px = float(Xf["close"].iloc[-1])
        say(f"{Xf['datetime'].iloc[-1]} {symbol} px={last_px:.4f} P(long)={pL:.2f} P(short)={pS:.2f}")
        time.sleep(poll)

def live_loop(symbol, timeframe, model_path, size_usd):
    try:
        import ccxt, joblib
    except Exception:
        print(f"{RED}Butuh ccxt & joblib. Install: pip install ccxt joblib{RESET}")
        return
    ex = ccxt.bitget({
        "apiKey": os.getenv("BITGET_KEY"),
        "secret": os.getenv("BITGET_SECRET"),
        "password": os.getenv("BITGET_PASSWORD"),
        "enableRateLimit": True,
    })
    ex.options["defaultType"] = "swap"; ex.load_markets()
    bundle = joblib.load(model_path); clf = bundle["model"]; cols = bundle["feat_cols"]
    say(f"{GREEN}LIVE mulai (order MARKET reduceOnly saat edge > 0.2). Ctrl+C untuk stop.{RESET}")
    while True:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=normalize_timeframe(timeframe), limit=400)
        df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        Xf, _ = build_features(df[["datetime","open","high","low","close","volume"]])
        prob = clf.predict_proba(Xf[cols])
        classes = list(clf.classes_)
        ip = classes.index(1) if 1 in classes else None
        im = classes.index(-1) if -1 in classes else None
        pL = prob[-1, ip] if ip is not None else 0.0
        pS = prob[-1, im] if im is not None else 0.0
        last_px = float(Xf["close"].iloc[-1])
        say(f"{Xf['datetime'].iloc[-1]} {symbol} px={last_px:.4f} P(long)={pL:.2f} P(short)={pS:.2f}")
        edge = pL - pS
        if abs(edge) > 0.2:
            mkt = ex.market(symbol); csize = mkt.get("contractSize") or 1.0
            amount = max(1e-6, size_usd / last_px / csize)
            side = "buy" if edge > 0 else "sell"
            try:
                ex.create_order(symbol, "market", side, amount, None, {"reduceOnly": True})
                say(f"{GREEN}ORDER {side.upper()} {amount:.6f} @~{last_px}{RESET}")
            except Exception as e:
                say(f"{YELLOW}Order gagal: {e}{RESET}")
        time.sleep(30)

# ---------- Wizard ----------
def run():
    header("Fibonacci AI Futures — API Wizard (Bitget) — WITH CHARTS")
    mode = ask("Pilih MODE: TRAIN / BACKTEST / PAPER / LIVE / STATUS", default="BACKTEST", example="TRAIN").upper()

    if mode == "TRAIN":
        symbol = ask("Symbol (USDT perp):", default="BTC/USDT:USDT", example="ARB/USDT:USDT")
        timeframe = ask("Timeframe (angka menit atau 5m/15m/1h/4h/1d):", default="15m", example="15")
        candles = int(ask("Jumlah candle historis:", default="5000", example="8000"))
        model_path = ask("Simpan model ke:", default="models/fibo_rf_api.pkl", example="models/BTC_15m.pkl")
        horizon = int(ask("Horizon label (bar):", default="6", example="8"))
        thr = float(ask("Ambang label |return|:", default="0.001", example="0.0015"))
        lbl = ask("Skema label? 2=naik/turun (drop netral), 3=naik/netral/turun", default="2", example="2").strip()
        three_class = (lbl == "3")
        header("RINGKASAN"); print(json.dumps({
            "symbol":symbol,"timeframe":timeframe,"candles":candles,"model":model_path,
            "horizon":horizon,"threshold":thr,"labels":"3-class" if three_class else "2-class"
        }, indent=2))
        if ask("Lanjut TRAIN? (Y/N)", default="Y", example="Y").upper()=="Y":
            train_model_api(symbol, timeframe, candles, model_path, horizon, thr, three_class=three_class)

    elif mode == "BACKTEST":
        symbol = ask("Symbol (USDT perp):", default="BTC/USDT:USDT", example="ETH/USDT:USDT")
        timeframe = ask("Timeframe:", default="15m", example="5m / 15m / 1h")
        candles = int(ask("Jumlah candle historis:", default="5000", example="8000"))
        model_path = ask("Path model:", default="models/fibo_rf_api.pkl", example="models/BTC_15m.pkl")
        fees = float(ask("Fee per side (desimal):", default="0.0004", example="0.0004"))
        slip = float(ask("Slippage ($):", default="0.5", example="0.5"))
        horizon = int(ask("Maks durasi trade (bar):", default="12", example="12"))
        tp = float(ask("TP ATR:", default="0.7", example="0.5–0.9"))
        sl = float(ask("SL ATR:", default="1.2", example="1.0–1.6"))
        fib_eps = float(ask("Max jarak ke 50/61.8 (relatif):", default="0.004", example="0.002–0.006"))
        proba = float(ask("Ambang probabilitas:", default="0.65", example="0.7–0.8 (lebih selektif)"))
        savec = ask("Simpan chart hasil backtest? (Y/N)", default="Y", example="Y").upper()=="Y"
        cdir = ask("Folder untuk chart:", default="charts", example="charts")
        header("RINGKASAN"); print(json.dumps({
            "symbol":symbol,"timeframe":timeframe,"candles":candles,"model":model_path,
            "fees":fees,"slippage":slip,"horizon":horizon,"tp_atr":tp,"sl_atr":sl,"fib_eps":fib_eps,"proba":proba,
            "save_charts": savec, "charts_dir": cdir
        }, indent=2))
        if ask("Jalankan BACKTEST? (Y/N)", default="Y", example="Y").upper()=="Y":
            backtest_api(symbol, timeframe, candles, model_path, fees, slip, horizon, tp, sl, fib_eps, proba,
                         save_charts=savec, charts_dir=cdir)

    elif mode == "PAPER":
        symbol = ask("Symbol:", default="BTC/USDT:USDT", example="ARB/USDT:USDT")
        timeframe = ask("Timeframe:", default="15m", example="5m / 15m")
        model_path = ask("Path model:", default="models/fibo_rf_api.pkl", example="models/BTC_15m.pkl")
        poll = int(ask("Interval refresh (detik):", default="60", example="60"))
        size = float(ask("Ukuran notional per sinyal (USD):", default="50", example="25"))
        header("INFO"); say("Menampilkan probabilitas long/short periodik.")
        if ask("Mulai PAPER sekarang? (Y/N)", default="N", example="Y").upper()=="Y":
            paper_loop(symbol, timeframe, model_path, poll, size)

    elif mode == "LIVE":
        say("Pastikan env: BITGET_KEY, BITGET_SECRET, BITGET_PASSWORD sudah di-set.")
        symbol = ask("Symbol:", default="BTC/USDT:USDT", example="ETH/USDT:USDT")
        timeframe = ask("Timeframe:", default="15m", example="5m / 15m")
        model_path = ask("Path model:", default="models/fibo_rf_api.pkl", example="models/BTC_15m.pkl")
        size = float(ask("Ukuran notional order (USD):", default="25", example="10"))
        header("INFO"); say("LIVE akan kirim order MARKET reduceOnly saat selisih proba > 0.2")
        if ask("Mulai LIVE sekarang? (Y/N)", default="N", example="Y").upper()=="Y":
            live_loop(symbol, timeframe, model_path, size)

    elif mode == "STATUS":
        header("STATUS: Spot & Futures")
        if ask("Tampilkan spot balances? (Y/N)", default="Y", example="Y").upper()=="Y":
            show_spot_balances()
        sym = ask("Cek posisi futures untuk symbol:", default="BTC/USDT:USDT", example="ARB/USDT:USDT")
        show_futures_position(sym)
        if ask("Close posisi futures sekarang? (Y/N)", default="N", example="Y").upper()=="Y":
            close_futures_position(sym)

    else:
        say("Mode tidak dikenal. Pilih TRAIN/BACKTEST/PAPER/LIVE/STATUS.")

if __name__ == "__main__":
    run()
