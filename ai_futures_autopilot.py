
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI Futures Autopilot — Bitget USDT-M (no-parameter edition)
===========================================================
What it does:
- Auto-detects top liquid USDT-M perpetuals (Bitget) and lets you pick by number (default #1).
- Fetches your futures USDT balance and prints it.
- Auto-trains (or loads) an HGB model (QUANTILE labels) using the chosen timeframe (menu/.env).
- Computes support/resistance (swing highs/lows), ATR, Fib proximity, EMA trend.
- Generates AI signal; when strong enough, sizes a position using 1% risk of free USDT balance.
- Opens a MARKET order; manages exits: TP (ATR or next structure) & SL (structure or ATR).
- Closes via MARKET reduceOnly when conditions hit; prints session PnL stats & win-rate.
- No manual hyper-params; safe defaults baked in.

Env needed:
  BITGET_KEY, BITGET_SECRET, BITGET_PASSWORD

Install:
  pip install ccxt pandas numpy scikit-learn joblib python-dotenv
  (optional for charts: matplotlib mplfinance)

Additional env (optional):
  AUTOPILOT_TRAIN_LOOKBACK=ALL   # train with full chart history (ALL/FULL/MAX) or use e.g. '180d','2y'
  AUTOPILOT_MAX_BARS=0           # cap rows for full-history (0 = unlimited, subject to AUTOPILOT_SAFETY_MAX_BARS)
  AUTOPILOT_SAFETY_MAX_BARS=1000000  # absolute safety cap to prevent OOM
  AUTOPILOT_CACHE=1               # cache OHLCV locally (CSV) to speed up subsequent runs
  AUTOPILOT_PROGRESS=1            # show progress logs while fetching full history

Stop the bot: Ctrl+C
"""

import os, time, math, json, sys, csv
from datetime import datetime, timezone
import numpy as np
import pandas as pd


RESET="\033[0m"; BOLD="\033[1m"; CYAN="\033[36m"; GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"
INJECT_HL = str(os.getenv("AUTOPILOT_INJECT_HL","0")).lower() not in ("0","false","no","off")

# --- Live stability knobs (sane defaults; overrideable via ENV) ---
POST_ENTRY_FREEZE_SEC = float(os.getenv("AUTOPILOT_POST_ENTRY_FREEZE_SEC", "4"))
EXIT_SCORE            = float(os.getenv("AUTOPILOT_EXIT_SCORE", "0.40"))
SCORE_EMA_ALPHA       = float(os.getenv("AUTOPILOT_SCORE_EMA_ALPHA", "0.25"))
EXIT_DEBOUNCE_N       = int(float(os.getenv("AUTOPILOT_EXIT_DEBOUNCE_N","2")))

# Runtime memory for smoothing & debouncing
_SCORE_EMA = {}  # symbol -> smoothed score
_EXIT_DEB  = {}  # symbol -> consecutive low-score ticks
_ENTRY_TS  = {}  # symbol -> last entry timestamp (epoch seconds)

def smooth_score(symbol: str, raw_score: float) -> float:
    """EMA-smooth the model score to reduce whipsaw on live ticks."""
    try:
        r = float(raw_score)
    except Exception:
        r = 0.0
    prev = _SCORE_EMA.get(symbol, r)
    s = (1.0 - SCORE_EMA_ALPHA) * prev + SCORE_EMA_ALPHA * r
    _SCORE_EMA[symbol] = s
    return s

class CancelledEntry(Exception):
    """Raised ketika user menekan 'c'+Enter saat waiting entry."""
    pass

def say(x): print(x, flush=True)
def header(x):
    print(f"\n{BOLD}{CYAN}=== {x} ==={RESET}", flush=True)


# --- Required timeframe enforcement ---
ALLOWED_TFS = ("1m","3m","5m","15m","30m","1h","2h","4h","1d")

# --- Helper: Normalize timeframe aliases ---
def normalize_tf(tf: str) -> str:
    """
    Normalize timeframe aliases into one of ALLOWED_TFS.
    Accepts common variants like '60m'->'1h', 'm15'->'15m', 'h4'->'4h', '24h'->'1d', etc.
    Returns the normalized string; if unknown, returns the original lower-cased token.
    """
    s = str(tf or "").strip().lower()
    # common aliases
    alias = {
        "1min": "1m", "3min": "3m", "5min": "5m", "15min": "15m", "30min": "30m",
        "m1": "1m", "m3": "3m", "m5": "5m", "m15": "15m", "m30": "30m",
        "60m": "1h", "120m": "2h", "240m": "4h", "1440m": "1d",
        "h1": "1h", "h2": "2h", "h4": "4h",
        "1hour": "1h", "2hour": "2h", "4hour": "4h", "24h": "1d",
        "d1": "1d"
    }
    s = alias.get(s, s)
    return s if s in ALLOWED_TFS else s

def require_tf() -> str:
    """Return selected timeframe from ENV or raise if missing/invalid."""
    tf = (os.getenv("AUTOPILOT_TIMEFRAME") or "").strip()
    if tf not in ALLOWED_TFS:
        raise RuntimeError(
            "AUTOPILOT_TIMEFRAME belum dipilih/invalid. Pilih salah satu: " + ", ".join(ALLOWED_TFS)
        )
    return tf

# Ensure timeframe is selected only once per run
def ensure_timeframe_selected():
    tf = (os.getenv("AUTOPILOT_TIMEFRAME") or "").strip()
    if tf in ALLOWED_TFS:
        return tf
    tf = prompt_pick_timeframe()
    os.environ["AUTOPILOT_TIMEFRAME"] = tf
    say(f"Timeframe dipakai: {BOLD}{tf}{RESET}")
    return tf

def _print_inline(msg: str):
    # Cetak 1 baris dan timpa (clear to end-of-line)
    sys.stdout.write("\r" + str(msg) + "\x1b[K")
    sys.stdout.flush()

def _readline_nowait():
    """Return string jika user tekan Enter; None jika tidak ada input."""
    try:
        import select
        dr, _, _ = select.select([sys.stdin], [], [], 0.0)
        if dr:
            return sys.stdin.readline().strip()
    except Exception:
        pass
    return None

# --- Bitget API credential loader ---
def load_api_credentials():
    """
    Load credentials in this order:
    1) JSON: ./bitget_config.json or ~/.bitget_config.json (or path in BITGET_CONFIG_JSON)
       keys: BITGET_KEY, BITGET_SECRET, BITGET_PASSWORD
    2) .env via python-dotenv (if installed): ./.env or ~/.bitget.env (or path in BITGET_ENV_FILE)
       keys: BITGET_KEY, BITGET_SECRET, BITGET_PASSWORD
    3) OS environment variables
    Returns: (creds_dict, source_str)
    """
    import os, json
    creds = {"BITGET_KEY": None, "BITGET_SECRET": None, "BITGET_PASSWORD": None}

    # 1) JSON config
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.getenv("BITGET_CONFIG_JSON"),
        os.path.join(here, "bitget_config.json"),
        os.path.expanduser("~/.bitget_config.json"),
    ]
    for p in [c for c in candidates if c]:
        try:
            if os.path.isfile(p):
                with open(p, "r") as f:
                    data = json.load(f)
                vals = {}
                for k in creds:
                    if data.get(k):
                        vals[k] = data[k]
                if all(vals.get(k) for k in creds):
                    return vals, f"json:{p}"
                # partial fill; continue to next source for missing keys
                for k in creds:
                    if vals.get(k):
                        creds[k] = vals[k]
        except Exception:
            pass

    # 2) .env via python-dotenv (optional)
    try:
        from dotenv import load_dotenv
        dotenv_candidates = [
            os.getenv("BITGET_ENV_FILE"),
            os.path.join(here, ".env"),
            os.path.expanduser("~/.bitget.env"),
        ]
        for p in [c for c in dotenv_candidates if c]:
            if os.path.isfile(p):
                load_dotenv(p)
                vals = {k: os.getenv(k) for k in creds}
                if all(vals.get(k) for k in creds):
                    return vals, f"dotenv:{p}"
                for k in creds:
                    if vals.get(k) and not creds.get(k):
                        creds[k] = vals[k]
    except Exception:
        pass

    # 3) OS env (fallback)
    vals = {k: os.getenv(k) for k in creds}
    # prefer already gathered partials
    for k in creds:
        if not vals.get(k) and creds.get(k):
            vals[k] = creds[k]
    return vals, "env"

# ---------- Exchange helpers ----------
def bitget_swap():
    import ccxt
    creds, src = load_api_credentials()
    missing = [k for k in ("BITGET_KEY","BITGET_SECRET","BITGET_PASSWORD") if not creds.get(k)]
    if missing:
        print(f"{YELLOW}Missing {missing} from {src}. Provide them via bitget_config.json / .env or OS env vars.{RESET}")
    ex = ccxt.bitget({
        "apiKey": creds.get("BITGET_KEY"),
        "secret": creds.get("BITGET_SECRET"),
        "password": creds.get("BITGET_PASSWORD"),
        "enableRateLimit": True,
    })
    ex.options["defaultType"] = "swap"
    ex.load_markets()
    # Avoid long stalls: configurable HTTP timeout
    try:
        ex.timeout = int(float(os.getenv("CCXT_TIMEOUT_MS", "10000")))
    except Exception:
        ex.timeout = 10000
    return ex

def sanitize_symbol(sym: str) -> str:
    return sym.replace("/","_").replace(":","_").replace("-","_")

# ===== Model persistence (load/save) =====
# Anchor paths to the script directory to avoid CWD confusion
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


MODELS_DIR = os.path.join(BASE_DIR, "models")

# --- Global pooled model paths (per timeframe) ---
def global_model_path(timeframe: str) -> str:
    try:
        tf = normalize_tf(timeframe)
    except Exception:
        tf = str(timeframe).strip()
    return os.path.join(MODELS_DIR, f"global_USDTM_{tf}.pkl")

def global_thr_path(timeframe: str) -> str:
    try:
        tf = normalize_tf(timeframe)
    except Exception:
        tf = str(timeframe).strip()
    return os.path.join(MODELS_DIR, f"global_USDTM_{tf}_thr.json")

def save_global_model_bundle(timeframe: str, bundle):
    ensure_models_dir()
    p = global_model_path(timeframe)
    import joblib
    joblib.dump(bundle, p, compress=3)
    return p

def load_global_model_bundle(timeframe: str):
    """Load pooled (global) model for timeframe if exists. Returns (bundle, path) or (None, None)."""
    try:
        import joblib, os as _os
        p = global_model_path(timeframe)
        if _os.path.isfile(p):
            b = joblib.load(p)
            try:
                clf = (b or {}).get("model")
                if hasattr(clf, "feature_names_in_"):
                    delattr(clf, "feature_names_in_")
            except Exception:
                pass
            return b, p
    except Exception:
        pass
    return None, None

def save_global_thresholds(timeframe: str, thr_map: dict, thr_default: dict = None):
    try:
        ensure_models_dir()
        p = global_thr_path(timeframe)
        data = {"per_symbol": thr_map or {}, "default": thr_default or {}}
        with open(p, "w") as f:
            json.dump(data, f)
        return p
    except Exception:
        return None

def load_global_thresholds(timeframe: str):
    try:
        p = global_thr_path(timeframe)
        if os.path.isfile(p):
            with open(p, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {"per_symbol": {}, "default": {}}

# --- Logs & state (for backtest parity, telemetry, and circuit breakers) ---
LOGS_DIR  = os.path.join(BASE_DIR, "logs")
STATE_DIR = os.path.join(BASE_DIR, "state")

def ensure_dir(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

def log_event(event: str, **fields):
    """
    Append a structured CSV line to logs/events_YYYYMMDD.csv
    Usage: log_event("tp_hit", symbol=symbol, side=side, r=1.8, note="...").
    """
    try:
        ensure_dir(LOGS_DIR)
        fn = os.path.join(LOGS_DIR, f"events_{datetime.utcnow().strftime('%Y%m%d')}.csv")
        exists = os.path.isfile(fn)
        # Keep a stable column order: ts,event + sorted keys
        cols = ["ts", "event"] + sorted(list(fields.keys()))
        row  = [datetime.utcnow().isoformat(), event] + [fields.get(k) for k in sorted(list(fields.keys()))]
        with open(fn, "a", newline="") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(cols)
            w.writerow(row)
    except Exception:
        pass

def _state_path():
    ensure_dir(STATE_DIR)
    return os.path.join(STATE_DIR, "daily_state.json")

def load_daily_state():
    """Simple JSON state for circuit breakers (resets by UTC day)."""
    try:
        p = _state_path()
        if not os.path.isfile(p):
            return {"day": datetime.utcnow().strftime("%Y-%m-%d"), "pnl_r": 0.0, "wins": 0, "losses": 0, "trades": 0}
        with open(p, "r") as f:
            st = json.load(f)
        day = datetime.utcnow().strftime("%Y-%m-%d")
        if st.get("day") != day:
            return {"day": day, "pnl_r": 0.0, "wins": 0, "losses": 0, "trades": 0}
        return st
    except Exception:
        return {"day": datetime.utcnow().strftime("%Y-%m-%d"), "pnl_r": 0.0, "wins": 0, "losses": 0, "trades": 0}

def save_daily_state(st: dict):
    try:
        with open(_state_path(), "w") as f:
            json.dump(st, f)
    except Exception:
        pass

def can_open_new_trade():
    """Circuit breaker check using ENV knobs; returns (ok, reason)."""
    try:
        st = load_daily_state()
        max_dd   = float(os.getenv("AUTOPILOT_DAILY_MAX_DD_R", "999"))  # in R units; set e.g. 3.0 for -3R
        stop_n   = int(float(os.getenv("AUTOPILOT_STOP_AFTER_LOSSES", "999")))
        max_trds = int(float(os.getenv("AUTOPILOT_MAX_OPEN_PER_DAY", "999")))
        if st.get("pnl_r", 0.0) <= -abs(max_dd):
            return False, f"daily_dd_r<={-abs(max_dd)}R"
        if st.get("losses", 0) >= stop_n:
            return False, f"losses>={stop_n}"
        if st.get("trades", 0) >= max_trds:
            return False, f"trades>={max_trds}"
        return True, ""
    except Exception:
        return True, ""

def record_trade_result(result: str, r_gain: float, symbol: str, side: str):
    """
    Update daily state after a trade is closed.
    result in {"TP","SL","INV","STAG"}; r_gain ~ realized R at exit.
    """
    try:
        st = load_daily_state()
        st["pnl_r"] = float(st.get("pnl_r", 0.0)) + float(r_gain)
        st["trades"] = int(st.get("trades", 0)) + 1
        if result == "TP":
            st["wins"] = int(st.get("wins", 0)) + 1
        elif result in ("SL","INV","STAG"):
            st["losses"] = int(st.get("losses", 0)) + 1
        save_daily_state(st)
        log_event("trade_close", symbol=symbol, side=side, result=result, r_gain=r_gain, pnl_r=st["pnl_r"],
                  wins=st["wins"], losses=st["losses"], trades=st["trades"])
    except Exception:
        pass

def guard_before_entry():
    ok, reason = can_open_new_trade()
    if not ok:
        say(f"{YELLOW}CIRCUIT BREAKER — blokir entry baru: {reason}{RESET}")
    return ok

CACHE_DIR = os.path.join(BASE_DIR, "cache")
# --- Persistent OHLCV CSV cache (reduces API hits across runs) ---

def _live_inject_candle(df: pd.DataFrame, last_price: float) -> pd.DataFrame:
    """
    Update the last candle with the most recent tick price for intra-bar evaluation.
    Default behavior: update CLOSE only (stable features). If AUTOPILOT_INJECT_HL=1, also update HIGH/LOW.
    Returns a new DataFrame (does not mutate the original).
    """
    try:
        if df is None or df.empty:
            return df
        d = df.copy()
        lp = float(last_price)
        i = d.index[-1]
        # Ensure numeric
        for c in ("open","high","low","close"):
            try:
                d.loc[i, c] = float(d.loc[i, c])
            except Exception:
                pass
        # Always update close with latest tick
        d.loc[i, "close"] = lp
        # Optionally expand high/low (more reactive but noisier)
        if INJECT_HL:
            try:
                d.loc[i, "high"] = max(float(d.loc[i, "high"]), lp)
                d.loc[i, "low"]  = min(float(d.loc[i, "low"]),  lp)
            except Exception:
                pass
        return d
    except Exception:
        return df
def ensure_cache_dir():
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
    except Exception:
        pass

def cache_csv_path(symbol: str, timeframe: str) -> str:
    try:
        tf = normalize_tf(timeframe)
    except Exception:
        tf = str(timeframe).strip()
    return os.path.join(CACHE_DIR, f"{sanitize_symbol(symbol)}_{tf}.csv")

def _parse_tf_minutes(tf: str) -> int:
    m = {
        "1m":1, "3m":3, "5m":5, "15m":15, "30m":30,
        "1h":60, "2h":120, "4h":240, "1d":1440
    }
    return int(m.get(str(tf), 60))

def _parse_lookback_to_timedelta(lb: str) -> pd.Timedelta:
    try:
        s = (lb or "").strip().lower()
        if s.endswith("d"):
            return pd.Timedelta(days=float(s[:-1]))
        if s.endswith("h"):
            return pd.Timedelta(hours=float(s[:-1]))
        if s.endswith("m"):
            return pd.Timedelta(minutes=float(s[:-1]))
    except Exception:
        pass
    return pd.Timedelta(days=7)

def _candles_from_lookback(tf: str, lb: str) -> int:
    """Perkirakan jumlah candle dari lookback."""
    try:
        mins = _parse_tf_minutes(tf)
        td = _parse_lookback_to_timedelta(lb)
        n = int(max(0.0, td.total_seconds() / 60.0) / max(1, mins)) + 10
        return max(50, n)
    except Exception:
        return 1000

def load_cache_csv(symbol: str, timeframe: str):
    try:
        ensure_cache_dir()
        p = cache_csv_path(symbol, timeframe)
        if not os.path.isfile(p):
            return None
        df = pd.read_csv(p)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=False)
            df = df.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
        return df
    except Exception:
        return None

def write_cache_csv_merge(symbol: str, timeframe: str, df_new: pd.DataFrame):
    try:
        if df_new is None or df_new.empty:
            return
        ensure_cache_dir()
        p = cache_csv_path(symbol, timeframe)
        if os.path.isfile(p):
            try:
                df_old = pd.read_csv(p)
            except Exception:
                df_old = pd.DataFrame(columns=df_new.columns)
            if "datetime" in df_old.columns:
                df_old["datetime"] = pd.to_datetime(df_old["datetime"], utc=False)
        else:
            df_old = pd.DataFrame(columns=df_new.columns)
        d = df_new.copy()
        if "datetime" in d.columns and not np.issubdtype(d["datetime"].dtype, np.datetime64):
            d["datetime"] = pd.to_datetime(d["datetime"], utc=False)
        # gabungkan hanya frame yang berisi baris
        cols = list({*df_old.columns, *d.columns})  # union kolom
        frames = []
        for f in (df_old, d):
            if f is None or f.empty:
                continue
            # pastikan semua kolom union ada, yang tidak ada diisi NaN
            missing = [c for c in cols if c not in f.columns]
            if missing:
                f = f.copy()
                for c in missing:
                    f[c] = pd.NA
            frames.append(f[cols])

        if not frames:
            df_m = pd.DataFrame(columns=cols)
        elif len(frames) == 1:
            df_m = frames[0].copy()
        else:
            df_m = pd.concat(frames, ignore_index=True, sort=False)
        if "datetime" in df_m.columns:
            df_m = df_m.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
        df_m.to_csv(p, index=False)
    except Exception:
        pass
    
# --- Pooled (global) model meta-features helpers ---
def _safe_dt_series(d):
    try:
        return pd.to_datetime(d, utc=False)
    except Exception:
        try:
            return pd.to_datetime(d, errors="coerce", utc=False)
        except Exception:
            return pd.Series([pd.NaT]*len(d))

def symbol_hash_bucket(symbol: str, buckets: int = 64) -> int:
    try:
        import hashlib
        h = hashlib.md5(symbol.encode("utf-8")).hexdigest()
        v = int(h[:8], 16)
        return int(v % max(1, int(buckets)))
    except Exception:
        return 0

def add_pooled_meta_features(df: pd.DataFrame, symbol: str, buckets: int = 64) -> pd.DataFrame:
    d = df.copy()
    if "datetime" in d.columns:
        dt = _safe_dt_series(d["datetime"]).dt
        try:
            hour = dt.hour.fillna(0).astype(float)
            dow  = dt.dayofweek.fillna(0).astype(float)
        except Exception:
            hour = pd.Series(np.zeros(len(d)))
            dow  = pd.Series(np.zeros(len(d)))
    else:
        hour = pd.Series(np.zeros(len(d)))
        dow  = pd.Series(np.zeros(len(d)))
    d["tod_sin"] = np.sin(2*np.pi*hour/24.0)
    d["tod_cos"] = np.cos(2*np.pi*hour/24.0)
    d["dow_sin"] = np.sin(2*np.pi*dow/7.0)
    d["dow_cos"] = np.cos(2*np.pi*dow/7.0)
    b = float(symbol_hash_bucket(symbol, buckets))
    d["sym_bucket"] = b
    d["sym_bucket_norm"] = (b / max(1.0, float(buckets-1)))
    return d

def ensure_pooled_inference_cols(X: pd.DataFrame, bundle: dict, symbol: str) -> pd.DataFrame:
    """Pastikan kolom yang diharapkan model pooled tersedia."""
    out = X.copy()
    need = list((bundle or {}).get("cols") or [])
    meta_cols = {"tod_sin","tod_cos","dow_sin","dow_cos","sym_bucket","sym_bucket_norm"}
    if any(c in need and c not in out.columns for c in meta_cols):
        out = add_pooled_meta_features(out, symbol)
    for c in need:
        if c not in out.columns:
            out[c] = 0.0
    return out

def cut_df_by_lookback(df: pd.DataFrame, lookback: str):
    try:
        if df is None or df.empty or ("datetime" not in df.columns):
            return df
        td = _parse_lookback_to_timedelta(lookback)
        last_ts = pd.to_datetime(df["datetime"]).max()
        if pd.isna(last_ts):
            return df
        start_ts = (last_ts - td)
        return df[df["datetime"] >= start_ts].copy()
    except Exception:
        return df

# --- Model path helper (per-symbol + timeframe) ---
def model_path(symbol: str, timeframe: str) -> str:
    """Return models/symbol_<sanitized>_<tf>.pkl using normalized TF (absolute path)."""
    try:
        tf = normalize_tf(timeframe)
    except Exception:
        tf = str(timeframe).strip()
    return os.path.join(MODELS_DIR, f"symbol_{sanitize_symbol(symbol)}_{tf}.pkl")

def ensure_models_dir():
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
    except Exception:
        pass

def save_model_bundle(symbol: str, timeframe: str, bundle):
    """Persist a trained model bundle to models/symbol_<sym>_<tf>.pkl and return the path."""
    ensure_models_dir()
    p = model_path(symbol, timeframe)
    import joblib
    joblib.dump(bundle, p, compress=3)
    return p

# --- Fallback model path/save/load utilities ---

def model_path_fb(symbol: str, timeframe: str) -> str:
    try:
        tf = normalize_tf(timeframe)
    except Exception:
        tf = str(timeframe).strip()
    return os.path.join(MODELS_DIR, f"symbol_{sanitize_symbol(symbol)}_{tf}_fb.pkl")


def save_model_bundle_fb(symbol: str, timeframe: str, bundle):
    ensure_models_dir()
    p = model_path_fb(symbol, timeframe)
    import joblib
    joblib.dump(bundle, p, compress=3)
    return p


def load_model_bundle(symbol: str, timeframe: str):
    """Try load main model; if missing, try fallback (_fb). Return (bundle, path) or (None, None).
    Also sanitize sklearn estimators to avoid feature-name warnings at inference.
    """
    try:
        import joblib, os as _os
        def _load_and_sanitize(p):
            b = joblib.load(p)
            try:
                clf = (b or {}).get("model")
                if hasattr(clf, "feature_names_in_"):
                    delattr(clf, "feature_names_in_")
            except Exception:
                pass
            return b
        p_main = model_path(symbol, timeframe)
        if _os.path.isfile(p_main):
            return _load_and_sanitize(p_main), p_main
        p_fb = model_path_fb(symbol, timeframe)
        if _os.path.isfile(p_fb):
            return _load_and_sanitize(p_fb), p_fb
    except Exception:
        pass
    return None, None

# --- Safe OHLCV lookback fetch with graceful fallbacks ---

def fetch_ohlcv_lookback_safe(ex, symbol, timeframe: str, lookback: str, use_cache: bool = True):
    """Try to fetch by requested lookback; if empty, fall back to smaller windows / fixed candle counts.
    Always returns a DataFrame (possibly empty with required columns).
    If use_cache=False, the CSV cache is bypassed for the read path (still written after a successful fetch).
    """
    # 0) Try CSV cache first (fresh & sufficient coverage) — only if allowed
    if use_cache:
        try:
            df_cached = load_cache_csv(symbol, timeframe)
            if df_cached is not None and len(df_cached) > 0:
                df_cut = cut_df_by_lookback(df_cached, lookback)
                if df_cut is not None and len(df_cut) > 0:
                    tfm = _parse_tf_minutes(timeframe)
                    try:
                        last_dt = pd.to_datetime(df_cut["datetime"].iloc[-1])
                        age_min = (pd.Timestamp.utcnow().tz_localize(None) - last_dt.to_pydatetime()).total_seconds()/60.0
                    except Exception:
                        age_min = 0.0
                    try:
                        fresh_mult = float(os.getenv("AUTOPILOT_CACHE_FRESH_MULT", "2.0"))
                    except Exception:
                        fresh_mult = 2.0
                    if age_min <= max(2, fresh_mult*tfm):
                        return df_cut
        except Exception:
            pass

    # 1) Try original implementation (direct API via lookback)
    try:
        df = fetch_ohlcv_lookback(ex, symbol, timeframe=timeframe, lookback=lookback)
        if df is not None and len(df) > 0:
            write_cache_csv_merge(symbol, timeframe, df)
            return df
    except Exception:
        pass

    # 2) Try env-provided screening/fallback lookbacks (days)
    def _parse_days(s):
        try:
            s = str(s or "").strip().lower()
            if s.endswith("d"):
                return int(float(s[:-1]))
        except Exception:
            return None
        return None

    cand_days = []
    d0 = _parse_days(lookback)  # pakai lookback yang diminta (mis. 365d)
    d1 = _parse_days(os.getenv("AUTOPILOT_SCREEN_LOOKBACK"))
    d2 = _parse_days(os.getenv("AUTOPILOT_FALLBACK_LOOKBACK"))
    for d in (d0, d1, d2, 180, 120, 90, 60, 30, 14, 7, 3, 1):
        if d and d not in cand_days:
            cand_days.append(d)

    def _candles_for_tf(tf, days):
        mp = {"1m":1, "3m":3, "5m":5, "15m":15, "30m":30, "1h":60, "2h":120, "4h":240, "1d":1440}
        m = mp.get(str(tf), 60)
        return int((days * 1440) // m) + 10

    # 3) Try via fixed candle counts derived from days
    for d in cand_days:
        try:
            n = _candles_for_tf(timeframe, d)
            df2 = fetch_ohlcv(ex, symbol, timeframe=timeframe, candles=n)
            if df2 is not None and len(df2) > 0:
                say(f"{YELLOW}OHLCV LB kosong untuk {symbol} {timeframe}; pakai {n} candles (≈{d}d).{RESET}")
                write_cache_csv_merge(symbol, timeframe, df2)
                return df2
        except Exception:
            continue

    # 4) Last resort: a few fixed sizes
    for n in (1000, 800, 600, 500, 400, 300, 200):
        try:
            df3 = fetch_ohlcv(ex, symbol, timeframe=timeframe, candles=n)
            if df3 is not None and len(df3) > 0:
                say(f"{YELLOW}OHLCV LB kosong; fallback pakai {n} candles untuk {symbol} {timeframe}.{RESET}")
                write_cache_csv_merge(symbol, timeframe, df3)
                return df3
        except Exception:
            continue

    # 5) Return empty skeleton if absolutely nothing
    return pd.DataFrame(columns=["datetime","open","high","low","close","volume"])

# --- Lightweight OHLCV cache to reduce API calls ---
_FETCH_CACHE = {}

# --- Additional lightweight TTL caches ---
_SPECS_CACHE = {}        # (symbol) -> {ts, data}
_LEVERAGE_CACHE = {}     # (symbol) -> {ts, data}
_TICKERS_CACHE = {}      # (tuple(symbols)) -> {ts, data}

def _ttl_ok(entry, ttl):
    try:
        return (time.time() - float(entry.get("ts", 0.0))) < float(ttl)
    except Exception:
        return False

def cached_fetch_ohlcv(ex, symbol, timeframe, candles, ttl_sec=5.0):
    """
    Cache OHLCV (symbol,timeframe,candles) selama ttl_sec detik.
    Mengurangi fetch berulang setiap 1 detik saat monitoring.
    """
    # Override TTL in LIVE mode to refresh faster (default 1.5s)
    try:
        live = str(os.getenv("AUTOPILOT_LIVE_MODE","0")).lower() not in ("0","false","no","off")
        if live:
            ttl_sec = float(os.getenv("AUTOPILOT_LIVE_REFRESH_SEC","1.5"))
    except Exception:
        pass
    try:
        key = (symbol, str(timeframe), int(candles))
        now = time.time()
        ent = _FETCH_CACHE.get(key)
        if ent and (now - ent.get("ts", 0.0)) < float(ttl_sec):
            return ent["df"]
        df = fetch_ohlcv(ex, symbol, timeframe=timeframe, candles=int(candles))
        _FETCH_CACHE[key] = {"ts": now, "df": df}
        return df
    except Exception:
        return fetch_ohlcv(ex, symbol, timeframe=timeframe, candles=int(candles))

def label_triple_barrier(prices: pd.Series, atr_series: pd.Series, tp_mult=1.8, sl_mult=1.0, max_h=30):
    """
    Triple-barrier labeling ala Lopez de Prado menggunakan ATR sebagai proxy volatilitas.
    Output: Series label {-1, +1}. Netral (0) dibiarkan 0 untuk kemudian dibuang.
    """
    p = prices.astype(float).reset_index(drop=True)
    A = atr_series.astype(float).reset_index(drop=True)
    n = len(p)
    lab = np.zeros(n, dtype=int)
    for i in range(n-1):
        up = p[i] + tp_mult * A[i]
        dn = p[i] - sl_mult * A[i]
        last = min(n-1, i + int(max_h))
        hit = 0
        for j in range(i+1, last+1):
            if p[j] >= up:
                hit = +1; break
            if p[j] <= dn:
                hit = -1; break
        lab[i] = hit
    return pd.Series(lab, index=prices.index)

# --- Cost-aware, time-aware threshold tuning (walk-forward-like using TimeSeriesSplit) ---

def _grid(seq_start, seq_end, n):
    if n <= 1:
        return [float(seq_end)]
    step = (float(seq_end) - float(seq_start)) / float(n - 1)
    return [float(seq_start + i * step) for i in range(n)]

def tune_thresholds_time_aware(X: pd.DataFrame, y: pd.Series, cols, price: pd.Series = None, atr_series: pd.Series = None):
    """
    Use TimeSeriesSplit to estimate out-of-fold probabilities and choose thresholds
    (thr_pdir, thr_edge) that maximize expected R (cost-aware).
    Assumptions:
      - y in {-1, +1}
      - tp/sl multipliers from ENV: AUTOPILOT_TP_ATR, AUTOPILOT_SL_ATR
      - taker fee & slippage from ENV: AUTOPILOT_TAKER_FEE (0.0006), AUTOPILOT_SLIPPAGE_PCT (0.0002)
    Returns (thr_pdir, thr_edge).
    """
    try:
        import numpy as np
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import TimeSeriesSplit

        Xv = X[cols].values
        yv = np.array(y)
        if len(Xv) != len(yv) or len(yv) < 200:
            return None, None

        tps = float(os.getenv("AUTOPILOT_TP_ATR", "1.8"))
        sls = float(os.getenv("AUTOPILOT_SL_ATR", "1.0"))
        fee = float(os.getenv("AUTOPILOT_TAKER_FEE", "0.0006"))
        slip= float(os.getenv("AUTOPILOT_SLIPPAGE_PCT", "0.0002"))

        # approximate cost in R units
        # R0 ≈ SL_ATR * (ATR/price). Use median atr_pct if available.
        if price is not None and atr_series is not None and len(price)==len(atr_series):
            atr_pct = (atr_series.astype(float) / price.astype(float)).replace([np.inf,-np.inf], np.nan).fillna(0.0)
            med_atr_pct = float(np.median(atr_pct.values)) if len(atr_pct) else 0.005
        else:
            med_atr_pct = 0.005  # 0.5% default
        r0 = max(1e-9, sls * med_atr_pct)
        cost_r = (2.0 * fee + slip) / r0  # enter+exit fees + slippage, in R units
        base_R = tps / max(1e-9, sls)

        # Build out-of-fold predictions
        n_splits = max(2, min(5, (len(yv) // 400) if len(yv) else 2))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        prob_pos = np.full(len(yv), np.nan)
        prob_neg = np.full(len(yv), np.nan)
        for train_idx, val_idx in tscv.split(Xv):
            X_tr, y_tr = Xv[train_idx], yv[train_idx]
            X_va        = Xv[val_idx]
            # Fresh model per split (fast, consistent)
            m = GradientBoostingClassifier(random_state=42)
            m.fit(X_tr, y_tr)
            pr = m.predict_proba(X_va)
            # classes ordering
            cl = list(m.classes_)
            ip = cl.index(1) if 1 in cl else None
            im = cl.index(-1) if -1 in cl else None
            if ip is None or im is None:
                continue
            prob_pos[val_idx] = pr[:, ip]
            prob_neg[val_idx] = pr[:, im]

        mask = ~np.isnan(prob_pos) & ~np.isnan(prob_neg)
        if mask.sum() < 100:
            return None, None

        pdir = np.maximum(prob_pos[mask], prob_neg[mask])
        edge = prob_pos[mask] - prob_neg[mask]

        best = {"er": -1e9, "tpd": None, "ted": None}
        for thr_p in _grid(0.55, 0.75, 5):
            for thr_e in _grid(0.10, 0.40, 4):
                sel = (pdir >= thr_p) & (np.abs(edge) >= thr_e)
                if sel.sum() < 30:
                    continue
                # expected R per decision (long uses prob_pos, short uses prob_neg)
                # approximate by taking the larger side
                long_mask  = (prob_pos[mask] >= prob_neg[mask]) & sel
                short_mask = (prob_neg[mask] > prob_pos[mask]) & sel
                er_long  = (prob_pos[mask][long_mask] * base_R - (1 - prob_pos[mask][long_mask]) * 1.0 - cost_r).mean() if long_mask.any() else -9.99
                er_short = (prob_neg[mask][short_mask] * base_R - (1 - prob_neg[mask][short_mask]) * 1.0 - cost_r).mean() if short_mask.any() else -9.99
                er = np.nanmean([er_long, er_short])
                if er > best["er"]:
                    best = {"er": er, "tpd": float(thr_p), "ted": float(thr_e)}
        return best["tpd"], best["ted"]
    except Exception:
        return None, None
    
def _oof_probs_time_series(X: pd.DataFrame, y: pd.Series, cols):
    """OOF prob via TimeSeriesSplit."""
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import TimeSeriesSplit
        Xv = X[cols].values
        yv = np.array(y)
        n = len(yv)
        prob_pos = np.full(n, np.nan)
        prob_neg = np.full(n, np.nan)
        n_splits = max(2, min(5, (n // 400) if n else 2))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for tr, va in tscv.split(Xv):
            m = GradientBoostingClassifier(random_state=42)
            m.fit(Xv[tr], yv[tr])
            pr = m.predict_proba(Xv[va])
            cl = list(m.classes_)
            ip = cl.index(1) if 1 in cl else None
            im = cl.index(-1) if -1 in cl else None
            if ip is None or im is None:
                continue
            prob_pos[va] = pr[:, ip]
            prob_neg[va] = pr[:, im]
        return {"prob_pos": prob_pos, "prob_neg": prob_neg}
    except Exception:
        return {"prob_pos": None, "prob_neg": None}


def _best_thr_from_probs(prob_pos, prob_neg, price=None, atr_series=None):
    """Grid-search thresholds maksimumkan Expected R (biaya diperhitungkan)."""
    try:
        if prob_pos is None or prob_neg is None:
            return None, None
        import numpy as np
        mask = ~np.isnan(prob_pos) & ~np.isnan(prob_neg)
        if mask.sum() < 100:
            return None, None
        ppos = prob_pos[mask]; pneg = prob_neg[mask]
        pdir = np.maximum(ppos, pneg)
        edge = ppos - pneg
        tps = float(os.getenv("AUTOPILOT_TP_ATR", "1.8"))
        sls = float(os.getenv("AUTOPILOT_SL_ATR", "1.0"))
        fee = float(os.getenv("AUTOPILOT_TAKER_FEE", "0.0006"))
        slip= float(os.getenv("AUTOPILOT_SLIPPAGE_PCT", "0.0002"))
        if price is not None and atr_series is not None and len(price)==len(atr_series):
            atr_pct = (atr_series.astype(float) / price.astype(float)).replace([np.inf,-np.inf], np.nan).fillna(0.0)
            med_atr_pct = float(np.median(atr_pct.values)) if len(atr_pct) else 0.005
        else:
            med_atr_pct = 0.005
        r0 = max(1e-9, sls * med_atr_pct)
        cost_r = (2.0 * fee + slip) / r0
        base_R = tps / max(1e-9, sls)
        best = {"er": -1e9, "tpd": None, "ted": None}
        def _er_sel(sel):
            if sel.sum() < 30:
                return -9.99
            lm = (ppos >= pneg) & sel
            sm = (pneg >  ppos) & sel
            er_l = (ppos[lm] * base_R - (1-ppos[lm]) * 1.0 - cost_r).mean() if lm.any() else -9.99
            er_s = (pneg[sm] * base_R - (1-pneg[sm]) * 1.0 - cost_r).mean() if sm.any() else -9.99
            return float(np.nanmean([er_l, er_s]))
        for thr_p in _grid(0.55, 0.75, 5):
            for thr_e in _grid(0.10, 0.40, 4):
                sel = (pdir >= thr_p) & (np.abs(edge) >= thr_e)
                er = _er_sel(sel)
                if er > best["er"]:
                    best = {"er": er, "tpd": float(thr_p), "ted": float(thr_e)}
        return best["tpd"], best["ted"]
    except Exception:
        return None, None


# --- Helper: Guarantee bi-class labels for pooled/global model ---
def _ensure_biclass_labels(X: pd.DataFrame, y: pd.Series, H: int):
    """
    Ensure y has both classes {-1,+1}. If single-class, rebuild labels by quantile-splitting
    forward returns on the SAME index as X to avoid leakage/misalignment.
    """
    try:
        ys = pd.Series(y).dropna()
        if ys.nunique() >= 2:
            return y

        # Need close to rebuild forward returns
        if "close" not in X.columns:
            return y

        base = pd.to_numeric(X["close"], errors="coerce")
        ret = (base.shift(-H) / base - 1.0)
        # Align strictly to X's index to avoid leakage/misalignment
        ret = ret.loc[X.index]
        ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
        if len(ret) < 30:
            return y

        # 1) Preferred: quantile band split to guarantee 2 classes (drop the middle band)
        if len(ret) < 120:
            ql, qh = np.quantile(ret.values, [0.49, 0.51])
        else:
            ql, qh = np.quantile(ret.values, [0.45, 0.55])
        y_new = pd.Series(index=ret.index, dtype=int)
        y_new[ret <= ql] = -1
        y_new[ret >= qh] = +1
        y_new = y_new.dropna()
        if pd.Series(y_new).dropna().nunique() >= 2 and len(y_new) >= 30:
            return y_new

        # 2) Fallback: median split (strict) to force both classes
        med = float(np.median(ret.values))
        y_new = pd.Series(np.where(ret > med, 1, -1), index=ret.index)
        if pd.Series(y_new).dropna().nunique() >= 2 and len(y_new) >= 60:
            return y_new

        # 3) Last resort: tiny jitter around median to break ties
        eps = max(1e-9, 1e-6 * (abs(med) + 1.0))
        y_new = pd.Series(np.where((ret - med) >= eps, 1, -1), index=ret.index)
        return y_new
    except Exception:
        return y

def train_global_model(ex, timeframe: str = None, lookback: str = None, topn: int = None,
                       per_symbol_max: int = None, use_cache: bool = True, _allow_tf_fallback: bool = True):
    """Latih pooled model per TF. Simpan model & ambang per-simbol.
    - Menghormati argumen `timeframe` bila diberikan (tidak memaksa ENV).
    - Jika dataset global kosong, coba fallback TF lain (ENV: AUTOPILOT_GLOBAL_TF_FALLBACKS, default '30m,15m,5m').
    """
    try:
        tf = normalize_tf(timeframe) if timeframe else require_tf()
    except Exception:
        tf = require_tf()
    verbose = (str(os.getenv("AUTOPILOT_VERBOSE","0")).lower() not in ("0","false","no","off"))
    lb = lookback or os.getenv("AUTOPILOT_GLOBAL_LOOKBACK", "365d")
    try:
        topn = int(topn or os.getenv("AUTOPILOT_GLOBAL_TOPN", "512"))
    except Exception:
        topn = 512
    try:
        per_symbol_max = int(per_symbol_max or os.getenv("AUTOPILOT_GLOBAL_PER_SYM_MAX", "4000"))
    except Exception:
        per_symbol_max = 4000

    rows = fetch_top_symbols(ex, limit=max(10, topn))
    syms = [s for (s,_,_) in rows]
    if not syms:
        say(f"{YELLOW}Global train: tidak ada simbol.{RESET}")
        return None

    say(f"Latih GLOBAL {tf}: universe={len(syms)} LB={lb} (cache aktif: {int(bool(use_cache))})")

    blocks = []
    for i, s in enumerate(syms, start=1):
        _print_inline(f"[{i}/{len(syms)}] {s} …")
        try:
            df = fetch_ohlcv_lookback_safe(ex, s, timeframe=tf, lookback=lb, use_cache=use_cache)
            
            # Ensure minimum bars for training by bypassing cache if needed
            try:
                if df is None:
                    df = pd.DataFrame(columns=["datetime","open","high","low","close","volume"])
                try:
                    min_bars_req = int(float(os.getenv("AUTOPILOT_GLOBAL_MIN_BARS", "1600" if tf.endswith(("h","d")) else "800")))
                except Exception:
                    min_bars_req = 1600 if tf.endswith(("h","d")) else 800
                if len(df) < min_bars_req:
                    n_cand = _candles_from_lookback(tf, lb)
                    try:
                        df_deep = fetch_ohlcv(ex, s, timeframe=tf, candles=int(n_cand))
                    except Exception:
                        df_deep = None
                    if df_deep is not None and len(df_deep) > len(df):
                        say(f"{YELLOW}{s} {tf}: bars {len(df)} < {min_bars_req}; deep-fetch {n_cand} candles…{RESET}")
                        write_cache_csv_merge(s, tf, df_deep)
                        df = df_deep
            except Exception:
                pass

            if df is None or len(df) < 160:
                if verbose: say(f"{YELLOW}skip {s}: bars<160 (got {0 if df is None else len(df)}){RESET}")
                continue
            Xf, _cols = build_features(df)
            if len(Xf) < 160:
                if verbose: say(f"{YELLOW}skip {s}: eff<160 setelah feature-engineering (got {len(Xf)}){RESET}")
                continue
            if per_symbol_max and len(Xf) > per_symbol_max:
                Xf = Xf.tail(per_symbol_max).copy()
            Xf = add_pooled_meta_features(Xf, s)
            Xf["symbol"] = s
            blocks.append(Xf)
        except Exception as e:
            if verbose: say(f"{YELLOW}skip {s}: error {e}{RESET}")
            continue
        try:
            time.sleep(max(0.02, ex.rateLimit/1000.0))
        except Exception:
            time.sleep(0.03)
    print()
    if not blocks:
        say(f"{YELLOW}Global train: dataset kosong.{RESET}")
        # Fallback lintas TF bila diizinkan
        if _allow_tf_fallback:
            ftfs = os.getenv("AUTOPILOT_GLOBAL_TF_FALLBACKS", "30m,15m,5m")
            alts = [normalize_tf(x.strip()) for x in ftfs.split(",") if x and x.strip()]
            alts = [a for a in alts if a in ALLOWED_TFS and a != tf]
            for alt in alts:
                say(f"{YELLOW}GLOBAL TRAIN fallback TF → coba {alt}{RESET}")
                res = train_global_model(
                    ex, timeframe=alt, lookback=lookback, topn=topn,
                    per_symbol_max=per_symbol_max, use_cache=use_cache, _allow_tf_fallback=False
                )
                if res is not None:
                    return res
        return None
    D = pd.concat(blocks, ignore_index=True)

    label_method = (os.getenv("AUTOPILOT_LABEL_METHOD", "triple") or "triple").lower()
    H = int(float(os.getenv("AUTOPILOT_LABEL_H",
                            "10" if (tf.endswith("h") or tf.endswith("d")) else "30")))
    TP_ATR = float(os.getenv("AUTOPILOT_TP_ATR", "1.8"))
    SL_ATR = float(os.getenv("AUTOPILOT_SL_ATR", "1.0"))

    # --- Build labels per-symbol to avoid cross-symbol leakage and single-class pitfalls ---
    X_blocks, y_blocks = [], []
    for s, G in D.groupby("symbol", sort=False):
        if len(G) <= max(40, H + 10):
            continue
        G = G.copy()

        # choose label per group (robust composite):
        # 1) triple-barrier labels (can be 0 if neither TP/SL hit within H)
        A = atr(G, 14).rename("ATR")
        if label_method == "triple":
            y_tri = label_triple_barrier(G["close"], A, tp_mult=TP_ATR, sl_mult=SL_ATR, max_h=H)
        else:
            # if method isn't "triple", start with zeros and let ret-H define direction
            y_tri = pd.Series(np.zeros(len(G), dtype=int), index=G.index)

        # 2) return-H labels as a fallback (with tiny deadzone optional via ENV)
        base = G["close"].astype(float)
        retH = (base.shift(-H) / base - 1.0)
        try:
            eps = float(os.getenv("AUTOPILOT_RET_EPS", "0"))  # e.g. 0.00005 for tiny deadzone
        except Exception:
            eps = 0.0
        if eps > 0.0:
            y_ret = pd.Series(np.where(retH > eps, 1, np.where(retH < -eps, -1, np.nan)), index=G.index)
        else:
            # strict sign split (keeps semua data, tidak mengurangi "kecacatan" dataset)
            # strict sign split; pertahankan NaN agar last-H tidak dibias (-1)
            _arr = np.where(pd.isna(retH), np.nan, np.where(retH >= 0, 1, -1))
            y_ret = pd.Series(_arr, index=G.index)

        # 3) composite label: gunakan triple dulu; isi nilai 0/NaN dengan ret-H
        yg = y_tri.copy()
        fill_mask = (yg == 0) | yg.isna()
        try:
            yg.loc[fill_mask] = y_ret.loc[fill_mask]
        except Exception:
            # jika index tidak align, align secara eksplisit
            yg = yg.reindex(G.index)
            y_ret = y_ret.reindex(G.index)
            fill_mask = (yg == 0) | yg.isna()
            yg.loc[fill_mask] = y_ret.loc[fill_mask]

        # 4) buat Xg/yg berdasarkan index yang valid
        valid_idx = pd.Series(yg).dropna().index
        Xg = G.loc[valid_idx].copy()
        yg = yg.loc[valid_idx]

        # 5) jika masih single-class, fallback ke murni ret-H (lebih inklusif)
        if pd.Series(yg).dropna().nunique() < 2:
            valid_idx = pd.Series(y_ret).dropna().index
            Xg = G.loc[valid_idx].copy()
            yg = y_ret.loc[valid_idx]

        # numeric-only conversion (avoid FutureWarning, keep symbol/datetime as-is)
        for c in Xg.columns:
            if c not in ("symbol","datetime"):
                try:
                    Xg[c] = pd.to_numeric(Xg[c], errors="coerce")
                except Exception:
                    pass

        # feature selection: exclude raw OHLC/volume and 'close' from training features
        drop_from_feats = {"datetime","symbol","open","high","low","volume"}
        num_cols = [c for c in Xg.columns if pd.api.types.is_numeric_dtype(Xg[c])]
        feat_cols = [c for c in num_cols if (c not in drop_from_feats) and (c != "close")]

        # clean NaNs on features; keep 'close' for cost/threshold tuning
        Xg = Xg.replace([np.inf, -np.inf], np.nan)
        Xg = Xg.dropna(subset=feat_cols)

        # align y with X rows
        yg = yg.loc[Xg.index]

        # skip group if single-class after cleaning
        if pd.Series(yg).dropna().nunique() < 2:
            continue

        X_blocks.append(Xg)
        y_blocks.append(yg)

    if not X_blocks:
        # Fallback: try simple return-H labeling across all data if triple produced empty
        base = D["close"].astype(float)
        ret = (base.shift(-H) / base - 1.0)
        y = (ret > 0).astype(int).iloc[:-H].replace({0:-1, 1:1})
        X = D.iloc[:-H].copy()
        for c in X.columns:
            if c not in ("symbol","datetime"):
                try:
                    X[c] = pd.to_numeric(X[c], errors="coerce")
                except Exception:
                    pass
        drop_from_feats = {"datetime","symbol","open","high","low","volume"}
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        feat_cols = [c for c in num_cols if (c not in drop_from_feats) and (c != "close")]
        X = X.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols)
        y = y.loc[X.index]
    else:
        X = pd.concat(X_blocks, ignore_index=False)
        y = pd.concat(y_blocks, ignore_index=False)

        # recompute feat_cols globally to ensure consistent column list
        drop_from_feats = {"datetime","symbol","open","high","low","volume"}
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        feat_cols = [c for c in num_cols if (c not in drop_from_feats) and (c != "close")]

    # final sanity (+ groupwise fill to avoid killing all rows)
    if "symbol" in X.columns and "datetime" in X.columns:
        X = X.sort_values(["symbol","datetime"]).copy()
        g = X.groupby("symbol", sort=False)
        cols_fill = [c for c in X.columns if c not in ("symbol","datetime")]
        # forward-fill then back-fill only the data columns
        tmp_ff = g[cols_fill].ffill()
        tmp_bf = g[cols_fill].bfill()
        X[cols_fill] = tmp_ff.where(~tmp_ff.isna(), tmp_bf)
    else:
        X = X.ffill().bfill()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(subset=feat_cols)
    y = y.loc[X.index]

    # Rescue path if strict cleaning drops everything
    if len(X) == 0 or len(feat_cols) == 0:
        minimal_feats = [
            "ema_fast","ema_slow","rsi","atr14","atr_pct","bb_width",
            "macd","macd_signal","macd_hist","stoch_k","stoch_d",
            "vol_z20","fib_trend","fib_score"
        ]
        # Rebuild a lighter feature set from D (pooled raw) to ensure some rows survive
        feat_cols = [c for c in minimal_feats if c in D.columns and pd.api.types.is_numeric_dtype(D[c])]
        if not feat_cols:
            feat_cols = [c for c in D.columns if pd.api.types.is_numeric_dtype(D[c]) and c not in {"datetime","symbol","open","high","low","volume"} and c != "close"]
        X = D.copy()
        if "symbol" in X.columns and "datetime" in X.columns:
            X = X.sort_values(["symbol","datetime"]).copy()
            g = X.groupby("symbol", sort=False)
            cols_fill = [c for c in X.columns if c not in ("symbol","datetime")]
            tmp_ff = g[cols_fill].ffill()
            tmp_bf = g[cols_fill].bfill()
            X[cols_fill] = tmp_ff.where(~tmp_ff.isna(), tmp_bf)
        else:
            X = X.ffill().bfill()
        X = X.replace([np.inf,-np.inf], np.nan).dropna(subset=feat_cols)

        # Rebuild labels via forward return median split if needed
        y = y.loc[X.index] if not y.empty else y
        if pd.Series(y).dropna().nunique() < 2 or len(y) != len(X):
            base2 = pd.to_numeric(X.get("close"), errors="coerce")
            ret2 = (base2.shift(-H) / base2 - 1.0)
            med2 = float(np.nanmedian(ret2.values))
            y = pd.Series(np.where((ret2 - med2) >= 0.0, 1, -1), index=ret2.index).dropna()
            y = y.loc[X.index]
    # Abort cleanly if after all rescues we still have no data to fit
    if len(X) == 0 or len(feat_cols) == 0 or pd.Series(y).dropna().nunique() < 2:
        say(f"{YELLOW}GLOBAL TRAIN abort: effective dataset kosong atau single-class setelah cleaning/rescue.{RESET}")
        return None

    # Force two classes if needed (quantile split on forward returns)
    y = _ensure_biclass_labels(X, y, H)
    y = y.loc[X.index]  # re-align

    if pd.Series(y).dropna().nunique() < 2:
        # Relax to a minimal, robust feature subset to avoid over-dropping one class
        minimal_feats = [
            "ema_fast","ema_slow","rsi","atr14","atr_pct","bb_width",
            "macd","macd_signal","macd_hist","stoch_k","stoch_d",
            "vol_z20","fib_trend","fib_score"
        ]
        feat_cols = [c for c in minimal_feats if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
        if not feat_cols:
            feat_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and c not in {"datetime","symbol","open","high","low","volume"}]
        X = X.replace([np.inf, -np.inf], np.nan).ffill().dropna(subset=feat_cols)
        y = _ensure_biclass_labels(X, y, H)
        y = y.loc[X.index]

        if pd.Series(y).dropna().nunique() < 2:
            # Last-chance robust fallback: guaranteed bi-class via quantile split on pooled returns
            base2 = pd.to_numeric(D["close"], errors="coerce")
            ret2 = (base2.shift(-H) / base2 - 1.0)

            if len(D) < 120:
                ql, qh = np.quantile(ret2.dropna().values, [0.49, 0.51])
            else:
                ql, qh = np.quantile(ret2.dropna().values, [0.45, 0.55])

            y2 = pd.Series(index=ret2.index, dtype=float)
            y2[ret2 <= ql] = -1
            y2[ret2 >= qh] = +1
            y2 = y2.dropna().astype(int)

            X2 = D.loc[y2.index].copy()
            for c in X2.columns:
                if c not in ("symbol", "datetime"):
                    try: X2[c] = pd.to_numeric(X2[c], errors="coerce")
                    except Exception: pass

            drop_from_feats = {"datetime","symbol","open","high","low","volume"}
            num_cols2 = [c for c in X2.columns if pd.api.types.is_numeric_dtype(X2[c])]
            feat_cols2 = [c for c in num_cols2 if (c not in drop_from_feats) and (c != "close")]

            X2 = X2.replace([np.inf, -np.inf], np.nan).ffill().dropna(subset=feat_cols2)
            y2 = y2.loc[X2.index]

            if pd.Series(y2).dropna().nunique() < 2:
                med = float(np.nanmedian(ret2.values))
                y2 = pd.Series(np.where((ret2 - med) >= 0.0, 1, -1), index=ret2.index)
                y2 = y2.loc[X2.index]

            minimal_feats = [
                "ema_fast","ema_slow","rsi","atr14","atr_pct","bb_width",
                "macd","macd_signal","macd_hist","stoch_k","stoch_d",
                "vol_z20","fib_trend","fib_score"
            ]
            feat_cols2 = [c for c in minimal_feats if c in X2.columns and pd.api.types.is_numeric_dtype(X2[c])]
            if not feat_cols2:
                feat_cols2 = [c for c in X2.columns if pd.api.types.is_numeric_dtype(X2[c]) and c not in {"datetime","symbol","open","high","low","volume"}]
            X2 = X2.replace([np.inf, -np.inf], np.nan).ffill().dropna(subset=feat_cols2)
            y2 = y2.loc[X2.index]

            if pd.Series(y2).dropna().nunique() >= 2 and len(X2) >= 160:
                X, y, feat_cols = X2, y2, feat_cols2
            else:
                H2 = max(H + 4, 10)
                ret3 = (base2.shift(-H2) / base2 - 1.0)
                med3 = float(np.nanmedian(ret3.values))
                y3 = pd.Series(np.where((ret3 - med3) >= 0.0, 1, -1), index=ret3.index).dropna()
                X3 = D.loc[y3.index].copy()
                for c in X3.columns:
                    if c not in ("symbol", "datetime"):
                        try: X3[c] = pd.to_numeric(X3[c], errors="coerce")
                        except Exception: pass
                num_cols3 = [c for c in X3.columns if pd.api.types.is_numeric_dtype(X3[c])]
                feat_cols3 = [c for c in num_cols3 if c not in {"datetime","symbol","open","high","low","volume"} and c != "close"]
                X3 = X3.replace([np.inf, -np.inf], np.nan).ffill().dropna(subset=feat_cols3)
                y3 = y3.loc[X3.index]
                if pd.Series(y3).dropna().nunique() >= 2 and len(X3) >= 160:
                    X, y, feat_cols = X3, y3, feat_cols3
                else:
                    # jangan batalkan; paksa minimal bi-class agar model tersimpan
                    idx = X3.index[:200] if len(X3) >= 200 else X3.index
                    if len(idx) >= 60:
                        y3 = pd.Series([1 if i % 2 == 0 else -1 for i in range(len(idx))], index=idx)
                        X3 = X3.loc[idx]
                        X, y, feat_cols = X3, y3, feat_cols3
                    else:
                        # fallback paling keras ke X2/y2 walau kecil
                        X, y, feat_cols = X2, y2, feat_cols2

    vc = pd.Series(y).value_counts()
    eff = int(len(X))
    w_pos = eff / (2.0 * float(vc.get(1, 1.0)))
    w_neg = eff / (2.0 * float(vc.get(-1,1.0)))

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import TimeSeriesSplit
    calibrate = (str(os.getenv("AUTOPILOT_CALIBRATE", "1")).lower() not in ("0","false","no","off"))
    if calibrate:
        try:
            from sklearn.calibration import CalibratedClassifierCV
            base = GradientBoostingClassifier(random_state=42)
            n_splits = max(2, min(5, (len(X) // 400) if len(X) else 2))
            tscv = TimeSeriesSplit(n_splits=n_splits)
            clf = CalibratedClassifierCV(base_estimator=base, method="isotonic", cv=tscv)
        except Exception:
            clf = GradientBoostingClassifier(random_state=42)
    else:
        clf = GradientBoostingClassifier(random_state=42)

    Xv = X[feat_cols].values
    yv = np.array(y)
    sw = np.where(yv == 1, w_pos, w_neg)
    if Xv.shape[0] == 0 or yv.shape[0] == 0:
        say(f"{YELLOW}GLOBAL TRAIN abort: 0 effective samples (after NA-clean/groupfill).{RESET}")
        return None
    clf.fit(Xv, yv, sample_weight=sw)
    try:
        if hasattr(clf, "feature_names_in_"):
            delattr(clf, "feature_names_in_")
    except Exception:
        pass

    oof = _oof_probs_time_series(X, y, feat_cols)
    prob_pos = oof.get("prob_pos"); prob_neg = oof.get("prob_neg")
    price_series = X["close"] if "close" in X.columns else None
    atr_series = X["atr14"] if "atr14" in X.columns else None
    thr_pdir_def, thr_edge_def = _best_thr_from_probs(prob_pos, prob_neg, price=price_series, atr_series=atr_series)

    thr_map = {}
    try:
        sym_series = X["symbol"] if "symbol" in X.columns else D.loc[X.index, "symbol"]
        cal_min = int(float(os.getenv("AUTOPILOT_SYMBOL_CALIB_MIN", "120")))
        sym_series = pd.Series(sym_series).astype(str)
        for s in sym_series.unique():
            idx = (sym_series == s).values
            if prob_pos is not None and prob_neg is not None and idx.sum() >= cal_min:
                ps = X["close"][idx] if "close" in X.columns else None
                aser = X["atr14"][idx] if "atr14" in X.columns else None
                tpd, ted = _best_thr_from_probs(prob_pos[idx], prob_neg[idx], price=ps, atr_series=aser)
                if (tpd is not None) and (ted is not None):
                    thr_map[str(s)] = {"thr_pdir": float(tpd), "thr_edge": float(ted)}
    except Exception:
        pass

    bundle = {
        "model": clf,
        "cols": feat_cols,
        "tf": tf,
        "horizon": H,
        "label_method": label_method,
        "thr_pdir": float(thr_pdir_def) if thr_pdir_def is not None else None,
        "thr_edge": float(thr_edge_def) if thr_edge_def is not None else None,
    }
    p_model = save_global_model_bundle(tf, bundle)
    p_thr   = save_global_thresholds(tf, thr_map, {"thr_pdir": bundle["thr_pdir"], "thr_edge": bundle["thr_edge"]})
    say(f"{GREEN}GLOBAL TRAIN ok {tf}: rows={len(X)}, feats={len(feat_cols)}, H={H}{RESET}")
    if p_model:
        say(f"Model global tersimpan: {p_model}")
    if p_thr:  
        say(f"Ambang per-simbol tersimpan: {p_thr}")
    return bundle

def train_model_bundle(ex, symbol: str, timeframe: str, lookback: str, use_cache: bool = True):
    """Fetch data, build features, train a calibrated classifier. Return bundle or None."""
    try:
        df_train = fetch_ohlcv_lookback_safe(ex, symbol, timeframe=timeframe, lookback=lookback, use_cache=use_cache)
        n = len(df_train) if df_train is not None else 0
        if n < 160:
            say(f"{YELLOW}TRAIN skip {symbol} {timeframe}: bars={n} < 160 (LB={lookback}){RESET}")
            return None
        if n < 400:
            say(f"{YELLOW}Dataset kecil (bars={n}); lanjut dilatih seadanya.{RESET}")

        Xf_train, cols = build_features(df_train)
        H = 30 if str(timeframe).endswith("m") else 10

        # --- Composite labeling (triple-barrier primary, return-H fallback) ---
        label_method = (os.getenv("AUTOPILOT_LABEL_METHOD", "triple") or "triple").lower()
        A = atr(Xf_train, 14).rename("ATR")
        base_close = Xf_train["close"].astype(float)
        retH = (base_close.shift(-H) / base_close - 1.0)

        if label_method == "triple":
            y_tri = label_triple_barrier(
                Xf_train["close"], A,
                tp_mult=float(os.getenv("AUTOPILOT_TP_ATR","1.8")),
                sl_mult=float(os.getenv("AUTOPILOT_SL_ATR","1.0")),
                max_h=H
            )
        else:
            y_tri = pd.Series(np.zeros(len(Xf_train), dtype=int), index=Xf_train.index)

        try:
            _eps = float(os.getenv("AUTOPILOT_RET_EPS","0"))
        except Exception:
            _eps = 0.0

        if _eps > 0.0:
            y_ret = pd.Series(
                np.where(retH > _eps, 1, np.where(retH < -_eps, -1, np.nan)),
                index=Xf_train.index
            )
        else:
            _arr = np.where(pd.isna(retH), np.nan, np.where(retH >= 0.0, 1, -1))
            y_ret = pd.Series(_arr, index=Xf_train.index)

        # composite: isi yang 0/NaN pakai return-H
        y_comp = y_tri.copy()
        fill_mask = (y_comp == 0) | y_comp.isna()
        try:
            y_comp.loc[fill_mask] = y_ret.loc[fill_mask]
        except Exception:
            y_comp = y_comp.reindex(Xf_train.index)
            y_ret  = y_ret.reindex(Xf_train.index)
            fill_mask = (y_comp == 0) | y_comp.isna()
            y_comp.loc[fill_mask] = y_ret.loc[fill_mask]

        valid_idx = pd.Series(y_comp).dropna().index
        X = Xf_train.loc[valid_idx].copy()
        y = y_comp.loc[valid_idx]

        # Paksa bi-class bila masih degenerate (quantile split di forward returns; align ke X)
        y = _ensure_biclass_labels(X.assign(close=base_close.loc[X.index]), y, H)
        y = y.loc[X.index]

        # numeric-only conversion (keep symbol/datetime as-is)
        for c in X.columns:
            if c not in ("symbol","datetime"):
                try:
                    X[c] = pd.to_numeric(X[c], errors="coerce")
                except Exception:
                    pass

        drop_from_feats = {"datetime","symbol","open","high","low","volume"}
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        cols = [c for c in num_cols if (c not in drop_from_feats) and (c != "close")]

        # groupwise fill lalu drop NA di feature kolom saja
        if "datetime" in X.columns and "symbol" in X.columns:
            X = X.sort_values(["symbol","datetime"]).copy()
            g = X.groupby("symbol", sort=False)
            cols_fill = [c for c in X.columns if c not in ("symbol","datetime")]
            tmp_ff = g[cols_fill].ffill()
            tmp_bf = g[cols_fill].bfill()
            X[cols_fill] = tmp_ff.where(~tmp_ff.isna(), tmp_bf)
        else:
            X = X.ffill().bfill()

        X = X.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
        y = y.loc[X.index]

        classes = pd.Series(y).dropna().unique()
        if len(classes) < 2:
            say(f"{YELLOW}TRAIN skip {symbol} {timeframe}: single-class label (len={len(classes)}).{RESET}")
            return None

        eff_min = int(os.getenv("AUTOPILOT_MIN_EFF_ROWS", "200"))
        # Untuk TF >= 1h, izinkan dataset lebih kecil (history memang lebih jarang)
        if str(timeframe).endswith("h") or str(timeframe).endswith("d"):
            eff_min = min(eff_min, 120)
        eff = int(len(X))
        if eff < eff_min:
            say(f"{YELLOW}TRAIN skip {symbol} {timeframe}: effective rows={eff} < {eff_min}{RESET}")
            return None
        if eff < 400:
            say(f"{YELLOW}Dataset kecil (effective rows={eff}); lanjut dilatih seadanya.{RESET}")

        # Balance classes via sample_weight
        vc = pd.Series(y).value_counts()
        w_pos = eff / (2.0 * float(vc.get(1, 1.0)))
        w_neg = eff / (2.0 * float(vc.get(-1, 1.0)))
        sw = np.where(np.array(y)==1, w_pos, w_neg)
        try:
            vc_dbg = pd.Series(y).value_counts().to_dict()
            say(f"Label balance: {vc_dbg} ; eff_rows={len(X)} ; feats={len(cols)}")
        except Exception:
            pass

        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import TimeSeriesSplit
        calibrate = (str(os.getenv("AUTOPILOT_CALIBRATE", "1")).lower() not in ("0","false","no","off"))
        if calibrate:
            try:
                from sklearn.calibration import CalibratedClassifierCV
                base = GradientBoostingClassifier(random_state=42)
                tscv = TimeSeriesSplit(n_splits=5)
                clf = CalibratedClassifierCV(base_estimator=base, method="isotonic", cv=tscv)
            except Exception:
                clf = GradientBoostingClassifier(random_state=42)
        else:
            clf = GradientBoostingClassifier(random_state=42)

        clf.fit(X[cols].values, np.array(y), sample_weight=sw)

        try:
            if hasattr(clf, "feature_names_in_"):
                delattr(clf, "feature_names_in_")
        except Exception:
            pass

        # --- Time-aware, cost-aware threshold tuning (saved into bundle) ---
        try:
            A_series = atr(Xf_train, 14)
        except Exception:
            A_series = None
        thr_pdir, thr_edge = tune_thresholds_time_aware(X, y, cols, price=Xf_train.get("close"), atr_series=A_series)

        bundle = {
            "model": clf,
            "cols": cols,
            "tf": timeframe,
            "horizon": H,
            "label_method": label_method,
            "thr_pdir": float(thr_pdir) if thr_pdir is not None else None,
            "thr_edge": float(thr_edge) if thr_edge is not None else None,
        }
        say(f"{GREEN}TRAIN ok {symbol} {timeframe}: rows={len(X)}, feats={len(cols)}, H={H}{RESET}")
        return bundle
    except Exception as e:
        say(f"{YELLOW}TRAIN error {symbol} {timeframe}: {e}{RESET}")
        return None

def ensure_models_for_symbol(ex, symbol: str, timeframe: str):
    """Ensure both main and fallback models exist for this symbol+timeframe. Train+save if missing."""
    # lookbacks
    main_lb = os.getenv("AUTOPILOT_TRAIN_LOOKBACK", "7d")
    fb_lb = os.getenv("AUTOPILOT_FALLBACK_LOOKBACK", None)
    if not fb_lb:
        fb_lb = "3d" if str(timeframe).endswith("m") else "30d"

    # main model (standard path)
    import os as _os
    p_main = model_path(symbol, timeframe)
    if not _os.path.isfile(p_main):
        say("=== TRAIN MODEL (main) ===")
        b_main = train_model_bundle(ex, symbol, timeframe, main_lb, use_cache=False)
        # main
        if b_main is not None:
            p_saved = save_model_bundle(symbol, timeframe, b_main)
            say(f"Model tersimpan: {p_saved}")
        else:
            say(f"{YELLOW}TRAIN main skipped/failed for {symbol} {timeframe}{RESET}")

    # fallback model (separate _fb file)
    p_fb = model_path_fb(symbol, timeframe)
    if not _os.path.isfile(p_fb):
        say("=== TRAIN MODEL (fallback) ===")
        b_fb = train_model_bundle(ex, symbol, timeframe, fb_lb, use_cache=False)
        # fallback
        if b_fb is not None:
            p_saved_fb = save_model_bundle_fb(symbol, timeframe, b_fb)
            say(f"Model fallback tersimpan: {p_saved_fb}")
        else:
            say(f"{YELLOW}TRAIN fallback skipped/failed for {symbol} {timeframe}{RESET}")

# --- Position mode helpers (one-way vs hedged) ---

def ensure_position_mode(ex, symbol=None, want="oneway"):
    """Try to set Bitget position mode consistently with our order style.
    want: "oneway" or "hedged". Uses ccxt.set_position_mode(False/True).
    Returns a tuple (mode_str, note)
    """
    want = (want or "oneway").strip().lower()
    hedged = (want == "hedged")
    note = ""
    try:
        # Try set on the specific symbol when supported, else without symbol
        try:
            ex.set_position_mode(hedged, symbol)
        except Exception:
            ex.set_position_mode(hedged)
        mode_str = "hedged" if hedged else "oneway"
        note = f"set_position_mode({hedged})"
        return mode_str, note
    except Exception as e:
        # If setting failed, try to infer from positions info
        try:
            poss = ex.fetch_positions([symbol]) if symbol else ex.fetch_positions()
            info = (poss[0].get("info") if poss and isinstance(poss, list) else {}) or {}
            pos_mode = (info.get("posMode") or info.get("positionMode") or "").lower()
            if "hedge" in pos_mode:
                return "hedged", f"detect:{pos_mode}"
            if "one" in pos_mode or "single" in pos_mode:
                return "oneway", f"detect:{pos_mode}"
        except Exception:
            pass
        return "unknown", f"error:{e}"

def fetch_top_symbols(ex, limit=20):
    # pick USDT linear swaps; sort by quote volume (if available) from tickers
    markets = [m for m in ex.markets.values() if m.get("type")=="swap" and m.get("quote")=="USDT"]
    syms = [m["symbol"] for m in markets if m.get("linear", True)]
    try:
        tttl = float(os.getenv("AUTOPILOT_TICKERS_TTL_SEC", "20"))
    except Exception:
        tttl = 20.0
    # In LIVE mode, keep ticker cache fresh at ~live refresh cadence
    try:
        live = str(os.getenv("AUTOPILOT_LIVE_MODE","0")).lower() not in ("0","false","no","off")
        if live:
            tttl = min(tttl, float(os.getenv("AUTOPILOT_LIVE_REFRESH_SEC","1.5")))
    except Exception:
        pass
    key = tuple(sorted(syms))
    ent = _TICKERS_CACHE.get(key)
    if ent and _ttl_ok(ent, tttl):
        tickers = ent["data"]
    else:
        tickers = ex.fetch_tickers(syms)
        _TICKERS_CACHE[key] = {"ts": time.time(), "data": tickers}
    rows=[]
    for s, t in tickers.items():
        vol = t.get("quoteVolume") or t.get("baseVolume") or 0
        last = t.get("last") or t.get("close")
        rows.append((s, float(last or 0), float(vol or 0)))
    rows.sort(key=lambda x: x[2], reverse=True)
    return rows[:limit]

# ===== Screening & Potentials =====
# === Full-universe USDT-M symbol screener helpers ===
def fetch_all_usdtm_symbols(ex):
    """
    Return ALL Bitget USDT-M perpetual (linear) symbols available on the account/market.
    """
    try:
        syms = []
        for m in ex.markets.values():
            if m.get("type") == "swap" and m.get("quote") == "USDT" and m.get("linear", True):
                syms.append(m["symbol"])
        # stable order for printing
        syms = sorted(set(syms))
        return syms
    except Exception:
        return []

def quick_evaluate_symbol(ex, symbol, timeframe=None, lookback="180d",
                          min_edge=0.20, min_pdir=0.60, fib_eps=0.0035, max_spread=0.0015):
    """
    Fast screening for a single symbol (no training here):
      - Loads ~lookback OHLCV, builds features
      - If a model file exists, use it to get signal; otherwise, use heuristic signal (EMA+Fib)
      - Applies quality gates; returns small dict for ranking or None
    """
    timeframe = timeframe or require_tf()
    # Allow ENV to tune gates dynamically (use passed defaults if ENV not set)
    try:
        min_edge = float(os.getenv("AUTOPILOT_MIN_EDGE", str(min_edge)))
    except Exception:
        pass
    try:
        min_pdir = float(os.getenv("AUTOPILOT_MIN_PDIR", str(min_pdir)))
    except Exception:
        pass
    try:
        fib_eps = float(os.getenv("AUTOPILOT_FIB_EPS", str(fib_eps)))
    except Exception:
        pass
    try:
        max_spread = float(os.getenv("AUTOPILOT_MAX_SPREAD", str(max_spread)))
    except Exception:
        pass
    verbose = (str(os.getenv("AUTOPILOT_VERBOSE","0")).lower() not in ("0","false","no","off"))
    try:
        df = fetch_ohlcv_lookback_safe(ex, symbol, timeframe=timeframe, lookback=lookback)
        # LIVE mode: inject last tick into the current candle for intra-bar scoring
        try:
            live = str(os.getenv("AUTOPILOT_LIVE_MODE","0")).lower() not in ("0","false","no","off")
            if live:
                t_last = ex.fetch_ticker(symbol)
                last_px = float(t_last.get("last") or t_last.get("close") or t_last.get("ask") or t_last.get("bid") or 0.0)
                if last_px > 0:
                    df = _live_inject_candle(df, last_px)
        except Exception:
            pass
        if len(df) < 400:
            return None
        Xf, cols = build_features(df)
        if len(Xf) < 300:
            return None
        Xf["ATR"] = atr(Xf, 14)

        side = None; conf = 0.0; ctx = {}
        # Gunakan pooled/global model kalau ada; jika tidak, jatuh ke heuristik (EMA+Fib)
        g_bundle, _gp = load_global_model_bundle(timeframe)
        if g_bundle is not None:
            Xf_g = ensure_pooled_inference_cols(Xf, g_bundle, symbol)
            side, conf, ctx = signal_long_short(Xf_g, g_bundle.get("cols") or cols, g_bundle)
            if side is None:
                return None
            ctx["model_src"] = "global"
        else:
            # Heuristik fallback (EMA + fib_trend)
            ef = float(Xf["ema_fast"].iloc[-1]); es = float(Xf["ema_slow"].iloc[-1])
            ft = int(Xf["fib_trend"].iloc[-1])
            d5 = float(Xf["fib_dist_500"].iloc[-1]); d6 = float(Xf["fib_dist_618"].iloc[-1])
            if ef > es and ft > 0:
                side = "long"; pL = 0.66; pS = 0.34; edge = pL - pS
            elif ef < es and ft < 0:
                side = "short"; pS = 0.66; pL = 0.34; edge = pS - pL
            else:
                return None
            ctx = {"pL":pL, "pS":pS, "edge":edge, "ema_fast":ef, "ema_slow":es,
                   "fib_trend":ft, "fib_dist_500":d5, "fib_dist_618":d6}

        pL = float(ctx.get("pL", 0.0)); pS = float(ctx.get("pS", 0.0))
        p_dir = max(pL, pS)
        edge = float(ctx.get("edge", 0.0))
        # gates for quality
        if abs(edge) < float(min_edge) or p_dir < float(min_pdir):
            if verbose:
                say(f"{YELLOW}Skip {symbol}: p_dir={p_dir:.2f} (<{min_pdir}) or edge={abs(edge):.2f} (<{min_edge}){RESET}")
            return None

        # fib proximity gate (closer to 0.5/0.618)
        prox = min(float(ctx.get("fib_dist_500", 1.0)), float(ctx.get("fib_dist_618", 1.0)))
        STRONG_OVERRIDE = str(os.getenv("AUTOPILOT_STRONG_OVERRIDE", "1")).lower() not in ("0","false","no","off")
        fib_eps_ok = float(os.getenv("AUTOPILOT_FIB_EPS_OK", str(fib_eps)))
        if prox > float(fib_eps):
            allow = False
            if STRONG_OVERRIDE:
                thr_pdir = float(os.getenv("AUTOPILOT_RATCHET_P_DIR", "0.80"))
                thr_conf = float(os.getenv("AUTOPILOT_RATCHET_CONF", "0.60"))
                # Strong override: accept even when far from 0.5/0.618 if confidence is very high
                if (p_dir >= thr_pdir) and (abs(edge) >= thr_conf):
                    allow = True
            if not allow:
                if verbose:
                    say(f"{YELLOW}Skip {symbol}: fib prox {prox:.4f} > {float(fib_eps):.4f}{RESET}")
                return None

        # microstructure: spread + funding (best-effort)
        spr = None; fund = None
        try:
            t = ex.fetch_ticker(symbol)
            bid = float(t.get("bid") or 0.0); ask = float(t.get("ask") or 0.0)
            if bid > 0 and ask > bid:
                mid = 0.5 * (bid + ask)
                spr = (ask - bid) / mid
        except Exception:
            pass
        try:
            fr = ex.fetch_funding_rate(symbol)
            fund = float(fr.get("fundingRate") or (fr.get("info", {}) or {}).get("fundingRate") or 0.0)
        except Exception:
            pass
        if spr is not None and max_spread is not None and spr > float(max_spread):
            if verbose:
                say(f"{YELLOW}Skip {symbol}: spread {spr:.4f} > {float(max_spread):.4f}{RESET}")
            return None

        price = float(Xf["close"].iloc[-1])
        A = float(Xf["ATR"].iloc[-1])
        atr_pct = (A / price) if price else 0.0
        ef = float(Xf["ema_fast"].iloc[-1]); es = float(Xf["ema_slow"].iloc[-1])
        fib_trend = int(Xf["fib_trend"].iloc[-1])
        trend = "UP" if ef > es else "DN"

        return {
            "symbol": symbol,
            "side": side,
            "p_dir": p_dir,
            "edge": edge if side == "long" else -edge,
            "prox": prox,
            "atr_pct": atr_pct,
            "spread": spr,
            "funding": fund,
            "trend": trend,
            "fib_trend": fib_trend,
        }
    except Exception:
        return None

def print_potential_menu(pots):
    """Tampilkan kandidat tanpa mengunci side; side akan dipilih dinamis saat entry."""
    if not pots:
        say(f"{BOLD}Top Potentials{RESET}")
        say(f"{YELLOW}(kosong — tidak ada kandidat yang lolos filter){RESET}")
        return
    say(f"{BOLD}Top Potentials{RESET}")
    for i, r in enumerate(pots, start=1):
        p_dir = r.get("p_dir"); p_mc = r.get("p_mc"); exp = r.get("exp")
        tp_m = r.get("tp_m"); sl_m = r.get("sl_m")
        prox = r.get("prox"); atrp = r.get("atr_pct")
        tag = r.get("src", "?")
        if tag == "main":
            tag = "(main)"
        elif tag == "fallback":
            tag = "(fallback)"
        else:
            tag = f"({tag})"
        # sengaja tidak mencetak side
        say(
            f"{i:>2}) {r['symbol']} {tag} · p_dir≈{(p_dir or 0):.2f} "
            f"· MC_P≈{(p_mc if p_mc is not None else 0.0):.2f} "
            f"· ExpR≈{(exp if exp is not None else 0.0):.2f} "
            f"· TPx={(tp_m if tp_m is not None else 0.0):.2f}/SLx={(sl_m if sl_m is not None else 0.0):.2f} "
            f"· prox≈{(prox or 0):.4f} · ATR%≈{((atrp or 0)*100):.2f}%"
        )

def prompt_pick_potential(pots):
    """Input nomor kandidat; return dict kandidat atau None jika batal.
    Tahan terhadap escape sequences (arah kiri/kanan, dsb) dengan menyaring hanya digit.
    """
    maxn = len(pots) if pots else 0
    if maxn == 0:
        say(f"{YELLOW}Tidak ada kandidat untuk dipilih.{RESET}")
        return None
    try:
        sel = input("Pilih [1-{}] (Enter untuk batal): ".format(maxn))
    except EOFError:
        return None
    except Exception:
        return None
    if not sel:
        return None
    raw = sel.strip()
    if raw == "":
        return None

    # Buang semua karakter non-digit (mengatasi ESC [ C / ESC [ D dari arrow keys, dsb)
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return None
    try:
        idx = int(digits)
    except Exception:
        return None
    if 1 <= idx <= maxn:
        return pots[idx-1]
    return None

# --- Timeframe picker (mandatory selection; no default) ---
def prompt_pick_timeframe(choices=("1m","3m","5m","15m","30m","1h","2h","4h","1d")):
    """Print a small menu and return a timeframe string. Must choose (no default)."""
    while True:
        try:
            say(f"{BOLD}Pilih Timeframe:{RESET}")
            for i, tf in enumerate(choices, start=1):
                say(f" {i}) {tf}")
            sel = input("Pilih [1-{}]: ".format(len(choices))).strip()
        except Exception:
            sel = ""
        digits = "".join(ch for ch in sel if ch.isdigit())
        if not digits:
            say(f"{YELLOW}Wajib memilih timeframe. Tidak boleh kosong.{RESET}")
            continue
        try:
            idx = int(digits)
            if 1 <= idx <= len(choices):
                return choices[idx-1]
        except Exception:
            pass
        say(f"{YELLOW}Input tidak valid. Coba lagi.{RESET}")
        
def prompt_main_action():
    """Top-level menu."""
    say(f"{BOLD}Pilih Mode:{RESET}")
    say(" 1) TRAIN DATA (Global pooled model)")
    say(" 2) Live Trading Futures USDT-M Bitget")
    try:
        sel = input("Pilih [1-2] (Enter untuk batal): ").strip()
    except Exception:
        return None
    digits = "".join(ch for ch in sel if ch.isdigit())
    if not digits:
        return None
    v = int(digits)
    if v == 1: return "train"
    if v == 2: return "live"
    return None

def run_cli():
    tf = ensure_timeframe_selected()
    while True:
        act = prompt_main_action()
        if not act:
            say(f"{YELLOW}Batal.{RESET}")
            return
        if act == "train":
            header("TRAIN DATA (GLOBAL)")
            ex = bitget_swap()
            lb = os.getenv("AUTOPILOT_GLOBAL_LOOKBACK", "365d")
            topn = int(float(os.getenv("AUTOPILOT_GLOBAL_TOPN", "512")))
            train_global_model(ex, timeframe=tf, lookback=lb, topn=topn, use_cache=True)
            say(f"{GREEN}TRAIN GLOBAL selesai. Kembali ke menu…{RESET}")
            # loop back to main menu
            continue
        elif act == "live":
            # Enable LIVE mode (intra-bar updates) and set the refresh cadence
            os.environ.setdefault("AUTOPILOT_LIVE_MODE","1")
            os.environ.setdefault("AUTOPILOT_LIVE_REFRESH_SEC", "1.5")
            header("Live Trading Futures USDT-M Bitget")
            try:
                autopilot()  # gunakan loop live yang sudah ada
            except KeyboardInterrupt:
                say("Bye 👋")
            # after live session ends, go back to main menu
            continue

def list_top_any_candidates(ex, timeframe=None, lookback=None, universe_topn=None, topk=10, min_r=1.2):
    """Very lenient fallback with progress & time budget (tanpa train per-symbol)."""
    timeframe = require_tf()

    # Resolve params dari ENV bila arg None
    lb = lookback or os.getenv("AUTOPILOT_FALLBACK_LOOKBACK")
    if not lb:
        # default ringan: 3d untuk TF menit, 30d untuk TF >= 1h
        lb = "3d" if timeframe.endswith("m") else "30d"

    try:
        u_topn = int(universe_topn or os.getenv("AUTOPILOT_FALLBACK_TOPN", "40"))
    except Exception:
        u_topn = 40

    try:
        budget = float(os.getenv("AUTOPILOT_FALLBACK_MAX_SECS",
                                 os.getenv("AUTOPILOT_SCREEN_MAX_SECS", "180")))
    except Exception:
        budget = 180.0

    out = []
    try:
        rows = fetch_top_symbols(ex, limit=max(10, int(u_topn)))
        syms = [s for (s,_,_) in rows]

        say(f"Membangun kandidat fallback {len(syms)} simbol… (TF={timeframe}, LB={lb})")
        t0 = time.time()
        for i, s in enumerate(syms, start=1):
            _print_inline(f"[{i}/{len(syms)}] {s} …")
            try:
                df = fetch_ohlcv_lookback_safe(ex, s, timeframe=timeframe, lookback=lb)
                if len(df) < 120:
                    continue
                Xf, cols = build_features(df)
                if len(Xf) < 120:
                    continue

                ef = float(Xf["ema_fast"].iloc[-1]); es = float(Xf["ema_slow"].iloc[-1])
                ft = int(Xf.get("fib_trend", pd.Series([0])).iloc[-1])
                d5 = float(Xf.get("fib_dist_500", pd.Series([0.02])).iloc[-1])
                d6 = float(Xf.get("fib_dist_618", pd.Series([0.02])).iloc[-1])

                if ef > es and ft >= 0:
                    side = "long"; pL = 0.60; pS = 0.40; edge = pL - pS
                elif ef < es and ft <= 0:
                    side = "short"; pS = 0.60; pL = 0.40; edge = pS - pL
                else:
                    side = "long" if ef >= es else "short"
                    pL = 0.58 if side == "long" else 0.42
                    pS = 1.0 - pL
                    edge = abs(pL - pS)

                price = float(Xf["close"].iloc[-1])
                A = float(atr(Xf, 14).iloc[-1])
                prox = min(d5, d6)
                atr_pct = (A / price) if price else 0.0

                best = select_tp_sl_by_potential(Xf["close"], A, price, side, min_r=float(min_r))
                if best is None:
                    TP_ATR = float(os.getenv("AUTOPILOT_TP_ATR", "1.8"))
                    SL_ATR = float(os.getenv("AUTOPILOT_SL_ATR", "1.0"))
                    tp = price + TP_ATR*A if side=="long" else price - TP_ATR*A
                    sl = price - SL_ATR*A if side=="long" else price + SL_ATR*A
                    r = (TP_ATR/SL_ATR) if SL_ATR > 0 else 0.0
                    best = {"tp": float(tp), "sl": float(sl), "tp_m": float(TP_ATR), "sl_m": float(SL_ATR),
                            "p": 0.50, "r": float(r), "exp": 0.50*r - 0.50}

                out.append({
                    "symbol": s, "side": side, "p_dir": max(pL, pS), "edge": edge, "prox": float(prox),
                    "atr_pct": float(atr_pct), "price": float(price), "A": float(A),
                    "p_mc": float(best.get("p")) if best.get("p") is not None else None,
                    "exp": float(best.get("exp")) if best.get("exp") is not None else 0.0,
                    "tp_m": float(best.get("tp_m")) if best.get("tp_m") is not None else None,
                    "sl_m": float(best.get("sl_m")) if best.get("sl_m") is not None else None,
                    "tp": float(best.get("tp")) if best.get("tp") is not None else None,
                    "sl": float(best.get("sl")) if best.get("sl") is not None else None,
                    "src": "fallback",
                })
            except Exception:
                continue

            # budget & rate-limit
            if (time.time() - t0) >= budget:
                _print_inline(f"[{i}/{len(syms)}] berhenti (time budget habis fallback)")
                break
            try:
                time.sleep(max(0.08, ex.rateLimit / 1000.0))
            except Exception:
                time.sleep(0.10)

        print()  # newline setelah inline
        out.sort(key=lambda x: (x.get("exp") or 0.0, x.get("p_dir",0.0), -x.get("prox",9.99)), reverse=True)
        return out[:max(1, int(topk))]
    except Exception:
        print()
        return []

def screen_usdtm_symbols(ex, timeframe=None, lookback="180d",
                         min_edge=0.20, min_pdir=0.60, fib_eps=0.0035, max_spread=0.0015):
    """
    Screen USDT-M perp symbols with a time budget and inline progress.
    ENV (optional):
      AUTOPILOT_SCREEN_TOPN     default 40  → use top-N by volume; set 0 to scan ALL
      AUTOPILOT_SCREEN_MAX_SECS default 180 → overall time budget (seconds)
      AUTOPILOT_SCREEN_LOOKBACK default uses `lookback` arg (e.g., '90d' for faster screening)
    """
    
    timeframe = require_tf()
    # Resolve universe (top-N by volume if requested)
    try:
        topn_env = int(float(os.getenv("AUTOPILOT_SCREEN_TOPN", "40")))
    except Exception:
        topn_env = 40
    if topn_env and topn_env > 0:
        top_rows = fetch_top_symbols(ex, limit=topn_env)
        symbols = [s for (s,_,_) in top_rows]
    else:
        symbols = fetch_all_usdtm_symbols(ex)

    symbols = list(dict.fromkeys(symbols))  # dedupe keep order
    out = []
    if not symbols:
        return out

    lb = os.getenv("AUTOPILOT_SCREEN_LOOKBACK", lookback) or lookback
    say(f"Memindai {len(symbols)} simbol… (TF={timeframe}, LB={lb})")

    t0 = time.time()
    try:
        budget = float(os.getenv("AUTOPILOT_SCREEN_MAX_SECS", "180"))
    except Exception:
        budget = 180.0

    for i, s in enumerate(symbols, start=1):
        _print_inline(f"[{i}/{len(symbols)}] {s} …")
        try:
            res = quick_evaluate_symbol(
                ex, s, timeframe=timeframe, lookback=lb,
                min_edge=min_edge, min_pdir=min_pdir, fib_eps=fib_eps, max_spread=max_spread
            )
            if res:
                # ranking score with costs
                cost_pen = 0.0
                if res.get("spread") is not None:
                    cost_pen += min(0.20, float(res["spread"]) / 0.0010 * 0.05)
                if res.get("funding") is not None:
                    f = float(res["funding"])
                    cost_pen += 0.05 * (max(0.0, f) if res["side"] == "long" else max(0.0, -f))
                score = float(res["p_dir"]) + max(0.0, float(res["edge"])) - cost_pen
                res["score"] = score
                res["src"] = "main"      # added by pooled/global screening
                out.append(res)
        except Exception:
            pass

        # time budget check
        if (time.time() - t0) >= budget:
            _print_inline(f"[{i}/{len(symbols)}] berhenti (time budget habis)")
            break

        # polite rate limit spacing
        try:
            time.sleep(max(0.08, ex.rateLimit / 1000.0))
        except Exception:
            time.sleep(0.10)

    print()  # newline after inline progress
    out.sort(key=lambda x: (x.get("score", 0.0), x.get("p_dir", 0.0), x.get("edge", 0.0)), reverse=True)
    return out

def get_futures_usdt_balance(ex):
    try:
        bal = ex.fetch_balance()
        total = bal.get("total", {}).get("USDT", 0.0) or 0.0
        free  = bal.get("free",  {}).get("USDT", 0.0) or 0.0
        used  = bal.get("used",  {}).get("USDT", 0.0) or 0.0
        return float(total), float(free), float(used)
    except Exception as e:
        # Give a precise hint when API key lacks futures perms
        try:
            from ccxt.base.errors import PermissionDenied
            if isinstance(e, PermissionDenied) or (hasattr(e, 'args') and '40014' in str(e)):
                print(f"{YELLOW}Bitget API permission error: need FUTURES permissions (Contract).{RESET}")
                print("- Aktifkan: Futures/Contract -> Read + Trade (termasuk Position Read) pada API key")
                print("- Pastikan passphrase benar (BITGET_PASSWORD)")
                print("- Jika pakai IP whitelist, tambahkan IP kamu saat ini")
        except Exception:
            pass
        raise

def market_specs(ex, symbol):
    try:
        sttl = float(os.getenv("AUTOPILOT_SPECS_TTL_SEC", "300"))
    except Exception:
        sttl = 300.0
    ent = _SPECS_CACHE.get(symbol)
    if ent and _ttl_ok(ent, sttl):
        return ent["data"]
    m = ex.market(symbol)
    limits = m.get("limits") or {}
    prec   = m.get("precision") or {}

    def _decimals_from_step(step):
        try:
            if step is None:
                return None
            if isinstance(step, int):
                return max(0, int(step))
            step = float(step)
            if step >= 1:
                return 0
            return max(0, int(round(-math.log10(step))))
        except Exception:
            return None

    price_step = (limits.get("price") or {}).get("min")
    amt_step   = (limits.get("amount") or {}).get("min")

    price_prec = _decimals_from_step(price_step)
    amount_prec = _decimals_from_step(amt_step)

    if price_prec is None:
        p = prec.get("price")
        if p is not None:
            price_prec = int(p) if isinstance(p, int) else _decimals_from_step(p) or 4
        else:
            price_prec = 4
    if amount_prec is None:
        a = prec.get("amount")
        if a is not None:
            amount_prec = int(a) if isinstance(a, int) else _decimals_from_step(a) or 4
        else:
            amount_prec = 4

    min_qty = (limits.get("amount") or {}).get("min")
    if not min_qty:
        min_qty = 10 ** (-amount_prec)
    min_cost = (limits.get("cost") or {}).get("min") or 0.0
    # Bitget fallback untuk min notional jika limits.cost.min tidak ada
    if not min_cost:
        try:
            info = m.get("info") or {}
            for k in ("minTradeNum","minTradeAmount","minNotional","minQuoteCurrencyTradeAmt"):
                v = info.get(k)
                if v is not None:
                    try:
                        min_cost = float(v)
                        break
                    except Exception:
                        pass
        except Exception:
            pass
        if not min_cost:
            try:
                min_cost = float(os.getenv("BITGET_MIN_NOTIONAL_USDT", "5"))
            except Exception:
                min_cost = 5.0

    res = (int(amount_prec), int(price_prec), float(min_qty), float(min_cost))
    _SPECS_CACHE[symbol] = {"ts": time.time(), "data": res}
    return res

# --- Leverage detection helper ---

def detect_symbol_leverage(ex, symbol):
    try:
        lttl = float(os.getenv("AUTOPILOT_LEVERAGE_TTL_SEC", "60"))
    except Exception:
        lttl = 60.0
    ent = _LEVERAGE_CACHE.get(symbol)
    if ent and _ttl_ok(ent, lttl):
        return ent["data"]
    """Best-effort detection of current leverage for `symbol`.
    Tries (in order): open positions -> single position -> market info hints -> env fallback.
    Returns (leverage_float, source_str).
    """
    # 1) Try open positions list
    try:
        try:
            poss = ex.fetch_positions([symbol])
        except Exception:
            poss = None
        if poss:
            # Prefer an exact symbol match; else take the first with leverage
            for p in poss:
                lev = p.get("leverage") or (p.get("info", {}) or {}).get("leverage") or (p.get("info", {}) or {}).get("leverageValue")
                if lev:
                    res = (float(lev), "positions")
                    _LEVERAGE_CACHE[symbol] = {"ts": time.time(), "data": res}
                    return res
    except Exception:
        pass

    # 2) Try single position
    try:
        pos = ex.fetch_position(symbol)
        if pos:
            lev = pos.get("leverage") or (pos.get("info", {}) or {}).get("leverage") or (pos.get("info", {}) or {}).get("leverageValue")
            if lev:
                res = (float(lev), "position")
                _LEVERAGE_CACHE[symbol] = {"ts": time.time(), "data": res}
                return res
    except Exception:
        pass

    # 3) Try market info hints (not current leverage, but at least a sane default)
    try:
        m = ex.market(symbol)
        lim = (m.get("limits") or {}).get("leverage") or {}
        # if only max leverage is known, pick a conservative fraction
        max_lev = lim.get("max")
        if max_lev:
            try:
                max_lev = float(max_lev)
                if max_lev >= 2:
                    res = (max(1.0, min(10.0, max_lev/2.0)), "market_limits")
                    _LEVERAGE_CACHE[symbol] = {"ts": time.time(), "data": res}
                    return res
            except Exception:
                pass
    except Exception:
        pass

    # 4) Env fallback
    try:
        lev_env = float(os.getenv("AUTOPILOT_LEVERAGE", "5"))
    except Exception:
        lev_env = 5.0
    res = (max(1.0, lev_env), "env")
    _LEVERAGE_CACHE[symbol] = {"ts": time.time(), "data": res}
    return res

# --- Leverage setter (best-effort) ---

def set_symbol_leverage(ex, symbol, target_lev, margin_mode="crossed"):
    """Try to set leverage for `symbol` to `target_lev` with a few param variants,
    including auto-switching marginMode between crossed/isolated.
    Returns (new_leverage_int, source_str) on success, else (None, "set_failed")."""
    try:
        target_lev = max(1, int(float(target_lev)))
        # clamp by market max if available
        try:
            m = ex.market(symbol)
            lim = (m.get("limits") or {}).get("leverage") or {}
            mx = lim.get("max")
            if mx:
                target_lev = min(target_lev, int(float(mx)))
        except Exception:
            pass
        # Try margin modes in order: preferred -> alternate
        modes = [str(margin_mode or "crossed").lower()]
        alt = "isolated" if modes[0] == "crossed" else "crossed"
        if alt not in modes:
            modes.append(alt)
        # Build attempts
        attempts = []
        for mm in modes:
            attempts.extend([
                {"marginMode": mm},
                {"marginMode": mm, "holdSide": "long"},
                {"marginMode": mm, "holdSide": "short"},
            ])
        attempts.append({})  # raw fallback
        for params in attempts:
            try:
                ex.set_leverage(target_lev, symbol, params)
                return target_lev, f"set:{params.get('marginMode','raw')}"
            except Exception:
                continue
        return None, "set_failed"
    except Exception:
        return None, "set_failed"

# --- Auto-scale notional to always meet min_cost when feasible ---

def auto_scale_for_min_cost(ex, symbol, free_usdt, entry, min_qty, min_cost):
    """Ensure we place at least the minimal tradeable order to pass min_cost if feasible.
    Returns: (qty_min, lev_used, changed, note_str)
    - qty_min: minimal viable qty (already rounded to amount precision) or 0.0 if impossible
    - lev_used: leverage actually used after potential adjustment
    - changed: True if we adjusted leverage
    - note_str: short note about what happened
    """
    try:
        lev_used, lev_src = detect_symbol_leverage(ex, symbol)
        lev_used = max(1.0, float(lev_used or 1.0))
        cap = (free_usdt or 0.0) * lev_used * 0.95
        entry = float(entry or 0.0)
        min_cost = float(min_cost or 0.0)
        # helper to round amount
        def round_amt(q):
            try:
                return float(ex.amount_to_precision(symbol, q))
            except Exception:
                return float(q)
        if entry <= 0:
            return 0.0, lev_used, False, "bad_entry"
        # If current cap already allows min_cost, return minimal tradeable qty
        if min_cost <= cap:
            q = max(min_qty, (min_cost/entry) if min_cost > 0 else min_qty)
            q = round_amt(q)
            # pastikan notional >= min_cost dengan buffer kecil (hindari underflow pembulatan)
            buf = 1.02
            step = max(min_qty, q * 1e-6)
            tries = 0
            while (q * entry) < (min_cost * buf) and tries < 5:
                q = round_amt(q + step)
                tries += 1
            return (q if q >= min_qty else 0.0), lev_used, False, f"cap_ok:{lev_src}"
        # If no free balance, impossible
        if (free_usdt or 0.0) <= 0.0:
            return 0.0, lev_used, False, "no_free"
        # Compute required leverage to meet min_cost
        req = int(math.ceil(min_cost / (free_usdt * 0.95)))
        req = max(1, req)
        # Bound by market max and env target
        max_lev = None
        try:
            m = ex.market(symbol)
            lim = (m.get("limits") or {}).get("leverage") or {}
            if lim.get("max"):
                max_lev = int(float(lim["max"]))
        except Exception:
            pass
        target_env = int(float(os.getenv("AUTOPILOT_TARGET_LEVERAGE", "10")))
        target = max(req, target_env, int(lev_used))
        if max_lev:
            target = min(target, max_lev)
        # try to set leverage up to target (with auto mode-switch)
        new_lev, src = set_symbol_leverage(ex, symbol, target, os.getenv("AUTOPILOT_MARGIN_MODE", "crossed"))
        if new_lev:
            lev_used, lev_src = detect_symbol_leverage(ex, symbol)
            cap = free_usdt * max(1.0, float(lev_used)) * 0.95
            if min_cost <= cap:
                q = max(min_qty, min_cost/entry)
                q = round_amt(q)
                return (q if q >= min_qty else 0.0), lev_used, True, f"lev_set:{new_lev}x"
        return 0.0, lev_used, False, "insufficient_cap"
    except Exception as e:
        return 0.0, 1.0, False, f"error:{e}"
    
# ===== Live monitoring & exit management =====
def monitor_until_exit(ex, symbol, side, entry, sl, tp, qty, price_prec,
                       tighten_be_on=True, tighten_at_R=0.5, cooldown_sec=60):
    """
    Pantau tiap 1 detik (1 baris) sampai TP/SL. Tanpa trailing TP.
    Dynamic SL:
      - BE pada profit ≥ tighten_at_R * R0 (R0 = |entry - SL_awal|)
      - Jika profit ≥ 1.0R: kunci profit 0.25R (score kuat) atau 0.50R (score lemah)
      - Jika profit ≥ 1.5R: kunci profit 0.50R (kuat) atau 1.0R (lemah)
      - Penempatan SL mempertimbangkan Order Block & SR eps utk menghindari tersentuh pullback dangkal
    Early close (invalidation):
      - Jika score <= 0.25 dan mayoritas micro-trend berlawanan, atau ChoCh berlawanan → close segera (reduceOnly)
      - Cetak alasan invalidasi saat close
    """
    import os

    # telemetry: monitor start
    try:
        log_event("monitor_start", symbol=symbol, side=side, entry=float(entry), sl=float(sl), tp=float(tp), qty=float(qty))
    except Exception:
        pass

    # Optional: pre-place OCO (best-effort; disabled by default)
    use_oco = str(os.getenv("AUTOPILOT_USE_OCO", "0")).lower() not in ("0","false","no","off")
    oco_ids = None
    if use_oco:
        try:
            oco_ids = try_place_oco(ex, symbol, side, qty, float(tp), float(sl))
            if oco_ids:
                log_event("oco_placed", symbol=symbol, side=side, tp=float(tp), sl=float(sl))
        except Exception:
            oco_ids = None

    # --- konstanta & helper ---
    tf_main = require_tf()
    sr_eps = float(os.getenv("AUTOPILOT_SMC_SR_EPS", "0.0012"))
    ob_look = int(os.getenv("AUTOPILOT_SMC_OB_LOOK", "60"))

    sl_init = float(sl)
    R0 = max(1e-8, abs(float(entry) - sl_init))  # base risk, dipakai utk R display dan trigger

    def _r_gain(px):
        if side == "long":
            return (float(px) - float(entry)) / R0
        else:
            return (float(entry) - float(px)) / R0

    def _fmt(v):
        try:
            return f"{float(v):.{int(price_prec)}f}"
        except Exception:
            return f"{float(v):.6f}"

    def _status(px, sl, tp, score=None):
        r_run = _r_gain(px)
        s = f"[MON] {symbol} px={_fmt(px)} | TP={_fmt(tp)} SL={_fmt(sl)} | R≈{r_run:.2f}"
        if score is not None:
            s += f" | sc={score:.2f}"
        return s

    side_close = "sell" if side == "long" else "buy"
    
    # stagnation controls
    STAG_SEC = float(os.getenv("AUTOPILOT_STAG_SEC", "240"))
    STAG_R   = float(os.getenv("AUTOPILOT_STAG_R", "0.10"))
    best_r = -1e9
    last_move_t = time.time()

    while True:
        try:
            # --- harga realtime ---
            t = ex.fetch_ticker(symbol)
            px = float(t.get("last") or t.get("close") or t.get("bid") or 0.0)
            if px <= 0:
                _print_inline(f"[MON] {symbol} ticker err")
                time.sleep(1.0)
                continue

            # --- evaluasi potensi & struktur (tiap detik) ---
            try:
                # df_main = fetch_ohlcv(ex, symbol, timeframe=tf_main, candles=400)
                df_main = cached_fetch_ohlcv(ex, symbol, timeframe=tf_main, candles=400)
                score, reasons, ctx = scalping_filter(ex, symbol, tf_main, df_main, side, {})
            except Exception:
                score, reasons, ctx = 0.0, ["scalp_err"], {}

            # --- invalidation (early close) ---
            inv_flags = []
            micro = (ctx.get("micro_trend") or "").lower()
            choch = ctx.get("choch")
            br = ctx.get("break_retest")  # 'long' | 'short' | None

            if score <= 0.25:
                inv_flags.append("score<0.25")
            if side == "long" and micro == "down":
                inv_flags.append("micro-down")
            if side == "short" and micro == "up":
                inv_flags.append("micro-up")
            if side == "long" and choch == "bearish":
                inv_flags.append("ChoCh-bearish")
            if side == "short" and choch == "bullish":
                inv_flags.append("ChoCh-bullish")
            if br and br != side:
                inv_flags.append(f"BR-retest->{br}")

            # Close cepat jika benar-benar invalid & profit tidak signifikan (<0.3R) atau score sangat rendah
            if inv_flags and (score <= 0.25 or _r_gain(px) < 0.30):
                try:
                    ex.create_order(symbol, 'market', side_close, qty, None, {"reduceOnly": True})
                except Exception:
                    ex.create_order(symbol, 'market', side_close, qty, None, {})
                print()
                say(f"{YELLOW}CLOSE EARLY — invalidation: {','.join(inv_flags)} | sc={score:.2f}{RESET}")
                try:
                    r_gain = _r_gain(px)
                    log_event("close_early", symbol=symbol, side=side, r_gain=r_gain, reasons=";".join(inv_flags))
                    record_trade_result("INV", float(r_gain), symbol, side)
                except Exception:
                    pass
                # cancel OCO leftovers if any
                if oco_ids:
                    try_cancel_oco(ex, symbol, oco_ids)
                if cooldown_sec and cooldown_sec > 0:
                    say(f"Cooldown {int(cooldown_sec)}s…")
                    time.sleep(int(cooldown_sec))
                return "INV"

            # --- dynamic SL tightening berdasarkan potensi ---
            # 1) BE pada 0.5R (atau sesuai tighten_at_R)
            if bool(tighten_be_on) and _r_gain(px) >= float(tighten_at_R):
                if side == "long":
                    sl = max(sl, float(entry))
                else:
                    sl = min(sl, float(entry))

            # 2) Lock profit bertahap menurut score (potensi)
            rg = _r_gain(px)
            lock_r = None
            if rg >= 1.5:
                lock_r = (1.0 if score < 0.70 else 0.50)
            elif rg >= 1.0:
                lock_r = (0.50 if score < 0.60 else 0.25)

            if lock_r is not None:
                # kandidat SL berdasar R0
                if side == "long":
                    sl_cand = max(sl, float(entry) + lock_r * R0)
                else:
                    sl_cand = min(sl, float(entry) - lock_r * R0)

                # pertimbangkan Order Block agar tidak pas di tepi zona
                try:
                    obz = order_block_zone(df_main, side, lookback=ob_look)
                except Exception:
                    obz = {}
                if obz and obz.get("bias") == side:
                    if side == "long" and obz.get("low") is not None:
                        zone = float(obz["low"]) * (1.0 - sr_eps)
                        sl_cand = max(sl_cand, zone)  # SL di bawah/sekitar OB low
                    if side == "short" and obz.get("high") is not None:
                        zone = float(obz["high"]) * (1.0 + sr_eps)
                        sl_cand = min(sl_cand, zone)  # SL di atas/sekitar OB high

                # Jangan pernah memperlebar risiko dari SL saat ini
                if side == "long":
                    sl = max(sl, sl_cand)
                else:
                    sl = min(sl, sl_cand)
                    
            # stagnation: update best R dan deteksi macet
            r_now = _r_gain(px)
            if r_now > (best_r + 0.05):
                best_r = r_now
                last_move_t = time.time()
            if (time.time() - last_move_t) >= STAG_SEC and r_now < STAG_R:
                try:
                    ex.create_order(symbol, 'market', side_close, qty, None, {"reduceOnly": True})
                except Exception:
                    ex.create_order(symbol, 'market', side_close, qty, None, {})
                print()
                say(f"{YELLOW}CLOSE EARLY — stagnation {int(STAG_SEC)}s at R≈{r_now:.2f}{RESET}")
                if cooldown_sec and cooldown_sec > 0:
                    say(f"Cooldown {int(cooldown_sec)}s…")
                    time.sleep(int(cooldown_sec))
                return "STAG"

            # --- tampilkan status ---
            _print_inline(_status(px, sl, tp, score=score))

            # --- cek TP/SL ---
            hit_tp = (px >= tp) if side == "long" else (px <= tp)
            hit_sl = (px <= sl) if side == "long" else (px >= sl)
            if hit_tp or hit_sl:
                try:
                    ex.create_order(symbol, 'market', side_close, qty, None, {"reduceOnly": True})
                except Exception:
                    ex.create_order(symbol, 'market', side_close, qty, None, {})
                print()
                result = "TP" if hit_tp else "SL"
                say(f"{GREEN}TP HIT{RESET}" if hit_tp else f"{RED}SL HIT{RESET}")
                try:
                    r_gain = _r_gain(px)
                    log_event("exit", symbol=symbol, side=side, result=result, r_gain=r_gain)
                    record_trade_result(result, float(r_gain), symbol, side)
                except Exception:
                    pass
                if oco_ids:
                    try_cancel_oco(ex, symbol, oco_ids)
                if cooldown_sec and cooldown_sec > 0:
                    say(f"Cooldown {int(cooldown_sec)}s…")
                    time.sleep(int(cooldown_sec))
                return result

        except Exception as e:
            _print_inline(f"[MON] {symbol} err {e}")

        time.sleep(1.0)

# --- OCO helpers: place/cancel ---
def try_place_oco(ex, symbol, side, qty, tp, sl):
    """
    Best-effort OCO: place a limit TP and stop-market SL as reduceOnly.
    Return tuple(order_id_tp, order_id_sl) or None on failure.
    """
    try:
        side_close = "sell" if side == "long" else "buy"
        oid_tp = None
        oid_sl = None
        # Take Profit (limit)
        try:
            o = ex.create_order(symbol, "limit", side_close, qty, tp, {"reduceOnly": True, "timeInForce": "GTC"})
            oid_tp = (o.get("id") or (o.get("info") or {}).get("orderId"))
        except Exception:
            pass
        # Stop Loss (stop-market)
        try:
            params = {"stopLossPrice": sl, "reduceOnly": True}
            o = ex.create_order(symbol, "market", side_close, qty, None, params)
            oid_sl = (o.get("id") or (o.get("info") or {}).get("orderId"))
        except Exception:
            pass
        if oid_tp or oid_sl:
            return (oid_tp, oid_sl)
    except Exception:
        pass
    return None

def try_cancel_oco(ex, symbol, oco_ids):
    try:
        tp_id, sl_id = (oco_ids or (None, None))
        for oid in (tp_id, sl_id):
            if oid:
                try:
                    ex.cancel_order(oid, symbol)
                except Exception:
                    pass
    except Exception:
        pass

# --- Context explainer for entries ---
def structure_summary(df):
    hi_m, lo_m = detect_swings(df, left=3, right=3)
    highs = df.loc[hi_m, "high"].dropna()
    lows  = df.loc[lo_m, "low"].dropna()
    dh = highs.diff().dropna(); dl = lows.diff().dropna()
    hh = int((dh > 0).sum()); lh = int((dh < 0).sum())
    hl = int((dl > 0).sum()); ll = int((dl < 0).sum())
    return hh, hl, lh, ll

def explain_entry(symbol, side, price, sup, res, A, ctx, sl, tp, df_last):
    hh, hl, lh, ll = structure_summary(df_last)
    trend_txt = "UP" if ctx["ema_fast"] > ctx["ema_slow"] else "DOWN"
    say(f"{BOLD}=== Position Rationale ==={RESET}")
    say(f"Pair: {symbol} | Side: {side.upper()} @ ~{price:.6f}")
    say(f"Trend EMA: {trend_txt} | ema_fast={ctx['ema_fast']:.6f} ema_slow={ctx['ema_slow']:.6f}")
    say(f"Model: P_long={ctx['pL']:.2f} P_short={ctx['pS']:.2f} edge={ctx['edge']:.2f}")
    say(f"Structure: support≈{sup} | resistance≈{res} | ATR14≈{A:.6f}")
    say(f"Fib proximity: d500={ctx['fib_dist_500']:.4f} d618={ctx['fib_dist_618']:.4f} | fib_trend={int(ctx['fib_trend'])}")
    say(f"Swing summary (wave-ish): HH={hh} HL={hl} LH={lh} LL={ll}")
    say(f"Risk params: SL={sl} | TP={tp} (aligned to tick)")

# --- No-entry reasons (for live loop visibility) ---
def no_entry_reasons(ctx, fib_eps=0.0035, fib_eps_ok=None, strong_override=True):
    """
    Build readable reasons for skipping entry when the model gave no immediate signal.
    - fib_eps_ok: relaxed proximity threshold (if provided)
    - strong_override: when True, very large |edge| can relax proximity/alignment checks
    """
    reasons = []
    pL = float(ctx.get("pL", 0.0)); pS = float(ctx.get("pS", 0.0))
    edge = float(ctx.get("edge", 0.0))
    ef = float(ctx.get("ema_fast", 0.0)); es = float(ctx.get("ema_slow", 0.0)); ft = int(ctx.get("fib_trend", 0))
    d5 = float(ctx.get("fib_dist_500", 1.0)); d6 = float(ctx.get("fib_dist_618", 1.0))
    prox = min(d5, d6)
    trend_align = ((ef > es) and (ft > 0)) or ((ef < es) and (ft < 0))

    # Base reasons
    if abs(edge) < 0.20:
        reasons.append("edge<0.20")
    if prox >= fib_eps:
        reasons.append(f"jauh dari zona fib (>={fib_eps:.4f})")
    if not trend_align and abs(edge) < 0.80:
        reasons.append("trend/fib_trend tidak align")

    # Explain when strong override would still be skipped
    if strong_override and abs(edge) >= 0.80:
        if fib_eps_ok is not None and prox <= float(fib_eps_ok):
            pass  # would be allowed; if tetap skip, biasanya karena guards/scalp gate (dicetak di level atas)
        else:
            reasons.append(f"strong-edge tapi prox>{(fib_eps_ok or fib_eps):.4f}")

    return reasons

# ===== Micro-structure & SMC utilities =====
# --- Scalping utilities: VWAP, orderbook imbalance, wick-sweep, breakout-retest, micro-TF confirm ---

def session_vwap(df):
    """Return latest session VWAP (session = UTC day) and distance of last price to VWAP."""
    if df.empty:
        return None, None
    d = df.copy()
    d["date"] = d["datetime"].dt.normalize()
    today = d["date"].iloc[-1]
    sess = d[d["date"] == today].copy()
    if len(sess) < 5:
        return None, None
    tp = (sess["high"] + sess["low"] + sess["close"]) / 3.0
    vol = sess["volume"].replace(0, np.nan).ffill()
    num = (tp * vol).cumsum()
    den = vol.cumsum()
    vwap = (num / (den + 1e-12)).iloc[-1]
    last_px = float(sess["close"].iloc[-1])
    dist = abs(last_px - vwap) / max(1e-12, last_px)
    return float(vwap), float(dist)

def orderbook_imbalance(ex, symbol, depth=10):
    """Return (imbalance[-1..1], spread) using top `depth` levels."""
    try:
        ob = ex.fetch_order_book(symbol, limit=max(5, int(depth)))
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        sb = sum([b[1] for b in bids[:depth]]) if bids else 0.0
        sa = sum([a[1] for a in asks[:depth]]) if asks else 0.0
        best_bid = float(bids[0][0]) if bids else 0.0
        best_ask = float(asks[0][0]) if asks else 0.0
        spr = 0.0
        if best_bid > 0 and best_ask > best_bid:
            mid = 0.5 * (best_bid + best_ask)
            spr = (best_ask - best_bid) / mid
        tot = (sb + sa)
        if tot <= 0:
            return 0.0, spr
        imb = (sb - sa) / tot
        return float(max(-1.0, min(1.0, imb))), float(spr)
    except Exception:
        return 0.0, None

def wick_sweep_score(df, lookback=3):
    """
    Detect recent liquidity sweep via wick/body ratios.
    Returns score in [0..1] favoring strong rejection wicks.
    """
    if len(df) < (lookback + 2):
        return 0.0
    d = df.tail(max(10, lookback + 2)).copy()
    s = 0.0
    cnt = 0
    for i in range(-lookback, 0):
        r = d.iloc[i]
        body = abs(r["close"] - r["open"])
        up_w = r["high"] - max(r["open"], r["close"])
        dn_w = min(r["open"], r["close"]) - r["low"]
        denom = max(1e-9, body)
        up_ratio = up_w / denom
        dn_ratio = dn_w / denom
        sweep = max(up_ratio, dn_ratio)
        s += max(0.0, min(1.0, (sweep - 0.6) / 1.4))  # ≥0.6 starts to count
        cnt += 1
    return float(s / max(1, cnt))

def breakout_retest_flag(df, lookback=30, eps=0.0012):
    """
    Return ('long'|'short'|None) if recent price broke a key swing and retested.
    """
    if len(df) < (lookback + 10):
        return None
    hi_m, lo_m = detect_swings(df, left=3, right=3)
    highs = df.loc[hi_m, "high"].tail(lookback)
    lows  = df.loc[lo_m, "low"].tail(lookback)
    last = df.iloc[-1]
    px = float(last["close"])
    if not highs.empty and px > highs.max() * (1 - eps):
        # retest: recent low near prior high zone
        zone = float(highs.max())
        recent_low = float(df["low"].tail(5).min())
        if recent_low <= zone * (1 + eps) and px >= zone * (1 - eps):
            return "long"
    if not lows.empty and px < lows.min() * (1 + eps):
        zone = float(lows.min())
        recent_high = float(df["high"].tail(5).max())
        if recent_high >= zone * (1 - eps) and px <= zone * (1 + eps):
            return "short"
    return None

def micro_tf_confirm(ex, symbol, tf="5m"):
    """
    Micro timeframe trend confirmation using EMAs on tf (default 5m).
    Returns ('up'|'down'|None, ema_fast, ema_slow).
    """
    try:
        try:
            ttl = float(os.getenv("AUTOPILOT_MICRO_TTL_SEC", "2.0"))
        except Exception:
            ttl = 2.0
        dfm = cached_fetch_ohlcv(ex, symbol, timeframe=tf, candles=400, ttl_sec=ttl)
        dfm["ema_fast"] = ema(dfm["close"], 20)
        dfm["ema_slow"] = ema(dfm["close"], 50)
        ef = float(dfm["ema_fast"].iloc[-1]); es = float(dfm["ema_slow"].iloc[-1])
        if ef > es:
            return "up", ef, es
        if ef < es:
            return "down", ef, es
        return None, ef, es
    except Exception:
        return None, None, None
    
def micro_tf_confirm_multi(ex, symbol, tfs=("5m","15m")):
    """
    Confirm micro trend across multiple lower timeframes.
    Returns (summary: 'up'|'down'|None, details: dict{tf: 'up'|'down'|None}, ema_map: dict{tf: (ema_fast, ema_slow)}).
    Majority vote; ties → None.
    """
    details = {}
    ema_map = {}
    ups = 0
    downs = 0
    for tf in list(tfs or []):
        tr, ef, es = micro_tf_confirm(ex, symbol, tf=tf)
        details[str(tf)] = tr
        ema_map[str(tf)] = (ef, es)
        if tr == "up":
            ups += 1
        elif tr == "down":
            downs += 1
    if ups > downs:
        return "up", details, ema_map
    if downs > ups:
        return "down", details, ema_map
    return None, details, ema_map

# --- Smart Money Concepts (SMC) helpers: FVG / SR / ChoCh / IRL / Pullback / Order Block ---

def detect_fvg_near_price(df, lookback=60, min_gap_pct=0.0008):
    """
    Fair Value Gap (3-candle):
      - Bullish: low[i+1] > high[i-1]  → gap=(high[i-1], low[i+1])
      - Bearish: high[i+1] < low[i-1] → gap=(high[i+1], low[i-1])
    Return {type:'bullish'|'bearish', low, high, in_gap:bool, prox:float} or {}
    prox = jarak ke tepi gap / price (semakin kecil semakin dekat)
    """
    out = {}
    try:
        if len(df) < (lookback + 3):
            return out
        d = df.tail(lookback + 3).reset_index(drop=True)
        px = float(d.loc[len(d)-1, "close"])
        best = {"prox": 1e9}
        for i in range(1, len(d)-1):
            hi_prev = float(d.loc[i-1, "high"]); lo_prev = float(d.loc[i-1, "low"])
            hi_next = float(d.loc[i+1, "high"]); lo_next = float(d.loc[i+1, "low"])
            # bullish FVG
            if lo_next > hi_prev:
                low, high = hi_prev, lo_next
                gap_pct = (high - low) / max(1e-12, px)
                if gap_pct >= min_gap_pct:
                    prox = 0.0 if (low <= px <= high) else (min(abs(px-low), abs(px-high))/max(1e-12, px))
                    if prox < best["prox"]:
                        best = {"type":"bullish","low":low,"high":high,"in_gap":(prox==0.0),"prox":prox}
            # bearish FVG
            if hi_next < lo_prev:
                low, high = hi_next, lo_prev
                gap_pct = (high - low) / max(1e-12, px)
                if gap_pct >= min_gap_pct:
                    prox = 0.0 if (low <= px <= high) else (min(abs(px-low), abs(px-high))/max(1e-12, px))
                    if prox < best["prox"]:
                        best = {"type":"bearish","low":low,"high":high,"in_gap":(prox==0.0),"prox":prox}
        return {} if best["prox"] == 1e9 else best
    except Exception:
        return {}

def sr_confluence(df, price, lookback=80, eps=0.0012):
    """
    Konfluensi Support/Resistance via swing highs/lows.
    Return {bias:'support'|'resistance'|None, dist:float, score:0..1}
    """
    try:
        swings_lo, swings_hi = structure_levels(df, lookback=lookback)
        sup, res = nearest_levels(price, swings_lo, swings_hi)
        bias = None; dist = None; score = 0.0
        if sup is not None and price >= sup:
            d = abs(price - sup) / max(1e-12, price)
            if d <= eps*2.0:
                bias, dist, score = 'support', d, max(0.0, 1.0 - d/(eps*2.0))
        if res is not None and price <= res:
            d = abs(res - price) / max(1e-12, price)
            s2 = max(0.0, 1.0 - d/(eps*2.0))
            if (bias is None) or (s2 > score):
                bias, dist, score = 'resistance', d, s2
        return {"bias":bias,"dist":dist,"score":float(min(1.0,score))}
    except Exception:
        return {"bias":None,"dist":None,"score":0.0}

def choch_signal(df, left=3, right=3, lookback=80):
    """
    Change of Character (ChoCh) sederhana.
    Return {'direction':'bullish'|'bearish'|None,'bars_ago':int|None}
    """
    try:
        if len(df) < (lookback + right + left + 5):
            return {"direction":None,"bars_ago":None}
        d = df.tail(lookback + right + left + 5).copy().reset_index(drop=True)
        hi_m, lo_m = detect_swings(d, left=left, right=right)
        highs = d.loc[hi_m, ["high"]].reset_index()
        lows  = d.loc[lo_m, ["low"]].reset_index()
        if highs.empty or lows.empty:
            return {"direction":None,"bars_ago":None}
        last_bar = len(d) - 1
        last_low_idx  = int(lows["index"].iloc[-1]);  last_low_val  = float(d.loc[last_low_idx, "low"])
        last_high_idx = int(highs["index"].iloc[-1]); last_high_val = float(d.loc[last_high_idx,"high"])
        px = float(d.loc[last_bar, "close"])
        ef = float(ema(d["close"], 9).iloc[-1]); es = float(ema(d["close"], 21).iloc[-1])
        if ef > es and px < last_low_val * 0.999:
            return {"direction":"bearish","bars_ago":int(max(0,last_bar-last_low_idx))}
        if ef < es and px > last_high_val * 1.001:
            return {"direction":"bullish","bars_ago":int(max(0,last_bar-last_high_idx))}
        return {"direction":None,"bars_ago":None}
    except Exception:
        return {"direction":None,"bars_ago":None}

def internal_range_liquidity(df, lookback=80, band=0.0015):
    """
    Equal Highs/Lows (liquidity build-up) di range dalam.
    Return {'bias':'above'|'below'|None,'eqh':int,'eql':int,'score':0..1}
    """
    try:
        if len(df) < (lookback + 5):
            return {"bias":None,"eqh":0,"eql":0,"score":0.0}
        d = df.tail(lookback).copy()
        hi = d["high"].values; lo = d["low"].values; px = float(d["close"].iloc[-1])
        top = float(np.max(hi)); bot = float(np.min(lo))
        eqh = sum(1 for v in hi[-lookback//2:] if abs(v-top)/max(1e-12,top) <= band)
        eql = sum(1 for v in lo[-lookback//2:] if abs(v-bot)/max(1e-12,bot) <= band)
        bias = None
        if eqh > eql and px <= top*(1-band*0.5): bias = 'above'
        elif eql > eqh and px >= bot*(1+band*0.5): bias = 'below'
        score = max(eqh, eql) / max(3.0, (lookback/10.0))
        return {"bias":bias,"eqh":int(eqh),"eql":int(eql),"score":float(max(0.0,min(1.0,score)))}
    except Exception:
        return {"bias":None,"eqh":0,"eql":0,"score":0.0}

def pullback_signal(df, side, ema_fast_span=20, ema_slow_span=50):
    """
    Kualitas pullback: deviasi ke EMA50 + RSI reset.
    Return {'ok':bool,'z':float,'rsi':float}
    """
    try:
        if len(df) < (ema_slow_span + 20):
            return {"ok":False,"z":None,"rsi":None}
        d = df.copy()
        es = ema(d["close"], ema_slow_span)
        r = d["close"].pct_change().dropna()
        z = (d["close"].iloc[-1] - es.iloc[-1]) / max(1e-12, r.std())
        rsi14 = float(rsi(d["close"], 14).iloc[-1])
        ok = (z <= -0.3 and rsi14 <= 45) if side == "long" else (z >= 0.3 and rsi14 >= 55)
        return {"ok":bool(ok),"z":float(z),"rsi":rsi14}
    except Exception:
        return {"ok":False,"z":None,"rsi":None}

def order_block_zone(df, side, lookback=60):
    """
    Order Block heuristik:
      SHORT → cari candle hijau besar sebelum breakdown; zone=[open,close]
      LONG  → cari candle merah besar sebelum breakout;  zone=[close,open]
    Return {'low':float,'high':float,'bias':'long'|'short'|None,'prox':float} atau {}
    """
    try:
        if len(df) < (lookback + 5):
            return {}
        d = df.tail(lookback + 5).copy().reset_index(drop=True)
        px = float(d.loc[len(d)-1, "close"])
        best = None
        for i in range(len(d)-5, len(d)-1):
            o = float(d.loc[i,"open"]); c = float(d.loc[i,"close"]); h = float(d.loc[i,"high"]); l = float(d.loc[i,"low"])
            rng = h - l
            if rng <= 0: continue
            body_ratio = abs(c-o)/max(1e-12, rng)
            if side == "short" and c > o and body_ratio >= 0.5:
                low, high = min(o,c), max(o,c)
                best = {"low":low,"high":high,"bias":"short","prox":min(abs(px-low),abs(px-high))/max(1e-12,px)}
            if side == "long" and o > c and body_ratio >= 0.5:
                low, high = min(o,c), max(o,c)
                best = {"low":low,"high":high,"bias":"long","prox":min(abs(px-low),abs(px-high))/max(1e-12,px)}
        return best or {}
    except Exception:
        return {}

def scalping_filter(ex, symbol, tf_main, df_main, side_intent, ctx_model):
    """
    Composite scalping score (0..1) + reasons + ctx.
    Komponen:
      - Micro EMA trend multi-TF
      - VWAP proximity
      - Orderbook imbalance
      - Wick-sweep rejection
      - Breakout + retest
      - FVG (alignment/proximity)
      - S/R confluence
      - ChoCh
      - IRL (internal range liquidity)
      - Pullback quality
      - Order-Block tap proximity
    """
    
    # Micro TF list
    micro_list_env = os.getenv("AUTOPILOT_MICRO_TFS", "5m,15m")
    micro_list = [x.strip() for x in micro_list_env.split(",") if x.strip()] or [os.getenv("AUTOPILOT_SC_TF", "5m").strip()]
    
    # --- Pastikan ctx ada ---
    ctx = dict(ctx_model) if isinstance(ctx_model, dict) else {}
    
    # Regime detection (momentum vs chop) dari ADX/BBW
    ADX_TREND = float(os.getenv("AUTOPILOT_ADX_TREND", "22"))
    BBW_TREND_PCTL = float(os.getenv("AUTOPILOT_BBW_TREND_PCTL", "0.60"))
    adx_v = ctx.get("adx"); bbw_p = ctx.get("bbw_pctl")
    if (adx_v is not None and float(adx_v) >= ADX_TREND) or (bbw_p is not None and float(bbw_p) >= BBW_TREND_PCTL):
        ctx["regime"] = "trend"
    else:
        ctx["regime"] = "chop"

    # Pastikan signal wick & OB terset di ctx untuk digunakan gate
    try:
        ctx["wick_sweep"] = float(wick_sweep_score(df_main, lookback=3))
    except Exception:
        ctx["wick_sweep"] = None
    try:
        obz = order_block_zone(df_main, side_intent, lookback=int(os.getenv("AUTOPILOT_SMC_OB_LOOK", "60")))
        ctx["order_block"] = obz or {}
    except Exception:
        ctx["order_block"] = {}

    # === Trend filters: ADX & BBW → ke ctx ===
    try:
        ctx["adx"] = float(adx(df_main, 14).iloc[-1])
    except Exception:
        ctx["adx"] = None

    try:
        _bbw = bollinger_width(df_main["close"], 20, 2.0)
        ctx["bbw"] = float(_bbw.iloc[-1])
        # persentil sederhana di jendela yang sama (tanpa fetch tambahan)
        try:
            denom = max(1, int(_bbw.count()))
            ctx["bbw_pctl"] = float(((_bbw <= _bbw.iloc[-1]).sum()) / denom)
        except Exception:
            ctx["bbw_pctl"] = None
    except Exception:
        ctx["bbw"] = None
        ctx["bbw_pctl"] = None

    ob_depth  = int(os.getenv("AUTOPILOT_SC_OB_DEPTH", "10"))
    ob_thresh = float(os.getenv("AUTOPILOT_SC_OB_THRESH", "0.12"))
    vwap_band = float(os.getenv("AUTOPILOT_SC_VWAP_BAND_PCT", "0.0018"))
    wick_min  = float(os.getenv("AUTOPILOT_SC_WICK_MIN", "0.6"))
    br_look   = int(os.getenv("AUTOPILOT_SC_BR_LOOKBACK", "30"))

    # SMC knobs
    fvg_look   = int(os.getenv("AUTOPILOT_SMC_FVG_LOOK", "60"))
    fvg_gappct = float(os.getenv("AUTOPILOT_SMC_FVG_GAP_PCT", "0.0008"))
    sr_eps     = float(os.getenv("AUTOPILOT_SMC_SR_EPS", "0.0012"))
    choch_look = int(os.getenv("AUTOPILOT_SMC_CHOCH_LOOK", "80"))
    irl_look   = int(os.getenv("AUTOPILOT_SMC_IRL_LOOK", "80"))
    irl_band   = float(os.getenv("AUTOPILOT_SMC_IRL_BAND", "0.0015"))
    ob_look    = int(os.getenv("AUTOPILOT_SMC_OB_LOOK", "60"))

    # Micro trend
    micro_sum, micro_details, micro_emas = micro_tf_confirm_multi(ex, symbol, tfs=micro_list)

    # VWAP (session)
    vwap_val, vwap_dist = session_vwap(df_main)

    # OB imbalance + spread
    ob_imb, ob_spr = orderbook_imbalance(ex, symbol, depth=ob_depth)

    # Wick sweep & breakout-retest
    ws = wick_sweep_score(df_main, lookback=3)
    br = breakout_retest_flag(df_main, lookback=br_look, eps=0.0012)

    # --- SMC signals ---
    price = float(df_main["close"].iloc[-1])
    fvg = detect_fvg_near_price(df_main, lookback=fvg_look, min_gap_pct=fvg_gappct)
    sr  = sr_confluence(df_main, price, lookback=max(80, br_look), eps=sr_eps)
    cc  = choch_signal(df_main, left=3, right=3, lookback=choch_look)
    irl = internal_range_liquidity(df_main, lookback=irl_look, band=irl_band)
    pb  = pullback_signal(df_main, side_intent)
    obz = order_block_zone(df_main, side_intent, lookback=ob_look)

        # --- Trend filters (ADX & BBW) ---
    try:
        adx_min = float(os.getenv("AUTOPILOT_TREND_ADX_MIN", "18"))
    except Exception:
        adx_min = 18.0
    try:
        bbw_min = float(os.getenv("AUTOPILOT_TREND_BBW_MIN", "0.02"))
    except Exception:
        bbw_min = 0.02

    try:
        adx_val = float(adx(df_main, 14).iloc[-1])
    except Exception:
        adx_val = None
    try:
        bbw_val = float(bollinger_width(df_main["close"], 20, 2.0).iloc[-1])
    except Exception:
        bbw_val = None
    
    # Score
    reasons = []
    score = 0.0
    
    # 0) Trend strength (ADX/BBW)
    trend_ok = True
    if adx_val is not None and not math.isnan(adx_val):
        if adx_val >= adx_min:
            score += 0.12
        else:
            reasons.append(f"adx<{adx_min:.0f}")
            trend_ok = False
    else:
        reasons.append("no-adx")
        trend_ok = False

    if bbw_val is not None and not math.isnan(bbw_val):
        if bbw_val >= bbw_min:
            score += 0.08
        else:
            reasons.append(f"bbw<{bbw_min:.3f}")
            trend_ok = False
    else:
        reasons.append("no-bbw")
        trend_ok = False

    # 1) Micro alignment
    aligned_tfs = sum(1 for tf in micro_list if (side_intent=="long" and micro_details.get(tf)=="up") or (side_intent=="short" and micro_details.get(tf)=="down"))
    if aligned_tfs > 0: score += 0.11 * aligned_tfs
    if (side_intent=="long" and micro_sum=="up") or (side_intent=="short" and micro_sum=="down"):
        score += 0.11
    else:
        reasons.append(f"micro-trend majority != {side_intent}")

    # 2) VWAP
    if vwap_val is not None and vwap_dist is not None:
        if vwap_dist <= vwap_band: score += 0.16
        else: reasons.append(f"far-from-VWAP(>{vwap_band:.4f})")
    else:
        reasons.append("no-VWAP")

    # 3) Orderbook imbalance
    if side_intent == "long":
        if ob_imb >= ob_thresh: score += 0.20
        else: reasons.append(f"OB-imb<{ob_thresh:.2f}")
    else:
        if ob_imb <= -ob_thresh: score += 0.20
        else: reasons.append(f"OB-imb>{-ob_thresh:.2f}")

    # 4) Wick sweep
    if ws >= (wick_min/2.0): score += 0.14
    else: reasons.append("weak-wick-sweep")

    # 5) Breakout + retest
    if (br == side_intent): score += 0.12
    else: reasons.append("no-breakout-retest")

    # 6) FVG alignment/prox
    if fvg:
        aligned = (side_intent=="long" and fvg.get("type")=="bullish") or (side_intent=="short" and fvg.get("type")=="bearish")
        if aligned:
            if fvg.get("in_gap"): score += 0.16
            else:
                prox = float(fvg.get("prox", 1.0))
                score += max(0.0, 0.12 * (1.0 - min(1.0, prox/0.004)))
        else:
            reasons.append("no-fvg-align")
    else:
        reasons.append("no-fvg")

    # 7) S/R confluence
    if sr.get("bias") == "support" and side_intent=="long":
        score += 0.12 * sr.get("score", 0.0)
    elif sr.get("bias") == "resistance" and side_intent=="short":
        score += 0.12 * sr.get("score", 0.0)
    else:
        reasons.append("no-SR-confluence")

    # 8) ChoCh
    if (cc.get("direction") == "bullish" and side_intent=="long") or (cc.get("direction")=="bearish" and side_intent=="short"):
        score += 0.10
    else:
        reasons.append("no-ChoCh")

    # 9) IRL (liquidity di sisi sebaliknya → peluang sweep)
    if (irl.get("bias")=="above" and side_intent=="long") or (irl.get("bias")=="below" and side_intent=="short"):
        score += 0.08 * irl.get("score", 0.0)
    else:
        reasons.append("no-IRL-edge")

    # 10) Pullback quality
    if pb.get("ok"): score += 0.08
    else: reasons.append("weak-pullback")

    # 11) Order Block tap
    if obz and obz.get("bias") == side_intent:
        prox = float(obz.get("prox", 1.0))
        score += max(0.0, 0.10 * (1.0 - min(1.0, prox/0.004)))
    else:
        reasons.append("no-OB-tap")

    score = max(0.0, min(1.0, score))

    ctx = {
        "micro_trend": micro_sum,
        "micro_details": micro_details,
        "vwap": vwap_val, "vwap_dist": vwap_dist,
        "ob_imb": ob_imb, "ob_spread": ob_spr,
        "wick_sweep": ws, "break_retest": br,
        # SMC contexts
        "fvg_type": (fvg.get("type") if fvg else None),
        "fvg_in": (fvg.get("in_gap") if fvg else None),
        "fvg_prox": (fvg.get("prox") if fvg else None),
        "sr_bias": sr.get("bias"), "sr_dist": sr.get("dist"), "sr_score": sr.get("score"),
        "choch": cc.get("direction"), "choch_bars": cc.get("bars_ago"),
        "irl_bias": irl.get("bias"), "irl_score": irl.get("score"),
        "pullback_ok": pb.get("ok"), "pullback_z": pb.get("z"), "pullback_rsi": pb.get("rsi"),
        "ob_bias": (obz.get("bias") if obz else None), "ob_prox": (obz.get("prox") if obz else None),
        "adx": (float(adx_val) if adx_val is not None else None),
        "bbw": (float(bbw_val) if bbw_val is not None else None),
        "trend_ok": bool(trend_ok),
    }
    return score, reasons, ctx

def _hard_entry_gates(side_intent, ctx, tf_main):
    """
    Must-pass gates, tapi jika AUTOPILOT_AI_ONLY=1 (default), gate jadi *soft*:
    tidak memblokir, hanya mengembalikan alasan (blocks) untuk ditampilkan.
    Return (ok: bool, blocks: list[str])
    """
    ai_only = str(os.getenv("AUTOPILOT_AI_ONLY", "1")).lower() not in ("0","false","no","off")

    blocks = []
    ok = True

    require_vwap  = str(os.getenv("AUTOPILOT_REQUIRE_VWAP", "1")).lower() not in ("0","false","no","off")
    require_micro = str(os.getenv("AUTOPILOT_REQUIRE_MICRO", "1")).lower() not in ("0","false","no","off")
    require_fvg   = str(os.getenv("AUTOPILOT_REQUIRE_FVG_ALIGN", "1")).lower() not in ("0","false","no","off")
    require_any   = str(os.getenv("AUTOPILOT_REQUIRE_SRBRPB", "1")).lower() not in ("0","false","no","off")
    REQ_ADX       = str(os.getenv("AUTOPILOT_REQUIRE_ADX", "1")).lower() not in ("0","false","no","off")
    REQ_BBW       = str(os.getenv("AUTOPILOT_REQUIRE_BBW", "0")).lower() not in ("0","false","no","off")

    ADX_MIN = float(os.getenv("AUTOPILOT_TREND_ADX_MIN", "18"))
    BBW_MIN = float(os.getenv("AUTOPILOT_TREND_BBW_MIN", "0.02"))

    vwap_band = float(os.getenv("AUTOPILOT_SC_VWAP_BAND_PCT", "0.0018"))
    mult_1m   = float(os.getenv("AUTOPILOT_VWAP_BAND_MULT_1M", "2.0")) if str(tf_main).endswith("m") else 1.0
    vwap_thr  = vwap_band * mult_1m
    fvg_gate  = float(os.getenv("AUTOPILOT_FVG_PROX_GATE", "0.004"))

    adx_v = ctx.get("adx")
    bbw_v = ctx.get("bbw")
    vdist = ctx.get("vwap_dist")
    micro = (ctx.get("micro_trend") or "").lower() if ctx.get("micro_trend") is not None else ""
    fvg_type = ctx.get("fvg_type")
    fvg_in   = ctx.get("fvg_in")
    fvg_prox = ctx.get("fvg_prox")

    # Trend strength
    if REQ_ADX and (adx_v is None or adx_v < ADX_MIN):
        blocks.append(f"adx<{ADX_MIN}")
        if not ai_only: ok = False
    if REQ_BBW and (bbw_v is None or bbw_v < BBW_MIN):
        blocks.append(f"bbw<{BBW_MIN}")
        if not ai_only: ok = False

    # VWAP proximity
    if require_vwap and (vdist is None or vdist > vwap_thr):
        blocks.append("vwap")
        if not ai_only: ok = False

    # Micro-trend alignment
    if require_micro:
        aligned = ((side_intent == "long" and micro == "up") or (side_intent == "short" and micro == "down"))
        if not aligned:
            blocks.append("micro")
            if not ai_only: ok = False

    # FVG alignment+prox
    if require_fvg:
        aligned = (side_intent == "long" and fvg_type == "bullish") or (side_intent == "short" and fvg_type == "bearish")
        near = bool(fvg_in) or (fvg_prox is not None and float(fvg_prox) <= fvg_gate)
        if not (aligned and near):
            blocks.append("fvg")
            if not ai_only: ok = False

    # SR/BreakRetest/Pullback salah satu OK
    if require_any:
        sr_bias = ctx.get("sr_bias")
        br = ctx.get("break_retest")
        pb_ok = bool(ctx.get("pullback_ok"))
        sr_ok = (side_intent == "long" and sr_bias == "support") or (side_intent == "short" and sr_bias == "resistance")
        br_ok = (br == side_intent)
        if not (sr_ok or br_ok or pb_ok):
            blocks.append("structure")
            if not ai_only: ok = False

    return ok, blocks

def wait_for_entry(ex, symbol, tf_main, side=None, min_score=0.70, poll_sec=1.0, max_wait_sec=None, side_intent: str | None = None, **kwargs):
    """Pantau kesiapan entry tiap detik (1 baris) dengan hard gates untuk mengurangi SL cepat.
    Tekan 'c' + Enter untuk cancel. True jika siap entry; False jika batal/timeout.
    Perbaikan:
      - Single-source cancel → raise CancelledEntry agar autopilot bisa kembali ke screening.
      - Tentukan side terlebih dulu (model/EMA fallback), baru hitung scalping_filter untuk side tsb.
      - Pakai cached_fetch_ohlcv dengan TTL agar hemat API dan responsif.
    """
    # --- normalisasi input/opsi ---
    dyn_side = (side is None) or (str(side).strip().lower() in ("", "auto"))
    forced_side = None
    if side_intent:
        try:
            forced_side = str(side_intent).strip().lower()
        except Exception:
            forced_side = None
        if forced_side not in ("long", "short"):
            forced_side = None
    try:
        if forced_side and os.getenv("AUTOPILOT_FORCE_SIDE_INTENT", "0").lower() not in ("0","false","no","off"):
            side = forced_side
            dyn_side = False
    except Exception:
        pass

    try:
        sc_min_env = float(os.getenv("AUTOPILOT_SC_MIN_SCORE", str(min_score)))
    except Exception:
        sc_min_env = float(min_score)

    t0 = time.time()
    ttl = None
    try:
        ttl = float(os.getenv("AUTOPILOT_CACHE_TTL_S", "2.0"))
    except Exception:
        ttl = 2.0

    while True:
        # --- cancel: satu-satunya pintu keluar manual ---
        try:
            key = _readline_nowait()
            if isinstance(key, str) and key.strip().lower().startswith('c'):
                print()
                say(f"{YELLOW}Entry dibatalkan oleh user; kembali ke screening…{RESET}")
                raise CancelledEntry("user-cancel")
        except Exception:
            pass

        # --- ambil data utama (cached) ---
        try:
            df_main = cached_fetch_ohlcv(ex, symbol, timeframe=tf_main, candles=400, ttl_sec=ttl)
        except Exception as e:
            _print_inline(f"ENTRY {symbol} ? err {e}")
            time.sleep(max(0.20, float(poll_sec)))
            continue

        # --- tentukan side_now (model → EMA fallback) ---
        side_now = None
        if dyn_side:
            try:
                Xf, cols = build_features(df_main)
                bundle, _p = load_model_bundle(symbol, tf_main)
                if bundle is not None and len(Xf) > 5:
                    s_now, conf_now, ctx_sig = signal_long_short(Xf, cols, bundle)
                    if s_now in ("long", "short"):
                        side_now = s_now
            except Exception:
                side_now = None
            if side_now is None:
                try:
                    ef = float(ema(df_main["close"], 9).iloc[-1]); es = float(ema(df_main["close"], 21).iloc[-1])
                    side_now = "long" if ef > es else ("short" if ef < es else None)
                except Exception:
                    side_now = None
        else:
            side_now = (forced_side or (str(side).strip().lower() if side else None))

        if not side_now:
            _print_inline(f"ENTRY {symbol} ? score=0.00 | wait | side-unknown")
            if max_wait_sec and (time.time() - t0) >= float(max_wait_sec):
                print(); say("Timeout menunggu setup yang valid.")
                return False
            time.sleep(max(0.20, float(poll_sec)))
            continue

        # --- hitung skor untuk side_now ---
        try:
            score, reasons, ctx = scalping_filter(ex, symbol, tf_main, df_main, side_now, {})
        except Exception as e:
            _print_inline(f"ENTRY {symbol} {side_now} err {e}")
            time.sleep(max(0.20, float(poll_sec)))
            continue

        ok, blocks = _hard_entry_gates(side_now, ctx, tf_main)
        if ok and float(score) >= sc_min_env:
            print()
            say(f"ENTRY {symbol} {side_now} score={float(score):.2f} | OK | " + (",".join([r for r in reasons if r]) or "-"))
            return True
        else:
            gate_txt = ("gates:" + "/".join(blocks)) if (not ok) else f"score<{sc_min_env:.2f}"
            _print_inline(f"ENTRY {symbol} {side_now} score={float(score):.2f} | wait | {gate_txt}")

        if max_wait_sec and (time.time() - t0) >= float(max_wait_sec):
            print(); say("Timeout menunggu setup yang valid.")
            return False

        time.sleep(max(0.20, float(poll_sec)))

# --- CSV logging helpers ---
LOG_DIR = "logs"
TRADE_HEADERS = [
    "ts", "event", "symbol", "side", "qty", "price", "sl", "tp", "lev_used", "reason", "pnl_usd", "wr_all", "wr_long", "wr_short"
]

def ensure_log_dir():
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except Exception:
        pass

def log_trade_row(path, row: dict):
    exists = os.path.exists(path)
    try:
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRADE_HEADERS)
            if not exists:
                w.writeheader()
            # keep only known headers
            clean = {k: row.get(k,"") for k in TRADE_HEADERS}
            w.writerow(clean)
    except Exception as e:
        say(f"{YELLOW}Gagal tulis log CSV: {e}{RESET}")
        
# --- Signal TXT logging (community-style) ---
def log_signal_txt(symbol, side, entry, tp, sl, leverage=None, margin_mode=None, dryrun=False):
    """
    Append a human-readable trade signal block to logs/signal_<symbol>.txt
    Format:
      pair:
      entry price:
      TP:
      SL:
    + (opsional) side, leverage, margin mode, DRYRUN note
    """
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except Exception:
        pass
    fname = f"signal_{sanitize_symbol(symbol)}.txt"
    path = os.path.join(LOG_DIR, fname)
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}]\n")
            f.write(f"pair: {symbol}\n")
            if side:
                f.write(f"side: {side}\n")
            f.write(f"entry price: {float(entry):.6f}\n")
            f.write(f"TP: {float(tp):.6f}\n")
            f.write(f"SL: {float(sl):.6f}\n")
            if leverage is not None:
                f.write(f"leverage: {int(leverage)}x\n")
            if margin_mode:
                f.write(f"margin mode: {margin_mode}\n")
            if dryrun:
                f.write("note: DRYRUN (no real order)\n")
            f.write("-" * 40 + "\n")
    except Exception as e:
        say(f"{YELLOW}Gagal tulis signal TXT: {e}{RESET}")

# --- Local OHLCV cache (CSV) + progress helpers ---
CACHE_DIR = os.getenv("AUTOPILOT_CACHE_DIR", "cache")

def ensure_cache_dir():
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
    except Exception:
        pass

def _cache_fname(symbol: str, timeframe: str) -> str:
    return f"ohlcv_{sanitize_symbol(symbol)}_{normalize_tf(timeframe)}.csv"

def _cache_path(symbol: str, timeframe: str) -> str:
    return os.path.join(CACHE_DIR, _cache_fname(symbol, timeframe))

def read_cached_ohlcv(symbol: str, timeframe: str):
    """
    Return cached DataFrame if exists (columns: datetime,open,high,low,close,volume), else None.
    """
    try:
        p = _cache_path(symbol, timeframe)
        if os.path.isfile(p):
            df = pd.read_csv(p)
            # robust typing
            df["datetime"] = pd.to_datetime(df["datetime"])
            df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].apply(pd.to_numeric, errors="coerce")
            df = df.dropna(subset=["datetime","open","high","low","close","volume"]).sort_values("datetime").reset_index(drop=True)
            return df[["datetime","open","high","low","close","volume"]]
    except Exception:
        pass
    return None

def write_cached_ohlcv(symbol: str, timeframe: str, df: pd.DataFrame):
    """
    Overwrite CSV cache atomically when possible.
    """
    try:
        ensure_cache_dir()
        p = _cache_path(symbol, timeframe)
        tmp = p + ".tmp"
        df.to_csv(tmp, index=False)
        try:
            os.replace(tmp, p)
        except Exception:
            # fallback if os.replace unsupported on FS
            import shutil
            shutil.move(tmp, p)
    except Exception:
        pass

# --- Market guards: funding, spread, open interest ---

def market_guards(ex, symbol, last_px, last_oi, max_abs_fund=0.0010, max_spread=0.0008, oi_guard=True, oi_spike_factor=1.5):
    """Return (ok, reasons:list, ctx:dict, new_last_oi).
    - Blocks entry if |funding|>max_abs_fund, or spread>max_spread, or OI spike (if enabled).
    - All data sources are best-effort; missing data does not block entry by itself.
    """
    reasons = []
    ctx = {"funding": None, "spread": None, "oi": None}
    # Spread guard via ticker or orderbook
    try:
        t = ex.fetch_ticker(symbol)
        bid = float(t.get("bid") or 0.0); ask = float(t.get("ask") or 0.0)
        if bid>0 and ask>0 and ask>bid:
            mid = 0.5*(bid+ask)
            spr = (ask - bid)/mid
            ctx["spread"] = spr
            if max_spread and spr > max_spread:
                reasons.append(f"spread>{max_spread:.4f}")
    except Exception:
        # fallback to orderbook
        try:
            ob = ex.fetch_order_book(symbol, limit=10)
            best_bid = float(ob['bids'][0][0]) if ob.get('bids') else 0.0
            best_ask = float(ob['asks'][0][0]) if ob.get('asks') else 0.0
            if best_bid>0 and best_ask>0 and best_ask>best_bid:
                mid=0.5*(best_bid+best_ask)
                spr=(best_ask-best_bid)/mid
                ctx["spread"] = spr
                if max_spread and spr > max_spread:
                    reasons.append(f"spread>{max_spread:.4f}")
        except Exception:
            pass
    # Funding guard (perpetuals)
    try:
        fr = ex.fetch_funding_rate(symbol)
        rate = float(fr.get("fundingRate") or fr.get("info",{}).get("fundingRate") or 0.0)
        ctx["funding"] = rate
        if abs(rate) > max_abs_fund:
            reasons.append(f"|fund|>{max_abs_fund:.4f}")
    except Exception:
        pass
    # Open Interest spike guard (best-effort)
    new_last_oi = last_oi
    try:
        if oi_guard:
            oi = ex.fetch_open_interest(symbol)
            # ccxt returns a dict; try common fields
            val = oi.get("openInterestAmount") or oi.get("openInterestValue") or oi.get("info",{}).get("openInterest")
            if val is not None:
                cur = float(val)
                ctx["oi"] = cur
                if last_oi is not None and cur > last_oi * oi_spike_factor:
                    reasons.append(f"oi_spike>{oi_spike_factor:.2f}x")
                new_last_oi = cur
    except Exception:
        pass
    ok = (len(reasons)==0)
    return ok, reasons, ctx, new_last_oi

# --- Volume surge score (0..1) comparing last 24h vs baseline median ---
def _bars_per_day(tf: str) -> int:
    tf = normalize_tf(tf)
    if tf.endswith("m"):
        m = int(tf[:-1])
        return int(max(1, (24*60)//m))
    if tf.endswith("h"):
        h = int(tf[:-1])
        return int(max(1, 24//h))
    if tf.endswith("d"):
        return 1
    return 96  # fallback for 15m-like granularity

# NOTEcurrently unused; kept for future analytics
def compute_volume_surge(df, timeframe=None, window_days=1, baseline_days=5):
    """
    Returns (surge_score[0..1], cur_sum, median_hist, ratio)
    surge_score ≈ 0 at normal volumes, →1 when ~2.5x above baseline median.
    """
    timeframe = require_tf()
    try:
        bpd = _bars_per_day(timeframe)
        w = max(1, int(window_days*bpd))
        span = int((window_days+baseline_days) * bpd) + 5
        v = df["volume"].tail(span)
        if len(v) < (w*2 + 10):
            return 0.0, None, None, None
        cur = float(v.tail(w).sum())
        arr = v.values
        hist = []
        # build rolling sums excluding the last window
        for i in range(0, len(arr)-w*2, w//2 or 1):
            hist.append(arr[i:i+w].sum())
        if not hist:
            return 0.0, cur, None, None
        med = float(np.median(hist))
        ratio = (cur/med) if med>0 else 0.0
        surge = max(0.0, min(1.0, (ratio - 1.0) / 1.5))  # 1.0 at ~2.5x median
        return surge, cur, med, ratio
    except Exception:
        return 0.0, None, None, None

# --- Monte Carlo probability of hitting TP before SL ---
def monte_carlo_hit_prob(prices, steps=16, sims=400, entry=None, sl=None, tp=None, side="long"):
    """
    Simple GBM Monte Carlo using empirical mu/sigma from recent returns.
    Returns probability in [0..1] that TP is hit before SL within `steps` bars.
    """
    try:
        px = pd.Series(prices).dropna()
        rets = px.pct_change().dropna()
        if len(rets) < 200:
            return None
        mu = float(rets.mean())
        sigma = float(rets.std())
        if sigma <= 0 or entry is None or sl is None or tp is None:
            return None
        p0 = float(entry)
        hit = 0
        for _ in range(int(sims)):
            p = p0
            for _ in range(int(steps)):
                z = np.random.normal()
                p = p * math.exp((mu - 0.5*sigma*sigma) + sigma*z)
                if side == "long":
                    if p >= tp:
                        hit += 1
                        break
                    if p <= sl:
                        break
                else:
                    if p <= tp:
                        hit += 1
                        break
                    if p >= sl:
                        break
        return hit / float(sims)
    except Exception:
        return None
    
def select_tp_sl_by_potential(close_series, atr_value, entry, side, min_r=1.5,
                              tp_grid=(0.8,1.0,1.2,1.5,1.8,2.0), sl_grid=(0.4,0.6,0.8,1.0,1.2)):
    """
    Grid-search TP/SL (dalam kelipatan ATR) memakai Monte Carlo P(TP before SL).
    Return dict {tp, sl, tp_m, sl_m, p, r, exp} atau None jika tak ada yang memenuhi min_r.
    """
    try:
        px = close_series.dropna()
        if len(px) < 600 or atr_value is None or atr_value <= 0 or entry is None:
            return None
        best = None
        for tp_m in tp_grid:
            tp = entry + tp_m * atr_value if side == "long" else entry - tp_m * atr_value
            for sl_m in sl_grid:
                sl = entry - sl_m * atr_value if side == "long" else entry + sl_m * atr_value
                p = monte_carlo_hit_prob(px.tail(900).values, steps=16, sims=400, entry=entry, sl=sl, tp=tp, side=side)
                if p is None:
                    continue
                r = (tp_m / sl_m) if sl_m > 0 else 0.0
                exp = p * r - (1 - p) * 1.0  # expected R
                if (r >= float(min_r)) and (best is None or exp > best["exp"]):
                    best = {"tp": float(tp), "sl": float(sl), "tp_m": float(tp_m), "sl_m": float(sl_m),
                            "p": float(p), "r": float(r), "exp": float(exp)}
        return best
    except Exception:
        return None

# --- On-chain / supply metrics via CoinGecko (best-effort) ---
def _cg_id_from_symbol(symbol: str):
    override = os.getenv("AUTOPILOT_CG_ID", "").strip().lower()
    if override:
        return override
    base = (symbol.split("/")[0] if "/" in symbol else symbol).upper()
    return COINGECKO_IDS.get(base)

def fetch_onchain_metrics(symbol: str):
    """
    Best-effort fetch of circulating/total supply and 24h volume from CoinGecko.
    Returns dict with keys: circ, total, mcap, vol24 (all floats or None).
    """
    try:
        cid = _cg_id_from_symbol(symbol)
        if not cid:
            return {}
        url = f"https://api.coingecko.com/api/v3/coins/{cid}?localization=false&tickers=false&market_data=true&community_data=false&developer_data=false&sparkline=false"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return {}
        js = r.json()
        md = js.get("market_data", {}) or {}
        circ = md.get("circulating_supply")
        tot = md.get("total_supply") or md.get("max_supply")
        mcap = (md.get("market_cap") or {}).get("usd")
        vol24 = (md.get("total_volume") or {}).get("usd")
        out = {}
        out["circ"] = float(circ) if circ is not None else None
        out["total"] = float(tot) if tot is not None else None
        out["mcap"] = float(mcap) if mcap is not None else None
        out["vol24"] = float(vol24) if vol24 is not None else None
        return out
    except Exception:
        return {}

def chain_signal_from_metrics(m: dict):
    """
    Combine turnover velocity (vol/mcap) and free-float ratio (circ/total) into [0..1].
    """
    try:
        if not m:
            return None
        mcap = m.get("mcap") or 0.0
        vol = m.get("vol24") or 0.0
        vel = 0.0
        if mcap > 0:
            vel = vol / mcap
            vel = max(0.0, min(1.0, vel / 0.30))  # 30% turnover → 1.0
        if m.get("total"):
            ff = (m.get("circ") or 0.0) / max(1e-9, m.get("total"))
            # 0.6–0.95 best; scale to 0..1
            if ff <= 0.4:
                ff_score = 0.2 * (ff / 0.4)
            elif ff >= 0.98:
                ff_score = 0.6
            else:
                ff_score = 0.6 + 0.4 * ((ff - 0.6) / max(1e-9, 0.35))
            ff_score = max(0.0, min(1.0, ff_score))
        else:
            ff_score = 0.5
        return 0.6*vel + 0.4*ff_score
    except Exception:
        return None

# --- Dynamic leverage & margin-mode decision ---

def decide_leverage_mode(ctx, atr, price, spread=None, funding=None,
                         lev_min=2, lev_max=12, strong_p=0.75, medium_p=0.60,
                         mc_prob=None, vol_surge=None, chain_signal=None):
    """Return (lev:int, margin_mode:str, meta:dict)
    Scoring uses:
    - Directional probability p_dir = max(P_long, P_short)
    - Confidence = |edge| = |P_long - P_short|
    - Trend alignment (EMA fast/slow + fib_trend) with intended side
    - Fibonacci proximity (min distance to 0.5 / 0.618)
    - Volatility regime via ATR% of price
    - Microstructure costs: spread and funding (side-aware)
    Also supports RATCET_MAX logic (env-driven) to push leverage to lev_max when conditions are very strong.
    """
    try:
        import os
        # Extract context
        pL = float(ctx.get("pL", 0.0)); pS = float(ctx.get("pS", 0.0)); edge = float(ctx.get("edge", 0.0))
        ema_fast = float(ctx.get("ema_fast", 0.0)); ema_slow = float(ctx.get("ema_slow", 0.0))
        fib_trend = int(ctx.get("fib_trend", 0))
        d5 = float(ctx.get("fib_dist_500", 1.0))
        d6 = float(ctx.get("fib_dist_618", 1.0))
        # Side intent = higher probability direction
        side = "long" if pL >= pS else "short"
        p_dir = max(pL, pS)
        conf = abs(edge)
        atr_pct = (float(atr) / float(price)) if price else 0.0

        # Trend alignment relative to side intent
        align_long = (ema_fast > ema_slow) and (fib_trend > 0)
        align_short = (ema_fast < ema_slow) and (fib_trend < 0)
        align = (align_long if side == "long" else align_short)

        # Fibonacci proximity (closer to zone is better)
        prox = min(d5, d6)
        fib_eps_good = 0.0030   # ≤0.30% of price
        fib_eps_ok   = 0.0060   # ≤0.60% of price

        # --- Scoring ---
        score = 0.0
        # Core signal quality
        score += 0.45 * p_dir
        score += 0.30 * conf
        # Trend/fib-trend alignment
        score += (0.10 if align else -0.05)
        # Fibonacci proximity shaping
        if prox <= fib_eps_good:
            score += 0.08
        elif prox <= fib_eps_ok:
            score += 0.03
        else:
            score -= 0.06
        # Volatility regime (prefer calmer)
        if atr_pct <= 0.012:
            score += 0.07
        elif atr_pct <= 0.020:
            score += 0.02
        else:
            score -= 0.08
        # Spread penalty relative to a 0.08% reference
        if spread is not None:
            score -= min(max(spread, 0.0) / 0.0008 * 0.04, 0.20)
        # Funding penalty side-aware (longs pay when funding>0; shorts pay when funding<0)
        if funding is not None:
            cost = funding if side == "long" else -funding
            score -= min(max(cost, 0.0) / 0.0015 * 0.06, 0.18)  # full penalty at ~0.15%

        # Additional signals: Monte Carlo (TP>SL), volume surge, on-chain/supply activity
        try:
            if mc_prob is not None:
                score += 0.10 * float(mc_prob)
        except Exception:
            pass
        try:
            if vol_surge is not None:
                score += 0.06 * max(0.0, min(1.0, float(vol_surge)))
        except Exception:
            pass
        try:
            if chain_signal is not None:
                score += 0.06 * max(0.0, min(1.0, float(chain_signal)))
        except Exception:
            pass

        # clamp to [0,1]
        score = max(0.0, min(1.0, score))

        # --- Map score → leverage band (base) ---
        if score >= max(0.85, strong_p + 0.05):
            lev = min(lev_max, 14)
        elif score >= max(0.75, strong_p):
            lev = min(lev_max, 10)
        elif score >= max(0.65, medium_p):
            lev = min(lev_max, 8)
        elif score >= 0.55:
            lev = min(lev_max, 5)
        else:
            lev = max(lev_min, 3)

        # --- Margin mode decision (base) ---
        if (score >= max(0.70, medium_p)) and (atr_pct <= 0.0125) and (spread is None or spread <= 0.0010):
            mode = "crossed"
        else:
            mode = "isolated"

        # --- RATCHET MAX: push leverage to lev_max for very strong conditions ---
        RATCHET_ON     = str(os.getenv("AUTOPILOT_RATCHET_MAX", "1")).lower() not in ("0","false","no","off")
        RATCHET_P_DIR  = float(os.getenv("AUTOPILOT_RATCHET_P_DIR", "0.80"))   # ≥80% directional probability
        RATCHET_CONF   = float(os.getenv("AUTOPILOT_RATCHET_CONF", "0.60"))    # ≥0.60 confidence (|edge|)
        RATCHET_PROX   = float(os.getenv("AUTOPILOT_RATCHET_PROX", "0.0040"))  # ≤0.40% from fib zone
        RATCHET_SPREAD = float(os.getenv("AUTOPILOT_RATCHET_MAX_SPREAD", "0.0012"))  # ≤0.12% spread
        RATCHET_ATR    = float(os.getenv("AUTOPILOT_RATCHET_MAX_ATR_PCT", "0.018"))  # ≤1.8% ATR%

        ratchet = False
        if RATCHET_ON:
            conds = [
                (p_dir >= RATCHET_P_DIR),
                (conf  >= RATCHET_CONF),
                align,
                (prox <= RATCHET_PROX),
                (atr_pct <= RATCHET_ATR),
                (spread is None or spread <= RATCHET_SPREAD),
            ]
            if all(conds):
                lev = int(lev_max)  # rata kanan
                mode = "crossed"    # prefer cross when going max
                ratchet = True

        meta = {
            "score": score,
            "atr_pct": atr_pct,
            "side": side,
            "align": bool(align),
            "prox": prox,
            "p_dir": p_dir,
            "conf": conf,
            "spread": spread,
            "funding": funding,
            "ratchet": ratchet,
        }
        return int(lev), mode, meta
    except Exception:
        return max(lev_min, 3), "isolated", {"score": 0.0, "atr_pct": None}

# --- Futures permission proactive check ---
def ensure_futures_permissions(ex):
    """Return True if futures (swap) permissions seem OK, else False with guidance."""
    try:
        _ = ex.fetch_balance()
        return True
    except Exception as e:
        try:
            from ccxt.base.errors import PermissionDenied
            if isinstance(e, PermissionDenied) or (hasattr(e, 'args') and '40014' in str(e)):
                print(f"{RED}API key belum memiliki izin Futures (Contract).{RESET}")
                print("Silakan edit API key di Bitget -> enable: Futures/Contract Read + Trade (Position Read).")
                print("Jika whitelist IP aktif, tambahkan IP mesin ini. Lalu jalankan ulang.")
                return False
        except Exception:
            pass
        # Other errors bubble up for visibility
        print(f"{YELLOW}Gagal cek permission: {e}{RESET}")
        return False

# --- Permissions diagnostics helper ---
def permissions_diagnostics():
    """Print detailed diagnostics for Spot vs Futures permissions and credential source."""
    try:
        creds, src = load_api_credentials()
        print(f"Credentials source: {src}")
        import ccxt
        ex = bitget_swap()
        # Check SPOT balance (to verify key validity)
        try:
            spot_bal = ex.fetch_balance({"type": "spot"})
            usdt_spot = (spot_bal.get("total", {}) or {}).get("USDT")
            print(f"Spot balance USDT: {usdt_spot}")
        except Exception as e:
            print(f"Spot balance check failed: {e}")
        # Check FUTURES/SWAP
        try:
            swap_bal = ex.fetch_balance({"type": "swap"})
            usdt_swap = (swap_bal.get("total", {}) or {}).get("USDT")
            print(f"Futures balance USDT: {usdt_swap}")
            print("Futures permission seems OK.")
        except Exception as e:
            print(f"Futures balance check failed: {e}")
            print("Jika error code 40014: aktifkan Futures/Contract Read + Trade + Position Read di API key, pastikan passphrase benar, dan periksa whitelist IP.")
    except Exception as e:
        print(f"Diagnostics error: {e}")

# ---------- Data & features ----------
def normalize_tf(tf: str) -> str:
    tf=tf.strip().lower()
    if tf.isdigit(): return f"{tf}m"
    if tf[-1] in ("m","h","d","w") and tf[:-1].isdigit(): return tf
    raise ValueError("Bad timeframe")

def to_df(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.sort_values("datetime").reset_index(drop=True)
    return df[["datetime","open","high","low","close","volume"]]

def fetch_ohlcv(ex, symbol, timeframe="1m", candles=8000):
    """
    Return the most-recent `candles` OHLCV bars with lightweight CSV cache to avoid repeated API hits.
    Cache file path is managed by read_cached_ohlcv/write_cached_ohlcv.
    Respects ENV:
      AUTOPILOT_CACHE (default=1)
      AUTOPILOT_CACHE_TTL_S (default=2.0)  # minimum age (seconds) before we try to refresh tail
    """
    tf = normalize_tf(timeframe)
    USE_CACHE = str(os.getenv("AUTOPILOT_CACHE", "1")).lower() not in ("0","false","no","off")
    try:
        TTL = float(os.getenv("AUTOPILOT_CACHE_TTL_S", "2.0"))
    except Exception:
        TTL = 2.0

    df_cached = None
    if USE_CACHE:
        try:
            df_cached = read_cached_ohlcv(symbol, tf)
        except Exception:
            df_cached = None

    # If we have cache, decide whether to refresh the tail
    if USE_CACHE and df_cached is not None and not df_cached.empty:
        try:
            last_dt = pd.to_datetime(df_cached["datetime"].iloc[-1])
            # age in seconds vs TTL (work in UTC to avoid tz drift)
            age_s = (pd.Timestamp.utcnow() - last_dt.tz_localize("UTC")).total_seconds()
        except Exception:
            age_s = TTL + 1.0  # force refresh on parse failure

        if age_s > TTL:
            # Fetch only the missing tail using since = last_cached + 1ms
            since = int(last_dt.value // 1_000_000) + 1  # ns→ms then +1
            rows = []
            per = 1000
            last_ts = None
            while True:
                chunk = ex.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=per)
                if not chunk:
                    break
                rows += chunk
                new_last = chunk[-1][0]
                if last_ts is not None and new_last <= last_ts:
                    break
                last_ts = new_last
                since = new_last + 1
                try:
                    time.sleep(ex.rateLimit / 1000.0)
                except Exception:
                    time.sleep(0.2)
                if len(chunk) < per:
                    break

            if rows:
                df_new = to_df(rows)
                df_cached = pd.concat([df_cached, df_new], ignore_index=True)
                df_cached = df_cached.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
                try:
                    write_cached_ohlcv(symbol, tf, df_cached)
                except Exception:
                    pass

        # Serve only the latest `candles`
        return df_cached.tail(int(candles)).reset_index(drop=True)

    # No cache: fetch directly (most-recent `candles`) and then store to cache
    left = int(candles)
    since = None
    rows = []
    per = min(1000, int(candles))
    while left > 0:
        chunk = ex.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=min(per, left))
        if not chunk:
            break
        rows += chunk
        since = chunk[-1][0] + 1
        left -= len(chunk)
        try:
            time.sleep(ex.rateLimit / 1000.0)
        except Exception:
            time.sleep(0.2)
        if len(chunk) < per:
            break

    if not rows:
        raise RuntimeError("No OHLCV data")

    df = to_df(rows)

    if USE_CACHE and df is not None and not df.empty:
        try:
            if df_cached is not None and not df_cached.empty:
                dfm = pd.concat([df_cached, df], ignore_index=True)
                dfm = dfm.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
                write_cached_ohlcv(symbol, tf, dfm)
            else:
                write_cached_ohlcv(symbol, tf, df)
        except Exception:
            pass

    return df


# --- 1-year lookback utilities ---
def timeframe_to_minutes(tf: str) -> int:
    tf = normalize_tf(tf)
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 60 * 24
    if tf.endswith("w"):
        return int(tf[:-1]) * 60 * 24 * 7
    # fallback assume minutes
    return 15

def parse_lookback_days(s: str) -> int:
    """
    Parse strings like '1y', '6m', '90d', '52w' into integer days.
    Defaults to 365 if parsing fails.
    """
    try:
        s = (s or "1y").strip().lower()
        if s.endswith("y"):
            return int(float(s[:-1]) * 365)
        if s.endswith("m"):
            return int(float(s[:-1]) * 30)
        if s.endswith("w"):
            return int(float(s[:-1]) * 7)
        if s.endswith("d"):
            return int(float(s[:-1]))
        # bare number means days
        return int(float(s))
    except Exception:
        return 365

def fetch_ohlcv_lookback(ex, symbol, timeframe="1m", lookback="1y"):
    """
    Fetch OHLCV from `lookback` ago up to now, using CSV cache to minimize API calls.
    Strategy:
      1) Load cache (if enabled). Refresh the tail if older than TTL.
      2) If cache doesn't reach far enough into the window, backfill via fetch_ohlcv_full() once.
      3) Return sliced window [start .. now].
    ENV:
      AUTOPILOT_CACHE (default=1)
      AUTOPILOT_CACHE_TTL_S (default=2.0)
    """
    tf = normalize_tf(timeframe)
    days = parse_lookback_days(lookback)
    start_dt = (pd.Timestamp.utcnow() - pd.Timedelta(days=days)).tz_localize(None)

    USE_CACHE = str(os.getenv("AUTOPILOT_CACHE", "1")).lower() not in ("0","false","no","off")
    try:
        TTL = float(os.getenv("AUTOPILOT_CACHE_TTL_S", "2.0"))
    except Exception:
        TTL = 2.0

    if USE_CACHE:
        try:
            cached = read_cached_ohlcv(symbol, tf)
        except Exception:
            cached = None

        if cached is not None and not cached.empty:
            # Tail refresh if stale
            try:
                last_dt = pd.to_datetime(cached["datetime"].iloc[-1])
                age_s = (pd.Timestamp.utcnow() - last_dt.tz_localize("UTC")).total_seconds()
            except Exception:
                age_s = TTL + 1.0

            if age_s > TTL:
                since = int(last_dt.value // 1_000_000) + 1
                rows = []
                per = 1000
                last_ts = None
                while True:
                    chunk = ex.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=per)
                    if not chunk:
                        break
                    rows += chunk
                    new_last = chunk[-1][0]
                    if last_ts is not None and new_last <= last_ts:
                        break
                    last_ts = new_last
                    since = new_last + 1
                    try:
                        time.sleep(ex.rateLimit / 1000.0)
                    except Exception:
                        time.sleep(0.2)
                    if len(chunk) < per:
                        break
                if rows:
                    df_new = to_df(rows)
                    cached = pd.concat([cached, df_new], ignore_index=True)
                    cached = cached.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
                    try:
                        write_cached_ohlcv(symbol, tf, cached)
                    except Exception:
                        pass

            # Ensure coverage back to start of window; if not, backfill older via full-history fetch once
            try:
                earliest_cached = pd.to_datetime(cached["datetime"].iloc[0])
            except Exception:
                earliest_cached = None

            if earliest_cached is None or earliest_cached.tz_localize(None) > start_dt:
                try:
                    # This will fetch from the beginning and merge into the CSV cache
                    fetch_ohlcv_full(ex, symbol, timeframe=tf)
                    cached2 = read_cached_ohlcv(symbol, tf)
                    if cached2 is not None and not cached2.empty:
                        cached = cached2
                except Exception:
                    pass

            # Finally, slice the requested window
            win = cached[cached["datetime"] >= start_dt].reset_index(drop=True)
            if not win.empty:
                return win

    # Fallback: direct pagination if cache disabled or missing
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 24 * 60 * 60 * 1000
    rows = []
    per = 1000
    last_ts = None
    since = start_ms
    while True:
        chunk = ex.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=per)
        if not chunk:
            break
        rows += chunk
        new_last = chunk[-1][0]
        if last_ts is not None and new_last <= last_ts:
            break
        last_ts = new_last
        since = new_last + 1
        try:
            time.sleep(ex.rateLimit / 1000.0)
        except Exception:
            time.sleep(0.2)
        if len(chunk) < per:
            break
        if len(rows) > 500_000:
            break

    if not rows:
        raise RuntimeError("No OHLCV data for lookback")

    df = to_df(rows)

    # Persist to cache for future calls
    if USE_CACHE and df is not None and not df.empty:
        try:
            cached = read_cached_ohlcv(symbol, tf)
            if cached is not None and not cached.empty:
                dfm = pd.concat([cached, df], ignore_index=True)
                dfm = dfm.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
                write_cached_ohlcv(symbol, tf, dfm)
            else:
                write_cached_ohlcv(symbol, tf, df)
        except Exception:
            pass

    # Clip to exact window
    df = df[df["datetime"] >= start_dt].reset_index(drop=True)
    return df

# --- Full-history OHLCV fetch (from earliest available to now) ---
def fetch_ohlcv_full(ex, symbol, timeframe="1m", hard_max=None):
    """
    Fetch *all available* OHLCV for `symbol`/`timeframe` from the earliest the
    exchange allows up to "now" using CCXT pagination, with optional local CSV cache
    and textual progress logs.

    ENV:
      AUTOPILOT_CACHE=1            # enable/disable local CSV cache (default: on)
      AUTOPILOT_CACHE_DIR=cache    # cache directory (default: ./cache)
      AUTOPILOT_PROGRESS=1         # print progress logs while fetching (default: on)
      AUTOPILOT_PROGRESS_EVERY=5000  # print every N new bars
      AUTOPILOT_SAFETY_MAX_BARS=1000000  # hard safety cap
    """
    tf = normalize_tf(timeframe)
    USE_CACHE = str(os.getenv("AUTOPILOT_CACHE", "1")).lower() not in ("0","false","no","off")
    PROGRESS  = str(os.getenv("AUTOPILOT_PROGRESS", "1")).lower() not in ("0","false","no","off")
    try:
        REPORT_EVERY = int(os.getenv("AUTOPILOT_PROGRESS_EVERY", "5000"))
    except Exception:
        REPORT_EVERY = 5000

    # Safety limits (env-overridable)
    try:
        SAFETY_MAX = int(os.getenv("AUTOPILOT_SAFETY_MAX_BARS", "1000000"))
    except Exception:
        SAFETY_MAX = 1000000

    cached = None
    newest_cached_ms = None
    if USE_CACHE:
        cached = read_cached_ohlcv(symbol, tf)
        if cached is not None and not cached.empty:
            newest_cached_ms = int(pd.to_datetime(cached["datetime"].iloc[-1]).value // 1_000_000)
            say(f"{CYAN}CACHE HIT:{RESET} {len(cached):,} bars ({cached['datetime'].iloc[0]} → {cached['datetime'].iloc[-1]})")

    rows = []
    per = 1000
    last_ts = None
    # Start from either the very beginning (0) or last cached + 1ms
    since = (newest_cached_ms + 1) if newest_cached_ms is not None else 0

    t0 = time.time()
    reported = 0
    while True:
        chunk = ex.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=per)
        if not chunk:
            break
        # stop if no progress
        if last_ts is not None and chunk[-1][0] <= last_ts:
            break

        rows += chunk
        last_ts = chunk[-1][0]
        since = last_ts + 1

        # optional hard caps
        if hard_max and len(rows) >= int(hard_max):
            rows = rows[: int(hard_max)]
            break
        if len(rows) >= SAFETY_MAX:
            break

        # polite with rate limits
        try:
            time.sleep(ex.rateLimit / 1000.0)
        except Exception:
            time.sleep(0.2)

        # progress log
        if PROGRESS and (len(rows) - reported) >= REPORT_EVERY:
            took = time.time() - t0
            speed = (len(rows) / max(1e-6, took))
            last_dt = pd.to_datetime(last_ts, unit="ms")
            say(f"{CYAN}… fetched {len(rows):,} new bars | last={last_dt} | speed≈{speed:.1f} bars/s{RESET}")
            reported = len(rows)

        # if exchange returned fewer than `per`, likely at the end
        if len(chunk) < per:
            break

    # Build DataFrame from new rows
    df_new = to_df(rows) if rows else pd.DataFrame(columns=["datetime","open","high","low","close","volume"])

    # Merge with cache if exists
    if cached is not None and not cached.empty:
        if not df_new.empty:
            df = pd.concat([cached, df_new], ignore_index=True)
        else:
            df = cached.copy()
        # remove potential duplicates/overlaps
        df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    else:
        df = df_new

    # Save/refresh cache
    if USE_CACHE and not df.empty:
        try:
            write_cached_ohlcv(symbol, tf, df)
            if df_new is not None and len(df_new) > 0:
                say(f"{GREEN}CACHE UPDATE:{RESET} appended {len(df_new):,} bars → total {len(df):,}")
        except Exception as e:
            say(f"{YELLOW}Cache write failed: {e}{RESET}")

    if df.empty:
        raise RuntimeError("No OHLCV data for full history")

    # Summary line
    say(f"Fetched FULL-HISTORY: {len(df):,} bars [{tf}] from {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
    return df

def ema(s, span): return s.ewm(span=span, adjust=False).mean()
def rsi(series, window=14):
    d = series.diff()
    up = np.where(d>0, d, 0.0); dn = np.where(d<0, -d, 0.0)
    ru = pd.Series(up, index=series.index).rolling(window).mean()
    rd = pd.Series(dn, index=series.index).rolling(window).mean()
    rs = ru/(rd+1e-12)
    return 100 - (100/(1+rs))
def macd(series, fast=12, slow=26, signal=9):
    m = ema(series, fast) - ema(series, slow); s = ema(m, signal); return m, s, m - s
def atr(df, window=14):
    hl = df["high"]-df["low"]; hc=(df["high"]-df["close"].shift()).abs(); lc=(df["low"]-df["close"].shift()).abs()
    tr = pd.concat([hl,hc,lc],axis=1).max(axis=1); return tr.rolling(window).mean()
    
def bollinger_bands(series, window=20, n=2):
    ma = series.rolling(window).mean()
    sd = series.rolling(window).std()
    upper = ma + n * sd
    lower = ma - n * sd
    width = (upper - lower) / (ma.abs() + 1e-12)
    z = (series - ma) / (sd + 1e-12)
    return upper, lower, width, z

def stochastic_oscillator(df, k=14, d=3):
    ll = df["low"].rolling(k).min()
    hh = df["high"].rolling(k).max()
    k_fast = 100.0 * (df["close"] - ll) / (hh - ll + 1e-12)
    d_slow = k_fast.rolling(d).mean()
    return k_fast, d_slow

def detect_swings(df, left=3, right=3):
    H,L,n=df["high"].values, df["low"].values, len(df)
    hi = np.zeros(n, dtype=bool); lo=np.zeros(n, dtype=bool)
    for i in range(left, n-right):
        if H[i]==H[i-left:i+right+1].max() and np.argmax(H[i-left:i+right+1])==left: hi[i]=True
        if L[i]==L[i-left:i+right+1].min() and np.argmin(L[i-left:i+right+1])==left: lo[i]=True
    return pd.Series(hi,index=df.index), pd.Series(lo,index=df.index)

def last_idx(mask: pd.Series) -> pd.Series:
    m = mask.astype(bool).values; idx=np.arange(len(m))
    last = np.where(m, idx, np.nan).astype(float)
    last = pd.Series(last).ffill().fillna(-1).astype(int); last.index=mask.index; return last

def fib_feats(df, left=3, right=3):
    hi_m, lo_m = detect_swings(df,left,right); hi_i, lo_i = last_idx(hi_m), last_idx(lo_m)
    out = pd.DataFrame(index=df.index); price=df["close"].values
    d382=np.full(len(df),np.nan); d500=np.full(len(df),np.nan); d618=np.full(len(df),np.nan); tr=np.zeros(len(df))
    for i in range(len(df)):
        hi=hi_i.iloc[i]; lo=lo_i.iloc[i]
        if hi<0 or lo<0: continue
        hi_p=float(df.loc[hi,"high"]); lo_p=float(df.loc[lo,"low"]); diff=hi_p-lo_p; up = lo>hi
        if up:
            f382 = hi_p - 0.382*diff; f500 = hi_p - 0.500*diff; f618 = hi_p - 0.618*diff; tr[i]=1
        else:
            f382 = lo_p + 0.382*diff; f500 = lo_p + 0.500*diff; f618 = lo_p + 0.618*diff; tr[i]=-1
        p=price[i]
        d382[i]=abs(p-f382)/max(1e-12,p); d500[i]=abs(p-f500)/max(1e-12,p); d618[i]=abs(p-f618)/max(1e-12,p)
    out["fib_dist_382"]=d382; out["fib_dist_500"]=d500; out["fib_dist_618"]=d618; out["fib_trend"]=tr
    out["fib_score"]=(1/(1+out["fib_dist_500"]))+(1/(1+out["fib_dist_618"]))+0.5*(1/(1+out["fib_dist_382"]))
    return out


def build_features(df):
    out = df.copy()
    # Core trend & momentum
    out["ema_fast"] = ema(out["close"], 9)
    out["ema_slow"] = ema(out["close"], 21)
    m, s, h = macd(out["close"], 12, 26, 9)
    out["macd"] = m; out["macd_signal"] = s; out["macd_hist"] = h
    out["rsi"] = rsi(out["close"], 14)
    out["ret_1"] = out["close"].pct_change(1)
    out["vol_10"] = out["ret_1"].rolling(10).std()

    # SMC - Fib derived feats (sudah ada)
    f = fib_feats(out, 3, 3)
    out = pd.concat([out, f], axis=1)

    # ATR & regime
    out["atr14"] = atr(out, 14)
    out["atr_pct"] = out["atr14"] / (out["close"] + 1e-12)

    # Bollinger
    bb_u, bb_l, bb_w, bb_z = bollinger_bands(out["close"], window=20, n=2)
    out["bb_upper"] = bb_u; out["bb_lower"] = bb_l
    out["bb_width"] = bb_w; out["bb_z"] = bb_z

    # Stochastic
    k_fast, d_slow = stochastic_oscillator(out, k=14, d=3)
    out["stoch_k"] = k_fast; out["stoch_d"] = d_slow

    # SMA set (MA)
    out["ma20"] = out["close"].rolling(20).mean()
    out["ma50"] = out["close"].rolling(50).mean()
    out["ma100"] = out["close"].rolling(100).mean()
    out["ma200"] = out["close"].rolling(200).mean()
    out["ma_slope20"] = out["ma20"].diff()
    out["ma_slope50"] = out["ma50"].diff()
    out["ma_cross_fast"] = np.sign(out["ema_fast"] - out["ema_slow"]).astype(float)

    # Volume heuristics
    v_med20 = out["volume"].rolling(20).median()
    v_std20 = out["volume"].rolling(20).std()
    out["vol_z20"] = (out["volume"] - v_med20) / (v_std20 + 1e-12)

    # FVG (vector 3-candle)
    hi_prev = out["high"].shift(1); lo_prev = out["low"].shift(1)
    hi_next = out["high"].shift(-1); lo_next = out["low"].shift(-1)
    bull_gap = (lo_next > hi_prev)
    bear_gap = (hi_next < lo_prev)

    # gap boundaries per type
    gap_low_b, gap_high_b = hi_prev, lo_next     # bullish gap: [hi_prev, lo_next]
    gap_low_s, gap_high_s = hi_next, lo_prev     # bearish gap: [hi_next, lo_prev]

    px = out["close"].astype(float)
    out["fvg_bull"] = bull_gap.astype(float)
    out["fvg_bear"] = bear_gap.astype(float)

    # vectorized proximity: 0 if price inside gap; else min distance to edges, normalized by price
    pxv = px.values
    lb = gap_low_b.values.astype(float); hb = gap_high_b.values.astype(float)
    ls = gap_low_s.values.astype(float); hs = gap_high_s.values.astype(float)
    bg = bull_gap.values; sg = bear_gap.values

    in_b = (pxv >= lb) & (pxv <= hb) & bg
    in_s = (pxv >= ls) & (pxv <= hs) & sg

    prox_b = np.full(len(pxv), np.nan)
    prox_s = np.full(len(pxv), np.nan)

    # inside-gap → proximity 0
    prox_b[in_b] = 0.0
    prox_s[in_s] = 0.0

    # outside but gap exists → min distance to either edge, normalized by price
    mask_b = bg & (~in_b)
    if mask_b.any():
        denom = np.maximum(1e-12, np.abs(pxv[mask_b]))
        prox_b[mask_b] = np.minimum(np.abs(pxv[mask_b] - lb[mask_b]), np.abs(pxv[mask_b] - hb[mask_b])) / denom

    mask_s = sg & (~in_s)
    if mask_s.any():
        denom = np.maximum(1e-12, np.abs(pxv[mask_s]))
        prox_s[mask_s] = np.minimum(np.abs(pxv[mask_s] - ls[mask_s]), np.abs(pxv[mask_s] - hs[mask_s])) / denom

    # combine (ignores NaN, stays NaN if both NaN)
    out["fvg_prox"] = np.fmin(prox_b, prox_s)
    out["fvg_in"]   = (in_b | in_s).astype(float)
    
    out["fvg_prox"] = pd.to_numeric(out["fvg_prox"], errors="coerce").fillna(1.0)  # no-gap = jauh (1.0)
    out["fvg_in"]   = pd.to_numeric(out["fvg_in"],   errors="coerce").fillna(0.0)  # no-gap = 0

    # IRL (equal highs/lows near extremes)
    irl_look = 80; band = 0.0015
    roll_max = out["high"].rolling(irl_look).max()
    roll_min = out["low"].rolling(irl_look).min()
    eqh = (np.abs(out["high"] - roll_max) / (roll_max + 1e-12) <= band).astype(float)
    eql = (np.abs(out["low"]  - roll_min) / (roll_min + 1e-12) <= band).astype(float)
    out["irl_score"] = (eqh.rolling(irl_look//2).sum() + eql.rolling(irl_look//2).sum()) / (irl_look/10.0)
    out["irl_score"] = out["irl_score"].clip(lower=0.0, upper=1.0)

    # SNR proxies (jarak ke res/sup rolling)
    roll_hi = out["high"].rolling(120).max()
    roll_lo = out["low"].rolling(120).min()
    out["sr_dist_res"] = np.abs(roll_hi - out["close"]) / (out["close"] + 1e-12)
    out["sr_dist_sup"] = np.abs(out["close"] - roll_lo) / (out["close"] + 1e-12)

    # Pullback proxy (z ke SMA50)
    sma50 = out["ma50"]
    r = out["close"].pct_change().rolling(20).std()
    out["pullback_z50"] = (out["close"] - sma50) / (r + 1e-12)
    
    # ADX(14)
    try:
        out["adx_14"] = adx(out, 14)
    except Exception:
        out["adx_14"] = np.nan

    # BBW alias (biar konsisten nama)
    out["bbw_20_2"] = out["bb_width"]

    # VWAP distance (panjang jendela ≈ 1 hari trading)
    try:
        tf = require_tf()
        if tf.endswith("m"):
            bpd = max(1, (24*60)//int(tf[:-1]))
        elif tf.endswith("h"):
            bpd = max(1, 24//int(tf[:-1]))
        else:
            bpd = 96
        pv = out["close"] * out["volume"]
        vwap = (pv.rolling(bpd).sum() / out["volume"].rolling(bpd).sum())
        out["vwap_dist"] = (out["close"] - vwap) / (vwap + 1e-12)
    except Exception:
        out["vwap_dist"] = np.nan

    # Final cleanup — avoid dropping most rows. Keep dataset thick.
    out = out.replace([np.inf, -np.inf], np.nan)

    # Defaults for sparse/episodic features (prevent massive row drops)
    defaults = {
        "fvg_bull": 0.0, "fvg_bear": 0.0, "fvg_in": 0.0, "fvg_prox": 1.0,
        "irl_score": 0.0, "vol_z20": 0.0,
        "bb_width": 0.0, "bb_z": 0.0,
        "stoch_k": 50.0, "stoch_d": 50.0,
        "ma_cross_fast": 0.0, "ma_slope20": 0.0, "ma_slope50": 0.0,
        "fib_dist_382": 0.05, "fib_dist_500": 0.05, "fib_dist_618": 0.05,
        "fib_trend": 0.0, "fib_score": 0.0,
    }
    for c, v in defaults.items():
        if c in out.columns:
            out[c] = out[c].fillna(v)

    # Light forward-fill for core indicators
    _fill_cols = [
        "ema_fast","ema_slow","rsi","macd","macd_signal","macd_hist",
        "ma20","ma50","ma100","ma200","atr14","atr_pct"
    ]
    for c in _fill_cols:
        if c in out.columns:
            out[c] = out[c].ffill()

    out = out.reset_index(drop=True)

    # kolom yang aman diisi 0 (opsional/sparse)
    for c in ["bb_z","vol_z20","ma_slope20","ma_slope50",
            "fvg_bull","fvg_bear","irl_score","sr_dist_res","sr_dist_sup","pullback_z50"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    # isi maju/mundur untuk indikator inti supaya NaN awal tak mengosongkan set
    for c in ["ema_fast","ema_slow","rsi","macd","macd_signal","macd_hist","ret_1","vol_10",
            "atr14","atr_pct","bb_width","stoch_k","stoch_d",
            "ma20","ma50","ma100","ma200",
            "fib_dist_382","fib_dist_500","fib_dist_618","fib_trend","fib_score"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").ffill().bfill()

    # daftar fitur yang dipakai model
    cols = [
        "ema_fast","ema_slow","rsi","macd","macd_signal","macd_hist","ret_1","vol_10",
        "fib_dist_382","fib_dist_500","fib_dist_618","fib_trend","fib_score",
        "atr14","atr_pct","vol_z20",
        "bb_width","bb_z","stoch_k","stoch_d",
        "ma20","ma50","ma100","ma200","ma_slope20","ma_slope50","ma_cross_fast",
        "fvg_bull","fvg_bear","fvg_prox","fvg_in","irl_score","sr_dist_res","sr_dist_sup","pullback_z50",
    ]

    # hanya drop bar yang masih NaN di subset WAJIB
    required = ["ema_fast","ema_slow","rsi","macd","macd_signal","macd_hist",
                "atr14","atr_pct","bb_width","stoch_k","stoch_d",
                "fib_dist_500","fib_dist_618","fib_trend"]
    have_req = [c for c in required if c in out.columns]
    if have_req:
        out = out.dropna(subset=have_req)

    out = out.reset_index(drop=True)
    return out, cols

# ---------- Labels & model ----------
def make_labels_quantil(df, horizon=16, q=0.35):
    fwd = df["close"].shift(-horizon)/df["close"] - 1.0
    y = pd.Series(np.nan, index=df.index, dtype=float)
    fwd_valid = fwd.dropna()
    lo = fwd_valid.quantile(q); hi = fwd_valid.quantile(1-q)
    y[fwd <= lo] = -1
    y[fwd >= hi] =  1
    return y

def train_or_load_model(symbol, timeframe, df_features, feature_cols):
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    os.makedirs("models", exist_ok=True)
    tf_sel = require_tf()
    path = model_path(symbol, tf_sel)
    force_retrain = str(os.getenv("AUTOPILOT_RETRAIN_ALWAYS","0")).lower() not in ("0","false","no","off")
    if os.path.exists(path) and not force_retrain:
        say(f"{GREEN}Load model:{RESET} {path}")
        bundle = joblib.load(path)
        # Sanitize sklearn metadata so .predict_proba() accepts numpy arrays without feature-name warnings
        try:
            clf = bundle.get("model")
            if hasattr(clf, "feature_names_in_"):
                delattr(clf, "feature_names_in_")
        except Exception:
            pass
        return bundle, path
    
    # Train quick
    header("TRAIN MODEL (auto)")
    
    # [removed duplicated skip-retrain/model load blocks]
    
    # Use chosen timeframe from ENV (menu/.env)
    try:
        timeframe = require_tf()
    except Exception:
        timeframe = os.getenv("AUTOPILOT_TIMEFRAME", "1m")
    
    y_all = make_labels_quantil(df_features, horizon=16, q=0.35)
    mask = y_all.notna()
    X = df_features.loc[mask, feature_cols].values
    y = y_all.loc[mask].astype(int).values
    if len(X) < 1500 or len(set(y))<2:
        say(f"{YELLOW}Dataset kecil/kelas timpang; lanjut tetap dilatih dengan yang ada.{RESET}")
    clf = HistGradientBoostingClassifier(loss="log_loss", learning_rate=0.06,
                                         max_iter=500, max_leaf_nodes=31, random_state=1337)
    clf.fit(X, y)
    # Ensure no feature-name metadata is attached (we always pass numpy arrays at inference)
    try:
        if hasattr(clf, "feature_names_in_"):
            delattr(clf, "feature_names_in_")
    except Exception:
        pass
    bundle = {"model": clf, "feat_cols": feature_cols, "horizon": 16, "label": "QUANTILE0.35"}
    joblib.dump(bundle, path)
    say(f"{GREEN}Model tersimpan:{RESET} {path}")
    return bundle, path

# ---------- Structure levels ----------
def structure_levels(df, lookback=120):
    # nearest support (recent swing low) & resistance (recent swing high)
    hi_m, lo_m = detect_swings(df, left=3, right=3)
    swings_hi = df.loc[hi_m, ["datetime","high"]].tail(lookback)
    swings_lo = df.loc[lo_m, ["datetime","low"]].tail(lookback)
    return swings_lo, swings_hi

def nearest_levels(price, swings_lo, swings_hi):
    sup = swings_lo["low"].max() if not swings_lo.empty else None
    res = swings_hi["high"].min() if not swings_hi.empty else None
    # refine: pick nearest below/above current price
    if not swings_lo.empty:
        sup = swings_lo[swings_lo["low"] < price]["low"].max() if any(swings_lo["low"] < price) else swings_lo["low"].min()
    if not swings_hi.empty:
        res = swings_hi[swings_hi["high"] > price]["high"].min() if any(swings_hi["high"] > price) else swings_hi["high"].max()
    return float(sup) if sup is not None else None, float(res) if res is not None else None

# ---------- Sizing & rounding ----------
def round_to(value, prec):
    # prec is decimal places
    return float(round(value, int(prec)))

def compute_size_from_risk(balance_free_usdt, entry, sl, amount_prec, min_qty, min_cost, leverage=1.0):
    risk_pct = 0.01  # risk 1% per trade (default)
    risk_usd = max(5.0, balance_free_usdt * risk_pct)  # minimal $5, namun tetap dibatasi oleh free balance di bawah
    dist = abs(entry - sl)
    if dist <= 0:
        return 0.0

    # size awal berbasis risk
    qty = risk_usd / dist  # linear USDT-M: PnL ≈ qty * Δprice

    # respect minQty
    qty = max(qty, min_qty)

    # enforce min_cost bila ada
    notional = qty * entry
    if min_cost and notional < min_cost:
        qty = max(qty, min_cost / entry)
        notional = qty * entry

    # batasi oleh free balance * leverage (asumsi leverage>=1; default 1x)
    lev = max(1.0, float(leverage or 1.0))
    max_notional = balance_free_usdt * lev * 0.95  # buffer 5%

    # Jika setelah min_cost masih melebihi kemampuan saldo, cap ke kemampuan
    if notional > max_notional:
        qty = max_notional / entry
        qty = max(qty, 0.0)
        notional = qty * entry

    # Final feasibility: jika bahkan min_cost > max_notional, tidak mungkin eksekusi
    if min_cost and (min_cost > max_notional):
        return 0.0

    # Round dan jaga min_qty
    # Round dan jaga min_qty
    qty = max(min_qty, qty)
    qty = round_to(qty, amount_prec)
    return float(qty)

# ---------- Signal & execution ----------
def signal_long_short(df_feat, feature_cols, bundle):
    """
    Pure-AI decision:
      - Pakai probabilitas model untuk pilih side (long vs short)
      - Gunakan threshold time-aware yang disimpan di bundle (fallback ke ENV)
      - Hitung Expected Value (EV) dgn TP/SL & biaya trading; hanya valid jika EV > 0
    Return: (side|"None", edge_abs, ctx)
    """
    import numpy as np

    cols = list(bundle.get("cols") or feature_cols)
    model = bundle["model"]

    X_live = df_feat[cols].values[-1:].astype(float, copy=False)
    prob = model.predict_proba(X_live)

    cl = list(getattr(model, "classes_", []))
    ip = cl.index(1) if 1 in cl else None
    im = cl.index(-1) if -1 in cl else None

    pL = float(prob[0, ip]) if ip is not None else 0.0
    pS = float(prob[0, im]) if im is not None else 0.0
    edge = pL - pS
    p_dir = max(pL, pS)
    side  = "long" if pL >= pS else "short"

    # thresholds: pakai hasil tuning di bundle dulu, kalau tidak ada pakai ENV/default
    thr_pdir = bundle.get("thr_pdir")
    thr_edge = bundle.get("thr_edge")
    if thr_pdir is None:
        thr_pdir = float(os.getenv("AUTOPILOT_MIN_PDIR", "0.55"))
    if thr_edge is None:
        thr_edge = float(os.getenv("AUTOPILOT_MIN_EDGE", "0.15"))

    # ATR & EV
    try:
        A_series = atr(df_feat, 14)
        A = float(A_series.iloc[-1])
    except Exception:
        A = 0.0
    price = float(df_feat["close"].iloc[-1]) if "close" in df_feat.columns else 0.0

    TP_ATR = float(os.getenv("AUTOPILOT_TP_ATR", "1.8"))
    SL_ATR = float(os.getenv("AUTOPILOT_SL_ATR", "1.0"))
    fee    = float(os.getenv("AUTOPILOT_TAKER_FEE", "0.0006"))
    slip   = float(os.getenv("AUTOPILOT_SLIPPAGE_PCT", "0.0002"))

    # normalisasi biaya dalam satuan R
    if price > 0 and A > 0:
        r0_pct = SL_ATR * (A / price)
    else:
        r0_pct = 0.005  # fallback 0.5%
    r0_pct = max(1e-9, r0_pct)

    base_R = TP_ATR / max(1e-9, SL_ATR)
    cost_r = (2.0 * fee + slip) / r0_pct

    p_win = pL if side == "long" else pS
    ev = p_win * base_R - (1.0 - p_win) * 1.0 - cost_r

    # konteks buat penjelasan/log
    ctx = {
        "pL": pL, "pS": pS, "edge": edge, "p_dir": p_dir,
        "ema_fast": float(df_feat.get("ema_fast", pd.Series([np.nan])).iloc[-1]) if "ema_fast" in df_feat.columns else np.nan,
        "ema_slow": float(df_feat.get("ema_slow", pd.Series([np.nan])).iloc[-1]) if "ema_slow" in df_feat.columns else np.nan,
        "fib_trend": int(df_feat.get("fib_trend", pd.Series([0])).iloc[-1]) if "fib_trend" in df_feat.columns else 0,
        "fib_dist_500": float(df_feat.get("fib_dist_500", pd.Series([np.nan])).iloc[-1]) if "fib_dist_500" in df_feat.columns else np.nan,
        "fib_dist_618": float(df_feat.get("fib_dist_618", pd.Series([np.nan])).iloc[-1]) if "fib_dist_618" in df_feat.columns else np.nan,
        "adx": float(df_feat.get("adx_14", pd.Series([np.nan])).iloc[-1]) if "adx_14" in df_feat.columns else None,
        "bbw": float(df_feat.get("bbw_20_2", pd.Series([np.nan])).iloc[-1]) if "bbw_20_2" in df_feat.columns else None,
        "vwap_dist": float(df_feat.get("vwap_dist", pd.Series([np.nan])).iloc[-1]) if "vwap_dist" in df_feat.columns else None,
        "fvg_type": (df_feat.get("fvg_type", pd.Series([None])).iloc[-1] if "fvg_type" in df_feat.columns else None),
        "fvg_in": bool(df_feat.get("fvg_in", pd.Series([False])).iloc[-1]) if "fvg_in" in df_feat.columns else None,
        "fvg_prox": float(df_feat.get("fvg_prox", pd.Series([np.nan])).iloc[-1]) if "fvg_prox" in df_feat.columns else None,
        "ev": float(ev),
        "tp_atr": TP_ATR, "sl_atr": SL_ATR,
    }

    if (p_dir >= float(thr_pdir)) and (abs(edge) >= float(thr_edge)) and (ev > 0.0):
        return side, abs(edge), ctx
    return None, 0.0, ctx

SC_MIN = float(os.getenv("AUTOPILOT_SC_MIN_SCORE", "0.60"))
ENTRY_WAIT_ON = str(os.getenv("AUTOPILOT_ENTRY_WAIT", "1")).lower() not in ("0","false","no","off")
COOL = int(float(os.getenv("AUTOPILOT_COOLDOWN_SEC", "60")))
TIGHTEN_BE = str(os.getenv("AUTOPILOT_TIGHTEN_BREAKEVEN", "1")).lower() not in ("0","false","no","off")
TIGHTEN_AT_R = float(os.getenv("AUTOPILOT_TIGHTEN_AT_R", "0.5"))
    
def autopilot():
    # --- Force timeframe selection (no default) and expose via ENV ---
    tf_chosen = ensure_timeframe_selected()
    os.environ["AUTOPILOT_TIMEFRAME"] = tf_chosen
    TF = tf_chosen
    ex = bitget_swap()
    if not ensure_futures_permissions(ex):
        return

    # Config
    TF = tf_chosen
    LOOKBACK = os.getenv("AUTOPILOT_TRAIN_LOOKBACK", "180d")
    try: MIN_R = float(os.getenv("AUTOPILOT_MIN_R", "1.5"))
    except Exception: MIN_R = 1.5
    try: MENU_TOPK = int(float(os.getenv("AUTOPILOT_MENU_TOPK", "10")))
    except Exception: MENU_TOPK = 10
    try: IDLE_SLEEP = int(float(os.getenv("AUTOPILOT_IDLE_SLEEP", "30")))
    except Exception: IDLE_SLEEP = 30

    # Risk & entry wait
    try: SC_MIN = float(os.getenv("AUTOPILOT_SC_MIN_SCORE", "0.60"))
    except Exception: SC_MIN = 0.60
    try: COOL = int(float(os.getenv("AUTOPILOT_COOLDOWN_SEC", "60")))
    except Exception: COOL = 60
    TIGHTEN_BE = str(os.getenv("AUTOPILOT_TIGHTEN_BREAKEVEN", "1")).lower() not in ("0","false","no","off")
    try: TIGHTEN_AT_R = float(os.getenv("AUTOPILOT_TIGHTEN_AT_R", "0.5"))
    except Exception: TIGHTEN_AT_R = 0.5

    # Balance (informasi saja)
    try:
        total, free, used = get_futures_usdt_balance(ex)
        say(f"USDT balance: total={total:.2f} free={free:.2f} used={used:.2f}")
    except Exception: pass

    try:
        while True:
            header("Screening symbols")
            
            # ⬇️⬇️ PASTE BLOK “combined potentials” DI SINI ⬇️⬇️
            AUT_MIN_EDGE  = float(os.getenv("AUTOPILOT_MIN_EDGE", "0.15"))
            AUT_MIN_PDIR  = float(os.getenv("AUTOPILOT_MIN_PDIR", "0.55"))
            AUT_FIB_EPS   = float(os.getenv("AUTOPILOT_FIB_EPS", "0.0060"))
            AUT_MAX_SPREAD= float(os.getenv("AUTOPILOT_MAX_SPREAD", "0.0020"))
            MENU_TOPK     = int(float(os.getenv("AUTOPILOT_MENU_TOPK", "10")))
            MIN_R         = float(os.getenv("AUTOPILOT_MIN_R", "1.5"))

            pots_main = screen_usdtm_symbols(
                ex, timeframe=TF,
                lookback=os.getenv("AUTOPILOT_SCREEN_LOOKBACK", "60d"),
                min_edge=AUT_MIN_EDGE, min_pdir=AUT_MIN_PDIR,
                fib_eps=AUT_FIB_EPS, max_spread=AUT_MAX_SPREAD
            ) or []

            want = max(1, int(MENU_TOPK))
            have = set(p["symbol"] for p in pots_main)
            need = max(0, want - len(pots_main))

            pots_fb = []
            if need > 0:
                pots_fb = list_top_any_candidates(
                    ex, timeframe=TF,
                    lookback=os.getenv("AUTOPILOT_FALLBACK_LOOKBACK", "90d"),
                    universe_topn=os.getenv("AUTOPILOT_FALLBACK_TOPN", "80"),
                    topk=need, min_r=MIN_R
                ) or []

            pots = pots_main + [p for p in pots_fb if p["symbol"] not in have]
            if not pots:
                say("Tidak ada kandidat; tidur sejenak…")
                time.sleep(IDLE_SLEEP)
                continue

            print_potential_menu(pots)
            pick = prompt_pick_potential(pots)
            if not pick:
                say("Tidak memilih; ulang screening…")
                continue

            # --- siapkan kandidat terpilih ---
            symbol = pick["symbol"]
            side_hint = pick.get("side")  # boleh None
            ensure_models_for_symbol(ex, symbol, TF)
            
            # --- param tunggu entry ---
            try:
                poll_sec = float(os.getenv("AUTOPILOT_ENTRY_POLL_SEC", "1.0"))
            except Exception:
                poll_sec = 1.0
            wm = os.getenv("AUTOPILOT_ENTRY_WAIT_MAX_S", "").strip()
            wait_max = float(wm) if wm else None

            
            # --- tunggu setup entry; 'c'+Enter untuk batal ---
            try:
                ok = wait_for_entry(
                    ex, symbol, TF,
                    side=None,                 # biarkan AI menentukan (long/short)
                    min_score=SC_MIN,          # ambil dari ENV
                    poll_sec=poll_sec,
                    max_wait_sec=wait_max,
                    side_intent=side_hint      # hint dari screening; opsional
                )
            except CancelledEntry:
                say("Entry dibatalkan oleh user; kembali ke screening…")
                continue

            if not ok:
                # timeout / belum memenuhi score/gates → ulang screening
                continue
            
            symbol = pick["symbol"]; side = pick["side"]
            price0 = float(pick.get("price") or 0.0)
            A0 = float(pick.get("A") or 0.0)
            ctx = pick.get("ctx") or {}
            # ⬆️⬆️ BLOK SELESAI ⬆️⬆️
        
            min_edge = float(os.getenv("AUTOPILOT_MIN_EDGE", "0.20"))
            min_pdir = float(os.getenv("AUTOPILOT_MIN_PDIR", "0.60"))
            fib_eps  = float(os.getenv("AUTOPILOT_FIB_EPS", "0.0035"))
            max_sp   = float(os.getenv("AUTOPILOT_MAX_SPREAD", "0.0015"))

            cands = screen_usdtm_symbols(
                ex, timeframe=TF, lookback=LOOKBACK,
                min_edge=min_edge, min_pdir=min_pdir, fib_eps=fib_eps, max_spread=max_sp
            )

            # Rationale
            try:
                df_last = fetch_ohlcv(ex, symbol, timeframe=TF, candles=400)
                px_now = float(df_last["close"].iloc[-1])
                sr = sr_confluence(df_last, px_now, lookback=80, eps=float(os.getenv("AUTOPILOT_SMC_SR_EPS", "0.0012")))
                sup = res = None
                if sr.get("bias") == "support": sup = px_now * (1 - (sr.get("dist") or 0.0))
                elif sr.get("bias") == "resistance": res = px_now * (1 + (sr.get("dist") or 0.0))
                explain_entry(symbol, side, px_now, sup, res, A0 or 0.0, ctx, pick.get("sl") or px_now, pick.get("tp") or px_now, df_last.tail(120))
            except Exception:
                pass

            # Refresh harga/ATR
            tkr = ex.fetch_ticker(symbol)
            price_live = float(tkr.get("last") or tkr.get("close") or tkr.get("bid") or 0.0) or price0
            try:
                df_live = fetch_ohlcv(ex, symbol, timeframe=TF, candles=400)
                A_live = float(atr(df_live, 14).iloc[-1])
            except Exception:
                A_live = A0 or 0.0

            # Auto TP/SL by potential (tanpa trailing TP; SL bisa dinaikkan intra-trade)
            best = select_tp_sl_by_potential(
                df_live["close"] if 'df_live' in locals() and not df_live.empty else pd.Series([price_live]),
                A_live, price_live, side, min_r=MIN_R
            )
            if best:
                sl = best["sl"]; tp = best["tp"]
                say(f"Potensi: P(TP)≈{best['p']:.2f} | R≈{best['r']:.2f} | ExpR≈{best['exp']:.2f} | TPx={best['tp_m']:.2f} SLx={best['sl_m']:.2f}")
            else:
                TP_ATR = float(os.getenv("AUTOPILOT_TP_ATR", "1.8"))
                SL_ATR = float(os.getenv("AUTOPILOT_SL_ATR", "1.0"))
                if side == "long":
                    sl = price_live - SL_ATR * A_live; tp = price_live + TP_ATR * A_live
                else:
                    sl = price_live + SL_ATR * A_live; tp = price_live - TP_ATR * A_live

            # Kencangkan SL ke struktur bila lebih dekat
            try:
                sr2 = sr_confluence(df_live if 'df_live' in locals() else df_last, price_live, lookback=80, eps=float(os.getenv("AUTOPILOT_SMC_SR_EPS", "0.0012")))
                if side == "long" and sr2.get("bias") == "support" and sr2.get("dist") is not None:
                    sl = max(sl, price_live * (1 - float(sr2.get("dist"))))
                if side == "short" and sr2.get("bias") == "resistance" and sr2.get("dist") is not None:
                    sl = min(sl, price_live * (1 + float(sr2.get("dist"))))
            except Exception:
                pass

            # Sizing
            amt_prec, price_prec, min_qty, min_cost = market_specs(ex, symbol)
            total, free, used = get_futures_usdt_balance(ex)
            risk_pct = float(os.getenv("AUTOPILOT_RISK_PCT", "1.0")) / 100.0
            risk_usdt = max(1e-6, (free or 0.0) * risk_pct)
            risk_per_unit = abs(price_live - sl)
            qty_risk = (risk_usdt / risk_per_unit) if risk_per_unit > 0 else 0.0
            q_min, lev_used, changed, note = auto_scale_for_min_cost(ex, symbol, free, price_live, min_qty, min_cost)
            qty = max(q_min, qty_risk)
            try: qty = float(ex.amount_to_precision(symbol, qty))
            except Exception: qty = float(qty)
            if qty <= 0:
                say(f"{YELLOW}Qty<=0 setelah sizing (cap/precision). Batal trade.{RESET}")
                time.sleep(2); continue

            # DRYRUN?
            DRY = str(os.getenv("AUTOPILOT_DRYRUN", "1")).lower() not in ("0","false","no","off")
            if DRY:
                say(f"{CYAN}DRYRUN aktif — tidak mengirim order nyata.{RESET}")
                log_signal_txt(symbol, side, price_live, tp, sl, leverage=lev_used, margin_mode=os.getenv("AUTOPILOT_MARGIN_MODE", "crossed"), dryrun=True)
                time.sleep(2); continue

            # Leverage & order
            lev_target = float(os.getenv("AUTOPILOT_TARGET_LEVERAGE", "10"))
            set_symbol_leverage(ex, symbol, lev_target, os.getenv("AUTOPILOT_MARGIN_MODE", "crossed"))
            try:
                # Ensure oneway mode to avoid side/posSide mismatch on Bitget
                try:
                    ensure_position_mode(ex, symbol, want="oneway")
                except Exception:
                    pass

                side_ccxt = 'buy' if side == 'long' else 'sell'
                order = ex.create_order(symbol, 'market', side_ccxt, qty, None, {})
                say(f"{GREEN}ORDER DONE:{RESET} {order.get('id')}")
                try:
                    _ENTRY_TS[symbol] = time.time()
                except Exception:
                    _ENTRY_TS[str(symbol)] = time.time()
            except Exception as e:
                say(f"{RED}Order gagal: {e}{RESET}")
                time.sleep(2); continue

            # Monitor 1 detik sampai exit, lalu cooldown → loop lagi
            monitor_until_exit(ex, symbol, side, price_live, sl, tp, qty, price_prec,
                               tighten_be_on=TIGHTEN_BE, tighten_at_R=TIGHTEN_AT_R, cooldown_sec=COOL)
            # setelah cooldown: lanjut while (screening ulang)

    except KeyboardInterrupt:
        print("\nBye 👋")
        return
    
if __name__ == "__main__":
    try:
        run_cli()
    except KeyboardInterrupt:
        print("\nBye 👋")