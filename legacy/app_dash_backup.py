# app_dash.py - Optimized AI Trading Dashboard
# Enhanced Bitget Futures Trading Dashboard with Resource Optimization
# Indicators: EMA(20/50), Volume, MACD, RSI, Stochastic, S/R, Elliott Wave, BB
# Features:
# - Memory-optimized real-time refresh with smart caching
# - Elliott Wave pattern recognition and labeling  
# - Resource-efficient chart updates with delta compression
# - AI-powered signal generation with GPT integration
# - Advanced position management with trailing stops
# - Multi-timeframe analysis with minimal API calls

import os, time, threading, webbrowser
import re
import xml.etree.ElementTree as ET
from datetime import datetime
import numpy as np
import pandas as pd
import gc
np.seterr(all="ignore")
pd.options.mode.copy_on_write = True

from dotenv import load_dotenv
from functools import lru_cache

# --- HTTP session with retry ---
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import ccxt
import requests
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from zoneinfo import ZoneInfo
LOCAL_TZ = ZoneInfo("Asia/Jakarta")  # WIB

from dash import Dash, html, dcc, Input, Output, State, no_update, ctx
from dash.dependencies import ClientsideFunction
# =============================================
# HEDGE FUND PROFESSIONAL UI STYLING
# =============================================
# Premium dark theme inspired by institutional trading platforms

# Professional Color Palette
COLORS = {
    'primary': '#0A0E27',      # Deep navy background
    'secondary': '#1A1D3A',    # Secondary dark blue
    'accent': '#00D4FF',       # Bright cyan accent
    'success': '#00FF88',      # Bright green for profits
    'danger': '#FF3366',       # Bright red for losses
    'warning': '#FFB800',      # Amber for warnings
    'info': '#7C3AED',         # Purple for info
    'text_primary': '#FFFFFF', # White text
    'text_secondary': '#B4BCD0', # Light gray text
    'border': '#2A2D47',       # Border color
    'surface': '#141629',      # Card surface
    'glass': 'rgba(26, 29, 58, 0.8)', # Glass effect
}

# Premium Card Styling
CARD_STYLE = {
    "background": f"linear-gradient(135deg, {COLORS['surface']}, {COLORS['secondary']})",
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "12px",
    "padding": "16px",
    "backdropFilter": "blur(20px)",
    "boxShadow": "0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1)",
    "color": COLORS['text_primary'],
    "transition": "all 0.3s ease",
}

# Advanced Grid Layouts
GRID_2COL = {
    "display": "grid",
    "gridTemplateColumns": "1fr 1fr", 
    "gap": "16px",
    "alignItems": "start"
}

GRID_3COL = {
    "display": "grid", 
    "gridTemplateColumns": "1fr 1fr 1fr",
    "gap": "16px",
    "alignItems": "start"
}

GRID_4COL = {
    "display": "grid",
    "gridTemplateColumns": "repeat(4, 1fr)",
    "gap": "12px",
    "alignItems": "start"
}

# Professional Dashboard Layout
DASHBOARD_LAYOUT = {
    "display": "grid",
    "gridTemplateColumns": "280px 1fr",  # Sidebar + Main
    "gridTemplateRows": "60px 1fr",      # Header + Content
    "height": "100vh",
    "gap": "0",
    "background": f"linear-gradient(135deg, {COLORS['primary']}, #0F1329)"
}

# Sidebar Styling
SIDEBAR_STYLE = {
    "background": f"linear-gradient(180deg, {COLORS['secondary']}, {COLORS['surface']})",
    "borderRight": f"1px solid {COLORS['border']}",
    "padding": "20px",
    "height": "100vh",
    "overflowY": "auto",
    "backdropFilter": "blur(20px)",
    "boxShadow": "2px 0 20px rgba(0, 0, 0, 0.3)"
}

# Header Bar Styling
HEADER_STYLE = {
    "background": f"linear-gradient(90deg, {COLORS['surface']}, {COLORS['secondary']})",
    "borderBottom": f"1px solid {COLORS['border']}",
    "padding": "0 20px",
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "space-between",
    "backdropFilter": "blur(20px)",
    "boxShadow": "0 2px 20px rgba(0, 0, 0, 0.3)"
}

# ===============================================
# RESOURCE OPTIMIZATION & PERFORMANCE SETTINGS  
# ===============================================
# Memory-optimized configuration for real-time trading dashboard
# Designed for minimal resource usage while maintaining full functionality

DEBUG_LOG = os.getenv("DEBUG_LOG", "false").lower() in ("1","true","yes","on")
DF_FLOAT_DTYPE = "float32"  # Always use float32 for 50% memory reduction

# Data Processing Limits (Reduced for Performance)
IND_MAX_LEN = int(os.getenv("IND_MAX_LEN", "800"))      # Max indicators length (was 2000)
OB_SCAN_FACTOR = float(os.getenv("OB_SCAN_FACTOR", "1.5"))  # Order block scan efficiency
PLOT_BARS = int(os.getenv("PLOT_BARS", "200"))          # Chart bars (was 320)

# Garbage Collection & Memory Management
GC_TUNE       = True  # Force enable optimized garbage collection
GC_EVERY      = int(os.getenv("GC_EVERY","25"))         # More frequent GC (was 80)

# Cache Optimization (Aggressive Memory Management)
FIG_CACHE_MAX = int(os.getenv("FIG_CACHE_MAX","6"))     # Chart cache (was 24)
OBJ_CACHE_MAX = int(os.getenv("OBJ_CACHE_MAX","12"))    # Object cache (was 64)
TICKER_CACHE_MAX = int(os.getenv("TICKER_CACHE_MAX","24"))  # Ticker cache (was 128)
CACHE_WRITE_SKIP = int(os.getenv("CACHE_WRITE_SKIP","8"))   # Disk write frequency

# Elliott Wave Analysis Settings
EW_LOOKBACK = int(os.getenv("EW_LOOKBACK", "100"))      # Analysis window for EW patterns
EW_MIN_SWING = float(os.getenv("EW_MIN_SWING", "0.008")) # Min swing: 0.8% for noise filtering
EW_CACHE_TTL = int(os.getenv("EW_CACHE_TTL", "300"))     # Cache TTL: 5 minutes

# Performance Monitoring
PERF_MONITOR = os.getenv("PERF_MONITOR", "false").lower() in ("1","true","yes","on")
MAX_MEMORY_MB = int(os.getenv("MAX_MEMORY_MB", "500"))   # Memory limit warning threshold

# -------------------- ENV & Exchange --------------------
load_dotenv()
BITGET_KEY      = os.getenv("BITGET_KEY")
BITGET_SECRET   = os.getenv("BITGET_SECRET")
BITGET_PASSWORD = os.getenv("BITGET_PASSWORD")

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-5-thinking")

# --- AI & Partial TP / News settings ---
# --- Partial TP by ROE (percent) ---
# --- Trailing Stop settings (by ROE & ATR lock) ---
TRAIL_ENABLE       = os.getenv("TRAIL_ENABLE","true").lower() in ("1","true","yes","on")
TRAIL_BE_ROE       = float(os.getenv("TRAIL_BE_ROE","20"))    # move SL to breakeven when ROE >= 20%
TRAIL_STEP_ROE     = float(os.getenv("TRAIL_STEP_ROE","50"))  # every +50% ROE, tighten SL
TRAIL_LOCK_ATR     = float(os.getenv("TRAIL_LOCK_ATR","0.8"))  # lock distance in ATR multiples
TRAIL_MAX_LOCK_ATR = float(os.getenv("TRAIL_MAX_LOCK_ATR","2.5"))

ROE_TP_ENABLE    = os.getenv("ROE_TP_ENABLE","true").lower() in ("1","true","yes","on")
ROE_TP_STEPS_ENV = os.getenv("ROE_TP_STEPS","[(50,0.25),(100,0.25),(200,0.25)]")  # [(roe%, fraction)]
PARTIAL_TP       = os.getenv("PARTIAL_TP","true").lower() in ("1","true","yes","on")
TP_SCALE_ENV     = os.getenv("TP_SCALE","[(1.0,0.5),(2.0,0.25)]")  # list of (R-multiple, fraction)
NEWS_ENABLE      = os.getenv("NEWS_ENABLE","true").lower() in ("1","true","yes","on")
def parse_tp_scale(env_val: str):
    try:
        arr = json.loads(env_val)
        out = []
        for r_mult, frac in arr:
            r_mult = float(r_mult); frac = float(frac)
            if r_mult > 0 and 0 < frac < 1:
                out.append((r_mult, frac))
        if out:
            return out
    except Exception:
        pass
    return [(1.0,0.5),(2.0,0.25)]

DEFAULT_TP_SCALE = parse_tp_scale(TP_SCALE_ENV)

NEWS_REFRESH_SEC = int(os.getenv("NEWS_REFRESH_SEC","300"))
NEWS_MAX_ITEMS   = int(os.getenv("NEWS_MAX_ITEMS","60"))

# --- Trading runtime config (override-able via UI) ---
LIVE_TRADING    = os.getenv("LIVE_TRADING", "false").lower() in ("1","true","yes","on")
LEVERAGE_BASE   = 5
PLOT_BARS       = int(os.getenv("PLOT_BARS", "320"))  # jumlah bar yang dirender (perhitungan masih pakai frame penuh)
BASE_ORDER_USDT = float(os.getenv("BASE_ORDER_USDT", "25"))  # notional per entry
RISK_PCT        = float(os.getenv("RISK_PCT", "0.005"))       # risk mgmt (tidak dipakai agresif di versi ini)
ENTRY_CONF      = int(os.getenv("ENTRY_CONF", "60"))          # min confidence %


TP_ATR_MULT     = float(os.getenv("TP_ATR_MULT", "1.5"))
SL_ATR_MULT     = float(os.getenv("SL_ATR_MULT", "1.0"))

# --- Accuracy / Filter knobs (to improve selectivity & win rate) ---
ADX_MIN = float(os.getenv("ADX_MIN","18"))                    # min trend strength to allow trend trades
SQUEEZE_MIN_BBWIDTH = float(os.getenv("SQUEEZE_MIN_BBWIDTH","0.005"))  # skip when BB squeeze is too tight
P_UP_MIN_BUY = float(os.getenv("P_UP_MIN_BUY","0.58"))        # min up-probability for BUY
P_UP_MAX_SELL = float(os.getenv("P_UP_MAX_SELL","0.42"))      # max up-probability for SELL
ATR_PCT_MIN = float(os.getenv("ATR_PCT_MIN","0.0012"))        # skip dead markets (very low ATR%)
ATR_PCT_MAX = float(os.getenv("ATR_PCT_MAX","0.04"))          # skip erratic spikes (too high ATR%)
BREAKOUT_N = int(os.getenv("BREAKOUT_N","3"))                 # confirm breakout with last N bars
ENTRY_COOLDOWN_SEC = int(os.getenv("ENTRY_COOLDOWN_SEC","90"))# avoid rapid re-entries
RISK_AVERSION   = float(os.getenv("RISK_AVERSION", "0.5"))  # 0..1, makin besar = entry lebih ketat & risk lebih konservatif

# Dynamic margin & leverage controls
FULL_MARGIN      = os.getenv("FULL_MARGIN","false").lower() in ("1","true","yes","on","max","full")
DYNAMIC_LEVERAGE = os.getenv("DYNAMIC_LEVERAGE","true").lower() in ("1","true","yes","on")
MAX_LEV_FALLBACK = int(os.getenv("MAX_LEV_FALLBACK","50"))  # used if market metadata has no leverage info
MARGIN_USE_PCT   = float(os.getenv("MARGIN_USE_PCT","0.95")) # when FULL_MARGIN, use this fraction of available USDT

# Order Block influence (soft only): adjusts confidence, never gates entries
OB_SOFT = True  # keep True to use OB as confidence feature
OB_PAD_ATR = float(os.getenv("OB_PAD_ATR", "0.30"))  # pad around OB zone in ATR units
OB_BONUS = int(os.getenv("OB_BONUS", "8"))           # +/- confidence points when near an OB

# runtime state (in-memory)
STATE = {
    "pos": None,
    "live_toggle": (["on"] if LIVE_TRADING else []),
    "watch": (None, None),       # (symbol, timeframe) yang diminta UI
    "bg_alive": False,           # status background fetcher
    "ai_cache": {},              # {(symbol, tf, candle_ts_sec): teks}
    "news_alive": False,         # status news fetcher
    "news_pulse": {              # latest news/macro snapshot
        "ts": 0,
        "politics_sent": 0.0,    # -1..+1
        "crypto_sent": 0.0,      # -1..+1
        "rate_bias": "uncertain", # cut/hold/hike/uncertain
        "rate_prob": None,       # 0..1 if available
        "highlights": []         # list[str] of notable crypto-positive headlines
    },
    "yrange": {},           # smoothed y-axis cache per (symbol|tf)
    "fig_cache": {},       # cached figures per (symbol|tf) with signature to avoid rebuilds
    "ob_cache": {},         # small cache for OB zones keyed by last bar ts
    "perf": {"equity_peak": None, "equity_now": None, "drawdown_pct": 0.0,
         "trades": 0, "wins": 0, "losses": 0,
         "actions": {"BUY": {"ok": 0, "total": 0}, "SELL": {"ok": 0, "total": 0}}},
}

GC_COUNTER = 0
SAVE_COUNTER = {}

def prune_caches(light: bool = True):
    """Enhanced cache pruning with Elliott Wave cache management."""
    # fig_cache - more aggressive pruning
    try:
        fc = STATE.setdefault("fig_cache", {})
        if len(fc) > FIG_CACHE_MAX:
            # Remove oldest 50% of entries for better performance
            drop_n = max(1, len(fc) // 2)
            for k in list(fc.keys())[:drop_n]:
                fc.pop(k, None)
    except Exception:
        pass
    
    # ob_cache - more aggressive pruning 
    try:
        oc = STATE.setdefault("ob_cache", {})
        if len(oc) > OBJ_CACHE_MAX:
            drop_n = max(1, len(oc) // 2) 
            for k in list(oc.keys())[:drop_n]:
                oc.pop(k, None)
    except Exception:
        pass
        
    # ticker cache - keep only recent entries
    try:
        if len(TICKER_CACHE) > TICKER_CACHE_MAX:
            current_time = time.time()
            # Remove entries older than 5 minutes
            expired_keys = [k for k, v in TICKER_CACHE.items() 
                          if current_time - v.get("ts", 0) / 1000 > 300]
            for k in expired_keys:
                TICKER_CACHE.pop(k, None)
            
            # If still too many, remove oldest
            if len(TICKER_CACHE) > TICKER_CACHE_MAX:
                keys = sorted(TICKER_CACHE.keys(), key=lambda k: TICKER_CACHE[k].get("ts", 0))
                for k in keys[:len(TICKER_CACHE) - TICKER_CACHE_MAX]:
                    TICKER_CACHE.pop(k, None)
    except Exception:
        pass
        
    # Elliott Wave cache pruning
    try:
        global EW_CACHE
        current_time = time.time()
        expired_ew = [k for k, v in EW_CACHE.items() if current_time - v[1] > EW_CACHE_TTL]
        for k in expired_ew:
            EW_CACHE.pop(k, None)
    except Exception:
        pass
        
    # ATR cache pruning (prevent memory leak)
    try:
        if len(ATR_CACHE) > 20:
            # Remove oldest half
            keys_to_remove = list(ATR_CACHE.keys())[:len(ATR_CACHE)//2]
            for k in keys_to_remove:
                ATR_CACHE.pop(k, None)
    except Exception:
        pass

def tick_gc():
    """Enhanced garbage collection with memory monitoring."""
    if not GC_TUNE:
        return
    global GC_COUNTER
    GC_COUNTER += 1
    
    # More frequent cache pruning
    if GC_COUNTER % (GC_EVERY // 3) == 0:
        prune_caches(light=True)
    
    # Full GC less frequently
    if GC_COUNTER % GC_EVERY == 0:
        try:
            # Force garbage collection for all generations
            gc.collect(0)
            gc.collect(1) 
            gc.collect(2)
            
            if DEBUG_LOG:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                print(f"[GC] Memory usage: {memory_mb:.1f} MB")
        except Exception as e:
            if DEBUG_LOG:
                print(f"[GC] Error: {e}")

def perf_init():
    p = STATE.get("perf")
    if not isinstance(p, dict):
        STATE["perf"] = {"equity_peak": None, "equity_now": None, "drawdown_pct": 0.0,
                         "trades": 0, "wins": 0, "losses": 0,
                         "actions": {"BUY": {"ok": 0, "total": 0}, "SELL": {"ok": 0, "total": 0}}}

def perf_update_equity(equity_now: float | None):
    perf_init()
    if equity_now is None:
        return
    p = STATE["perf"]
    try:
        eq = float(equity_now)
    except Exception:
        return
    p["equity_now"] = eq
    if p.get("equity_peak") is None or eq > float(p.get("equity_peak")):
        p["equity_peak"] = eq
    peak = float(p.get("equity_peak") or eq)
    dd = 0.0 if peak <= 0 else max(0.0, (peak - eq) / peak * 100.0)
    p["drawdown_pct"] = dd

def perf_log_action(action: str, success: bool | None):
    perf_init()
    a = STATE["perf"]["actions"].setdefault(action, {"ok": 0, "total": 0})
    a["total"] += 1
    if success:
        a["ok"] += 1

def perf_log_trade(pnl_usdt: float):
    perf_init()
    p = STATE["perf"]
    p["trades"] += 1
    if pnl_usdt >= 0:
        p["wins"] += 1
    else:
        p["losses"] += 1

def perf_snapshot():
    perf_init()
    p = STATE["perf"]
    wr_pos = (p["wins"] / max(1, p["trades"])) * 100.0
    a_buy = p["actions"].get("BUY", {"ok": 0, "total": 0})
    a_sell = p["actions"].get("SELL", {"ok": 0, "total": 0})
    wr_act_buy = (a_buy["ok"] / max(1, a_buy["total"])) * 100.0
    wr_act_sell = (a_sell["ok"] / max(1, a_sell["total"])) * 100.0
    return {
        "trades": p["trades"],
        "wins": p["wins"],
        "losses": p["losses"],
        "winrate_pos": wr_pos,
        "winrate_action_buy": wr_act_buy,
        "winrate_action_sell": wr_act_sell,
        "drawdown_pct": float(p.get("drawdown_pct") or 0.0),
    }
    
# -------------------- AI Predict Target --------------------
def sanitize_ai_text(text: str) -> str:
    """
    Singkatkan keluaran AI agar tidak mengulang metrik yang sudah tampil di tiles.
    Aman kalau input bukan string.
    """
    if not isinstance(text, str) or not text:
        return text
    import re
    # buang potongan "Key=Number" yang bersifat metrik
    text = re.sub(
        r"\b(Close|EMA20|EMA50|MACD(?:_hist)?|RSI|StochK|StochD|BB ?Mid|ATR|Volume|HTF Trend)"
        r"\s*=\s*[-+]?\d*\.?\d+(?:e[-+]?\d+)?",
        "", text, flags=re.I
    )
    # rapikan separator & spasi ganda
    text = re.sub(r"(;\s*){2,}", "; ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip(" ;,")
    return text or "—"
    
def compose_top_metrics(symbol: str, tf: str, di: pd.DataFrame, pos: dict | None) -> str:
    # Balance
    avail = fetch_usdt_futures_balance("available")
    eq    = fetch_usdt_futures_balance("equity")

    # Perf snapshot
    ps = perf_snapshot()

    # News pulse
    pulse = STATE.get("news_pulse", {}) or {}
    pol_s = float(pulse.get("politics_sent") or 0.0)
    cry_s = float(pulse.get("crypto_sent") or 0.0)
    rbias = str(pulse.get("rate_bias") or "uncertain")
    rprob = pulse.get("rate_prob")
    hi    = pulse.get("highlights") or []

    # Price + ROE
    last = float(di["close"].iloc[-1]) if di is not None and not di.empty else float("nan")
    roe  = None
    if pos and np.isfinite(last):
        roe = compute_roe(pos, last)

    # Direction probability (heuristic)
    try:
        p_up, _ = ai_predict_direction(di)
    except Exception:
        p_up = None

    # Build compact, non-duplicative lines
    lines = []
    lines.append(f"**{symbol} @ {tf}**")
    if avail is not None and eq is not None:
        lines.append(f"Equity ${eq:.2f} | Avail ${avail:.2f}")
    else:
        lines.append("Balance: (fetching...)")

    # Merge winrate + actions on one line (avoid duplication)
    lines.append(
        f"Winrate {ps['winrate_pos']:.1f}% | BUY {ps['winrate_action_buy']:.1f}% | SELL {ps['winrate_action_sell']:.1f}% • Trades {ps['trades']} • DD {ps['drawdown_pct']:.1f}%"
    )

    if roe is not None:
        lines.append(f"ROE {roe:+.2f}%")
    if p_up is not None:
        lines.append(f"P(up) {p_up*100:.1f}%")

    # Condense news sentiment to one line and optional 1-3 headlines
    rate_txt = rbias + (f" {int(rprob*100)}%" if rprob is not None else "")
    lines.append(f"Sentiment: crypto {cry_s:+.2f} | politics {pol_s:+.2f} | rates {rate_txt}")
    for t in hi[:3]:
        lines.append(f"• {t}")

    return "\n".join(lines)

# ---- UI helper: render position & progress text, and live position snapshot ----
def render_pos_text(symbol: str, pos: dict | None) -> str:
    if not isinstance(pos, dict) or not pos:
        return "—"
    try:
        side = str(pos.get("side","")).upper() or "—"
        amt  = float(pos.get("amount") or 0)
        entry= float(pos.get("entry") or 0)
        tp   = float(pos.get("tp") or (pos.get("tp_calc") or 0))
        sl   = float(pos.get("sl") or (pos.get("sl_calc") or 0))
        lev  = float(pos.get("lev") or 0)
        parts = [f"{side} {symbol} x{int(lev) if lev else '-'}", f"Size {amt:g}"]
        if entry: parts.append(f"Entry {entry:.6g}")
        if tp:    parts.append(f"TP {tp:.6g}")
        if sl:    parts.append(f"SL {sl:.6g}")
        return " • ".join(parts)
    except Exception:
        return "—"

def _r_multiple(side: str, entry: float, sl: float, price: float) -> float | None:
    try:
        side = (side or "").lower()
        if not all(np.isfinite(x) for x in (entry, sl, price)): return None
        risk = (entry - sl) if side == "long" else (sl - entry)
        move = (price - entry) if side == "long" else (entry - price)
        if risk <= 0: return None
        return float(move / risk)
    except Exception:
        return None

def render_progress_text(pos: dict | None, last_price: float | None) -> str:
    if not isinstance(pos, dict) or not pos or last_price is None or not np.isfinite(last_price):
        return "—"
    try:
        roe = compute_roe(pos, float(last_price))
        rm  = _r_multiple(str(pos.get("side")), float(pos.get("entry") or 0), float(pos.get("sl") or pos.get("sl_calc") or 0), float(last_price))
        out = []
        if roe is not None:
            out.append(f"ROE {roe:+.2f}%")
        if rm is not None:
            out.append(f"{rm:.2f}R")
        return " • ".join(out) if out else "—"
    except Exception:
        return "—"

def get_position_snapshot(symbol: str) -> dict | None:
    """
    Fetch live futures position via CCXT and normalize to our internal dict.
    Returns dict: {'side','amount','entry','tp','sl','lev','unrealized','margin'} or None.
    """
    try:
        # Throttle a bit using POS_CACHE
        ent = POS_CACHE.get("data", {}).get(symbol)
        now = now_ms()
        if isinstance(ent, dict) and (now - int(ent.get("ts",0)) < 900):
            return ent.get("pos")

        lst = EX.fetch_positions([symbol])
        pos = None
        for p in lst or []:
            amt = num(p.get("contracts") or (p.get("info") or {}).get("total") or p.get("contractsAmount"))
            side = str(p.get("side") or "").lower()
            if amt is None:
                amt = num((p.get("info") or {}).get("holdVol"))
            if amt is None:
                continue
            amt = float(amt)
            if abs(amt) <= 0:
                continue
            if not side:
                side = "long" if amt > 0 else "short"

            info = p.get("info") or {}
            entry = num(p.get("entryPrice") or info.get("avgPrice") or info.get("averageOpenPrice"))
            tp    = num(p.get("takeProfit") or info.get("takeProfitPrice") or info.get("tp"))
            sl    = num(p.get("stopLoss") or info.get("stopLossPrice") or info.get("sl"))
            lev   = num(p.get("leverage") or info.get("leverage"))
            unreal= num(p.get("unrealizedPnl") or info.get("unrealizedPL") or info.get("unrealizedPnl"))
            margin= num(p.get("initialMargin") or info.get("margin") or info.get("marginBalance"))

            pos = {
                "side": side, "amount": abs(float(amt)), "entry": entry or 0.0,
                "tp": tp or None, "sl": sl or None, "lev": lev or LEVERAGE_BASE,
                "unrealized": unreal, "margin": margin
            }
            break  # first non-zero position
        POS_CACHE["data"][symbol] = {"ts": now, "pos": pos}
        return pos
    except Exception as e:
        print(f"[POS] warn: {e}")
        return None

def prepare_ui_snap(symbol: str, tf: str, di: pd.DataFrame) -> tuple[dict | None, str, str, str]:
    """
    Returns (pos_dict, pos_text, progress_text, ov_metrics_text)
    Ensures TP/SL fallback is computed when exchange doesn't provide it.
    """
    pos = get_position_snapshot(symbol)
    # Compute fallback TP/SL if missing
    try:
        if pos:
            hti = None
            try:
                a_tf = anchor_tf(tf)
                if a_tf and a_tf != tf:
                    hti, _ = ensure_clean_tf_df(symbol, a_tf)
            except Exception:
                hti = None
            tp_calc, sl_calc, rr_est = derive_tp_sl_mtf(di, hti, pos.get("side","long"), tf)
            if not pos.get("tp"):
                pos["tp"] = tp_calc
                pos["tp_calc"] = tp_calc
            if not pos.get("sl"):
                pos["sl"] = sl_calc
                pos["sl_calc"] = sl_calc
            pos["rr"] = float(rr_est or 0.0)
    except Exception:
        pass

    last = float(di["close"].iloc[-1]) if di is not None and not di.empty else None
    pos_text = render_pos_text(symbol, pos)
    prog_text = render_progress_text(pos, last)
    ov_text = compose_top_metrics(symbol, tf, di, pos)
    return pos, pos_text, prog_text, ov_text

# pos dict: {"side": "long"/"short", "amount": float, "entry": float, "tp": float, "sl": float}

def bitget():
    ex = ccxt.bitget({
        "apiKey": BITGET_KEY,
        "secret": BITGET_SECRET,
        "password": BITGET_PASSWORD,
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",
            "defaultSubType": "umcbl",
            "defaultSettle": "USDT",
            "defaultMarket": "swap",
        },
        "timeout": 10000
    })
    ex.load_markets()
    return ex

EX = bitget()

# --- Monotonic clock helper (avoid try/except spam on EX.milliseconds) ---
def now_ms() -> int:
    try:
        return int(EX.milliseconds())
    except Exception:
        return int(time.time() * 1000)

# --- Reusable HTTP session with retries (cuts latency & 429s) ---
HTTP = requests.Session()
_retry = Retry(
    total=3,
    backoff_factor=0.3,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
_adapter = HTTPAdapter(max_retries=_retry)
HTTP.mount("https://", _adapter)
HTTP.mount("http://", _adapter)

# -------------------- Background OHLCV fetcher (non-blocking) --------------------
def ensure_bg_fetcher():
    if STATE.get("bg_alive"):
        return
    def _bg():
        while True:
            sym, tf = STATE.get("watch", (None, None))
            if not sym or not tf:
                time.sleep(0.8)
                continue
            try:
                # muat cache
                df, _ = load_cache(sym, tf)
                # backfill awal
                if df.empty or len(df) < 300:
                    df2 = fetch_initial_bars(sym, tf, target=600)
                    if not df2.empty:
                        save_cache(sym, tf, df2)
                else:
                    # incremental
                    last_ms = int(df["datetime"].max().timestamp() * 1000)
                    inc = fetch_incremental_bars(sym, tf, last_ms)
                    if not inc.empty:
                        df2 = _concat_dedup(df, inc).tail(1000)
                        save_cache(sym, tf, df2)
            except Exception as e:
                print(f"[BG] fetch warn: {e}")
            time.sleep(1.0)  # ~1 dtk
    t = threading.Thread(target=_bg, daemon=True)
    t.start()
    STATE["bg_alive"] = True

# ---- Simple on-disk + in-memory OHLCV cache ----
CACHE = {}  # key -> {"df": DataFrame, "ts": last_ts_ms, "last_update": ms}

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# futures balance cache/throttle (avoid 429)
BAL_CACHE = {"equity": None, "avail": None, "ts": 0, "cooldown_until": 0}
# Precision/ticker/position caches for speed
PREC_CACHE = {"amount": {}, "price": {}}
TICKER_CACHE = {}  # {symbol: {"ts": ms, "val": {"last": float|None, "mark": float|None}}}
POS_CACHE = {"data": {}}  # per-symbol throttled live position snapshots

# --- Extra caches ---
LEV_CACHE = {}   # remember last leverage set per symbol
ATR_CACHE = {}   # cache ATR series by (id(df), n, last_ts)

def _cache_key(symbol: str, timeframe: str) -> str:
    return f"{symbol}|{timeframe}"

def _cache_fp(symbol: str, timeframe: str) -> str:
    safe = symbol.replace("/", "").replace(":", "_")
    return os.path.join(CACHE_DIR, f"{safe}_{timeframe}.csv")

def _log(msg: str):
    if DEBUG_LOG:
        try:
            print(msg)
        except Exception:
            pass
        
def num(x):
    """Best-effort float parser; returns None if not parseable."""
    try:
        if x is None: return None
        if isinstance(x, (int, float)): return float(x)
        if isinstance(x, str) and x.strip() != "": return float(x)
    except Exception:
        pass
    return None

def _downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast OHLCV/indicator float columns to save RAM. Safe for empty/None."""
    if df is None or getattr(df, "empty", True):
        return df
    cols = [c for c in ("open","high","low","close","volume",
                        "ema20","ema50","rsi",
                        "macd_line","macd_signal","macd_hist",
                        "stoch_k","stoch_d",
                        "bb_mid","bb_upper","bb_lower","bb_width", "adx","vwap","obv","adl","psar","williams_r","vwma20","vwma50","hma20","hma50",
                        "ppo","ppo_signal","ppo_hist","cmf","std20","aroon_up","aroon_down","aroon_osc",
                        "rvi","roc12","vroc","tsi","ibs","ara","arb","pump_score")
            if c in df.columns]
    for c in cols:
        try:
            if DF_FLOAT_DTYPE == "float32":
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32, copy=False)
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(float, copy=False)
        except Exception:
            pass
    return df

# --- Robust tail helper ---
def stable_tail(df: pd.DataFrame, n: int | float) -> pd.DataFrame:
    try:
        k = int(n or 0)
    except Exception:
        k = 0
    if not hasattr(df, "__len__"):
        return df
    L = len(df)
    if k <= 0:
        return df.iloc[0:0].copy()
    if k >= L:
        return df.copy()
    return df.iloc[L - k : L].copy()

def _smooth_y_range(symbol: str, tf: str, y_min: float, y_max: float, pad: float = 0.02) -> tuple[float, float]:
    """Persist y-range per (symbol|tf) untuk mencegah chart jumping. Hanya melebarkan jika perlu."""
    try:
        if (not np.isfinite(y_min)) or (not np.isfinite(y_max)) or (y_max <= y_min):
            return float(y_min), float(y_max)
    except Exception:
        return float(y_min), float(y_max)
    span = y_max - y_min
    y0 = y_min - pad * span
    y1 = y_max + pad * span
    key = f"{symbol}|{tf}"
    store = STATE.setdefault("yrange", {})
    old = store.get(key)
    if old:
        o0, o1 = old
        # jika range baru masih muat dalam range lama (dengan toleransi kecil), pakai yang lama (hindari jitter)
        if y0 >= o0 * 0.999 and y1 <= o1 * 1.001:
            return o0, o1
        # kalau keluar, lebarkan agar tidak mengecil tiba-tiba
        y0 = min(y0, o0)
        y1 = max(y1, o1)
    store[key] = (float(y0), float(y1))
    # cap cache size to avoid unbounded growth when switching many symbols/TFs
    if len(store) > 50:
        drop_n = len(store) - 50
        for k in list(store.keys())[:drop_n]:
            try:
                if k != key:
                    store.pop(k, None)
            except Exception:
                pass
    return float(y0), float(y1)

def get_ob_zones_cached(di: pd.DataFrame, tf: str) -> list:
    """Cache OB berdasarkan state candle terakhir (ts,len,close) agar tidak bentrok lintas simbol & menghindari recompute tiap detik di bar yang sama."""
    if di is None or di.empty:
        return []
    try:
        last_ts = int(pd.to_datetime(di["datetime"].iloc[-1]).timestamp())
        last_close = float(di["close"].iloc[-1])
    except Exception:
        last_ts = int(time.time())
        last_close = float("nan")
    key = f"{tf}|{len(di)}|{last_ts}|{last_close:.8f}"
    cache = STATE.setdefault("ob_cache", {})
    if key in cache:
        return cache[key]
    zones = find_order_blocks(di)
    cache[key] = zones
    # trim to 24 entries (LRU-ish)
    if len(cache) > 24:
        for k in list(cache.keys())[: len(cache) - 24]:
            cache.pop(k, None)
    return zones

def load_cache(symbol: str, timeframe: str):
    key = _cache_key(symbol, timeframe)
    if key in CACHE:
        entry = CACHE[key]
        df = entry["df"].copy()
        ts = int(entry.get("ts", 0) or 0)
        return df, ts

    fp = _cache_fp(symbol, timeframe)
    if os.path.exists(fp):
        try:
            df = pd.read_csv(fp, usecols=["datetime","open","high","low","close","volume"])
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
            df = df.dropna(subset=["datetime"]).reset_index(drop=True)
            df = _downcast_df(df)
            ts = int(df["datetime"].max().timestamp() * 1000) if not df.empty else 0
            return df, ts
        except Exception as e:
            print(f"[CACHE] fail load {fp}: {e}")

    return pd.DataFrame(columns=["datetime","open","high","low","close","volume"]), 0

def save_cache(symbol: str, timeframe: str, df: pd.DataFrame):
    key = _cache_key(symbol, timeframe)
    df2 = df[["datetime","open","high","low","close","volume"]].copy(deep=False)
    df2["datetime"] = pd.to_datetime(df2["datetime"], utc=True)
    df2 = _downcast_df(df2)
    ts = int(df2["datetime"].max().timestamp()*1000) if not df2.empty else 0
    CACHE[key] = {"df": df2, "ts": ts, "last_update": now_ms()}

    # throttle penulisan file (kurangi alokasi & GC)
    cnt = SAVE_COUNTER.get(key, 0) + 1
    SAVE_COUNTER[key] = cnt
    if (cnt % (CACHE_WRITE_SKIP + 1)) == 0:
        df2.to_csv(_cache_fp(symbol, timeframe), index=False)
        _log(f"[CACHE] saved {len(df2)} bars for {symbol}@{timeframe}")

# -------------------- Professional Theme Configuration --------------------
# Advanced trading constants and hedge fund styling

ALLOWED_TF = ["1m","3m","5m","15m","30m","1h","2h","4h","1d"]

# Professional chart theme
PROFESSIONAL_THEME = {
    'template': 'plotly_dark',
    'paper_bgcolor': COLORS['primary'],
    'plot_bgcolor': COLORS['surface'],
    'font': {
        'family': "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
        'size': 12,
        'color': COLORS['text_primary']
    },
    'colorway': [COLORS['accent'], COLORS['success'], COLORS['danger'], COLORS['warning'], COLORS['info']]
}

@lru_cache(maxsize=64)
def timeframe_ms(tf: str) -> int:
    try:
        sec = EX.parse_timeframe(tf)
        return int(sec * 1000)
    except Exception:
        fallback = {"1m":60,"3m":180,"5m":300,"15m":900,"30m":1800,"1h":3600,"2h":7200,"4h":14400,"1d":86400}
        return int(fallback.get(tf, 60) * 1000)

@lru_cache(maxsize=64)
def bitget_granularity(tf: str) -> str | None:
    tf = (tf or "").lower()
    m = {
        "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "12h": "12H",
        "1d": "1D", "3d": "3D", "1w": "1W", "1mth": "1M"
    }
    return m.get(tf)

# --- Anchor timeframe mapping (HTF for MTF bias) ---
@lru_cache(maxsize=64)
def anchor_tf(tf: str) -> str:
    m = {
        "1m": "5m", "3m": "15m", "5m": "15m",
        "15m": "1h", "30m": "2h", "1h": "4h",
        "2h": "4h", "4h": "1d", "1d": "1d"
    }
    return m.get(tf, "1h")

# --- Bitget raw HTTP (v2) ---
def fetch_ohlcv_bitget_raw(symbol: str, timeframe: str = "5m", limit: int = 200, start_ms: int | None = None, end_ms: int | None = None) -> pd.DataFrame:
    try:
        m = EX.market(symbol)
        base_id = (m.get("id") or "").replace("_UMCBL", "").replace("_CMCBL", "").strip()
    except Exception:
        base_id = symbol.replace(":USDT", "").replace("/", "").strip()

    gran = bitget_granularity(timeframe)
    if not gran:
        return pd.DataFrame(columns=["datetime","open","high","low","close","volume"])

    url = "https://api.bitget.com/api/v2/mix/market/history-candles"
    params = {
        "symbol": base_id,
        "granularity": gran,
        "productType": "USDT-FUTURES",
        "limit": min(max(int(limit), 50), 200)
    }
    if start_ms is not None:
        params["startTime"] = int(start_ms)
    if end_ms is not None:
        params["endTime"] = int(end_ms)

    try:
        r = HTTP.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        arr = data if isinstance(data, list) else data.get("data") or []
        if not arr:
            return pd.DataFrame(columns=["datetime","open","high","low","close","volume"])
        rows = []
        for row in arr:  # newest-first: [ts, o,h,l,c, baseVol, quoteVol]
            ts = int(float(row[0]))
            o,h,l,c = map(float, row[1:5])
            v = float(row[5]) if len(row) > 5 else float("nan")
            rows.append([ts,o,h,l,c,v])
        rows.sort(key=lambda x: x[0])
        df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df[["datetime","open","high","low","close","volume"]]
        for col in ["open","high","low","close","volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        return _downcast_df(df)
    except Exception as e:
        print(f"[OHLCV][raw v2] fail {symbol}@{timeframe}: {e}")
        return pd.DataFrame(columns=["datetime","open","high","low","close","volume"])

def _concat_dedup(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    if a is None or a.empty:
        return b.copy()
    if b is None or b.empty:
        return a.copy()
    out = pd.concat([a, b], ignore_index=True)
    out = out.drop_duplicates(subset=["datetime"], keep="last").sort_values("datetime").reset_index(drop=True)
    return out

def fetch_initial_bars(symbol: str, timeframe: str, target: int = 600) -> pd.DataFrame:
    acc = pd.DataFrame(columns=["datetime","open","high","low","close","volume"])
    end_ms = None
    pages = 0
    while len(acc) < target and pages < 10:
        chunk = fetch_ohlcv_bitget_raw(symbol, timeframe, limit=200, end_ms=end_ms)
        if chunk.empty:
            break
        chunk = _downcast_df(chunk)
        acc = _concat_dedup(chunk, acc)
        end_ms = int(chunk["datetime"].min().timestamp()*1000) - 1
        pages += 1
    if not acc.empty:
        _log(f"[OHLCV] backfilled(v2) {len(acc)} bars for {symbol}@{timeframe} in {pages} page(s)")
        return _downcast_df(acc.tail(target))
    # ccxt fallback (one-shot)
    try:
        rows = EX.fetch_ohlcv(symbol, timeframe=timeframe, limit=min(target, 300))
        df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df[["datetime","open","high","low","close","volume"]]
        for col in ["open","high","low","close","volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
        _log(f"[OHLCV] backfilled(ccxt) {len(df)} bars for {symbol}@{timeframe}")
        return _downcast_df(df)
    except Exception as e:
        print(f"[OHLCV] ccxt initial fail {symbol}@{timeframe}: {e}")
        return acc

def fetch_incremental_bars(symbol: str, timeframe: str, last_ts_ms: int) -> pd.DataFrame:
    inc = fetch_ohlcv_bitget_raw(symbol, timeframe, limit=200, start_ms=last_ts_ms + 1)
    if not inc.empty:
        return inc
    try:
        rows = EX.fetch_ohlcv(symbol, timeframe=timeframe, since=last_ts_ms + 1, limit=300)
        if rows:
            df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
            df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            df = df[["datetime","open","high","low","close","volume"]]
            for col in ["open","high","low","close","volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
            return _downcast_df(df)
    except Exception as e:
        print(f"[OHLCV] ccxt since fail {symbol}@{timeframe}: {e}")
    return pd.DataFrame(columns=["datetime","open","high","low","close","volume"])

def fetch_ohlcv_df(symbol: str, timeframe: str="1m", limit: int=600) -> pd.DataFrame:
    target = min(int(limit), 1000)
    key = _cache_key(symbol, timeframe)

    df, _ = load_cache(symbol, timeframe)
    if df.empty:
        df = fetch_initial_bars(symbol, timeframe, target=target)
        if not df.empty:
            save_cache(symbol, timeframe, df)
        return _downcast_df(df.tail(target))

    now = now_ms()
    last_update = int(CACHE.get(key, {}).get("last_update", 0))
    if now - last_update < 900:   # throttle ~0.9s
        return _downcast_df(df.tail(target))

    last_ts_ms = int(df["datetime"].max().timestamp()*1000)
    inc = fetch_incremental_bars(symbol, timeframe, last_ts_ms)
    if not inc.empty:
        df = _concat_dedup(df, inc).tail(target)
        save_cache(symbol, timeframe, df)
        _log(f"[OHLCV] updated +{len(inc)} bars → total {len(df)} for {symbol}@{timeframe}")
        return _downcast_df(df)

    CACHE[key]["last_update"] = now_ms()
    return _downcast_df(df.tail(target))

def ensure_clean_tf_df(symbol: str, tf: str, min_bars: int = 200, max_age_mult: int = 2) -> tuple[pd.DataFrame, str]:
    """
    Pastikan data chart sesuai TF & cukup bersih.
    - Jika cache kosong/kurang/stale: coba incremental, jika gagal → full backfill.
    - Return (df, status) di mana status: "ok"|"inc"|"reloaded"|"stale"
    """
    per_ms = timeframe_ms(tf)
    df, _ = load_cache(symbol, tf)

    def _is_bad(d: pd.DataFrame) -> bool:
        if d is None or d.empty: return True
        if len(d) < min_bars: return True
        try:
            last_ts = int(pd.to_datetime(d["datetime"].iloc[-1]).timestamp() * 1000)
        except Exception:
            return True
        lag = now_ms() - last_ts
        if lag > max_age_mult * per_ms:  # bar terakhir terlalu tua dibanding TF
            return True
        if not d["datetime"].is_monotonic_increasing:
            return True
        return False

    if not _is_bad(df):
        return df, "ok"

    # coba incremental
    if df is not None and not df.empty:
        try:
            last_ts = int(df["datetime"].max().timestamp() * 1000)
            inc = fetch_incremental_bars(symbol, tf, last_ts)
            if not inc.empty:
                df = _concat_dedup(df, inc).tail(max(min_bars, 1000))
                save_cache(symbol, tf, df)
                if not _is_bad(df):
                    return df, "inc"
        except Exception:
            pass

    # full backfill terakhir
    df2 = fetch_initial_bars(symbol, tf, target=max(min_bars, 600))
    if not df2.empty:
        save_cache(symbol, tf, df2)
        return df2, "reloaded"

    # fallback: kembalikan apa adanya, tandai stale
    return (df if df is not None else pd.DataFrame(columns=["datetime","open","high","low","close","volume"])), "stale"

def fetch_ticker_fast(symbol: str, ttl_ms: int | None = None) -> dict:
    """
    Cached ticker to avoid hammering the exchange every tick.
    Returns {"last": float|None, "mark": float|None}.
    """
    if ttl_ms is None:
        ttl_ms = int(os.getenv("TICKER_TTL_MS", "500"))
    try:
        now = now_ms()
    except Exception:
        now = int(time.time()*1000)
    ent = TICKER_CACHE.get(symbol)
    if ent and (now - int(ent.get("ts", 0)) < ttl_ms):
        return ent["val"]
    try:
        t = EX.fetch_ticker(symbol)
        val = {
            "last": num(t.get("last") or t.get("close")),
            "mark": num((t.get("info") or {}).get("markPrice") or t.get("mark")),
        }
    except Exception:
        val = {"last": None, "mark": None}
    TICKER_CACHE[symbol] = {"ts": now, "val": val}
    prune_caches(light=True)
    return val

def inject_last_price(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    try:
        tk = fetch_ticker_fast(symbol, ttl_ms=500)
        lp = tk.get("mark") or tk.get("last") or 0.0
        lp = float(lp or 0.0)
        if lp <= 0 or df is None or df.empty:
            return _downcast_df(df)

        d = df.copy(deep=False)
        i = d.index[-1]
        o = float(d.loc[i, "open"])
        h = float(d.loc[i, "high"])
        l = float(d.loc[i, "low"])
        c_new = lp

        # jaga konsistensi candle: high/low harus mengurung open & close
        h = max(h, o, c_new)
        l = min(l, o, c_new)
        d.loc[i, "close"] = c_new
        d.loc[i, "high"]  = h
        d.loc[i, "low"]   = l
        return _downcast_df(d)
    except Exception:
        return _downcast_df(df)

# --- Futures balance helper (USDT-M) ---
def fetch_usdt_futures_balance(kind: str = "equity") -> float | None:
    """Return USDT-M futures balance.
    kind: "equity" (mark-to-market, includes PnL) or "available" (free USDT).
    Caches results and backs off on 429s to avoid breaking the UI.
    """
    refresh_ms = int(os.getenv("BAL_REFRESH_MS", "3000"))
    debug = os.getenv("BAL_DEBUG", "false").lower() in ("1","true","yes","on")

    try:
        now = now_ms()
    except Exception:
        now = int(time.time()*1000)

    # hard backoff if we recently hit 429
    if now < int(BAL_CACHE.get("cooldown_until", 0)):
        return BAL_CACHE.get("equity") if kind=="equity" else BAL_CACHE.get("avail")

    # throttle reads
    if now - int(BAL_CACHE.get("ts", 0)) < refresh_ms:
        return BAL_CACHE.get("equity") if kind=="equity" else BAL_CACHE.get("avail")

    params_try = [
        {"type": "swap", "productType": "USDT-FUTURES", "marginCoin": "USDT"},
        {"type": "swap", "productType": "USDT-FUTURES"},
        {"type": "swap"},
        {},
    ]

    equity_val, avail_val = None, None

    for prm in params_try:
        try:
            bal = EX.fetch_balance(prm)
            if debug:
                try:
                    print("[BAL][dbg] params=", prm, "type=", type(bal).__name__, "keys=", list(bal.keys())[:10] if isinstance(bal, dict) else "-")
                except Exception:
                    pass
            # ccxt standard bucket
            if isinstance(bal, dict):
                # Prefer v2 data list
                info = bal.get("info") if isinstance(bal.get("info"), (dict, list)) else None
                data_list = None
                if isinstance(info, dict) and isinstance(info.get("data"), list):
                    data_list = info.get("data")
                elif isinstance(info, list):
                    data_list = info
                if isinstance(data_list, list):
                    for acct in data_list:
                        if not isinstance(acct, dict):
                            continue
                        if str(acct.get("marginCoin", "")).upper() != "USDT":
                            continue
                        # Try equity fields
                        for k in ("equity","usdtEquity","total","balance","cash"):
                            v = num(acct.get(k))
                            if v is not None:
                                equity_val = v
                                break
                        # Try available/free fields
                        for k in ("available","availableBalance","free","usdtBalance","maxTransferOut"):
                            v = num(acct.get(k))
                            if v is not None:
                                avail_val = v
                                break
                        break  # first USDT account is enough

                if equity_val is None or avail_val is None:
                    bucket = bal.get("USDT") or bal.get("usdt")
                    if isinstance(bucket, dict):
                        if equity_val is None:
                            for k in ("equity","total","balance"):
                                v = num(bucket.get(k))
                                if v is not None:
                                    equity_val = v
                                    break
                        if avail_val is None:
                            for k in ("free","available"):
                                v = num(bucket.get(k))
                                if v is not None:
                                    avail_val = v
                                    break
            # If we got anything usable, stop trying further params
            if equity_val is not None or avail_val is not None:
                break
        except Exception as e:
            emsg = str(e)
            if "429" in emsg or "Too Many Requests" in emsg:
                BAL_CACHE["cooldown_until"] = now + 20000  # 20s cooldown
                break
            if debug:
                print(f"[BAL][dbg] fetch_balance error with {prm}: {e}")
            continue

    # Keep previous good numbers if new fetch failed
    if equity_val is None:
        equity_val = BAL_CACHE.get("equity")
    if avail_val is None:
        avail_val = BAL_CACHE.get("avail")

    try:
        perf_update_equity(equity_val)
    except Exception:
        pass

    BAL_CACHE.update({"equity": equity_val, "avail": avail_val, "ts": now})
    val = equity_val if kind=="equity" else avail_val
    try:
        return 0.0 if val is None else float(val)
    except Exception:
        return 0.0

# -------------------- News & Macro Sentiment (optional AI) --------------------
CRYPTO_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
]
POL_FEEDS = [
    "https://feeds.reuters.com/reuters/worldNews",
]
RATES_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
]

def _rss_fetch(url: str, limit: int = 40):
    try:
        r = HTTP.get(url, timeout=10)
        r.raise_for_status()
        return _rss_parse(r.text)[:limit]
    except Exception:
        return []

def _rss_parse(xml_text: str):
    out = []
    try:
        root = ET.fromstring(xml_text)
        for item in root.findall('.//item'):
            title = (item.findtext('title') or '').strip()
            link  = (item.findtext('link') or '').strip()
            pub   = (item.findtext('pubDate') or '').strip()
            if title:
                out.append({"title": title, "link": link, "pub": pub})
    except Exception:
        pass
    return out

def fetch_news_bundle(max_items: int = NEWS_MAX_ITEMS):
    items = []
    for u in (CRYPTO_FEEDS + POL_FEEDS + RATES_FEEDS):
        items.extend(_rss_fetch(u, limit=max_items//3))
    # de-dup by title
    seen = set(); dedup = []
    for it in items:
        t = it.get("title","")
        if t in seen: continue
        seen.add(t); dedup.append(it)
    return dedup[:max_items]

POS_WORDS_CRYPTO = [
    "approval","etf","adoption","partnership","upgrade","integration","listing","raises","funding",
    "launch","mainnet","reduce fees","halving","institutional","custody","adds support","sec approves"
]
NEG_WORDS_CRYPTO = [
    "hack","outage","lawsuit","ban","penalty","exploit","delist","selloff","fraud","insolvency","restrict"
]
POS_WORDS_POL = ["deal","peace","truce","ceasefire","agreement","talks progress"]
NEG_WORDS_POL = ["war","conflict","tension","sanction","protest","coup","unrest"]
CUT_WORDS  = ["rate cut","cuts rates","slashes rates","eases policy","easing","dovish"]
HIKE_WORDS = ["rate hike","raises rates","tightening","hawkish"]
HOLD_WORDS = ["holds rates","keeps rates","unchanged"]

def _kw_score(title: str, pos_list, neg_list):
    t = title.lower()
    score = 0
    for w in pos_list:
        if w in t: score += 1
    for w in neg_list:
        if w in t: score -= 1
    return score

def _heuristic_news_pulse(items):
    pol, cry = 0, 0
    pol_n, cry_n = 0, 0
    cuts, hikes, holds = 0, 0, 0
    hi = []
    for it in items:
        t = it.get("title","")
        if not t: continue
        # crypto
        cs = _kw_score(t, POS_WORDS_CRYPTO, NEG_WORDS_CRYPTO)
        if any(k in t.lower() for k in ["bitcoin","crypto","ethereum","token","defi","blockchain","sec","etf","binance","coinbase","solana"]):
            cry += cs; cry_n += 1
            if cs > 0 and len(hi) < 6:
                hi.append(t)
        # politics
        ps = _kw_score(t, POS_WORDS_POL, NEG_WORDS_POL)
        if any(k in t.lower() for k in ["election","president","parliament","sanction","war","conflict","government","policy"]):
            pol += ps; pol_n += 1
        # rates
        tl = t.lower()
        if any(w in tl for w in CUT_WORDS): cuts += 1
        if any(w in tl for w in HIKE_WORDS): hikes += 1
        if any(w in tl for w in HOLD_WORDS): holds += 1
    def _norm(x,n):
        if n<=0: return 0.0
        return max(-1.0, min(1.0, x/ max(1, n)))
    politics_sent = _norm(pol, pol_n)
    crypto_sent   = _norm(cry, cry_n)
    # decide rate bias
    if cuts>max(hikes,holds): rb = "cut"
    elif hikes>max(cuts,holds): rb = "hike"
    elif holds>max(cuts,hikes): rb = "hold"
    else: rb = "uncertain"
    total = cuts + hikes + holds
    prob = (max(cuts,hikes,holds) / total) if total>0 else None
    return {
        "ts": int(time.time()),
        "politics_sent": float(politics_sent),
        "crypto_sent": float(crypto_sent),
        "rate_bias": rb,
        "rate_prob": float(prob) if prob is not None else None,
        "highlights": hi[:5]
    }

def analyze_news_pulse_with_ai(items):
    if not OPENAI_API_KEY:
        return _heuristic_news_pulse(items)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        titles = [it.get("title","") for it in items if it.get("title")]
        titles = titles[:80]
        payload = {"titles": titles}
        rsp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role":"system","content":"Return JSON only with keys politics_sent (-1..1), crypto_sent (-1..1), rate_bias one of cut/hold/hike/uncertain, rate_prob 0..1, highlights (up to 5 crypto-positive headlines as strings)."},
                {"role":"user","content": json.dumps(payload, separators=(",",":"))}
            ],
            temperature=0.2,
            max_tokens=200,
        )
        raw = rsp.choices[0].message.content.strip()
        obj = json.loads(raw)
        out = _heuristic_news_pulse(items)
        out.update({
            "politics_sent": float(obj.get("politics_sent", out["politics_sent"])),
            "crypto_sent": float(obj.get("crypto_sent", out["crypto_sent"])),
            "rate_bias": str(obj.get("rate_bias", out["rate_bias"])) or out["rate_bias"],
            "rate_prob": float(obj.get("rate_prob")) if obj.get("rate_prob") is not None else out["rate_prob"],
            "highlights": list(obj.get("highlights", out["highlights"]))[:5]
        })
        return out
    except Exception:
        return _heuristic_news_pulse(items)

def ensure_news_fetcher():
    if not NEWS_ENABLE or STATE.get("news_alive"):
        return
    def _bg_news():
        while True:
            try:
                items = fetch_news_bundle(NEWS_MAX_ITEMS)
                pulse = analyze_news_pulse_with_ai(items)
                STATE["news_pulse"] = pulse
            except Exception as e:
                print(f"[NEWS] warn: {e}")
            # sleep persis NEWS_REFRESH_SEC detik
            time.sleep(max(1, int(NEWS_REFRESH_SEC)))
    t = threading.Thread(target=_bg_news, daemon=True)
    t.start()
    STATE["news_alive"] = True

# ---- Partial TP scale parsing + AI plan ----

def parse_roe_steps(env_val: str):
    try:
        arr = json.loads(env_val)
        out = []
        for thr, frac in arr:
            thr = float(thr); frac = float(frac)
            if thr > 0 and 0 < frac < 1:
                out.append((thr, frac))
        if out:
            return out
    except Exception:
        pass
    return [(50.0,0.25),(100.0,0.25),(200.0,0.25)]

DEFAULT_ROE_STEPS = parse_roe_steps(ROE_TP_STEPS_ENV)

def compute_roe(pos: dict, last_price: float) -> float | None:
    """Return ROE% (positive bagus). Utamakan unrealized/margin dari exchange; fallback price-change × leverage."""
    try:
        entry = float(pos.get("entry"))
        side  = str(pos.get("side"))
        lev   = float(pos.get("lev") or LEVERAGE_BASE)
        unreal = pos.get("unrealized")
        margin = pos.get("margin")
        if unreal is not None and margin not in (None, 0):
            roe = (float(unreal) / float(margin)) * 100.0
        else:
            if not np.isfinite(entry) or entry <= 0 or not np.isfinite(last_price) or last_price <= 0:
                return None
            change_pct = ((last_price - entry) / entry) * 100.0 if side == "long" else ((entry - last_price) / entry) * 100.0
            roe = change_pct * lev
        # clamp absurd values to avoid UI spikes
        if not np.isfinite(roe):
            return None
        return float(max(-10000.0, min(10000.0, roe)))
    except Exception:
        return None

def ai_decide_scale_plan(di: pd.DataFrame, hti: pd.DataFrame|None, pos: dict, p_up: float, news_pulse: dict|None):
    # Heuristic first; optionally upgraded by OpenAI if available
    bullish_bias = p_up >= 0.65
    cry_sent = (news_pulse or {}).get("crypto_sent", 0.0)
    rate_bias = (news_pulse or {}).get("rate_bias", "uncertain")
    # base plan
    plan = list(DEFAULT_TP_SCALE)
    try:
        # widen targets if very bullish + supportive macro
        if (pos.get("side") == "long" and bullish_bias and cry_sent > 0.15 and rate_bias in ("cut","hold")):
            plan = [(1.2,0.4),(2.5,0.3)]
        # tighten targets if bearish/regime-risky
        if (pos.get("side") == "long" and (cry_sent < -0.15 or rate_bias == "hike")):
            plan = [(0.8,0.5),(1.5,0.3)]
        if (pos.get("side") == "short" and cry_sent < -0.15):
            plan = [(1.0,0.5),(2.0,0.25)]
    except Exception:
        pass
    if not OPENAI_API_KEY:
        return plan
    # Let the model refine the plan when available
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        last = di.iloc[-1]
        payload = {
            "side": pos.get("side"),
            "p_up": float(p_up),
            "rsi": float(last.get("rsi", 50) or 50),
            "macd_hist": float(last.get("macd_hist", 0) or 0),
            "ema20_gt_ema50": bool(float(last.get("ema20",0)) > float(last.get("ema50",0))),
            "news": news_pulse or {},
            "base_plan": plan
        }
        rsp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role":"system","content":"Output JSON only: list of pairs under key 'plan' where each pair is [R_multiple, fraction]. Keep 1-3 items. Fractions between 0 and 1."},
                {"role":"user","content": json.dumps(payload, separators=(",",":"))}
            ],
            temperature=0.2,
            max_tokens=120,
        )
        raw = rsp.choices[0].message.content.strip()
        obj = json.loads(raw)
        cand = obj.get("plan")
        out = []
        for a,b in cand:
            a = float(a); b = float(b)
            if a>0 and 0<b<1:
                out.append((a,b))
        if out:
            return out[:3]
    except Exception:
        pass
    return plan

# -------------------- Indicators --------------------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def sma(s, n): return s.rolling(n).mean()

def wma(s: pd.Series, n: int) -> pd.Series:
    if n <= 1: return s.copy()
    w = np.arange(1, n+1)
    return s.rolling(n).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)

def hma(s: pd.Series, n: int) -> pd.Series:
    if n <= 1: return s.copy()
    n2 = max(1, int(n/2)); sqrt_n = max(1, int(np.sqrt(n)))
    return wma(2*wma(s, n2) - wma(s, n), sqrt_n)

def vwap_session(df: pd.DataFrame) -> pd.Series:
    g = df["datetime"].dt.floor("D")
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    return pv.groupby(g).cumsum() / df["volume"].groupby(g).cumsum().replace(0, np.nan)

def obv_series(df: pd.DataFrame) -> pd.Series:
    c = df["close"].astype(float); v = df["volume"].astype(float)
    sign = np.sign(c.diff().fillna(0.0))
    return (sign * v).cumsum()

def adl_series(df: pd.DataFrame) -> pd.Series:
    mf = ((df["close"]-df["low"]) - (df["high"]-df["close"])) / ((df["high"]-df["low"]).replace(0,np.nan))
    return (mf * df["volume"]).cumsum()

def psar_series(df: pd.DataFrame, af: float=0.02, af_max: float=0.2) -> pd.Series:
    n = len(df); 
    if n==0: return pd.Series(dtype=float)
    H = df["high"].to_numpy(float); L = df["low"].to_numpy(float)
    ps, bull, afc, ep = np.zeros(n), True, af, H[0]; ps[0] = L[0]
    for i in range(1,n):
        prev = ps[i-1]
        if bull:
            ps[i] = min(prev + afc*(ep-prev), L[i-1], L[i])
            if H[i] > ep: ep = H[i]; afc = min(afc+af, af_max)
            if L[i] < ps[i]: bull=False; ps[i]=ep; ep=L[i]; afc=af
        else:
            ps[i] = max(prev + afc*(ep-prev), H[i-1], H[i])
            if L[i] < ep: ep = L[i]; afc = min(afc+af, af_max)
            if H[i] > ps[i]: bull=True; ps[i]=ep; ep=H[i]; afc=af
    return pd.Series(ps, index=df.index)

def williams_r(df: pd.DataFrame, period: int=14) -> pd.Series:
    hh = df["high"].rolling(period).max()
    ll = df["low"].rolling(period).min()
    return -100.0 * (hh - df["close"]) / (hh - ll + 1e-12)

def vwma(p: pd.Series, v: pd.Series, n: int) -> pd.Series:
    return (p*v).rolling(n).sum() / v.rolling(n).sum().replace(0,np.nan)

def ppo(series: pd.Series, fast: int=12, slow: int=26, signal: int=9):
    f = ema(series, fast); s = ema(series, slow)
    line = 100.0 * (f - s) / s.replace(0,np.nan)
    sig = ema(line, signal); hist = line - sig
    return line, sig, hist

def cmf(df: pd.DataFrame, n: int=20) -> pd.Series:
    mfm = ((df["close"]-df["low"]) - (df["high"]-df["close"])) / ((df["high"]-df["low"]).replace(0,np.nan))
    mfv = mfm * df["volume"]
    return mfv.rolling(n).sum() / df["volume"].rolling(n).sum().replace(0,np.nan)

def stddev(s: pd.Series, n: int=20) -> pd.Series:
    return s.rolling(n).std(ddof=0)

def aroon(df: pd.DataFrame, period: int=25):
    upd = df["high"].rolling(period+1).apply(lambda x: period - np.argmax(x[::-1]), raw=True)
    dnd = df["low" ].rolling(period+1).apply(lambda x: period - np.argmin(x[::-1]), raw=True)
    up = 100.0 * (period - upd) / period
    dn = 100.0 * (period - dnd) / period
    return up, dn, (up - dn)

def rvi(df: pd.DataFrame, n: int=10) -> pd.Series:
    num = ((df["close"] - df["open"]) / (df["high"] - df["low"]).replace(0,np.nan)).replace([np.inf,-np.inf], np.nan)
    return num.rolling(n).mean()

def roc(s: pd.Series, n: int=12) -> pd.Series:
    return (s / s.shift(n) - 1.0) * 100.0

def vroc(v: pd.Series, n: int=12) -> pd.Series:
    return (v / v.shift(n) - 1.0) * 100.0

def tsi(series: pd.Series, r: int=25, s_: int=13) -> pd.Series:
    pc = series.diff(); apc = pc.abs()
    ema1 = pc.ewm(span=r, adjust=False).mean()
    ema2 = ema1.ewm(span=s_, adjust=False).mean()
    ema1a = apc.ewm(span=r, adjust=False).mean()
    ema2a = ema1a.ewm(span=s_, adjust=False).mean()
    return 100.0 * (ema2 / ema2a.replace(0,np.nan))

def ibs(df: pd.DataFrame) -> pd.Series:
    return ((df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0,np.nan)).clip(0.0,1.0)

def ara_arb(df: pd.DataFrame):
    rng = (df["high"] - df["low"]).astype(float)
    a = atr_cached(df, 14)
    ara = ((a / a.shift(1)).replace([np.inf,-np.inf], np.nan) - 1.0) * 100.0
    arb = (rng / a.replace(0,np.nan)).clip(0.0, 10.0)
    return ara, arb

def pump_score(df: pd.DataFrame, n_vol: int=20) -> pd.Series:
    vol_ma = df["volume"].rolling(n_vol).mean()
    a = atr_cached(df, 14)
    body = (df["close"] - df["open"]).abs()
    return (body / a.replace(0,np.nan)) * (df["volume"] / vol_ma.replace(0,np.nan))

def detect_breakout_fakeout(df: pd.DataFrame, lookback: int=20):
    try:
        hh = df["high"].shift(1).rolling(lookback).max().iloc[-1]
        ll = df["low" ].shift(1).rolling(lookback).min().iloc[-1]
        c0 = float(df["close"].iloc[-1])
        c1 = float(df["close"].iloc[-2]) if len(df)>1 else c0
        brk_up  = c0 > hh; brk_dn = c0 < ll
        fake_up = (c1 > hh) and (c0 <= hh) if len(df)>1 else False
        fake_dn = (c1 < ll) and (c0 >= ll) if len(df)>1 else False
        return brk_up, brk_dn, fake_up, fake_dn
    except Exception:
        return False, False, False, False

def fib_levels(df: pd.DataFrame, lookback: int=120) -> dict:
    d = df.tail(lookback)
    try:
        hi = float(d["high"].max()); lo = float(d["low"].min())
    except Exception:
        return {}
    if not np.isfinite(hi) or not np.isfinite(lo) or hi <= lo: return {}
    diff = hi - lo
    return {"0.0": hi, "0.236": hi - 0.236*diff, "0.382": hi - 0.382*diff,
            "0.5": hi - 0.5*diff, "0.618": hi - 0.618*diff, "0.786": hi - 0.786*diff, "1.0": lo}

# ============ ELLIOTT WAVE ANALYSIS ============

def find_swing_points(df: pd.DataFrame, min_swing_pct: float = 0.008) -> dict:
    """
    Identify swing highs and lows for Elliott Wave analysis.
    Returns dict with 'highs' and 'lows' containing (index, price) tuples.
    """
    if len(df) < 10:
        return {"highs": [], "lows": []}
    
    highs = []
    lows = []
    
    # Find local extremes
    for i in range(2, len(df) - 2):
        current_high = df["high"].iloc[i]
        current_low = df["low"].iloc[i]
        
        # Check for swing high (higher than 2 bars before and after)
        if (current_high > df["high"].iloc[i-1] and current_high > df["high"].iloc[i-2] and
            current_high > df["high"].iloc[i+1] and current_high > df["high"].iloc[i+2]):
            
            # Check minimum swing size
            if not highs or abs(current_high - highs[-1][1]) / highs[-1][1] >= min_swing_pct:
                highs.append((i, current_high))
        
        # Check for swing low (lower than 2 bars before and after)
        if (current_low < df["low"].iloc[i-1] and current_low < df["low"].iloc[i-2] and
            current_low < df["low"].iloc[i+1] and current_low < df["low"].iloc[i+2]):
            
            # Check minimum swing size
            if not lows or abs(current_low - lows[-1][1]) / lows[-1][1] >= min_swing_pct:
                lows.append((i, current_low))
    
    return {"highs": highs, "lows": lows}

def elliott_wave_analysis(df: pd.DataFrame, lookback: int = 100) -> dict:
    """
    Elliott Wave pattern recognition with resource optimization.
    Returns wave count, trend direction, and key levels.
    """
    if len(df) < 50:
        return {"wave_count": 0, "direction": "sideways", "pattern": "incomplete", 
                "support": None, "resistance": None, "impulse_strength": 0}
    
    # Use limited lookback for performance
    data = df.tail(lookback).copy()
    swings = find_swing_points(data, EW_MIN_SWING)
    
    highs = swings["highs"]
    lows = swings["lows"]
    
    if len(highs) < 3 or len(lows) < 3:
        return {"wave_count": 0, "direction": "sideways", "pattern": "incomplete",
                "support": None, "resistance": None, "impulse_strength": 0}
    
    # Combine and sort swing points
    all_swings = [(idx, price, "H") for idx, price in highs] + [(idx, price, "L") for idx, price in lows]
    all_swings.sort(key=lambda x: x[0])
    
    if len(all_swings) < 5:
        return {"wave_count": len(all_swings), "direction": "sideways", "pattern": "incomplete",
                "support": None, "resistance": None, "impulse_strength": 0}
    
    # Analyze last 5 waves for pattern recognition
    recent_swings = all_swings[-5:]
    
    # Determine overall trend direction
    first_price = recent_swings[0][1]
    last_price = recent_swings[-1][1]
    direction = "bullish" if last_price > first_price else "bearish" if last_price < first_price else "sideways"
    
    # Calculate wave ratios for Fibonacci relationships
    wave_ratios = []
    for i in range(1, len(recent_swings)):
        ratio = abs(recent_swings[i][1] - recent_swings[i-1][1]) / abs(recent_swings[0][1] - recent_swings[-1][1])
        wave_ratios.append(ratio)
    
    # Identify potential Elliott Wave patterns
    pattern = "incomplete"
    impulse_strength = 0
    
    if len(recent_swings) >= 5:
        # Check for 5-wave impulse pattern
        w1 = abs(recent_swings[1][1] - recent_swings[0][1])
        w3 = abs(recent_swings[3][1] - recent_swings[2][1])
        w5 = abs(recent_swings[4][1] - recent_swings[3][1])
        
        # Wave 3 should be longest in impulse
        if w3 > w1 and w3 > w5:
            # Check Fibonacci ratios
            ratio_31 = w3 / w1 if w1 > 0 else 0
            ratio_51 = w5 / w1 if w1 > 0 else 0
            
            # Common Elliott Wave ratios: 1.618, 2.618, 0.618
            if 1.4 <= ratio_31 <= 2.8 or 0.5 <= ratio_51 <= 0.8:
                pattern = "impulse"
                impulse_strength = min(100, max(0, (ratio_31 - 1) * 50))
        
        # Check for corrective patterns (ABC)
        elif len(recent_swings) == 3:
            pattern = "corrective"
    
    # Support and resistance from swing points
    support = min([swing[1] for swing in recent_swings if swing[2] == "L"], default=None)
    resistance = max([swing[1] for swing in recent_swings if swing[2] == "H"], default=None)
    
    return {
        "wave_count": len(recent_swings),
        "direction": direction,
        "pattern": pattern,
        "support": support,
        "resistance": resistance,
        "impulse_strength": impulse_strength,
        "swings": recent_swings[-5:] if len(recent_swings) >= 5 else recent_swings
    }

# Cache for Elliott Wave analysis to reduce computation
EW_CACHE = {}

def get_elliott_wave_cached(df: pd.DataFrame, symbol: str) -> dict:
    """
    Cached Elliott Wave analysis with TTL to optimize performance.
    """
    if df is None or len(df) < 50:
        return {"wave_count": 0, "direction": "sideways", "pattern": "incomplete",
                "support": None, "resistance": None, "impulse_strength": 0}
    
    # Create cache key from last few candles timestamp
    last_ts = int(df.index[-1].timestamp()) if hasattr(df.index[-1], 'timestamp') else int(time.time())
    cache_key = f"{symbol}_{last_ts // EW_CACHE_TTL}"  # Group by TTL window
    
    # Check cache
    if cache_key in EW_CACHE:
        cached_data, cached_time = EW_CACHE[cache_key]
        if time.time() - cached_time < EW_CACHE_TTL:
            return cached_data
    
    # Compute new analysis
    ew_data = elliott_wave_analysis(df, EW_LOOKBACK)
    EW_CACHE[cache_key] = (ew_data, time.time())
    
    # Clean old cache entries (keep max 10 entries)
    if len(EW_CACHE) > 10:
        oldest_key = min(EW_CACHE.keys())
        del EW_CACHE[oldest_key]
    
    return ew_data

def rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta>0, delta, 0.0)
    loss = np.where(delta<0, -delta, 0.0)
    roll_up = pd.Series(gain, index=series.index).rolling(period).mean()
    roll_dn = pd.Series(loss, index=series.index).rolling(period).mean()
    rs = roll_up / (roll_dn.replace(0, np.nan))
    r = 100 - (100/(1+rs))
    return pd.Series(r, index=series.index).bfill()

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    line = ema_fast - ema_slow
    sig  = ema(line, signal)
    hist = line - sig
    return line, sig, hist

def atr(df, n=14):
    h = df["high"]; l = df["low"]; c = df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# Cached ATR to avoid recomputing on every tick for the same candle
def atr_cached(df: pd.DataFrame, n: int = 14) -> pd.Series:
    if df is None or df.empty:
        return atr(df, n)
    try:
        last_ts = int(pd.to_datetime(df["datetime"].iloc[-1]).timestamp())
    except Exception:
        last_ts = len(df)
    key = (id(df), int(n), last_ts)
    ent = ATR_CACHE.get(key)
    if ent is not None:
        return ent
    s = atr(df, n)
    ATR_CACHE[key] = s
    # keep cache tiny (LRU-like) instead of clearing every call
    if len(ATR_CACHE) > 8:
        for old_k in list(ATR_CACHE.keys())[:len(ATR_CACHE)-8]:
            if old_k != key:
                ATR_CACHE.pop(old_k, None)
    return s

# --- ADX helper ---
def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """
    Lightweight ADX (Wilder-style approximation using SMA smoothing).
    Returns Series of ADX values (0..100).
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)
    high = df["high"].astype(float)
    low  = df["low"].astype(float)
    close = df["close"].astype(float)

    up_move = high.diff()
    down_move = (-low.diff())
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_n = tr.rolling(n).mean()

    plus_di  = 100.0 * (plus_dm.rolling(n).mean()  / atr_n.replace(0, np.nan))
    minus_di = 100.0 * (minus_dm.rolling(n).mean() / atr_n.replace(0, np.nan))

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100.0
    adx_val = dx.rolling(n).mean()
    return adx_val.bfill()

def stoch(df, k=14, d=3):
    low_min  = df["low"].rolling(k).min()
    high_max = df["high"].rolling(k).max()
    k_fast = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-12)
    d_slow = k_fast.rolling(d).mean()
    return k_fast, d_slow

def bbands(series: pd.Series, period: int = 20, mult: float = 2.0):
    mid = sma(series, period)
    std = series.rolling(period).std(ddof=0)
    upper = mid + mult * std
    lower = mid - mult * std
    width = (upper - lower) / mid.replace(0, np.nan)
    return mid, upper, lower, width

def swing_levels(df, left=5, right=5, max_lines=6):
    # Use compact tail to limit CPU
    win = min(len(df), max(100, 5*left + 5*right + 60))
    d = stable_tail(df, win)
    if d is None or d.empty or len(d) < (left + right + 3):
        return [], []

    H = d["high"].to_numpy(dtype=float, copy=False)
    L = d["low"].to_numpy(dtype=float, copy=False)

    idx_high, idx_low = [], []
    rng = range(left, len(d) - right)
    # Single pass to collect pivots
    for i in rng:
        seg = slice(i-left, i+right+1)
        if H[i] == np.max(H[seg]):
            idx_high.append(H[i])
        if L[i] == np.min(L[seg]):
            idx_low.append(L[i])

    levels = idx_high + idx_low
    if not levels:
        return [], []

    lv = np.asarray(sorted(levels), dtype=float)
    # Pick evenly spread quantiles as representative levels
    qs = np.quantile(lv, np.linspace(0, 1, max_lines + 2))[1:-1]

    # Deduplicate close-by levels (relative epsilon ~0.1%)
    eps_rel = 1e-3
    def dedup(vals):
        out = []
        for v in sorted(set([float(x) for x in vals])):
            if not out or abs(v - out[-1]) > eps_rel * max(1.0, v):
                out.append(float(v))
        return out

    cand = dedup(qs.tolist() + lv.tolist())
    px = float(d["close"].iloc[-1])
    supports = [x for x in cand if x <= px][-max_lines:]
    resist   = [x for x in cand if x >= px][:max_lines]
    return supports, resist

def candle_patterns(df):
    o,h,l,c = [df[k] for k in ("open","high","low","close")]
    body = (c - o).abs()
    rng  = (h - l).replace(0, np.nan)
    upper = (h - c).where(c>=o, h - o)
    lower = (o - l).where(c>=o, c - l)
    doji = (body <= 0.1 * rng)
    prev_o, prev_c = o.shift(1), c.shift(1)
    bull_eng = (c > o) & (prev_c < prev_o) & (c >= prev_o) & (o <= prev_c)
    bear_eng = (c < o) & (prev_c > prev_o) & (c <= prev_o) & (o >= prev_c)
    hammer = (lower >= 2*body) & (upper <= 0.3*body)
    shoot  = (upper >= 2*body) & (lower <= 0.3*body)
    return {
        "doji": doji.fillna(False),
        "bull_engulf": bull_eng.fillna(False),
        "bear_engulf": bear_eng.fillna(False),
        "hammer": hammer.fillna(False),
        "shooting": shoot.fillna(False),
    }

def compute_indicators(df: pd.DataFrame, symbol: str = "UNKNOWN") -> pd.DataFrame:
    """Optimized indicator computation with Elliott Wave analysis."""
    d = df.copy()
    if len(d) > IND_MAX_LEN:
        d = d.tail(IND_MAX_LEN).copy()

    # Memory optimization: use float32 consistently
    d = d.astype(DF_FLOAT_DTYPE)
    
    # Core indicators (most important first)
    d["ema20"] = ema(d["close"], 20).astype(DF_FLOAT_DTYPE)
    d["ema50"] = ema(d["close"], 50).astype(DF_FLOAT_DTYPE)
    d["rsi"]   = rsi(d["close"], 14).astype(DF_FLOAT_DTYPE)
    
    # MACD
    macd_line, macd_sig, macd_hist = macd(d["close"])
    d["macd_line"] = macd_line.astype(DF_FLOAT_DTYPE)
    d["macd_signal"] = macd_sig.astype(DF_FLOAT_DTYPE) 
    d["macd_hist"] = macd_hist.astype(DF_FLOAT_DTYPE)
    
    # Stochastic
    k, kk = stoch(d, 14, 3)
    d["stoch_k"] = k.astype(DF_FLOAT_DTYPE)
    d["stoch_d"] = kk.astype(DF_FLOAT_DTYPE)
    
    # Bollinger Bands
    mid, up, low, bw = bbands(d["close"], 20, 2.0)
    d["bb_mid"] = mid.astype(DF_FLOAT_DTYPE)
    d["bb_upper"] = up.astype(DF_FLOAT_DTYPE) 
    d["bb_lower"] = low.astype(DF_FLOAT_DTYPE)
    d["bb_width"] = bw.astype(DF_FLOAT_DTYPE)
    
    # ADX for trend strength
    d["adx"] = adx(d, 14).astype(DF_FLOAT_DTYPE)
    
    # Elliott Wave Analysis (cached for performance)
    try:
        ew_data = get_elliott_wave_cached(d, symbol)
        d["ew_direction"] = ew_data["direction"]
        d["ew_pattern"] = ew_data["pattern"] 
        d["ew_wave_count"] = ew_data["wave_count"]
        d["ew_impulse_strength"] = ew_data["impulse_strength"]
        d["ew_support"] = ew_data["support"] if ew_data["support"] else np.nan
        d["ew_resistance"] = ew_data["resistance"] if ew_data["resistance"] else np.nan
    except Exception as e:
        if DEBUG_LOG:
            print(f"Elliott Wave error: {e}")
        # Fallback values
        d["ew_direction"] = "sideways"
        d["ew_pattern"] = "incomplete"
        d["ew_wave_count"] = 0
        d["ew_impulse_strength"] = 0
        d["ew_support"] = np.nan
        d["ew_resistance"] = np.nan

    # Candle patterns (tail only)
    _pat_win = min(len(d), 300)
    d[["doji","bull_engulf","bear_engulf","hammer","shooting"]] = False
    if _pat_win > 20:
        tail_idx = d.tail(_pat_win).index
        pats = candle_patterns(d.tail(_pat_win))
        for kx, vx in pats.items():
            d.loc[tail_idx, kx] = vx.values

    # New indicators
    try: d["vwap"] = vwap_session(d).astype(float)
    except: d["vwap"] = np.nan
    try: d["obv"] = obv_series(d).astype(float)
    except: d["obv"] = np.nan
    try: d["adl"] = adl_series(d).astype(float)
    except: d["adl"] = np.nan
    try: d["psar"] = psar_series(d).astype(float)
    except: d["psar"] = np.nan
    try: d["williams_r"] = williams_r(d, 14).astype(float)
    except: d["williams_r"] = np.nan
    try:
        d["vwma20"] = vwma(d["close"], d["volume"], 20).astype(float)
        d["vwma50"] = vwma(d["close"], d["volume"], 50).astype(float)
    except:
        d["vwma20"], d["vwma50"] = np.nan, np.nan
    try:
        d["hma20"] = hma(d["close"], 20).astype(float)
        d["hma50"] = hma(d["close"], 50).astype(float)
    except:
        d["hma20"], d["hma50"] = np.nan, np.nan
    try:
        ppo_line, ppo_sig, ppo_hist = ppo(d["close"], 12, 26, 9)
        d["ppo"], d["ppo_signal"], d["ppo_hist"] = ppo_line, ppo_sig, ppo_hist
    except:
        d["ppo"], d["ppo_signal"], d["ppo_hist"] = np.nan, np.nan, np.nan
    try: d["cmf"] = cmf(d, 20).astype(float)
    except: d["cmf"] = np.nan
    try: d["std20"] = stddev(d["close"], 20).astype(float)
    except: d["std20"] = np.nan
    try:
        au, adw, ao = aroon(d, 25)
        d["aroon_up"], d["aroon_down"], d["aroon_osc"] = au.astype(float), adw.astype(float), ao.astype(float)
    except:
        d["aroon_up"], d["aroon_down"], d["aroon_osc"] = np.nan, np.nan, np.nan
    try: d["rvi"] = rvi(d, 10).astype(float)
    except: d["rvi"] = np.nan
    try: d["roc12"] = roc(d["close"], 12).astype(float)
    except: d["roc12"] = np.nan
    try: d["vroc"] = vroc(d["volume"], 12).astype(float)
    except: d["vroc"] = np.nan
    try: d["tsi"] = tsi(d["close"], 25, 13).astype(float)
    except: d["tsi"] = np.nan
    try: d["ibs"] = ibs(d).astype(float)
    except: d["ibs"] = np.nan
    try:
        ara, arb = ara_arb(d)
        d["ara"], d["arb"] = ara.astype(float), arb.astype(float)
    except:
        d["ara"], d["arb"] = np.nan, np.nan
    try: d["pump_score"] = pump_score(d).astype(float)
    except: d["pump_score"] = np.nan

    # Downcast once via shared helper (hindari duplikasi logika)
    return _downcast_df(d)

def build_optimized_chart(di: pd.DataFrame, pos: dict | None, ob_zones: list, symbol: str, tf: str, overlay: dict | None = None) -> go.Figure:
    """
    Memory-optimized chart rendering with Elliott Wave visualization.
    Features: Smart caching, minimal recomputation, Elliott Wave overlays.
    """
    # Reduced column set for memory optimization
    essential_cols = ["datetime","open","high","low","close","volume"]
    indicator_cols = ["ema20","ema50","vwap","bb_upper","bb_lower","ew_support","ew_resistance"]
    
    # Build visible data with only necessary columns
    available_cols = essential_cols + [c for c in indicator_cols if c in di.columns]
    vis = di.loc[:, available_cols].tail(PLOT_BARS).copy()
    
    # Use float32 to reduce memory usage
    numeric_cols = [c for c in vis.columns if c != 'datetime']
    vis[numeric_cols] = vis[numeric_cols].astype(DF_FLOAT_DTYPE)
    
    # Optimized cache key generation (reduced signature)
    last_price = float(vis["close"].iloc[-1])
    last_time = vis["datetime"].iloc[-1]
    pos_hash = hash(str(pos)) if pos else 0
    ob_hash = len(ob_zones) if ob_zones else 0
    
    cache_key = f"{symbol}_{tf}_{len(vis)}_{hash(str(last_time))}_{int(last_price*10000)}_{pos_hash}_{ob_hash}"
    
    # Check cache first
    cache = STATE.setdefault("fig_cache", {})
    if cache_key in cache:
        cached_fig = cache[cache_key]
        if cached_fig is not None:
            return cached_fig
    
    # Overlay toggles (default: semua ON)
    _ov = {"vwap": True, "psar": True, "fib": True, "brkfake": True}
    if isinstance(overlay, dict):
        for k in list(_ov.keys()):
            try:
                _ov[k] = bool(overlay.get(k, _ov[k]))
            except Exception:
                pass

    # --- Figure cache signature (last bar + pos + zones count) ---
    try:
        last_dt = vis["datetime"].iloc[-1]
        last_c  = round(float(vis["close"].iloc[-1]), 4)
    except Exception:
        last_dt, last_c = (None, None)
    pos_sig = None
    if pos:
        pos_sig = (
            float(pos.get("entry") or 0),
            float(pos.get("tp") or 0),
            float(pos.get("sl") or 0),
        )
    try:
        zc = len(ob_zones) if ob_zones else 0
        if zc and isinstance(ob_zones, list):
            z0 = ob_zones[-1]
            z1 = ob_zones[0] if len(ob_zones) > 1 else z0
            z_sig = (
                float(z0.get("low", 0.0)), float(z0.get("high", 0.0)),
                float(z1.get("low", 0.0)), float(z1.get("high", 0.0)),
            )
        else:
            z_sig = (0.0, 0.0, 0.0, 0.0)
    except Exception:
        z_sig = (0.0, 0.0, 0.0, 0.0)
    sig = (str(symbol), str(tf), len(vis), str(last_dt), last_c, pos_sig, z_sig,
        bool(_ov.get("vwap", True)), bool(_ov.get("psar", True)),
        bool(_ov.get("fib", True)), bool(_ov.get("brkfake", True)))

    cache = STATE.setdefault("fig_cache", {})
    key = f"{symbol}|{tf}"
    ent = cache.get(key)
    if ent and ent.get("sig") == sig and ent.get("fig") is not None:
        return ent["fig"]

    # Create optimized subplot layout
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03,  # Reduced spacing
        row_heights=[0.8, 0.2],  # More space for main chart
        subplot_titles=(f"{symbol} {tf}", "Volume")
    )
    
    # Configure for better performance
    fig.update_layout(
        template="plotly_dark",
        showlegend=True,
        legend=dict(x=0, y=1, bgcolor="rgba(0,0,0,0.5)"),
        margin=dict(l=10, r=10, t=30, b=10),  # Reduced margins
        height=600,  # Fixed height for consistency
        hovermode='x unified'  # Better hover performance
    )

    # Main candlestick chart with reduced data points if needed
    sample_rate = max(1, len(vis) // 500)  # Downsample if too many points
    if sample_rate > 1:
        vis_sampled = vis.iloc[::sample_rate].copy()
    else:
        vis_sampled = vis
        
    fig.add_trace(go.Candlestick(
        x=vis_sampled["datetime"], 
        open=vis_sampled["open"], 
        high=vis_sampled["high"],
        low=vis_sampled["low"], 
        close=vis_sampled["close"], 
        name=f"{tf}",
        showlegend=False  # Reduce legend clutter
    ), row=1, col=1)

    # Essential EMAs with optimized rendering
    if "ema20" in vis.columns:
        fig.add_trace(go.Scatter(
            x=vis_sampled["datetime"], 
            y=vis_sampled["ema20"], 
            name="EMA20", 
            mode="lines",
            line=dict(width=1.5, color="#FFA500"),  # Orange
            hovertemplate="EMA20: %{y:.6f}<extra></extra>"
        ), row=1, col=1)
        
    if "ema50" in vis.columns:
        fig.add_trace(go.Scatter(
            x=vis_sampled["datetime"], 
            y=vis_sampled["ema50"], 
            name="EMA50", 
            mode="lines",
            line=dict(width=1.5, color="#00CED1"),  # Dark Turquoise
            hovertemplate="EMA50: %{y:.6f}<extra></extra>"
        ), row=1, col=1)

    # Elliott Wave Support/Resistance levels
    if "ew_support" in vis.columns and vis["ew_support"].notna().any():
        support_level = vis["ew_support"].iloc[-1]
        if not np.isnan(support_level):
            fig.add_hline(
                y=support_level,
                line_dash="dash",
                line_color="#00FF00",  # Green
                annotation_text=f"EW Support: {support_level:.6f}",
                annotation_position="bottom right",
                row=1, col=1
            )
            
    if "ew_resistance" in vis.columns and vis["ew_resistance"].notna().any():
        resistance_level = vis["ew_resistance"].iloc[-1]
        if not np.isnan(resistance_level):
            fig.add_hline(
                y=resistance_level,
                line_dash="dash", 
                line_color="#FF0000",  # Red
                annotation_text=f"EW Resistance: {resistance_level:.6f}",
                annotation_position="top right",
                row=1, col=1
            )

    # Optimized volume bars (reduced opacity for performance)
    fig.add_trace(go.Bar(
        x=vis_sampled["datetime"], 
        y=vis_sampled["volume"], 
        name="Volume",
        opacity=0.6,
        showlegend=False,
        marker_color="#404040"
    ), row=2, col=1)

    # Optional overlays (lightweight & from active TF only)
    # 1) VWAP line (session-based)
    try:
        if _ov["vwap"] and "vwap" in vis.columns and vis["vwap"].notna().any():
            fig.add_trace(
                go.Scatter(x=vis["datetime"], y=vis["vwap"], name="VWAP", mode="lines"),
                row=1, col=1
            )
    except Exception:
        pass

    # 2) PSAR markers
    try:
        if _ov["psar"] and "psar" in vis.columns and vis["psar"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=vis["datetime"], y=vis["psar"], name="PSAR",
                    mode="markers", marker=dict(size=5),
                ),
                row=1, col=1
            )
    except Exception:
        pass

    # 3) Fibonacci retracement
    try:
        if _ov["fib"]:
            fl = fib_levels(vis, lookback=min(240, len(vis)))
            if isinstance(fl, dict) and fl:
                for lvl_key in ("0.236","0.382","0.5","0.618"):
                    if lvl_key in fl:
                        yv = float(fl[lvl_key])
                        try:
                            fig.add_hline(y=yv, line=dict(width=1, dash="dot"),
                                        annotation_text=f"Fib {lvl_key}", row=1, col=1)
                        except Exception:
                            fig.add_shape(type="line", x0=vis["datetime"].iloc[0], x1=vis["datetime"].iloc[-1],
                                        y0=yv, y1=yv, xref="x", yref="y",
                                        line=dict(width=1, dash="dot"), row=1, col=1)
    except Exception:
        pass

    # 4) Breakout / Fakeout marker
    try:
        if _ov["brkfake"]:
            brk_up, brk_dn, fake_up, fake_dn = detect_breakout_fakeout(vis, max(10, BREAKOUT_N*4))
            x_last = vis["datetime"].iloc[-1]
            y_mid  = float(vis["close"].iloc[-1])
            note = "Breakout↑" if brk_up else ("Breakout↓" if brk_dn else ("Fakeout↑" if fake_up else ("Fakeout↓" if fake_dn else None)))
            if note:
                try:
                    fig.add_vline(x=x_last, line=dict(width=1, dash="dot"), annotation_text=note, row=1, col=1)
                except Exception:
                    fig.add_shape(type="line", x0=x_last, x1=x_last,
                                y0=y_mid*0.999, y1=y_mid*1.001,
                                xref="x", yref="y", line=dict(width=1, dash="dot"), row=1, col=1)
    except Exception:
        pass

    # Stabilize axes across updates (hindari loncat/lompat autoscale)
    try:
        y_min = float(np.nanmin(vis["low"]))
        y_max = float(np.nanmax(vis["high"]))
        if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
            yr0, yr1 = _smooth_y_range(symbol, tf, y_min, y_max, pad=0.02)
            fig.update_yaxes(range=[yr0, yr1], row=1, col=1, autorange=False)
    except Exception:
        pass

    # Optimized Order Block zones rendering (max 5 zones)
    if ob_zones and len(ob_zones) > 0:
        x0 = vis["datetime"].iloc[0]
        x1 = vis["datetime"].iloc[-1]
        
        # Limit to most recent 5 zones for performance
        zones_to_render = ob_zones[-5:] if len(ob_zones) > 5 else ob_zones
        
        for i, zone in enumerate(zones_to_render):
            try:
                zone_low = float(zone.get("low", 0))
                zone_high = float(zone.get("high", 0))
                zone_type = zone.get("type", "unknown")
                
                color = "rgba(0,255,0,0.1)" if zone_type == "bull" else "rgba(255,0,0,0.1)"
                
                fig.add_shape(
                    type="rect",
                    x0=x0, x1=x1,
                    y0=zone_low, y1=zone_high,
                    fillcolor=color,
                    line=dict(width=0),
                    layer="below",
                    row=1, col=1
                )
            except Exception:
                continue
    
    # Add Elliott Wave pattern annotations
    try:
        if hasattr(di, 'index') and len(di) > 0:
            ew_data = get_elliott_wave_cached(di, symbol)
            if ew_data.get("pattern") != "incomplete" and "swings" in ew_data:
                swings = ew_data["swings"]
                if len(swings) >= 3:
                    # Draw swing lines
                    for i in range(len(swings)-1):
                        try:
                            idx1, price1, type1 = swings[i]
                            idx2, price2, type2 = swings[i+1]
                            
                            # Convert index to datetime if possible
                            if idx1 < len(vis) and idx2 < len(vis):
                                dt1 = vis["datetime"].iloc[min(idx1, len(vis)-1)]
                                dt2 = vis["datetime"].iloc[min(idx2, len(vis)-1)]
                                
                                # Color based on wave direction
                                line_color = "#FFD700" if ew_data["direction"] == "bullish" else "#FF6347"
                                
                                fig.add_trace(go.Scatter(
                                    x=[dt1, dt2],
                                    y=[price1, price2],
                                    mode="lines+markers",
                                    line=dict(color=line_color, width=2, dash="dot"),
                                    marker=dict(size=6, color=line_color),
                                    name=f"EW-{i+1}",
                                    showlegend=False,
                                    hovertemplate=f"Wave {i+1}: %{{y:.6f}}<extra></extra>"
                                ), row=1, col=1)
                        except Exception:
                            continue
                    
                    # Add pattern annotation
                    pattern_text = f"Elliott Wave: {ew_data['pattern'].upper()}"
                    if ew_data.get("impulse_strength", 0) > 0:
                        pattern_text += f" (Strength: {ew_data['impulse_strength']:.0f}%)"
                        
                    fig.add_annotation(
                        x=vis["datetime"].iloc[-1],
                        y=vis["high"].iloc[-1],
                        text=pattern_text,
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="#FFD700",
                        bgcolor="rgba(0,0,0,0.8)",
                        bordercolor="#FFD700",
                        font=dict(color="white", size=10),
                        row=1, col=1
                    )
    except Exception as e:
        if DEBUG_LOG:
            print(f"Elliott Wave visualization error: {e}")
    
    # Position overlays (entry/TP/SL)
    if pos:
        try:
            entry = float(pos.get("entry", 0))
            tp = float(pos.get("tp", 0)) or float(pos.get("tp_calc", 0))
            sl = float(pos.get("sl", 0)) or float(pos.get("sl_calc", 0))
            side = pos.get("side", "unknown")
            
            if entry > 0:
                fig.add_hline(
                    y=entry,
                    line_dash="solid",
                    line_color="#FFFF00",  # Yellow
                    annotation_text=f"Entry: {entry:.6f}",
                    annotation_position="left",
                    row=1, col=1
                )
            
            if tp > 0:
                fig.add_hline(
                    y=tp,
                    line_dash="dash",
                    line_color="#00FF00",  # Green
                    annotation_text=f"TP: {tp:.6f}",
                    annotation_position="right",
                    row=1, col=1
                )
                
            if sl > 0:
                fig.add_hline(
                    y=sl,
                    line_dash="dash",
                    line_color="#FF0000",  # Red
                    annotation_text=f"SL: {sl:.6f}",
                    annotation_position="right",
                    row=1, col=1
                )
        except Exception as e:
            if DEBUG_LOG:
                print(f"Position overlay error: {e}")
    
    # Optimize y-axis range for better visibility
    try:
        y_min = float(np.nanmin(vis["low"])) * 0.999
        y_max = float(np.nanmax(vis["high"])) * 1.001
        fig.update_yaxes(range=[y_min, y_max], row=1, col=1)
    except Exception:
        pass
    
    # Final layout optimizations
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    # Cache the result
    cache[cache_key] = fig
    
    # Clean cache if too large
    if len(cache) > FIG_CACHE_MAX:
        # Remove oldest entries
        keys_to_remove = list(cache.keys())[:-FIG_CACHE_MAX//2]
        for k in keys_to_remove:
            cache.pop(k, None)
    
    return fig

# Alias for backward compatibility
build_clean_chart = build_optimized_chart

# Strengthened signal with HTF bias and ATR-based S/R proximity
def simple_signal(d: pd.DataFrame, htf_bias: str = "flat", near_sup_atr: float = 1e9, near_res_atr: float = 1e9):
    last = d.iloc[-1]
    reasons = []

    # Regime checks
    ap = atr_percent(d)
    if ap < ATR_PCT_MIN:
        return "HOLD", 0, "Low-vol regime (ATR% too small); skip", 0.0
    if ap > ATR_PCT_MAX:
        return "HOLD", 0, "Too-volatile regime (ATR% too high); skip", 0.0
    try:
        if float(last.get("bb_width", np.nan)) < float(SQUEEZE_MIN_BBWIDTH):
            return "HOLD", 0, "Squeeze (BB width low); wait breakout", 0.0
    except Exception:
        pass

    # Risk aversion
    rv = float(RISK_AVERSION)
    adx_gate = max(0.0, ADX_MIN + 5.0 * rv)
    p_up_buy_gate  = min(0.70, P_UP_MIN_BUY  + 0.06 * rv)
    p_up_sell_gate = max(0.30, P_UP_MAX_SELL - 0.06 * rv)

    votes = []
    # Trend / momentum / oscillator
    if last["ema20"] > last["ema50"]: votes.append(+1); reasons.append("MA trend: Bullish")
    elif last["ema20"] < last["ema50"]: votes.append(-1); reasons.append("MA trend: Bearish")
    else: reasons.append("MA trend: Flat")

    votes.append(+1 if last.get("macd_hist",0) > 0 else -1); reasons.append(f"MACD hist: {'+' if last.get('macd_hist',0)>0 else '-'}")
    if last.get("rsi",50) >= 55: votes.append(+1); reasons.append("RSI>55")
    elif last.get("rsi",50) <= 45: votes.append(-1); reasons.append("RSI<45")
    else: reasons.append("RSI mid")
    votes.append(+1 if float(last.get("stoch_k",50)) > float(last.get("stoch_d",50)) else -1); reasons.append("Stoch K vs D")
    if last.get("bb_mid")==last.get("bb_mid"):
        votes.append(+1 if last["close"] > last["bb_mid"] else -1); reasons.append("BB vs mid")

    # Leading signals
    try:
        ibsv = float(last.get("ibs", 0.5))
        if ibsv >= 0.6: votes.append(+1); reasons.append("IBS>0.6")
        if ibsv <= 0.4: votes.append(-1); reasons.append("IBS<0.4")
    except Exception: pass
    try:
        wr = float(last.get("williams_r", -50))
        if wr > -50: votes.append(+1); reasons.append("W%R>-50")
        if wr < -80: votes.append(-1); reasons.append("W%R<-80")
    except Exception: pass
    try:
        ao = float(last.get("aroon_osc", 0))
        if ao > 0: votes.append(+1); reasons.append("Aroon>0")
        if ao < 0: votes.append(-1); reasons.append("Aroon<0")
    except Exception: pass
    try:
        if float(last.get("ppo_hist", 0)) > 0: votes.append(+1); reasons.append("PPO hist +")
        else: votes.append(-1); reasons.append("PPO hist -")
    except Exception: pass
    try:
        if float(last.get("cmf", 0)) > 0: votes.append(+1); reasons.append("CMF+")
        else: votes.append(-1); reasons.append("CMF-")
    except Exception: pass
    try:
        if float(last.get("vwap", last["close"])) <= float(last.get("close")):
            votes.append(+1); reasons.append("Above VWAP")
        else:
            votes.append(-1); reasons.append("Below VWAP")
    except Exception: pass

    # OBV slope
    try:
        obv_tail = d["obv"].tail(5)
        if len(obv_tail.dropna()) >= 2 and (obv_tail.iloc[-1] - obv_tail.iloc[0]) > 0:
            votes.append(+1); reasons.append("OBV rising")
        elif len(obv_tail.dropna()) >= 2:
            votes.append(-1); reasons.append("OBV falling")
    except Exception: pass

    # S/R proximity (ATR units)
    if near_sup_atr < 0.6: votes.append(+1); reasons.append("Near support (<0.6 ATR)")
    if near_res_atr < 0.6: votes.append(-1); reasons.append("Near resistance (<0.6 ATR)")

    # HTF bias
    if htf_bias == "bull": votes += [+1, +1]; reasons.append("HTF bias: Bull")
    elif htf_bias == "bear": votes += [-1, -1]; reasons.append("HTF bias: Bear")
    else: reasons.append("HTF bias: Flat")

    # Breakout & Fakeout
    brk_up, brk_dn, fake_up, fake_dn = detect_breakout_fakeout(d, max(10, BREAKOUT_N*4))
    if brk_up: votes.append(+1); reasons.append("Breakout↑")
    if brk_dn: votes.append(-1); reasons.append("Breakout↓")
    if fake_up: votes.append(-2); reasons.append("Fakeout↑")
    if fake_dn: votes.append(+2); reasons.append("Fakeout↓")

    score = float(np.tanh(np.mean(votes) if votes else 0.0))  # -1..+1

    # Up-probability (existing heuristic)
    p_up, sc_dir = ai_predict_direction(d)

    # ADX gate
    adx_val = float(last.get("adx", np.nan))
    adx_ok = (np.isfinite(adx_val) and adx_val >= adx_gate)

    # Decision
    action = "HOLD"
    if score > 0.15:
        if (p_up >= p_up_buy_gate) and adx_ok and (brk_up or near_sup_atr < 0.6 or last["close"] > last.get("bb_mid", last["close"])) and not fake_up:
            action = "BUY"; reasons.append(f"BUY ok: p_up={p_up:.2f}, ADX={adx_val:.1f}")
        else:
            reasons.append(f"Blocked BUY: p_up={p_up:.2f}, ADX={adx_val:.1f}, fake_up={fake_up}")
    elif score < -0.15:
        if (p_up <= p_up_sell_gate) and adx_ok and (brk_dn or near_res_atr < 0.6 or last["close"] < last.get("bb_mid", last["close"])) and not fake_dn:
            action = "SELL"; reasons.append(f"SELL ok: p_up={p_up:.2f}, ADX={adx_val:.1f}")
        else:
            reasons.append(f"Blocked SELL: p_up={p_up:.2f}, ADX={adx_val:.1f}, fake_dn={fake_dn}")

    # Confidence (masukkan drawdown & risk aversion) — boosted for trend alignment (helps 1H trending up reach ≥70%)
    conf_raw = float(abs(score))
    adx_norm = 0.0 if not np.isfinite(adx_val) else min(1.0, max(0.0, (adx_val - 10.0) / 35.0))

    # Trend-aligned factor (0..1): favors EMA trend, MACD sign, VWAP side, and ADX strength in the direction of action
    try:
        ema_up = float(last.get("ema20", 0)) > float(last.get("ema50", 0))
        ema_dn = float(last.get("ema20", 0)) < float(last.get("ema50", 0))
        macd_pos = float(last.get("macd_hist", 0)) > 0
        macd_neg = float(last.get("macd_hist", 0)) < 0
        above_vwap = float(last.get("close", 0)) > float(last.get("vwap", last.get("close", 0)))
        below_vwap = float(last.get("close", 0)) < float(last.get("vwap", last.get("close", 0)))
        if action == "BUY":
            trend_factor = (0.4 * float(ema_up) + 0.2 * float(macd_pos) + 0.2 * float(above_vwap) + 0.2 * adx_norm)
        elif action == "SELL":
            trend_factor = (0.4 * float(ema_dn) + 0.2 * float(macd_neg) + 0.2 * float(below_vwap) + 0.2 * adx_norm)
        else:
            trend_factor = 0.0
    except Exception:
        trend_factor = 0.0

    # Small bonus if HTF bias matches the action
    htf_bonus = 0.05 if ((action == "BUY" and htf_bias == "bull") or (action == "SELL" and htf_bias == "bear")) else 0.0

    # Re-weight confidence to include trend_factor and htf_bonus
    conf = 100 * min(
        1.0,
        0.30 * conf_raw +
        0.35 * abs(p_up - 0.5) * 2.0 +
        0.20 * adx_norm +
        0.15 * float(trend_factor) +
        float(htf_bonus)
    )

    # Tambahkan snapshot performa (winrate & DD) ke alasan
    ps = perf_snapshot()
    reasons.append(
        f"Perf: winrate_pos={ps['winrate_pos']:.1f}% on {ps['trades']} trades; "
        f"wr_buy={ps['winrate_action_buy']:.1f}%, wr_sell={ps['winrate_action_sell']:.1f}%, "
        f"DD={ps['drawdown_pct']:.1f}%"
    )

    return action, int(round(conf)), "; ".join(reasons), float(score)


def derive_tp_sl(df: pd.DataFrame, side: str, tp_mult: float = TP_ATR_MULT, sl_mult: float = SL_ATR_MULT):
    d = df.copy()
    d["atr"] = atr_cached(d, 14)
    last = d.iloc[-1]
    a = float(last["atr"] or 0.0) if not np.isnan(last["atr"]) else max(1e-6, float(d["close"].iloc[-1]) * 0.002)
    px = float(last["close"])
    if side == "long":
        tp = px + tp_mult * a
        sl = px - sl_mult * a
    else:
        tp = px - tp_mult * a
        sl = px + sl_mult * a
    return tp, sl

# ---- MTF-aware TP/SL for wide targets & structural stops ----
def recent_swings(di: pd.DataFrame, left: int = 3, right: int = 3):
    """Return (last_swing_low, last_swing_high) prices or (None,None)."""
    try:
        H = di["high"].values; L = di["low"].values
        idx_high, idx_low = [], []
        for i in range(left, len(di)-right):
            if H[i] == max(H[i-left:i+right+1]): idx_high.append(i)
            if L[i] == min(L[i-left:i+right+1]): idx_low.append(i)
        sw_hi = float(di["high"].iloc[idx_high[-1]]) if idx_high else None
        sw_lo = float(di["low"].iloc[idx_low[-1]]) if idx_low else None
        return sw_lo, sw_hi
    except Exception:
        return None, None

def _tf_big_move_atr_mult(tf: str) -> float:
    """ATR multiples for far TP depending on timeframe (bigger TF = farther TP)."""
    m = {
        "1m": 3.0, "3m": 3.5, "5m": 4.0, "15m": 5.0, "30m": 5.5,
        "1h": 6.0, "2h": 6.5, "4h": 7.5, "1d": 9.0
    }
    return float(m.get(tf, 5.0))

def _tf_sl_floor_atr(tf: str) -> float:
    """Minimum SL distance in ATR to avoid too-tight stops (structural)."""
    m = {"1m":1.2, "3m":1.3, "5m":1.5, "15m":1.8, "30m":1.9, "1h":2.0, "2h":2.1, "4h":2.2, "1d":2.5}
    return float(m.get(tf, 1.8))

def derive_tp_sl_mtf(di: pd.DataFrame, hti: pd.DataFrame | None, side: str, tf: str, rr_min: float = 1.8):
    """
    Multi-timeframe TP/SL:
      - SL: below/above recent swing and nearest S/R with padding, and at least `_tf_sl_floor_atr(tf)` ATR away.
      - TP: next HTF resistance/support if available; else ATR extension `_tf_big_move_atr_mult(tf)`.
      - Returns (tp, sl, rr). If any issue, falls back to classic ATR-based `derive_tp_sl`.
    """
    try:
        px = float(di["close"].iloc[-1])
        atr14 = float(atr_cached(di, 14).iloc[-1])
        if not np.isfinite(atr14) or atr14 <= 0:
            atr14 = max(1e-6, px * 0.002)

        # Collect LTF & HTF S/R levels
        sups_l, ress_l = swing_levels(di)
        if hti is not None and not hti.empty:
            sups_h, ress_h = swing_levels(hti)
        else:
            sups_h, ress_h = [], []
        sups_all = sorted(set([*sups_l, *sups_h]))
        ress_all = sorted(set([*ress_l, *ress_h]))

        # Recent structural swings
        sw_lo, sw_hi = recent_swings(di)
        pad = 0.25 * atr14  # small structural padding
        sl_floor = _tf_sl_floor_atr(tf) * atr14

        if side == "long":
            sup_below = max([s for s in sups_all if s < px], default=None)
            sl_candidates = []
            if sw_lo is not None: sl_candidates.append(sw_lo - pad)
            if sup_below is not None: sl_candidates.append(sup_below - pad)
            sl = min(sl_candidates) if sl_candidates else (px - sl_floor)
            # Enforce minimum distance
            if (px - sl) < sl_floor:
                sl = px - sl_floor
            # TP via next HTF resistance or ATR extension
            res_above = min([r for r in ress_all if r > px], default=None)
            if res_above is not None:
                tp = float(res_above)
            else:
                tp = px + _tf_big_move_atr_mult(tf) * atr14
        else:
            res_above = min([r for r in ress_all if r > px], default=None)
            sl_candidates = []
            if sw_hi is not None: sl_candidates.append(sw_hi + pad)
            if res_above is not None: sl_candidates.append(res_above + pad)
            sl = max(sl_candidates) if sl_candidates else (px + sl_floor)
            if (sl - px) < sl_floor:
                sl = px + sl_floor
            sup_below = max([s for s in sups_all if s < px], default=None)
            if sup_below is not None:
                tp = float(sup_below)
            else:
                tp = px - _tf_big_move_atr_mult(tf) * atr14

        # Final sanity + RR gating
        rr = None
        try:
            r = abs(px - sl)
            rr = abs(tp - px) / r if r > 1e-12 else None
        except Exception:
            rr = None
        if rr is None or not np.isfinite(rr):
            tp_fallback, sl_fallback = derive_tp_sl(di, side)
            return float(tp_fallback), float(sl_fallback), None
        # If RR too small, push TP out to meet rr_min (do NOT pull SL tighter)
        if rr < rr_min:
            if side == "long":
                tp = px + rr_min * abs(px - sl)
            else:
                tp = px - rr_min * abs(px - sl)
            rr = rr_min
        return float(tp), float(sl), float(rr)
    except Exception:
        tp_fallback, sl_fallback = derive_tp_sl(di, side)
        return float(tp_fallback), float(sl_fallback), None

def find_order_blocks(di: pd.DataFrame, lookback: int = 40, atr_mult: float = 1.0, max_backtrack: int = 6):
    """
    Lightweight LuxAlgo-like Order Block approximation:
      - Bullish OB: last down candle before a bullish displacement that breaks prior HH in `lookback` bars
      - Bearish OB: last up candle before a bearish displacement that breaks prior LL in `lookback` bars
      - Displacement requires body dominance and range ≥ atr_mult * ATR(14)
    Returns list of dicts: {"type":"bull"/"bear","idx":j,"low":low_j,"high":high_j,"ts":TimestampUTC}
    """
    if di is None or di.empty or len(di) <= lookback + 5:
        return []
    d = di.tail(int(min(len(di), lookback*OB_SCAN_FACTOR + 80))).reset_index(drop=True)
    d["atr14"] = atr_cached(d, 14).bfill().fillna(0)

    O = d["open"].values
    H = d["high"].values
    L = d["low"].values
    C = d["close"].values

    n = len(d)
    if n <= lookback + 2:
        return []

    # Precompute rolling extremes of the *previous* window (exclude current bar)
    roll_max_prev = pd.Series(H).rolling(lookback, min_periods=lookback).max().shift(1).to_numpy()
    roll_min_prev = pd.Series(L).rolling(lookback, min_periods=lookback).min().shift(1).to_numpy()

    is_down = C < O
    is_up   = C > O

    res = []
    for i in range(lookback, n):
        atr_i = float(d["atr14"].iloc[i] or 0.0)
        if atr_i <= 0:
            continue
        rng_i  = float(H[i] - L[i])
        body_i = float(abs(C[i] - O[i]))
        if rng_i <= 0:
            continue

        broke_HH = np.isfinite(roll_max_prev[i]) and (H[i] >= roll_max_prev[i])
        broke_LL = np.isfinite(roll_min_prev[i]) and (L[i] <= roll_min_prev[i])

        # Displacement thresholds
        body_ok = (body_i >= 0.55 * rng_i)
        atr_ok  = (rng_i >= atr_mult * atr_i)

        # Bullish OB: green displacement breaking prior HH
        if is_up[i] and body_ok and atr_ok and broke_HH:
            # last down candle within `max_backtrack`
            j = None
            k0 = max(lookback, i - max_backtrack)
            seg = np.where(is_down[k0:i])[0]
            if seg.size:
                j = int(k0 + seg[-1])
            if j is not None:
                res.append({
                    "type":"bull",
                    "idx": j,
                    "low": float(L[j]),
                    "high": float(H[j]),
                    "ts": pd.to_datetime(d["datetime"].iloc[j])
                })

        # Bearish OB: red displacement breaking prior LL
        if (not is_up[i]) and body_ok and atr_ok and broke_LL:
            j = None
            k0 = max(lookback, i - max_backtrack)
            seg = np.where(is_up[k0:i])[0]
            if seg.size:
                j = int(k0 + seg[-1])
            if j is not None:
                res.append({
                    "type":"bear",
                    "idx": j,
                    "low": float(L[j]),
                    "high": float(H[j]),
                    "ts": pd.to_datetime(d["datetime"].iloc[j])
                })

    # Merge overlapping zones, keep most recent
    def _overlap(a,b):
        return not (a["high"] <= b["low"] or a["low"] >= b["high"])

    out = []
    for z in sorted(res, key=lambda x: x["idx"]):
        if not out:
            out.append(z); continue
        last = out[-1]
        if (z["type"] == last["type"]) and _overlap(z, last):
            last["low"]  = min(last["low"],  z["low"])
            last["high"] = max(last["high"], z["high"])
            last["idx"]  = max(last["idx"],  z["idx"])
            last["ts"]   = max(last["ts"],   z["ts"])
        else:
            out.append(z)

    bulls = [z for z in out if z["type"] == "bull"][-3:]
    bears = [z for z in out if z["type"] == "bear"][-3:]
    return [*bulls, *bears]

def near_order_block(price: float, side: str, zones: list, atr_val: float, pad_atr: float = 0.25) -> tuple[bool, dict|None]:
    if not zones:
        return False, None
    atr_safe = float(atr_val if np.isfinite(atr_val) and atr_val > 0 else 1e-6)
    pad = abs(float(pad_atr)) * atr_safe
    t = "bull" if side == "long" else "bear"
    cand = [z for z in zones if (z.get("type") == t)]
    if not cand:
        return False, None
    z = sorted(cand, key=lambda x: x.get("idx", 0))[-1]
    lo = float(z["low"])  - pad
    hi = float(z["high"]) + pad
    ok = (price >= lo) and (price <= hi)
    return ok, z

# -------- Dynamic leverage & sizing helpers --------
@lru_cache(maxsize=512)
def get_max_leverage(symbol: str) -> int:
    """Best-effort read of max leverage from market metadata; fallback via env."""
    try:
        m = EX.market(symbol)
        # try standard CCXT shape first
        lev = None
        try:
            lev = m.get("limits",{}).get("leverage",{}).get("max")
        except Exception:
            lev = None
        if lev is None:
            try:
                lev = m.get("leverage",{}).get("max")
            except Exception:
                lev = None
        if lev is not None:
            return int(max(1, min(125, float(lev))))
    except Exception:
        pass
    return int(MAX_LEV_FALLBACK)

def atr_percent(di: pd.DataFrame) -> float:
    """ATR(14) as a fraction of price for volatility-aware decisions."""
    try:
        a = float(atr_cached(di, 14).iloc[-1])
        p = float(di["close"].iloc[-1])
        if not np.isfinite(a) or not np.isfinite(p) or p <= 0:
            return 0.005
        return max(1e-6, a / p)
    except Exception:
        return 0.005

def choose_leverage(symbol: str, timeframe: str, di: pd.DataFrame) -> int:
    """Return leverage to use. If FULL_MARGIN, use exchange max; else scale by volatility."""
    maxlev = get_max_leverage(symbol)
    if FULL_MARGIN:
        return maxlev
    if not DYNAMIC_LEVERAGE:
        return int(LEVERAGE_BASE)
    v = atr_percent(di)
    # Lower volatility -> higher leverage. Very rough buckets.
    if v <= 0.002:
        frac = 1.0
    elif v <= 0.005:
        frac = 0.8
    elif v <= 0.01:
        frac = 0.6
    else:
        frac = 0.4
    dyn = int(max(1, min(maxlev, round(maxlev * frac))))
    # Never go below user baseline
    return max(dyn, int(LEVERAGE_BASE))

def compute_notional_usdt(symbol: str, last_price: float) -> float:
    """Decide notional in USDT for new entries."""
    if FULL_MARGIN:
        fav = fetch_usdt_futures_balance("available")
        try:
            fav = float(fav or 0.0)
        except Exception:
            fav = 0.0
        return max(0.0, fav * MARGIN_USE_PCT)
    return float(BASE_ORDER_USDT)

# --- Amount filters & rounding (min/step safe for partial TP) ---
@lru_cache(maxsize=512)
def get_amount_filters(symbol: str) -> tuple[float, float, float]:
    """
    Return (min_amount, max_amount, step_amount) untuk market.
    Aman kalau beberapa field tidak ada di exchange metadata.
    """
    try:
        m = EX.market(symbol)
        lims = (m.get("limits") or {}).get("amount") or {}
        prec = (m.get("precision") or {})
        min_amt = lims.get("min")
        max_amt = lims.get("max") or float("inf")
        step_amt = lims.get("step")

        # Fallback step dari precision.amount (jumlah desimal)
        if step_amt is None:
            pa = prec.get("amount")
            if isinstance(pa, int) and pa >= 0:
                step_amt = 10 ** (-pa)

        # Fallback min dari info.*
        if min_amt is None:
            info = m.get("info") or {}
            cand = info.get("minTradeNum") or info.get("minSz") or info.get("minSize")
            try:
                min_amt = float(cand)
            except Exception:
                min_amt = 0.0

        return float(min_amt or 0.0), float(max_amt if max_amt is not None else float("inf")), float(step_amt or 0.0)
    except Exception:
        return 0.0, float("inf"), 0.0

def round_amount(symbol: str, amount: float) -> float:
    """Bulatkan amount ke presisi market; cache precision & fallback aman."""
    try:
        return float(EX.amount_to_precision(symbol, amount))
    except Exception:
        pa = PREC_CACHE["amount"].get(symbol)
        if pa is None:
            try:
                m = EX.market(symbol)
                pa = (m.get("precision") or {}).get("amount")
            except Exception:
                pa = None
            PREC_CACHE["amount"][symbol] = pa
        try:
            if isinstance(pa, int) and pa >= 0:
                return float(f"{float(amount):.{pa}f}")
            return float(f"{float(amount):.6f}")
        except Exception:
            return float(amount)

def ai_predict_direction(di: pd.DataFrame) -> tuple[float, float]:
    """Heuristic probability the next leg is UP. Returns (p_up 0..1, score -1..1)."""
    s = di.tail(60).copy()
    try:
        slope = (s["ema20"].iloc[-1] - s["ema20"].iloc[-5]) / 5.0
    except Exception:
        slope = 0.0
    ap = atr_percent(di)
    try:
        bb_mid_ok = float(s["bb_mid"].iloc[-1])
        above_mid = 1.0 if float(s["close"].iloc[-1]) > bb_mid_ok else -1.0
    except Exception:
        above_mid = 0.0
    macdh = float(di["macd_hist"].iloc[-1]) if np.isfinite(di["macd_hist"].iloc[-1]) else 0.0
    st_rel = 1.0 if float(di["stoch_k"].iloc[-1]) > float(di["stoch_d"].iloc[-1]) else -1.0
    # Normalize slope by volatility to avoid scale bias
    slope_norm = 0.0 if ap <= 0 else slope / (ap * 5.0)
    score = 0.0
    score += np.tanh(slope_norm)
    score += np.tanh(macdh) * 0.6
    score += st_rel * 0.3
    score += above_mid * 0.4
    score = float(np.tanh(score / 2.2))
    p_up = 0.5 * (score + 1.0)
    return float(max(0.0, min(1.0, p_up))), score

def ai_explain(symbol, timeframe, action, conf, reason, last_row):
    if not OPENAI_API_KEY:
        return f"Aksi: {action} (Confidence {conf}%). Alasan: {reason}."
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        txt = (
            f"Symbol {symbol} TF {timeframe}. Close={last_row['close']:.6f}. "
            f"EMA20={last_row['ema20']:.6f}, EMA50={last_row['ema50']:.6f}, "
            f"MACD_hist={last_row['macd_hist']:.6f}, RSI={last_row['rsi']:.2f}, "
            f"StochK={last_row['stoch_k']:.2f}, StochD={last_row['stoch_d']:.2f}. "
            f"Decision={action} with {conf}% confidence. Reasons: {reason}."
        )
        msg = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role":"system","content":"You are a concise trading assistant. Avoid hype. Be specific and actionable."},
                {"role":"user","content": txt}
            ],
            temperature=0.3,
            max_tokens=220,
        )
        return msg.choices[0].message.content.strip()
    except Exception:
        return f"Aksi: {action} (Confidence {conf}%). Alasan: {reason}."
    
def _cache_put(cache: dict, key, val, max_items: int = 180):
    cache[key] = val
    if len(cache) > max_items:
        for k in list(cache.keys())[: max(1, len(cache) - max_items)]:
            cache.pop(k, None)

# -------------------- AI Predict Target --------------------
def ai_predict_target(symbol, timeframe, last_row, sups_all, ress_all, atr_val):
    """
    Returns (target_price: float | None, basis: str). Uses OpenAI if available; otherwise a deterministic heuristic.
    """
    try:
        px = float(last_row["close"])
    except Exception:
        return None, "No price"
    # --- Heuristic fallback (also used if OpenAI fails) ---
    def _heuristic():
        # pick nearest R above if bullish bias; nearest S below if bearish; else 2*ATR move
        ema20 = float(last_row.get("ema20", np.nan))
        ema50 = float(last_row.get("ema50", np.nan))
        macdh = float(last_row.get("macd_hist", np.nan))
        bullish = (ema20 > ema50) and (macdh > 0)
        bearish = (ema20 < ema50) and (macdh < 0)
        tgt = None
        if bullish:
            above = [r for r in ress_all if r > px]
            if above:
                tgt = min(above, key=lambda v: v - px)
                return float(tgt), "Heuristic: next resistance"
            return float(px + 2.0 * max(atr_val, 1e-6)), "Heuristic: ATR extension up"
        if bearish:
            below = [s for s in sups_all if s < px]
            if below:
                tgt = max(below, key=lambda v: px - v)
                return float(tgt), "Heuristic: next support"
            return float(px - 2.0 * max(atr_val, 1e-6)), "Heuristic: ATR extension down"
        # sideways
        return float(px), "Heuristic: sideways"
    # If no API key, return heuristic
    if not OPENAI_API_KEY:
        return _heuristic()
    # Prepare concise context for the model
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        # take up to 4 nearest S/R levels for context
        near_res = sorted([r for r in ress_all if r > px])[:4]
        near_sup = sorted([s for s in sups_all if s < px], reverse=True)[:4]
        prompt = {
            "symbol": symbol,
            "timeframe": timeframe,
            "price": px,
            "ema20": float(last_row.get("ema20", np.nan)),
            "ema50": float(last_row.get("ema50", np.nan)),
            "rsi": float(last_row.get("rsi", np.nan)),
            "macd_hist": float(last_row.get("macd_hist", np.nan)),
            "stoch_k": float(last_row.get("stoch_k", np.nan)),
            "stoch_d": float(last_row.get("stoch_d", np.nan)),
            "bb_mid": float(last_row.get("bb_mid", np.nan)),
            "atr": float(atr_val),
            "supports": near_sup,
            "resistances": near_res,
        }
        msg = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Return ONLY one minified JSON with keys target_price (float) and basis (short, <60 chars). No prose."},
                {"role": "user", "content": json.dumps(prompt, separators=(",",":"))}
            ],
            temperature=0.2,
            max_tokens=60,
        )
        raw = msg.choices[0].message.content.strip()
        try:
            obj = json.loads(raw)
            tgt = float(obj.get("target_price"))
            bas = str(obj.get("basis") or "AI")
            # sanity: if absurd, fall back
            if not np.isfinite(tgt) or tgt <= 0:
                return _heuristic()
            return float(tgt), bas
        except Exception:
            # attempt to extract the first number
            import re
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw or "")
            if m:
                return float(m.group()), "AI (parsed)"
            return _heuristic()
    except Exception:
        return _heuristic()
    
# -------------------- Order helpers (live & paper) --------------------
def set_leverage_if_needed(symbol: str, lev: int):
    try:
        key = str(symbol)
        last = LEV_CACHE.get(key)
        if isinstance(last, dict):
            prev_lev = last.get("lev")
            ts = last.get("ts", 0)
        else:
            prev_lev, ts = None, 0
        now = now_ms()
        # only set if changed or older than 5 minutes
        if prev_lev == int(lev) and (now - ts) < 5*60*1000:
            return
        EX.set_leverage(int(lev), symbol, params={"marginMode":"cross"})
        LEV_CACHE[key] = {"lev": int(lev), "ts": now}
    except Exception as e:
        print(f"[TRADE] set_leverage warn: {e}")

def place_market_order(symbol: str, side: str, amount: float, reduce_only: bool=False, lev: int | None = None):
    amount = round_amount(symbol, float(amount))
    if amount <= 0:
        raise ValueError("amount must be > 0")
    if "on" not in (STATE.get("live_toggle") or []):
        print(f"[PAPER] {side} {amount} {symbol}")
        try:
            if not reduce_only:
                STATE["last_entry_ts"] = now_ms()
        except Exception:
            if not reduce_only:
                STATE["last_entry_ts"] = now_ms()
        return {"id": f"paper-{int(time.time()*1000)}", "status": "filled"}
    try:
        params = {"reduceOnly": reduce_only}
        set_leverage_if_needed(symbol, int(lev if lev is not None else LEVERAGE_BASE))
        order = EX.create_order(symbol=symbol, type="market",
                                side=("buy" if side=="long" else "sell"),
                                amount=amount, params=params)
        print(f"[ORDER] {side.upper()} {amount} {symbol} → {order.get('id')}")
        try:
            if not reduce_only:
                STATE["last_entry_ts"] = now_ms()
        except Exception:
            if not reduce_only:
                STATE["last_entry_ts"] = now_ms()
        return order
    except Exception as e:
        print(f"[ORDER] fail {side} {amount} {symbol}: {e}")
        return {"id": None, "status": "error", "error": str(e)}

def close_position_market(symbol: str, pos: dict):
    side = "short" if pos["side"]=="long" else "long"
    amt  = float(pos["amount"])
    return place_market_order(symbol, side, amt, reduce_only=True)

def round_price(symbol: str, price: float) -> float:
    """Bulatkan price ke presisi market; cache precision & fallback aman."""
    try:
        return float(EX.price_to_precision(symbol, price))
    except Exception:
        pp = PREC_CACHE["price"].get(symbol)
        if pp is None:
            try:
                m = EX.market(symbol)
                pp = (m.get("precision") or {}).get("price")
            except Exception:
                pp = None
            PREC_CACHE["price"][symbol] = pp
        try:
            if isinstance(pp, int) and pp >= 0:
                return float(f"{float(price):.{pp}f}")
            return float(f"{float(price):.8f}")
        except Exception:
            return float(price)

def place_limit_order(symbol: str, side: str, amount: float, price: float, reduce_only: bool=False, post_only: bool=True, lev: int | None = None):
    amount = round_amount(symbol, float(amount))
    price  = round_price(symbol, float(price))
    if amount <= 0 or price <= 0:
        raise ValueError("amount/price must be > 0")
    if "on" not in (STATE.get("live_toggle") or []):
        print(f"[PAPER][LMT] {side} {amount} {symbol} @ {price}")
        return {"id": f"paper-lmt-{int(time.time()*1000)}", "status": "open"}
    try:
        params = {"reduceOnly": reduce_only}
        if post_only:
            params["postOnly"] = True
        set_leverage_if_needed(symbol, int(lev if lev is not None else LEVERAGE_BASE))
        order = EX.create_order(symbol=symbol, type="limit",
                                side=("buy" if side=="long" else "sell"),
                                amount=amount, price=price, params=params)
        print(f"[ORDER][LMT] {side.upper()} {amount} {symbol} @ {price} → {order.get('id')}")
        return order
    except Exception as e:
        print(f"[ORDER][LMT] fail {side} {amount} {symbol} @ {price}: {e}")
        return {"id": None, "status": "error", "error": str(e)}

def manage_partial_tp_by_roe(symbol: str, pos: dict, roe_now: float, last_price: float):
    """
    Tutup sebagian posisi ketika ROE melewati threshold yang dikonfigurasi.
    Menggunakan market reduceOnly untuk jamin fill.
    """
    if pos is None:
        return
    steps = DEFAULT_ROE_STEPS
    if not isinstance(pos.get("tp_book"), dict):
        pos["tp_book"] = {}
    # tarik remaining amount live (kalau ada)
    live = fetch_exchange_position(symbol)
    if live and isinstance(live.get("amount"), (int, float)):
        try:
            pos["amount"] = float(live["amount"])
        except Exception:
            pass

    remaining = float(pos.get("amount") or 0.0)
    if remaining <= 0:
        return

    min_amt, _max_amt, step_amt = get_amount_filters(symbol)
    changed = False

    for thr, frac in steps:
        key = f"{thr:.2f}"
        if roe_now >= float(thr) and not pos["tp_book"].get(key):
            qty_raw = remaining * float(frac)
            # Enforce min and step
            qty = max(float(min_amt or 0.0), float(qty_raw))
            if step_amt and step_amt > 0:
                qty = np.floor(qty / step_amt) * step_amt
            qty = min(qty, remaining)
            qty = round_amount(symbol, qty)
            if qty <= 0 or qty > remaining + 1e-12:
                continue
            # Debounce within 1s per threshold
            hit_key = f"hit_ts_{key}"
            now_ms = int(time.time()*1000)
            last_hit = pos["tp_book"].get(hit_key, 0)
            if now_ms - int(last_hit) < 900:
                continue

            close_side = "short" if pos.get("side") == "long" else "long"
            od = place_market_order(symbol, close_side, qty, reduce_only=True, lev=pos.get("lev"))
            if od.get("status") != "error":
                pos["tp_book"][key] = True
                pos["tp_book"][hit_key] = now_ms
                remaining = max(0.0, remaining - qty)
                pos["amount"] = remaining
                STATE["pos"] = pos
                changed = True
                print(f"[TP][ROE] {symbol} hit {thr}% ➜ close {qty} ({frac:.2%}) remaining {remaining}")
    if changed:
        STATE["pos"] = pos

def update_trailing_sl(symbol: str, pos: dict | None, roe_now: float | None, last_price: float | None, di: pd.DataFrame, tf: str):
    """
    Move SL forward when position is in profit.
    Rules:
      - If ROE >= TRAIL_BE_ROE: move SL to breakeven (entry) at minimum.
      - For each TRAIL_STEP_ROE beyond BE, lock SL at (price ± TRAIL_LOCK_ATR * ATR14) in favorable direction.
      - SL never moves backwards.
    """
    if not TRAIL_ENABLE or not pos:
        return
    try:
        roe = float(roe_now) if roe_now is not None else None
        if roe is None or not np.isfinite(roe):
            return
        side = str(pos.get("side"))
        entry = float(pos.get("entry") or 0.0)
        if entry <= 0:
            return
        px = float(last_price or 0.0)
        if px <= 0:
            return

        a = atr_cached(di, 14).iloc[-1]
        try:
            atr14 = float(a if np.isfinite(a) else max(1e-6, px * 0.002))
        except Exception:
            atr14 = max(1e-6, px * 0.002)

        cur_sl = pos.get("sl")
        cur_sl_f = float(cur_sl) if cur_sl is not None else None
        moved = False

        # 1) Breakeven minimal
        if roe >= TRAIL_BE_ROE:
            be = entry
            if side == "long":
                new_sl = be if cur_sl_f is None else max(cur_sl_f, be)
            else:
                new_sl = be if cur_sl_f is None else min(cur_sl_f, be)
            if cur_sl_f is None or (side == "long" and new_sl > cur_sl_f) or (side == "short" and new_sl < cur_sl_f):
                pos["sl"] = float(new_sl)
                cur_sl_f = float(new_sl)
                moved = True

        # 2) Trailing berbasis ATR setelah melewati separuh step pertama
        if roe >= (TRAIL_BE_ROE + 0.5 * TRAIL_STEP_ROE):
            lock = min(float(TRAIL_MAX_LOCK_ATR), float(TRAIL_LOCK_ATR)) * atr14
            if side == "long":
                target_sl = px - lock
                target_sl = max(target_sl, entry)  # jangan di bawah BE
                if cur_sl_f is None or target_sl > float(pos.get("sl", -1e30)):
                    pos["sl"] = float(target_sl)
                    moved = True
            else:
                target_sl = px + lock
                target_sl = min(target_sl, entry)  # jangan di atas BE (untung)
                if cur_sl_f is None or target_sl < float(pos.get("sl", 1e30)):
                    pos["sl"] = float(target_sl)
                    moved = True

        if moved:
            STATE["pos"] = pos
            print(f"[SL][trail] {symbol} → SL {pos['sl']} (roe={roe:.2f}%)")
    except Exception as e:
        print(f"[SL] trailing warn: {e}")

# --- Fetch live position snapshot for a symbol
def fetch_exchange_position(symbol: str) -> dict | None:
    """Return a live position snapshot for the given symbol if it exists, plus lev/unrealized/margin if available. Throttled per-symbol via POS_REFRESH_MS."""
    ttl_ms = int(os.getenv("POS_REFRESH_MS","1000"))
    try:
        now = now_ms()
    except Exception:
        now = int(time.time()*1000)

    ent = POS_CACHE["data"].get(symbol)
    if ent and (now - int(ent.get("ts", 0)) < ttl_ms):
        return ent.get("val")

    result = None
    try:
        poss = EX.fetch_positions([symbol])
        for p in poss or []:
            # amount / side
            amt = None
            for k in ("contracts", "size", "contractsSize", "positionAmt", "amount"):
                v = num(p.get(k))
                if v is not None:
                    amt = v
                    break
            if amt is None or abs(amt) == 0:
                continue

            side = p.get("side") or ("long" if amt > 0 else "short")

            # entry price
            entry = None
            for k in ("entryPrice", "avgEntryPrice", "averagePrice", "entry"):
                v = num(p.get(k))
                if v is not None:
                    entry = v
                    break
            if entry is None:
                try:
                    tk = fetch_ticker_fast(symbol)
                    entry = tk.get("last") or tk.get("mark")
                except Exception:
                    entry = None

            # leverage
            lev = num(p.get("leverage"))
            info = p.get("info") if isinstance(p.get("info"), dict) else {}
            if lev is None:
                lev = num((info or {}).get("leverage") or (info or {}).get("lever"))

            # unrealized PnL & margin
            unreal = None
            for k in ("unrealizedPnl","unrealizedProfit","pnl"):
                v = num(p.get(k))
                if v is not None:
                    unreal = v
                    break
            if unreal is None and isinstance(info, dict):
                for k in ("unrealizedPnl","unrealizedProfit","pnl","upl"):
                    v = num(info.get(k))
                    if v is not None:
                        unreal = v
                        break

            margin = None
            for k in ("initialMargin","margin","collateral","isolatedMargin"):
                v = num(p.get(k))
                if v is not None:
                    margin = v
                    break
            if margin is None and isinstance(info, dict):
                for k in ("initialMargin","margin","marginBalance","isolatedMargin","im"):
                    v = num(info.get(k))
                    if v is not None:
                        margin = v
                        break

            if lev is None:
                try:
                    lev = float(LEVERAGE_BASE)
                except Exception:
                    lev = 5.0

            result = {
                "side": side,
                "amount": abs(float(amt)),
                "entry": float(entry) if entry is not None else None,
                "tp": None, "sl": None,
                "lev": float(lev) if lev is not None else None,
                "unrealized": float(unreal) if unreal is not None else None,
                "margin": float(margin) if margin is not None else None,
            }
            break
    except Exception as e:
        print(f"[POS] fetch warn: {e}")
        result = None

    POS_CACHE["data"][symbol] = {"ts": now, "val": result}
    return result

# -------------------- UI (Dash) --------------------
def resolve_symbol(raw: str) -> str | None:
    if not raw:
        return None
    s = str(raw).strip().upper()
    if s in EX.markets:
        return s
    if "/" in s and ":" not in s:
        cand = s + ":USDT"
        if cand in EX.markets:
            return cand
    if ":" not in s and "/" not in s and s.endswith("USDT") and len(s) > 4:
        base = s[:-4]
        cand = f"{base}/USDT:USDT"
        if cand in EX.markets:
            return cand
    return s if s in EX.markets else None

# =============================================
# HEDGE FUND PROFESSIONAL UI COMPONENTS
# =============================================

def create_logo_header():
    """Professional hedge fund style logo header."""
    return html.Div([
        html.Div([
            html.H1("QUANTUM CAPITAL", 
                   style={
                       "margin": "0",
                       "fontSize": "24px",
                       "fontWeight": "700",
                       "background": f"linear-gradient(45deg, {COLORS['accent']}, {COLORS['info']})",
                       "WebkitBackgroundClip": "text",
                       "WebkitTextFillColor": "transparent",
                       "letterSpacing": "0.5px"
                   }),
            html.P("AI FUTURES TRADING PLATFORM", 
                   style={
                       "margin": "0",
                       "fontSize": "11px",
                       "color": COLORS['text_secondary'],
                       "letterSpacing": "2px",
                       "fontWeight": "500"
                   })
        ]),
        html.Div([
            html.Div(id="connection-status", children=[
                html.Span("●", style={"color": COLORS['success'], "fontSize": "16px", "marginRight": "8px"}),
                html.Span("LIVE", style={"color": COLORS['success'], "fontSize": "12px", "fontWeight": "600"})
            ], style={"display": "flex", "alignItems": "center"})
        ])
    ], style={
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
        "padding": "0 20px",
        "height": "100%"
    })

def create_metric_card(title, value, change=None, trend=None, icon=None):
    """Premium metric card for hedge fund dashboard."""
    change_color = COLORS['success'] if change and change > 0 else COLORS['danger'] if change and change < 0 else COLORS['text_secondary']
    change_icon = "↗" if change and change > 0 else "↘" if change and change < 0 else "→"
    
    return html.Div([
        html.Div([
            html.Div([
                html.H3(title, style={
                    "margin": "0 0 8px 0",
                    "fontSize": "14px",
                    "fontWeight": "600",
                    "color": COLORS['text_secondary'],
                    "textTransform": "uppercase",
                    "letterSpacing": "0.5px"
                }),
                html.Div([
                    html.Span(str(value), style={
                        "fontSize": "28px",
                        "fontWeight": "700",
                        "color": COLORS['text_primary'],
                        "lineHeight": "1"
                    }),
                    html.Div([
                        html.Span(change_icon, style={
                            "fontSize": "16px",
                            "marginRight": "4px",
                            "color": change_color
                        }) if change is not None else "",
                        html.Span(f"{change:+.2f}%" if change is not None else "", style={
                            "fontSize": "14px",
                            "fontWeight": "600",
                            "color": change_color
                        })
                    ], style={
                        "display": "flex",
                        "alignItems": "center",
                        "marginTop": "4px"
                    }) if change is not None else None
                ])
            ], style={"flex": "1"}),
            html.Div(icon, style={
                "fontSize": "24px",
                "color": COLORS['accent'],
                "opacity": "0.7"
            }) if icon else None
        ], style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "flex-start"
        })
    ], style={
        **CARD_STYLE,
        "minHeight": "100px",
        "cursor": "pointer",
        "position": "relative",
        "overflow": "hidden"
    })

def create_sidebar_nav():
    """Professional sidebar navigation."""
    return html.Div([
        # Logo section
        html.Div([
            html.Div("Q", style={
                "width": "40px",
                "height": "40px",
                "borderRadius": "8px",
                "background": f"linear-gradient(45deg, {COLORS['accent']}, {COLORS['info']})",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "fontSize": "20px",
                "fontWeight": "700",
                "color": "white",
                "marginRight": "12px"
            }),
            html.Div([
                html.Div("QUANTUM", style={
                    "fontSize": "14px",
                    "fontWeight": "700",
                    "color": COLORS['text_primary'],
                    "lineHeight": "1"
                }),
                html.Div("CAPITAL", style={
                    "fontSize": "12px",
                    "color": COLORS['text_secondary'],
                    "lineHeight": "1"
                })
            ])
        ], style={
            "display": "flex",
            "alignItems": "center",
            "marginBottom": "40px"
        }),
        
        # Trading controls
        create_trading_controls()
    ], style=SIDEBAR_STYLE)

def create_trading_controls():
    """Advanced trading control panel."""
    return html.Div([
        html.H4("TRADING CONTROLS", style={
            "margin": "0 0 20px 0",
            "fontSize": "14px",
            "fontWeight": "700",
            "color": COLORS['text_primary'],
            "textTransform": "uppercase",
            "letterSpacing": "1px"
        }),
        
        # Symbol selection
        html.Div([
            html.Label("SYMBOL", style={
                "fontSize": "11px",
                "fontWeight": "600",
                "color": COLORS['text_secondary'],
                "marginBottom": "8px",
                "display": "block",
                "textTransform": "uppercase",
                "letterSpacing": "0.5px"
            }),
            dcc.Dropdown(
                id="symbol-input",
                placeholder="Select trading pair...",
                style={
                    "backgroundColor": COLORS['surface'],
                    "border": "none",
                    "borderRadius": "6px"
                }
            )
        ], style={"marginBottom": "20px"}),
        
        # Timeframe selection
        html.Div([
            html.Label("TIMEFRAME", style={
                "fontSize": "11px",
                "fontWeight": "600",
                "color": COLORS['text_secondary'],
                "marginBottom": "8px",
                "display": "block",
                "textTransform": "uppercase",
                "letterSpacing": "0.5px"
            }),
            dcc.Dropdown(
                id="timeframe-input",
                options=[
                    {"label": "1M", "value": "1m"},
                    {"label": "5M", "value": "5m"},
                    {"label": "15M", "value": "15m"},
                    {"label": "1H", "value": "1h"},
                    {"label": "4H", "value": "4h"},
                    {"label": "1D", "value": "1d"}
                ],
                value="5m",
                style={
                    "backgroundColor": COLORS['surface'],
                    "border": "none",
                    "borderRadius": "6px"
                }
            )
        ], style={"marginBottom": "20px"}),
        
        # Live trading toggle
        html.Div([
            html.Label("TRADING MODE", style={
                "fontSize": "11px",
                "fontWeight": "600",
                "color": COLORS['text_secondary'],
                "marginBottom": "12px",
                "display": "block",
                "textTransform": "uppercase",
                "letterSpacing": "0.5px"
            }),
            dcc.Checklist(
                id="live-toggle",
                options=[{"label": "Enable Live Trading", "value": "on"}],
                style={
                    "color": COLORS['text_primary']
                }
            )
        ])
    ], style={
        **CARD_STYLE,
        "marginTop": "20px"
    })

# Initialize Dash app with professional theme
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Quantum Capital - AI Trading Platform"
server = app.server

# Disable dev tools for production-like experience
try:
    app.enable_dev_tools(dev_tools_hot_reload=False, dev_tools_ui=False)
except Exception:
    pass

def get_symbol_options(limit=60):
    mkts = [m for m in EX.markets.values() if m.get("type")=="swap" and m.get("quote")=="USDT" and m.get("linear",True)]
    syms = [m["symbol"] for m in mkts]
    try:
        tick = EX.fetch_tickers(syms)
        rows = []
        for s,t in tick.items():
            vol = t.get("quoteVolume") or t.get("baseVolume") or 0
            rows.append((s, float(vol or 0)))
        syms = [s for s,_ in sorted(rows, key=lambda x:x[1], reverse=True)][:limit]
    except Exception:
        syms = sorted(syms)[:limit]
    return [{"label": s.replace(":USDT",""), "value": s} for s in syms]

# Professional hedge fund layout with custom CSS
app.layout = html.Div([
    # Global stores
    dcc.Store(id="data-store"),
    dcc.Store(id="pos-store"),
    dcc.Interval(id="timer", interval=1000, disabled=True),
    
    # Custom CSS injection for professional styling
    html.Style('''
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    body {
        margin: 0;
        padding: 0;
        background: linear-gradient(135deg, #0A0E27 0%, #0F1329 100%) !important;
        color: #FFFFFF;
        overflow-x: hidden;
    }
    
    .professional-grid {
        display: grid;
        gap: 16px;
        animation: fadeIn 0.6s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 48px rgba(0, 212, 255, 0.15) !important;
    }
    
    .glass-effect {
        backdrop-filter: blur(20px);
        border: 1px solid rgba(180, 188, 208, 0.1);
    }
    
    .dropdown .Select-control {
        background-color: #141629 !important;
        border: 1px solid #2A2D47 !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
    }
    
    .dropdown .Select-menu-outer {
        background-color: #1A1D3A !important;
        border: 1px solid #2A2D47 !important;
        border-radius: 8px !important;
    }
    '''),
    
    # Main layout container
    html.Div([
        # Professional header bar
        html.Div(
            create_logo_header(),
            style={
                **HEADER_STYLE,
                "gridColumn": "1 / -1"
            }
        ),
        
        # Sidebar navigation
        html.Div(
            create_sidebar_nav(),
            style={
                "gridRow": "2",
                "gridColumn": "1"
            }
        ),

        # Main content area
        html.Div([
            # Premium metrics dashboard
            html.Div([
                html.H3("MARKET OVERVIEW", style={
                    "margin": "0 0 24px 0",
                    "fontSize": "16px",
                    "fontWeight": "700",
                    "color": COLORS['text_primary'],
                    "textTransform": "uppercase",
                    "letterSpacing": "1px",
                    "borderBottom": f"2px solid {COLORS['accent']}",
                    "paddingBottom": "8px",
                    "display": "inline-block"
                }),
                
                # Professional metrics grid
                html.Div([
                    create_metric_card("CURRENT PRICE", "", None, None, "💰"),
                    create_metric_card("USDT BALANCE", "", None, None, "💳"),
                    create_metric_card("STRATEGY SIGNAL", "WAIT", None, None, "🎯"),
                    create_metric_card("AI CONFIDENCE", "--", None, None, "🧠"),
                    create_metric_card("POSITION", "NONE", None, None, "📈"),
                    create_metric_card("RISK:REWARD", "--", None, None, "⚖️"),
                    create_metric_card("PROGRESS", "--", None, None, "📉"),
                    create_metric_card("AI TARGET", "--", None, None, "🎯")
                ], style={
                    **GRID_4COL,
                    "gridTemplateColumns": "repeat(auto-fit, minmax(200px, 1fr))",
                    "marginBottom": "32px"
                }, className="professional-grid"),
    
                # Professional overlay controls and AI analysis
                html.Div([
                    # Left: Advanced Controls
                    html.Div([
                        html.H4("CHART OVERLAYS", style={
                            "margin": "0 0 16px 0",
                            "fontSize": "14px",
                            "fontWeight": "700",
                            "color": COLORS['text_primary'],
                            "textTransform": "uppercase",
                            "letterSpacing": "1px"
                        }),
                        
                        html.Div([
                            dcc.Checklist(
                                id="overlay_opts",
                                options=[
                                    {"label": "VWAP", "value": "vwap"},
                                    {"label": "PARABOLIC SAR", "value": "psar"},
                                    {"label": "FIBONACCI", "value": "fib"},
                                    {"label": "ELLIOTT WAVE", "value": "ew"},
                                    {"label": "BREAKOUT/FAKEOUT", "value": "brkfake"}
                                ],
                                value=["vwap", "psar", "fib", "ew", "brkfake"],
                                style={
                                    "color": COLORS['text_primary'],
                                    "fontSize": "12px",
                                    "fontWeight": "500"
                                },
                                inputStyle={"margin": "0 8px 0 0"},
                                labelStyle={"display": "block", "margin": "8px 0"}
                            )
                        ]),
                        
                        html.Div([
                            html.H5("MARKET SNAPSHOT", style={
                                "margin": "20px 0 12px 0",
                                "fontSize": "12px",
                                "fontWeight": "600",
                                "color": COLORS['text_secondary'],
                                "textTransform": "uppercase",
                                "letterSpacing": "0.5px"
                            }),
                            dcc.Markdown(
                                id="top_metrics", 
                                style={
                                    "whiteSpace": "pre-wrap",
                                    "fontFamily": "'JetBrains Mono', monospace",
                                    "fontSize": "11px",
                                    "lineHeight": "1.6",
                                    "color": COLORS['text_secondary'],
                                    "background": COLORS['surface'],
                                    "padding": "12px",
                                    "borderRadius": "8px",
                                    "border": f"1px solid {COLORS['border']}"
                                }
                            )
                        ])
                    ], style={**CARD_STYLE, "flex": "1"}),
                    
                    # Right: AI Analysis
                    html.Div([
                        html.H4("AI MARKET ANALYSIS", style={
                            "margin": "0 0 16px 0",
                            "fontSize": "14px",
                            "fontWeight": "700",
                            "color": COLORS['text_primary'],
                            "textTransform": "uppercase",
                            "letterSpacing": "1px"
                        }),
                        
                        html.Div([
                            html.Div(
                                id="ai-reco", 
                                style={
                                    "whiteSpace": "pre-wrap",
                                    "lineHeight": "1.6",
                                    "fontSize": "12px",
                                    "color": COLORS['text_secondary'],
                                    "background": COLORS['surface'],
                                    "padding": "16px",
                                    "borderRadius": "8px",
                                    "border": f"1px solid {COLORS['border']}",
                                    "minHeight": "200px"
                                }
                            )
                        ])
                    ], style={**CARD_STYLE, "flex": "1"})
                ], style={
                    "display": "flex",
                    "gap": "20px",
                    "marginBottom": "24px"
                }),

                # Professional technical analysis section
                html.Div([
                    html.H3("TECHNICAL INDICATORS", style={
                        "margin": "0 0 20px 0",
                        "fontSize": "16px",
                        "fontWeight": "700",
                        "color": COLORS['text_primary'],
                        "textTransform": "uppercase",
                        "letterSpacing": "1px",
                        "borderBottom": f"2px solid {COLORS['accent']}",
                        "paddingBottom": "8px",
                        "display": "inline-block"
                    }),
                    html.Div(
                        id="ta-cards", 
                        style={
                            **GRID_4COL,
                            "gridTemplateColumns": "repeat(auto-fit, minmax(150px, 1fr))",
                            "marginBottom": "24px"
                        },
                        className="professional-grid"
                    ),
                    # Professional chart with advanced configuration
                    html.Div([
                        dcc.Graph(
                            id="chart",
                            config={
                                "displaylogo": False,
                                "scrollZoom": True,
                                "doubleClick": "reset",
                                "responsive": True,
                                "displayModeBar": "hover",
                                "modeBarButtonsToRemove": ['lasso2d', 'select2d'],
                                "modeBarButtonsToAdd": ["autoScale2d", "resetScale2d"],
                                "toImageButtonOptions": {
                                    "format": "png",
                                    "filename": "quantum_capital_chart",
                                    "height": 800,
                                    "width": 1200,
                                    "scale": 2
                                }
                            },
                            style={
                                "height": "700px",
                                "backgroundColor": "transparent",
                                "borderRadius": "12px"
                            }
                        )
                    ], style={
                        **CARD_STYLE,
                        "padding": "20px",
                        "marginBottom": "20px"
                    })
                ])
            ], style={
                "gridRow": "2",
                "gridColumn": "2", 
                "padding": "20px",
                "overflowY": "auto",
                "background": f"linear-gradient(135deg, {COLORS['primary']}, #0F1329)"
            })
        ]
    ], style={
        **DASHBOARD_LAYOUT,
        "fontFamily": "'Inter', -apple-system, BlinkMacSystemFont, sans-serif"
    }),

    # Hidden elements for backward compatibility 
    html.Div([
        html.Div(id="price", style={"display": "none"}),
        html.Div(id="bal", style={"display": "none"}),
        html.Div(id="sig", style={"display": "none"}),
        html.Div(id="aisig", style={"display": "none"}),
        html.Div(id="pos", style={"display": "none"}),
        html.Div(id="rr", style={"display": "none"}),
        html.Div(id="progress", style={"display": "none"}),
        html.Div(id="ai-target", style={"display": "none"}),
        html.Div(id="status", style={"display": "none"}),
        html.Div(id="conn", style={"display": "none"}),
        html.Button(id="start", style={"display": "none"}),
        html.Button(id="stop", style={"display": "none"}),
        dcc.Dropdown(id="symbol", style={"display": "none"}),
        dcc.Dropdown(id="tf", style={"display": "none"}),
        dcc.Checklist(id="live", style={"display": "none"})
    ])
])

app.config.suppress_callback_exceptions = True

tick_gc()

# -------------------- Callbacks --------------------
@app.callback(
    Output("timer","disabled"),
    Output("timer","n_intervals"),
    Output("status","children"),
    Output("start","style"),
    Output("stop","style"),
    Input("start","n_clicks"), Input("stop","n_clicks"),
    State("symbol","value"), State("tf","value"),
    prevent_initial_call=True
)

def start_stop(n_start, n_stop, sym, tf):
    print(f"[DASH] start_stop triggered by: {ctx.triggered_id}")
    # base styles
    start_base = {"width":"100%","marginBottom":"6px","background":"#2b78e4","color":"white","height":"40px","borderRadius":"8px"}
    stop_base  = {"width":"100%","background":"#e06666","color":"white","height":"40px","borderRadius":"8px"}

    trig = ctx.triggered_id
    if trig == "start":
        STATE["force_stop"] = False
        if not sym or not tf:
            # keep stopped, show Start only
            start_style = dict(start_base, **{"display":"block"})
            stop_style  = dict(stop_base,  **{"display":"none"})
            return True, no_update, "Stopped (pilih symbol & timeframe dulu)", start_style, stop_style
        # go running, show Stop only
        start_style = dict(start_base, **{"display":"none"})
        stop_style  = dict(stop_base,  **{"display":"block"})
        return False, 0, f"Running — {resolve_symbol(sym) or sym} @ {tf} (WIB)", start_style, stop_style

    if trig == "stop":
        STATE["force_stop"] = True
        STATE["warmup"] = 0
        # Do not clear STATE["pos"] here; keep detected exchange position visible
        start_style = dict(start_base, **{"display":"block"})
        stop_style  = dict(stop_base,  **{"display":"none"})
        return True, 0, "Stopped — bot is idle", start_style, stop_style

    # default: do nothing
    return no_update, no_update, no_update, no_update, no_update

@app.callback(
    Output("pos-store","data"),
    Output("data-store","data"),
    Output("price","children"),
    Output("bal","children"),
    Output("sig","children"),
    Output("aisig","children"),
    Output("ta-cards","children"),
    Output("chart","figure"),
    Output("ai-reco","children"),
    Output("pos","children"),
    Output("rr","children"),
    Output("progress","children"),
    Output("ai-target","children"),
    Output("top_metrics","children"),
    Input("timer","n_intervals"),
    Input("timer","disabled"),
    Input("symbol","value"),
    Input("tf","value"),
    Input("start","n_clicks"),
    Input("stop","n_clicks"),
    State("live","value"),
    State("pos-store","data"),
    State("overlay_opts","value"),
)

def refresh(_tick, timer_disabled, symbol, tf, _n_start, _n_stop, live_value, pos_store, overlay_vals):
    # cache live toggle
    STATE["live_toggle"] = live_value or (["on"] if LIVE_TRADING else [])

    if not symbol or not tf:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="Pick symbol & timeframe", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return (pos_store or STATE.get("pos")), no_update, "--", "--", "--", "--", [], empty_fig, "—", "—", "—", "—", "—", "—"

    norm = resolve_symbol(symbol)
    if not norm:
        fig = go.Figure()
        fig.add_annotation(text="Unknown symbol. Please pick from the list.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return (pos_store or STATE.get("pos")), no_update, "--", "—", "—", "—", [], fig, "Symbol tidak dikenali. Pilih dari dropdown.", "—", "—", "—", "—", "—"
    symbol = norm
    np.seterr(all='ignore')
    running = not bool(timer_disabled)
    # arahkan background fetcher dan pastikan aktif
    STATE["watch"] = (symbol, tf)
    ensure_bg_fetcher()
    ensure_news_fetcher()
    
    if not running:
        # keep any detected live position visible, only stop trading logic
        STATE["warmup"] = 0

    # --- Detect live exchange position on this symbol (even if opened outside the bot) ---
    try:
        now_ms = EX.milliseconds()
        last_check = int(STATE.get("pos_last_check", 0) or 0)
        need_check = (now_ms - last_check) > 900  # ~1s untuk ROE realtime
    except Exception:
        need_check = True
        now_ms = 0

    pos = pos_store or STATE.get("pos")
    if need_check:
        live_pos = fetch_exchange_position(symbol)
        STATE["pos_last_check"] = now_ms
        if live_pos:
            if pos is None:
                pos = live_pos
            else:
                # Selalu segarkan field dinamis walau side/amount tidak berubah
                for k in ("unrealized","margin","lev"):
                    v = live_pos.get(k)
                    if v is not None:
                        pos[k] = v
                # Ganti field statis hanya bila berubah signifikan
                if (pos.get("side") != live_pos.get("side") or
                    abs(float(pos.get("amount", 0.0)) - float(live_pos.get("amount", 0.0))) > 1e-12):
                    for k in ("side","amount","entry"):
                        if live_pos.get(k) is not None:
                            pos[k] = live_pos[k]
            STATE["pos"] = pos
        else:
            # No live position detected on exchange
            if pos and "paper-" not in str(pos.get("id","")):
                pos = None
                STATE["pos"] = None

    if ctx.triggered_id == "timer" and running:
        STATE["warmup"] = min(5, int(STATE.get("warmup", 0)) + 1)

    allow_trade = running and (ctx.triggered_id == "timer") and not STATE.get("force_stop") and int(STATE.get("warmup", 0)) >= 2

    # Data (cache-only, non-blocking)
    # Data chart sesuai TF (auto re-request bila kurang/stale)
    df, df_state = ensure_clean_tf_df(symbol, tf, min_bars=max(160, PLOT_BARS+40))
    if df is None:
        df = pd.DataFrame(columns=["datetime","open","high","low","close","volume"])
    _log(f"[DASH] df rows={len(df)} ({df_state}) for {symbol}@{tf}")
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"Re-loading {symbol} @ {tf} …", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return (pos_store or STATE.get("pos")), no_update, "--", "—", "—", "—", [], fig, "Memuat data…", "—", "—", "—", "—", "—"

    df = inject_last_price(df, symbol)
    di = compute_indicators(df, symbol)
    # Detect recent Order Blocks (LuxAlgo-style approximation)
    ob_zones = get_ob_zones_cached(di, tf)
    # price snapshot (used by RR calc and others) — define early
    last_price = float(di["close"].iloc[-1])
    
    # --- Progress text (PnL% dan R multiple) ---
    # --- Progress (ROE realtime) ---
    progress_txt = "—"
    try:
        if pos:
            roe = compute_roe(pos, last_price)
            # simpan R dasar untuk referensi (tetap mendukung trailing/TP R)
            sl_for_r0 = float(pos.get("sl") or pos.get("entry") or last_price)
            r0 = float(pos.get("r0") or abs(float(pos.get("entry")) - sl_for_r0))
            pos["r0"] = r0
            chg = (last_price - float(pos["entry"])) if pos["side"]=="long" else (float(pos["entry"]) - last_price)
            multi = chg / max(1e-12, r0)
            progress_txt = f"ROE {roe:+.2f}% • {multi:.2f}R" if roe is not None else f"{multi:.2f}R"
    except Exception:
        pass
    
    # Overlay state handled below after BB overlays
    
    # --- HTF context for MTF bias ---
    htf = anchor_tf(tf)
    htf_df = fetch_ohlcv_df(symbol, htf, limit=600)
    if not htf_df.empty:
        hti = compute_indicators(htf_df, symbol)
        htf_bias = (
            "bull" if (hti["ema20"].iloc[-1] > hti["ema50"].iloc[-1] and hti["macd_hist"].iloc[-1] > 0)
            else ("bear" if (hti["ema20"].iloc[-1] < hti["ema50"].iloc[-1] and hti["macd_hist"].iloc[-1] < 0) else "flat")
        )
    else:
        hti, htf_bias = None, "flat"

    # --- Combine LTF & HTF S/R and compute ATR-based proximity ---
    sups_ltf, ress_ltf = swing_levels(di)
    if hti is not None:
        sups_htf, ress_htf = swing_levels(hti)
    else:
        sups_htf, ress_htf = [], []
    sups_all = sorted(set([*sups_ltf, *sups_htf]))
    ress_all = sorted(set([*ress_ltf, *ress_htf]))

    di["atr14"] = atr(di, 14)
    _last = di.iloc[-1]
    _atr_val = float(_last["atr14"]) if pd.notna(_last["atr14"]) else max(1e-6, float(_last["close"]) * 0.002)
    _px = float(_last["close"])
    near_sup_atr = min([abs(_px - s) / _atr_val for s in sups_all], default=1e9)
    near_res_atr = min([abs(_px - r) / _atr_val for r in ress_all], default=1e9)
    # --- AI Target Price (cached per last candle) ---
    _ts = pd.to_datetime(di["datetime"].iloc[-1])
    try:
        # if already tz-aware use tz_convert, else tz_localize
        last_candle_ts = int((_ts.tz_convert("UTC") if getattr(_ts, "tzinfo", None) else _ts.tz_localize("UTC")).timestamp())
    except Exception:
        last_candle_ts = int(pd.Timestamp(_ts).timestamp())
    t_key = (symbol, tf, last_candle_ts, "target")
    tgt_val, tgt_note = STATE["ai_cache"].get(t_key, (None, ""))
    if tgt_val is None:
        tgt_val, tgt_note = ai_predict_target(symbol, tf, _last, sups_all, ress_all, _atr_val)
        _cache_put(STATE["ai_cache"], t_key, (tgt_val, tgt_note), max_items=180)

    # subset khusus plotting (lebih ringan), logika tetap pakai 'di' penuh
    vis = di.tail(PLOT_BARS).copy()

    action, conf, reason, score = simple_signal(di, htf_bias, near_sup_atr, near_res_atr)
    # OB as soft feature → adjust confidence only (no gating)
    if OB_SOFT:
        ob_long_ok, _zL = near_order_block(last_price, "long",  ob_zones, _atr_val, pad_atr=OB_PAD_ATR)
        ob_short_ok, _zS = near_order_block(last_price, "short", ob_zones, _atr_val, pad_atr=OB_PAD_ATR)
        ob_delta = 0
        notes = []
        if action == "BUY"  and ob_long_ok:
            ob_delta += OB_BONUS; notes.append("near bull OB → +%d" % OB_BONUS)
        if action == "SELL" and ob_short_ok:
            ob_delta += OB_BONUS; notes.append("near bear OB → +%d" % OB_BONUS)
        if action == "BUY"  and ob_short_ok:
            ob_delta -= OB_BONUS; notes.append("inside bear OB → -%d" % OB_BONUS)
        if action == "SELL" and ob_long_ok:
            ob_delta -= OB_BONUS; notes.append("inside bull OB → -%d" % OB_BONUS)
        conf = int(np.clip(conf + ob_delta, 0, 100))
        if notes:
            reason += "; OB: " + ", ".join(notes)
    # Enhanced AI direction probability (short-horizon, good for 1m)
    p_up, dir_score = ai_predict_direction(di)
    news = STATE.get("news_pulse") or {}

    # Balance (USDT-M Futures) — show both Available (free) and Equity (mark PnL)
    try:
        feq = fetch_usdt_futures_balance("equity")
        fav = fetch_usdt_futures_balance("available")
        def _fmt(x):
            try: return f"${float(x):,.2f}"
            except: return "—"
        bal_txt = f"Avail {_fmt(fav)} | Equity {_fmt(feq)}"
    except Exception as e:
        print(f"[BAL] warn: {e}")
        bal_txt = "—"

    price_txt = f"${di['close'].iloc[-1]:,.6f}"
    sig_txt = f"{action} ({conf}% confidence)"
    ai_txt  = f"{'UP' if p_up>=0.5 else 'DOWN'} ({int(p_up*100)}%)"

    # --- Risk : Reward (current or planned) ---
    rr_txt = "—"
    try:
        if pos:
            entry = float(pos.get("entry"))
            tp_v  = float(pos.get("tp"))
            sl_v  = float(pos.get("sl"))
            denom = abs(entry - sl_v)
            numer = abs(tp_v - entry)
            if denom > 1e-12:
                rr_txt = f"{(numer/denom):.2f}"
        else:
            if action in ("BUY","SELL"):
                side_plan = "long" if action == "BUY" else "short"
                tp_p, sl_p, rr_p = derive_tp_sl_mtf(di, hti, side_plan, tf)
                if rr_p is not None:
                    rr_txt = f"{rr_p:.2f} (plan‑MTF)"
                else:
                    denom = abs(last_price - sl_p)
                    numer = abs(tp_p - last_price)
                    if denom > 1e-12:
                        rr_txt = f"{(numer/denom):.2f} (plan)"
    except Exception:
        rr_txt = "—"

    # --- Auto-trade logic ---
    # `pos` may have been set from exchange detection above

    if allow_trade:
        try:
            if pos is None and conf >= ENTRY_CONF and action in ("BUY","SELL"):
                side = "long" if action == "BUY" else "short"
                # Leverage selection & notional sizing
                lev_use = choose_leverage(symbol, tf, di)
                notional = compute_notional_usdt(symbol, last_price)
                amt = max(1e-8, notional / max(1e-12, last_price))
                tp, sl, rr_plan = derive_tp_sl_mtf(di, hti, side, tf)
                # Gate on minimum RR so entry is only at strong locations
                if rr_plan is None or rr_plan < 1.8:
                    raise Exception("RR below threshold; skip entry")
                od = place_market_order(symbol, side, amt, reduce_only=False, lev=lev_use)
                if od.get("status") != "error":
                    r0 = abs(float(last_price) - float(sl))
                    pos = {
                        "side": side,
                        "amount": float(amt),
                        "entry": float(last_price),
                        "tp": float(tp),
                        "sl": float(sl),
                        "lev": int(lev_use),
                        "r0": float(r0),
                        "tp_book": {}  # NEW: penanda TP bertahap yang sudah dieksekusi
                    }
                    STATE["pos"] = pos
        except Exception as e:
            print(f"[TRADE] entry error: {e}")

        try:
            if pos is not None:
                # ensure TP/SL exist for externally-detected positions
                tp_val = pos.get("tp")
                sl_val = pos.get("sl")
                try:
                    tp_ok = (tp_val is not None) and np.isfinite(float(tp_val))
                    sl_ok = (sl_val is not None) and np.isfinite(float(sl_val))
                except Exception:
                    tp_ok = sl_ok = False
                if not (tp_ok and sl_ok):
                    new_tp, new_sl, _rr = derive_tp_sl_mtf(di, hti, pos["side"], tf)
                    pos["tp"], pos["sl"] = float(new_tp), float(new_sl)
                    if not pos.get("r0"):
                        pos["r0"] = abs(float(pos["entry"]) - float(pos["sl"]))
                    STATE["pos"] = pos
                    if pos.get("r0") is None:
                        try:
                            pos["r0"] = float(abs(float(pos.get("entry")) - float(pos["sl"])))
                        except Exception:
                            pass
                    STATE["pos"] = pos

                    # --- Trailing SL by R multiples (BE di 1R, kunci +1R di 2R) ---
                    try:
                        r0 = float(pos.get("r0") or abs(float(pos["entry"]) - float(pos.get("sl") or pos["entry"])))
                        pos["r0"] = r0
                        chg = (last_price - float(pos["entry"])) if pos["side"] == "long" else (float(pos["entry"]) - last_price)
                        multi = chg / max(1e-12, r0)
                        new_sl = float(pos.get("sl") or pos["entry"])
                        # 1R: breakeven
                        if multi >= 1.0:
                            new_sl = max(new_sl, float(pos["entry"])) if pos["side"] == "long" else min(new_sl, float(pos["entry"]))
                        # 2R: lock +1R
                        if multi >= 2.0:
                            target = float(pos["entry"]) + (r0 if pos["side"] == "long" else -r0)
                            new_sl = max(new_sl, target) if pos["side"] == "long" else min(new_sl, target)
                        if abs(new_sl - float(pos.get("sl") or new_sl)) > 1e-12:
                            pos["sl"] = float(new_sl)
                            STATE["pos"] = pos
                    except Exception as e:
                        print(f"[SL] trailing warn: {e}")
                    
                    if not pos.get("r0"):
                        pos["r0"] = abs(float(pos["entry"]) - float(pos["sl"]))

                # Trailing stop based on R multiples (breakeven at 1R, lock 1R at 2R)
                try:
                    r0 = float(pos.get("r0") or abs(float(pos["entry"]) - float(pos.get("sl") or pos["entry"])))
                    pos["r0"] = r0
                    progress = (last_price - float(pos["entry"])) if pos["side"]=="long" else (float(pos["entry"]) - last_price)
                    tr = int(pos.get("trail", 0) or 0)
                    if progress >= 1.0 * r0 and tr < 1:
                        pos["sl"] = float(pos["entry"])  # move to BE
                        pos["trail"] = 1
                    if progress >= 2.0 * r0 and tr < 2:
                        pos["sl"] = float(pos["entry"]) + (r0 if pos["side"]=="long" else -r0)
                        pos["trail"] = 2
                    STATE["pos"] = pos
                except Exception:
                    pass

                # --- Partial Take-Profit (AI/heuristic plan) ---
                # --- Partial Take-Profit (by ROE or by R) ---
                try:
                    if PARTIAL_TP and pos:
                        # --- ROE-based partial TP ---
                        if ROE_TP_ENABLE:
                            if not pos.get("roe_plan"):
                                pos["roe_plan"] = DEFAULT_ROE_STEPS  # [(50,0.25),(100,0.25),(200,0.25)] via ENV
                                pos["roe_done"] = []

                            roe_now = compute_roe(pos, last_price)
                            
                            try:
                                update_trailing_sl(symbol, pos, float(roe_now), last_price, di, tf)
                            except Exception as e:
                                print(f"[SL] trail call warn: {e}")
                            
                            if roe_now is not None:
                                plan = list(pos.get("roe_plan") or [])
                                done = set(pos.get("roe_done") or [])

                                # market filters & sisa size
                                min_amt, max_amt, step_amt = get_amount_filters(symbol)
                                remaining = float(pos.get("amount") or 0.0)
                                last_thr = max([thr for thr, _ in plan]) if plan else None

                                # jalankan threshold dari kecil ke besar
                                for (thr, frac) in sorted(plan, key=lambda x: float(x[0])):
                                    if thr in done:
                                        continue
                                    if float(roe_now) < float(thr):
                                        continue

                                    # size yang mau direduksi di step ini
                                    take_target = max(0.0, remaining * float(frac))
                                    take_amt = min(remaining, take_target)
                                    take_amt = round_amount(symbol, take_amt)

                                    # hormati min lot/step; untuk step terakhir, ambil semua sisa jika sisa >= min
                                    if take_amt <= 0:
                                        continue
                                    if take_amt < max(min_amt, step_amt or 0.0):
                                        if (last_thr is not None) and (float(thr) == float(last_thr)) and (remaining >= max(min_amt, step_amt or 0.0)):
                                            take_amt = round_amount(symbol, remaining)
                                        else:
                                            # terlalu kecil untuk step ini; jangan tandai "done"
                                            continue

                                    side_close = "short" if pos.get("side") == "long" else "long"
                                    od = place_market_order(symbol, side_close, float(take_amt), reduce_only=True, lev=pos.get("lev"))
                                    if od.get("status") != "error":
                                        remaining = max(0.0, remaining - float(take_amt))
                                        pos["amount"] = remaining
                                        done.add(thr)                          # hindari eksekusi berulang
                                        pos["roe_done"] = list(sorted(done))   # simpan progress
                                        STATE["pos"] = pos
                                        print(f"[TP] ROE {roe_now:.2f}% >= {thr}% → reduce {take_amt} ({float(frac)*100:.1f}% step)")

                                # jika sisa sangat kecil (<0.5 min), ratakan saja biar bersih
                                if 0.0 < remaining < (max(min_amt, step_amt or 0.0) * 0.5):
                                    side_close = "short" if pos.get("side") == "long" else "long"
                                    take_amt = round_amount(symbol, remaining)
                                    od = place_market_order(symbol, side_close, float(take_amt), reduce_only=True, lev=pos.get("lev"))
                                    if od.get("status") != "error":
                                        pos["amount"] = 0.0
                                        STATE["pos"] = pos
                                        print(f"[TP] Remainder dust closed: {take_amt}")

                        # --- Fallback by R-multiple (kalau ROE TP dimatikan) ---
                        elif TP_SCALE_ENV and (pos.get("r0") is not None):
                            try:
                                base_plan = json.loads(TP_SCALE_ENV)   # contoh [(1.0,0.5),(2.0,0.25)]
                            except Exception:
                                base_plan = [(1.0,0.5),(2.0,0.25)]

                            r0 = float(pos.get("r0"))
                            progress = (last_price - float(pos["entry"])) if pos["side"]=="long" else (float(pos["entry"]) - last_price)
                            curr_R = progress / max(1e-12, r0)

                            if not pos.get("r_done"):
                                pos["r_done"] = []
                            doneR = set(pos["r_done"])

                            min_amt, max_amt, step_amt = get_amount_filters(symbol)
                            remaining = float(pos.get("amount") or 0.0)

                            for (Rm, frac) in sorted(base_plan, key=lambda x: float(x[0])):
                                if Rm in doneR:
                                    continue
                                if float(curr_R) < float(Rm):
                                    continue
                                take_target = max(0.0, remaining * float(frac))
                                take_amt = min(remaining, take_target)
                                take_amt = round_amount(symbol, take_amt)
                                if take_amt < max(min_amt, step_amt or 0.0):
                                    continue
                                side_close = "short" if pos["side"] == "long" else "long"
                                od = place_market_order(symbol, side_close, float(take_amt), reduce_only=True, lev=pos.get("lev"))
                                if od.get("status") != "error":
                                    remaining = max(0.0, remaining - float(take_amt))
                                    pos["amount"] = remaining
                                    doneR.add(Rm)
                                    pos["r_done"] = list(sorted(doneR))
                                    STATE["pos"] = pos
                                    print(f"[TP] {curr_R:.2f}R >= {Rm}R → reduce {take_amt} ({float(frac)*100:.1f}% step)")
                except Exception as e:
                    print(f"[TP] warn: {e}")

                try:
                    if float(pos.get("amount", 0.0)) <= 1e-9:
                        pos = None
                        STATE["pos"] = None
                except Exception:
                    pass

                hit_tp = (last_price >= tp_val) if pos["side"] == "long" else (last_price <= tp_val)
                hit_sl = (last_price <= sl_val) if pos["side"] == "long" else (last_price >= sl_val)
                opp_signal = (conf >= max(ENTRY_CONF+10, 70)) and ((action == "SELL" and pos["side"] == "long") or (action == "BUY" and pos["side"] == "short"))
                if hit_tp or hit_sl or opp_signal:
                    od2 = close_position_market(symbol, pos)
                    if od2.get("status") != "error":
                        pos = None
                        STATE["pos"] = None
        except Exception as e:
            print(f"[TRADE] exit error: {e}")
    else:
        # when not running, never auto open/close; keep current pos_store as-is
        pass

    # TA tiles
    last = di.iloc[-1]
    def to_float(x):
        try:
            if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
                return None
            return float(x)
        except Exception:
            return None

    def tile(label, val):
        return html.Div([html.B(f"{label}: "), html.Span(val if val is not None else "-")], className="tiny-tile")

    # Enhanced cards with Elliott Wave data
    ew_direction = last.get('ew_direction', 'sideways')
    ew_pattern = last.get('ew_pattern', 'incomplete')
    ew_strength = to_float(last.get('ew_impulse_strength', 0))
    ew_wave_count = int(to_float(last.get('ew_wave_count', 0)) or 0)
    
    cards = [
        tile("HTF Trend", htf_bias.title()),
        tile("EW Direction", ew_direction.title()),
        tile("EW Pattern", ew_pattern.title()),
        tile("EW Strength", f"{ew_strength:.0f}%" if ew_strength and ew_strength > 0 else "-"),
        tile("Wave Count", str(ew_wave_count) if ew_wave_count > 0 else "-"),
        tile("RSI", f"{to_float(last['rsi']):.2f}" if to_float(last['rsi']) is not None else "-"),
        tile("EMA 20", f"{to_float(last['ema20']):.6f}" if to_float(last['ema20']) is not None else "-"),
        tile("EMA 50", f"{to_float(last['ema50']):.6f}" if to_float(last['ema50']) is not None else "-"),
        tile("MACD", f"{to_float(last['macd_line']):.6f}" if to_float(last['macd_line']) is not None else "-"),
        tile("Volume", f"{to_float(last['volume']):.2f}" if to_float(last['volume']) is not None else "-"),
        tile("BB Width", f"{to_float(last.get('bb_width', np.nan))*100:.2f}%" if to_float(last.get('bb_width', np.nan)) is not None else "-")
    ]

    # Plot fewer bars to keep DOM light; keep full DI for signals
    plotN = int(os.getenv("PLOT_BARS", "300"))
    vis = di.tail(plotN).copy()
    # Plotly x (datetime to WIB)
    xdt = vis["datetime"]
    try:
        xdt = xdt.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)  # show in WIB
    except Exception:
        try:
            xdt = xdt.dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
        except Exception:
            pass

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.03,
                        specs=[[{"secondary_y": False}], [{"secondary_y": False}]])
    green = "#16a34a"  # strong green
    red   = "#ef4444"  # strong red
    fig.add_trace(
        go.Candlestick(
            x=xdt,
            open=vis["open"], high=vis["high"], low=vis["low"], close=vis["close"],
            name="Price",
            showlegend=False,
            increasing=dict(line=dict(color=green, width=1.2), fillcolor=green),
            decreasing=dict(line=dict(color=red,   width=1.2), fillcolor=red),
            whiskerwidth=0.4,
            opacity=1.0,
        ),
        row=1, col=1
    )
    # Close line overlay (TradingView-style smooth path)
    fig.add_trace(
        go.Scatter(
            x=xdt,
            y=vis["close"],
            mode="lines",
            name="Close",
            line=dict(width=1),
            opacity=0.9,
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=xdt, y=vis["ema20"], mode="lines", name="EMA 20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xdt, y=vis["ema50"], mode="lines", name="EMA 50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xdt, y=vis["bb_lower"], mode="lines", name="BB Lower", showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=xdt, y=vis["bb_upper"], mode="lines", name="BB Upper", fill="tonexty", showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=xdt, y=vis["bb_mid"],   mode="lines", name="BB Mid",   showlegend=True), row=1, col=1)

    # === Overlay toggles (VWAP / PSAR / Fib / Break-Fake) ===
    try:
        _ov = set(overlay_vals or [])
    except Exception:
        _ov = {"vwap","psar","fib","brkfake"}

    # VWAP line
    try:
        if ("vwap" in _ov) and ("vwap" in vis.columns) and vis["vwap"].notna().any():
            fig.add_trace(
                go.Scatter(x=xdt, y=vis["vwap"], name="VWAP", mode="lines"),
                row=1, col=1
            )
    except Exception:
        pass

    # PSAR markers
    try:
        if ("psar" in _ov) and ("psar" in vis.columns) and vis["psar"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=xdt,
                    y=vis["psar"],
                    name="PSAR",
                    mode="markers",
                    marker=dict(size=5),
                    showlegend=True,
                ),
                row=1, col=1
            )
    except Exception:
        pass

    # Fibonacci retracement levels (last swing window)
    try:
        if "fib" in _ov:
            fl = fib_levels(vis, lookback=min(240, len(vis)))
            if isinstance(fl, dict) and fl:
                for lvl_key in ("0.236","0.382","0.5","0.618"):
                    if lvl_key in fl and np.isfinite(float(fl[lvl_key])):
                        fig.add_hline(y=float(fl[lvl_key]), line=dict(width=1, dash="dot"),
                                      annotation_text=f"Fib {lvl_key}", row=1, col=1)
    except Exception:
        pass

    # Breakout / Fakeout marker on last bar
    try:
        if "brkfake" in _ov:
            brk_up, brk_dn, fake_up, fake_dn = detect_breakout_fakeout(vis, max(10, BREAKOUT_N*4))
            x_last = xdt.iloc[-1]
            note = "Breakout↑" if brk_up else ("Breakout↓" if brk_dn else ("Fakeout↑" if fake_up else ("Fakeout↓" if fake_dn else None)))
            if note:
                fig.add_vline(x=x_last, line=dict(width=1, dash="dot"), annotation_text=note, row=1, col=1)
    except Exception:
        pass

    # Horizontal last price line + right-side label (TradingView-like)
    _sym_label = (symbol or "").replace(":USDT", "")
    fig.add_hline(
        y=float(last_price),
        line=dict(width=2, dash="solid", color="rgba(255,190,0,0.85)"),
        annotation_text=f"{_sym_label}  {last_price:.6f}",
        annotation_position="right",
        annotation_font=dict(size=12),
        annotation_bgcolor="rgba(255,190,0,0.85)",
    )

    # --- Position overlays: Entry / SL / TP + TP1/TP2 dari scale_plan ---
    try:
        if pos:
            entry = float(pos["entry"])
            tp_v  = float(pos["tp"])
            sl_v  = float(pos["sl"])
            fig.add_hline(y=entry, line_dash="dash",
                        annotation_text=f"Entry {entry:.6f}", annotation_position="left")
            fig.add_hline(y=sl_v, line_dash="dot",
                        annotation_text=f"SL {sl_v:.6f}", annotation_position="left")
            fig.add_hline(y=tp_v, line_dash="dot",
                        annotation_text=f"TP {tp_v:.6f}", annotation_position="left")

            # TP bertahap dari r0 & scale_plan (TP1/TP2/TP3…)
            r0 = float(pos.get("r0") or abs(entry - sl_v))
            plan = pos.get("scale_plan") or DEFAULT_TP_SCALE
            for i, (rm, frac) in enumerate(plan[:3], start=1):
                level = entry + (rm * r0 if pos["side"]=="long" else -rm * r0)
                fig.add_hline(
                    y=float(level),
                    line_dash="dot",
                    opacity=0.5,
                    annotation_text=f"TP{i} {rm}R ({int(frac*100)}%)",
                    annotation_position="right"
                )
    except Exception as _:
        pass

    # S/R
    sups, ress = swing_levels(di)
    for s in sups:
        fig.add_hline(y=s, line=dict(width=1, dash="dot"), row=1, col=1)
    for r in ress:
        fig.add_hline(y=r, line=dict(width=1, dash="dot"), row=1, col=1)
    # HTF S/R overlays (stronger lines)
    if 'hti' in locals() and hti is not None:
        sups_h, ress_h = swing_levels(hti)
        for s in sups_h:
            fig.add_hline(y=s, line=dict(width=2, dash="dash"), row=1, col=1)
        for r in ress_h:
            fig.add_hline(y=r, line=dict(width=2, dash="dash"), row=1, col=1)

    # Order Block zones (visual, shaded to the right)
    try:
        x_last = xdt.iloc[-1]
        # show only the most recent few zones and those that appear in the visible window
        ob_vis = sorted(ob_zones, key=lambda x: x["idx"])[-6:]
        for z in ob_vis:
            z_ts = pd.to_datetime(z["ts"])
            try:
                z_x = z_ts.tz_convert(LOCAL_TZ).tz_localize(None)
            except Exception:
                try:
                    z_x = z_ts.tz_localize("UTC").tz_convert(LOCAL_TZ).tz_localize(None)
                except Exception:
                    z_x = z_ts
            color = "rgba(22,163,74,0.25)" if z["type"] == "bull" else "rgba(239,68,68,0.25)"  # stronger fill
            linec = "rgba(22,163,74,0.80)" if z["type"] == "bull" else "rgba(239,68,68,0.80)"  # stronger border
            fig.add_hrect(
                y0=float(z["low"]), y1=float(z["high"]),
                x0=z_x, x1=x_last,
                line=dict(width=2, color=linec, dash="dot"),
                fillcolor=color, layer="below", row=1, col=1
            )
            # OB label
            label = "Bull OB" if z["type"] == "bull" else "Bear OB"
            fig.add_annotation(
                x=z_x, y=float((float(z["low"]) + float(z["high"])) / 2.0),
                xanchor="left", yanchor="middle",
                text=label, showarrow=False,
                font=dict(size=10, color="#0b1324"),
                bgcolor=("rgba(22,163,74,0.7)" if z["type"]=="bull" else "rgba(239,68,68,0.7)"),
                borderpad=2, opacity=0.95,
                row=1, col=1,
            )
    except Exception as _e:
        pass

    # AI Target overlay
    if tgt_val is not None and np.isfinite(float(tgt_val)):
        fig.add_hline(y=float(tgt_val), line=dict(color="rgba(255,165,0,0.8)", width=2, dash="dot"),
                      annotation_text="AI Target", row=1, col=1)

    # Pattern markers (last 50)
    tail = di.tail(50)
    tail_xdt = tail["datetime"]
    try:
        tail_xdt = tail_xdt.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    except Exception:
        try:
            tail_xdt = tail_xdt.dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
        except Exception:
            pass
    def add_marks(mask, text):
        mk = tail[mask].copy()
        if mk.empty: return
        mk_xdt = mk["datetime"]
        try:
            mk_xdt = mk_xdt.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
        except Exception:
            try:
                mk_xdt = mk_xdt.dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
            except Exception:
                pass
        fig.add_trace(go.Scatter(
            x=mk_xdt, y=mk["close"], mode="markers", name=text,
            marker=dict(size=9, symbol="triangle-up" if "Bull" in text or "Hammer" in text else "triangle-down"),
            text=[text]*len(mk), hoverinfo="text"), row=1, col=1)
    add_marks(tail["bull_engulf"], "Bull Engulf")
    add_marks(tail["bear_engulf"], "Bear Engulf")
    add_marks(tail["hammer"],      "Hammer")
    add_marks(tail["shooting"],    "Shooting Star")
    add_marks(tail["doji"],        "Doji")

    # Volume + MACD
    fig.add_trace(go.Bar(x=xdt, y=vis["volume"], name="Volume"), row=2, col=1)
    fig.add_trace(go.Scatter(x=xdt, y=vis["macd_line"], name="MACD", mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(x=xdt, y=vis["macd_signal"], name="Signal", mode="lines"), row=2, col=1)
    fig.add_trace(go.Bar(x=xdt, y=vis["macd_hist"], name="Hist", opacity=0.4), row=2, col=1)

    fig.update_layout(
        margin=dict(l=10,r=10,t=10,b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=640,
        xaxis_rangeslider_visible=True
    )
    fig.update_layout(dragmode="pan")
    fig.update_layout(uirevision="chart")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    fig.update_yaxes(tickformat=".6f")

    # Dark theme + vertical zoom/pan on y-axis
    fig.update_layout(template="plotly_dark",
                      paper_bgcolor="#0b1324",
                      plot_bgcolor="#0f172a",
                      font=dict(color="#e5e7eb"))
    fig.update_yaxes(fixedrange=False, rangemode="normal", automargin=True, gridcolor="rgba(255,255,255,0.08)")
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")

    # TradingView-like interaction: unified hover + crosshair spikelines
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1, spikesnap="cursor")
    fig.update_yaxes(showspikes=True, spikemode="across", spikethickness=1)

    # AI explanation (cache per candle)
    last_dt = pd.to_datetime(di["datetime"].iloc[-1])
    try:
        last_dt = last_dt.tz_convert(LOCAL_TZ)
    except Exception:
        try:
            last_dt = last_dt.tz_localize("UTC").tz_convert(LOCAL_TZ)
        except Exception:
            pass
    ai_key = (symbol, tf, int(last_dt.timestamp()))
    
    # Sanitasi teks AI agar tidak duplikatif
    try:
        rec = sanitize_ai_text(rec)
    except Exception:
        pass

    rec = STATE["ai_cache"].get(ai_key)
    if rec is None:
        rec = ai_explain(symbol, tf, action, conf, reason, last)
        STATE["ai_cache"][ai_key] = rec
    if tgt_val is not None and np.isfinite(float(tgt_val)):
        rec = f"{rec}\nTarget AI: {float(tgt_val):.6f} ({tgt_note})"
    
    pos_txt = "—"
    if pos:
        lev_disp = f" x{int(pos.get('lev', LEVERAGE_BASE))}" if pos.get('lev') else ""
        pos_txt = f"{pos['side'].upper()}{lev_disp} {pos['amount']:.6f} @ {pos['entry']:.6f} | TP {pos['tp']:.6f} | SL {pos['sl']:.6f}"

    ai_target_txt = "—"
    try:
        if tgt_val is not None and np.isfinite(float(tgt_val)):
            ai_target_txt = f"{float(tgt_val):,.6f} ({tgt_note})"
    except Exception:
        pass
        ai_target_txt = "—"

    # JSON-safe snapshot for Store (avoid pandas/np scalars)
    snap_dt = pd.to_datetime(di["datetime"].iloc[-1])
    try:
        snap_dt = snap_dt.tz_convert(LOCAL_TZ)
    except Exception:
        try:
            snap_dt = snap_dt.tz_localize("UTC").tz_convert(LOCAL_TZ)
        except Exception:
            pass
    snap = {
        "datetime": snap_dt.isoformat(),
        "open": float(di["open"].iloc[-1]),
        "high": float(di["high"].iloc[-1]),
        "low": float(di["low"].iloc[-1]),
        "close": float(di["close"].iloc[-1]),
        "volume": float(di["volume"].iloc[-1]),
        "rsi": float(last["rsi"]) if np.isfinite(last["rsi"]) else None,
        "ema20": float(last["ema20"]) if np.isfinite(last["ema20"]) else None,
        "ema50": float(last["ema50"]) if np.isfinite(last["ema50"]) else None,
        "macd": float(last["macd_line"]) if np.isfinite(last["macd_line"]) else None,
        "macd_signal": float(last["macd_signal"]) if np.isfinite(last["macd_signal"]) else None,
    }

    # Compose top metrics panel
    try:
        top_md = compose_top_metrics(symbol, tf, di, pos)
    except Exception:
        top_md = "—"

    return pos, snap, price_txt, bal_txt, sig_txt, ai_txt, cards, fig, rec, pos_txt, rr_txt, progress_txt, ai_target_txt, top_md

# -------------------- Styling --------------------
app.index_string = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  {%metas%}
  <title>Crypto Trading Bot (Visual)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/flowbite/2.2.1/flowbite.min.css" rel="stylesheet" />
  <script>tailwind.config = { darkMode: 'class' }</script>
  <style>
    .box-title { font-weight:600; margin-bottom:6px; }
    .panel { background:#ffffff; border-radius:12px; padding:12px; box-shadow:0 2px 8px rgba(0,0,0,0.06); margin-bottom:12px; }
    .tile { background:#ffffff; border-radius:12px; padding:12px; box-shadow:0 2px 8px rgba(0,0,0,0.06); text-align:center; }
    .lbl { font-size:12px; color:#666; }
    .tiny-tile { background:#f5f7fb; border-radius:10px; padding:8px; }
    body { margin:0; }
    #react-entry-point { width: 100%; margin: 0; }
    /* Dark theme overrides */
    .dark .panel { background:#0f172a; color:#e5e7eb; }
    .dark .tile { background:#0b1324; color:#e5e7eb; }
    .dark .tiny-tile { background:#1e293b; color:#e5e7eb; }

    /* Dash Dropdown (react-select) dark overrides */
    .dark .dark-select .Select-control,
    .dark .dark-select .Select-menu-outer,
    .dark .dark-select .Select-menu {
      background-color: #0f172a !important;
      color: #e5e7eb !important;
      border-color: #334155 !important;
    }
    .dark .dark-select .Select-value-label,
    .dark .dark-select .VirtualizedSelectOption,
    .dark .dark-select .Select-option {
      color: #e5e7eb !important;
    }
    .dark .dark-select .Select-placeholder { color: #94a3b8 !important; }
    .dark .dark-select .Select-option.is-focused { background-color: #1e293b !important; }
    .dark .dark-select .Select-option.is-selected { background-color: #334155 !important; }
    .dark .dark-select .Select-arrow { border-top-color: #e5e7eb !important; }
    /* React-Select portal menu (dcc.Dropdown) — global dark overrides */
    .dark .Select-menu-outer,
    .dark .Select-menu {
      background-color: #0f172a !important;
      color: #e5e7eb !important;
      border: 1px solid #334155 !important;
      box-shadow: 0 6px 18px rgba(0,0,0,0.45) !important;
    }
    .dark .VirtualizedSelectOption,
    .dark .Select-option,
    .dark .Select-noresults {
      color: #e5e7eb !important;
    }
    .dark .Select-option.is-focused { background-color: #1e293b !important; color: #e5e7eb !important; }
    .dark .Select-option.is-selected { background-color: #334155 !important; color: #e5e7eb !important; }
    /* keep the menu above charts/panels */
    .dark .Select-menu-outer { z-index: 10000 !important; }
    /* Chart container background hardening */
    .dark #chart { background: #0f172a; }
  </style>
</head>
<body class="dark bg-slate-900 text-slate-200">
  {%app_entry%}
  <footer>
    {%config%}
    {%scripts%}
    <script src="https://cdnjs.cloudflare.com/ajax/libs/flowbite/2.2.1/flowbite.min.js"></script>
    {%renderer%}
    {%favicon%}
  </footer>
</body>
</html>
"""

def open_browser():
    time.sleep(1.0)
    try: webbrowser.open("http://127.0.0.1:8050")
    except: pass

try:
    app.enable_dev_tools(dev_tools_hot_reload=False, dev_tools_ui=False)
except Exception:
    pass

if __name__ == "__main__":
    threading.Thread(target=open_browser).start()
    app.run(debug=True, host="127.0.0.1", port=8050, threaded=True, use_reloader=False)