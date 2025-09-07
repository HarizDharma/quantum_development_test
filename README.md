Quantum Development Test — Tools & Artifacts Backup

Tujuan
- Arsip rapi alat/skrip pengembangan dan artefak yang berguna untuk referensi/eksperimen.
- Tidak menyertakan folder produksi `dash_app/` (punya repo khusus).
- Tidak menyertakan secrets/artefak runtime (`.env`, venv, logs, cache, state, __pycache__).

Struktur Folder
- legacy/
  - app_dash_backup.py — Backup/versi lama dari Dash app (referensi pola UI/logic).
  - app_dash.py.backup — Alternatif backup Dash app.
- tools/
  - ai_trading_chatgpt_bitget.py — Tool trading Bitget terintegrasi ChatGPT/OpenAI.
  - fibo_wizard_api_with_charts.py — Util analisis Fibonacci + charting.
  - helper_functions.txt — Catatan/snippet bantuan.
- experiments/
  - ai_futures_autopilot*.py — Berbagai iterasi autopilot futures untuk eksperimen.
- models/
  - global_USDTM_*.json / *.pkl — Artefak model/threshold pendukung eksperimen.
- docs/
  - HEDGE_FUND_UI_DOCUMENTATION.md — Panduan UI profesional (hedge fund style).
  - OPTIMIZATION_CHANGELOG.md — Catatan optimisasi performa/memori.
- tests/
  - test_syntax.py — Cek sintaks cepat.

Dependensi per Skrip (ringkas)
- tools/ai_trading_chatgpt_bitget.py
  - ccxt, openai, pandas, TA-Lib, python-dotenv, asyncio
- tools/fibo_wizard_api_with_charts.py
  - numpy, pandas, scikit-learn (UndefinedMetricWarning), math/warnings (built-in)
- legacy/app_dash_backup.py dan legacy/app_dash.py.backup
  - dash, plotly, pandas, numpy, ccxt, requests, python-dotenv, urllib3
- experiments/ai_futures_autopilot*.py
  - numpy, pandas, (beberapa versi menggunakan) requests

Instalasi Cepat (opsional untuk eksplorasi)
- Buat venv Python 3.11, lalu:
  - `pip install -r requirements-dev.txt`
  - Atau instal paket sesuai kebutuhan setiap skrip.

Catatan Keamanan
- Jangan commit `.env` atau kredensial.
- Hindari menjalankan skrip trading legacy pada mode live tanpa review menyeluruh.

Catatan
- Artefak di repo ini bersifat referensi; tidak dijamin jalan “as‑is”.
- Untuk penggunaan produksi, gunakan repo khusus `dash_app/` (dashboard minimal).
