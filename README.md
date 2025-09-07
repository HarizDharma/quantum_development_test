Quantum Development Test — Tools & Artifacts Backup

This repository preserves development tools and artifacts that are useful for future reference and testing. It intentionally excludes the production Dash app folder (`dash_app/`), which lives in its own dedicated repository.

Included files
- app_dash_backup.py — Legacy/backup version of the Dash app (reference implementation and patterns).
- app_dash.py.backup — Alternate backup of the Dash app code (useful for diffing and recovery).
- ai_futures_autopilot.py (+ copies) — Experimental futures trading autopilot scripts; various iterations kept for comparison and reuse.
- ai_trading_chatgpt_bitget.py — Bitget trading tooling with ChatGPT/OpenAI interaction patterns.
- fibo_wizard_api_with_charts.py — Fibo-based analysis utilities with chart generation.
- helper_functions.txt — Notes and helper snippets used during development.
- test_syntax.py — Quick syntax sanity check script.
- HEDGE_FUND_UI_DOCUMENTATION.md — Hedge fund-grade UI documentation used to guide the app UX redesign.
- OPTIMIZATION_CHANGELOG.md — Notes on performance and memory optimizations applied across iterations.
- models/ — Small model/threshold artifacts used by previous experiments (JSON/PKL).

Not included
- `dash_app/` — The production/minimal app is kept in its own repo.
- Secrets or runtime files (`.env`, logs, caches, venv, state) are intentionally excluded.

Quick notes
- These artifacts are not guaranteed to run as-is; they serve as a reference and source material for future work.
- If you adapt any of the scripts, create a fresh virtual environment and install required libraries (dash, plotly, pandas, numpy, ccxt, openai, etc.) based on the script’s imports.

Safety
- Do not add `.env` or any credentials to this repository.
- Avoid running legacy trading scripts against live exchanges without thorough review.
