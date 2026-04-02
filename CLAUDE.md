# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ERMTool (Equity Research Management Tool) — a Django web app for managing stock research, portfolio analysis, and trading strategy backtesting. Users create research posts per ticker, vote on portfolio inclusion, and run regime/technical analysis.

## Common Commands

```bash
# Activate virtual environment (Windows)
.\venv\Scripts\activate

# Development server
python manage.py runserver

# Database migrations
python manage.py makemigrations
python manage.py migrate

# Run tests
python manage.py test

# Collect static files (production)
python manage.py collectstatic

# Deploy to Heroku
./deploy.bat
```

## Architecture

### Apps
- **`config/`** — Django project settings, root URL config, WSGI/ASGI entry points
- **`accounts/`** — User registration only (`SignUpView`); authentication handled by Django's built-in auth
- **`blog/`** — All core functionality: research posts, comments, votes, portfolio analysis

### Core `blog/` modules
- **`models.py`** — `Post` (stock research with ticker, prices, thesis), `Comment`, `Vote` (unique per user/post, used for portfolio inclusion)
- **`views.py`** — 20+ views handling CRUD, voting, likes, PDF generation, portfolio/regime analysis. Business logic lives directly in views.
- **`finance.py`** — Portfolio metrics: VaR (historic, Cornish-Fisher, CVaR), stock stats via yfinance
- **`regime.py`** — Market regime detection using Hurst exponent and portfolio optimization (cvxpy)
- **`ta.py`** — `TABacktester` class for SMA crossover, RSI, and mean-reversion backtesting
- **`utils.py`** — `render_to_pdf()` helper using xhtml2pdf

### Data flow for portfolio views
`/portfolio` and `/summary` pull tickers from `Post` objects where `include=True`, fetch price data via yfinance, run regime analysis from `regime.py`, and render results. `/pdf/` endpoints generate downloadable reports using `utils.render_to_pdf()`.

### Database
SQLite3 in development; PostgreSQL in production via `DATABASE_URL` env var. 12 migrations under `blog/migrations/`.

### Deployment
Heroku via `Procfile` (`gunicorn config.wsgi`). Static files served by WhiteNoise. Environment variables (`SECRET_KEY`, `DATABASE_URL`) loaded from `.env` via python-dotenv.

## Key Dependencies
- **yfinance / pandas-datareader** — market data fetching
- **cvxpy** — portfolio optimization in `regime.py`
- **hurst** — Hurst exponent calculation for regime detection
- **xhtml2pdf / reportlab** — PDF report generation
- **django-crispy-forms + bootstrap4** — form rendering
