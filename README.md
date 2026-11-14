# Public Portfolio Research Dashboard

Adapted from the attached **Advanced Multi-Stock Research, Risk & Prediction Dashboard** notebook.

## Run locally

1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Run `pip install -r requirements.txt`.
4. Run `streamlit run app.py`.

## Publish a link with Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py` and `requirements.txt` to its root.
2. Visit https://share.streamlit.io/ and sign in.
3. Select **Create app**, choose the repository, branch and `app.py` as the main file.
4. Deploy. Share the resulting public URL.

A public repository means the application code is public. Do not put passwords, API keys, or client data in source files.

## What the app includes

- Editable public holdings intake form (ticker, shares, average cost per share)
- Market-value weighted allocation, unrealized P/L and downloadable CSV
- Current-weight historical risk proxy, correlation and Monte Carlo simulation
- Notebook-derived RSI, MACD, moving averages, and transparent risk monitor
- Optional per-stock gradient-boosting 5-day and 20-day return estimates

## Limitations / production checklist

- Uses Yahoo Finance through `yfinance`; data may be delayed, incomplete, unavailable or subject to provider restrictions. Review data licensing and hosting terms before commercial use.
- Portfolio returns and simulations use **today's weights**, not actual historical trades or cash flows. Cost basis is user-entered and excludes dividends, fees, taxes, and corporate-action adjustments.
- Forecasts are exploratory; not investment advice. Historical training does not imply future accuracy.
- No database, account, authentication, or intentional persistent portfolio storage is included. Streamlit sessions and hosting infrastructure may still process submitted values; do not advertise a guarantee of zero retention.
- For a widely shared public site, add abuse/rate limiting, data validation, privacy terms, error monitoring, security review, and scalable hosting.
- The app is a web adaptation, not a cell-for-cell execution of all 49 notebook cells. In particular it does not yet expose the notebook's directional classifier, permutation feature importance, all comparison dashboards or monitoring flags.

## Test checklist

- Try multiple tickers and fractional shares; confirm cost basis and market value calculations.
- Try duplicate tickers; verify share-weighted average purchase cost.
- Try an invalid ticker; verify the warning and that valid holdings still work.
- Verify charts with a one-stock portfolio and with tickers having limited overlapping histories.
- Verify CSV downloads, mobile display, and forecasts after deployment.
