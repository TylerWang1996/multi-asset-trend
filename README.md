# Multi-Asset Trend Following Strategy

This repository contains a Python implementation of a multi-asset trend-following strategy targeting 10% volatility. The strategy applies to various asset classes including equities, commodities, rates, and FX using time series momentum (TSMOM) as a simple signal for trend.

The Python script tsmom.py takes in multi-asset futures data in the from of data_pulled_updated.csv and produces csvs that show the trades for different portfolios. The multi-asset aggregate portfolio can be found in cta_results.csv and single asset class portfolios can be found in the respective result CSVs.

This also shows backtested performance of portfolios in said CSVs.

NOTE: This is meant to be a simple illustrative example of a multi-asset trend strategy as an educational exercise and not a live-tested investment strategy.