import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

def load_data(tickers, start_date, end_date):
    """Load adjusted close prices for Indian stocks from Yahoo Finance"""
    # Add .NS suffix for NSE stocks if not already present
    tickers = [t if '.' in t else f"{t}.NS" for t in tickers]
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data.dropna()

def calculate_returns(prices):
    """Calculate daily simple returns"""
    return prices.pct_change().dropna()

def calculate_metrics(weights, returns, risk_free_rate=0.0, mar=0.0, benchmark_returns=None):
    """Calculate comprehensive portfolio metrics"""
    portfolio_returns = returns.dot(weights)
    cum_returns = (1 + portfolio_returns).cumprod()
    
    metrics = {
        "expected_return": portfolio_returns.mean() * 252,
        "volatility": portfolio_returns.std() * np.sqrt(252),
        "sharpe_ratio": (portfolio_returns.mean() - risk_free_rate/252) / portfolio_returns.std() * np.sqrt(252),
        "max_drawdown": calculate_max_drawdown(cum_returns),
        "cumulative_return": cum_returns.iloc[-1] - 1
    }
    
    # Sortino ratio
    downside_returns = portfolio_returns[portfolio_returns < mar]
    downside_deviation = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(252)
    metrics["sortino_ratio"] = (metrics["expected_return"] - risk_free_rate) / downside_deviation if downside_deviation != 0 else 0
    
    # CVaR (95% confidence)
    var = np.percentile(portfolio_returns, 5)
    cvar = portfolio_returns[portfolio_returns <= var].mean()
    metrics["cvar"] = cvar * np.sqrt(252)
    
    # Omega ratio
    gains = portfolio_returns[portfolio_returns > mar] - mar
    losses = mar - portfolio_returns[portfolio_returns < mar]
    metrics["omega_ratio"] = gains.sum() / losses.sum() if losses.sum() != 0 else np.inf
    
    # Benchmark-relative metrics
    if benchmark_returns is not None:
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(252)
        info_ratio = active_returns.mean() * 252 / tracking_error if tracking_error != 0 else 0
        
        metrics.update({
            "tracking_error": tracking_error,
            "information_ratio": info_ratio,
            "active_return": active_returns.mean() * 252
        })
    
    return metrics, cum_returns

def calculate_max_drawdown(cum_returns):
    """Calculate maximum drawdown"""
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    return drawdown.min()