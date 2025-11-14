from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from utils import load_data, calculate_returns, calculate_metrics
from optimizers import *

app = FastAPI()

class OptimizationRequest(BaseModel):
    tickers: List[str]
    start_date: str
    end_date: str
    method: str
    risk_free_rate: float = 0.0
    mar: float = 0.0
    confidence_level: float = 0.95
    benchmark_ticker: Optional[str] = None
    long_only: bool = True
    weight_bounds: Optional[List[float]] = None
    n_points: int = 20

@app.post("/load-data")
async def load_data_endpoint(request: OptimizationRequest):
    prices = load_data(request.tickers, request.start_date, request.end_date)
    returns = calculate_returns(prices)
    return {
        "tickers": request.tickers,
        "data": returns.to_dict(),
        "start_date": request.start_date,
        "end_date": request.end_date
    }

@app.post("/optimize")
async def optimize_portfolio(request: OptimizationRequest):
    # Load data
    prices = load_data(request.tickers, request.start_date, request.end_date)
    returns = calculate_returns(prices)
    
    # Handle benchmark if needed
    benchmark_returns = None
    if request.benchmark_ticker:
        bench_prices = load_data([request.benchmark_ticker], request.start_date, request.end_date)
        benchmark_returns = calculate_returns(bench_prices).squeeze()
    
    # Initialize optimizer
    kwargs = {
        'risk_free_rate': request.risk_free_rate,
        'mar': request.mar,
        'confidence_level': request.confidence_level,
        'long_only': request.long_only,
        'weight_bounds': tuple(request.weight_bounds) if request.weight_bounds else (0, 1),
        'n_points': request.n_points,
        'benchmark_returns': benchmark_returns
    }
    
    optimizer_map = {
        'mvo': MeanVarianceOptimizer,
        'cvar': CVarOptimizer,
        'risk_parity': RiskParityOptimizer,
        'tracking_error': TrackingErrorOptimizer,
        'info_ratio': InformationRatioOptimizer,
        'kelly': KellyOptimizer,
        'sortino': SortinoOptimizer,
        'omega': OmegaOptimizer,
        'min_drawdown': MinDrawdownOptimizer
    }
    
    optimizer = optimizer_map[request.method](returns, **kwargs)
    result = optimizer.optimize()
    
    # Calculate metrics
    weights = pd.Series(result['weights'])
    metrics, cum_returns = calculate_metrics(
        weights, 
        returns, 
        request.risk_free_rate, 
        request.mar,
        benchmark_returns
    )
    
    return {
        "weights": result['weights'],
        "metrics": metrics,
        "cumulative_returns": cum_returns.to_dict()
    }

@app.get("/frontier")
async def get_frontier(request: OptimizationRequest):
    # Similar to optimize but for MVO frontier
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Backend server is running"}