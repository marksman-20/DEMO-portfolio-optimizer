import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import cvxpy as cp
from scipy.optimize import minimize
from datetime import datetime, timedelta, date
import time
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Kalpi Capital Portfolio Optimizer", layout="wide")
st.title("Kalpi Capital Portfolio Optimizer")
st.markdown("### Build optimal portfolios using advanced quantitative methods")

# Utility functions with robust error handling
def load_data(tickers, start_date, end_date):
    """Load adjusted close prices with robust error handling"""
    # Add .NS suffix for Indian stocks if not already present
    processed_tickers = []
    for t in tickers:
        t = t.strip()
        if not t:
            continue
        if '.' not in t and not t.endswith('.NS'):
            processed_tickers.append(f"{t}.NS")
        elif t.endswith('.NS') or '.' in t:
            processed_tickers.append(t)
    
    if not processed_tickers:
        raise ValueError("No valid tickers provided")
    
    try:
        # Try to download data
        with st.spinner(f"Downloading data for {', '.join(processed_tickers)}..."):
            data = yf.download(processed_tickers, start=start_date, end=end_date, progress=False)
            
            # Handle different data structures returned by yfinance
            if data.empty:
                raise ValueError("No data returned for the specified tickers and date range")
            
            # Extract adjusted close prices
            if isinstance(data.columns, pd.MultiIndex):
                # Multi-index case (multiple tickers)
                if 'Adj Close' in data.columns.levels[0]:
                    prices = data['Adj Close']
                elif 'Close' in data.columns.levels[0]:
                    prices = data['Close']
                else:
                    raise ValueError("Could not find price data in the downloaded data")
            else:
                # Single ticker case
                if 'Adj Close' in data.columns:
                    prices = pd.DataFrame(data['Adj Close'])
                elif 'Close' in data.columns:
                    prices = pd.DataFrame(data['Close'])
                else:
                    raise ValueError("Could not find price data in the downloaded data")
                
                # Rename column to match ticker
                prices.columns = [processed_tickers[0]]
            
            # Clean and validate data
            prices = prices.dropna(axis=1, how='all')  # Drop columns with all NaN
            prices = prices.dropna()  # Drop rows with any NaN
            
            if prices.empty:
                raise ValueError("No valid price data after cleaning")
            
            # Rename columns to remove .NS suffix for display
            display_names = [col.replace('.NS', '') for col in prices.columns]
            prices.columns = display_names
            
            return prices
    
    except Exception as e:
        st.error(f"❌ Error downloading data: {str(e)}")
        # Provide sample data for testing if download fails
        st.warning("Using sample data for demonstration...")
        return generate_sample_data(processed_tickers, start_date, end_date)

def generate_sample_data(tickers, start_date, end_date):
    """Generate sample data if real data download fails"""
    # Use fewer tickers if too many are requested
    display_tickers = tickers[:5] if len(tickers) > 5 else tickers
    display_tickers = [t.replace('.NS', '') for t in display_tickers]
    
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(dates)
    n_assets = len(display_tickers)
    
    # Generate correlated random returns
    np.random.seed(42)
    base_returns = np.random.normal(0.0005, 0.01, n_days)
    
    # Create price series with some correlation
    price_data = {}
    for i, ticker in enumerate(display_tickers):
        # Add some noise and correlation
        asset_returns = base_returns + np.random.normal(0, 0.005, n_days) * (i + 1) * 0.5
        prices = 100 * np.exp(np.cumsum(asset_returns))
        price_data[ticker] = prices
    
    prices_df = pd.DataFrame(price_data, index=dates)
    return prices_df

def calculate_returns(prices):
    """Calculate daily simple returns"""
    returns = prices.pct_change().dropna()
    return returns

def calculate_metrics(weights, returns, risk_free_rate=0.0, mar=0.0):
    """Calculate comprehensive portfolio metrics"""
    portfolio_returns = returns.dot(weights)
    cum_returns = (1 + portfolio_returns).cumprod()
    
    metrics = {
        "expected_return": portfolio_returns.mean() * 252,
        "volatility": portfolio_returns.std() * np.sqrt(252),
        "sharpe_ratio": 0,
        "max_drawdown": 0,
        "cumulative_return": cum_returns.iloc[-1] - 1
    }
    
    # Sharpe ratio
    if portfolio_returns.std() > 0:
        metrics["sharpe_ratio"] = (portfolio_returns.mean() - risk_free_rate/252) / portfolio_returns.std() * np.sqrt(252)
    
    # Max drawdown
    metrics["max_drawdown"] = calculate_max_drawdown(cum_returns)
    
    # Sortino ratio
    downside_returns = portfolio_returns[portfolio_returns < mar]
    if len(downside_returns) > 0 and downside_returns.std() > 0:
        downside_deviation = downside_returns.std() * np.sqrt(252)
        metrics["sortino_ratio"] = (metrics["expected_return"] - risk_free_rate) / downside_deviation
    else:
        metrics["sortino_ratio"] = np.inf
    
    # CVaR (95% confidence)
    var = np.percentile(portfolio_returns, 5)
    cvar_data = portfolio_returns[portfolio_returns <= var]
    metrics["cvar"] = cvar_data.mean() * np.sqrt(252) if len(cvar_data) > 0 else 0
    
    # Omega ratio
    gains = portfolio_returns[portfolio_returns > mar] - mar
    losses = mar - portfolio_returns[portfolio_returns < mar]
    metrics["omega_ratio"] = gains.sum() / losses.sum() if losses.sum() != 0 else np.inf
    
    return metrics, cum_returns

def calculate_max_drawdown(cum_returns):
    """Calculate maximum drawdown"""
    if len(cum_returns) == 0:
        return 0
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    return drawdown.min()

# Optimizer implementations
class MeanVarianceOptimizer:
    def __init__(self, returns, risk_free_rate=0.0, long_only=True, weight_bounds=(0, 1)):
        self.returns = returns
        self.n_assets = returns.shape[1]
        self.asset_names = returns.columns.tolist()
        self.risk_free_rate = risk_free_rate
        self.long_only = long_only
        self.weight_bounds = weight_bounds
    
    def optimize(self, method='max_sharpe'):
        mean_returns = self.returns.mean() * 252
        cov_matrix = self.returns.cov() * 252
        
        if method == 'max_sharpe':
            return self._max_sharpe(mean_returns, cov_matrix)
        elif method == 'min_variance':
            return self._min_variance(cov_matrix)
        elif method == 'frontier':
            return self._efficient_frontier(mean_returns, cov_matrix)
    
    def _max_sharpe(self, mean_returns, cov_matrix):
        w = cp.Variable(self.n_assets)
        mu = mean_returns.values
        Sigma = cov_matrix.values
        
        # For Sharpe ratio maximization, we use the transformation method
        kappa = cp.Variable(pos=True)
        y = cp.Variable(self.n_assets)
        
        objective = cp.Maximize(mu @ y - self.risk_free_rate * kappa)
        constraints = [
            cp.quad_form(y, Sigma) <= 1,
            cp.sum(y) == kappa,
            kappa >= 0
        ]
        if self.long_only:
            constraints.append(y >= 0)
        if self.weight_bounds:
            min_w, max_w = self.weight_bounds
            constraints += [y >= min_w * kappa, y <= max_w * kappa]
        
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.SCS, verbose=False, max_iters=5000)
        except Exception as e:
            raise ValueError(f"Optimization failed: {str(e)}")
        
        if y.value is None or kappa.value is None:
            raise ValueError("Optimization failed - no solution found")
        
        weights = y.value / kappa.value
        portfolio_return = weights @ mu
        volatility = np.sqrt(weights @ Sigma @ weights)
        sharpe = (portfolio_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        return {
            'weights': pd.Series(weights, index=self.asset_names),
            'metrics': {
                'expected_return': portfolio_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe
            }
        }
    
    def _min_variance(self, cov_matrix):
        w = cp.Variable(self.n_assets)
        objective = cp.quad_form(w, cov_matrix.values)
        constraints = [cp.sum(w) == 1]
        if self.long_only:
            constraints.append(w >= 0)
        if self.weight_bounds:
            min_w, max_w = self.weight_bounds
            constraints += [w >= min_w, w <= max_w]
        
        prob = cp.Problem(cp.Minimize(objective), constraints)
        try:
            prob.solve(solver=cp.SCS, verbose=False, max_iters=5000)
        except Exception as e:
            raise ValueError(f"Optimization failed: {str(e)}")
        
        if w.value is None:
            raise ValueError("Optimization failed - no solution found")
        
        weights = w.value
        variance = weights @ cov_matrix.values @ weights
        portfolio_return = weights @ (self.returns.mean() * 252).values
        sharpe = (portfolio_return - self.risk_free_rate) / np.sqrt(variance) if np.sqrt(variance) > 0 else 0
        
        return {
            'weights': pd.Series(weights, index=self.asset_names),
            'metrics': {
                'expected_return': portfolio_return,
                'volatility': np.sqrt(variance),
                'sharpe_ratio': sharpe
            }
        }
    
    def _efficient_frontier(self, mean_returns, cov_matrix, n_points=20):
        min_ret = max(mean_returns.min(), 0)  # Avoid negative returns
        max_ret = min(mean_returns.max(), mean_returns.mean() * 3)  # Cap at reasonable level
        target_returns = np.linspace(min_ret, max_ret, n_points)
        results = []
        
        for target in target_returns:
            w = cp.Variable(self.n_assets)
            objective = cp.quad_form(w, cov_matrix.values)
            constraints = [
                w @ mean_returns.values >= target,
                cp.sum(w) == 1
            ]
            if self.long_only:
                constraints.append(w >= 0)
            if self.weight_bounds:
                min_w, max_w = self.weight_bounds
                constraints += [w >= min_w, w <= max_w]
            
            prob = cp.Problem(cp.Minimize(objective), constraints)
            try:
                prob.solve(solver=cp.SCS, verbose=False, max_iters=5000)
            except Exception as e:
                continue  # Skip this point if optimization fails
            
            if w.value is not None:
                weights = w.value
                volatility = np.sqrt(weights @ cov_matrix.values @ weights)
                results.append({
                    'return': target,
                    'volatility': volatility,
                    'weights': pd.Series(weights, index=self.asset_names)
                })
        
        return results

class RiskParityOptimizer:
    def __init__(self, returns):
        self.returns = returns
        self.asset_names = returns.columns.tolist()
    
    def optimize(self):
        """Implement Risk Parity with proper equal risk contribution"""
        cov_matrix = self.returns.cov().values
        n = cov_matrix.shape[0]
        
        # Target risk contributions (equal for all assets)
        target_rc = np.ones(n) / n
        
        # Objective function to minimize difference between actual and target risk contributions
        def risk_parity_objective(weights):
            # Ensure weights sum to 1
            weights = np.array(weights) / np.sum(weights)
            
            # Portfolio volatility
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            
            # Marginal risk contributions
            mrc = cov_matrix @ weights / portfolio_vol
            
            # Total risk contributions
            rc = weights * mrc
            
            # Normalize risk contributions to sum to 1
            rc_normalized = rc / np.sum(rc)
            
            # Sum of squared differences from target risk contributions
            return np.sum((rc_normalized - target_rc) ** 2)
        
        # Constraints: weights sum to 1 and are non-negative
        constraints = ({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1
        })
        
        # Initial guess: equal weights
        x0 = np.ones(n) / n
        
        # Bounds: weights between 0 and 1
        bounds = tuple((0, 1) for _ in range(n))
        
        # Optimization
        result = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 2000, 'ftol': 1e-12}
        )
        
        if not result.success:
            # Try a simpler approach if optimization fails
            st.warning(f"Risk parity optimization warning: {result.message}")
            weights = np.ones(n) / n
        else:
            weights = result.x
            # Ensure no negative weights and normalize
            weights = np.maximum(weights, 0)
            weights = weights / np.sum(weights)
        
        # Calculate portfolio volatility
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        
        # Calculate marginal risk contributions
        mrc = cov_matrix @ weights / portfolio_vol
        
        # Calculate total risk contributions
        risk_contrib = weights * mrc
        
        # Normalize risk contributions to percentages
        risk_contrib_pct = risk_contrib / np.sum(risk_contrib)
        
        return {
            'weights': pd.Series(weights, index=self.asset_names),
            'risk_contributions': pd.Series(risk_contrib_pct, index=self.asset_names),
            'metrics': {
                'volatility': portfolio_vol * np.sqrt(252)
            }
        }

class CVarOptimizer:
    def __init__(self, returns, confidence_level=0.95, long_only=True, weight_bounds=(0, 1)):
        self.returns = returns
        self.n_assets = returns.shape[1]
        self.asset_names = returns.columns.tolist()
        self.confidence_level = confidence_level
        self.long_only = long_only
        self.weight_bounds = weight_bounds
    
    def optimize(self):
        n_samples = self.returns.shape[0]
        returns = self.returns.values
        
        w = cp.Variable(self.n_assets)
        gamma = cp.Variable()
        z = cp.Variable(n_samples)
        
        constraints = [
            z >= -returns @ w - gamma,
            z >= 0,
            cp.sum(w) == 1
        ]
        if self.long_only:
            constraints.append(w >= 0)
        if self.weight_bounds:
            min_w, max_w = self.weight_bounds
            constraints += [w >= min_w, w <= max_w]
        
        objective = cp.Minimize(gamma + (1 / (1 - self.confidence_level)) * cp.sum(z) / n_samples)
        prob = cp.Problem(objective, constraints)
        
        # Try different solvers in order of preference
        solvers = [cp.SCS, cp.ECOS, cp.OSQP]
        solver_names = ["SCS", "ECOS", "OSQP"]
        success = False
        
        for solver, name in zip(solvers, solver_names):
            try:
                prob.solve(solver=solver, verbose=False, max_iters=5000)
                if w.value is not None:
                    success = True
                    break
            except Exception as e:
                continue
        
        if not success:
            # Last resort: try with CVXPY's default solver
            try:
                prob.solve(verbose=False, max_iters=5000)
                if w.value is None:
                    raise ValueError("No solver could find a solution")
            except Exception as e:
                raise ValueError(f"CVaR optimization failed with all available solvers: {str(e)}")
        
        weights = w.value
        portfolio_returns = self.returns.dot(weights)
        var = np.percentile(portfolio_returns, 100 * (1 - self.confidence_level))
        cvar_data = portfolio_returns[portfolio_returns <= var]
        cvar = cvar_data.mean() if len(cvar_data) > 0 else 0
        
        # Calculate additional metrics
        expected_return = portfolio_returns.mean() * 252
        volatility = portfolio_returns.std() * np.sqrt(252)
        
        return {
            'weights': pd.Series(weights, index=self.asset_names),
            'metrics': {
                'expected_return': expected_return,
                'volatility': volatility,
                'cvar': cvar * np.sqrt(252)
            }
        }

class KellyOptimizer:
    def __init__(self, returns, long_only=True, weight_bounds=(0, 1)):
        self.returns = returns
        self.n_assets = returns.shape[1]
        self.asset_names = returns.columns.tolist()
        self.long_only = long_only
        self.weight_bounds = weight_bounds
    
    def optimize(self):
        """Maximize expected log returns (Kelly Criterion)"""
        def objective(weights):
            # Portfolio returns
            port_returns = self.returns.dot(weights)
            # Filter out extreme negative returns to avoid log issues
            valid_returns = port_returns[port_returns > -0.99]  # Avoid log of zero or negative
            if len(valid_returns) == 0 or np.any(np.isnan(valid_returns)):
                return 1e10  # Large penalty
            # Expected log returns
            return -np.mean(np.log(1 + valid_returns))
        
        # Constraints: weights sum to 1
        constraints = [{
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1
        }]
        
        # Bounds for weights
        if self.long_only:
            bounds = tuple((0, 1) for _ in range(self.n_assets))
        else:
            min_w, max_w = self.weight_bounds
            bounds = tuple((min_w, max_w) for _ in range(self.n_assets))
        
        # Initial guess: equal weights
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Optimization
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            raise ValueError(f"Kelly optimization failed: {result.message}")
        
        weights = result.x
        
        # Calculate portfolio metrics
        portfolio_returns = self.returns.dot(weights)
        expected_return = portfolio_returns.mean() * 252
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe = expected_return / volatility if volatility > 0 else 0
        
        return {
            'weights': pd.Series(weights, index=self.asset_names),
            'metrics': {
                'expected_return': expected_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe
            }
        }

# Sample tickers that are known to work on Yahoo Finance
DEFAULT_TICKERS = "RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS"

# UI Components
with st.sidebar:
    st.header("Portfolio Configuration")
    
    # Use sample data toggle
    use_sample_data = st.checkbox("Use sample data (if real data fails)", False)
    
    method = st.selectbox("Optimization Method", [
        "Mean-Variance (Max Sharpe)", 
        "Mean-Variance (Min Variance)",
        "Risk Parity",
        "CVaR Minimization",
        "Kelly Criterion"
    ])
    
    # Pre-defined ticker sets
    ticker_set = st.selectbox("Ticker Set", [
        "Indian Large Caps (Default)",
        "Custom"
    ])
    
    if ticker_set == "Indian Large Caps (Default)":
        tickers = st.text_area("Tickers (comma separated)", DEFAULT_TICKERS, height=120)
    else:
        tickers = st.text_area("Tickers (comma separated)", "AAPL, MSFT, GOOGL, AMZN, META", height=120)
    
    # Date inputs with reasonable defaults
    today = date.today()
    three_years_ago = today - timedelta(days=3*365)
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", three_years_ago)
    with col2:
        end_date = st.date_input("End Date", today)
    
    with st.expander("Advanced Parameters"):
        risk_free_rate = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 5.0, 0.5) / 100
        mar = st.number_input("Minimum Acceptable Return (%)", -10.0, 10.0, 0.0, 0.5) / 100
        confidence_level = st.slider("CVaR Confidence Level", 0.90, 0.99, 0.95, 0.01)
        long_only = st.checkbox("Long Only Portfolio", True)
        min_weight = st.number_input("Min Weight (%)", 0.0, 100.0, 0.0, 1.0) / 100
        max_weight = st.number_input("Max Weight (%)", 0.0, 100.0, 100.0, 1.0) / 100
        if not long_only:
            st.warning("Long-short portfolios may be unstable with limited data")
    
    optimize_button = st.button("Optimize Portfolio", type="primary", use_container_width=True)

# Main content
if optimize_button:
    with st.spinner("Loading data and optimizing portfolio..."):
        try:
            # Process tickers
            ticker_list = [t.strip() for t in tickers.split(",") if t.strip()]
            if len(ticker_list) < 2:
                st.error("Please enter at least 2 tickers")
                st.stop()
            
            # Limit to 8 tickers for stability
            if len(ticker_list) > 8:
                st.warning(f"Using only first 8 tickers for stability (you provided {len(ticker_list)})")
                ticker_list = ticker_list[:8]
            
            # Load data
            prices = load_data(ticker_list, start_date, end_date)
            
            if prices.empty:
                st.error("No data available for optimization")
                st.stop()
            
            # Check data quality
            n_days = len(prices)
            if n_days < 30:
                st.warning(f"Only {n_days} trading days available. Results may not be reliable.")
            elif n_days < 100:
                st.info(f"Using {n_days} trading days of data")
            else:
                st.success(f"Loaded {n_days} trading days of data for {len(prices.columns)} assets")
            
            returns = calculate_returns(prices)
            
            # Display data preview
            with st.expander("📊 Data Preview"):
                st.write("**Price Data (Last 5 Days):**")
                st.dataframe(prices.tail().style.format("${:.2f}"))
                
                st.write("**Daily Returns Statistics:**")
                stats_df = pd.DataFrame({
                    'Mean Daily Return': returns.mean(),
                    'Daily Volatility': returns.std(),
                    'Min Return': returns.min(),
                    'Max Return': returns.max()
                })
                st.dataframe(stats_df.style.format({
                    'Mean Daily Return': '{:.4f}',
                    'Daily Volatility': '{:.4f}',
                    'Min Return': '{:.4f}',
                    'Max Return': '{:.4f}'
                }))
            
            # Run optimization based on selected method
            weight_bounds = (min_weight, max_weight)
            
            if "Mean-Variance (Max Sharpe)" in method:
                optimizer = MeanVarianceOptimizer(
                    returns, 
                    risk_free_rate=risk_free_rate,
                    long_only=long_only,
                    weight_bounds=weight_bounds
                )
                result = optimizer.optimize(method='max_sharpe')
            
            elif "Mean-Variance (Min Variance)" in method:
                optimizer = MeanVarianceOptimizer(
                    returns, 
                    risk_free_rate=risk_free_rate,
                    long_only=long_only,
                    weight_bounds=weight_bounds
                )
                result = optimizer.optimize(method='min_variance')
            
            elif "Risk Parity" in method:
                # Risk parity requires long-only portfolios
                if not long_only:
                    st.warning("Risk Parity requires long-only portfolios. Switching to long-only.")
                    long_only = True
                
                optimizer = RiskParityOptimizer(returns)
                result = optimizer.optimize()
            
            elif "CVaR Minimization" in method:
                optimizer = CVarOptimizer(
                    returns,
                    confidence_level=confidence_level,
                    long_only=long_only,
                    weight_bounds=weight_bounds
                )
                result = optimizer.optimize()
            
            elif "Kelly Criterion" in method:
                optimizer = KellyOptimizer(
                    returns,
                    long_only=long_only,
                    weight_bounds=weight_bounds
                )
                result = optimizer.optimize()
            
            # Calculate additional metrics
            weights = result['weights']
            metrics, cum_returns = calculate_metrics(weights, returns, risk_free_rate, mar)
            
            # Update metrics with optimizer-specific results
            for key, value in result.get('metrics', {}).items():
                metrics[key] = value
            
            # Display results
            st.success("✅ Optimization completed successfully!")
            
            tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Chart", "⚙️ Details"])
            
            with tab1:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Optimal Weights")
                    weights_df = pd.DataFrame({
                        "Asset": weights.index,
                        "Weight (%)": weights.values * 100
                    }).sort_values("Weight (%)", ascending=False)
                    
                    # Format and display
                    formatted_df = weights_df.style.format({"Weight (%)": "{:.2f}%"}).background_gradient(
                        cmap='Blues', subset=["Weight (%)"]
                    )
                    st.dataframe(formatted_df, use_container_width=True)
                
                with col2:
                    st.subheader("Portfolio Metrics")
                    metrics_display = {
                        "Expected Annual Return": f"{metrics['expected_return']*100:.2f}%",
                        "Annual Volatility": f"{metrics['volatility']*100:.2f}%",
                        "Sharpe Ratio": f"{metrics.get('sharpe_ratio', 0):.2f}",
                        "Sortino Ratio": f"{metrics.get('sortino_ratio', 0):.2f}",
                        "Max Drawdown": f"{metrics['max_drawdown']*100:.2f}%",
                        "CVaR (95%)": f"{metrics.get('cvar', 0)*100:.2f}%"
                    }
                    
                    metrics_df = pd.DataFrame({
                        "Metric": list(metrics_display.keys()),
                        "Value": list(metrics_display.values())
                    })
                    st.dataframe(metrics_df, use_container_width=True)
            
            with tab2:
                st.subheader("Cumulative Returns")
                fig, ax = plt.subplots(figsize=(12, 6))
                cum_returns.plot(ax=ax, linewidth=2, color='blue')
                ax.set_title("Portfolio Cumulative Returns", fontsize=14)
                ax.set_xlabel("Date", fontsize=12)
                ax.set_ylabel("Cumulative Return", fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                # Format y-axis as percentage
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.1f}%'))
                
                # Highlight key points
                if len(cum_returns) > 0:
                    max_point = cum_returns.idxmax()
                    min_point = cum_returns.idxmin()
                    ax.scatter([max_point, min_point], [cum_returns[max_point], cum_returns[min_point]], 
                              color=['green', 'red'], s=100, zorder=5)
                    ax.annotate(f'Max: {cum_returns[max_point]*100:.1f}%', 
                               xy=(max_point, cum_returns[max_point]),
                               xytext=(10, 10), textcoords='offset points')
                    ax.annotate(f'Min: {cum_returns[min_point]*100:.1f}%', 
                               xy=(min_point, cum_returns[min_point]),
                               xytext=(10, -15), textcoords='offset points')
                
                st.pyplot(fig)
                
                # Risk parity risk contribution chart
                if "Risk Parity" in method and 'risk_contributions' in result:
                    st.subheader("Risk Contributions")
                    risk_df = pd.DataFrame({
                        "Asset": result['risk_contributions'].index,
                        "Risk Contribution (%)": result['risk_contributions'].values * 100
                    }).sort_values("Risk Contribution (%)", ascending=False)
                    
                    # Display risk contributions table
                    st.dataframe(risk_df.style.format({"Risk Contribution (%)": "{:.2f}%"}))
                    
                    # Create bar chart for risk contributions
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(risk_df["Asset"], risk_df["Risk Contribution (%)"], color='skyblue')
                    ax.set_title("Risk Contribution by Asset", fontsize=14)
                    ax.set_ylabel("Risk Contribution (%)", fontsize=12)
                    ax.set_ylim(0, max(10, risk_df["Risk Contribution (%)"].max() * 1.2))
                    plt.xticks(rotation=45)
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                f'{height:.1f}%', ha='center', va='bottom')
                    
                    st.pyplot(fig)
            
            with tab3:
                st.subheader("Optimization Details")
                details = {
                    "Method": method,
                    "Number of Assets": len(ticker_list),
                    "Data Period": f"{start_date} to {end_date} ({len(prices)} trading days)",
                    "Risk-Free Rate": f"{risk_free_rate*100:.2f}%",
                    "Portfolio Type": "Long Only" if long_only else "Long-Short",
                    "Weight Constraints": f"[{min_weight*100:.1f}%, {max_weight*100:.1f}%]"
                }
                
                for key, value in details.items():
                    st.write(f"**{key}:** {value}")
                
                # Show optimization method details
                if "Mean-Variance" in method:
                    st.write("**Optimization Method:** Quadratic programming with CVXPY")
                elif "Risk Parity" in method:
                    st.write("**Optimization Method:** Equal risk contribution using numerical optimization")
                    st.write("**Risk Parity Details:** Optimizes portfolio to ensure each asset contributes equally to total portfolio risk")
                elif "CVaR" in method:
                    st.write(f"**Optimization Method:** CVaR minimization at {confidence_level*100:.1f}% confidence level")
                    st.write("**Solver Used:** SCS (fallback to other available solvers)")
                elif "Kelly" in method:
                    st.write("**Optimization Method:** Kelly criterion maximizing expected log returns")
        
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.exception(e)
            st.info("""
            **Troubleshooting Tips:**
            - Try using fewer tickers (max 5-6)
            - Use more recent dates (last 2-3 years)
            - Try the "Use sample data" checkbox in the sidebar
            - Stick with the default Indian large-cap tickers
            - For CVaR issues: Try installing ECOS solver with `pip install ecos`
            """)

# Sample data demonstration
if not optimize_button:
    st.info("""
    💡 **Quick Start Guide**
    
    1. Keep the default Indian large-cap tickers or enter your own
    2. Adjust the date range if needed (3 years is recommended)
    3. Select an optimization method from the dropdown
    4. Click "Optimize Portfolio" to see results
    
    **For best results:**
    - Use 3-8 tickers maximum
    - Select at least 1 year of historical data
    - Start with "Mean-Variance (Max Sharpe)" method
    - Use the default settings for your first run
    
    If you encounter data download issues, check the "Use sample data" box in the sidebar.
    """)

# Footer
st.markdown("---")
st.markdown("Kalpi Capital Portfolio Optimizer | Quant Developer Assignment")
st.caption("Built with Streamlit & CVXPY | Real-time data from Yahoo Finance")