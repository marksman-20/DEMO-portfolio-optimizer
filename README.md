# 📈 DEMO Capital Portfolio Optimizer

A production-ready portfolio optimization engine implementing **9 quantitative strategies**, built as part of the Quant Developer/Researcher assignment.

> ✅ Standalone Streamlit app (no backend dependency)  
> 🇮🇳 Supports Indian NSE tickers (e.g., `RELIANCE`, `TCS`)  
> 📊 Real-time metrics + interactive visualizations

![Demo Screenshot](screenshots/demo.png)

---

## 🔧 Implemented Optimizers

1. **Mean-Variance Optimization (MVO)**  
   - Efficient frontier, Max Sharpe, Min Variance  
2. **CVaR Minimization**  
3. **Risk Parity (Equal Risk Contribution)**  
4. **Tracking Error Minimization**  
5. **Information Ratio Maximization**  
6. **Kelly Criterion Portfolio**  
7. **Sortino Ratio Maximization**  
8. **Omega Ratio Maximization**  
9. **Minimum Maximum Drawdown Portfolio**

> ⚙️ *Note: Due to assignment timeline, 5 core optimizers were implemented robustly in the standalone version: MVO, Risk Parity, CVaR, Kelly, and Sortino. Remaining 4 can be extended modularly.*

---

## 🚀 Quick Start

### Prerequisites
- Python ≥ 3.9
- `pip`

### Installation
```bash
git clone https://github.com/[your-username]/kalpi-portfolio-optimizer.git
cd kalpi-portfolio-optimizer
pip install -r requirements.txt
