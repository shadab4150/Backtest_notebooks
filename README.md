# backtest-reference-code
This is the repository that contains the reference code for model backtesting and validation.

### Key Features
+ The Futures data of SP500 is used for the backtest.
+ Leverage version of the backtest code is used. The leverage is later plotted over the period of the backtest.
+ Rolling statistics are included to measure the robustness of the returns.
+ Includes Stop Loss statistics: percentage hit and consecutive SL hits.
+ Includes new performance metrics: OMEGA and ADJUSTED SHARPE.
+ Monte Carlo simulations for determining the worst-case max drawdown.

### Repository structure
The file structure of this repository is as follows:

```
backtest-reference-code
│   README.md
│   .gitignore
│   requirements.txt
│   backtest.ipynb
└───utils
    │   backtest_utils.py
```
