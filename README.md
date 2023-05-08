# backtest-reference-code
This is the repository that contains the code for model backtesting and validation.

### Key Features
+ The Futures data of SP500 is used for the backtest.
+ Rolling statistics are included to measure the robustness of the returns.
+ Includes Stop Loss statistics: percentage hit and consecutive SL hits.

### Repository structure
The file structure of this repository is as follows:

```
backtest-reference-code
│   README.md
│   requirements.txt
│   sample_backtest.ipynb
|   sample_strategy_ibs.ipynb
└───utils
    │   backtest_utils.py
```
