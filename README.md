# PyBacktestPro

A lightweight, script-first Python backtesting toolkit with:
- a vectorized indicator engine and Strategy base class (engine.py),
- a TradingView Pine Script → Python converter (pine_parser.py),
- a notebook strategy loader (notebook_runner.py),
- plotting & chart helpers (charts.py),
- a small web/API surface (app.py) and example strategy (my_strategy.py).

Designed for traders and researchers who want to turn TradingView strategies or Jupyter notebooks into runnable Python backtests and inspect results interactively.

## Features
- Core backtest engine and Strategy base (engine.py) with enums and config classes for backtests, walk-forward, Monte Carlo and optimization options.
- Indicator engine (IND) with standard indicators used by strategies.
- Pine Script parser / converter to auto-generate Python Strategy classes for backtesting.
- Notebook loader to extract and run Strategy classes from .ipynb files.
- Charting utilities and an embedded HTML chart (assets/tv_chart.html).
- Example strategy (my_strategy.py) that demonstrates a simple SMA + ATR strategy.

---

## Quickstart

1. Clone and install dependencies:
```bash
git clone https://github.com/raviwijerathna1/PyBacktestPro.git
cd PyBacktestPro
python -m pip install -r requirements.txt
```

2. Run the web UI / API (app.py):
```bash
python app.py
```
If the repository is used as a lightweight web interface, app.py exposes endpoints to run backtests, view charts, and fetch chart data (see `app.py` for the available routes).

3. Convert TradingView Pine Script → Python Strategy:
```python
from pine_parser import convert_pine_to_python

pine_code = """// your pine v5 code here"""
py_code, errors, warnings = convert_pine_to_python(pine_code)
print(py_code)
print("errors:", errors)
print("warnings:", warnings)
```
This returns a Python Strategy class (string) you can save as a .py strategy file and run with the engine.

4. Extract a Strategy from a Jupyter Notebook:
```python
from notebook_runner import load_ipynb, save_ipynb_as_strategy

# load from a notebook file and get the assembled Python code
py_code, errors, warnings = load_ipynb(file_path="my_notebook.ipynb")
# or save the notebook's strategy directly to my_strategy.py
save_ipynb_as_strategy(file_path="my_notebook.ipynb", output_path="my_strategy_from_nb.py")
```

5. Example: inspect or modify the shipped example strategy
```python
# my_strategy.py (example)
from engine import Strategy, IndicatorEngine as IND
# ... see the example file below
```

```python
# Run a script that imports your strategy and drives the engine.
# The exact run interface depends on how you wire Engine/BacktestConfig,
# but basic building blocks are `Strategy` and `IndicatorEngine` (IND).
```

---

## Project layout

Top-level files and folders:
```
app.py                 # Web/API surface (Flask-like app endpoints)
charts.py              # Charting helpers and plotting utilities
engine.py              # Core engine, Strategy base class, configs, indicators
my_strategy.py         # Example strategy (SMA crossover + ATR SL/TP)
notebook_runner.py     # Extract and execute strategies from .ipynb notebooks
pine_parser.py         # TradingView Pine Script → Python converter
requirements.txt       # Python dependencies
assets/                # Static assets (e.g. assets/tv_chart.html)
cache/                 # local cache used by the app/tools
pybacktest_cache/      # another cache dir
sessions/              # session data storage for web UI
__pycache__/           # Python byte-compiled cache (ignored)
.gitignore
```

How it fits together:
- engine.py contains the Strategy base class and IndicatorEngine used by strategies. Strategies are vectorized in `init()` (indicators) and implement per-bar logic in `next(bar, data)`.
- pine_parser.py can convert Pine v5 scripts into a Python Strategy class (auto-generated code uses IND helper functions).
- notebook_runner.py extracts Strategy classes and helper code from Jupyter notebooks and can save or execute them as Python modules.
- app.py ties these pieces into a small web UI / API to run backtests and show charts; charts.py builds the plotting and chart export functions used by the web UI.

---

## Example strategy (from repository)

```python name=my_strategy.py url=https://github.com/raviwijerathna1/PyBacktestPro/blob/main/my_strategy.py
"""Imported from Jupyter"""
import pandas as pd
import numpy as np
from engine import Strategy, IndicatorEngine as IND

def cross_above(a,b,bar):
    if bar<1: return False
    try: return a.iloc[bar-1]<=b.iloc[bar-1] and a.iloc[bar]>b.iloc[bar]
    except: return False

def cross_below(a,b,bar):
    if bar<1: return False
    try: return a.iloc[bar-1]>=b.iloc[bar-1] and a.iloc[bar]<b.iloc[bar]
    except: return False

# Paste your Jupyter cell here
class MyStrategy(Strategy):
    fast_len = 10
    slow_len = 30
    atr_sl = 2.0
    atr_tp = 3.0

    def init(self):
        c = self._data['Close']
        h = self._data['High']
        l = self._data['Low']
        self._data['fast'] = IND.sma(c, self.fast_len)
        self._data['slow'] = IND.sma(c, self.slow_len)
        self._data['ATR'] = IND.atr(h, l, c, 14)
        self._data['RSI'] = IND.rsi(c, 14)

    def next(self, bar, data):
        if bar < self.slow_len + 1:
            return
        c = data['Close'].iloc[bar]
        atr = data['ATR'].iloc[bar]
        if pd.isna(atr) or atr == 0:
            return
        if cross_above(data['fast'], data['slow'], bar) and self.is_flat:
            self.buy(sl=c - atr * self.atr_sl, tp=c + atr * self.atr_tp, tag='Long')
        elif cross_below(data['fast'], data['slow'], bar) and self.is_long:
            self.close_position(tag='Exit')
```

---

## Dependencies

Primary dependencies are declared in requirements.txt:
- yfinance
- pandas
- numpy
- matplotlib
- scipy
- jinja2
- requests

Install with:
```bash
python -m pip install -r requirements.txt
```

---

## Notes & tips
- The repository includes tools to convert Pine scripts and Jupyter notebooks; these return Python code strings which you can save and run with the engine.
- app.py exposes API endpoints for chart data and backtest control — open the file to see available routes and how the web UI expects to call the engine.
- assets/tv_chart.html is provided for chart rendering/export; charts.py contains helpers to generate plots used by the UI.
- No LICENSE file detected — consider adding an open source license if you plan to share or accept contributions.

---

## Contributing 
- Read the code in engine.py, pine_parser.py and notebook_runner.py to understand expected interfaces (Strategy.init, Strategy.next, IND functions).
- Add tests or example notebooks in a /examples or /notebooks directory to help users reproduce workflows.
- If adding web features, ensure requirements.txt includes web framework dependencies.

---

## Questions to consider next
- Do you want a CLI wrapper to run backtests directly from the command line (e.g., `pybacktest run --strategy my_strategy.py --data SYMBOL`)?
- Should we add automated tests and a CI workflow that runs example backtests and validates parser outputs?
- Would you like an example Jupyter notebook demonstrating the full workflow: convert Pine → save Strategy → run backtest → show charts?

If you want, I can create a polished README.md file in the repository with this content and an example CLI snippet wired to the engine (if you confirm the desired run interface).

***********************
