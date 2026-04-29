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
