"""
════════════════════════════════════════════════════════
✏️  ඔබේ STRATEGY මෙතන ලියන්න
    Save කර Dashboard එකේ "▶ Run" click → auto reload!
════════════════════════════════════════════════════════

TradingView Pine Script          Python equivalent
─────────────────────            ─────────────────
ta.sma(close, 10)         →     ta_sma(close, 10)
ta.ema(close, 21)         →     ta_ema(close, 21)
ta.rsi(close, 14)         →     ta_rsi(close, 14)
ta.macd(close)            →     ta_macd(close)
ta.atr(14)                →     ta_atr(high, low, close, 14)
ta.bb(close, 20, 2)       →     ta_bb(close, 20, 2)
ta.crossover(a, b)        →     cross_above(a, b, bar)
ta.crossunder(a, b)       →     cross_below(a, b, bar)
close[1]                  →     close.iloc[bar - 1]
strategy.entry("L",long)  →     self.buy(...)
strategy.close("L")       →     self.close_position()
════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from engine import Strategy, IndicatorEngine as IND


# ── Pine Script Helper Functions ─────────────────────
def ta_sma(src, length):    return IND.sma(src, length)
def ta_ema(src, length):    return IND.ema(src, length)
def ta_rsi(src, length=14): return IND.rsi(src, length)
def ta_atr(h, l, c, length=14): return IND.atr(h, l, c, length)

def ta_macd(src, fast=12, slow=26, sig=9):
    return IND.macd(src, fast, slow, sig)

def ta_bb(src, length=20, mult=2.0):
    return IND.bollinger_bands(src, length, mult)

def ta_stoch(h, l, c, k=14, d=3):
    return IND.stochastic(h, l, c, k, d)

def ta_supertrend(h, l, c, length=10, mult=3.0):
    return IND.supertrend(h, l, c, length, mult)

def cross_above(a, b, bar):
    """ta.crossover(a, b)"""
    if bar < 1: return False
    try:
        return (a.iloc[bar-1] <= b.iloc[bar-1] and
                a.iloc[bar] > b.iloc[bar])
    except: return False

def cross_below(a, b, bar):
    """ta.crossunder(a, b)"""
    if bar < 1: return False
    try:
        return (a.iloc[bar-1] >= b.iloc[bar-1] and
                a.iloc[bar] < b.iloc[bar])
    except: return False


# ═════════════════════════════════════════════════════
# 📌 ACTIVE STRATEGY — මේක Edit කරන්න!
# ═════════════════════════════════════════════════════

class MyStrategy(Strategy):
    """
    SMA Crossover + ATR Stop/Target
    ──────────────────────────────
    Pine Script:
        fast = ta.sma(close, 10)
        slow = ta.sma(close, 30)
        if ta.crossover(fast, slow)
            strategy.entry("Long", strategy.long)
        if ta.crossunder(fast, slow)
            strategy.close("Long")
    """

    # ── Parameters ───────────────────────────────
    fast_len = 10
    slow_len = 30
    atr_sl = 2.0      # ATR × 2 stop loss
    atr_tp = 3.0      # ATR × 3 take profit

    def init(self):
        """Indicators ගණනය (vectorized — fast)"""
        c = self._data['Close']
        h = self._data['High']
        l = self._data['Low']

        self._data['fast'] = ta_sma(c, self.fast_len)
        self._data['slow'] = ta_sma(c, self.slow_len)
        self._data['ATR']  = ta_atr(h, l, c, 14)
        self._data['RSI']  = ta_rsi(c, 14)

    def next(self, bar, data):
        """Bar-by-bar trading logic"""
        if bar < self.slow_len + 1:
            return

        c   = data['Close'].iloc[bar]
        atr = data['ATR'].iloc[bar]
        rsi = data['RSI'].iloc[bar]

        if pd.isna(atr) or atr == 0:
            return

        # ── BUY: fast crosses above slow ─────────
        if (cross_above(data['fast'], data['slow'], bar)
                and self.is_flat):
            self.buy(
                sl=c - atr * self.atr_sl,
                tp=c + atr * self.atr_tp,
                tag='SMA_Cross_Up'
            )

        # ── SELL: fast crosses below slow ────────
        elif (cross_below(data['fast'], data['slow'], bar)
              and self.is_long):
            self.close_position(tag='SMA_Cross_Down')


# ═════════════════════════════════════════════════════
# 📝 EXAMPLE STRATEGIES — uncomment කර MyStrategy replace
# ═════════════════════════════════════════════════════

# class MyStrategy(Strategy):
#     """RSI + Bollinger Bands Mean Reversion"""
#     rsi_len = 14
#     bb_len = 20
#
#     def init(self):
#         c = self._data['Close']
#         self._data['RSI'] = ta_rsi(c, self.rsi_len)
#         u, m, l = ta_bb(c, self.bb_len)
#         self._data['BB_Upper'] = u
#         self._data['BB_Mid'] = m
#         self._data['BB_Lower'] = l
#
#     def next(self, bar, data):
#         if bar < 30: return
#         c = data['Close'].iloc[bar]
#         rsi = data['RSI'].iloc[bar]
#         bbl = data['BB_Lower'].iloc[bar]
#         bbm = data['BB_Mid'].iloc[bar]
#         if pd.isna(rsi): return
#
#         if rsi < 30 and c <= bbl * 1.01 and self.is_flat:
#             self.buy(sl=c*0.97, tp=bbm, tag='RSI_BB')
#         elif rsi > 70 and self.is_long:
#             self.close_position(tag='RSI_OB')


# class MyStrategy(Strategy):
#     """MACD + ATR (Long & Short)"""
#     def init(self):
#         c = self._data['Close']
#         ml, sl, hist = ta_macd(c)
#         self._data['MACD'] = ml
#         self._data['MACD_Signal'] = sl
#         self._data['MACD_Hist'] = hist
#         self._data['ATR'] = ta_atr(
#             self._data['High'], self._data['Low'], c, 14)
#
#     def next(self, bar, data):
#         if bar < 35: return
#         hist = data['MACD_Hist']
#         c = data['Close'].iloc[bar]
#         atr = data['ATR'].iloc[bar]
#         if pd.isna(atr) or atr == 0: return
#
#         if hist.iloc[bar-1] <= 0 and hist.iloc[bar] > 0:
#             if self.is_short: self.close_position()
#             if self.is_flat:
#                 self.buy(sl=c-atr*2, tp=c+atr*3, tag='MACD_L')
#         elif hist.iloc[bar-1] >= 0 and hist.iloc[bar] < 0:
#             if self.is_long: self.close_position()
#             if self.is_flat:
#                 self.sell(sl=c+atr*2, tp=c-atr*3, tag='MACD_S')