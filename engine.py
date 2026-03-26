"""
═══════════════════════════════════════════════════════════════
PyBacktest Pro — Professional Strategy Backtesting Engine
All 15 issues resolved ✅
═══════════════════════════════════════════════════════════════
"""

import os, sys, time, json, hashlib, logging, sqlite3, warnings
import io, base64
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Tuple, Any, Callable, Union
from copy import deepcopy
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

warnings.filterwarnings('ignore')

# ─── Logging ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('PyBacktestPro')


# ══════════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════════
class OrderType(Enum):
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()
    TRAILING_STOP = auto()

class OrderSide(Enum):
    BUY = auto()
    SELL = auto()

class OrderStatus(Enum):
    PENDING = auto()
    FILLED = auto()
    CANCELLED = auto()
    EXPIRED = auto()
    PARTIALLY_FILLED = auto()

class PositionSide(Enum):
    LONG = auto()
    SHORT = auto()
    FLAT = auto()

class SizingMethod(Enum):
    FIXED_AMOUNT = auto()
    FIXED_FRACTIONAL = auto()
    KELLY = auto()
    ATR_BASED = auto()
    VOLATILITY_TARGET = auto()
    FULL_EQUITY = auto()


# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_pct: float = 0.001
    commission_fixed: float = 0.0
    slippage_pct: float = 0.0005
    slippage_fixed: float = 0.0
    volume_impact_factor: float = 0.1
    use_volume_slippage: bool = True
    margin_requirement: float = 0.5
    short_borrow_rate: float = 0.02
    allow_short: bool = True
    sizing_method: SizingMethod = SizingMethod.FIXED_FRACTIONAL
    sizing_param: float = 0.02
    max_position_pct: float = 0.95
    execute_on_close: bool = False
    enable_fractional: bool = True
    risk_free_rate: float = 0.04

@dataclass
class CacheConfig:
    cache_dir: str = './pybacktest_cache'
    db_path: str = './pybacktest_cache/market_data.db'
    cache_ttl_hours: int = 12
    max_retries: int = 3
    retry_delay: float = 2.0

@dataclass
class WalkForwardConfig:
    n_splits: int = 5
    in_sample_pct: float = 0.7
    anchored: bool = False

@dataclass
class MonteCarloConfig:
    n_simulations: int = 1000
    confidence_levels: List[float] = field(
        default_factory=lambda: [0.95, 0.99]
    )
    block_size: int = 5


# ══════════════════════════════════════════════════════════════
# DATABASE & CACHE — Issues #1, #15
# ══════════════════════════════════════════════════════════════
class DatabaseManager:
    def __init__(self, config: CacheConfig = CacheConfig()):
        self.config = config
        os.makedirs(config.cache_dir, exist_ok=True)
        self.conn = sqlite3.connect(
            config.db_path, check_same_thread=False, timeout=30
        )
        self._init_tables()
        logger.info(f"Database: {config.db_path}")

    def _init_tables(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS ohlcv_cache (
            cache_key TEXT PRIMARY KEY, symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL, data_json TEXT NOT NULL,
            row_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS trade_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL, strategy_name TEXT,
            symbol TEXT, side TEXT, entry_date TEXT,
            exit_date TEXT, entry_price REAL, exit_price REAL,
            quantity REAL, pnl REAL, pnl_pct REAL,
            commission REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS backtest_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL, strategy_name TEXT,
            symbol TEXT, start_date TEXT, end_date TEXT,
            total_return REAL, sharpe_ratio REAL,
            max_drawdown REAL, win_rate REAL,
            total_trades INTEGER, config_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        self.conn.commit()

    def _make_key(self, sym, start, end, interval):
        raw = f"{sym}_{start}_{end}_{interval}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get_cached_data(self, sym, start, end, interval):
        key = self._make_key(sym, start, end, interval)
        try:
            cur = self.conn.cursor()
            cur.execute(
                'SELECT data_json, expires_at FROM ohlcv_cache '
                'WHERE cache_key=?', (key,)
            )
            row = cur.fetchone()
            if not row:
                return None
            if datetime.now() > datetime.fromisoformat(row[1]):
                cur.execute(
                    'DELETE FROM ohlcv_cache WHERE cache_key=?', (key,)
                )
                self.conn.commit()
                return None
            df = pd.read_json(io.StringIO(row[0]), orient='split')
            df.index = pd.to_datetime(df.index)
            logger.info(f"✅ Cache hit: {sym} ({len(df)} rows)")
            return df
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None

    def save_to_cache(self, df, sym, start, end, interval):
        key = self._make_key(sym, start, end, interval)
        expires = datetime.now() + timedelta(
            hours=self.config.cache_ttl_hours
        )
        try:
            data_json = df.to_json(orient='split', date_format='iso')
            self.conn.execute(
                'INSERT OR REPLACE INTO ohlcv_cache '
                '(cache_key,symbol,timeframe,data_json,row_count,'
                'expires_at) VALUES(?,?,?,?,?,?)',
                (key, sym, interval, data_json,
                 len(df), expires.isoformat())
            )
            self.conn.commit()
            logger.info(f"✅ Cached: {sym} ({len(df)} rows)")
        except Exception as e:
            logger.error(f"Cache write error: {e}")

    def save_trades(self, trades, session_id, strategy_name, symbol):
        cur = self.conn.cursor()
        for t in trades:
            cur.execute(
                'INSERT INTO trade_history '
                '(session_id,strategy_name,symbol,side,entry_date,'
                'exit_date,entry_price,exit_price,quantity,pnl,'
                'pnl_pct,commission) '
                'VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                (session_id, strategy_name, symbol,
                 t.get('side',''), str(t.get('entry_date','')),
                 str(t.get('exit_date','')),
                 t.get('entry_price',0), t.get('exit_price',0),
                 t.get('quantity',0), t.get('pnl',0),
                 t.get('pnl_pct',0), t.get('commission',0))
            )
        self.conn.commit()
        logger.info(f"✅ Saved {len(trades)} trades to DB")

    def save_backtest_result(self, result, session_id):
        self.conn.execute(
            'INSERT INTO backtest_results '
            '(session_id,strategy_name,symbol,start_date,end_date,'
            'total_return,sharpe_ratio,max_drawdown,win_rate,'
            'total_trades,config_json) '
            'VALUES(?,?,?,?,?,?,?,?,?,?,?)',
            (session_id, result.get('strategy_name',''),
             result.get('symbol',''),
             result.get('start_date',''),
             result.get('end_date',''),
             result.get('total_return',0),
             result.get('sharpe_ratio',0),
             result.get('max_drawdown',0),
             result.get('win_rate',0),
             result.get('total_trades',0),
             json.dumps(result.get('config',{})))
        )
        self.conn.commit()

    def get_all_results(self):
        return pd.read_sql(
            'SELECT * FROM backtest_results ORDER BY created_at DESC',
            self.conn
        )

    def close(self):
        self.conn.close()


# ══════════════════════════════════════════════════════════════
# DATA MANAGER — Issues #2, #12, #14
# ══════════════════════════════════════════════════════════════
class DataManager:
    def __init__(self, cache_config=CacheConfig(),
                 csv_data_dir='./csv_data'):
        self.db = DatabaseManager(cache_config)
        self.cache_config = cache_config
        self.csv_data_dir = csv_data_dir

    def _fix_multiindex(self, df):
        """Issue #14: MultiIndex fix"""
        if isinstance(df.columns, pd.MultiIndex):
            logger.info("Fixing MultiIndex columns...")
            if df.columns.nlevels == 2:
                symbols = df.columns.get_level_values(1).unique()
                if len(symbols) == 1:
                    df.columns = df.columns.get_level_values(0)
                else:
                    df.columns = [
                        f"{c[0]}_{c[1]}" for c in df.columns
                    ]
        col_map = {}
        for col in df.columns:
            cl = str(col).lower().strip()
            if cl in ('open','adj open'):
                col_map[col] = 'Open'
            elif cl in ('high','adj high'):
                col_map[col] = 'High'
            elif cl in ('low','adj low'):
                col_map[col] = 'Low'
            elif cl in ('close','adj close','adj_close'):
                col_map[col] = 'Close'
            elif cl in ('volume','vol'):
                col_map[col] = 'Volume'
        if col_map:
            df = df.rename(columns=col_map)
        return df

    def _validate(self, df):
        required = ['Open','High','Low','Close']
        for c in required:
            if c not in df.columns:
                return False
        return len(df) >= 2 and not df[required].isnull().all().any()

    def _fetch_yfinance(self, sym, start, end, interval):
        try:
            import yfinance as yf
        except ImportError:
            return None
        for attempt in range(self.cache_config.max_retries):
            try:
                logger.info(
                    f"yfinance: {sym} (attempt {attempt+1})"
                )
                df = yf.Ticker(sym).history(
                    start=start, end=end, interval=interval,
                    auto_adjust=True, actions=False
                )
                if df is not None and len(df) > 0:
                    df = self._fix_multiindex(df)
                    if self._validate(df):
                        return df
            except Exception as e:
                logger.warning(
                    f"yfinance attempt {attempt+1} failed: {e}"
                )
                time.sleep(
                    self.cache_config.retry_delay * (attempt + 1)
                )
        return None

    def _fetch_stooq(self, sym, start, end, interval):
        try:
            import requests
            url = (
                f"https://stooq.com/q/d/l/"
                f"?s={sym.replace('-','.')}"
                f"&d1={start.replace('-','')}"
                f"&d2={end.replace('-','')}&i=d"
            )
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and len(resp.text) > 100:
                df = pd.read_csv(io.StringIO(resp.text))
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.set_index('Date').sort_index()
                    df = self._fix_multiindex(df)
                    if self._validate(df):
                        return df
        except Exception as e:
            logger.warning(f"Stooq failed: {e}")
        return None

    def _fetch_csv(self, sym, start, end, interval):
        patterns = [f"{sym}.csv", f"{sym.lower()}.csv",
                    f"{sym.upper()}.csv"]
        for p in patterns:
            fp = os.path.join(self.csv_data_dir, p)
            if os.path.exists(fp):
                try:
                    df = pd.read_csv(fp, parse_dates=True,
                                     index_col=0)
                    df = self._fix_multiindex(df)
                    df.index = pd.to_datetime(df.index)
                    mask = (df.index >= start) & (df.index <= end)
                    df = df.loc[mask]
                    if self._validate(df):
                        return df
                except Exception as e:
                    logger.warning(f"CSV error: {e}")
        return None

    def fetch(self, symbol, start='2020-01-01', end=None,
              interval='1d', force_refresh=False):
        if end is None:
            end = datetime.now().strftime('%Y-%m-%d')
        if not force_refresh:
            cached = self.db.get_cached_data(
                symbol, start, end, interval
            )
            if cached is not None:
                return cached
        methods = [
            ('yfinance', self._fetch_yfinance),
            ('stooq', self._fetch_stooq),
            ('csv', self._fetch_csv),
        ]
        df = None
        for name, fn in methods:
            try:
                df = fn(symbol, start, end, interval)
                if df is not None and len(df) > 0:
                    logger.info(f"✅ {name}: {symbol} ({len(df)})")
                    break
            except Exception as e:
                logger.warning(f"{name} failed: {e}")
        if df is None or len(df) == 0:
            raise ValueError(
                f"❌ No data for {symbol} from any source!\n"
                f"Place CSV: {self.csv_data_dir}/{symbol}.csv"
            )
        df = df.dropna(subset=['Open','High','Low','Close'])
        if 'Volume' not in df.columns:
            df['Volume'] = 0
        df['Volume'] = df['Volume'].fillna(0)
        df = df.sort_index()
        self.db.save_to_cache(df, symbol, start, end, interval)
        return df

    def fetch_multi_timeframe(self, symbol, start='2020-01-01',
                               end=None, base_interval='1d',
                               higher_intervals=None):
        if higher_intervals is None:
            higher_intervals = ['1wk', '1mo']
        result = {}
        base = self.fetch(symbol, start, end, base_interval)
        result[base_interval] = base
        rmap = {'1h':'h','4h':'4h','1d':'D',
                '1wk':'W','1mo':'ME','W':'W','M':'ME'}
        for htf in higher_intervals:
            try:
                rule = rmap.get(htf, htf)
                hdf = base.resample(rule).agg({
                    'Open':'first','High':'max','Low':'min',
                    'Close':'last','Volume':'sum'
                }).dropna()
                result[htf] = hdf
            except Exception as e:
                logger.warning(f"Resample {htf}: {e}")
        return result


# ══════════════════════════════════════════════════════════════
# INDICATOR ENGINE — Issue #3 (Look-ahead bias free)
# ══════════════════════════════════════════════════════════════
class IndicatorEngine:
    @staticmethod
    def sma(close, period=20):
        return close.rolling(window=period,
                             min_periods=period).mean()

    @staticmethod
    def ema(close, period=20):
        return close.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(close, period=14):
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        ag = gain.ewm(com=period-1, min_periods=period).mean()
        al = loss.ewm(com=period-1, min_periods=period).mean()
        rs = ag / al.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(close, fast=12, slow=26, signal=9):
        ef = close.ewm(span=fast, adjust=False).mean()
        es = close.ewm(span=slow, adjust=False).mean()
        ml = ef - es
        sl = ml.ewm(span=signal, adjust=False).mean()
        return ml, sl, ml - sl

    @staticmethod
    def bollinger_bands(close, period=20, std_dev=2.0):
        mid = close.rolling(period, min_periods=period).mean()
        std = close.rolling(period, min_periods=period).std()
        return mid + std * std_dev, mid, mid - std * std_dev

    @staticmethod
    def atr(high, low, close, period=14):
        prev = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev).abs(),
            (low - prev).abs()
        ], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    @staticmethod
    def stochastic(high, low, close, k_period=14, d_period=3):
        ll = low.rolling(k_period).min()
        hh = high.rolling(k_period).max()
        k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
        d = k.rolling(d_period).mean()
        return k, d

    @staticmethod
    def adx(high, low, close, period=14):
        pdm = high.diff()
        mdm = -low.diff()
        pdm = pdm.where((pdm > mdm) & (pdm > 0), 0.0)
        mdm = mdm.where((mdm > pdm) & (mdm > 0), 0.0)
        a = IndicatorEngine.atr(high, low, close, period)
        pdi = 100*(pdm.ewm(span=period,adjust=False).mean()
                    / a.replace(0,np.nan))
        mdi = 100*(mdm.ewm(span=period,adjust=False).mean()
                    / a.replace(0,np.nan))
        dx = 100*((pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan))
        return dx.ewm(span=period, adjust=False).mean()

    @staticmethod
    def vwap(high, low, close, volume):
        tp = (high + low + close) / 3
        return (tp * volume).cumsum() / volume.cumsum().replace(
            0, np.nan
        )

    @staticmethod
    def supertrend(high, low, close, period=10, mult=3.0):
        a = IndicatorEngine.atr(high, low, close, period)
        hl2 = (high + low) / 2
        ub = hl2 + mult * a
        lb = hl2 - mult * a
        st = pd.Series(np.nan, index=close.index)
        d = pd.Series(1, index=close.index)
        for i in range(1, len(close)):
            if close.iloc[i] > ub.iloc[i-1]:
                d.iloc[i] = 1
            elif close.iloc[i] < lb.iloc[i-1]:
                d.iloc[i] = -1
            else:
                d.iloc[i] = d.iloc[i-1]
            st.iloc[i] = lb.iloc[i] if d.iloc[i]==1 else ub.iloc[i]
        return st, d

    @staticmethod
    def safe_indicator(func, *args, shift_bars=0, **kwargs):
        result = func(*args, **kwargs)
        if shift_bars > 0:
            if isinstance(result, pd.Series):
                result = result.shift(shift_bars)
            elif isinstance(result, tuple):
                result = tuple(
                    s.shift(shift_bars) if isinstance(s, pd.Series)
                    else s for s in result
                )
        return result

IND = IndicatorEngine


# ══════════════════════════════════════════════════════════════
# ORDER MANAGEMENT — Issue #5
# ══════════════════════════════════════════════════════════════
@dataclass
class Order:
    order_type: OrderType
    side: OrderSide
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_pct: Optional[float] = None
    trail_amount: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_bar: int = 0
    filled_bar: Optional[int] = None
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    expiry_bars: int = 0
    tag: str = ''
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    _trail_extreme: Optional[float] = None

    @property
    def is_active(self):
        return self.status in (
            OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED
        )


class OrderManager:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.pending_orders: List[Order] = []
        self.filled_orders: List[Order] = []
        self.cancelled_orders: List[Order] = []

    def submit(self, order: Order):
        self.pending_orders.append(order)
        return order

    def cancel_all(self):
        for o in self.pending_orders:
            o.status = OrderStatus.CANCELLED
            self.cancelled_orders.append(o)
        self.pending_orders.clear()

    def cancel_by_tag(self, tag):
        remaining = []
        for o in self.pending_orders:
            if o.tag == tag:
                o.status = OrderStatus.CANCELLED
                self.cancelled_orders.append(o)
            else:
                remaining.append(o)
        self.pending_orders = remaining

    def process_bar(self, bar_idx, open_p, high, low, close, vol):
        filled = []
        still_pending = []
        for order in self.pending_orders:
            if (order.expiry_bars > 0 and
                    bar_idx - order.created_bar >= order.expiry_bars):
                order.status = OrderStatus.EXPIRED
                self.cancelled_orders.append(order)
                continue

            done = False
            fp = None

            if order.order_type == OrderType.MARKET:
                fp = open_p
                done = True

            elif order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY:
                    if low <= order.price:
                        fp = min(order.price, open_p)
                        done = True
                else:
                    if high >= order.price:
                        fp = max(order.price, open_p)
                        done = True

            elif order.order_type == OrderType.STOP:
                if order.side == OrderSide.BUY:
                    if high >= order.stop_price:
                        fp = max(order.stop_price, open_p)
                        done = True
                else:
                    if low <= order.stop_price:
                        fp = min(order.stop_price, open_p)
                        done = True

            elif order.order_type == OrderType.STOP_LIMIT:
                triggered = False
                if order.side == OrderSide.BUY:
                    triggered = high >= order.stop_price
                else:
                    triggered = low <= order.stop_price
                if triggered:
                    if order.side == OrderSide.BUY:
                        if low <= order.price:
                            fp = min(order.price, open_p)
                            done = True
                    else:
                        if high >= order.price:
                            fp = max(order.price, open_p)
                            done = True

            elif order.order_type == OrderType.TRAILING_STOP:
                if order._trail_extreme is None:
                    order._trail_extreme = (
                        high if order.side == OrderSide.SELL
                        else low
                    )
                if order.side == OrderSide.SELL:
                    order._trail_extreme = max(
                        order._trail_extreme, high
                    )
                    trigger = (
                        order._trail_extreme * (1 - order.trail_pct)
                        if order.trail_pct
                        else order._trail_extreme - order.trail_amount
                    )
                    if low <= trigger:
                        fp = min(trigger, open_p)
                        done = True
                else:
                    order._trail_extreme = min(
                        order._trail_extreme, low
                    )
                    trigger = (
                        order._trail_extreme * (1 + order.trail_pct)
                        if order.trail_pct
                        else order._trail_extreme + order.trail_amount
                    )
                    if high >= trigger:
                        fp = max(trigger, open_p)
                        done = True

            if done and fp is not None:
                fp = self._apply_slippage(fp, order.side, vol)
                order.status = OrderStatus.FILLED
                order.filled_bar = bar_idx
                order.filled_price = fp
                order.filled_quantity = order.quantity
                self.filled_orders.append(order)
                filled.append(order)
            else:
                still_pending.append(order)

        self.pending_orders = still_pending
        return filled

    def _apply_slippage(self, price, side, volume):
        slip = self.config.slippage_fixed
        slip += price * self.config.slippage_pct
        if self.config.use_volume_slippage and volume > 0:
            slip += price * self.config.volume_impact_factor / max(
                volume, 1
            ) * 1000
        if side == OrderSide.BUY:
            return price + slip
        return price - slip

    def reset(self):
        self.pending_orders.clear()
        self.filled_orders.clear()
        self.cancelled_orders.clear()


# ══════════════════════════════════════════════════════════════
# POSITION SIZING — Issue #7
# ══════════════════════════════════════════════════════════════
class PositionSizer:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self._history: List[float] = []

    def record_trade(self, pnl_pct):
        self._history.append(pnl_pct)

    def calculate_size(self, equity, price, atr=0.0,
                       volatility=0.0, stop_distance=0.0,
                       method=None):
        if method is None:
            method = self.config.sizing_method
        max_val = equity * self.config.max_position_pct
        size = 0.0

        if method == SizingMethod.FIXED_AMOUNT:
            size = self.config.sizing_param / price
        elif method == SizingMethod.FIXED_FRACTIONAL:
            risk = equity * self.config.sizing_param
            size = risk / stop_distance if stop_distance > 0 else risk / price
        elif method == SizingMethod.KELLY:
            size = self._kelly(equity, price)
        elif method == SizingMethod.ATR_BASED:
            if atr > 0:
                size = (equity * self.config.sizing_param) / (atr * 2)
            else:
                size = (equity * 0.02) / price
        elif method == SizingMethod.VOLATILITY_TARGET:
            if volatility > 0:
                size = (equity * self.config.sizing_param) / (
                    price * volatility * np.sqrt(252)
                )
            else:
                size = (equity * 0.02) / price
        elif method == SizingMethod.FULL_EQUITY:
            size = max_val / price

        size = min(size, max_val / price)
        if not self.config.enable_fractional:
            size = int(size)
        return max(size, 0)

    def _kelly(self, equity, price):
        if len(self._history) < 10:
            return (equity * 0.02) / price
        wins = [t for t in self._history if t > 0]
        losses = [t for t in self._history if t < 0]
        if not wins or not losses:
            return (equity * 0.02) / price
        b = np.mean(wins) / abs(np.mean(losses))
        p = len(wins) / len(self._history)
        q = 1 - p
        kf = (b * p - q) / b
        kf = max(0, min(kf * 0.5, 0.25))
        return (equity * kf) / price


# ══════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME — Issue #10
# ══════════════════════════════════════════════════════════════
class MultiTimeframe:
    @staticmethod
    def resample(df, rule='W'):
        return df.resample(rule).agg({
            'Open':'first','High':'max',
            'Low':'min','Close':'last','Volume':'sum'
        }).dropna()

    @staticmethod
    def merge_higher_tf(base, htf, cols, prefix='HTF_'):
        result = base.copy()
        for col in cols:
            if col in htf.columns:
                s = htf[col].shift(1)
                result[f'{prefix}{col}'] = s.reindex(
                    base.index, method='ffill'
                )
        return result

    @staticmethod
    def add_htf_indicators(base_df, htf_rule='W',
                           indicators=None):
        if indicators is None:
            indicators = {
                'SMA_20': {'func':'sma','params':{'period':20}},
                'SMA_50': {'func':'sma','params':{'period':50}},
                'RSI_14': {'func':'rsi','params':{'period':14}},
            }
        htf = MultiTimeframe.resample(base_df, htf_rule)
        for name, spec in indicators.items():
            fn = spec['func']
            params = spec.get('params', {})
            if fn == 'sma':
                htf[name] = IND.sma(htf['Close'], **params)
            elif fn == 'ema':
                htf[name] = IND.ema(htf['Close'], **params)
            elif fn == 'rsi':
                htf[name] = IND.rsi(htf['Close'], **params)
            elif fn == 'atr':
                htf[name] = IND.atr(
                    htf['High'], htf['Low'], htf['Close'], **params
                )
        return MultiTimeframe.merge_higher_tf(
            base_df, htf, list(indicators.keys()),
            prefix=f'HTF_{htf_rule}_'
        )


# ══════════════════════════════════════════════════════════════
# ENGINE CORE — Issues #3, #4, #6
# ══════════════════════════════════════════════════════════════
@dataclass
class Position:
    side: PositionSide
    quantity: float
    entry_price: float
    entry_date: Any
    entry_bar: int
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    tag: str = ''

@dataclass
class TradeRecord:
    side: str
    entry_date: Any
    exit_date: Any
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    commission: float
    bars_held: int
    tag: str = ''


class Strategy(ABC):
    def __init__(self):
        self.name = self.__class__.__name__
        self.params: Dict[str, Any] = {}
        self._engine: Optional['BacktestEngine'] = None
        self._data: Optional[pd.DataFrame] = None
        self._bar_idx: int = 0

    def init(self):
        pass

    @abstractmethod
    def next(self, bar: int, data: pd.DataFrame):
        pass

    def buy(self, qty=0, price=None, sl=None, tp=None,
            order_type=OrderType.MARKET, tag=''):
        self._engine._submit_order(
            OrderSide.BUY, qty, price, None,
            order_type, sl, tp, tag
        )

    def sell(self, qty=0, price=None, sl=None, tp=None,
             order_type=OrderType.MARKET, tag=''):
        self._engine._submit_order(
            OrderSide.SELL, qty, price, None,
            order_type, sl, tp, tag
        )

    def buy_limit(self, qty, price, sl=None, tp=None, tag=''):
        self._engine._submit_order(
            OrderSide.BUY, qty, price, None,
            OrderType.LIMIT, sl, tp, tag
        )

    def sell_limit(self, qty, price, sl=None, tp=None, tag=''):
        self._engine._submit_order(
            OrderSide.SELL, qty, price, None,
            OrderType.LIMIT, sl, tp, tag
        )

    def buy_stop(self, qty, stop_price, sl=None, tp=None, tag=''):
        self._engine._submit_order(
            OrderSide.BUY, qty, None, stop_price,
            OrderType.STOP, sl, tp, tag
        )

    def sell_stop(self, qty, stop_price, sl=None, tp=None, tag=''):
        self._engine._submit_order(
            OrderSide.SELL, qty, None, stop_price,
            OrderType.STOP, sl, tp, tag
        )

    def close_position(self, tag=''):
        self._engine._close_position(tag)

    def cancel_all_orders(self):
        self._engine.order_manager.cancel_all()

    @property
    def position(self):
        return self._engine.current_position

    @property
    def is_long(self):
        p = self._engine.current_position
        return p is not None and p.side == PositionSide.LONG

    @property
    def is_short(self):
        p = self._engine.current_position
        return p is not None and p.side == PositionSide.SHORT

    @property
    def is_flat(self):
        return self._engine.current_position is None

    @property
    def equity(self):
        return self._engine.current_equity

    @property
    def cash(self):
        return self._engine.current_cash


class BacktestEngine:
    def __init__(self, config=BacktestConfig()):
        self.config = config
        self.order_manager = OrderManager(config)
        self.position_sizer = PositionSizer(config)
        self.current_cash = config.initial_capital
        self.current_equity = config.initial_capital
        self.current_position: Optional[Position] = None
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[float] = []
        self.cash_curve: List[float] = []
        self.drawdown_curve: List[float] = []
        self.dates: List[Any] = []
        self._data = None
        self._bar_idx = 0
        self._peak = config.initial_capital
        self._session_id = ''

    def reset(self):
        self.current_cash = self.config.initial_capital
        self.current_equity = self.config.initial_capital
        self.current_position = None
        self.trades.clear()
        self.equity_curve.clear()
        self.cash_curve.clear()
        self.drawdown_curve.clear()
        self.dates.clear()
        self.order_manager.reset()
        self._peak = self.config.initial_capital
        self._bar_idx = 0

    def run(self, strategy, data, progress=True):
        self.reset()
        self._session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        data = data.copy()
        for col in ['Open','High','Low','Close']:
            if col not in data.columns:
                raise ValueError(f"Missing column: {col}")
        if 'Volume' not in data.columns:
            data['Volume'] = 0
        self._data = data
        strategy._engine = self
        strategy._data = data
        strategy.init()

        n = len(data)
        ri = max(1, n // 20)
        t0 = time.time()
        if progress:
            logger.info(
                f"🚀 {strategy.name} | {n} bars | "
                f"${self.config.initial_capital:,.0f}"
            )

        for bar in range(n):
            self._bar_idx = bar
            row = data.iloc[bar]
            o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
            v = row.get('Volume', 0)

            filled = self.order_manager.process_bar(bar, o, h, l, c, v)
            for order in filled:
                self._handle_fill(order, bar, data.index[bar])

            self._check_sl_tp(bar, h, l, o, v, data.index[bar])
            self._update_equity(c)

            try:
                strategy.next(bar, data)
            except Exception as e:
                logger.error(f"Strategy error bar {bar}: {e}")

            self.equity_curve.append(self.current_equity)
            self.cash_curve.append(self.current_cash)
            self.dates.append(data.index[bar])

            if self.current_equity > self._peak:
                self._peak = self.current_equity
            dd = (self._peak - self.current_equity) / self._peak
            self.drawdown_curve.append(dd)

            if progress and bar % ri == 0 and bar > 0:
                print(
                    f"\r  {bar/n*100:.0f}% | "
                    f"${self.current_equity:,.0f} | "
                    f"{len(self.trades)} trades",
                    end='', flush=True
                )

        if self.current_position is not None:
            self._close_at_price(
                data['Close'].iloc[-1], n-1,
                data.index[-1], 'end'
            )
            self._update_equity(data['Close'].iloc[-1])
            self.equity_curve[-1] = self.current_equity

        if progress:
            print(
                f"\r✅ Done {time.time()-t0:.1f}s | "
                f"{n} bars | {len(self.trades)} trades"
                + " " * 20
            )

        return BacktestResult(self, strategy, data)

    def _submit_order(self, side, qty, price=None,
                      stop_price=None,
                      order_type=OrderType.MARKET,
                      sl=None, tp=None, tag=''):
        cp = self._data['Close'].iloc[self._bar_idx]
        if qty <= 0:
            atr_val = 0
            if 'ATR' in self._data.columns:
                atr_val = self._data['ATR'].iloc[self._bar_idx]
            elif self._bar_idx >= 14:
                atr_val = IND.atr(
                    self._data['High'][:self._bar_idx+1],
                    self._data['Low'][:self._bar_idx+1],
                    self._data['Close'][:self._bar_idx+1], 14
                ).iloc[-1]
            sd = abs(cp - sl) if sl else 0
            vol = (
                self._data['Close'][:self._bar_idx+1]
                .pct_change().std()
            ) if self._bar_idx > 5 else 0
            qty = self.position_sizer.calculate_size(
                self.current_equity, cp, atr_val, vol, sd
            )
        if qty <= 0:
            return
        if (side == OrderSide.SELL and
                not self.config.allow_short and
                self.current_position is None):
            return
        order = Order(
            order_type=order_type, side=side, quantity=qty,
            price=price, stop_price=stop_price,
            created_bar=self._bar_idx,
            sl_price=sl, tp_price=tp, tag=tag
        )
        self.order_manager.submit(order)

    def _handle_fill(self, order, bar, date):
        comm = self._calc_comm(order.filled_price,
                               order.filled_quantity)
        self.current_cash -= comm

        if order.side == OrderSide.BUY:
            if (self.current_position and
                    self.current_position.side == PositionSide.SHORT):
                self._close_at_price(
                    order.filled_price, bar, date, order.tag
                )
                return
            if (self.current_position and
                    self.current_position.side == PositionSide.LONG):
                pos = self.current_position
                tc = (pos.entry_price * pos.quantity +
                      order.filled_price * order.filled_quantity)
                tq = pos.quantity + order.filled_quantity
                pos.entry_price = tc / tq
                pos.quantity = tq
                self.current_cash -= (
                    order.filled_price * order.filled_quantity
                )
                if order.sl_price:
                    pos.sl_price = order.sl_price
                if order.tp_price:
                    pos.tp_price = order.tp_price
                return
            self.current_position = Position(
                PositionSide.LONG, order.filled_quantity,
                order.filled_price, date, bar,
                order.sl_price, order.tp_price, order.tag
            )
            self.current_cash -= (
                order.filled_price * order.filled_quantity
            )

        elif order.side == OrderSide.SELL:
            if (self.current_position and
                    self.current_position.side == PositionSide.LONG):
                self._close_at_price(
                    order.filled_price, bar, date, order.tag
                )
                return
            if (self.current_position and
                    self.current_position.side == PositionSide.SHORT):
                pos = self.current_position
                tc = (pos.entry_price * pos.quantity +
                      order.filled_price * order.filled_quantity)
                tq = pos.quantity + order.filled_quantity
                pos.entry_price = tc / tq
                pos.quantity = tq
                self.current_cash += (
                    order.filled_price * order.filled_quantity
                )
                if order.sl_price:
                    pos.sl_price = order.sl_price
                if order.tp_price:
                    pos.tp_price = order.tp_price
                return
            if self.config.allow_short:
                margin = (
                    order.filled_price * order.filled_quantity
                    * self.config.margin_requirement
                )
                if self.current_cash >= margin:
                    self.current_position = Position(
                        PositionSide.SHORT, order.filled_quantity,
                        order.filled_price, date, bar,
                        order.sl_price, order.tp_price, order.tag
                    )
                    self.current_cash += (
                        order.filled_price * order.filled_quantity
                    )

    def _close_position(self, tag=''):
        if not self.current_position:
            return
        side = (OrderSide.SELL
                if self.current_position.side == PositionSide.LONG
                else OrderSide.BUY)
        self.order_manager.submit(Order(
            OrderType.MARKET, side,
            self.current_position.quantity,
            created_bar=self._bar_idx, tag=tag
        ))

    def _close_at_price(self, price, bar, date, tag=''):
        if not self.current_position:
            return
        pos = self.current_position
        comm = self._calc_comm(price, pos.quantity)
        if pos.side == PositionSide.LONG:
            pnl = (price - pos.entry_price) * pos.quantity
            self.current_cash += price * pos.quantity
        else:
            pnl = (pos.entry_price - price) * pos.quantity
            self.current_cash -= price * pos.quantity
            days = max(1, bar - pos.entry_bar)
            borrow = (pos.entry_price * pos.quantity *
                      self.config.short_borrow_rate * days / 252)
            pnl -= borrow
        pnl -= comm
        self.current_cash -= comm
        pnl_pct = pnl / (pos.entry_price * pos.quantity)
        self.trades.append(TradeRecord(
            'LONG' if pos.side == PositionSide.LONG else 'SHORT',
            pos.entry_date, date, pos.entry_price, price,
            pos.quantity, pnl, pnl_pct, comm,
            bar - pos.entry_bar, tag
        ))
        self.position_sizer.record_trade(pnl_pct)
        self.current_position = None

    def _check_sl_tp(self, bar, h, l, o, v, date):
        if not self.current_position:
            return
        pos = self.current_position
        if pos.side == PositionSide.LONG:
            if pos.sl_price and l <= pos.sl_price:
                fp = min(pos.sl_price, o)
                fp = self.order_manager._apply_slippage(
                    fp, OrderSide.SELL, v
                )
                self._close_at_price(fp, bar, date, 'SL')
                return
            if pos.tp_price and h >= pos.tp_price:
                self._close_at_price(
                    max(pos.tp_price, o), bar, date, 'TP'
                )
                return
        else:
            if pos.sl_price and h >= pos.sl_price:
                fp = max(pos.sl_price, o)
                fp = self.order_manager._apply_slippage(
                    fp, OrderSide.BUY, v
                )
                self._close_at_price(fp, bar, date, 'SL')
                return
            if pos.tp_price and l <= pos.tp_price:
                self._close_at_price(
                    min(pos.tp_price, o), bar, date, 'TP'
                )
                return

    def _update_equity(self, price):
        if not self.current_position:
            self.current_equity = self.current_cash
        else:
            pos = self.current_position
            if pos.side == PositionSide.LONG:
                u = (price - pos.entry_price) * pos.quantity
            else:
                u = (pos.entry_price - price) * pos.quantity
            self.current_equity = self.current_cash + u

    def _calc_comm(self, price, qty):
        return max(self.config.commission_fixed,
                   price * qty * self.config.commission_pct)


# ══════════════════════════════════════════════════════════════
# BACKTEST RESULT
# ══════════════════════════════════════════════════════════════
class BacktestResult:
    def __init__(self, engine, strategy, data):
        self.engine = engine
        self.strategy_name = strategy.name
        self.config = engine.config
        self.data = data
        self.trades = engine.trades
        self.equity_curve = pd.Series(
            engine.equity_curve, index=engine.dates
        )
        self.drawdown_curve = pd.Series(
            engine.drawdown_curve, index=engine.dates
        )
        self.metrics = self._calc_metrics()

    def _calc_metrics(self):
        m = {}
        eq = self.equity_curve
        ini = self.config.initial_capital
        m['initial_capital'] = ini
        m['final_equity'] = eq.iloc[-1] if len(eq) else ini
        m['total_return'] = (m['final_equity'] - ini) / ini
        m['total_return_pct'] = m['total_return'] * 100
        nd = (eq.index[-1]-eq.index[0]).days if len(eq)>1 else 1
        ny = max(nd / 365.25, 0.01)
        m['n_years'] = ny
        m['cagr'] = (
            (m['final_equity']/ini)**(1/ny)-1
            if m['final_equity'] > 0 else 0
        )
        dr = eq.pct_change().dropna()
        m['daily_returns'] = dr
        if len(dr) > 1 and dr.std() > 0:
            m['sharpe_ratio'] = (
                (dr.mean() - self.config.risk_free_rate/252)
                / dr.std() * np.sqrt(252)
            )
        else:
            m['sharpe_ratio'] = 0
        ds = dr[dr < 0]
        m['sortino_ratio'] = (
            (dr.mean()-self.config.risk_free_rate/252)
            / ds.std()*np.sqrt(252)
            if len(ds) > 0 and ds.std() > 0 else 0
        )
        m['max_drawdown'] = (
            self.drawdown_curve.max() if len(self.drawdown_curve) else 0
        )
        m['max_drawdown_pct'] = m['max_drawdown'] * 100
        m['calmar_ratio'] = (
            m['cagr']/m['max_drawdown'] if m['max_drawdown']>0 else 0
        )
        m['annual_volatility'] = (
            dr.std()*np.sqrt(252) if len(dr)>1 else 0
        )
        m['total_trades'] = len(self.trades)
        if self.trades:
            pnls = [t.pnl for t in self.trades]
            pcts = [t.pnl_pct for t in self.trades]
            wins = [t for t in self.trades if t.pnl > 0]
            losses = [t for t in self.trades if t.pnl <= 0]
            m['winning_trades'] = len(wins)
            m['losing_trades'] = len(losses)
            m['win_rate'] = len(wins)/len(self.trades)
            m['total_pnl'] = sum(pnls)
            m['avg_pnl'] = np.mean(pnls)
            m['avg_pnl_pct'] = np.mean(pcts)*100
            m['avg_win'] = np.mean([t.pnl for t in wins]) if wins else 0
            m['avg_loss'] = np.mean([t.pnl for t in losses]) if losses else 0
            m['largest_win'] = max(pnls)
            m['largest_loss'] = min(pnls)
            wsum = sum(t.pnl for t in wins)
            lsum = abs(sum(t.pnl for t in losses))
            m['profit_factor'] = wsum/lsum if lsum > 0 else float('inf')
            m['avg_bars_held'] = np.mean([t.bars_held for t in self.trades])
            m['total_commission'] = sum(t.commission for t in self.trades)
            streaks = [1 if t.pnl>0 else -1 for t in self.trades]
            mw = ml = c = 0
            for s in streaks:
                if s > 0:
                    c = max(0,c)+1; mw = max(mw,c)
                else:
                    c = min(0,c)-1; ml = max(ml,-c)
            m['max_win_streak'] = mw
            m['max_loss_streak'] = ml
            longs = [t for t in self.trades if t.side=='LONG']
            shorts = [t for t in self.trades if t.side=='SHORT']
            m['long_trades'] = len(longs)
            m['short_trades'] = len(shorts)
            m['long_pnl'] = sum(t.pnl for t in longs)
            m['short_pnl'] = sum(t.pnl for t in shorts)
        else:
            for k in ['winning_trades','losing_trades','win_rate',
                       'total_pnl','avg_pnl','avg_pnl_pct',
                       'avg_win','avg_loss','largest_win',
                       'largest_loss','profit_factor',
                       'avg_bars_held','total_commission',
                       'max_win_streak','max_loss_streak',
                       'long_trades','short_trades',
                       'long_pnl','short_pnl']:
                m[k] = 0
        return m

    def summary(self):
        m = self.metrics
        return f"""
{'═'*55}
  📊 BACKTEST RESULTS: {self.strategy_name}
{'═'*55}
  Period:           {self.data.index[0].strftime('%Y-%m-%d')} → {self.data.index[-1].strftime('%Y-%m-%d')}
  Initial Capital:  ${m['initial_capital']:>12,.2f}
  Final Equity:     ${m['final_equity']:>12,.2f}
  Total Return:      {m['total_return_pct']:>11.2f}%
  CAGR:              {m['cagr']*100:>11.2f}%
{'─'*55}
  Sharpe Ratio:      {m['sharpe_ratio']:>11.2f}
  Sortino Ratio:     {m['sortino_ratio']:>11.2f}
  Calmar Ratio:      {m['calmar_ratio']:>11.2f}
  Max Drawdown:      {m['max_drawdown_pct']:>11.2f}%
  Annual Vol:        {m['annual_volatility']*100:>11.2f}%
{'─'*55}
  Total Trades:      {m['total_trades']:>11d}
  Win Rate:          {m['win_rate']*100:>11.2f}%
  Profit Factor:     {m['profit_factor']:>11.2f}
  Avg Trade PnL:    ${m['avg_pnl']:>12,.2f}
  Avg Win:          ${m['avg_win']:>12,.2f}
  Avg Loss:         ${m['avg_loss']:>12,.2f}
  Largest Win:      ${m['largest_win']:>12,.2f}
  Largest Loss:     ${m['largest_loss']:>12,.2f}
{'─'*55}
  Long:  {m['long_trades']:>3d} trades | PnL: ${m['long_pnl']:>10,.2f}
  Short: {m['short_trades']:>3d} trades | PnL: ${m['short_pnl']:>10,.2f}
  Commission:       ${m['total_commission']:>12,.2f}
{'═'*55}"""

    def trades_df(self):
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([{
            'Side': t.side, 'Entry': t.entry_date,
            'Exit': t.exit_date,
            'Entry$': round(t.entry_price,2),
            'Exit$': round(t.exit_price,2),
            'Qty': round(t.quantity,2),
            'PnL': round(t.pnl,2),
            'PnL%': round(t.pnl_pct*100,2),
            'Bars': t.bars_held, 'Tag': t.tag
        } for t in self.trades])

    def monthly_returns(self):
        dr = self.equity_curve.pct_change().dropna()
        mr = dr.resample('ME').apply(lambda x: (1+x).prod()-1)
        df = pd.DataFrame({
            'Year': mr.index.year, 'Month': mr.index.month,
            'Return': mr.values
        })
        p = df.pivot_table('Return','Year','Month',aggfunc='sum')
        months = ['Jan','Feb','Mar','Apr','May','Jun',
                  'Jul','Aug','Sep','Oct','Nov','Dec']
        p.columns = months[:len(p.columns)]
        return p * 100

    def plot(self, figsize=(16, 12)):
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.equity_curve, color='#2196F3', lw=1.5)
        ax1.axhline(self.config.initial_capital,
                     color='gray', ls='--', alpha=0.5)
        ax1.set_title(f'Equity — {self.strategy_name}',
                      fontweight='bold')
        ax1.set_ylabel('Equity ($)')
        ax1.grid(True, alpha=0.3)
        ax1.fill_between(
            self.equity_curve.index,
            self.config.initial_capital,
            self.equity_curve.values,
            where=self.equity_curve.values>=self.config.initial_capital,
            alpha=0.15, color='green'
        )
        ax1.fill_between(
            self.equity_curve.index,
            self.config.initial_capital,
            self.equity_curve.values,
            where=self.equity_curve.values<self.config.initial_capital,
            alpha=0.15, color='red'
        )

        ax2 = fig.add_subplot(gs[1, :])
        ax2.fill_between(
            self.drawdown_curve.index, 0,
            -self.drawdown_curve.values*100,
            color='#f44336', alpha=0.4
        )
        ax2.set_title('Drawdown', fontweight='bold')
        ax2.set_ylabel('DD (%)')
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[2, 0])
        if self.trades:
            pnls = [t.pnl for t in self.trades]
            colors = ['green' if p>0 else 'red' for p in pnls]
            ax3.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
            ax3.set_title('Trade PnL', fontweight='bold')
            ax3.axhline(0, color='black', lw=0.5)
            ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[2, 1])
        if self.trades:
            pp = [t.pnl_pct*100 for t in self.trades]
            ax4.hist(pp, bins=30, color='#2196F3', alpha=0.7,
                     edgecolor='white')
            ax4.axvline(np.mean(pp), color='red', ls='--',
                        label=f'Mean: {np.mean(pp):.2f}%')
            ax4.set_title('PnL Distribution', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(gs[3, :])
        try:
            mr = self.monthly_returns()
            if len(mr) > 0:
                im = ax5.imshow(mr.values, cmap='RdYlGn',
                                aspect='auto')
                ax5.set_xticks(range(len(mr.columns)))
                ax5.set_xticklabels(mr.columns)
                ax5.set_yticks(range(len(mr.index)))
                ax5.set_yticklabels(mr.index)
                for i in range(len(mr.index)):
                    for j in range(len(mr.columns)):
                        v = mr.values[i,j]
                        if not np.isnan(v):
                            ax5.text(j,i,f'{v:.1f}%',
                                     ha='center',va='center',
                                     fontsize=8)
                ax5.set_title('Monthly Returns (%)',
                              fontweight='bold')
                plt.colorbar(im, ax=ax5, shrink=0.8)
        except:
            pass

        plt.suptitle(f'📊 {self.strategy_name}',
                     fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(f'{self.strategy_name}_report.png',
                    dpi=150, bbox_inches='tight')
        plt.show()
        return fig

    def __repr__(self):
        return self.summary()


# ══════════════════════════════════════════════════════════════
# WALK-FORWARD ANALYSIS — Issue #8
# ══════════════════════════════════════════════════════════════
class WalkForwardAnalysis:
    def __init__(self, config=WalkForwardConfig()):
        self.config = config
        self.results = []

    def run(self, strategy_class, data, param_grid,
            bt_config=BacktestConfig(), metric='sharpe_ratio',
            progress=True):
        n = len(data)
        ns = self.config.n_splits
        self.results = []

        if self.config.anchored:
            ss = n // (ns + 1)
            splits = [(0, ss*(i+2), ss*(i+2),
                       min(ss*(i+3), n)) for i in range(ns)]
        else:
            fs = n // ns
            iss = int(fs * self.config.in_sample_pct)
            oss = fs - iss
            splits = []
            for i in range(ns):
                s = i * fs
                splits.append((s, s+iss, s+iss, min(s+fs, n)))

        pnames = list(param_grid.keys())
        combos = list(product(*param_grid.values()))

        for fi, (is_s, is_e, os_s, os_e) in enumerate(splits):
            if os_e <= os_s or os_e > n:
                continue
            if progress:
                print(f"\n📊 Fold {fi+1}/{len(splits)}")

            is_data = data.iloc[is_s:is_e].copy()
            os_data = data.iloc[os_s:os_e].copy()

            best_val = -np.inf
            best_params = {}
            for combo in combos:
                params = dict(zip(pnames, combo))
                try:
                    s = strategy_class()
                    for k, v in params.items():
                        setattr(s, k, v)
                    r = BacktestEngine(bt_config).run(
                        s, is_data, progress=False
                    )
                    val = r.metrics.get(metric, -np.inf)
                    if val > best_val:
                        best_val = val
                        best_params = params.copy()
                except:
                    continue

            s = strategy_class()
            for k, v in best_params.items():
                setattr(s, k, v)
            osr = BacktestEngine(bt_config).run(
                s, os_data, progress=False
            )
            self.results.append({
                'fold': fi+1, 'best_params': best_params,
                'is_metric': best_val,
                'oos_metric': osr.metrics.get(metric, 0),
                'oos_return': osr.metrics['total_return'],
                'oos_sharpe': osr.metrics['sharpe_ratio'],
                'oos_max_dd': osr.metrics['max_drawdown'],
                'oos_trades': osr.metrics['total_trades'],
            })
            if progress:
                print(f"   Best: {best_params} | "
                      f"OOS ret={osr.metrics['total_return']*100:.2f}%")

        summary = {
            'folds': self.results,
            'avg_oos_return': np.mean(
                [r['oos_return'] for r in self.results]
            ),
            'avg_oos_sharpe': np.mean(
                [r['oos_sharpe'] for r in self.results]
            ),
            'avg_oos_maxdd': np.mean(
                [r['oos_max_dd'] for r in self.results]
            ),
            'oos_consistency': sum(
                1 for r in self.results if r['oos_return'] > 0
            ) / max(len(self.results), 1),
            'is_oos_correlation': (
                np.corrcoef(
                    [r['is_metric'] for r in self.results],
                    [r['oos_metric'] for r in self.results]
                )[0,1] if len(self.results) > 1 else 0
            ),
        }
        if progress:
            print(f"\n{'═'*50}")
            print(f"  Avg OOS Return: "
                  f"{summary['avg_oos_return']*100:.2f}%")
            print(f"  Consistency: "
                  f"{summary['oos_consistency']*100:.0f}%")
            print(f"{'═'*50}")
        return summary


# ══════════════════════════════════════════════════════════════
# MONTE CARLO — Issue #9
# ══════════════════════════════════════════════════════════════
class MonteCarloSimulation:
    def __init__(self, config=MonteCarloConfig()):
        self.config = config
        self.sim_curves = None
        self.sim_maxdd = None
        self.sim_final = None

    def run(self, result, progress=True):
        if not result.trades:
            print("❌ No trades")
            return {}
        rets = np.array([t.pnl for t in result.trades])
        nt = len(rets)
        ns = self.config.n_simulations
        ini = result.config.initial_capital
        bs = min(self.config.block_size, nt // 2)
        if bs < 1:
            bs = 1

        self.sim_curves = np.zeros((ns, nt + 1))
        self.sim_curves[:, 0] = ini
        self.sim_maxdd = np.zeros(ns)
        self.sim_final = np.zeros(ns)

        if progress:
            print(f"🎲 {ns} simulations ({nt} trades)...")

        for sim in range(ns):
            sr = []
            while len(sr) < nt:
                si = np.random.randint(0, max(1, nt - bs))
                sr.extend(rets[si:si + bs])
            sr = np.array(sr[:nt])
            eq = ini
            peak = ini
            mdd = 0
            for j, r in enumerate(sr):
                eq += r
                self.sim_curves[sim, j + 1] = eq
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak if peak > 0 else 0
                mdd = max(mdd, dd)
            self.sim_maxdd[sim] = mdd
            self.sim_final[sim] = eq

        s = {
            'n_simulations': ns, 'n_trades': nt,
            'initial_capital': ini,
            'original_final': result.metrics['final_equity'],
            'median_final': np.median(self.sim_final),
            'mean_final': np.mean(self.sim_final),
            'prob_profit': np.mean(self.sim_final > ini),
            'prob_ruin': np.mean(self.sim_final < ini * 0.5),
        }
        for cl in self.config.confidence_levels:
            lo = (1 - cl) / 2 * 100
            hi = (1 + cl) / 2 * 100
            ci = int(cl * 100)
            s[f'final_ci_{ci}_low'] = np.percentile(
                self.sim_final, lo
            )
            s[f'final_ci_{ci}_high'] = np.percentile(
                self.sim_final, hi
            )
            s[f'maxdd_ci_{ci}'] = np.percentile(
                self.sim_maxdd, hi
            )

        if progress:
            print(f"  Median: ${s['median_final']:,.0f}")
            print(f"  P(Profit): {s['prob_profit']*100:.1f}%")
            print(f"  P(Ruin): {s['prob_ruin']*100:.1f}%")

        return s

    def plot(self, n_show=200):
        if self.sim_curves is None:
            return
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        np_ = min(n_show, len(self.sim_curves))
        for i in range(np_):
            axes[0,0].plot(self.sim_curves[i], alpha=0.05,
                           color='blue', lw=0.5)
        med = np.median(self.sim_curves, axis=0)
        axes[0,0].plot(med, color='red', lw=2, label='Median')
        axes[0,0].set_title('Simulated Equity')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        axes[0,1].hist(self.sim_final, bins=50, color='#2196F3',
                       alpha=0.7, edgecolor='white')
        axes[0,1].set_title('Final Equity Distribution')
        axes[0,1].grid(True, alpha=0.3)

        axes[1,0].hist(self.sim_maxdd*100, bins=50,
                       color='#f44336', alpha=0.7, edgecolor='white')
        axes[1,0].set_title('Max Drawdown Distribution')
        axes[1,0].grid(True, alpha=0.3)

        se = np.sort(self.sim_final)
        cp = np.arange(1, len(se)+1) / len(se)
        axes[1,1].plot(se, cp, color='#2196F3')
        axes[1,1].set_title('CDF Final Equity')
        axes[1,1].grid(True, alpha=0.3)

        plt.suptitle('🎲 Monte Carlo', fontweight='bold')
        plt.tight_layout()
        plt.savefig('monte_carlo.png', dpi=150, bbox_inches='tight')
        plt.show()


# ══════════════════════════════════════════════════════════════
# HTML REPORT — Issue #13
# ══════════════════════════════════════════════════════════════
class HTMLReportGenerator:
    @staticmethod
    def _fig_to_b64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120,
                    bbox_inches='tight', facecolor='white')
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return b64

    @staticmethod
    def generate(result, mc_results=None, wf_results=None,
                 filepath='backtest_report.html'):
        m = result.metrics

        # Equity chart
        fig1, (a1, a2) = plt.subplots(2, 1, figsize=(14, 7),
                                       height_ratios=[3, 1])
        a1.plot(result.equity_curve, color='#1976D2', lw=1.5)
        a1.axhline(result.config.initial_capital,
                    color='gray', ls='--', alpha=0.5)
        a1.set_title(f'Equity — {result.strategy_name}',
                     fontweight='bold')
        a1.grid(True, alpha=0.3)
        a2.fill_between(result.drawdown_curve.index, 0,
                        -result.drawdown_curve.values*100,
                        color='#EF5350', alpha=0.5)
        a2.set_ylabel('DD (%)')
        a2.grid(True, alpha=0.3)
        plt.tight_layout()
        c1 = HTMLReportGenerator._fig_to_b64(fig1)

        # Trades chart
        fig2, ax = plt.subplots(figsize=(14, 4))
        if result.trades:
            pnls = [t.pnl for t in result.trades]
            colors = ['#4CAF50' if p>0 else '#EF5350' for p in pnls]
            ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.8)
            ax.axhline(0, color='black', lw=0.5)
        ax.set_title('Trade PnL ($)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        c2 = HTMLReportGenerator._fig_to_b64(fig2)

        trades_html = ''
        if result.trades:
            trades_html = result.trades_df().to_html(
                index=False, classes='trades-table',
                float_format=lambda x: f'{x:.2f}'
            )

        pos_neg = 'positive' if m['total_return'] >= 0 else 'negative'

        html = f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Report — {result.strategy_name}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',sans-serif;background:#f5f5f5;
      max-width:1200px;margin:0 auto;padding:20px}}
.header{{background:linear-gradient(135deg,#1976D2,#1565C0);
         color:white;padding:30px;border-radius:10px;
         margin-bottom:20px;text-align:center}}
.section{{background:white;border-radius:10px;padding:25px;
          margin-bottom:20px;box-shadow:0 2px 10px rgba(0,0,0,0.08)}}
.section h2{{color:#1976D2;margin-bottom:15px;
             border-bottom:2px solid #e3f2fd;padding-bottom:8px}}
.mg{{display:grid;grid-template-columns:repeat(4,1fr);
     gap:15px;margin-bottom:20px}}
.mc{{background:#f8f9fa;border-radius:8px;padding:15px;
     text-align:center;border-left:4px solid #1976D2}}
.mc .l{{font-size:12px;color:#666;text-transform:uppercase}}
.mc .v{{font-size:22px;font-weight:bold;margin-top:5px}}
.positive{{color:#4CAF50}} .negative{{color:#EF5350}}
.mt{{width:100%;border-collapse:collapse}}
.mt td{{padding:8px 15px;border-bottom:1px solid #eee}}
.mt td:first-child{{font-weight:600;color:#555;width:50%}}
.trades-table{{width:100%;border-collapse:collapse;font-size:13px}}
.trades-table th{{background:#1976D2;color:white;padding:10px}}
.trades-table td{{padding:8px 10px;border-bottom:1px solid #eee}}
img{{max-width:100%;border-radius:5px}}
</style></head><body>
<div class="header"><h1>📊 {result.strategy_name}</h1>
<p>{result.data.index[0].strftime('%Y-%m-%d')} →
{result.data.index[-1].strftime('%Y-%m-%d')} |
{len(result.data)} bars</p></div>
<div class="section"><div class="mg">
<div class="mc"><div class="l">Return</div>
<div class="v {pos_neg}">{m['total_return_pct']:.2f}%</div></div>
<div class="mc"><div class="l">Sharpe</div>
<div class="v">{m['sharpe_ratio']:.2f}</div></div>
<div class="mc"><div class="l">Max DD</div>
<div class="v negative">-{m['max_drawdown_pct']:.2f}%</div></div>
<div class="mc"><div class="l">Win Rate</div>
<div class="v">{m['win_rate']*100:.1f}%</div></div>
</div></div>
<div class="section"><h2>📈 Equity</h2>
<img src="data:image/png;base64,{c1}"></div>
<div class="section"><h2>📊 Metrics</h2>
<table class="mt">
<tr><td>Initial</td><td>${m['initial_capital']:,.2f}</td></tr>
<tr><td>Final</td><td>${m['final_equity']:,.2f}</td></tr>
<tr><td>CAGR</td><td>{m['cagr']*100:.2f}%</td></tr>
<tr><td>Trades</td><td>{m['total_trades']}</td></tr>
<tr><td>Profit Factor</td><td>{m['profit_factor']:.2f}</td></tr>
<tr><td>Avg Win/Loss</td>
<td>${m['avg_win']:,.2f} / ${m['avg_loss']:,.2f}</td></tr>
</table></div>
<div class="section"><h2>💹 Trade PnL</h2>
<img src="data:image/png;base64,{c2}"></div>
<div class="section"><h2>📋 Trades ({m['total_trades']})</h2>
<div style="overflow-x:auto">{trades_html}</div></div>
</body></html>'''

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"✅ HTML Report: {filepath}")
        return filepath