"""
═══════════════════════════════════════════════════════════
charts.py — Data, Charts, Calculations, Symbol Database
═══════════════════════════════════════════════════════════
"""

import os, io, json, time, hashlib, importlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html

from engine import (
    BacktestConfig, SizingMethod, BacktestEngine,
    HTMLReportGenerator
)

# ═════════════════════════════════════════════════════
# GLOBAL STATE
# ═════════════════════════════════════════════════════
RESULTS = {}
DATA_CACHE = {}
RUN_LOG = []
ALERTS = []
SESSION_DIR = './sessions'
os.makedirs(SESSION_DIR, exist_ok=True)
os.makedirs('./exports', exist_ok=True)

# ═════════════════════════════════════════════════════
# THEMES
# ═════════════════════════════════════════════════════
DARK = {
    'bg':'#131722','panel':'#1e222d','border':'#2a2e39',
    'text':'#d1d4dc','dim':'#787b86','accent':'#2962ff',
    'up':'#26a69a','dn':'#ef5350',
    'vu':'rgba(38,166,154,0.5)','vd':'rgba(239,83,80,0.5)',
    'buy':'#2196F3','sell':'#FF9800',
    'eq':'#2962ff','eqf':'rgba(41,98,255,0.08)',
    'bal':'#2196F3','bal_f':'rgba(33,150,243,0.06)',
    'sma':['#f7a21b','#ff6d00','#2962ff','#e91e63','#00bcd4'],
    'bbl':'rgba(33,150,243,0.3)','bbf':'rgba(33,150,243,0.05)',
    'grid':'#1e222d','input_bg':'#2a2e39',
    'overlay':['#2196F3','#FF9800','#e91e63','#26a69a','#9c27b0','#00bcd4','#f9a825','#8bc34a'],
    'template':'plotly_dark','card_shadow':'0 2px 8px rgba(0,0,0,0.3)',
    'tab_bg':'#1e222d','tab_text':'#787b86',
    'row_alt':'rgba(255,255,255,0.02)','paper':'#131722',
    'crosshair':'#9598a1','rsi':'#7e57c2',
}
LIGHT = {
    'bg':'#f0f2f5','panel':'#ffffff','border':'#dee2e6',
    'text':'#212529','dim':'#6c757d','accent':'#0d6efd',
    'up':'#198754','dn':'#dc3545',
    'vu':'rgba(25,135,84,0.4)','vd':'rgba(220,53,69,0.4)',
    'buy':'#0d6efd','sell':'#fd7e14',
    'eq':'#198754','eqf':'rgba(25,135,84,0.06)',
    'bal':'#0d6efd','bal_f':'rgba(13,110,253,0.06)',
    'sma':['#e67e22','#ff6d00','#0d6efd','#e91e63','#0dcaf0'],
    'bbl':'rgba(13,110,253,0.35)','bbf':'rgba(13,110,253,0.05)',
    'grid':'#e9ecef','input_bg':'#ffffff',
    'overlay':['#0d6efd','#fd7e14','#e91e63','#198754','#6f42c1','#0dcaf0','#e67e22','#20c997'],
    'template':'plotly_white','card_shadow':'0 2px 8px rgba(0,0,0,0.06)',
    'tab_bg':'#e9ecef','tab_text':'#6c757d',
    'row_alt':'rgba(0,0,0,0.02)','paper':'#ffffff',
    'crosshair':'#9e9e9e','rsi':'#7e57c2',
}
CURRENT_THEME = {'mode': 'light'}
def T():
    return DARK if CURRENT_THEME['mode'] == 'dark' else LIGHT


# ═════════════════════════════════════════════════════
# SYMBOL DATABASE + BROKERS
# ═════════════════════════════════════════════════════
BROKERS = {
    'OANDA':{'icon':'🔵','color':'#1a8cff','badge':'✓'},
    'FXCM':{'icon':'🔴','color':'#e74c3c','badge':'Ⓩ'},
    'FOREX.COM':{'icon':'🟢','color':'#27ae60','badge':'Ⓖ'},
    'IG':{'icon':'🔴','color':'#dc3545','badge':'IG'},
    'FP Markets':{'icon':'🔵','color':'#0d6efd','badge':'✓'},
    'Pepperstone':{'icon':'🟢','color':'#198754','badge':'P'},
    'IC Markets':{'icon':'🔵','color':'#0dcaf0','badge':'IC'},
    'NASDAQ':{'icon':'🔵','color':'#0096d6','badge':'Q'},
    'NYSE':{'icon':'🟡','color':'#f0ad4e','badge':'N'},
    'LSE':{'icon':'🔵','color':'#003399','badge':'L'},
    'XETRA':{'icon':'🟡','color':'#ffc107','badge':'X'},
    'TSE':{'icon':'🔴','color':'#dc3545','badge':'T'},
    'HKEX':{'icon':'🟢','color':'#198754','badge':'HK'},
    'NSE':{'icon':'🔵','color':'#0d6efd','badge':'IN'},
    'ASX':{'icon':'🟡','color':'#ffc107','badge':'AU'},
    'TSX':{'icon':'🔴','color':'#dc3545','badge':'CA'},
    'KRX':{'icon':'🔵','color':'#0d6efd','badge':'KR'},
    'BINANCE':{'icon':'🟡','color':'#f0b90b','badge':'B'},
    'COINBASE':{'icon':'🔵','color':'#0052ff','badge':'CB'},
    'KRAKEN':{'icon':'🟣','color':'#7b2ff7','badge':'K'},
    'BYBIT':{'icon':'🟡','color':'#f7a600','badge':'BB'},
    'TVC':{'icon':'🟢','color':'#198754','badge':'TV'},
    'COMEX':{'icon':'🟡','color':'#ffc107','badge':'CX'},
    'NYMEX':{'icon':'🔴','color':'#dc3545','badge':'NM'},
    'CME':{'icon':'🔵','color':'#0d6efd','badge':'CM'},
    'CBOE':{'icon':'🟢','color':'#198754','badge':'VO'},
}

FOREX = {
    'EURUSD':{'name':'Euro / US Dollar','flag':'🇪🇺','brokers':['OANDA','FXCM','FOREX.COM','IG','FP Markets'],'featured':True},
    'GBPUSD':{'name':'British Pound / US Dollar','flag':'🇬🇧','brokers':['OANDA','FXCM','FOREX.COM','IG','FP Markets'],'featured':True},
    'USDJPY':{'name':'US Dollar / Japanese Yen','flag':'🇯🇵','brokers':['OANDA','FXCM','FOREX.COM','IG'],'featured':True},
    'USDCHF':{'name':'US Dollar / Swiss Franc','flag':'🇨🇭','brokers':['OANDA','FXCM','FOREX.COM'],'featured':False},
    'AUDUSD':{'name':'Australian Dollar / US Dollar','flag':'🇦🇺','brokers':['OANDA','FXCM','FOREX.COM','Pepperstone','IC Markets'],'featured':True},
    'USDCAD':{'name':'US Dollar / Canadian Dollar','flag':'🇨🇦','brokers':['OANDA','FXCM','FOREX.COM'],'featured':False},
    'NZDUSD':{'name':'New Zealand Dollar / US Dollar','flag':'🇳🇿','brokers':['OANDA','FXCM'],'featured':False},
    'EURGBP':{'name':'Euro / British Pound','flag':'🇪🇺','brokers':['OANDA','FXCM','IG'],'featured':False},
    'EURJPY':{'name':'Euro / Japanese Yen','flag':'🇪🇺','brokers':['OANDA','FXCM','IG'],'featured':False},
    'GBPJPY':{'name':'British Pound / Japanese Yen','flag':'🇬🇧','brokers':['OANDA','FXCM','IG','FP Markets'],'featured':True},
    'EURCHF':{'name':'Euro / Swiss Franc','flag':'🇪🇺','brokers':['OANDA','FXCM'],'featured':False},
    'EURAUD':{'name':'Euro / Australian Dollar','flag':'🇪🇺','brokers':['OANDA','FXCM'],'featured':False},
    'GBPAUD':{'name':'British Pound / Australian Dollar','flag':'🇬🇧','brokers':['OANDA','FXCM'],'featured':False},
    'AUDJPY':{'name':'Australian Dollar / Japanese Yen','flag':'🇦🇺','brokers':['OANDA','FXCM'],'featured':False},
    'AUDNZD':{'name':'Australian Dollar / NZ Dollar','flag':'🇦🇺','brokers':['OANDA','FXCM'],'featured':False},
    'CADJPY':{'name':'Canadian Dollar / Japanese Yen','flag':'🇨🇦','brokers':['OANDA','FXCM'],'featured':False},
    'USDTRY':{'name':'US Dollar / Turkish Lira','flag':'🇹🇷','brokers':['OANDA','FXCM'],'featured':False},
    'USDMXN':{'name':'US Dollar / Mexican Peso','flag':'🇲🇽','brokers':['OANDA','FXCM'],'featured':False},
    'USDZAR':{'name':'US Dollar / South African Rand','flag':'🇿🇦','brokers':['OANDA','FXCM'],'featured':False},
    'USDSGD':{'name':'US Dollar / Singapore Dollar','flag':'🇸🇬','brokers':['OANDA'],'featured':False},
    'USDCNH':{'name':'US Dollar / Chinese Yuan','flag':'🇨🇳','brokers':['OANDA','FXCM'],'featured':False},
}

STOCKS = {
    'AAPL':{'name':'Apple Inc.','flag':'🇺🇸','brokers':['NASDAQ'],'featured':True},
    'MSFT':{'name':'Microsoft Corp.','flag':'🇺🇸','brokers':['NASDAQ'],'featured':True},
    'GOOGL':{'name':'Alphabet (Google)','flag':'🇺🇸','brokers':['NASDAQ'],'featured':True},
    'AMZN':{'name':'Amazon.com','flag':'🇺🇸','brokers':['NASDAQ'],'featured':True},
    'NVDA':{'name':'NVIDIA Corp.','flag':'🇺🇸','brokers':['NASDAQ'],'featured':True},
    'META':{'name':'Meta Platforms','flag':'🇺🇸','brokers':['NASDAQ'],'featured':True},
    'TSLA':{'name':'Tesla Inc.','flag':'🇺🇸','brokers':['NASDAQ'],'featured':True},
    'JPM':{'name':'JPMorgan Chase','flag':'🇺🇸','brokers':['NYSE'],'featured':False},
    'V':{'name':'Visa Inc.','flag':'🇺🇸','brokers':['NYSE'],'featured':False},
    'NFLX':{'name':'Netflix Inc.','flag':'🇺🇸','brokers':['NASDAQ'],'featured':False},
    'AMD':{'name':'AMD Inc.','flag':'🇺🇸','brokers':['NASDAQ'],'featured':False},
    'BA':{'name':'Boeing Co.','flag':'🇺🇸','brokers':['NYSE'],'featured':False},
    'DIS':{'name':'Walt Disney','flag':'🇺🇸','brokers':['NYSE'],'featured':False},
    'COIN':{'name':'Coinbase Global','flag':'🇺🇸','brokers':['NASDAQ'],'featured':False},
    'PLTR':{'name':'Palantir','flag':'🇺🇸','brokers':['NYSE'],'featured':False},
    'VOD.L':{'name':'Vodafone Group','flag':'🇬🇧','brokers':['LSE'],'featured':False},
    'BP.L':{'name':'BP plc','flag':'🇬🇧','brokers':['LSE'],'featured':False},
    'SHEL.L':{'name':'Shell plc','flag':'🇬🇧','brokers':['LSE'],'featured':False},
    'AZN.L':{'name':'AstraZeneca','flag':'🇬🇧','brokers':['LSE'],'featured':False},
    'SAP.DE':{'name':'SAP SE','flag':'🇩🇪','brokers':['XETRA'],'featured':False},
    'BMW.DE':{'name':'BMW AG','flag':'🇩🇪','brokers':['XETRA'],'featured':False},
    '7203.T':{'name':'Toyota Motor','flag':'🇯🇵','brokers':['TSE'],'featured':False},
    '6758.T':{'name':'Sony Group','flag':'🇯🇵','brokers':['TSE'],'featured':False},
    '0700.HK':{'name':'Tencent Holdings','flag':'🇭🇰','brokers':['HKEX'],'featured':False},
    '9988.HK':{'name':'Alibaba Group','flag':'🇭🇰','brokers':['HKEX'],'featured':False},
    'RELIANCE.NS':{'name':'Reliance Industries','flag':'🇮🇳','brokers':['NSE'],'featured':False},
    'TCS.NS':{'name':'Tata Consultancy','flag':'🇮🇳','brokers':['NSE'],'featured':False},
    'BHP.AX':{'name':'BHP Group','flag':'🇦🇺','brokers':['ASX'],'featured':False},
    'SHOP.TO':{'name':'Shopify','flag':'🇨🇦','brokers':['TSX'],'featured':False},
    '005930.KS':{'name':'Samsung Electronics','flag':'🇰🇷','brokers':['KRX'],'featured':False},
}

CRYPTO = {
    'BTC':{'name':'Bitcoin','flag':'₿','brokers':['BINANCE','COINBASE','KRAKEN','BYBIT'],'featured':True},
    'ETH':{'name':'Ethereum','flag':'Ξ','brokers':['BINANCE','COINBASE','KRAKEN','BYBIT'],'featured':True},
    'SOL':{'name':'Solana','flag':'◎','brokers':['BINANCE','COINBASE','KRAKEN'],'featured':True},
    'XRP':{'name':'Ripple','flag':'✕','brokers':['BINANCE','KRAKEN','BYBIT'],'featured':True},
    'ADA':{'name':'Cardano','flag':'₳','brokers':['BINANCE','COINBASE'],'featured':False},
    'DOGE':{'name':'Dogecoin','flag':'Ð','brokers':['BINANCE','COINBASE'],'featured':False},
    'AVAX':{'name':'Avalanche','flag':'▲','brokers':['BINANCE','COINBASE'],'featured':False},
    'LINK':{'name':'Chainlink','flag':'⬡','brokers':['BINANCE','COINBASE'],'featured':False},
    'DOT':{'name':'Polkadot','flag':'●','brokers':['BINANCE','COINBASE'],'featured':False},
    'LTC':{'name':'Litecoin','flag':'Ł','brokers':['BINANCE','COINBASE','KRAKEN'],'featured':False},
    'UNI':{'name':'Uniswap','flag':'🦄','brokers':['BINANCE','COINBASE'],'featured':False},
    'SHIB':{'name':'Shiba Inu','flag':'🐕','brokers':['BINANCE','COINBASE'],'featured':False},
}

INDICES = {
    '^GSPC':{'name':'S&P 500','flag':'🇺🇸','brokers':['TVC','OANDA'],'featured':True},
    '^DJI':{'name':'Dow Jones Industrial','flag':'🇺🇸','brokers':['TVC','OANDA'],'featured':True},
    '^IXIC':{'name':'NASDAQ Composite','flag':'🇺🇸','brokers':['TVC'],'featured':True},
    '^VIX':{'name':'VIX Volatility Index','flag':'🇺🇸','brokers':['CBOE'],'featured':True},
    '^FTSE':{'name':'FTSE 100','flag':'🇬🇧','brokers':['TVC'],'featured':False},
    '^GDAXI':{'name':'DAX 40','flag':'🇩🇪','brokers':['TVC'],'featured':False},
    '^N225':{'name':'Nikkei 225','flag':'🇯🇵','brokers':['TVC'],'featured':False},
    '^HSI':{'name':'Hang Seng','flag':'🇭🇰','brokers':['TVC'],'featured':False},
}

FUTURES = {
    'GC=F':{'name':'Gold Futures','flag':'🥇','brokers':['COMEX','OANDA'],'featured':True},
    'SI=F':{'name':'Silver Futures','flag':'🥈','brokers':['COMEX'],'featured':False},
    'CL=F':{'name':'Crude Oil WTI','flag':'🛢️','brokers':['NYMEX','OANDA'],'featured':True},
    'BZ=F':{'name':'Brent Crude Oil','flag':'🛢️','brokers':['NYMEX'],'featured':False},
    'NG=F':{'name':'Natural Gas','flag':'🔥','brokers':['NYMEX'],'featured':False},
    'HG=F':{'name':'Copper Futures','flag':'🔶','brokers':['COMEX'],'featured':False},
    'ZC=F':{'name':'Corn Futures','flag':'🌽','brokers':['CME'],'featured':False},
    'ZW=F':{'name':'Wheat Futures','flag':'🌾','brokers':['CME'],'featured':False},
    'KC=F':{'name':'Coffee Futures','flag':'☕','brokers':['CME'],'featured':False},
}

ETFS = {
    'SPY':{'name':'SPDR S&P 500 ETF','flag':'🇺🇸','brokers':['NYSE'],'featured':True},
    'QQQ':{'name':'Invesco QQQ','flag':'🇺🇸','brokers':['NASDAQ'],'featured':True},
    'IWM':{'name':'iShares Russell 2000','flag':'🇺🇸','brokers':['NYSE'],'featured':False},
    'GLD':{'name':'SPDR Gold Shares','flag':'🥇','brokers':['NYSE'],'featured':True},
    'TLT':{'name':'iShares 20+ Treasury','flag':'🇺🇸','brokers':['NASDAQ'],'featured':False},
    'XLK':{'name':'Technology SPDR','flag':'🇺🇸','brokers':['NYSE'],'featured':False},
    'ARKK':{'name':'ARK Innovation','flag':'🇺🇸','brokers':['NYSE'],'featured':False},
    'VOO':{'name':'Vanguard S&P 500','flag':'🇺🇸','brokers':['NYSE'],'featured':False},
    'EEM':{'name':'iShares Emerging Mkt','flag':'🌍','brokers':['NYSE'],'featured':False},
}

ASSET_TYPES = ['all','stocks','funds','futures','forex','crypto','indices']
TYPE_LABELS = {'all':'All','stocks':'Stocks','funds':'Funds/ETFs',
               'futures':'Futures','forex':'Forex','crypto':'Crypto','indices':'Indices'}


def build_symbol_db():
    db = []
    type_map = {'forex':FOREX,'stocks':STOCKS,'crypto':CRYPTO,
                'indices':INDICES,'futures':FUTURES,'funds':ETFS}
    yf_map = {'forex':lambda t:f"{t}=X",'crypto':lambda t:f"{t}-USD",
              'stocks':lambda t:t,'indices':lambda t:t,'futures':lambda t:t,'funds':lambda t:t}
    for atype, symbols in type_map.items():
        for ticker, info in symbols.items():
            for broker in info['brokers']:
                db.append({'ticker':ticker,'name':info['name'],'type':atype,
                    'flag':info.get('flag',''),'broker':broker,
                    'featured':info.get('featured',False),
                    'yf_symbol':yf_map.get(atype,lambda t:t)(ticker),
                    'key':f"{ticker}_{broker}"})
    return db

SYMBOL_DB = build_symbol_db()
ALL_BROKERS = sorted(set(s['broker'] for s in SYMBOL_DB))

# TradingView Modal colors (always dark)
M = {
    'overlay_bg':'rgba(0,0,0,0.65)','modal_bg':'#1e222d',
    'modal_border':'#363a45','search_bg':'#2a2e39',
    'search_text':'#d1d4dc','tab_inactive':'#787b86',
    'tab_active_bg':'#2962ff','tab_active_text':'#ffffff',
    'ticker_color':'#d1d4dc','name_color':'#787b86',
    'featured_bg':'#2962ff','featured_text':'#ffffff',
    'broker_color':'#787b86','close_color':'#787b86',
}


# ═════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════
def clean_date(d, default):
    if not d or d == '': return default
    s = str(d)[:10]
    try: datetime.strptime(s, '%Y-%m-%d'); return s
    except: return default

def preset_to_dates(preset):
    end = datetime.now().strftime('%Y-%m-%d')
    m = {'7d':7,'30d':30,'90d':90,'1y':365,'2y':730,'5y':1825}
    if preset in m: return (datetime.now()-timedelta(days=m[preset])).strftime('%Y-%m-%d'), end
    if preset == 'all': return '2000-01-01', end
    return None, None

def search_symbols(query='', asset_type='all', broker='all'):
    q = query.strip().upper(); results = []
    for s in SYMBOL_DB:
        if asset_type != 'all' and s['type'] != asset_type: continue
        if broker != 'all' and s['broker'] != broker: continue
        if q and q not in s['ticker'].upper() and q not in s['name'].upper(): continue
        results.append(s)
    results.sort(key=lambda x: (not x['featured'], x['ticker']))
    return results[:50]

def card(label, value, color):
    t = T()
    return html.Div(style={'backgroundColor':t['panel'],'borderRadius':'8px','padding':'10px 8px',
        'textAlign':'center','borderTop':f'3px solid {color}','boxShadow':t['card_shadow']},
        children=[html.Div(label,style={'color':t['dim'],'fontSize':'9px','textTransform':'uppercase','letterSpacing':'0.5px'}),
        html.Div(value,style={'color':color,'fontSize':'15px','fontWeight':'bold','marginTop':'3px'})])


# ═════════════════════════════════════════════════════
# DATA FETCHING
# ═════════════════════════════════════════════════════
def fetch_data(symbol, tf, start='2018-01-01', end=None):
    start = clean_date(start, '2018-01-01')
    end = clean_date(end, datetime.now().strftime('%Y-%m-%d'))
    if start > end: start, end = end, start
    ck = f"{symbol}_{tf}_{start}_{end}"
    if ck in DATA_CACHE: return DATA_CACHE[ck], 'cache'

    try:
        import yfinance as yf
        if tf in ('1m','5m','15m','30m'):
            start = (datetime.now()-timedelta(days=7)).strftime('%Y-%m-%d')
        elif tf in ('1h','60m'):
            start = (datetime.now()-timedelta(days=729)).strftime('%Y-%m-%d')
        df = yf.Ticker(symbol).history(start=start,end=end,interval=tf,auto_adjust=True,actions=False)
        if df is not None and len(df) > 5:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            cm = {}
            for c in df.columns:
                cl = str(c).lower()
                if 'open' in cl: cm[c]='Open'
                elif 'high' in cl: cm[c]='High'
                elif 'low' in cl: cm[c]='Low'
                elif 'close' in cl: cm[c]='Close'
                elif 'vol' in cl: cm[c]='Volume'
            df = df.rename(columns=cm)
            for n in ['Open','High','Low','Close']:
                if n not in df.columns: raise ValueError(f"Missing {n}")
            if 'Volume' not in df.columns: df['Volume'] = 0
            df = df[['Open','High','Low','Close','Volume']].dropna().sort_index()
            if len(df) > 5: DATA_CACHE[ck] = df; return df, 'yfinance'
    except: pass

    if tf in ('1wk','1mo'):
        try:
            daily, _ = fetch_data(symbol, '1d', start, end)
            rule = 'W' if tf=='1wk' else 'ME'
            df = daily.resample(rule).agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
            if len(df)>5: DATA_CACHE[ck]=df; return df, 'resample'
        except: pass

    # Demo
    np.random.seed(abs(hash(symbol))%2**31)
    dates = pd.bdate_range(start, end)
    if tf=='1wk': dates=pd.date_range(start,end,freq='W')
    elif tf=='1mo': dates=pd.date_range(start,end,freq='ME')
    price = 1.0 if '=X' in symbol else (30000 if '-USD' in symbol else 100+abs(hash(symbol))%200)
    vol = 0.008 if '=X' in symbol else (0.035 if '-USD' in symbol else 0.018)
    prices = []; p = price
    for _ in dates: p *= np.exp(np.random.normal(0.0003,vol)); prices.append(p)
    pa = np.array(prices)
    df = pd.DataFrame({'Open':pa*(1+np.random.normal(0,0.003,len(pa))),'High':pa*(1+abs(np.random.normal(0,0.01,len(pa)))),
        'Low':pa*(1-abs(np.random.normal(0,0.01,len(pa)))),'Close':pa,'Volume':np.random.randint(100000,10000000,len(pa))},index=dates)
    DATA_CACHE[ck]=df; return df, 'demo'


# ═════════════════════════════════════════════════════
# CALCULATIONS
# ═════════════════════════════════════════════════════
def calc_benchmark(result):
    try:
        bm_df,_=fetch_data('^GSPC','1d',str(result.data.index[0].date()),str(result.data.index[-1].date()))
        if bm_df is None or len(bm_df)<10: return None
        eq=result.equity_curve; bm=bm_df['Close'].reindex(eq.index,method='ffill').dropna()
        ea=eq.reindex(bm.index,method='ffill').dropna(); c=ea.index.intersection(bm.index)
        if len(c)<10: return None
        ea=ea.loc[c]; bm=bm.loc[c]; sr=ea.pct_change().dropna(); br=bm.pct_change().dropna()
        c2=sr.index.intersection(br.index)
        if len(c2)<10: return None
        sr=sr.loc[c2]; br=br.loc[c2]; bt=(bm.iloc[-1]/bm.iloc[0]-1)*100
        cv=np.cov(sr.values,br.values); beta=cv[0,1]/cv[1,1] if cv[1,1]!=0 else 0
        alpha=(sr.mean()-beta*br.mean())*252*100; corr=sr.corr(br)
        ex=sr-br; ir=ex.mean()/ex.std()*np.sqrt(252) if ex.std()>0 else 0
        return {'benchmark_return':bt,'beta':beta,'alpha':alpha,'correlation':corr,'information_ratio':ir}
    except: return None

def calc_portfolio():
    if len(RESULTS)<2: return None
    try:
        rd = {k:r.equity_curve.pct_change().dropna() for k,r in RESULTS.items() if len(r.equity_curve.pct_change().dropna())>5}
        if len(rd)<2: return None
        df=pd.DataFrame(rd).dropna()
        if len(df)<10: return None
        corr=df.corr(); pr=df.mean(axis=1); pt=(1+pr).prod()-1
        ps=pr.mean()/pr.std()*np.sqrt(252) if pr.std()>0 else 0
        pe=(1+pr).cumprod(); pp=pe.expanding().max(); pdd=((pe-pp)/pp).min()
        iv=df.std()*np.sqrt(252); w=np.ones(len(df.columns))/len(df.columns)
        pv=pr.std()*np.sqrt(252); dr=np.dot(w,iv)/pv if pv>0 else 1
        return {'corr':corr,'ret':pt*100,'sharpe':ps,'maxdd':abs(pdd)*100,'div':dr,'n':len(df.columns)}
    except: return None

def calc_mt5_metrics(result):
    m=result.metrics; trades=result.trades; eq=result.equity_curve; ini=result.config.initial_capital; mt={}
    mt['bars_in_test']=len(result.data); mt['initial_deposit']=ini
    mt['total_net_profit']=m.get('total_pnl',0)
    mt['gross_profit']=sum(t.pnl for t in trades if t.pnl>0) if trades else 0
    mt['gross_loss']=sum(t.pnl for t in trades if t.pnl<=0) if trades else 0
    mt['profit_factor']=m.get('profit_factor',0); mt['expected_payoff']=m.get('avg_pnl',0)
    mt['sharpe_ratio']=m.get('sharpe_ratio',0); mt['sortino_ratio']=m.get('sortino_ratio',0)
    mt['cagr']=m.get('cagr',0)*100
    mda=m.get('max_drawdown',0)*ini; mt['recovery_factor']=mt['total_net_profit']/mda if mda>0 else 0
    if len(eq)>1:
        mt['balance_dd_absolute']=max(0,ini-eq.min()); pk=eq.expanding().max(); dd=pk-eq
        mt['balance_dd_maximal']=dd.max(); mt['balance_dd_maximal_pct']=m.get('max_drawdown_pct',0)
        dp=(pk-eq)/pk*100; mt['balance_dd_relative_pct']=dp.max()
    else:
        mt['balance_dd_absolute']=mt['balance_dd_maximal']=mt['balance_dd_maximal_pct']=mt['balance_dd_relative_pct']=0
    mt['total_trades']=len(trades)
    longs=[t for t in trades if t.side=='LONG']; shorts=[t for t in trades if t.side=='SHORT']
    mt['long_trades']=len(longs); mt['short_trades']=len(shorts)
    mt['long_won_pct']=len([t for t in longs if t.pnl>0])/len(longs)*100 if longs else 0
    mt['short_won_pct']=len([t for t in shorts if t.pnl>0])/len(shorts)*100 if shorts else 0
    wins=[t for t in trades if t.pnl>0]; losses=[t for t in trades if t.pnl<=0]
    mt['profit_trades']=len(wins); mt['loss_trades']=len(losses)
    mt['profit_trades_pct']=len(wins)/len(trades)*100 if trades else 0
    mt['largest_profit_trade']=max(t.pnl for t in trades) if trades else 0
    mt['largest_loss_trade']=min(t.pnl for t in trades) if trades else 0
    mt['avg_profit_trade']=np.mean([t.pnl for t in wins]) if wins else 0
    mt['avg_loss_trade']=np.mean([t.pnl for t in losses]) if losses else 0
    mt['avg_trade']=np.mean([t.pnl for t in trades]) if trades else 0
    mt['avg_bars_held']=np.mean([t.bars_held for t in trades]) if trades else 0
    mt['payoff_ratio']=abs(mt['avg_profit_trade']/mt['avg_loss_trade']) if mt['avg_loss_trade']!=0 else 0
    mt['total_commission']=sum(t.commission for t in trades) if trades else 0
    mt['annual_volatility']=m.get('annual_volatility',0)*100; mt['calmar_ratio']=m.get('calmar_ratio',0)
    # Consecutive
    if trades:
        mw=ml=cw=cl=0; mwp=mlp=cwp=clp=0; aws=[]; als=[]
        for t in trades:
            if t.pnl>0:
                cw+=1; cwp+=t.pnl
                if cl>0: als.append(cl); ml=max(ml,cl); mlp=min(mlp,clp); cl=0; clp=0
            else:
                cl+=1; clp+=t.pnl
                if cw>0: aws.append(cw); mw=max(mw,cw); mwp=max(mwp,cwp); cw=0; cwp=0
        if cw>0: aws.append(cw); mw=max(mw,cw); mwp=max(mwp,cwp)
        if cl>0: als.append(cl); ml=max(ml,cl); mlp=min(mlp,clp)
        mt['max_consecutive_wins']=mw; mt['max_consecutive_losses']=ml
        mt['max_consecutive_profit']=mwp; mt['max_consecutive_loss']=mlp
        mt['avg_consecutive_wins']=np.mean(aws) if aws else 0
        mt['avg_consecutive_losses']=np.mean(als) if als else 0
    else:
        for k in ['max_consecutive_wins','max_consecutive_losses','max_consecutive_profit',
                   'max_consecutive_loss','avg_consecutive_wins','avg_consecutive_losses']: mt[k]=0
    if len(result.data)>0:
        mt['start_date']=result.data.index[0].strftime('%Y-%m-%d')
        mt['end_date']=result.data.index[-1].strftime('%Y-%m-%d')
        mt['period_days']=(result.data.index[-1]-result.data.index[0]).days
    else: mt['start_date']=mt['end_date']=''; mt['period_days']=0
    return mt


# ═════════════════════════════════════════════════════
# EXPORT / SESSION / ALERTS
# ═════════════════════════════════════════════════════
def gen_csv(result,key):
    lines=[f"# {key}",""]
    for k,v in result.metrics.items():
        if k=='daily_returns': continue
        lines.append(f"# {k}: {v}")
    lines.append(""); lines.append("Trade#,Side,Entry,Exit,In,Out,Qty,PnL,PnL%,Bars,Comm,Tag")
    for i,t in enumerate(result.trades,1):
        lines.append(f"{i},{t.side},{str(t.entry_date)[:10]},{str(t.exit_date)[:10]},{t.entry_price:.4f},{t.exit_price:.4f},{t.quantity:.4f},{t.pnl:.2f},{t.pnl_pct*100:.2f},{t.bars_held},{t.commission:.2f},{t.tag}")
    return '\n'.join(lines)

def gen_excel(result,key):
    try:
        buf=io.BytesIO()
        with pd.ExcelWriter(buf,engine='openpyxl') as w:
            pd.DataFrame([{k:v for k,v in result.metrics.items() if k!='daily_returns'}]).to_excel(w,'Metrics',index=False)
            tdf=result.trades_df() if hasattr(result,'trades_df') else pd.DataFrame()
            if len(tdf)>0: tdf.to_excel(w,'Trades',index=False)
            result.equity_curve.to_frame('Equity').to_excel(w,'Equity')
        buf.seek(0); return buf.getvalue()
    except: return None

def gen_html_report(result,key):
    try:
        fp=f'./exports/report_{key.replace(" ","_").replace("•","")}.html'
        HTMLReportGenerator.generate(result,filepath=fp); return fp
    except: return None

def save_session(name):
    if not RESULTS: return False
    data={k:{'metrics':{mk:(float(mv) if isinstance(mv,(int,float,np.floating,np.integer)) else str(mv))
              for mk,mv in r.metrics.items() if mk!='daily_returns'}} for k,r in RESULTS.items()}
    with open(os.path.join(SESSION_DIR,f"{name}.json"),'w') as f:
        json.dump({'name':name,'saved_at':datetime.now().isoformat(),'results':data},f,indent=2,default=str)
    return True

def check_alerts(result,key):
    m=result.metrics; a=[]
    if m['max_drawdown_pct']>20: a.append(f"⚠️ {key}: DD {m['max_drawdown_pct']:.1f}%!")
    if m['sharpe_ratio']<0: a.append(f"⚠️ {key}: Sharpe {m['sharpe_ratio']:.2f}")
    if m['profit_factor']>3: a.append(f"🏆 {key}: PF {m['profit_factor']:.2f}!")
    if m['total_return_pct']>50: a.append(f"🏆 {key}: Return {m['total_return_pct']:.1f}%!")
    ALERTS.extend(a); return a


# ═════════════════════════════════════════════════════
# BACKTEST RUNNER
# ═════════════════════════════════════════════════════
def run_backtest(yf_symbol, tf, config, start, end, display_label=None):
    key = f"{display_label or yf_symbol} • {tf}"
    df, source = fetch_data(yf_symbol, tf, start, end)
    import my_strategy; importlib.reload(my_strategy)
    strat = my_strategy.MyStrategy()
    engine = BacktestEngine(config)
    result = engine.run(strat, df, progress=False)
    RESULTS[key] = result
    m = result.metrics
    RUN_LOG.append(f"{'🟢' if m['total_return']>=0 else '🔴'} {display_label or yf_symbol} {tf}: "
                   f"{m['total_return_pct']:+.1f}% | SR:{m['sharpe_ratio']:.2f} | T:{m['total_trades']} [{source}]")
    check_alerts(result, key)
    return result, key, source

def run_optimization(yf_sym, tf, param_name, param_values, config, start, end):
    df,_=fetch_data(yf_sym,tf,start,end); rl=[]
    for val in param_values:
        try:
            import my_strategy; importlib.reload(my_strategy)
            s=my_strategy.MyStrategy(); setattr(s,param_name,val)
            r=BacktestEngine(config).run(s,df,progress=False); m=r.metrics
            rl.append({'value':val,'return':m['total_return_pct'],'sharpe':m['sharpe_ratio'],
                       'maxdd':m['max_drawdown_pct'],'trades':m['total_trades'],'winrate':m['win_rate']*100,'pf':m['profit_factor']})
        except: rl.append({'value':val,'return':0,'sharpe':0,'maxdd':0,'trades':0,'winrate':0,'pf':0})
    return rl


# ═════════════════════════════════════════════════════
# CHART BUILDERS
# ═════════════════════════════════════════════════════
def empty_fig(msg="Select symbol → Run"):
    t=T(); fig=go.Figure()
    fig.update_layout(template=t['template'],paper_bgcolor=t['paper'],plot_bgcolor=t['paper'],height=300,
        annotations=[dict(text=msg,x=0.5,y=0.5,xref='paper',yref='paper',showarrow=False,font=dict(size=14,color=t['dim']))])
    return fig

def build_tv_chart(result, ctype='candlestick', show_t=True, show_v=True, show_i=True):
    t=T(); d=result.data; trades=result.trades if show_t else []
    has_rsi='RSI' in d.columns and show_i; has_macd=('MACD' in d.columns or 'MACD_Hist' in d.columns) and show_i
    has_v=show_v and 'Volume' in d.columns and d['Volume'].sum()>0
    panels,hts=['price'],[0.52]
    if has_v: panels.append('vol'); hts.append(0.08)
    if has_rsi: panels.append('rsi'); hts.append(0.15)
    if has_macd: panels.append('macd'); hts.append(0.15)
    n=len(panels); s=sum(hts); hts=[h/s for h in hts]; rm={x:i+1 for i,x in enumerate(panels)}
    fig=make_subplots(rows=n,cols=1,shared_xaxes=True,vertical_spacing=0.012,row_heights=hts)
    pr=rm['price']

    # Price
    if ctype=='candlestick':
        fig.add_trace(go.Candlestick(x=d.index,open=d['Open'],high=d['High'],low=d['Low'],close=d['Close'],
            increasing=dict(line=dict(color=t['up'],width=1),fillcolor=t['up']),
            decreasing=dict(line=dict(color=t['dn'],width=1),fillcolor='rgba(239,83,80,0.4)'),
            showlegend=False),row=pr,col=1)
    elif ctype=='ohlc':
        fig.add_trace(go.Ohlc(x=d.index,open=d['Open'],high=d['High'],low=d['Low'],close=d['Close'],
            increasing_line_color=t['up'],decreasing_line_color=t['dn'],showlegend=False),row=pr,col=1)
    elif ctype=='heikin':
        hc=(d['Open']+d['High']+d['Low']+d['Close'])/4; ho=d['Open'].copy()
        for i in range(1,len(d)): ho.iloc[i]=(ho.iloc[i-1]+hc.iloc[i-1])/2
        hh=pd.concat([d['High'],ho,hc],axis=1).max(axis=1); hl=pd.concat([d['Low'],ho,hc],axis=1).min(axis=1)
        fig.add_trace(go.Candlestick(x=d.index,open=ho,high=hh,low=hl,close=hc,
            increasing=dict(line=dict(color=t['up']),fillcolor=t['up']),
            decreasing=dict(line=dict(color=t['dn']),fillcolor='rgba(239,83,80,0.4)'),showlegend=False),row=pr,col=1)
    elif ctype=='area':
        fig.add_trace(go.Scatter(x=d.index,y=d['Close'],mode='lines',line=dict(color=t['up'],width=1.5),
            fill='tozeroy',fillcolor='rgba(38,166,154,0.08)',showlegend=False),row=pr,col=1)
    else:
        fig.add_trace(go.Scatter(x=d.index,y=d['Close'],mode='lines',line=dict(color=t['up'],width=1.5),showlegend=False),row=pr,col=1)

    # Indicators overlay
    if show_i:
        skip={'Open','High','Low','Close','Volume','ATR','RSI','MACD','MACD_Signal','MACD_Sig','MACD_Hist','BB_Upper','BB_Lower','BB_Mid','ST_Dir','Stoch_K','Stoch_D'}
        ci=0
        for col in d.columns:
            if col in skip: continue
            vals=d[col].dropna()
            if len(vals)<20 or d['Close'].mean()==0: continue
            if 0.3<vals.mean()/d['Close'].mean()<3.0:
                fig.add_trace(go.Scatter(x=d.index,y=d[col],mode='lines',name=col,
                    line=dict(color=t['sma'][ci%len(t['sma'])],width=1.3),opacity=0.85),row=pr,col=1); ci+=1
        if 'BB_Upper' in d.columns and 'BB_Lower' in d.columns:
            fig.add_trace(go.Scatter(x=d.index,y=d['BB_Upper'],mode='lines',name='BB',line=dict(color=t['bbl'],width=1,dash='dot')),row=pr,col=1)
            fig.add_trace(go.Scatter(x=d.index,y=d['BB_Lower'],mode='lines',line=dict(color=t['bbl'],width=1,dash='dot'),fill='tonexty',fillcolor=t['bbf'],showlegend=False),row=pr,col=1)

    # Trade markers (batched for speed)
    if trades:
        ex,ey,ec,es=[],[],[],[]; xx,xy,xc=[],[],[]
        lx,ly=[],[]
        for tr in trades:
            win=tr.pnl>0; il=tr.side=='LONG'
            ex.append(tr.entry_date); ey.append(tr.entry_price*(0.993 if il else 1.007))
            ec.append(t['buy'] if il else t['sell']); es.append('triangle-up' if il else 'triangle-down')
            xx.append(tr.exit_date); xy.append(tr.exit_price); xc.append(t['up'] if win else t['dn'])
            lx.extend([tr.entry_date,tr.exit_date,None]); ly.extend([tr.entry_price,tr.exit_price,None])
        fig.add_trace(go.Scatter(x=lx,y=ly,mode='lines',line=dict(color=t['dim'],width=0.8,dash='dot'),showlegend=False,hoverinfo='skip',opacity=0.3),row=pr,col=1)
        fig.add_trace(go.Scatter(x=ex,y=ey,mode='markers',marker=dict(symbol=es,size=11,color=ec,line=dict(color='white',width=1)),showlegend=False,name='Entry'),row=pr,col=1)
        fig.add_trace(go.Scatter(x=xx,y=xy,mode='markers',marker=dict(symbol='x',size=8,color=xc,line=dict(width=2)),showlegend=False,name='Exit'),row=pr,col=1)

    if has_v:
        vr=rm['vol']; vc=[t['vu'] if c>=o else t['vd'] for c,o in zip(d['Close'],d['Open'])]
        fig.add_trace(go.Bar(x=d.index,y=d['Volume'],marker_color=vc,showlegend=False,name='Volume'),row=vr,col=1)
    if has_rsi:
        rr=rm['rsi']
        fig.add_trace(go.Scatter(x=d.index,y=d['RSI'],mode='lines',name='RSI',line=dict(color=t['rsi'],width=1.5)),row=rr,col=1)
        fig.add_hrect(y0=70,y1=100,fillcolor='rgba(239,83,80,0.08)',line_width=0,row=rr,col=1)
        fig.add_hrect(y0=0,y1=30,fillcolor='rgba(38,166,154,0.08)',line_width=0,row=rr,col=1)
        for lv in [30,50,70]: fig.add_hline(y=lv,line_dash='dot',line_color=t['dim'],opacity=0.3,row=rr,col=1)
        fig.update_yaxes(range=[0,100],row=rr,col=1)
    if has_macd:
        mr=rm['macd']
        if 'MACD' in d.columns: fig.add_trace(go.Scatter(x=d.index,y=d['MACD'],mode='lines',name='MACD',line=dict(color='#2196F3',width=1.5)),row=mr,col=1)
        sc=next((c for c in ['MACD_Signal','MACD_Sig'] if c in d.columns),None)
        if sc: fig.add_trace(go.Scatter(x=d.index,y=d[sc],mode='lines',name='Signal',line=dict(color='#ff6d00',width=1.3)),row=mr,col=1)
        if 'MACD_Hist' in d.columns:
            h=d['MACD_Hist']; hc=[t['up'] if v>=0 else t['dn'] for v in h]
            fig.add_trace(go.Bar(x=d.index,y=h,marker_color=hc,showlegend=False,opacity=0.6),row=mr,col=1)
        fig.add_hline(y=0,line_color=t['dim'],opacity=0.3,row=mr,col=1)

    # Layout
    fig.update_layout(template=t['template'],paper_bgcolor=t['paper'],plot_bgcolor=t['paper'],
        font=dict(family="'Trebuchet MS','Segoe UI',sans-serif",color=t['text'],size=11),
        height=680,margin=dict(l=0,r=0,t=0,b=0),hovermode='x unified',spikedistance=-1,
        hoverlabel=dict(bgcolor=t['panel'],bordercolor=t['border'],font=dict(color=t['text'],size=11)),
        legend=dict(orientation='h',y=1.0,x=0.005,xanchor='left',yanchor='bottom',bgcolor='rgba(0,0,0,0)',font=dict(size=10,color=t['dim'])),
        xaxis=dict(rangeslider=dict(visible=False),rangeselector=dict(buttons=[
            dict(count=1,label='1M',step='month',stepmode='backward'),dict(count=3,label='3M',step='month',stepmode='backward'),
            dict(count=6,label='6M',step='month',stepmode='backward'),dict(count=1,label='1Y',step='year',stepmode='backward'),
            dict(count=2,label='2Y',step='year',stepmode='backward'),dict(label='ALL',step='all')],
            bgcolor=t['panel'],activecolor=t['accent'],bordercolor=t['border'],font=dict(color=t['text'],size=10),x=0,y=1.0,xanchor='left',yanchor='bottom')))

    for i in range(1,n+1):
        fig.update_xaxes(gridcolor=t['grid'],gridwidth=1,zeroline=False,showline=False,rangebreaks=[dict(bounds=["sat","mon"])],
            showspikes=True,spikemode='across',spikesnap='cursor',spikethickness=0.5,spikecolor=t['crosshair'],spikedash='solid',row=i,col=1)
        fig.update_yaxes(gridcolor=t['grid'],gridwidth=1,zeroline=False,showline=False,side='right',
            showspikes=True,spikemode='across',spikesnap='cursor',spikethickness=0.5,spikecolor=t['crosshair'],spikedash='solid',row=i,col=1)
    for i in range(1,n): fig.update_xaxes(showticklabels=False,row=i,col=1)
    return fig


def build_mt5_graph(result):
    t=T(); eq=result.equity_curve; dd=result.drawdown_curve; ini=result.config.initial_capital
    bal=pd.Series(dtype=float); bv=ini
    for tr in result.trades: bv+=tr.pnl; bal[tr.exit_date]=bv
    bal=bal.reindex(eq.index,method='ffill').fillna(ini) if len(bal)>0 else pd.Series(ini,index=eq.index)
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=0.03,row_heights=[0.55,0.25,0.20],
        subplot_titles=['<b>Balance / Equity</b>','<b>Drawdown %</b>','<b>Trade PnL ($)</b>'])
    fig.add_trace(go.Scatter(x=bal.index,y=bal.values,mode='lines',name='Balance',line=dict(color=t['bal'],width=2.5),fill='tozeroy',fillcolor=t['bal_f']),row=1,col=1)
    fig.add_trace(go.Scatter(x=eq.index,y=eq.values,mode='lines',name='Equity',line=dict(color=t['up'],width=1.5)),row=1,col=1)
    fig.add_hline(y=ini,line_dash='dash',line_color=t['dim'],opacity=0.4,row=1,col=1)
    fig.add_trace(go.Scatter(x=dd.index,y=-dd.values*100,mode='lines',name='DD',line=dict(color=t['dn'],width=1.5),fill='tozeroy',fillcolor='rgba(220,53,69,0.15)'),row=2,col=1)
    if result.trades:
        dates=[tr.exit_date for tr in result.trades]; pnls=[tr.pnl for tr in result.trades]
        fig.add_trace(go.Bar(x=dates,y=pnls,marker_color=[t['up'] if p>0 else t['dn'] for p in pnls],showlegend=False,opacity=0.8),row=3,col=1)
    fig.update_layout(template=t['template'],paper_bgcolor=t['paper'],plot_bgcolor=t['paper'],font=dict(color=t['text'],size=11),
        height=600,margin=dict(l=10,r=10,t=30,b=10),hovermode='x unified',legend=dict(orientation='h',y=1.02,x=0.5,xanchor='center'))
    for i in range(1,4):
        fig.update_xaxes(gridcolor=t['grid'],zeroline=False,rangebreaks=[dict(bounds=["sat","mon"])],row=i,col=1)
        fig.update_yaxes(gridcolor=t['grid'],zeroline=False,side='right',row=i,col=1)
    return fig

def build_pnl(r):
    t=T()
    if not r.trades: return empty_fig("")
    pnls=[tr.pnl for tr in r.trades]; cum=np.cumsum(pnls)
    fig=make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(x=list(range(1,len(pnls)+1)),y=pnls,marker_color=[t['up'] if p>0 else t['dn'] for p in pnls],opacity=0.8),secondary_y=False)
    fig.add_trace(go.Scatter(x=list(range(1,len(pnls)+1)),y=cum,mode='lines',line=dict(color=t['eq'],width=2.5)),secondary_y=True)
    fig.update_layout(template=t['template'],paper_bgcolor=t['paper'],plot_bgcolor=t['paper'],font=dict(color=t['text']),height=280,margin=dict(l=10,r=10,t=35,b=10),showlegend=False,title=dict(text='💹 Trade PnL',font=dict(size=13)))
    return fig

def build_dist(r):
    t=T()
    if not r.trades: return empty_fig("")
    pcts=[tr.pnl_pct*100 for tr in r.trades]
    fig=go.Figure()
    fig.add_trace(go.Histogram(x=pcts,nbinsx=30,marker_color=t['accent'],opacity=0.7,marker_line=dict(color='white',width=1)))
    fig.add_vline(x=np.mean(pcts),line_color='#e67e22',line_dash='dash',line_width=2)
    fig.update_layout(template=t['template'],paper_bgcolor=t['paper'],plot_bgcolor=t['paper'],font=dict(color=t['text']),height=280,margin=dict(l=10,r=10,t=35,b=10),title=dict(text='📊 Distribution %',font=dict(size=13)))
    return fig

def build_monthly_heatmap(result):
    t=T()
    try:
        eq=result.equity_curve; dr=eq.pct_change().dropna()
        monthly=dr.resample('ME').apply(lambda x:(1+x).prod()-1)*100
        df=pd.DataFrame({'Year':monthly.index.year,'Month':monthly.index.month,'Return':monthly.values})
        pivot=df.pivot_table(values='Return',index='Year',columns='Month',aggfunc='sum')
        ml=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        pivot.columns=[ml[m-1] for m in pivot.columns]
        pivot['YTD']=pivot.sum(axis=1)
        text=[[f'{v:+.1f}%' if not np.isnan(v) else '' for v in row] for row in pivot.values]
        ma=max(abs(pivot.min().min()),abs(pivot.max().max()),1)
        cs=[[0,'#b71c1c'],[0.25,'#ef5350'],[0.45,'#ffcdd2'],[0.5,t['paper']],[0.55,'#c8e6c9'],[0.75,'#26a69a'],[1.0,'#1b5e20']]
        fig=go.Figure(data=go.Heatmap(z=pivot.values,x=list(pivot.columns),y=[str(y) for y in pivot.index],
            colorscale=cs,zmid=0,zmin=-ma,zmax=ma,text=text,texttemplate='%{text}',textfont=dict(size=10,color=t['text']),
            xgap=2,ygap=2,colorbar=dict(title='%',len=0.8)))
        fig.update_layout(template=t['template'],paper_bgcolor=t['paper'],plot_bgcolor=t['paper'],font=dict(color=t['text'],size=11),
            height=max(200,len(pivot)*35+80),margin=dict(l=10,r=10,t=40,b=10),title=dict(text='📅 Monthly Returns (%)',font=dict(size=14),x=0.01),
            xaxis=dict(side='top'),yaxis=dict(autorange='reversed'))
        return fig
    except: return empty_fig("")

def build_drawdown_duration(result):
    t=T()
    try:
        eq=result.equity_curve; peak=eq.expanding().max(); dd_pct=(eq-peak)/peak*100
        in_dd=dd_pct<-0.01; periods=[]; si=None
        for i,(date,is_dd) in enumerate(zip(dd_pct.index,in_dd)):
            if is_dd and si is None: si=i
            elif not is_dd and si is not None:
                dur=(date-dd_pct.index[si]).days; mdd=dd_pct.iloc[si:i].min()
                periods.append({'start':dd_pct.index[si],'end':date,'duration':dur,'max_dd':mdd}); si=None
        if si is not None:
            dur=(dd_pct.index[-1]-dd_pct.index[si]).days; mdd=dd_pct.iloc[si:].min()
            periods.append({'start':dd_pct.index[si],'end':dd_pct.index[-1],'duration':dur,'max_dd':mdd})

        fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.05,row_heights=[0.6,0.4],
            subplot_titles=['<b>Drawdown %</b>','<b>Drawdown Duration (days)</b>'])
        fig.add_trace(go.Scatter(x=dd_pct.index,y=dd_pct.values,mode='lines',line=dict(color=t['dn'],width=1),
            fill='tozeroy',fillcolor='rgba(239,83,80,0.2)',name='DD%'),row=1,col=1)
        mi=dd_pct.idxmin(); mv=dd_pct.min()
        fig.add_trace(go.Scatter(x=[mi],y=[mv],mode='markers+text',marker=dict(size=10,color=t['dn'],symbol='diamond'),
            text=[f' Max: {mv:.1f}%'],textposition='middle right',textfont=dict(size=10,color=t['dn']),showlegend=False),row=1,col=1)
        if periods:
            for p in periods:
                bc=t['dn'] if p['duration']>60 else '#ff9800' if p['duration']>30 else t['dim']
                fig.add_trace(go.Bar(x=[p['start']],y=[p['duration']],width=86400000*max(p['duration'],3),marker_color=bc,opacity=0.7,showlegend=False),row=2,col=1)
            avg=np.mean([p['duration'] for p in periods])
            fig.add_hline(y=avg,line_dash='dash',line_color='#f7a21b',opacity=0.5,row=2,col=1,
                annotation_text=f'Avg: {avg:.0f}d',annotation_font_size=10,annotation_font_color='#f7a21b')
        fig.update_layout(template=t['template'],paper_bgcolor=t['paper'],plot_bgcolor=t['paper'],font=dict(color=t['text'],size=11),
            height=450,margin=dict(l=10,r=10,t=50,b=10),showlegend=False,hovermode='x unified')
        for i in range(1,3):
            fig.update_xaxes(gridcolor=t['grid'],zeroline=False,rangebreaks=[dict(bounds=["sat","mon"])],row=i,col=1)
            fig.update_yaxes(gridcolor=t['grid'],zeroline=False,side='right',row=i,col=1)
        return fig
    except: return empty_fig("")

def build_equity_overlay():
    t=T()
    if not RESULTS: return empty_fig("Run 2+ backtests")
    fig=go.Figure()
    for i,(k,r) in enumerate(RESULTS.items()):
        eq_n=r.equity_curve/r.equity_curve.iloc[0]*100
        fig.add_trace(go.Scatter(x=eq_n.index,y=eq_n.values,mode='lines',name=k,line=dict(color=t['overlay'][i%len(t['overlay'])],width=2.5)))
    fig.update_layout(template=t['template'],paper_bgcolor=t['paper'],plot_bgcolor=t['paper'],font=dict(color=t['text']),height=340,
        margin=dict(l=10,r=10,t=40,b=10),title=dict(text='📈 Equity Overlay',font=dict(size=13)),
        legend=dict(orientation='h',y=-0.15,x=0.5,xanchor='center'),hovermode='x unified')
    return fig

def build_corr():
    t=T(); pm=calc_portfolio()
    if pm is None: return empty_fig("Need 2+")
    corr=pm['corr']; labels=[k.replace(' • ','\n') for k in corr.columns]
    fig=go.Figure(data=go.Heatmap(z=corr.values,x=labels,y=labels,zmin=-1,zmax=1,
        colorscale=[[0,t['dn']],[0.5,t['paper']],[1,t['up']]],
        text=[[f"{v:.2f}" for v in row] for row in corr.values],texttemplate='%{text}',textfont=dict(size=12,color=t['text'])))
    fig.update_layout(template=t['template'],paper_bgcolor=t['paper'],plot_bgcolor=t['paper'],font=dict(color=t['text']),
        height=320,margin=dict(l=10,r=10,t=40,b=10),title=dict(text='🔗 Correlation',font=dict(size=13)))
    return fig

def build_opt_chart(opt_results,param_name):
    t=T()
    if not opt_results: return empty_fig("No results")
    vals=[r['value'] for r in opt_results]
    fig=make_subplots(rows=2,cols=2,subplot_titles=['Return%','Sharpe','MaxDD%','WinRate%'])
    fig.add_trace(go.Bar(x=[str(v) for v in vals],y=[r['return'] for r in opt_results],
        marker_color=[t['up'] if r['return']>0 else t['dn'] for r in opt_results],showlegend=False),row=1,col=1)
    fig.add_trace(go.Bar(x=[str(v) for v in vals],y=[r['sharpe'] for r in opt_results],marker_color=t['accent'],showlegend=False),row=1,col=2)
    fig.add_trace(go.Bar(x=[str(v) for v in vals],y=[r['maxdd'] for r in opt_results],marker_color=t['dn'],showlegend=False),row=2,col=1)
    fig.add_trace(go.Bar(x=[str(v) for v in vals],y=[r['winrate'] for r in opt_results],marker_color=t['up'],showlegend=False),row=2,col=2)
    fig.update_layout(template=t['template'],paper_bgcolor=t['paper'],plot_bgcolor=t['paper'],font=dict(color=t['text']),
        height=420,margin=dict(l=10,r=10,t=45,b=10),title=dict(text=f'🔧 {param_name}',font=dict(size=14)))
    return fig


# ═════════════════════════════════════════════════════
# MT5 REPORT + RESULTS HTML BUILDERS
# ═════════════════════════════════════════════════════
def build_mt5_report(result):
    t=T(); mt=calc_mt5_metrics(result)
    def row(l,v,c=None,b=False):
        c=c or t['text']; fw='bold' if b else 'normal'
        return html.Tr([html.Td(l,style={'padding':'6px 12px','color':t['dim'],'fontSize':'12px','borderBottom':f"1px solid {t['border']}","width":"55%"}),
            html.Td(v,style={'padding':'6px 12px','color':c,'fontWeight':fw,'fontSize':'12px','borderBottom':f"1px solid {t['border']}","textAlign":"right"})])
    def sec(title): return html.Tr([html.Td(title,colSpan=2,style={'padding':'10px 12px','color':t['accent'],'fontSize':'13px','fontWeight':'bold','backgroundColor':t['row_alt'],'borderBottom':f"2px solid {t['accent']}"})])
    pc=t['up'] if mt['total_net_profit']>=0 else t['dn']
    rows=[sec('📊 General'),row('Bars',f"{mt['bars_in_test']:,}"),row('Period',f"{mt.get('start_date','')} → {mt.get('end_date','')}"),
        row('Initial',f"${mt['initial_deposit']:,.2f}"),row('Commission',f"${mt['total_commission']:,.2f}",t['dn']),
        sec('💰 Profit'),row('Net Profit',f"${mt['total_net_profit']:+,.2f}",pc,True),
        row('Gross Profit',f"${mt['gross_profit']:+,.2f}",t['up']),row('Gross Loss',f"${mt['gross_loss']:+,.2f}",t['dn']),
        row('Profit Factor',f"{mt['profit_factor']:.2f}",t['up'] if mt['profit_factor']>1 else t['dn'],True),
        row('Recovery',f"{mt['recovery_factor']:.2f}"),row('CAGR',f"{mt['cagr']:+.2f}%"),
        sec('⚠️ Risk'),row('Sharpe',f"{mt['sharpe_ratio']:.2f}",t['up'] if mt['sharpe_ratio']>1 else t['text'],True),
        row('Sortino',f"{mt['sortino_ratio']:.2f}"),row('Payoff',f"{mt['payoff_ratio']:.2f}"),
        sec('📉 Drawdown'),row('DD Max',f"${mt['balance_dd_maximal']:,.2f} ({mt['balance_dd_maximal_pct']:.2f}%)",t['dn'],True),
        sec('📋 Trades'),row('Total',f"{mt['total_trades']}",bold=True),
        row('Long (won%)',f"{mt['long_trades']} ({mt['long_won_pct']:.1f}%)"),
        row('Short (won%)',f"{mt['short_trades']} ({mt['short_won_pct']:.1f}%)"),
        row('Profit Trades',f"{mt['profit_trades']} ({mt.get('profit_trades_pct',0):.1f}%)",t['up']),
        row('Loss Trades',f"{mt['loss_trades']} ({mt.get('loss_trades_pct',0):.1f}%)" if mt.get('loss_trades_pct') is not None else f"{mt['loss_trades']}",t['dn']),
        sec('💹 Details'),row('Largest Win',f"${mt['largest_profit_trade']:+,.2f}",t['up']),
        row('Largest Loss',f"${mt['largest_loss_trade']:+,.2f}",t['dn']),
        row('Avg Win',f"${mt['avg_profit_trade']:+,.2f}",t['up']),row('Avg Loss',f"${mt['avg_loss_trade']:+,.2f}",t['dn']),
        sec('🔗 Consecutive'),
        row('Max Wins',f"{mt['max_consecutive_wins']} (${mt['max_consecutive_profit']:+,.2f})",t['up']),
        row('Max Losses',f"{mt['max_consecutive_losses']} (${mt['max_consecutive_loss']:+,.2f})",t['dn'])]
    return html.Div(style={'backgroundColor':t['panel'],'borderRadius':'10px','padding':'16px'},
        children=[html.H3(f"📄 Report — {result.strategy_name}",style={'color':t['text'],'fontSize':'15px','marginBottom':'12px'}),
        html.Table(html.Tbody(rows),style={'width':'100%','borderCollapse':'collapse'})])

def build_mt5_results(result,key):
    t=T()
    if not result.trades: return html.Div("No trades",style={'color':t['dim'],'textAlign':'center','padding':'40px'})
    heads=['#','Open','Type','Qty','Entry$','Close','Exit$','Comm','Profit','Balance','Bars','Tag']
    hrow=html.Tr([html.Th(h,style={'padding':'8px','color':'#fff','backgroundColor':t['accent'],'fontSize':'10px',
        'textTransform':'uppercase','position':'sticky','top':'0','textAlign':'center','whiteSpace':'nowrap'}) for h in heads])
    rows=[]; bal=result.config.initial_capital
    for i,tr in enumerate(result.trades,1):
        bal+=tr.pnl; pc=t['up'] if tr.pnl>0 else t['dn']; sc=t['buy'] if tr.side=='LONG' else t['sell']
        cs={'padding':'5px 8px','fontSize':'11px','borderBottom':f"1px solid {t['border']}",'textAlign':'center','whiteSpace':'nowrap'}
        rows.append(html.Tr([
            html.Td(str(i),style={**cs,'color':t['dim']}),html.Td(str(tr.entry_date)[:10],style={**cs,'color':t['text'],'fontSize':'10px'}),
            html.Td(tr.side,style={**cs,'color':sc,'fontWeight':'bold'}),html.Td(f"{tr.quantity:.4f}",style={**cs,'color':t['text']}),
            html.Td(f"${tr.entry_price:.4f}",style={**cs,'color':t['text']}),html.Td(str(tr.exit_date)[:10],style={**cs,'color':t['text'],'fontSize':'10px'}),
            html.Td(f"${tr.exit_price:.4f}",style={**cs,'color':t['text']}),html.Td(f"${tr.commission:.2f}",style={**cs,'color':t['dim']}),
            html.Td(f"${tr.pnl:+,.2f}",style={**cs,'color':pc,'fontWeight':'bold'}),html.Td(f"${bal:,.0f}",style={**cs,'color':t['text']}),
            html.Td(str(tr.bars_held),style={**cs,'color':t['dim']}),html.Td(tr.tag,style={**cs,'color':t['dim'],'fontSize':'9px'})],
            style={'backgroundColor':t['row_alt'] if i%2==0 else 'transparent'}))
    return html.Div(style={'backgroundColor':t['panel'],'borderRadius':'10px','padding':'12px'},
        children=[html.H3(f"📋 {key} — {len(result.trades)} trades",style={'color':t['text'],'fontSize':'14px','marginBottom':'8px'}),
        html.Div(style={'maxHeight':'500px','overflowY':'auto','overflowX':'auto','border':f"1px solid {t['border']}",'borderRadius':'8px'},
            children=[html.Table([html.Thead(hrow),html.Tbody(rows)],style={'width':'100%','borderCollapse':'collapse'})])])

# ═════════════════════════════════════════════════════
# LIGHTWEIGHT CHARTS DATA API
# ═════════════════════════════════════════════════════

# Stores the current chart data for the iframe
LIVE_CHART_DATA = {'data': None, 'symbol': '', 'is_live': False}


def prepare_chart_json(result, symbol='', source='', show_trades=True):
    """Convert backtest result → Lightweight Charts JSON format"""
    d = result.data

    # Candle data: {time: unix, open, high, low, close, volume}
    candles = []
    for idx, row in d.iterrows():
        ts = int(idx.timestamp())
        candles.append({
            'time': ts,
            'open': round(float(row['Open']), 5),
            'high': round(float(row['High']), 5),
            'low': round(float(row['Low']), 5),
            'close': round(float(row['Close']), 5),
            'volume': int(row.get('Volume', 0)),
        })

    # SMA data
    sma_fast = []
    sma_slow = []
    if 'fast' in d.columns:
        for idx, val in d['fast'].dropna().items():
            sma_fast.append({'time': int(idx.timestamp()), 'value': round(float(val), 5)})
    if 'slow' in d.columns:
        for idx, val in d['slow'].dropna().items():
            sma_slow.append({'time': int(idx.timestamp()), 'value': round(float(val), 5)})

    # Trade markers
    trades = []
    exits = []
    if show_trades and result.trades:
        for tr in result.trades:
            trades.append({
                'time': int(pd.Timestamp(tr.entry_date).timestamp()),
                'side': tr.side,
                'price': round(float(tr.entry_price), 5),
                'pnl': round(float(tr.pnl), 2),
            })
            exits.append({
                'time': int(pd.Timestamp(tr.exit_date).timestamp()),
                'price': round(float(tr.exit_price), 5),
                'pnl': round(float(tr.pnl), 2),
            })

    return {
        'symbol': symbol,
        'source': source,
        'is_live': LIVE_CHART_DATA.get('is_live', False),
        'candles': candles,
        'sma_fast': sma_fast,
        'sma_slow': sma_slow,
        'fast_period': 10,
        'slow_period': 30,
        'trades': trades,
        'exits': exits,
    }


def fetch_live_bar(symbol, tf='1m'):
    """Fetch the latest bar for live updates"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period='1d', interval=tf)
        if df is not None and len(df) > 0:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            last = df.iloc[-1]
            return {
                'time': int(df.index[-1].timestamp()),
                'open': round(float(last.get('Open', last.get('open', 0))), 5),
                'high': round(float(last.get('High', last.get('high', 0))), 5),
                'low': round(float(last.get('Low', last.get('low', 0))), 5),
                'close': round(float(last.get('Close', last.get('close', 0))), 5),
                'volume': int(last.get('Volume', last.get('volume', 0))),
            }
    except:
        pass
    return None