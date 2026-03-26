"""
═══════════════════════════════════════════════════════════
app.py — Dash + TradingView Lightweight Charts + Live Data
═══════════════════════════════════════════════════════════
"""

import os, threading, time, webbrowser, json
from datetime import datetime
from flask import jsonify, request
import numpy as np

from dash import Dash, html, dcc, Output, Input, State, ctx, no_update, ALL
from engine import BacktestConfig, SizingMethod

from charts import (
    T, CURRENT_THEME, RESULTS, RUN_LOG, ALERTS,
    BROKERS, SYMBOL_DB, ALL_BROKERS, ASSET_TYPES, TYPE_LABELS, M,
    LIVE_CHART_DATA,
    clean_date, preset_to_dates, search_symbols, card,
    fetch_data, calc_benchmark, calc_portfolio,
    run_backtest, run_optimization,
    gen_csv, gen_excel, gen_html_report, save_session,
    prepare_chart_json, fetch_live_bar,
    build_tv_chart, build_mt5_graph, build_pnl, build_dist,
    build_monthly_heatmap, build_drawdown_duration,
    build_equity_overlay, build_corr, build_opt_chart,
    build_mt5_report, build_mt5_results, empty_fig,
)

# ═════════════════════════════════════════════════════
# DASH APP
# ═════════════════════════════════════════════════════
app = Dash(__name__, title="PyBacktest Pro",
           suppress_callback_exceptions=True,
           assets_folder='assets')

# Current state for live chart
CURRENT_STATE = {
    'yf_symbol': 'AAPL',
    'tf': '1d',
    'is_live': False,
}


# ═════════════════════════════════════════════════════
# FLASK API ROUTES (for Lightweight Charts iframe)
# ═════════════════════════════════════════════════════
@app.server.route('/api/chart-data')
def api_chart_data():
    """Serve chart data as JSON for Lightweight Charts"""
    if LIVE_CHART_DATA.get('data'):
        return jsonify(LIVE_CHART_DATA['data'])

    # If no backtest run yet, try to fetch basic price data
    sym = CURRENT_STATE.get('yf_symbol', 'AAPL')
    tf = CURRENT_STATE.get('tf', '1d')
    try:
        import pandas as pd
        df, source = fetch_data(sym, tf, '2023-01-01')
        candles = []
        for idx, row in df.iterrows():
            candles.append({
                'time': int(idx.timestamp()),
                'open': round(float(row['Open']), 5),
                'high': round(float(row['High']), 5),
                'low': round(float(row['Low']), 5),
                'close': round(float(row['Close']), 5),
                'volume': int(row.get('Volume', 0)),
            })
        return jsonify({
            'symbol': sym,
            'source': source,
            'is_live': CURRENT_STATE.get('is_live', False),
            'candles': candles,
            'trades': [], 'exits': [],
            'sma_fast': [], 'sma_slow': [],
        })
    except:
        return jsonify({'candles': [], 'symbol': sym, 'error': 'No data'})


@app.server.route('/api/live-bar')
def api_live_bar():
    """Serve latest bar for live updates"""
    after = request.args.get('after', 0, type=int)
    sym = CURRENT_STATE.get('yf_symbol', 'AAPL')
    tf = CURRENT_STATE.get('tf', '1m')
    bar = fetch_live_bar(sym, tf)
    if bar and bar['time'] > after:
        return jsonify({'bar': bar})
    return jsonify({'bar': None})


# ═════════════════════════════════════════════════════
# MODAL BUILDER
# ═════════════════════════════════════════════════════
def build_modal():
    tab_base = {'padding':'6px 16px','border':'none','borderRadius':'20px',
                'cursor':'pointer','fontSize':'13px','fontWeight':'500','marginRight':'4px'}
    return html.Div(id='symbol-modal',style={'display':'none'},children=[
        html.Div(id='modal-overlay-bg',n_clicks=0,
            style={'position':'fixed','top':0,'left':0,'right':0,'bottom':0,
                   'backgroundColor':M['overlay_bg'],'zIndex':9998}),
        html.Div(style={'position':'fixed','top':'50%','left':'50%',
            'transform':'translate(-50%,-50%)','width':'680px','maxWidth':'95vw',
            'maxHeight':'85vh','backgroundColor':M['modal_bg'],'borderRadius':'12px',
            'border':f"1px solid {M['modal_border']}",'boxShadow':'0 20px 60px rgba(0,0,0,0.5)',
            'zIndex':9999,'display':'flex','flexDirection':'column','overflow':'hidden'},
            children=[
                html.Div(style={'display':'flex','justifyContent':'space-between',
                    'alignItems':'center','padding':'16px 20px 12px'},children=[
                    html.Span("Symbol Search",style={'color':'#d1d4dc','fontSize':'18px','fontWeight':'600'}),
                    html.Button("✕",id='modal-close-btn',n_clicks=0,
                        style={'background':'none','border':'none','color':M['close_color'],
                               'fontSize':'22px','cursor':'pointer'})]),
                html.Div(style={'padding':'0 20px 12px'},children=[
                    html.Div(style={'display':'flex','alignItems':'center','backgroundColor':M['search_bg'],
                        'border':f"1px solid {M['modal_border']}",'borderRadius':'8px','padding':'0 12px'},children=[
                        html.Span("🔍",style={'fontSize':'16px','marginRight':'10px','color':M['tab_inactive']}),
                        dcc.Input(id='modal-search',type='text',value='',placeholder='Symbol, ISIN, or CUSIP',
                            debounce=False,style={'flex':'1','padding':'12px 0','backgroundColor':'transparent',
                                'color':M['search_text'],'border':'none','fontSize':'15px'})])]),
                html.Div(style={'display':'flex','padding':'0 20px 10px','gap':'2px','flexWrap':'wrap'},
                    children=[html.Button(TYPE_LABELS.get(cat,cat),id={'type':'cat-btn','cat':cat},n_clicks=0,
                        style={**tab_base,'backgroundColor':M['tab_active_bg'] if cat=='all' else 'transparent',
                               'color':M['tab_active_text'] if cat=='all' else M['tab_inactive']})
                        for cat in ASSET_TYPES]),
                html.Div(style={'padding':'0 20px 8px'},children=[
                    dcc.Dropdown(id='modal-source-filter',
                        options=[{'label':'  All sources','value':'all'}]+
                                [{'label':f"  {BROKERS[b]['icon']} {b}",'value':b} for b in ALL_BROKERS],
                        value='all',clearable=False,style={'fontSize':'12px','width':'200px'})]),
                html.Hr(style={'margin':'0','border':'none','borderTop':f"1px solid {M['modal_border']}"}),
                html.Div(id='modal-results',style={'flex':'1','overflowY':'auto','padding':'4px 0','maxHeight':'50vh'}),
                html.Div(style={'padding':'8px 20px','borderTop':f"1px solid {M['modal_border']}",'textAlign':'center'},
                    children=[html.Span("Type to search across all markets",style={'color':M['tab_inactive'],'fontSize':'12px'})])])])


def build_result_rows(symbols_list):
    if not symbols_list:
        return html.Div("No results found",style={'color':M['tab_inactive'],'textAlign':'center','padding':'40px'})
    rows = []
    for sym in symbols_list:
        bi = BROKERS.get(sym['broker'],{}); bc = bi.get('color','#787b86')
        rows.append(html.Div(id={'type':'sym-pick','index':sym['key']},n_clicks=0,
            style={'display':'flex','alignItems':'center','padding':'10px 20px','cursor':'pointer',
                   'borderBottom':f"1px solid {M['modal_border']}"},
            children=[
                html.Span(sym['flag'],style={'fontSize':'20px','marginRight':'14px','minWidth':'28px','textAlign':'center'}),
                html.Span(sym['ticker'],style={'color':M['ticker_color'],'fontSize':'14px','fontWeight':'700',
                    'minWidth':'100px','marginRight':'14px','letterSpacing':'0.5px'}),
                html.Span(sym['name'],style={'color':M['name_color'],'fontSize':'13px','flex':'1',
                    'overflow':'hidden','textOverflow':'ellipsis','whiteSpace':'nowrap'}),
                html.Span("FEATURED",style={'backgroundColor':M['featured_bg'],'color':M['featured_text'],
                    'padding':'2px 8px','borderRadius':'4px','fontSize':'10px','fontWeight':'700',
                    'marginRight':'10px'}) if sym.get('featured') else html.Span(),
                html.Div(style={'display':'flex','alignItems':'center','gap':'6px','minWidth':'120px','justifyContent':'flex-end'},
                    children=[
                        html.Span(sym['broker'],style={'color':M['broker_color'],'fontSize':'12px','fontWeight':'500'}),
                        html.Span(bi.get('badge','•'),style={'backgroundColor':bc,'color':'white','width':'20px','height':'20px',
                            'borderRadius':'50%','display':'inline-flex','alignItems':'center','justifyContent':'center',
                            'fontSize':'10px','fontWeight':'bold'})])]))
    return html.Div(rows)


# ═════════════════════════════════════════════════════
# LAYOUT
# ═════════════════════════════════════════════════════
def make_layout():
    t = T()
    ts = {'backgroundColor':t['tab_bg'],'color':t['tab_text'],'padding':'10px 20px',
          'borderRadius':'8px 8px 0 0','border':'none','fontSize':'12px','fontWeight':'600',
          'cursor':'pointer','marginRight':'2px'}
    tss = {**ts,'backgroundColor':t['panel'],'color':t['accent'],'borderBottom':f"3px solid {t['accent']}"}

    return html.Div(style={'backgroundColor':t['bg'],'minHeight':'100vh','padding':'10px 14px',
        'fontFamily':"'Segoe UI',-apple-system,sans-serif"},children=[

        build_modal(),

        # ── TOP BAR ──
        html.Div(style={'display':'flex','alignItems':'center','gap':'8px','padding':'10px 14px',
            'backgroundColor':t['panel'],'borderRadius':'10px','marginBottom':'10px',
            'borderBottom':f"3px solid {t['accent']}",'flexWrap':'wrap','boxShadow':t['card_shadow']},
            children=[
                html.Span("📊",style={'fontSize':'22px'}),
                html.Span("PyBacktest Pro",style={'fontSize':'16px','fontWeight':'bold','color':t['text'],'marginRight':'8px'}),

                # Symbol button
                html.Button(id='symbol-display-btn',n_clicks=0,children=[
                    html.Span("🔍 "),
                    html.Span(id='selected-symbol-text',children="AAPL",style={'fontWeight':'bold','fontSize':'14px'}),
                    html.Span(id='selected-broker-text',children=" • NASDAQ",style={'fontSize':'11px','opacity':'0.7','marginLeft':'4px'})],
                    style={'padding':'8px 18px','backgroundColor':t['input_bg'],'color':t['text'],
                        'border':f"1px solid {t['border']}",'borderRadius':'8px','cursor':'pointer','minWidth':'200px',
                        'display':'flex','alignItems':'center','gap':'4px'}),

                # Timeframe
                dcc.Dropdown(id='tf',options=[
                    {'label':'1m','value':'1m'},{'label':'5m','value':'5m'},{'label':'15m','value':'15m'},
                    {'label':'1h','value':'1h'},{'label':'Daily','value':'1d'},
                    {'label':'Weekly','value':'1wk'},{'label':'Monthly','value':'1mo'}],
                    value='1d',clearable=False,style={'width':'85px','fontSize':'11px'}),

                # Date preset
                html.Div(style={'display':'flex','alignItems':'center','gap':'4px',
                    'border':f"1px solid {t['border']}",'borderRadius':'6px','padding':'3px 8px',
                    'backgroundColor':t['input_bg']},
                    children=[html.Span("📅"),
                        dcc.Dropdown(id='date-preset',options=[
                            {'label':'Last 7 Days','value':'7d'},{'label':'Last 30 Days','value':'30d'},
                            {'label':'Last 90 Days','value':'90d'},{'label':'1 Year','value':'1y'},
                            {'label':'2 Years','value':'2y'},{'label':'5 Years','value':'5y'},
                            {'label':'All','value':'all'},{'label':'Custom...','value':'custom'}],
                            value='5y',clearable=False,style={'width':'140px','fontSize':'11px'})]),

                html.Div(id='custom-dates-box',style={'display':'none','alignItems':'center','gap':'4px'},children=[
                    dcc.Input(id='date-start',type='date',value='2020-01-01',
                        style={'width':'120px','padding':'5px','backgroundColor':t['input_bg'],'color':t['text'],
                            'border':f"1px solid {t['border']}",'borderRadius':'6px','fontSize':'11px'}),
                    html.Span("→",style={'color':t['dim']}),
                    dcc.Input(id='date-end',type='date',value=datetime.now().strftime('%Y-%m-%d'),
                        style={'width':'120px','padding':'5px','backgroundColor':t['input_bg'],'color':t['text'],
                            'border':f"1px solid {t['border']}",'borderRadius':'6px','fontSize':'11px'})]),

                # Config
                html.Div(style={'display':'flex','alignItems':'center','gap':'4px'},children=[
                    html.Label("$:",style={'color':t['dim'],'fontSize':'10px'}),
                    dcc.Input(id='capital',type='number',value=100000,style={'width':'80px','padding':'5px',
                        'backgroundColor':t['input_bg'],'color':t['text'],'border':f"1px solid {t['border']}",
                        'borderRadius':'6px','fontSize':'11px'}),
                    html.Label("C%:",style={'color':t['dim'],'fontSize':'10px'}),
                    dcc.Input(id='comm',type='number',value=0.1,step=0.01,style={'width':'50px','padding':'5px',
                        'backgroundColor':t['input_bg'],'color':t['text'],'border':f"1px solid {t['border']}",
                        'borderRadius':'6px','fontSize':'11px'}),
                    html.Label("S%:",style={'color':t['dim'],'fontSize':'10px'}),
                    dcc.Input(id='slip',type='number',value=0.05,step=0.01,style={'width':'50px','padding':'5px',
                        'backgroundColor':t['input_bg'],'color':t['text'],'border':f"1px solid {t['border']}",
                        'borderRadius':'6px','fontSize':'11px'})]),

                dcc.Checklist(id='shorts',options=[{'label':' Shorts','value':'y'}],value=['y'],inline=True,
                    style={'color':t['text'],'fontSize':'10px'}),

                # ── LIVE TOGGLE ──
                html.Button("📡 Live",id='live-btn',n_clicks=0,
                    style={'padding':'7px 14px','backgroundColor':'#2a2e39','color':'#787b86',
                        'border':'1px solid #363a45','borderRadius':'6px','cursor':'pointer',
                        'fontSize':'11px','fontWeight':'bold'}),

                html.Button("▶ Run",id='run-btn',n_clicks=0,
                    style={'padding':'8px 20px','backgroundColor':t['accent'],'color':'white','border':'none',
                        'borderRadius':'8px','cursor':'pointer','fontSize':'13px','fontWeight':'bold',
                        'boxShadow':f"0 2px 8px {t['accent']}40"}),
                html.Button("💾",id='save-btn',n_clicks=0,
                    style={'padding':'7px 12px','backgroundColor':t['input_bg'],'color':t['text'],
                        'border':f"1px solid {t['border']}",'borderRadius':'6px','cursor':'pointer'}),
                html.Button("🌙" if CURRENT_THEME['mode']=='light' else "☀️",id='theme-btn',n_clicks=0,
                    style={'padding':'7px 12px','backgroundColor':t['input_bg'],'color':t['text'],
                        'border':f"1px solid {t['border']}",'borderRadius':'6px','cursor':'pointer','fontSize':'14px'}),
                html.Div(style={'flex':'1'}),
                html.Label("View:",style={'color':t['dim'],'fontSize':'10px'}),
                dcc.Dropdown(id='view',options=[],value=None,clearable=False,placeholder='Run first...',
                    style={'width':'180px','fontSize':'11px'})]),

        # UI areas
        html.Div(id='alerts',style={'marginBottom':'6px'}),
        html.Div(id='status',style={'marginBottom':'6px'}),
        html.Div(id='cards',style={'marginBottom':'6px'}),
        html.Div(id='bm-cards',style={'marginBottom':'8px'}),

        # ── TABS ──
        dcc.Tabs(id='main-tabs',value='chart',style={'borderBottom':f"1px solid {t['border']}"},children=[
            dcc.Tab(label='📈 Chart',value='chart',style=ts,selected_style=tss),
            dcc.Tab(label='📊 Graph',value='graph',style=ts,selected_style=tss),
            dcc.Tab(label='📋 Results',value='results',style=ts,selected_style=tss),
            dcc.Tab(label='📄 Report',value='report',style=ts,selected_style=tss),
            dcc.Tab(label='🔧 Optimize',value='optimize',style=ts,selected_style=tss),
            dcc.Tab(label='🔗 Compare',value='compare',style=ts,selected_style=tss)]),

        html.Div(id='tab-content',style={'backgroundColor':t['panel'],'borderRadius':'0 0 10px 10px',
            'padding':'0','minHeight':'500px','boxShadow':t['card_shadow']}),

        # Export
        html.Div(style={'display':'flex','gap':'8px','marginTop':'10px','flexWrap':'wrap','alignItems':'center'},
            children=[
                html.Button("📥 CSV",id='exp-csv',n_clicks=0,style={'padding':'7px 16px','backgroundColor':t['panel'],
                    'color':t['text'],'border':f"1px solid {t['border']}",'borderRadius':'6px','cursor':'pointer','fontSize':'11px'}),
                html.Button("📊 Excel",id='exp-xl',n_clicks=0,style={'padding':'7px 16px','backgroundColor':t['panel'],
                    'color':t['text'],'border':f"1px solid {t['border']}",'borderRadius':'6px','cursor':'pointer','fontSize':'11px'}),
                html.Button("📄 HTML",id='exp-html',n_clicks=0,style={'padding':'7px 16px','backgroundColor':t['panel'],
                    'color':t['text'],'border':f"1px solid {t['border']}",'borderRadius':'6px','cursor':'pointer','fontSize':'11px'}),
                html.Div(style={'flex':'1'}),
                html.Div(id='sessions',style={'fontSize':'10px','color':t['dim']})]),
        dcc.Download(id='dl-csv'),dcc.Download(id='dl-xl'),dcc.Download(id='dl-html'),
        html.Div(id='log',style={'backgroundColor':t['panel'],'borderRadius':'10px','padding':'10px 14px',
            'marginTop':'8px','fontSize':'10px','color':t['dim']}),

        # Stores
        dcc.Store(id='theme-store',data='light'),
        dcc.Store(id='run-counter',data=0),
        dcc.Store(id='selected-symbol-data',data={'ticker':'AAPL','yf_symbol':'AAPL','broker':'NASDAQ','name':'Apple Inc.','type':'stocks'}),
        dcc.Store(id='active-cat',data='all'),
        dcc.Store(id='live-active',data=False),

        # Live update interval (disabled by default)
        dcc.Interval(id='live-interval',interval=5000,disabled=True),
    ])

app.layout = make_layout()


# ═════════════════════════════════════════════════════
# MODAL CALLBACKS
# ═════════════════════════════════════════════════════
@app.callback(Output('symbol-modal','style'),
    [Input('symbol-display-btn','n_clicks'),Input('modal-close-btn','n_clicks'),
     Input('modal-overlay-bg','n_clicks')],prevent_initial_call=True)
def toggle_modal(o,c,ov):
    return {'display':'block'} if ctx.triggered_id=='symbol-display-btn' else {'display':'none'}

@app.callback(Output('modal-results','children'),
    [Input('modal-search','value'),Input('active-cat','data'),Input('modal-source-filter','value')])
def update_search(q,cat,src):
    return build_result_rows(search_symbols(q or '',cat or 'all',src or 'all'))

@app.callback(Output('active-cat','data'),Input({'type':'cat-btn','cat':ALL},'n_clicks'),prevent_initial_call=True)
def cat_click(clicks):
    return ctx.triggered_id['cat'] if ctx.triggered_id else no_update

@app.callback([Output({'type':'cat-btn','cat':cat},'style') for cat in ASSET_TYPES],Input('active-cat','data'))
def tab_styles(active):
    b={'padding':'6px 16px','border':'none','borderRadius':'20px','cursor':'pointer','fontSize':'13px','fontWeight':'500','marginRight':'4px'}
    return [{**b,'backgroundColor':M['tab_active_bg'],'color':M['tab_active_text']} if cat==active
            else {**b,'backgroundColor':'transparent','color':M['tab_inactive']} for cat in ASSET_TYPES]

@app.callback([Output('selected-symbol-data','data'),Output('selected-symbol-text','children'),
    Output('selected-broker-text','children'),Output('symbol-modal','style',allow_duplicate=True)],
    Input({'type':'sym-pick','index':ALL},'n_clicks'),prevent_initial_call=True)
def pick_symbol(clicks):
    if not ctx.triggered_id or not any(c and c>0 for c in clicks): return no_update,no_update,no_update,no_update
    key=ctx.triggered_id['index']
    for s in SYMBOL_DB:
        if s['key']==key:
            CURRENT_STATE['yf_symbol'] = s['yf_symbol']
            return ({'ticker':s['ticker'],'yf_symbol':s['yf_symbol'],'broker':s['broker'],'name':s['name'],'type':s['type']},
                    s['ticker'],f" • {s['broker']}",{'display':'none'})
    return no_update,no_update,no_update,no_update


# ═════════════════════════════════════════════════════
# LIVE TOGGLE
# ═════════════════════════════════════════════════════
@app.callback(
    [Output('live-active','data'),
     Output('live-interval','disabled'),
     Output('live-btn','style'),
     Output('live-btn','children')],
    Input('live-btn','n_clicks'),
    State('live-active','data'),
    prevent_initial_call=True)
def toggle_live(n, is_active):
    new_state = not is_active
    CURRENT_STATE['is_live'] = new_state
    LIVE_CHART_DATA['is_live'] = new_state

    if new_state:
        return True, False, {
            'padding':'7px 14px','backgroundColor':'rgba(38,166,154,0.15)',
            'color':'#26a69a','border':'1px solid #26a69a','borderRadius':'6px',
            'cursor':'pointer','fontSize':'11px','fontWeight':'bold'
        }, "📡 LIVE ●"
    else:
        return False, True, {
            'padding':'7px 14px','backgroundColor':'#2a2e39',
            'color':'#787b86','border':'1px solid #363a45','borderRadius':'6px',
            'cursor':'pointer','fontSize':'11px','fontWeight':'bold'
        }, "📡 Live"


# ═════════════════════════════════════════════════════
# GENERAL CALLBACKS
# ═════════════════════════════════════════════════════
@app.callback([Output('date-start','value'),Output('date-end','value'),Output('custom-dates-box','style')],
    Input('date-preset','value'),prevent_initial_call=True)
def update_dates(p):
    if p=='custom': return no_update,no_update,{'display':'flex','alignItems':'center','gap':'4px'}
    s,e=preset_to_dates(p)
    return (s,e,{'display':'none'}) if s else (no_update,no_update,{'display':'none'})

@app.callback(Output('theme-store','data'),Input('theme-btn','n_clicks'),State('theme-store','data'),prevent_initial_call=True)
def toggle_theme(n,cur): new='light' if cur=='dark' else 'dark'; CURRENT_THEME['mode']=new; return new

@app.callback(Output('dl-csv','data'),Input('exp-csv','n_clicks'),State('view','value'),prevent_initial_call=True)
def csv_cb(n,s):
    if not s or s not in RESULTS: return no_update
    return dict(content=gen_csv(RESULTS[s],s),filename=f"bt_{s.replace(' ','_').replace('•','')}.csv")

@app.callback(Output('dl-xl','data'),Input('exp-xl','n_clicks'),State('view','value'),prevent_initial_call=True)
def xl_cb(n,s):
    if not s or s not in RESULTS: return no_update
    d=gen_excel(RESULTS[s],s); return dcc.send_bytes(d,f"bt_{s.replace(' ','_').replace('•','')}.xlsx") if d else no_update

@app.callback(Output('dl-html','data'),Input('exp-html','n_clicks'),State('view','value'),prevent_initial_call=True)
def html_cb(n,s):
    if not s or s not in RESULTS: return no_update
    fp=gen_html_report(RESULTS[s],s); return dcc.send_file(fp) if fp and os.path.exists(fp) else no_update

@app.callback(Output('sessions','children'),Input('save-btn','n_clicks'),prevent_initial_call=True)
def save_cb(n):
    t=T()
    if not RESULTS: return html.Span("❌",style={'color':t['dn']})
    name=datetime.now().strftime('s_%Y%m%d_%H%M%S'); save_session(name)
    return html.Span(f"✅ {name}",style={'color':t['up']})


# ═════════════════════════════════════════════════════
# RUN BACKTEST
# ═════════════════════════════════════════════════════
@app.callback(
    [Output('status','children'),Output('alerts','children'),Output('cards','children'),
     Output('bm-cards','children'),Output('view','options'),Output('view','value'),
     Output('run-counter','data'),Output('log','children')],
    Input('run-btn','n_clicks'),
    [State('selected-symbol-data','data'),State('tf','value'),State('date-start','value'),
     State('date-end','value'),State('capital','value'),State('comm','value'),
     State('slip','value'),State('shorts','value')],
    prevent_initial_call=True)
def run_cb(n,sym_data,tf,ds,de,cap,comm,slip,shorts):
    t=T()
    if not sym_data: return html.Div("❌ Select symbol",style={'color':t['dn']}),html.Div(),html.Div(),html.Div(),[],None,n,html.Div()

    yf_sym=sym_data.get('yf_symbol','AAPL'); ticker=sym_data.get('ticker','AAPL')
    broker=sym_data.get('broker',''); name=sym_data.get('name',''); atype=sym_data.get('type','stocks')
    ti={'forex':'💱','crypto':'₿','futures':'🛢️','indices':'📊','funds':'📈'}.get(atype,'📈')

    sd=clean_date(ds,'2020-01-01'); ed=clean_date(de,datetime.now().strftime('%Y-%m-%d'))
    tf_val = tf or '1d'
    CURRENT_STATE['yf_symbol'] = yf_sym
    CURRENT_STATE['tf'] = tf_val

    try: cf=float(cap) if cap else 100000
    except: cf=100000
    try: cm=float(comm)/100 if comm else 0.001
    except: cm=0.001
    try: sl=float(slip)/100 if slip else 0.0005
    except: sl=0.0005

    config=BacktestConfig(initial_capital=cf,commission_pct=cm,slippage_pct=sl,use_volume_slippage=True,
        allow_short='y' in (shorts or []),sizing_method=SizingMethod.FIXED_FRACTIONAL,sizing_param=0.02)

    try:
        result,key,source=run_backtest(yf_sym,tf_val,config,sd,ed,f"{ti} {ticker}")
        # Update chart data for Lightweight Charts
        LIVE_CHART_DATA['data'] = prepare_chart_json(result, f"{ticker} ({broker})", source)
        LIVE_CHART_DATA['symbol'] = ticker
    except Exception as e:
        return (html.Div(f"❌ {yf_sym}: {e}",style={'backgroundColor':'rgba(220,53,69,0.08)',
            'border':f"1px solid {t['dn']}",'borderRadius':'8px','padding':'8px 14px','fontSize':'12px','color':t['dn']}),
            html.Div(),html.Div(),html.Div(),
            [{'label':k,'value':k} for k in RESULTS],list(RESULTS.keys())[-1] if RESULTS else None,n,_log())

    m=result.metrics
    a_s=result.data.index[0].strftime('%Y-%m-%d'); a_e=result.data.index[-1].strftime('%Y-%m-%d')
    si={'yfinance':'🌐','cache':'💾','demo':'🎲','resample':'📊'}

    # Intraday warning
    warn = ''
    if tf_val in ('1m','5m','15m'): warn = f" ⚠️ {tf_val} data: limited history (yfinance)"

    status=html.Div(style={'backgroundColor':'rgba(25,135,84,0.08)','border':f"1px solid {t['up']}",
        'borderRadius':'8px','padding':'8px 14px','fontSize':'12px','color':t['up']},
        children=[html.Span(f"✅ {ti} {ticker}",style={'fontWeight':'bold'}),
            html.Span(f"  ({name})",style={'opacity':'0.7','marginRight':'6px'}),
            html.Span(f"📅 {a_s}→{a_e} • {len(result.data)} bars • {m['total_trades']} trades • {si.get(source,'')}{source} • ${cf:,.0f}",style={'opacity':'0.85'}),
            html.Span(warn,style={'color':'#fd7e14','fontSize':'11px'}) if warn else html.Span()])

    alerts_div = html.Div([html.Div(a) for a in ALERTS[-5:]],
        style={'backgroundColor':'rgba(253,126,20,0.08)','border':f"1px solid {t['sell']}",
            'borderRadius':'8px','padding':'8px 14px','fontSize':'11px','color':t['sell']}) if ALERTS[-5:] else html.Div()

    cards_div=html.Div(style={'display':'grid','gridTemplateColumns':'repeat(9,1fr)','gap':'6px'},children=[
        card('Return',f"{m['total_return_pct']:+.2f}%",t['up'] if m['total_return']>=0 else t['dn']),
        card('CAGR',f"{m['cagr']*100:+.1f}%",t['up'] if m['cagr']>=0 else t['dn']),
        card('Sharpe',f"{m['sharpe_ratio']:.2f}",t['accent']),card('Sortino',f"{m['sortino_ratio']:.2f}",t['accent']),
        card('MaxDD',f"-{m['max_drawdown_pct']:.1f}%",t['dn']),
        card('WinRate',f"{m['win_rate']*100:.1f}%",t['up'] if m['win_rate']>0.5 else t['sell']),
        card('PF',f"{m['profit_factor']:.2f}",t['up'] if m['profit_factor']>1 else t['dn']),
        card('Trades',f"{m['total_trades']}",t['text']),card('Final',f"${m['final_equity']:,.0f}",t['text'])])

    bm=calc_benchmark(result); bm_cards=html.Div()
    if bm:
        bm_cards=html.Div(style={'display':'grid','gridTemplateColumns':'repeat(5,1fr)','gap':'6px','marginTop':'6px'},children=[
            card('S&P500',f"{bm['benchmark_return']:+.1f}%",t['dim']),
            card('Alpha',f"{bm['alpha']:+.2f}%",t['up'] if bm['alpha']>0 else t['dn']),
            card('Beta',f"{bm['beta']:.2f}",t['accent']),card('Corr',f"{bm['correlation']:.2f}",t['accent']),
            card('IR',f"{bm['information_ratio']:.2f}",t['up'] if bm['information_ratio']>0 else t['dn'])])

    opts=[{'label':f'  {k}','value':k} for k in RESULTS]
    return status,alerts_div,cards_div,bm_cards,opts,key,n,_log()


# ═════════════════════════════════════════════════════
# TAB CONTENT
# ═════════════════════════════════════════════════════
@app.callback(Output('tab-content','children'),
    [Input('main-tabs','value'),Input('view','value'),Input('run-counter','data'),Input('theme-store','data')],
    prevent_initial_call=True)
def render_tab(tab,selected,rc,theme):
    t=T()
    r = None; key = None
    if selected and selected in RESULTS: r=RESULTS[selected]; key=selected
    elif RESULTS: key=list(RESULTS.keys())[-1]; r=RESULTS[key]

    # ── CHART TAB — Lightweight Charts (Real TradingView!) ──
    if tab == 'chart':
        return html.Div([
            html.Iframe(
                src='/assets/tv_chart.html',
                style={
                    'width': '100%',
                    'height': '680px',
                    'border': 'none',
                    'borderRadius': '0 0 10px 10px',
                },
                id='tv-chart-iframe',
            ),
        ])

    if not r:
        return html.Div(style={'textAlign':'center','padding':'80px'},children=[
            html.Div("📊",style={'fontSize':'48px','marginBottom':'12px'}),
            html.Div("Click symbol → ▶ Run",style={'fontSize':'16px','color':t['dim']})])

    if tab=='graph':
        return html.Div(style={'padding':'12px'},children=[
            dcc.Graph(figure=build_mt5_graph(r)),
            html.Div(style={'display':'grid','gridTemplateColumns':'1fr 1fr','gap':'8px','marginTop':'8px'},children=[
                dcc.Graph(figure=build_pnl(r),config={'displayModeBar':False}),
                dcc.Graph(figure=build_dist(r),config={'displayModeBar':False})]),
            dcc.Graph(figure=build_monthly_heatmap(r),config={'displayModeBar':False},style={'marginTop':'8px'}),
            dcc.Graph(figure=build_drawdown_duration(r),config={'displayModeBar':False},style={'marginTop':'8px'})])
    elif tab=='results': return html.Div(style={'padding':'12px'},children=[build_mt5_results(r,key)])
    elif tab=='report': return html.Div(style={'padding':'12px'},children=[build_mt5_report(r)])
    elif tab=='optimize':
        return html.Div(style={'padding':'12px'},children=[
            html.H3("🔧 Optimization",style={'color':t['text'],'fontSize':'15px','marginBottom':'12px'}),
            html.Div(style={'display':'flex','gap':'8px','alignItems':'center','flexWrap':'wrap','marginBottom':'12px'},children=[
                html.Label("Param:",style={'color':t['dim'],'fontSize':'11px'}),
                dcc.Input(id='opt-param',type='text',value='fast_len',style={'width':'120px','padding':'6px',
                    'backgroundColor':t['input_bg'],'color':t['text'],'border':f"1px solid {t['border']}",'borderRadius':'6px','fontSize':'11px'}),
                html.Label("Values:",style={'color':t['dim'],'fontSize':'11px'}),
                dcc.Input(id='opt-vals',type='text',value='5,10,15,20,25,30',style={'width':'200px','padding':'6px',
                    'backgroundColor':t['input_bg'],'color':t['text'],'border':f"1px solid {t['border']}",'borderRadius':'6px','fontSize':'11px'}),
                html.Button("🔍 Optimize",id='opt-btn',n_clicks=0,style={'padding':'7px 18px','backgroundColor':'#fd7e14',
                    'color':'white','border':'none','borderRadius':'6px','cursor':'pointer','fontSize':'12px','fontWeight':'bold'})]),
            dcc.Loading(dcc.Graph(id='fig-opt',figure=empty_fig("Set param → Optimize"),config={'displayModeBar':False}),type='dot',color='#fd7e14')])
    elif tab=='compare':
        return html.Div(style={'padding':'12px'},children=[_comp(),
            html.Div(style={'display':'grid','gridTemplateColumns':'1fr 1fr','gap':'8px','marginTop':'12px'},children=[
                dcc.Graph(figure=build_equity_overlay(),config={'displayModeBar':False}),
                dcc.Graph(figure=build_corr(),config={'displayModeBar':False})]),_port()])
    return html.Div()


# Optimization callback
@app.callback(Output('fig-opt','figure'),Input('opt-btn','n_clicks'),
    [State('selected-symbol-data','data'),State('tf','value'),State('date-start','value'),State('date-end','value'),
     State('opt-param','value'),State('opt-vals','value'),State('capital','value'),State('comm','value'),
     State('slip','value'),State('shorts','value')],prevent_initial_call=True)
def optimize(n,sym_data,tf,ds,de,param,vals_str,cap,comm,slip,shorts):
    if not sym_data or not param or not vals_str: return no_update
    yf_sym=sym_data.get('yf_symbol','AAPL')
    try: vals=[int(v.strip()) if v.strip().isdigit() else float(v.strip()) for v in vals_str.split(',')]
    except: return empty_fig("Invalid values")
    try: cf=float(cap) if cap else 100000
    except: cf=100000
    try: cm=float(comm)/100 if comm else 0.001
    except: cm=0.001
    try: sl=float(slip)/100 if slip else 0.0005
    except: sl=0.0005
    config=BacktestConfig(initial_capital=cf,commission_pct=cm,slippage_pct=sl,
        allow_short='y' in (shorts or []),sizing_method=SizingMethod.FIXED_FRACTIONAL,sizing_param=0.02)
    results=run_optimization(yf_sym,tf or '1d',param,vals,config,
        clean_date(ds,'2020-01-01'),clean_date(de,datetime.now().strftime('%Y-%m-%d')))
    return build_opt_chart(results,param)


# ═════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════
def _comp():
    t=T()
    if not RESULTS: return html.P("Run backtests to compare",style={'color':t['dim'],'textAlign':'center','padding':'20px'})
    sk=sorted(RESULTS.keys(),key=lambda k:RESULTS[k].metrics['sharpe_ratio'],reverse=True)
    heads=['🏆','Backtest','Return%','Sharpe','MaxDD%','WR%','PF','Trades','Final$']
    hrow=html.Tr([html.Th(h,style={'padding':'8px 10px','color':'#fff','backgroundColor':t['accent'],'fontSize':'10px','textTransform':'uppercase','textAlign':'left'}) for h in heads])
    rows=[]
    for rank,k in enumerate(sk):
        m=RESULTS[k].metrics; rc=t['up'] if m['total_return']>=0 else t['dn']
        badge=['🥇','🥈','🥉'][rank] if rank<3 else str(rank+1)
        cs={'padding':'6px 10px','fontSize':'11px','borderBottom':f"1px solid {t['border']}"}
        rows.append(html.Tr([html.Td(badge,style={**cs,'color':t['accent'] if rank==0 else t['dim']}),
            html.Td(k,style={**cs,'color':t['text'],'fontWeight':'bold' if rank==0 else 'normal'}),
            html.Td(f"{m['total_return_pct']:+.2f}%",style={**cs,'color':rc,'fontWeight':'bold'}),
            html.Td(f"{m['sharpe_ratio']:.2f}",style={**cs,'color':t['accent'],'fontWeight':'bold'}),
            html.Td(f"-{m['max_drawdown_pct']:.1f}%",style={**cs,'color':t['dn']}),
            html.Td(f"{m['win_rate']*100:.1f}%",style={**cs,'color':t['text']}),
            html.Td(f"{m['profit_factor']:.2f}",style={**cs,'color':t['up'] if m['profit_factor']>1 else t['dn']}),
            html.Td(str(m['total_trades']),style={**cs,'color':t['dim']}),
            html.Td(f"${m['final_equity']:,.0f}",style={**cs,'color':t['text']})],
            style={'backgroundColor':t['row_alt'] if rank%2==0 else 'transparent'}))
    return html.Div([html.H3(f"📊 Compare — {len(RESULTS)}",style={'color':t['text'],'fontSize':'14px','marginBottom':'12px'}),
        html.Table([html.Thead(hrow),html.Tbody(rows)],style={'width':'100%','borderCollapse':'collapse'})])

def _port():
    t=T(); pm=calc_portfolio()
    if pm is None: return html.Div()
    return html.Div(style={'marginTop':'12px','padding':'14px','backgroundColor':t['row_alt'],'borderRadius':'8px'},
        children=[html.H3("📊 Portfolio",style={'color':t['text'],'fontSize':'14px','marginBottom':'10px'}),
        html.Div(style={'display':'grid','gridTemplateColumns':'repeat(5,1fr)','gap':'6px'},children=[
            card('Return',f"{pm['ret']:+.1f}%",t['up'] if pm['ret']>0 else t['dn']),
            card('Sharpe',f"{pm['sharpe']:.2f}",t['accent']),card('MaxDD',f"-{pm['maxdd']:.1f}%",t['dn']),
            card('Diversif',f"{pm['div']:.2f}x",t['up'] if pm['div']>1 else t['dim']),
            card('Assets',f"{pm['n']}",t['text'])])])

def _log():
    t=T()
    if not RUN_LOG: return html.Div()
    return html.Div([html.Span("📝 Log:",style={'fontWeight':'bold','color':t['text']}),html.Br(),*[html.Div(l) for l in RUN_LOG[-10:]]])


# ═════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════
if __name__ == '__main__':
    PORT = 8050
    print(f"""
{'═'*60}
  📊 PyBacktest Pro — TradingView Lightweight Charts

  http://localhost:{PORT}

  📈 CHART: Real TradingView (Lightweight Charts v4)
  ├─ Smooth scrolling, zooming, crosshair
  ├─ Candlestick + Volume + SMA overlays
  ├─ Trade entry/exit markers
  ├─ 📡 Live market data (5s polling)
  └─ Dark theme (exact TV colors)

  📊 TABS: Graph | Results | Report | Optimize | Compare
  🔍 MODAL: TradingView symbol search
  💱 200+ symbols across all markets

  Ctrl+C to stop
{'═'*60}
""")
    threading.Thread(target=lambda:(time.sleep(1.5),webbrowser.open(f'http://localhost:{PORT}')),daemon=True).start()
    app.run(debug=False,port=PORT,host='127.0.0.1')