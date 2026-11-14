"""Public portfolio dashboard adapted from Advanced Multi-Stock Research notebook."""
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor

st.set_page_config(page_title="Portfolio Research Dashboard", page_icon="📈", layout="wide")
TRADING_DAYS = 252
FEATURES = ["Return_1D","Return_5D","Return_20D","Return_60D","RSI_14","MACD","MACD_Signal","MACD_Hist","ATR_Pct","Vol_20D","Vol_60D","Volume_Ratio","BB_Position","Drawdown","52W_Position","Above_SMA_20","Above_SMA_50","Above_SMA_200"]

def rsi(series, period=14):
    delta=series.diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
    avg_gain=gain.ewm(alpha=1/period,adjust=False,min_periods=period).mean()
    avg_loss=loss.ewm(alpha=1/period,adjust=False,min_periods=period).mean()
    return 100-100/(1+avg_gain/avg_loss.replace(0,np.nan))

def indicators(df):
    d=df.copy(); c=d['Close']; d['Return_1D']=c.pct_change()
    for p in [5,20,60]: d[f'Return_{p}D']=c.pct_change(p)
    for p in [20,50,200]: d[f'SMA_{p}']=c.rolling(p).mean()
    d['EMA_12']=c.ewm(span=12,adjust=False).mean(); d['EMA_26']=c.ewm(span=26,adjust=False).mean()
    d['MACD']=d.EMA_12-d.EMA_26; d['MACD_Signal']=d.MACD.ewm(span=9,adjust=False).mean(); d['MACD_Hist']=d.MACD-d.MACD_Signal
    d['RSI_14']=rsi(c)
    tr=pd.concat([d.High-d.Low,(d.High-c.shift()).abs(),(d.Low-c.shift()).abs()],axis=1).max(axis=1)
    d['ATR_Pct']=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()/c
    mid=c.rolling(20).mean(); sd=c.rolling(20).std(); d['BB_Position']=(c-(mid-2*sd))/(4*sd)
    d['Vol_20D']=d.Return_1D.rolling(20).std()*np.sqrt(TRADING_DAYS)
    d['Vol_60D']=d.Return_1D.rolling(60).std()*np.sqrt(TRADING_DAYS)
    d['Volume_Ratio']=d.Volume/d.Volume.rolling(20).mean()
    d['Drawdown']=c/c.cummax()-1
    hi=c.rolling(252,min_periods=20).max(); lo=c.rolling(252,min_periods=20).min(); d['52W_Position']=(c-lo)/(hi-lo)
    for p in [20,50,200]: d[f'Above_SMA_{p}']=(c>d[f'SMA_{p}']).astype(int)
    for p in [5,20]: d[f'Target_{p}D']=c.shift(-p)/c-1
    return d

def risk_score(d):
    x=d.iloc[-1]
    vol=np.clip((x.Vol_20D-.20)/.80*100,0,100)
    dd=np.clip(abs(min(x.Drawdown,0))/.60*100,0,100)
    atr=np.clip((x.ATR_Pct-.015)/.10*100,0,100)
    vr=x.Volume_Ratio
    liq=np.clip((1/max(vr,.05)-.5)/4*100,0,100) if np.isfinite(vr) else 0
    rv=x.RSI_14; rw=100 if rv>=80 or rv<=20 else (50 if rv>=70 or rv<=30 else 0)
    tw=100 if x.Close<x.SMA_50<x.SMA_200 else (50 if x.Close<x.SMA_50 else 0)
    return .30*vol+.25*dd+.15*atr+.10*liq+.10*rw+.10*tw

def forecasts(d):
    out={}
    for horizon in [5,20]:
        target=f'Target_{horizon}D'
        train=d[FEATURES+[target]].replace([np.inf,-np.inf],np.nan).dropna()
        latest=d[FEATURES].replace([np.inf,-np.inf],np.nan).dropna()
        if len(train)<260 or latest.empty: out[horizon]=np.nan;continue
        # Exclude the most recent horizon from training to prevent future-target leakage.
        split=int(len(train)*.8)
        model=HistGradientBoostingRegressor(max_iter=120,learning_rate=.05,max_leaf_nodes=15,l2_regularization=1,random_state=42)
        model.fit(train[FEATURES].iloc[:split],train[target].iloc[:split])
        out[horizon]=float(model.predict(latest[FEATURES].iloc[[-1]])[0])
    return out

@st.cache_data(ttl=900,show_spinner=False)
def load_history(ticker):
    frame=yf.download(ticker,period='5y',interval='1d',auto_adjust=True,progress=False,threads=False)
    if frame.empty: raise ValueError('No market data returned')
    if isinstance(frame.columns,pd.MultiIndex): frame.columns=frame.columns.get_level_values(0)
    if not {'Open','High','Low','Close','Volume'}.issubset(frame.columns): raise ValueError('Missing price columns')
    frame=frame[['Open','High','Low','Close','Volume']].dropna(subset=['Close']).copy()
    if len(frame)<65: raise ValueError('Insufficient price history (at least 65 trading days required)')
    frame.index=pd.to_datetime(frame.index).tz_localize(None) if frame.index.tz is not None else pd.to_datetime(frame.index)
    return frame

@st.cache_data(ttl=3600,show_spinner=False)
def cached_forecasts(ticker):
    return forecasts(indicators(load_history(ticker)))

def max_drawdown(r):
    wealth=(1+r).cumprod();return float((wealth/wealth.cummax()-1).min())

def var95(r):
    return float(-r.quantile(.05))

st.title('📈 Personal Portfolio Research Dashboard')
st.caption('Enter your positions and select Analyze Portfolio. Your holdings are used for this session only; this app does not intentionally store them in a database.')
st.info('Research and education only — not personalized investment advice. Market data may be delayed or incomplete. Forecasts and scenarios are uncertain, not guarantees.')
if 'holdings' not in st.session_state:
    st.session_state.holdings=pd.DataFrame([{'Ticker':'NVDA','Shares':25.0,'Average cost per share':120.0},{'Ticker':'KO','Shares':50.0,'Average cost per share':62.0}])
with st.form('holdings_form'):
    name=st.text_input('Portfolio name','My Portfolio',max_chars=80)
    holdings=st.data_editor(st.session_state.holdings,num_rows='dynamic',hide_index=True,use_container_width=True,
        column_config={'Ticker':st.column_config.TextColumn('Ticker (e.g. AAPL)',required=True,max_chars=15),
                       'Shares':st.column_config.NumberColumn('Shares',min_value=0.000001,format='%.4f',required=True),
                       'Average cost per share':st.column_config.NumberColumn('Average Purchase Price ($/Share)', min_value=0.,format='$%.2f', required=True, help="Enter your average purchase price per share."
)},key='holdings_editor')
    run=st.form_submit_button('Analyze Portfolio',type='primary',use_container_width=True)
if run:
    clean=holdings.copy();clean['Ticker']=clean['Ticker'].fillna('').astype(str).str.strip().str.upper()
    clean=clean[clean.Ticker!=''].copy()
    clean['Shares']=pd.to_numeric(clean['Shares'],errors='coerce')
    clean['Average cost per share']=pd.to_numeric(clean['Average cost per share'],errors='coerce')
    if clean.empty or len(clean)>20 or not clean.Ticker.str.fullmatch(r'[A-Z0-9^.=\-]{1,15}').all() or clean[['Shares','Average cost per share']].isna().any().any() or (clean.Shares<=0).any() or (clean['Average cost per share']<0).any().any():
        st.error('Enter 1–20 valid ticker symbols, positive share counts and nonnegative average costs.')
    else:
        clean['Extended cost']=clean['Shares']*clean['Average cost per share']
        clean=clean.groupby('Ticker',as_index=False).agg({'Shares':'sum','Extended cost':'sum'})
        clean['Average cost per share']=clean['Extended cost']/clean['Shares']
        st.session_state.positions=clean[['Ticker','Shares','Average cost per share']].copy()
        st.session_state.portfolio_name=name
if 'positions' not in st.session_state:
    st.stop()
positions=st.session_state.positions
histories={};bad={}
with st.spinner('Downloading stock prices and preparing your dashboard…'):
    for t in positions.Ticker:
        try: histories[t]=load_history(t)
        except Exception as e:bad[t]=str(e)
if bad:st.warning('Skipped unavailable tickers: '+', '.join(f'{t} ({reason})' for t,reason in bad.items()))
if not histories:st.error('No valid market data was available. Check the symbols and try again.');st.stop()
positions=positions[positions.Ticker.isin(histories)].copy()
last_prices={t:float(d.Close.iloc[-1]) for t,d in histories.items()}
positions['Latest price']=positions.Ticker.map(last_prices)
positions['Cost basis']=positions.Shares*positions['Average cost per share']
positions['Market value']=positions.Shares*positions['Latest price']
positions['Unrealized P/L']=positions['Market value']-positions['Cost basis']
positions['Unrealized P/L %']=np.where(positions['Cost basis']>0,positions['Unrealized P/L']/positions['Cost basis']*100,np.nan)
total=float(positions['Market value'].sum());basis=float(positions['Cost basis'].sum());pnl=total-basis
positions['Allocation %']=positions['Market value']/total*100
st.header(st.session_state.portfolio_name)
a,b,c,d=st.columns(4)
a.metric('Current market value',f'${total:,.2f}')
b.metric('Total cost basis',f'${basis:,.2f}')
c.metric('Unrealized gain/loss',f'${pnl:,.2f}',delta=f'{pnl/basis:.2%}' if basis>0 else None)
d.metric('Holdings',len(positions))
st.caption('Latest available daily close; not a live executable quote. Cost basis uses the average price you entered and excludes fees, taxes and dividends.')
st.subheader('Your holdings')
st.dataframe(positions.style.format({'Shares':'{:,.4f}','Average cost per share':'${:,.2f}','Latest price':'${:,.2f}','Cost basis':'${:,.2f}','Market value':'${:,.2f}','Unrealized P/L':'${:,.2f}','Unrealized P/L %':'{:+.2f}%','Allocation %':'{:.2f}%'}),hide_index=True,use_container_width=True)
left,right=st.columns(2)
with left:st.plotly_chart(px.pie(positions,names='Ticker',values='Market value',title='Allocation by market value',hole=.45),use_container_width=True)
with right:st.plotly_chart(px.bar(positions,x='Ticker',y='Unrealized P/L',title='Unrealized gain / loss ($)'),use_container_width=True)
close=pd.concat({t:df.Close for t,df in histories.items()},axis=1).sort_index().ffill()
returns=close.pct_change(fill_method=None).dropna(how='any')
if len(returns)>=30:
    # Historical proxy: constant weights equal to today's allocations, rebalanced daily.
    weights=positions.set_index('Ticker')['Allocation %'].reindex(returns.columns).div(100)
    portfolio_returns=returns.dot(weights)
    vol=float(portfolio_returns.std()*np.sqrt(TRADING_DAYS));ann=float(portfolio_returns.mean()*TRADING_DAYS)
    sharpe=ann/vol if vol>0 else np.nan
    tail=portfolio_returns[portfolio_returns<=portfolio_returns.quantile(.05)]
    st.subheader('Historical portfolio risk (constant-current-weight proxy)')
    cols=st.columns(5)
    for col,label,value in zip(cols,['Annualized return','Annualized volatility','Sharpe (rf=0)','Maximum drawdown','Daily historical VaR 95%'],[f'{ann:.1%}',f'{vol:.1%}',f'{sharpe:.2f}',f'{max_drawdown(portfolio_returns):.1%}',f'{var95(portfolio_returns):.1%}']):col.metric(label,value)
    st.caption('These are historical estimates using current portfolio weights rebalanced daily, not your actual historical account returns. VaR is a daily return-loss threshold, not a maximum possible loss.')
    growth=(1+portfolio_returns).cumprod();st.plotly_chart(px.line(x=growth.index,y=growth.values,title='Historical growth of $1 (constant-current-weight proxy)',labels={'x':'Date','y':'Growth multiple'}),use_container_width=True)
    st.plotly_chart(px.imshow(returns.corr(),text_auto='.2f',title='Daily return correlation'),use_container_width=True)
    rng=np.random.default_rng(42);simulations=2000;horizon=60
    mu=float(portfolio_returns.mean());sigma=float(portfolio_returns.std())
    shocks=rng.normal(mu,sigma,size=(horizon,simulations));terminal=total*np.prod(np.maximum(1+shocks,0),axis=0)
    st.subheader('60-trading-day Monte Carlo scenarios')
    q=np.percentile(terminal,[5,25,50,75,95]);st.plotly_chart(px.histogram(x=terminal,nbins=60,title='Simulated ending portfolio values',labels={'x':'Simulated portfolio value ($)'}),use_container_width=True)
    st.write(f'5th percentile: **${q[0]:,.0f}** · Median: **${q[2]:,.0f}** · 95th percentile: **${q[4]:,.0f}**')
    st.caption('Illustrative normal-return simulation calibrated to historical daily portfolio returns. It does not model fat tails, transaction costs, deposits, withdrawals or changing weights.')
else:st.warning('Not enough overlapping history for portfolio-level risk and Monte Carlo charts.')
st.subheader('Technical research and observed risk monitor')
research=[]
for ticker,history in histories.items():
    d=indicators(history);x=d.iloc[-1]
    research.append({'Ticker':ticker,'RSI (14)':x.RSI_14,'MACD':x.MACD,'20D volatility %':x.Vol_20D*100,'Observed risk score (0–100)':risk_score(d)})
research=pd.DataFrame(research)
st.dataframe(research.style.format({'RSI (14)':'{:.1f}','MACD':'{:.2f}','20D volatility %':'{:.1f}','Observed risk score (0–100)':'{:.1f}'}),hide_index=True,use_container_width=True)
st.caption('The observed risk score follows the notebook’s volatility, drawdown, ATR, volume, RSI and trend weighting. It is a monitoring heuristic, not a recommendation.')
selected=st.selectbox('Inspect a stock',list(histories))
d=indicators(histories[selected]);lookback=st.slider('Chart trading days',60,min(1000,len(d)),min(252,len(d)))
fig=go.Figure();view=d.tail(lookback)
fig.add_trace(go.Scatter(x=view.index,y=view.Close,name='Close'))
for p in [20,50,200]:fig.add_trace(go.Scatter(x=view.index,y=view[f'SMA_{p}'],name=f'SMA {p}'))
fig.update_layout(title=f'{selected}: price and moving averages',xaxis_title='Date',yaxis_title='Price ($)')
st.plotly_chart(fig,use_container_width=True)
with st.expander('Optional machine-learning forecasts (may take time)'):
    if st.button(f'Estimate {selected} forward returns'):
        with st.spinner('Training historical models…'):
            try:
                estimates=cached_forecasts(selected)
                st.write('5-trading-day return estimate:',f'{estimates[5]:+.2%}' if np.isfinite(estimates[5]) else 'Insufficient history')
                st.write('20-trading-day return estimate:',f'{estimates[20]:+.2%}' if np.isfinite(estimates[20]) else 'Insufficient history')
            except Exception as e:st.error(f'Forecast unavailable: {e}')
        st.caption('Adapted from the notebook’s gradient-boosting return model. Predictions are exploratory and not validated for trading use.')
st.download_button('Download holdings and results (CSV)',positions.to_csv(index=False).encode(),file_name='portfolio_results.csv',mime='text/csv')
