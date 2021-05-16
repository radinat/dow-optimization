# -*- coding: utf-8 -*-
"""
Created on Sun May 16 16:24:43 2021

@author: radina
"""
import pandas as pd
import yfinance as yf
from pandas_datareader import data as wb  #module has to be installed

#data = pd.read_csv("assets_data.csv", index_col=0)
# Import historical data

symbols = ['AXP', 'AMGN', 'AAPL', 'BA','CAT', 'CSCO', 'CVX',
           'GS','HD','HON','IBM','INTC','JNJ','KO','JPM',
           'MCD','MMM','MRK','MSFT','NKE','PG','TRV','UNH',
           'CRM','VZ','V','WBA','WMT','DIS']
dowSym = ['^DJI']

noa = len(symbols)
data = pd.DataFrame()
dow = pd.DataFrame()

for sym in symbols:
  data[sym] = wb.DataReader(sym, data_source='yahoo',
  start='2011-01-01', end='2021-01-01')['Adj Close']

for sym in dowSym:
  dow[sym] = wb.DataReader(sym, data_source='yahoo',
  start='2011-01-01', end='2021-01-01')['Adj Close']
  
data.isna().sum()
  
data.index=data.index.astype('datetime64[ns]')
dow.index=dow.index.astype('datetime64[ns]')

#data.to_csv(r'C:\Users\radina\Desktop\MASTERS\Python_for_Finance\assets_data.csv', index = True)
