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

symbols = [ 'AAPL','IBM', 'JPM',
           'MCD','MSFT','NKE', 'GC=F']
snpSym = ['^GSPC']

noa = len(symbols)
data = pd.DataFrame()
snp = pd.DataFrame()

for sym in symbols:
  data[sym] = wb.DataReader(sym, data_source='yahoo',
  start='2011-01-01', end='2021-01-01')['Adj Close']

for sym in snpSym:
  snp[sym] = wb.DataReader(sym, data_source='yahoo',
  start='2011-01-01', end='2021-01-01')['Adj Close']
  
data.isna().sum()
data = data.fillna(method='ffill') #fill na with last observation carried forward

#rename gold column
data = data.rename(columns={'GC=F': 'GOLD'})
#change index types to datetime
data.index=data.index.astype('datetime64[ns]')
snp.index=snp.index.astype('datetime64[ns]')

#data.to_csv(r'C:\Users\radina\Desktop\MASTERS\Techniques/assets_data.csv', index = True)
