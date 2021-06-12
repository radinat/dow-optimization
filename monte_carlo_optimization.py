# -*- coding: utf-8 -*-
"""
Created on Sun May 16 16:47:38 2021

@author: radina
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from pandas_datareader import data as wb  #module has to be installed
import scipy.optimize as sco
import scipy.interpolate as sci
import sys
import seaborn as sns
# set print options
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)


# Plot prices
(data / data.iloc[0] * 100).plot(figsize=(15,10))
#TSLA has surged in 2020

# Calculate log returns and other statistics
rets = np.log(data / data.shift(1))
rets_benchmark = np.log(snp / snp.shift(1))

rets=rets.iloc[1:]
rets_benchmark=rets_benchmark.iloc[1:]


plt.figure(figsize=(14, 7))
for c in rets.columns.values:
    plt.plot(rets.index, rets[c], lw=3, alpha=0.8,label=c)
plt.legend(loc='upper right', fontsize=12)
plt.ylabel('daily returns')

# Check metrics
rets.mean() * 252 # 252 trading days - annualize daily returns
rets.cov() * 252
rets.corr()

# generate random weights for every security and normalize them
weights = np.random.random(noa)
weights /= np.sum(weights)
weights

# calculate expected portfolio return from annualized return values
np.sum(rets.mean() * weights) * 252
# calculate expected portfolio variance
np.dot(weights.T, np.dot(rets.cov() * 252, weights))
# calculate expected portfolio standard deviation
np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))

# Monte Carlo simulation to generate random portfolio weights
# For every simulated allocation, we record the resulting expected portfolio return and variance
prets = []
pvols = []
for p in range (2500): 
 weights = np.random.random(noa)
 weights /= np.sum(weights)
 prets.append(np.sum(rets.mean() * weights) * 252)
 pvols.append(np.sqrt(np.dot(weights.T,
  np.dot(rets.cov() * 252, weights))))

prets = np.array(prets)
pvols = np.array(pvols)

# Plot the results from the simulation
plt.figure(figsize=(8, 4))
plt.scatter(pvols, prets, c=prets / pvols, marker='o')
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')

# a function for major portfolio statistics
def statistics(weights):
 weights = np.array(weights)
 pret = np.sum(rets.mean() * weights) * 252
 pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
 return np.array([pret, pvol, pret / pvol])


# function for minimizing the negative value of the Sharpe ration
def min_func_sharpe(weights):
 return -statistics(weights)[2]

cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) #constraint - all parameters add up to 1
bnds = tuple((0, 1) for x in range(noa)) # bound parameter values
noa * [1. / noa,] # initial guess of the weights - equal distribution

# call the function
opts = sco.minimize(min_func_sharpe, noa * [1. / noa,], method='SLSQP',
bounds=bnds, constraints=cons)
opts
# optimal portfolio composition results
opts['x'].round(3)
# calculate statistics
statistics(opts['x']).round(3)

# define a function that minimizes the variance
def min_func_variance(weights):
 return statistics(weights)[1] ** 2
# call the minimizing function
optv = sco.minimize(min_func_variance, noa * [1. / noa,],
method='SLSQP', bounds=bnds,
constraints=cons)
# absolute minimum variance portfolio results:
optv['x'].round(3)
# calculate statistics
statistics(optv['x']).round(3)

# Derive Efficient frontier
cons = ({'type': 'eq', 'fun': lambda x: statistics(x)[0] - tret},
{'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0, 1) for x in weights)

# define minimization function
def min_func_port(weights):
 return statistics(weights)[1]

# define target returns levels
trets = np.linspace(0.0, 0.25, 50)
tvols = [] #condition dictionary
# iterate over target returns levels
for tret in trets:
 cons = ({'type': 'eq', 'fun': lambda x: statistics(x)[0] - tret},
{'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
 res = sco.minimize(min_func_port, noa * [1. / noa,], method='SLSQP',
 bounds=bnds, constraints=cons)
 tvols.append(res['fun'])
tvols = np.array(tvols)

# Plot optimization results
plt.figure(figsize=(8, 4))
plt.scatter(pvols, prets, c=prets / pvols, marker='o')
# random portfolio composition
plt.scatter(tvols, trets,
c=trets / tvols, marker='x')
# efficient frontier
plt.plot(statistics(opts['x'])[1], statistics(opts['x'])[0],
'r*', markersize=15.0)
# portfolio with highest Sharpe ratio
plt.plot(statistics(optv['x'])[1], statistics(optv['x'])[0],
'y*', markersize=15.0)
# minimum variance portfolio
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')

# Capital market line

# select only portfolios from the efficient frontier
ind = np.argmin(tvols)
evols = tvols[ind:]
erets = trets[ind:]
# interpolation 
tck = sci.splrep(evols, erets)
# define function for the efficient frontier and the respective first derivative function
def f(x):
 "' Efficient frontier function (splines approximation). '"
 return sci.splev(x, tck, der=0)
def df(x):
 "' First derivative of efficient frontier function. '"
 return sci.splev(x, tck, der=1)

# function for mathematical conditions for the capital market line
def equations(p, rf=0.01):
 eq1 = rf - p[0]
 eq2 = rf + p[1] * p[2] - f(p[2])
 eq3 = p[1] - df(p[2])
 return eq1, eq2, eq3
# solve the equations function 
opt = sco.fsolve(equations, [0.01, 0.5, 0.15])
# results from the numerical optimization
opt
# equations' results
np.round(equations(opt), 6)

# plot optimal portfolio
plt.figure(figsize=(8, 4))
plt.scatter(pvols, prets,
c=(prets - 0.01) / pvols, marker='o')
# random portfolio composition
plt.plot(evols, erets, 'g', lw=4.0)
# efficient frontier
cx = np.linspace(0.0, 0.3)
plt.plot(cx, opt[0] + opt[1] * cx, lw=1.5)
# capital market line
plt.plot(opt[2], f(opt[2]), 'r*', markersize=15.0)
plt.grid(True)
plt.axhline(0, color='k', ls='-', lw=2.0)
plt.axvline(0, color='k', ls='-', lw=2.0)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')

# optimal portfolio weights
cons = ({'type': 'eq', 'fun': lambda x: statistics(x)[0] - f(opt[2])},
{'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
res = sco.minimize(min_func_port, noa * [1. / noa,], method='SLSQP',
bounds=bnds, constraints=cons)
# optimal portfolio weights print
res['x'].round(3)
portfolio=pd.DataFrame()
portfolio['assets']=data.columns
portfolio['weights']=res['x'].round(3)
portfolio=portfolio.drop(1)
portfolio=portfolio.drop(2)

fig = plt.figure(figsize =(10, 7))
plt.pie(portfolio['weights'], labels = portfolio['assets'], autopct='%1.1f%%', startangle = 90)
plt.show() 
# Check statistics
mc_stat=statistics(res['x'].round(3))
mc_stat
aa=res['x']

# Alpha and Beta

optPort_ret = (rets * res['x']).sum(axis = 1)


# plot optimal portfolio returns vs benchmark returns
sns.regplot(rets_benchmark.values,
optPort_ret.values)
plt.xlabel("Benchmark Returns")
plt.ylabel("Portfolio Returns")
plt.title("Portfolio Returns vs Benchmark Returns")
plt.show()

#We can see that our portfolio returns are highly correlated to the benchmark returns. 
#We can use the regression model to calculate the portfolio beta and the portfolio alpha. 
#We will us the linear regression model to calculate the alpha and the beta.
from scipy import stats
benchmark_ret = rets_benchmark.squeeze()
(beta, alpha) = stats.linregress(benchmark_ret.values,
                optPort_ret.values)[0:2]

#print beta
print("The portfolio beta is", round(beta, 4)) #0.9016
print("The portfolio alpha is", round(alpha,5))
