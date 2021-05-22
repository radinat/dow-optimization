# -*- coding: utf-8 -*-
"""
Created on Sun May 16 16:49:13 2021

@author: radina
"""

# PCA 
# https://thequantmba.wordpress.com/2017/01/24/principal-component-analysis-of-equity-returns-in-python/?fbclid=IwAR0m7eKN4DP8jQZBLRBXJHPyTi32oV57nXMlIq77jHMNE5yRMHXZZPR1lEY

from sklearn.decomposition import PCA 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# find optimal number of components
# PCA
pca = PCA()
x_pca = pca.fit_transform(rets)
plt.figure(figsize=(10,10))
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')
plt.grid()
# PCA with 3 components
pca = PCA(n_components=3) 
X = pca.fit_transform(rets)
sum(pca.explained_variance_ratio_)
print('explained variance ratio (first three components): %s' 
      % str(sum(pca.explained_variance_ratio_))) 
# explained variance by each component
pca.explained_variance_ratio_

# plot information content
fig = plt.figure(figsize=(12,6))
fig.add_subplot(1,2,1)
plt.bar(np.arange(pca.n_components_), 100 * pca.explained_variance_ratio_)
plt.title('Relative information content of PCA components')
plt.xlabel("PCA component number")
plt.ylabel("PCA component variance %")

pcs =pca.components_
pc1 = pcs[0,:]
# normalized to 1 
pc_w = np.asmatrix(pc1/sum(pc1)).T
pc1_ret = rets.values*pc_w

# plot the total return index of the first PC portfolio
pc_ret = pd.DataFrame(data =pc1_ret, index= rets.index)
pc_ret_idx = pc_ret+1
pc_ret_idx= pc_ret_idx.cumprod()
pc_ret_idx.columns =['pc1']
#plot PC portfolio vs market
pc_ret_idx.plot(subplots=True,title ='PC portfolio vs Market')

#portfolio weights
weights_df = pd.DataFrame(data = pc_w*100,index = rets.columns)
weights_df.columns=['weights']
weights_df.plot.bar(title='PCA portfolio weights',rot =45,fontsize =8)
weights_df


# Alpha and Beta
# transform to array
weights_arr=weights_arr.ravel()
weights_arr=weights_df.to_numpy()
#calculate optimal portfolio returns
optPortPC_ret = (rets * weights_arr).sum(axis = 1)
optPortPC_ret=optPortPC_ret.iloc[1:]
rets_benchmark=rets_benchmark.iloc[1:]

# plot optimal portfolio returns vs benchmark returns
sns.regplot(rets_benchmark.values,
optPortPC_ret.values)
plt.xlabel("Benchmark Returns")
plt.ylabel("Portfolio Returns")
plt.title("Portfolio Returns vs Benchmark Returns")
plt.show()


from scipy import stats
benchmark_ret = rets_benchmark.squeeze()
(beta, alpha) = stats.linregress(benchmark_ret.values,
                optPortPC_ret.values)[0:2]

#print beta
print("The portfolio beta is", round(beta, 4)) #103.6358
#We can see that this portfolio had a negative alpha. 
#The portfolio beta was 0.93. 
#This suggests that for every +1% move in the S&P 500 our portfolio will go up 0.93% in value.
print("The portfolio alpha is", round(alpha,5))
