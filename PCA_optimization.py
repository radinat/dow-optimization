# -*- coding: utf-8 -*-
"""
Created on Sun May 16 16:49:13 2021

@author: radina
"""

# PCA 
from sklearn.decomposition import PCA 
# check covariance
rets_cov=rets.cov()
rets=rets.iloc[1:,:]
# find optimal number of components
# PCA
pca = PCA()
x_pca = pca.fit_transform(rets)
plt.figure(figsize=(10,10))
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')
plt.grid()

pca = PCA(n_components=2) 
X = pca.fit_transform(rets)
sum(pca.explained_variance_ratio_)
print('explained variance ratio (first seven components): %s' 
      % str(sum(pca.explained_variance_ratio_))) 

pca.explained_variance_ratio_


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

#pc_ret_idx['indu'] = indu_index[1:]
pc_ret_idx.plot(subplots=True,title ='PC portfolio vs Market',layout =[1,2])

weights_df = pd.DataFrame(data = pc_w*100,index = rets.columns)
weights_df.columns=['weights']
weights_df.plot.bar(title='PCA portfolio weights',rot =45,fontsize =8)
