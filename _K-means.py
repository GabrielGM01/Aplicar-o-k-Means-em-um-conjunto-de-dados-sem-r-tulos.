#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt

dt = pd.read_csv("Wholesale customers data.csv")
x = dt.iloc[:, 0:8].values
K = [2,3,4,5,6,7,8,9,10]


# In[2]:


sil = []
kmax = 10

for k in range(2, kmax+1):
  kmeans = KMeans(n_clusters = k).fit(x)
  labels = kmeans.labels_
  sil.append(silhouette_score(x, labels, metric = 'euclidean'))

print(sil)


# In[3]:


plt.xlabel("Valor de K")
plt.ylabel("Pontuação")
xpoints = np.array(K)
ypoints = np.array(sil)

plt.plot(xpoints, ypoints)
plt.show()


