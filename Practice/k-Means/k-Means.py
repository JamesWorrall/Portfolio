import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#Import Data
mall_df = pd.read_csv('Mall_Customers.csv')
mall_stats = mall_df.to_numpy()

#Create Useful Variables
sps = (mall_stats[:, 4])/(mall_stats[:, 3]) #sps = Score Per Salary
age = mall_stats[:, 2]

#Transform Variables
sps = sps.tolist()
age = age.tolist()
data = tuple(zip(sps, age))

#Scale Variables
scaler = StandardScaler()
sd = scaler.fit_transform(data) #sd = Scaled Data

#Initiate and Fit Model
km = KMeans(
    init = 'random',
    n_clusters = 3,
    n_init = 10,
    max_iter = 300,
    random_state = 42
    )

km.fit(sd)

#Find Ideal Number of Clusters
km_kwargs = {
    'init': 'random',
    'n_init': 10,
    'max_iter': 300,
    'random_state': 42
    }

sse = []
for k in range(1,11):
    km = KMeans(n_clusters = k, **km_kwargs)
    km.fit(sd)
    sse.append(km.inertia_)

#Plot SSE by Number of Clusters
plt.style.use('fivethirtyeight')
plt.plot(range(1,11), sse)
plt.xticks(range(1,11))
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.show()

#Find Ideal Number of Clusters According to Curve Plotted Above
kl = KneeLocator(
    range(1,11), sse, curve = 'convex', direction = 'decreasing'
    )
print(kl.elbow)

#Find Ideal Number of Clusters According to Silhouette Coefficients
sc = [] #sc = Silhouette Coefficients
for k in range(2, 11):
    km = KMeans(n_clusters = k, **km_kwargs)
    km.fit(sd)
    score = silhouette_score(sd, km.labels_)
    sc.append(score)

plt.style.use('fivethirtyeight')
plt.plot(range(2,11), sc)
plt.xticks(range(2,11))
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficients')
plt.show()

#Plot Results of Tuned Model
km = KMeans(n_clusters = kl.elbow, **km_kwargs)
km.fit(sd)
y_km = km.predict(sd)
plt.scatter(*zip(*sd), c = y_km, s = 50, cmap = 'viridis')
centers = km.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c = 'black', s = 200, alpha = 0.5)
plt.xlabel('Spending Score per $1k Salary (Scaled)')
plt.ylabel('Age (Scaled)')
plt.show()
