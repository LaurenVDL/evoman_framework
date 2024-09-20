# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 16:38:55 2024

@author: charl
"""

from sklearn.cluster import DBSCAN
import numpy as np

# Example dataset
X = np.array([[1, 2], [2, 4], [2, 3],
              [8, 1], [10000, 1000], [25, 80]])

# Create DBSCAN model
dbscan = DBSCAN(eps=3, min_samples=2)

# Fit the model and get the labels
dbscan.fit(X)
labels = dbscan.labels_

# Get the number of clusters, ignoring noise (label = -1)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print(f'Number of clusters: {n_clusters}')