# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 15:06:42 2025

@author: yoyop
"""


# app.py
import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

# Set Streamlit page configuration
st.set_page_config(page_title="K-Means Clustering App", layout="centered")

# Title
st.title("🔍 K-Means Clustering App with Iris Dataset")


# Sidebar for user interaction
st.sidebar.header("Configure Clustering")
num_clusters = st.sidebar.slider("Select number of Clusters", min_value=2, max_value=10, value=3)

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Apply PCA for 2D projection
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
y_kmeans = kmeans.fit_predict(X_pca)

# Plot clusters
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, s=50, cmap='viridis')

# Add cluster labels
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5, marker='o', label="Centroids")

# Add title and labels
plt.title(f'Clusters (2D PCA Projection) - Iris Dataset')
plt.xlabel('PCA1')
plt.ylabel('PCA2')

# Add color legend
plt.legend()
plt.colorbar(scatter)
st.pyplot(plt)
