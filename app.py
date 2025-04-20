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

# Define a custom colormap (e.g., "viridis", "plasma", "inferno")
colors = plt.cm.viridis(np.linspace(0, 1, num_clusters))  # Using the 'viridis' colormap

# Plot clusters without centroids
plt.figure(figsize=(8, 6))
for i in range(num_clusters):
    plt.scatter(X_pca[y_kmeans == i, 0], X_pca[y_kmeans == i, 1], c=[colors[i]], s=50, label=f"Cluster {i}")

# Add title and labels
plt.title(f'Clusters (2D PCA Projection)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')

# Add color legend
plt.legend()
st.pyplot(plt)
