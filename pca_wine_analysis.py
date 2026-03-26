import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14
})

# Load dataset
data = load_wine()
X = data.data
y = data.target
feature_names = data.feature_names

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Covariance matrix
cov_matrix = np.cov(X_scaled.T)

# Eigen decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues and eigenvectors
sorted_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_idx]
eigenvectors = eigenvectors[:, sorted_idx]

# Project data onto first 2 principal components
W = eigenvectors[:, :2]
X_pca_manual = X_scaled @ W

# PCA using sklearn
pca = PCA(n_components=2)
X_pca_sklearn = pca.fit_transform(X_scaled)

# Explained variance
explained_variance = eigenvalues / np.sum(eigenvalues)

plt.figure(figsize=(7,6))
for i in range(3):
    plt.scatter(
        X_pca_manual[y == i, 0],
        X_pca_manual[y == i, 1],
        s=60,   # bigger points
        label=f"Class {i}"
    )
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA (Manual Implementation)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("pca_manual.png", dpi=300)
plt.show()

plt.figure(figsize=(7,6))
for i in range(3):
    plt.scatter(
        X_pca_sklearn[y == i, 0],
        X_pca_sklearn[y == i, 1],
        s=60,
        label=f"Class {i}"
    )
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA (Scikit-learn)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("pca_sklearn.png", dpi=300)
plt.show()

plt.figure(figsize=(7,5))
plt.plot(
    range(1, len(explained_variance)+1),
    explained_variance,
    'o-',
    markersize=7,
    linewidth=2
)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Explained Variance by Components")
plt.grid()
plt.tight_layout()
plt.savefig("explained_variance.png", dpi=300)
plt.show()

# Print variance info
print("Explained variance (first 5 components):")
for i in range(5):
    print(f"PC{i+1}: {explained_variance[i]:.4f}")
