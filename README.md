# PCA Analysis on Wine Dataset

This project demonstrates Principal Component Analysis (PCA) for dimensionality reduction on a real-world dataset. PCA is implemented both from first principles (using covariance matrix and eigenvalue decomposition) and using scikit-learn for validation.

The workflow includes data preprocessing, feature standardization, projection onto principal components, and analysis of explained variance. Visualizations are used to compare class separation and evaluate the effectiveness of dimensionality reduction.

## Features
- Data standardization
- Manual PCA implementation using eigen decomposition
- PCA using scikit-learn
- Comparison of both approaches
- Visualization of principal components
- Explained variance analysis

## Results

### PCA (Manual)
![Manual PCA](results/pca_manual.png)

### PCA (Scikit-learn)
![Sklearn PCA](results/pca_sklearn.png)

### Explained Variance
![Variance](results/explained_variance.png)

## Tools
- Python
- NumPy
- Matplotlib
- Scikit-learn
