from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the digits dataset
digits = load_digits()
print("Original shape:", digits.data.shape)  # (1797, 64)

# ---------------------- PCA --------------------------
pca = PCA(n_components=2)
pca_components = pca.fit_transform(digits.data)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=digits.target, palette='tab10', legend='full')
plt.title("📉 PCA - Digits Visualization (2D)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

# ---------------------- t-SNE --------------------------
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
tsne_components = tsne.fit_transform(digits.data)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=tsne_components[:, 0], y=tsne_components[:, 1], hue=digits.target, palette='tab10', legend='full')
plt.title("🎯 t-SNE - Digits Visualization (2D)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()
