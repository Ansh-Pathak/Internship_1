import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (silhouette_score, silhouette_samples,
                              davies_bouldin_score, calinski_harabasz_score,
                              adjusted_rand_score, confusion_matrix)
from scipy.cluster.hierarchy import dendrogram, linkage

OUT_DIR = "charts"
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.size"] = 12
PALETTE = sns.color_palette("Set2", 8)

data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
true_labels = data.target
target_names = data.target_names

print(df.shape)
print(df.isnull().sum().sum())
print(df.describe().T[["mean", "std", "min", "max"]])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.values)
X_scaled_df = pd.DataFrame(X_scaled, columns=df.columns)

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, orient="h", ax=ax, palette="Set2")
ax.set_title("Feature Scales Before Standardization", fontsize=15, fontweight="bold")
ax.set_xlabel("Raw Value")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/01_raw_feature_scales.png")
plt.close(fig)

corr = df.corr()
fig, ax = plt.subplots(figsize=(10, 8.5))
sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f",
            square=True, linewidths=0.5, cbar_kws={"label": "Pearson r"}, ax=ax)
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=12)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/02_feature_correlation.png")
plt.close(fig)

inertias = []
sil_scores = []
k_range = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels_k = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels_k))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(list(k_range), inertias, marker="o", color="#3B6E8F", linewidth=2)
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia (Within-Cluster SSE)")
axes[0].axvline(3, color="crimson", linestyle="--", linewidth=1.3, label="Chosen k=3")
axes[0].legend()

axes[1].plot(list(k_range), sil_scores, marker="o", color="#A64D79", linewidth=2)
axes[1].set_title("Silhouette Score by k")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Average Silhouette Score")
axes[1].axvline(3, color="crimson", linestyle="--", linewidth=1.3, label="Chosen k=3")
axes[1].legend()
fig.suptitle("Selecting the Number of Clusters", y=1.03, fontsize=15, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/03_elbow_silhouette.png")
plt.close(fig)

print("Inertias:", dict(zip(k_range, inertias)))
print("Silhouette scores:", dict(zip(k_range, [round(s, 3) for s in sil_scores])))

K = 3
kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
km_labels = kmeans.fit_predict(X_scaled)

sil_avg = silhouette_score(X_scaled, km_labels)
db_score = davies_bouldin_score(X_scaled, km_labels)
ch_score = calinski_harabasz_score(X_scaled, km_labels)
ari_true = adjusted_rand_score(true_labels, km_labels)

print("K-Means silhouette:", sil_avg)
print("K-Means Davies-Bouldin:", db_score)
print("K-Means Calinski-Harabasz:", ch_score)
print("K-Means ARI vs true cultivar:", ari_true)

sample_sil_values = silhouette_samples(X_scaled, km_labels)
fig, ax = plt.subplots(figsize=(9, 6.5))
y_lower = 10
for i in range(K):
    ith_vals = sample_sil_values[km_labels == i]
    ith_vals.sort()
    size_i = ith_vals.shape[0]
    y_upper = y_lower + size_i
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_vals,
                      facecolor=PALETTE[i], edgecolor=PALETTE[i], alpha=0.8)
    ax.text(-0.05, y_lower + 0.5 * size_i, str(i))
    y_lower = y_upper + 10
ax.axvline(sil_avg, color="red", linestyle="--", label=f"Average = {sil_avg:.2f}")
ax.set_title("Silhouette Plot for K-Means (k=3)", fontsize=15, fontweight="bold")
ax.set_xlabel("Silhouette Coefficient")
ax.set_ylabel("Cluster")
ax.set_yticks([])
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/04_silhouette_plot.png")
plt.close(fig)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print("Explained variance ratio (PC1, PC2):", pca.explained_variance_ratio_)
print("Total variance captured:", pca.explained_variance_ratio_.sum())

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for i in range(K):
    axes[0].scatter(X_pca[km_labels == i, 0], X_pca[km_labels == i, 1],
                     s=45, alpha=0.75, color=PALETTE[i], label=f"Cluster {i}")
centers_pca = pca.transform(kmeans.cluster_centers_)
axes[0].scatter(centers_pca[:, 0], centers_pca[:, 1], marker="X", s=250,
                c="black", edgecolor="white", linewidth=1.5, label="Centroids", zorder=5)
axes[0].set_title("K-Means Clusters (PCA Projection)")
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
axes[0].legend()

for i, name in enumerate(target_names):
    axes[1].scatter(X_pca[true_labels == i, 0], X_pca[true_labels == i, 1],
                     s=45, alpha=0.75, color=PALETTE[i], label=name)
axes[1].set_title("True Cultivar Labels (PCA Projection)")
axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
axes[1].legend()
fig.suptitle("Unsupervised Clusters Closely Recover the True Cultivars", y=1.03, fontsize=15, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/05_pca_clusters_vs_truth.png")
plt.close(fig)

linked = linkage(X_scaled, method="ward")
cut_height = 0.5 * (linked[-K, 2] + linked[-(K - 1), 2])
fig, ax = plt.subplots(figsize=(11, 6))
dendrogram(linked, truncate_mode="lastp", p=30, ax=ax,
           color_threshold=cut_height,
           above_threshold_color="gray")
ax.axhline(cut_height, color="crimson", linestyle="--", linewidth=1.3,
           label=f"Cut for k={K}")
ax.set_title("Hierarchical Clustering Dendrogram (Ward Linkage)", fontsize=15, fontweight="bold")
ax.set_xlabel("Sample Index or (Cluster Size)")
ax.set_ylabel("Ward Distance")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/06_dendrogram.png")
plt.close(fig)

agg = AgglomerativeClustering(n_clusters=K, linkage="ward")
agg_labels = agg.fit_predict(X_scaled)
agg_sil = silhouette_score(X_scaled, agg_labels)
ari_agg_vs_true = adjusted_rand_score(true_labels, agg_labels)
ari_km_vs_agg = adjusted_rand_score(km_labels, agg_labels)

print("Agglomerative silhouette:", agg_sil)
print("Agglomerative ARI vs true:", ari_agg_vs_true)
print("ARI between KMeans and Agglomerative:", ari_km_vs_agg)

fig, ax = plt.subplots(figsize=(7.5, 6.5))
for i in range(K):
    ax.scatter(X_pca[agg_labels == i, 0], X_pca[agg_labels == i, 1],
               s=45, alpha=0.75, color=PALETTE[i], label=f"Cluster {i}")
ax.set_title("Agglomerative (Ward) Clusters (PCA Projection)", fontsize=14, fontweight="bold")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/07_agglomerative_pca.png")
plt.close(fig)

profile = X_scaled_df.copy()
profile["cluster"] = km_labels
cluster_means_scaled = profile.groupby("cluster").mean()

fig, ax = plt.subplots(figsize=(11, 4.5))
sns.heatmap(cluster_means_scaled, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
            linewidths=0.5, cbar_kws={"label": "Mean Standardized Value"}, ax=ax,
            yticklabels=[f"Cluster {i}" for i in cluster_means_scaled.index])
ax.set_yticks([y + 0.5 for y in range(len(cluster_means_scaled))])
ax.set_yticklabels([f"Cluster {i}" for i in cluster_means_scaled.index], rotation=0, va="center")
ax.set_title("Cluster Chemical Profiles (Standardized Feature Means)", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Feature")
ax.set_ylabel("")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/08_cluster_profile_heatmap.png")
plt.close(fig)

raw_profile = df.copy()
raw_profile["cluster"] = km_labels
cluster_means_raw = raw_profile.groupby("cluster").mean().round(2)
print("\nRaw feature means per cluster:\n", cluster_means_raw.T)
print("\nCluster sizes:\n", pd.Series(km_labels).value_counts().sort_index())

key_features = ["alcohol", "flavanoids", "color_intensity", "proline", "total_phenols"]
fig, axes = plt.subplots(1, len(key_features), figsize=(4 * len(key_features), 5), sharex=False)
for ax, feat in zip(axes, key_features):
    sns.boxplot(data=raw_profile, x="cluster", y=feat, ax=ax, palette=PALETTE[:K], hue="cluster", legend=False)
    ax.set_title(feat, fontsize=12)
    ax.set_xlabel("Cluster")
fig.suptitle("Distinguishing Features Across Clusters", y=1.04, fontsize=15, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/09_key_feature_boxplots.png")
plt.close(fig)

cm = confusion_matrix(true_labels, km_labels)
cm_df = pd.DataFrame(cm, index=[f"True: {n}" for n in target_names],
                      columns=[f"Cluster {i}" for i in range(K)])
fig, ax = plt.subplots(figsize=(6.5, 5))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
ax.set_title("Cluster Assignment vs. True Cultivar", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/10_cluster_vs_truth_matrix.png")
plt.close(fig)
print("\nConfusion-style matrix (rows=true cultivar, cols=cluster):\n", cm_df)

print("\nAll charts saved to", OUT_DIR)
