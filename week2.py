import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

OUT_DIR = "charts"
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
PALETTE = "viridis"
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.size"] = 12


def money_fmt(x, pos):
    return f"${x:,.0f}"


df = sns.load_dataset("diamonds")

print("Raw shape:", df.shape)
print(df.head())

print("\nMissing values per column:\n", df.isnull().sum())
print("\nExact duplicate rows:", df.duplicated().sum())

invalid_dims = df[(df.x == 0) | (df.y == 0) | (df.z == 0)]
print("\nRows with a zero-mm dimension:", len(invalid_dims))
print(invalid_dims[["carat", "x", "y", "z", "price"]].head(10))

extreme = df[(df.y > 20) | (df.z > 10)]
print("\nRows with an implausibly large dimension:")
print(extreme[["carat", "x", "y", "z", "price"]])

cut_order = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
color_order = ["J", "I", "H", "G", "F", "E", "D"]
clarity_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

df["cut"] = pd.Categorical(df["cut"], categories=cut_order, ordered=True)
df["color"] = pd.Categorical(df["color"], categories=color_order, ordered=True)
df["clarity"] = pd.Categorical(df["clarity"], categories=clarity_order, ordered=True)

clean = df[(df.x > 0) & (df.y > 0) & (df.z > 0)].copy()
clean = clean[(clean.y < 20) & (clean.z < 10)]
clean = clean.drop_duplicates()

clean["price_per_carat"] = clean["price"] / clean["carat"]
clean["log_price"] = np.log10(clean["price"])

print("\nCleaned shape:", clean.shape)
print("Rows removed:", df.shape[0] - clean.shape[0])

print("\nNumeric summary:\n", clean[["carat", "depth", "table", "price"]].describe())
print("\n% of diamonds under 1 carat:", (clean.carat < 1).mean() * 100)
print("% of diamonds that are Ideal cut:", (clean.cut == "Ideal").mean() * 100)

print("\nAverage price by cut:\n", clean.groupby("cut", observed=True)["price"].mean().reindex(cut_order))
print("\nAverage price by color:\n", clean.groupby("color", observed=True)["price"].mean().reindex(color_order))
print("\nAverage price-per-carat by clarity:\n",
      clean.groupby("clarity", observed=True)["price_per_carat"].mean().reindex(clarity_order))

enc = clean.copy()
enc["cut_rank"] = enc["cut"].cat.codes
enc["color_rank"] = enc["color"].cat.codes
enc["clarity_rank"] = enc["clarity"].cat.codes

num_cols = ["carat", "depth", "table", "price", "x", "y", "z",
            "cut_rank", "color_rank", "clarity_rank"]
corr = enc[num_cols].corr()
print("\nCorrelation with price:\n", corr["price"].sort_values(ascending=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(clean["price"], bins=60, color="#3B6E8F", ax=axes[0])
axes[0].set_title("Price Distribution (Raw)")
axes[0].set_xlabel("Price (US$)")
axes[0].set_ylabel("Count of Diamonds")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(money_fmt))

sns.histplot(clean["log_price"], bins=60, color="#A64D79", ax=axes[1], kde=True)
axes[1].set_title("Price Distribution (Log10-Transformed)")
axes[1].set_xlabel("log10(Price)")
axes[1].set_ylabel("Count of Diamonds")
fig.suptitle("Diamond Price Is Strongly Right-Skewed", y=1.03, fontsize=15, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/01_price_distribution.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5.5))
sns.histplot(clean["carat"], bins=60, color="#3B6E8F", ax=ax)
ax.axvline(clean["carat"].median(), color="black", linestyle="--", linewidth=1.5,
           label=f"Median = {clean['carat'].median():.2f} ct")
ax.axvline(clean["carat"].mean(), color="red", linestyle=":", linewidth=1.5,
           label=f"Mean = {clean['carat'].mean():.2f} ct")
ax.set_title("Carat Weight Distribution", fontsize=15, fontweight="bold")
ax.set_xlabel("Carat")
ax.set_ylabel("Count of Diamonds")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/02_carat_distribution.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9.5, 6.5))
sample = clean.sample(6000, random_state=42)
sns.scatterplot(data=sample, x="carat", y="price", hue="cut", palette=PALETTE,
                 alpha=0.55, s=22, ax=ax, hue_order=cut_order)
ax.set_title("Price vs. Carat, Colored by Cut Quality", fontsize=15, fontweight="bold")
ax.set_xlabel("Carat")
ax.set_ylabel("Price (US$)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(money_fmt))
ax.legend(title="Cut", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/03_carat_vs_price_scatter.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 6))
sns.boxplot(data=clean, x="cut", y="price", order=cut_order, palette=PALETTE,
            ax=ax, hue="cut", legend=False)
ax.set_title("Price Distribution by Cut Quality", fontsize=15, fontweight="bold")
ax.set_xlabel("Cut Quality (Fair -> Ideal)")
ax.set_ylabel("Price (US$)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(money_fmt))
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/04_boxplot_price_by_cut.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(10.5, 6))
sns.boxplot(data=clean, x="clarity", y="price_per_carat", order=clarity_order, palette=PALETTE,
            ax=ax, hue="clarity", legend=False, showfliers=False)
ax.set_title("Price-per-Carat by Clarity Grade (outliers hidden)", fontsize=15, fontweight="bold")
ax.set_xlabel("Clarity Grade (I1 = worst -> IF = best)")
ax.set_ylabel("Price per Carat (US$)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(money_fmt))
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/05_boxplot_ppc_by_clarity.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 6))
grp = clean.groupby("color", observed=True)["price"].agg(["mean", "count"]).reindex(color_order)
bars = ax.bar(grp.index.astype(str), grp["mean"], color=sns.color_palette(PALETTE, len(grp)))
ax.set_title("Average Price by Color Grade", fontsize=15, fontweight="bold")
ax.set_xlabel("Color Grade (J = most tinted -> D = colorless)")
ax.set_ylabel("Average Price (US$)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(money_fmt))
for bar, n in zip(bars, grp["count"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 60, f"n={n:,}",
            ha="center", va="bottom", fontsize=9, color="dimgray")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/06_avg_price_by_color.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9.5, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, cbar_kws={"label": "Pearson r"}, ax=ax)
ax.set_title("Correlation Matrix (Numeric + Ordinal-Encoded Grades)", fontsize=14, fontweight="bold", pad=14)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/07_correlation_heatmap.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 5.5))
sns.countplot(data=clean, x="cut", order=cut_order, palette=PALETTE, ax=ax, hue="cut", legend=False)
ax.set_title("Count of Diamonds by Cut Quality", fontsize=15, fontweight="bold")
ax.set_xlabel("Cut Quality")
ax.set_ylabel("Count of Diamonds")
for p_ in ax.patches:
    ax.annotate(f"{int(p_.get_height()):,}", (p_.get_x() + p_.get_width() / 2, p_.get_height()),
                ha="center", va="bottom", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/08_countplot_cut.png")
plt.close(fig)

anomalies = df[(df.x == 0) | (df.y == 0) | (df.z == 0) | (df.y > 20) | (df.z > 10)]
fig, ax = plt.subplots(figsize=(9.5, 6.5))
good = df.drop(anomalies.index).sample(5000, random_state=1)
sns.scatterplot(data=good, x="carat", y="x", color="#3B6E8F", alpha=0.35, s=18, ax=ax, label="Normal records")
ax.scatter(anomalies["carat"], anomalies["x"], color="crimson", s=90, marker="X",
           label=f"Anomalous records (n={len(anomalies)})", zorder=5)
ax.set_title("Detected Anomalies: Zero or Implausible Physical Dimensions", fontsize=14, fontweight="bold")
ax.set_xlabel("Carat")
ax.set_ylabel("Length, x (mm)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/09_anomaly_scatter.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 6.5))
sns.violinplot(data=clean, x="clarity", y="log_price", order=clarity_order, palette=PALETTE,
               ax=ax, hue="clarity", legend=False, cut=0)
ax.set_title("Log-Price Distribution Shape by Clarity Grade", fontsize=15, fontweight="bold")
ax.set_xlabel("Clarity Grade (I1 = worst -> IF = best)")
ax.set_ylabel("log10(Price)")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/10_violin_price_by_clarity.png")
plt.close(fig)

print(f"\nDone. 10 charts saved to ./{OUT_DIR}/")
