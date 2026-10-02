import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, silhouette_score
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

DATA_PATH = os.environ.get("DATA_PATH", "/kaggle/diamonds.csv")
OUT_DIR = os.environ.get("OUT_DIR", "outputs")
SEED = 42
os.makedirs(OUT_DIR, exist_ok=True)
results = {}

def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, name), dpi=140)
    plt.close()

raw = pd.read_csv(DATA_PATH)
raw = raw.loc[:, ~raw.columns.str.contains("^Unnamed")]
print(raw.shape)
print(raw.head())
print(raw.dtypes)
print(raw.describe().T)
print(raw.isna().sum())

results["raw_rows"] = int(len(raw))
results["duplicates"] = int(raw.duplicated().sum())
results["zero_dims"] = int(((raw[["x", "y", "z"]] == 0).any(axis=1)).sum())
results["extreme_dims"] = int(((raw["y"] > 30) | (raw["z"] > 30)).sum())

df = raw.drop_duplicates().copy()
df = df[(df[["x", "y", "z"]] > 0).all(axis=1)]
df = df[(df["y"] < 30) & (df["z"] < 30)]

computed_depth = 2 * df["z"] / (df["x"] + df["y"]) * 100
inconsistent = (computed_depth - df["depth"]).abs() > 1.5
results["depth_inconsistent"] = int(inconsistent.sum())
df = df[~inconsistent]

df = df[(df["table"] >= 40) & (df["table"] <= 80) & (df["depth"] >= 50) & (df["depth"] <= 75)]
df = df.reset_index(drop=True)
results["clean_rows"] = int(len(df))
results["rows_removed"] = results["raw_rows"] - results["clean_rows"]
print("Rows after cleaning:", len(df))

CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER = ["J", "I", "H", "G", "F", "E", "D"]
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df["price"], bins=60, ax=axes[0], color="#2E5C94")
axes[0].set_title("Price distribution")
sns.histplot(np.log(df["price"]), bins=60, ax=axes[1], color="#2E5C94")
axes[1].set_title("log(Price) distribution")
sns.histplot(df["carat"], bins=60, ax=axes[2], color="#2E5C94")
axes[2].set_title("Carat distribution")
save_fig("eda_distributions.png")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, col, order in zip(axes, ["cut", "color", "clarity"], [CUT_ORDER, COLOR_ORDER, CLARITY_ORDER]):
    sns.boxplot(data=df, x=col, y="price", order=order, ax=ax, color="#9CC3EA", fliersize=1)
    ax.set_title(f"Price by {col}")
save_fig("eda_price_by_category.png")

sample = df.sample(8000, random_state=SEED)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.scatterplot(data=sample, x="carat", y="price", hue="clarity", hue_order=CLARITY_ORDER, s=8, ax=axes[0])
axes[0].set_title("Price vs carat (coloured by clarity)")
sns.scatterplot(data=sample, x=np.log(sample["carat"]), y=np.log(sample["price"]), s=8, ax=axes[1], color="#2E5C94")
axes[1].set_title("log(price) vs log(carat)")
axes[1].set_xlabel("log(carat)")
axes[1].set_ylabel("log(price)")
save_fig("eda_price_vs_carat.png")

num_cols = ["carat", "depth", "table", "x", "y", "z", "price"]
plt.figure(figsize=(7, 5.5))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation matrix")
save_fig("eda_correlation.png")

mean_price = df.groupby(["cut", "clarity"])["price"].mean().unstack().loc[CUT_ORDER, CLARITY_ORDER]
plt.figure(figsize=(9, 4.5))
sns.heatmap(mean_price, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Mean price by cut and clarity")
save_fig("eda_cut_clarity_heatmap.png")

results["price_skew"] = float(df["price"].skew())
results["log_price_skew"] = float(np.log(df["price"]).skew())
results["corr_carat_price"] = float(df["carat"].corr(df["price"]))
results["corr_depth_price"] = float(df["depth"].corr(df["price"]))
results["median_price_by_cut"] = df.groupby("cut")["price"].median().to_dict()
results["median_carat_by_cut"] = df.groupby("cut")["carat"].median().to_dict()
results["median_price_by_clarity"] = df.groupby("clarity")["price"].median().to_dict()
results["mean_price"] = float(df["price"].mean())
results["median_price"] = float(df["price"].median())

df["volume"] = df["x"] * df["y"] * df["z"]
df["log_carat"] = np.log(df["carat"])
df["log_volume"] = np.log(df["volume"])
df["table_depth_ratio"] = df["table"] / df["depth"]

ordinal_cols = ["cut", "color", "clarity"]
numeric_cols = ["carat", "depth", "table", "x", "y", "z", "volume", "log_carat", "log_volume", "table_depth_ratio"]
feature_cols = ordinal_cols + numeric_cols

X = df[feature_cols]
y = df["price"]
y_log = np.log(y)

X_train, X_test, y_train, y_test, ylog_train, ylog_test = train_test_split(
    X, y, y_log, test_size=0.2, random_state=SEED)
print("Train:", X_train.shape, "Test:", X_test.shape)

def make_preprocessor(scale):
    num_step = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("ord", OrdinalEncoder(categories=[CUT_ORDER, COLOR_ORDER, CLARITY_ORDER]), ordinal_cols),
        ("num", num_step, numeric_cols)])

models = {
    "Linear Regression": Pipeline([("prep", make_preprocessor(True)), ("model", LinearRegression())]),
    "Ridge": Pipeline([("prep", make_preprocessor(True)), ("model", Ridge(alpha=1.0))]),
    "Random Forest": Pipeline([("prep", make_preprocessor(False)),
                               ("model", RandomForestRegressor(n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=SEED))]),
    "Hist Gradient Boosting": Pipeline([("prep", make_preprocessor(False)),
                                        ("model", HistGradientBoostingRegressor(random_state=SEED))]),
}

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
cv_results = {}
for name, pipe in models.items():
    scores = cross_val_score(pipe, X_train, ylog_train, cv=kf, scoring="r2", n_jobs=1)
    cv_results[name] = {"cv_r2_mean": float(scores.mean()), "cv_r2_std": float(scores.std())}
    print(f"{name}: CV R2 on log price = {scores.mean():.4f} +/- {scores.std():.4f}")
results["cv"] = cv_results

param_dist = {
    "model__learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15],
    "model__max_iter": [300, 500, 800],
    "model__max_leaf_nodes": [15, 31, 63],
    "model__min_samples_leaf": [10, 20, 40],
    "model__l2_regularization": [0.0, 0.1, 1.0],
}
search = RandomizedSearchCV(models["Hist Gradient Boosting"], param_dist, n_iter=12, cv=3,
                            scoring="neg_root_mean_squared_error", random_state=SEED, n_jobs=1)
search.fit(X_train, ylog_train)
print("Best params:", search.best_params_)
results["best_params"] = {k: (float(v) if isinstance(v, float) else int(v)) for k, v in search.best_params_.items()}
models["HGB (tuned)"] = search.best_estimator_
tuned_scores = cross_val_score(search.best_estimator_, X_train, ylog_train, cv=kf, scoring="r2", n_jobs=1)
cv_results["HGB (tuned)"] = {"cv_r2_mean": float(tuned_scores.mean()), "cv_r2_std": float(tuned_scores.std())}
print(f"HGB (tuned): CV R2 on log price = {tuned_scores.mean():.4f} +/- {tuned_scores.std():.4f}")

def score_on_price(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
    }

test_scores, preds = {}, {}
for name, pipe in models.items():
    if name != "HGB (tuned)":
        pipe.fit(X_train, ylog_train)
    pred = np.exp(pipe.predict(X_test))
    preds[name] = pred
    test_scores[name] = score_on_price(y_test, pred)
    test_scores[name]["R2_log"] = float(r2_score(ylog_test, np.log(pred)))
    print(name, test_scores[name])
results["test"] = test_scores

best_name = max(cv_results, key=lambda k: cv_results[k]["cv_r2_mean"])
results["best_model"] = best_name
best_pipe = models[best_name]
best_pred = preds[best_name]
print("Best model:", best_name)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
axes[0].scatter(y_test, best_pred, s=4, alpha=0.3, color="#2E5C94")
lim = [0, max(y_test.max(), best_pred.max())]
axes[0].plot(lim, lim, "r--")
axes[0].set_xlabel("Actual price")
axes[0].set_ylabel("Predicted price")
axes[0].set_title(f"{best_name}: actual vs predicted")
resid = y_test.values - best_pred
axes[1].scatter(best_pred, resid, s=4, alpha=0.3, color="#2E5C94")
axes[1].axhline(0, color="r", ls="--")
axes[1].set_xlabel("Predicted price")
axes[1].set_ylabel("Residual")
axes[1].set_title("Residuals vs predicted")
rel = resid / y_test.values * 100
sns.histplot(rel[np.abs(rel) < 60], bins=60, ax=axes[2], color="#2E5C94")
axes[2].set_xlabel("Relative error (%)")
axes[2].set_title("Relative error distribution")
save_fig("model_diagnostics.png")

names = list(test_scores)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(names, [test_scores[n]["RMSE"] for n in names], color="#2E5C94")
axes[0].set_title("Test RMSE (USD)")
axes[1].bar(names, [test_scores[n]["MAE"] for n in names], color="#6AA6DE")
axes[1].set_title("Test MAE (USD)")
for ax in axes:
    ax.tick_params(axis="x", rotation=25)
save_fig("model_comparison.png")

bins = pd.qcut(y_test, 5, labels=["Q1 (cheapest)", "Q2", "Q3", "Q4", "Q5 (priciest)"])
seg = pd.DataFrame({"bin": bins.values, "abs_err": np.abs(resid), "ape": np.abs(rel)})
by_price = seg.groupby("bin").agg(MAE=("abs_err", "mean"), MAPE=("ape", "mean"))
print(by_price)
results["error_by_price_quintile"] = by_price.round(2).to_dict()

perm = permutation_importance(best_pipe, X_test, ylog_test, n_repeats=5, random_state=SEED, n_jobs=1, scoring="r2")
imp = pd.Series(perm.importances_mean, index=feature_cols).sort_values()
plt.figure(figsize=(7, 5))
imp.plot.barh(color="#2E5C94")
plt.title("Permutation importance (drop in R2, log price)")
save_fig("feature_importance.png")
results["importance"] = imp.sort_values(ascending=False).round(4).to_dict()

res_by_cut = pd.DataFrame({"cut": X_test["cut"].values, "ape": np.abs(rel)}).groupby("cut")["ape"].mean()
results["mape_by_cut"] = res_by_cut.round(2).to_dict()

cluster_features = ["carat", "depth", "table", "cut_score", "color_score", "clarity_score"]
cl = df.copy()
cl["cut_score"] = cl["cut"].map({c: i for i, c in enumerate(CUT_ORDER)})
cl["color_score"] = cl["color"].map({c: i for i, c in enumerate(COLOR_ORDER)})
cl["clarity_score"] = cl["clarity"].map({c: i for i, c in enumerate(CLARITY_ORDER)})

Z = StandardScaler().fit_transform(cl[cluster_features])

ks = range(2, 9)
inertias, sils = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(Z, km.labels_, sample_size=6000, random_state=SEED))
    print(k, round(inertias[-1], 1), round(sils[-1], 4))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(list(ks), inertias, "o-", color="#2E5C94")
axes[0].set_title("Elbow method")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia")
axes[1].plot(list(ks), sils, "o-", color="#2E5C94")
axes[1].set_title("Silhouette score")
axes[1].set_xlabel("k")
save_fig("cluster_selection.png")
results["silhouette_by_k"] = {int(k): round(float(s), 4) for k, s in zip(ks, sils)}
results["inertia_by_k"] = {int(k): round(float(s), 1) for k, s in zip(ks, inertias)}

best_k = int(list(ks)[int(np.argmax(sils))])
results["best_k"] = best_k
km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(Z)
cl["cluster"] = km.labels_

pca = PCA(n_components=2, random_state=SEED).fit(Z)
proj = pca.transform(Z)
results["pca_var"] = [round(float(v), 4) for v in pca.explained_variance_ratio_]
idx = np.random.RandomState(SEED).choice(len(cl), 8000, replace=False)
plt.figure(figsize=(7, 5.5))
sns.scatterplot(x=proj[idx, 0], y=proj[idx, 1], hue=cl["cluster"].values[idx], palette="tab10", s=10)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"K-Means clusters (k={best_k}) in PCA space")
save_fig("cluster_pca.png")

profile = cl.groupby("cluster").agg(
    size=("price", "size"), carat=("carat", "mean"), depth=("depth", "mean"), table=("table", "mean"),
    cut_score=("cut_score", "mean"), color_score=("color_score", "mean"),
    clarity_score=("clarity_score", "mean"), mean_price=("price", "mean"), median_price=("price", "median")).round(2)
print(profile)
results["cluster_profile"] = profile.reset_index().to_dict(orient="records")
sil_final = silhouette_score(Z, km.labels_, sample_size=6000, random_state=SEED)
results["final_silhouette"] = round(float(sil_final), 4)

plt.figure(figsize=(8, 4.5))
sns.boxplot(data=cl, x="cluster", y="price", color="#9CC3EA", fliersize=1)
plt.title("Price distribution by cluster")
save_fig("cluster_price.png")

with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)

