import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_validate
)
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, recall_score,
    precision_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, make_scorer
)

data = load_breast_cancer(as_frame=True)
X = data.data.copy()


y = (data.target == 0).astype(int)

print("Dataset shape:", X.shape)
print("Class counts (0=benign, 1=malignant):")
print(y.value_counts().sort_index())
print("Missing values:", X.isna().sum().sum())
print("Duplicate rows:", X.duplicated().sum())


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, X.columns)
    ]
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        solver="liblinear",
        max_iter=5000,
        random_state=42
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

parameter_grid = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__class_weight": [None, "balanced"]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    refit=True
)

grid_search.fit(X_train, y_train)

print("\nBest parameters:", grid_search.best_params_)
print("Best mean CV ROC-AUC:", round(grid_search.best_score_, 4))

scoring = {
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "malignant_recall": make_scorer(recall_score, pos_label=1),
    "malignant_f1": make_scorer(f1_score, pos_label=1)
}

cv_results = cross_validate(
    grid_search.best_estimator_,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)

print("\nCross-validation results:")
for metric, values in cv_results.items():
    if metric.startswith("test_"):
        print(f"{metric.replace('test_', '')}: "
              f"{values.mean():.4f} ± {values.std():.4f}")

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
y_probability = best_model.predict_proba(X_test)[:, 1]

print("\nHold-out test metrics")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Balanced accuracy:", round(balanced_accuracy_score(y_test, y_pred), 4))
print("Malignant recall (sensitivity):",
      round(recall_score(y_test, y_pred, pos_label=1), 4))
print("Malignant precision:",
      round(precision_score(y_test, y_pred, pos_label=1), 4))
print("Malignant F1-score:",
      round(f1_score(y_test, y_pred, pos_label=1), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_probability), 4))

print("\nConfusion matrix")
print("Rows: actual [benign, malignant]")
print("Columns: predicted [benign, malignant]")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\nClassification report")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Benign", "Malignant"]
))