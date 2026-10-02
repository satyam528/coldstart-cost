"""
Step 2: Train and compare regression models, run diagnostics, save the best one.

Models compared:
  1. SLR            - simple linear regression (memory only)  -> baseline
  2. MLR            - multiple linear regression (all 4 features)
  3. MLR (log y)    - MLR on log(target); fixes the curved relationship
  4. Ridge          - MLR with regularisation
  5. Random Forest  - non-linear model, to check "does MLR really win?"

Outputs:
  reports/results.csv, reports/*.png, models/model.pkl
Target is in micro-dollars (USD x 1,000,000) only to keep numbers readable.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)

# ---------------- load data ----------------
df = pd.read_csv(ROOT / "data" / "coldstart_data.csv")
df["y"] = df["cost_overhead_usd"] * 1e6          # micro-dollars
NUM = ["memory_mb", "package_mb", "idle_min"]
CAT = ["runtime"]
FEATURES = NUM + CAT
X, y = df[FEATURES], df["y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- EDA plots ----------------
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["y"], bins=40, ax=ax[0]); ax[0].set_title("Target distribution (micro-USD)")
sns.heatmap(df[NUM + ["y"]].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax[1])
ax[1].set_title("Correlation heatmap")
plt.tight_layout(); plt.savefig(ROOT / "reports" / "eda.png", dpi=130); plt.close()

# ---------------- model builders ----------------
def prep(cols_num, cols_cat):
    return ColumnTransformer([
        ("num", StandardScaler(), cols_num),
        ("cat", OneHotEncoder(drop="first"), cols_cat),
    ])

def make(model, cols_num=NUM, cols_cat=CAT, log_target=False):
    pipe = Pipeline([("prep", prep(cols_num, cols_cat)), ("model", model)])
    if log_target:
        return TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)
    return pipe

models = {
    "SLR (memory only)": (make(LinearRegression(), ["memory_mb"], []), ["memory_mb"]),
    "MLR": (make(LinearRegression()), FEATURES),
    "MLR (log target)": (make(LinearRegression(), log_target=True), FEATURES),
    "Ridge": (make(Ridge(alpha=1.0)), FEATURES),
    "Random Forest": (make(RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)), FEATURES),
}

# ---------------- train + evaluate ----------------
rows, fitted = [], {}
for name, (mdl, cols) in models.items():
    cv_r2 = cross_val_score(mdl, X_train[cols], y_train, cv=5, scoring="r2").mean()
    mdl.fit(X_train[cols], y_train)
    pred = mdl.predict(X_test[cols])
    rows.append({
        "model": name,
        "CV_R2": round(cv_r2, 4),
        "Test_R2": round(r2_score(y_test, pred), 4),
        "Test_RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),
        "Test_MAE": round(mean_absolute_error(y_test, pred), 4),
    })
    fitted[name] = (mdl, cols, pred)

results = pd.DataFrame(rows).sort_values("Test_RMSE")
results.to_csv(ROOT / "reports" / "results.csv", index=False)
print("\n=== MODEL COMPARISON (target in micro-USD) ===")
print(results.to_string(index=False))

best_name = results.iloc[0]["model"]
best_model, best_cols, best_pred = fitted[best_name]
print(f"\nBest model: {best_name}")

# ---------------- diagnostics ----------------
# 1) VIF (multicollinearity) on numeric features
Xn = sm.add_constant(df[NUM])
vif = pd.DataFrame({
    "feature": NUM,
    "VIF": [variance_inflation_factor(Xn.values, i + 1) for i in range(len(NUM))],
}).round(3)
vif.to_csv(ROOT / "reports" / "vif.csv", index=False)
print("\n=== VIF (rule of thumb: < 5 is fine) ===")
print(vif.to_string(index=False))

# 2) residual plots: plain MLR vs best model
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
mlr_pred = fitted["MLR"][2]
ax[0].scatter(mlr_pred, y_test - mlr_pred, s=8, alpha=0.5)
ax[0].axhline(0, color="red"); ax[0].set_title("MLR residuals vs predicted")
ax[0].set_xlabel("predicted"); ax[0].set_ylabel("residual")
ax[1].scatter(best_pred, y_test - best_pred, s=8, alpha=0.5)
ax[1].axhline(0, color="red"); ax[1].set_title(f"{best_name} residuals vs predicted")
ax[1].set_xlabel("predicted")
ax[2].scatter(y_test, best_pred, s=8, alpha=0.5)
lim = [0, max(y_test.max(), best_pred.max())]
ax[2].plot(lim, lim, color="red"); ax[2].set_title(f"{best_name}: actual vs predicted")
ax[2].set_xlabel("actual"); ax[2].set_ylabel("predicted")
plt.tight_layout(); plt.savefig(ROOT / "reports" / "residuals.png", dpi=130); plt.close()

# 3) model comparison bar chart
plt.figure(figsize=(7, 4))
sns.barplot(data=results, x="Test_R2", y="model", color="steelblue")
plt.title("Test R2 by model"); plt.tight_layout()
plt.savefig(ROOT / "reports" / "model_comparison.png", dpi=130); plt.close()

# 4) MLR coefficients (statsmodels summary on the plain MLR) for the report
Xd = pd.get_dummies(df[FEATURES], columns=CAT, drop_first=True).astype(float)
ols = sm.OLS(df["y"], sm.add_constant(Xd)).fit()
(ROOT / "reports" / "mlr_summary.txt").write_text(str(ols.summary()))

# ---------------- save best model ----------------
joblib.dump(best_model, ROOT / "models" / "model.pkl")
meta = {"best_model": best_name, "features": best_cols, "target_units": "micro-USD"}
(ROOT / "models" / "model_info.json").write_text(json.dumps(meta, indent=2))
print("\nSaved models/model.pkl and reports/ (png + csv + txt)")
