import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy.stats import mannwhitneyu

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

RANDOM_STATE = 42
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(current_dir, "Customer-Churn-Records.csv")
sns.set_theme(style="whitegrid")
os.makedirs("assets", exist_ok=True)
os.makedirs("models", exist_ok=True)

df = pd.read_csv(DATA_PATH)
print("Boyut:", df.shape)
df.head()

print("Eksik değerler:\n", df.isnull().sum()[df.isnull().sum() > 0], "\n")
print("Duplike satır:", df.duplicated().sum())
print("\nHedef dağılımı:\n", df["Exited"].value_counts(normalize=True).round(3))

DROP_COLS = ["RowNumber", "CustomerId", "Surname", "Complain"]
df = df.drop(columns=DROP_COLS, errors="ignore")

print("\nExited ile korelasyon:\n", df.corr(numeric_only=True)["Exited"].drop("Exited").sort_values().round(3))

stat, p = mannwhitneyu(df.loc[df.Exited == 1, "Age"], df.loc[df.Exited == 0, "Age"])
print(f"\nMann-Whitney U (Yaş ~ Exited): p-value = {p:.2e}")
print("\nÜlkeye göre churn oranı:\n", df.groupby("Geography")["Exited"].mean().round(3))

palette = {0: "#4C72B0", 1: "#C44E52"}
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.kdeplot(data=df, x="Age", hue="Exited", fill=True, common_norm=False,
            palette=palette, ax=axes[0])
axes[0].set_title("Yaş Dağılımı: Kalan (0) vs Terk Eden (1)")

sns.boxplot(data=df, x="Exited", y="Balance", hue="Exited",
            palette=palette, legend=False, ax=axes[1])
axes[1].set_title("Hesap Bakiyesi ~ Terk Durumu")

plt.tight_layout()
plt.savefig("assets/eda.png", dpi=150, bbox_inches="tight")
plt.show()
TARGET      = "Exited"
NUM_COLS    = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
OHE_COLS    = ["Geography"]          # 3 sınıf, sırasız -> One-Hot
LABEL_COLS  = ["Gender"]             # 2 sınıf -> Label (Ordinal) Encoding: Female=0, Male=1
BIN_COLS    = ["HasCrCard", "IsActiveMember"]   # zaten 0/1
FEATURES    = NUM_COLS + OHE_COLS + LABEL_COLS + BIN_COLS

preprocessor = ColumnTransformer(
    transformers=[
        ("num",    StandardScaler(), NUM_COLS),
        ("geo",    OneHotEncoder(drop="first", sparse_output=False), OHE_COLS),
        ("gender", OrdinalEncoder(categories=[["Female", "Male"]]), LABEL_COLS),
        ("bin",    "passthrough", BIN_COLS),
    ]
)
# %% 4. Train/Test Split + Random Forest
X, y = df[FEATURES], df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE   # stratify: churn oranı korunur
)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print(f"Churn oranı -> Train: {y_train.mean():.3f} | Test: {y_test.mean():.3f}")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,          # aşırı öğrenmeyi sınırlar
    class_weight="balanced",     # azınlık sınıfa (churn) daha yüksek ceza
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
model = Pipeline([("prep", preprocessor), ("clf", rf)])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for metric in ["recall", "f1", "roc_auc"]:
    s = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric)
    print(f"CV {metric:8s}: {s.mean():.3f} ± {s.std():.3f}")

model.fit(X_train, y_train)
# pip install xgboost
from sklearn.base import clone
from xgboost import XGBClassifier

spw = (y_train == 0).sum() / (y_train == 1).sum()      # dengesizlik düzeltmesi
xgb = Pipeline([
    ("prep", clone(preprocessor)),
    ("clf", XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                          subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=spw, eval_metric="logloss",
                          random_state=RANDOM_STATE)),
])
for metric in ["recall", "f1", "roc_auc"]:
    s = cross_val_score(xgb, X_train, y_train, cv=cv, scoring=metric)
    print(f"XGB CV {metric:8s}: {s.mean():.3f} ± {s.std():.3f}")
oof_proba = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
thr_grid  = np.arange(0.20, 0.71, 0.05)
f2_scores = [fbeta_score(y_train, (oof_proba >= t).astype(int), beta=2) for t in thr_grid]
BEST_THR  = round(float(thr_grid[int(np.argmax(f2_scores))]), 2)
print(f"Seçilen karar eşiği (max F2): {BEST_THR}")

y_proba = model.predict_proba(X_test)[:, 1]

def evaluate(thr):
    y_pred = (y_proba >= thr).astype(int)
    return {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1":        f1_score(y_test, y_pred),
    }

results = pd.DataFrame([evaluate(0.5), evaluate(BEST_THR)],
                       index=["Varsayılan eşik (0.50)", f"Optimize eşik ({BEST_THR})"]).round(3)
print("\n", results)
print(f"\nROC-AUC: {roc_auc_score(y_test, y_proba):.3f} | PR-AUC: {average_precision_score(y_test, y_proba):.3f}")

y_pred = (y_proba >= BEST_THR).astype(int)
print("\n", classification_report(y_test, y_pred, target_names=["Kalan", "Terk"]))

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                       display_labels=["Kalan", "Terk"]).plot(cmap="Blues", values_format="d", ax=ax)
ax.set_title(f"Confusion Matrix (eşik = {BEST_THR})")
plt.savefig("assets/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

feat_names  = model.named_steps["prep"].get_feature_names_out()
importances = pd.Series(model.named_steps["clf"].feature_importances_, index=feat_names).sort_values()
importances.plot.barh(figsize=(7, 5), color="#4C72B0", title="Random Forest Özellik Önemleri")
plt.tight_layout()
plt.savefig("assets/feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
artifact = {"pipeline": model, "threshold": BEST_THR, "features": FEATURES}
joblib.dump(artifact, "models/churn_model.joblib", compress=3)

loaded = joblib.load("models/churn_model.joblib")
assert np.allclose(loaded["pipeline"].predict_proba(X_test)[:, 1], y_proba)
print("✅ Model kaydedildi: models/churn_model.joblib")