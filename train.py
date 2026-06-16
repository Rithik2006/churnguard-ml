import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib
import os

print("=" * 50)
print("  ChurnGuard - Model Training Script")
print("=" * 50)

os.makedirs('data',   exist_ok=True)
os.makedirs('models', exist_ok=True)

# ── Generate dataset ────────────────────────────────
print("\n[1/5] Generating dataset...")
np.random.seed(42)
n = 8000

tenure                   = np.random.randint(1, 60, n)
num_of_orders            = np.random.randint(1, 100, n)
days_since_last_purchase = np.random.randint(1, 180, n)
total_spent              = np.random.uniform(100, 10000, n).round(2)
avg_order_value          = (total_spent / num_of_orders).round(2)
num_complaints           = np.random.randint(0, 10, n)
discount_used            = np.random.randint(0, 2, n)
login_frequency          = np.random.randint(0, 30, n)
preferred_category       = np.random.randint(0, 4, n)

score = (
    -0.05  * tenure +
    -0.01  * num_of_orders +
     0.02  * days_since_last_purchase +
    -0.0001 * total_spent +
     0.3   * num_complaints +
    -0.05  * login_frequency +
    np.random.normal(0, 0.5, n)
)
churn = (score > score.mean()).astype(int)

df = pd.DataFrame({
    'tenure':                   tenure,
    'num_of_orders':            num_of_orders,
    'days_since_last_purchase': days_since_last_purchase,
    'total_spent':              total_spent,
    'avg_order_value':          avg_order_value,
    'num_complaints':           num_complaints,
    'discount_used':            discount_used,
    'login_frequency':          login_frequency,
    'preferred_category':       preferred_category,
    'churn':                    churn
})

df.to_csv('data/churn_data.csv', index=False)
print(f"    ✅ Dataset created  →  {len(df)} records | Churn rate: {df['churn'].mean():.1%}")

# ── Prepare features ────────────────────────────────
print("\n[2/5] Preparing features...")
X = df.drop('churn', axis=1)
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"    ✅ Train: {len(X_train)} | Test: {len(X_test)}")

# ── Scale ───────────────────────────────────────────
print("\n[3/5] Scaling features...")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print("    ✅ StandardScaler applied")

# ── Train XGBoost ───────────────────────────────────
print("\n[4/5] Training XGBoost model...")
model = XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, eval_metric='logloss'
)
model.fit(X_train_sc, y_train)
print("    ✅ Model trained")

# ── Evaluate ────────────────────────────────────────
print("\n[5/5] Evaluating...")
y_pred = model.predict(X_test_sc)
y_prob = model.predict_proba(X_test_sc)[:, 1]
print(f"\n    Accuracy : {accuracy_score(y_test, y_pred):.2%}")
print(f"    ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
print(f"\n{classification_report(y_test, y_pred)}")

# ── Save ────────────────────────────────────────────
joblib.dump(model,  'models/churn_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("=" * 50)
print("  ✅ Model saved  →  models/churn_model.pkl")
print("  ✅ Scaler saved →  models/scaler.pkl")
print("  Now run  →  python app.py")
print("=" * 50)
