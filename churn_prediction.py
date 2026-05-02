import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression

# -----------------------------
# 1) Load dataset
# -----------------------------
df = pd.read_csv("churn.csv")

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nColumns:", df.columns.tolist())

# -----------------------------
# 2) Target & feature selection
# -----------------------------
# Try to detect churn column (case-insensitive)
target_col = None
for col in df.columns:
    if "churn" in col.lower():
        target_col = col
        break

if target_col is None:
    raise ValueError("No churn column found! Please check the dataset.")

# Convert target to binary (0/1)
y = df[target_col].apply(lambda v: 1 if str(v).strip().lower() in ["yes", "1", "true"] else 0)

# Remove target from features
X = df.drop(columns=[target_col])

# Keep only numeric features for simplicity
X = X.select_dtypes(include=[np.number])

# Handle missing values
X = X.fillna(0)

# -----------------------------
# 3) Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 4) Scale features
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 5) Perceptron (M-P neuron)
# -----------------------------
class Perceptron:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(self.n_iter):
            for xi, target in zip(X, y):
                linear_output = np.dot(xi, self.w) + self.b
                y_pred = 1 if linear_output >= 0 else 0
                update = self.lr * (target - y_pred)
                self.w += update * xi
                self.b += update
        return self

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.where(linear_output >= 0, 1, 0)

# Train Perceptron
perceptron = Perceptron(lr=0.01, n_iter=10)
perceptron.fit(X_train, y_train)
y_pred_p = perceptron.predict(X_test)

print("\n=== Perceptron Results ===")
print(classification_report(y_test, y_pred_p))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_p))

# -----------------------------
# 6) Logistic Regression (sigmoid)
# -----------------------------
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
y_prob_lr = log_reg.predict_proba(X_test)[:, 1]

print("\n=== Logistic Regression Results ===")
print(classification_report(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_lr))
