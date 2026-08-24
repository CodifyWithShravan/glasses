"""
===========================================================
 2_train_model.py  –  Train a Random Forest Sign Classifier
===========================================================
HOW IT WORKS:
  1. Reads  sign_data.csv  (created by 1_collect_data.py).
  2. Splits data into training (80%) and testing (20%).
  3. Trains a Random Forest classifier.
  4. Prints the accuracy & a per-sign classification report.
  5. Saves the trained model as  sign_model.pkl .

USAGE:
  python 2_train_model.py
===========================================================
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ─── CONFIG ────────────────────────────────────────────────
CSV_FILE = "dataset/sign_data.csv"
MODEL_FILE = "sign_model.pkl"
TEST_SIZE = 0.2          # 20% of data held out for testing
RANDOM_STATE = 42        # For reproducible results
N_ESTIMATORS = 100       # Number of trees in the forest
# ───────────────────────────────────────────────────────────

# ─── Step 1: Load the CSV ─────────────────────────────────
print("📂  Loading data from", CSV_FILE, "...")
df = pd.read_csv(CSV_FILE)

print(f"   Total samples : {len(df)}")
print(f"   Labels found  : {df['label'].unique().tolist()}")
print(f"   Samples per label:")
print(df["label"].value_counts().to_string(header=False))
print()

# ─── Step 2: Separate features (X) from labels (y) ───────
# Everything except the last column ('label') is a feature
X = df.drop(columns=["label"])
y = df["label"]

# ─── Step 3: Split into train and test sets ───────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,  # Keep label proportions equal in both sets
)

print(f"📊  Training samples : {len(X_train)}")
print(f"📊  Testing samples  : {len(X_test)}")
print()

# ─── Step 4: Train the Random Forest ─────────────────────
print("🌲  Training Random Forest with", N_ESTIMATORS, "trees...")
model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
)
model.fit(X_train, y_train)
print("   ✅  Training complete!\n")

# ─── Step 5: Evaluate on the test set ────────────────────
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"🎯  Test Accuracy: {accuracy * 100:.1f}%\n")

print("📋  Classification Report:")
print(classification_report(y_test, y_pred))

# ─── Step 6: Save the model ──────────────────────────────
joblib.dump(model, MODEL_FILE)
print(f"💾  Model saved to: {MODEL_FILE}")
print("   You can now use this in 3_test_live.py!")
