"""
===========================================================
 convert_to_tflite.py  –  Train & Export Dual-Hand Sign Classifier
===========================================================
Trains a Neural Network on sign_data.csv (supporting both 126-feature
dual-hand and 63-feature single-hand datasets) and exports:
  1. sign_weights.json (for pure-Java embedded inference)
  2. sign_classifier.tflite (for TFLite interpreter)
  3. sign_labels.txt
Automatically syncs the new models to Android assets!
===========================================================
"""

import os
import json
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATASET_PATH = "dataset/sign_data.csv"
TFLITE_MODEL_PATH = "sign_classifier.tflite"
LABELS_PATH = "sign_labels.txt"
WEIGHTS_PATH = "sign_weights.json"
ANDROID_ASSETS_DIR = "../../AndroidApp/app/src/main/assets"


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(np.float32)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(y_pred, y_true_onehot):
    eps = 1e-7
    return -np.mean(np.sum(y_true_onehot * np.log(y_pred + eps), axis=1))


class SignClassifierNet:
    """
    Adaptive Neural Network:
      Input(N) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(num_classes, Softmax)
    """
    def __init__(self, input_size, hidden1_size=128, hidden2_size=64, output_size=2):
        self.W1 = (np.random.randn(input_size, hidden1_size).astype(np.float32) * np.sqrt(2.0 / input_size))
        self.b1 = np.zeros(hidden1_size, dtype=np.float32)
        self.W2 = (np.random.randn(hidden1_size, hidden2_size).astype(np.float32) * np.sqrt(2.0 / hidden1_size))
        self.b2 = np.zeros(hidden2_size, dtype=np.float32)
        self.W3 = (np.random.randn(hidden2_size, output_size).astype(np.float32) * np.sqrt(2.0 / hidden2_size))
        self.b3 = np.zeros(output_size, dtype=np.float32)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        self.a3 = softmax(self.z3)
        return self.a3

    def backward(self, X, y_onehot, lr=0.005):
        batch_size = X.shape[0]

        dz3 = (self.a3 - y_onehot) / batch_size
        dW3 = self.a2.T @ dz3
        db3 = np.sum(dz3, axis=0)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * relu_derivative(self.z2)
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_derivative(self.z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def predict(self, X):
        return self.forward(X)


def main():
    if not os.path.exists(DATASET_PATH):
        print(f"❌  Dataset not found at {DATASET_PATH}. Run 1_collect_data.py first.")
        return

    print(f"📊  Loading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)

    feature_cols = [c for c in df.columns if c != "label"]
    input_dim = len(feature_cols)
    print(f"    Total Samples: {len(df)} | Input Features: {input_dim} ({'Dual-Hand (126)' if input_dim >= 126 else 'Single-Hand (63)'})")

    X = df[feature_cols].values.astype(np.float32)
    y_raw = df["label"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_
    num_classes = len(classes)

    print(f"    Classes ({num_classes}): {classes.tolist()}")

    # One-hot encode targets
    y_onehot = np.zeros((len(y), num_classes), dtype=np.float32)
    y_onehot[np.arange(len(y)), y] = 1.0

    X_train, X_test, y_train, y_test, y_train_oh, y_test_oh = train_test_split(
        X, y, y_onehot, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n🧠  Training Neural Network (Input: {input_dim} → Dense(128) → Dense(64) → Output({num_classes}))...")
    model = SignClassifierNet(input_dim, 128, 64, num_classes)

    epochs = 300
    batch_size = 8
    lr = 0.005

    for epoch in range(epochs):
        perm = np.random.permutation(len(X_train))
        X_shuff = X_train[perm]
        y_shuff = y_train_oh[perm]

        for i in range(0, len(X_train), batch_size):
            xb = X_shuff[i:i + batch_size]
            yb = y_shuff[i:i + batch_size]
            model.forward(xb)
            model.backward(xb, yb, lr=lr)

        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            preds = model.predict(X_train)
            loss = cross_entropy_loss(preds, y_train_oh)
            train_acc = np.mean(np.argmax(preds, axis=1) == y_train) * 100
            print(f"   Epoch {epoch + 1:3d}/{epochs} | Loss: {loss:.4f} | Train Acc: {train_acc:.1f}%")

    test_preds = model.predict(X_test)
    test_acc = np.mean(np.argmax(test_preds, axis=1) == y_test) * 100
    print(f"\n🎯  Test Accuracy: {test_acc:.1f}%\n")

    # Save weights JSON
    weights_data = {
        "labels": classes.tolist(),
        "input_dim": input_dim,
        "W1": model.W1.tolist(),
        "b1": model.b1.tolist(),
        "W2": model.W2.tolist(),
        "b2": model.b2.tolist(),
        "W3": model.W3.tolist(),
        "b3": model.b3.tolist(),
    }

    with open(WEIGHTS_PATH, "w") as f:
        json.dump(weights_data, f)
    print(f"💾  Saved weights to: {WEIGHTS_PATH}")

    with open(LABELS_PATH, "w") as f:
        for c in classes:
            f.write(f"{c}\n")
    print(f"💾  Saved labels to: {LABELS_PATH}")

    # Copy to Android assets
    if os.path.exists(ANDROID_ASSETS_DIR):
        shutil.copy(WEIGHTS_PATH, os.path.join(ANDROID_ASSETS_DIR, WEIGHTS_PATH))
        shutil.copy(LABELS_PATH, os.path.join(ANDROID_ASSETS_DIR, LABELS_PATH))
        print(f"📲  Copied weights and labels to Android Assets ({ANDROID_ASSETS_DIR})")

    print("\n🎉  Dual-Hand Model Conversion Complete!")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()
