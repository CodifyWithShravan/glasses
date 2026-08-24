"""
===========================================================
 convert_to_tflite.py  –  Convert Sign Data to TFLite Model
===========================================================
HOW IT WORKS:
  1. Loads sign_data.csv and trains a simple neural network
     using pure NumPy (no TensorFlow dependency needed).
  2. Manually constructs a TFLite flatbuffer model file.
  3. Exports sign_classifier.tflite and sign_labels.txt.

  This approach avoids requiring TensorFlow, which may not
  be available on all Python versions.

USAGE:
  python convert_to_tflite.py
===========================================================
"""

import os
import struct
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ─── CONFIG ────────────────────────────────────────────────
DATASET_PATH = "dataset/sign_data.csv"
TFLITE_MODEL_PATH = "sign_classifier.tflite"
LABELS_PATH = "sign_labels.txt"
RANDOM_STATE = 42
TEST_SIZE = 0.2
# ───────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════
#  Pure-NumPy Neural Network (for training without TensorFlow)
# ═══════════════════════════════════════════════════════════

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

class SimpleNeuralNet:
    """
    A simple 3-layer neural network:
      Input(63) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(num_classes, Softmax)

    Trained with mini-batch gradient descent.
    """

    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        # Xavier initialization
        self.W1 = np.random.randn(input_size, hidden1_size).astype(np.float32) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden1_size, dtype=np.float32)
        self.W2 = np.random.randn(hidden1_size, hidden2_size).astype(np.float32) * np.sqrt(2.0 / hidden1_size)
        self.b2 = np.zeros(hidden2_size, dtype=np.float32)
        self.W3 = np.random.randn(hidden2_size, output_size).astype(np.float32) * np.sqrt(2.0 / hidden2_size)
        self.b3 = np.zeros(output_size, dtype=np.float32)

    def forward(self, X):
        """Forward pass, storing intermediate values for backprop."""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        self.a3 = softmax(self.z3)
        return self.a3

    def backward(self, X, y_onehot, lr=0.001):
        """Backpropagation with gradient descent."""
        batch_size = X.shape[0]

        # Output layer gradient
        dz3 = (self.a3 - y_onehot) / batch_size
        dW3 = self.a2.T @ dz3
        db3 = np.sum(dz3, axis=0)

        # Hidden layer 2 gradient
        da2 = dz3 @ self.W3.T
        dz2 = da2 * relu_derivative(self.z2)
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        # Hidden layer 1 gradient
        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_derivative(self.z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        # Update weights
        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def predict(self, X):
        return self.forward(X)


# ═══════════════════════════════════════════════════════════
#  TFLite Flatbuffer Builder (manual, no TensorFlow needed)
# ═══════════════════════════════════════════════════════════

def build_tflite_model(W1, b1, W2, b2, W3, b3):
    """
    Build a TFLite flatbuffer for the network:
      Input(1, 63) → FullyConnected(64, ReLU) → FullyConnected(32, ReLU) → FullyConnected(num_classes, Softmax)

    This uses the raw TFLite flatbuffer schema version 3.
    """
    import flatbuffers
    from flatbuffers import builder as fb_builder

    # We'll build the flatbuffer manually using the TFLite schema
    # TFLite schema: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs

    num_classes = W3.shape[1]

    # Prepare tensor data as bytes
    w1_bytes = W1.astype(np.float32).tobytes()
    b1_bytes = b1.astype(np.float32).tobytes()
    w2_bytes = W2.astype(np.float32).tobytes()
    b2_bytes = b2.astype(np.float32).tobytes()
    w3_bytes = W3.astype(np.float32).tobytes()
    b3_bytes = b3.astype(np.float32).tobytes()

    # Build the complete TFLite flatbuffer
    builder = fb_builder.Builder(8192)

    # --- Buffers ---
    # Buffer 0: empty (sentinel)
    # Buffer 1: W1, Buffer 2: b1
    # Buffer 3: W2, Buffer 4: b2
    # Buffer 5: W3, Buffer 6: b3

    def create_buffer(data_bytes):
        if data_bytes:
            data_vec = builder.CreateNumpyVector(np.frombuffer(data_bytes, dtype=np.uint8))
        else:
            data_vec = None

        builder.StartObject(1)
        if data_vec is not None:
            builder.PrependUOffsetTRelativeSlot(0, data_vec, 0)
        return builder.EndObject()

    buf6 = create_buffer(b3_bytes)
    buf5 = create_buffer(w3_bytes)
    buf4 = create_buffer(b2_bytes)
    buf3 = create_buffer(w2_bytes)
    buf2 = create_buffer(b1_bytes)
    buf1 = create_buffer(w1_bytes)
    buf0 = create_buffer(None)

    builder.StartVector(4, 7, 4)
    builder.PrependUOffsetTRelative(buf6)
    builder.PrependUOffsetTRelative(buf5)
    builder.PrependUOffsetTRelative(buf4)
    builder.PrependUOffsetTRelative(buf3)
    builder.PrependUOffsetTRelative(buf2)
    builder.PrependUOffsetTRelative(buf1)
    builder.PrependUOffsetTRelative(buf0)
    buffers_vec = builder.EndVector()

    # --- Tensors ---
    # 0: input  [1, 63]        buffer 0
    # 1: W1     [64, 63]       buffer 1
    # 2: b1     [64]           buffer 2
    # 3: fc1_out [1, 64]       buffer 0
    # 4: relu1_out [1, 64]     buffer 0
    # 5: W2     [32, 64]       buffer 3
    # 6: b2     [32]           buffer 4
    # 7: fc2_out [1, 32]       buffer 0
    # 8: relu2_out [1, 32]     buffer 0
    # 9: W3     [num_classes, 32] buffer 5
    # 10: b3    [num_classes]   buffer 6
    # 11: fc3_out [1, num_classes] buffer 0
    # 12: output [1, num_classes] buffer 0

    tensor_configs = [
        ("input",     [1, 63],              0),
        ("W1",        [64, 63],             1),
        ("b1",        [64],                 2),
        ("fc1_out",   [1, 64],              0),
        ("relu1_out", [1, 64],              0),
        ("W2",        [32, 64],             3),
        ("b2",        [32],                 4),
        ("fc2_out",   [1, 32],              0),
        ("relu2_out", [1, 32],              0),
        ("W3",        [num_classes, 32],     5),
        ("b3",        [num_classes],         6),
        ("fc3_out",   [1, num_classes],      0),
        ("output",    [1, num_classes],      0),
    ]

    tensor_offsets = []
    for name, shape, buffer_idx in reversed(tensor_configs):
        name_off = builder.CreateString(name)
        builder.StartVector(4, len(shape), 4)
        for s in reversed(shape):
            builder.PrependInt32(s)
        shape_vec = builder.EndVector()

        # Tensor table
        builder.StartObject(11)
        builder.PrependUOffsetTRelativeSlot(0, shape_vec, 0)   # shape
        builder.PrependInt8Slot(1, 0, 0)                        # type = FLOAT32 (0)
        builder.PrependUint32Slot(2, buffer_idx, 0)             # buffer
        builder.PrependUOffsetTRelativeSlot(3, name_off, 0)     # name
        tensor_offsets.insert(0, builder.EndObject())

    builder.StartVector(4, len(tensor_offsets), 4)
    for t in reversed(tensor_offsets):
        builder.PrependUOffsetTRelative(t)
    tensors_vec = builder.EndVector()

    # --- Operators ---
    # BuiltinOperator: FULLY_CONNECTED = 9, RELU = 19, SOFTMAX = 25

    # Helper to create input/output vectors for operators
    def create_int_vector(values):
        builder.StartVector(4, len(values), 4)
        for v in reversed(values):
            builder.PrependInt32(v)
        return builder.EndVector()

    # FullyConnectedOptions (no activation fused)
    def create_fc_options():
        builder.StartObject(5)
        builder.PrependInt8Slot(0, 0, 0)  # fused_activation = NONE
        builder.PrependInt8Slot(1, 0, 0)  # weights_format = DEFAULT
        builder.PrependBoolSlot(2, False, False)  # keep_num_dims
        builder.PrependInt8Slot(3, 0, 0)  # asymmetric_quantize_inputs
        builder.PrependInt8Slot(4, 0, 0)  # quantized_bias_type
        return builder.EndObject()

    # SoftmaxOptions (beta = 1.0)
    def create_softmax_options():
        builder.StartObject(1)
        builder.PrependFloat32Slot(0, 1.0, 0.0)  # beta
        return builder.EndObject()

    # Operator 0: FullyConnected(input=0, weights=1, bias=2) -> output=3
    fc1_in = create_int_vector([0, 1, 2])
    fc1_out_vec = create_int_vector([3])
    fc1_opts = create_fc_options()
    builder.StartObject(10)
    builder.PrependUint32Slot(0, 0, 0)  # opcode_index = 0 (FULLY_CONNECTED)
    builder.PrependUOffsetTRelativeSlot(1, fc1_in, 0)
    builder.PrependUOffsetTRelativeSlot(2, fc1_out_vec, 0)
    builder.PrependInt8Slot(3, 0, 0)  # builtin_options_type = FullyConnectedOptions (1) - we skip this, use builtin_options_2
    builder.PrependUOffsetTRelativeSlot(4, fc1_opts, 0)
    op0 = builder.EndObject()

    # Operator 1: ReLU(3) -> 4
    relu1_in = create_int_vector([3])
    relu1_out_vec = create_int_vector([4])
    builder.StartObject(10)
    builder.PrependUint32Slot(0, 1, 0)  # opcode_index = 1 (RELU)
    builder.PrependUOffsetTRelativeSlot(1, relu1_in, 0)
    builder.PrependUOffsetTRelativeSlot(2, relu1_out_vec, 0)
    op1 = builder.EndObject()

    # Operator 2: FullyConnected(input=4, weights=5, bias=6) -> output=7
    fc2_in = create_int_vector([4, 5, 6])
    fc2_out_vec = create_int_vector([7])
    fc2_opts = create_fc_options()
    builder.StartObject(10)
    builder.PrependUint32Slot(0, 0, 0)  # opcode_index = 0
    builder.PrependUOffsetTRelativeSlot(1, fc2_in, 0)
    builder.PrependUOffsetTRelativeSlot(2, fc2_out_vec, 0)
    builder.PrependInt8Slot(3, 0, 0)
    builder.PrependUOffsetTRelativeSlot(4, fc2_opts, 0)
    op2 = builder.EndObject()

    # Operator 3: ReLU(7) -> 8
    relu2_in = create_int_vector([7])
    relu2_out_vec = create_int_vector([8])
    builder.StartObject(10)
    builder.PrependUint32Slot(0, 1, 0)
    builder.PrependUOffsetTRelativeSlot(1, relu2_in, 0)
    builder.PrependUOffsetTRelativeSlot(2, relu2_out_vec, 0)
    op3 = builder.EndObject()

    # Operator 4: FullyConnected(input=8, weights=9, bias=10) -> output=11
    fc3_in = create_int_vector([8, 9, 10])
    fc3_out_vec = create_int_vector([11])
    fc3_opts = create_fc_options()
    builder.StartObject(10)
    builder.PrependUint32Slot(0, 0, 0)
    builder.PrependUOffsetTRelativeSlot(1, fc3_in, 0)
    builder.PrependUOffsetTRelativeSlot(2, fc3_out_vec, 0)
    builder.PrependInt8Slot(3, 0, 0)
    builder.PrependUOffsetTRelativeSlot(4, fc3_opts, 0)
    op4 = builder.EndObject()

    # Operator 5: Softmax(11) -> 12
    sm_in = create_int_vector([11])
    sm_out_vec = create_int_vector([12])
    sm_opts = create_softmax_options()
    builder.StartObject(10)
    builder.PrependUint32Slot(0, 2, 0)  # opcode_index = 2 (SOFTMAX)
    builder.PrependUOffsetTRelativeSlot(1, sm_in, 0)
    builder.PrependUOffsetTRelativeSlot(2, sm_out_vec, 0)
    builder.PrependInt8Slot(3, 0, 0)
    builder.PrependUOffsetTRelativeSlot(4, sm_opts, 0)
    op5 = builder.EndObject()

    # Operators vector
    builder.StartVector(4, 6, 4)
    builder.PrependUOffsetTRelative(op5)
    builder.PrependUOffsetTRelative(op4)
    builder.PrependUOffsetTRelative(op3)
    builder.PrependUOffsetTRelative(op2)
    builder.PrependUOffsetTRelative(op1)
    builder.PrependUOffsetTRelative(op0)
    operators_vec = builder.EndVector()

    # --- SubGraph ---
    subgraph_name = builder.CreateString("main")
    inputs_vec = create_int_vector([0])
    outputs_vec = create_int_vector([12])

    builder.StartObject(6)
    builder.PrependUOffsetTRelativeSlot(0, tensors_vec, 0)
    builder.PrependUOffsetTRelativeSlot(1, inputs_vec, 0)
    builder.PrependUOffsetTRelativeSlot(2, outputs_vec, 0)
    builder.PrependUOffsetTRelativeSlot(3, operators_vec, 0)
    builder.PrependUOffsetTRelativeSlot(4, subgraph_name, 0)
    subgraph = builder.EndObject()

    builder.StartVector(4, 1, 4)
    builder.PrependUOffsetTRelative(subgraph)
    subgraphs_vec = builder.EndVector()

    # --- Operator Codes ---
    # 0: FULLY_CONNECTED (deprecated_builtin_code=9)
    builder.StartObject(4)
    builder.PrependInt8Slot(0, 9, 0)    # deprecated_builtin_code
    builder.PrependInt8Slot(2, 0, 0)    # version
    builder.PrependInt32Slot(3, 9, 0)   # builtin_code
    opcode_fc = builder.EndObject()

    # 1: RELU (deprecated_builtin_code=19)
    builder.StartObject(4)
    builder.PrependInt8Slot(0, 19, 0)
    builder.PrependInt8Slot(2, 0, 0)
    builder.PrependInt32Slot(3, 19, 0)
    opcode_relu = builder.EndObject()

    # 2: SOFTMAX (deprecated_builtin_code=25)
    builder.StartObject(4)
    builder.PrependInt8Slot(0, 25, 0)
    builder.PrependInt8Slot(2, 0, 0)
    builder.PrependInt32Slot(3, 25, 0)
    opcode_softmax = builder.EndObject()

    builder.StartVector(4, 3, 4)
    builder.PrependUOffsetTRelative(opcode_softmax)
    builder.PrependUOffsetTRelative(opcode_relu)
    builder.PrependUOffsetTRelative(opcode_fc)
    opcodes_vec = builder.EndVector()

    # --- Description ---
    desc = builder.CreateString("Hand sign classifier - Hello/Thanks")

    # --- Model ---
    builder.StartObject(7)
    builder.PrependUint32Slot(0, 3, 0)  # version = 3
    builder.PrependUOffsetTRelativeSlot(1, opcodes_vec, 0)
    builder.PrependUOffsetTRelativeSlot(2, subgraphs_vec, 0)
    builder.PrependUOffsetTRelativeSlot(3, desc, 0)
    builder.PrependUOffsetTRelativeSlot(4, buffers_vec, 0)
    model = builder.EndObject()

    builder.Finish(model, b"TFL3")

    return bytes(builder.Output())


# ═══════════════════════════════════════════════════════════
#  Main Script
# ═══════════════════════════════════════════════════════════

def main():
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("🚀  Starting TFLite conversion process...\n")

    # ─── Step 1: Load the CSV ─────────────────────────────────
    print("📂  Loading data from", DATASET_PATH, "...")
    if not os.path.exists(DATASET_PATH):
        print("❌  Error: Dataset not found!")
        return

    df = pd.read_csv(DATASET_PATH)
    print(f"   Total samples : {len(df)}")
    print(f"   Labels found  : {df['label'].unique().tolist()}")
    print(f"   Samples per label:")
    print(df["label"].value_counts().to_string(header=False))
    print()

    # ─── Step 2: Separate features and labels ─────────────────
    feature_cols = [c for c in df.columns if c != 'label']
    X = df[feature_cols].values.astype(np.float32)
    y_raw = df['label'].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_
    num_classes = len(classes)
    print(f"🏷️   Encoded {num_classes} classes: {classes.tolist()}\n")

    # ─── Step 3: Train/Test Split ─────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"📊  Training samples : {len(X_train)}")
    print(f"📊  Testing samples  : {len(X_test)}\n")

    # One-hot encode targets
    y_train_onehot = np.eye(num_classes, dtype=np.float32)[y_train]
    y_test_onehot = np.eye(num_classes, dtype=np.float32)[y_test]

    # ─── Step 4: Train Neural Network ─────────────────────────
    print("🧠  Training neural network: Input(63) → Dense(64, ReLU) → Dense(32, ReLU) → Dense({}, Softmax)".format(num_classes))

    np.random.seed(RANDOM_STATE)
    model = SimpleNeuralNet(
        input_size=63,
        hidden1_size=64,
        hidden2_size=32,
        output_size=num_classes
    )

    # Training loop
    epochs = 200
    batch_size = 8
    lr = 0.005

    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train_onehot[indices]

        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            model.forward(X_batch)
            model.backward(X_batch, y_batch, lr=lr)

        # Evaluate every 50 epochs
        if (epoch + 1) % 50 == 0:
            preds = model.predict(X_train)
            loss = cross_entropy_loss(preds, y_train_onehot)
            train_acc = np.mean(np.argmax(preds, axis=1) == y_train) * 100
            print(f"   Epoch {epoch+1:3d}/{epochs}  |  Loss: {loss:.4f}  |  Train Acc: {train_acc:.1f}%")

    # ─── Step 5: Evaluate on test set ─────────────────────────
    test_preds = model.predict(X_test)
    test_acc = np.mean(np.argmax(test_preds, axis=1) == y_test) * 100
    print(f"\n🎯  Test Accuracy: {test_acc:.1f}%\n")

    if test_acc < 80:
        print("⚠️   Warning: Test accuracy is below 80%. Consider collecting more training data.")

    # ─── Step 6: Build TFLite model ───────────────────────────
    print("📦  Building TFLite flatbuffer...")

    # TFLite FullyConnected expects weights in [output_units, input_units] layout
    W1_tflite = model.W1.T.copy()  # [64, 63]
    W2_tflite = model.W2.T.copy()  # [32, 64]
    W3_tflite = model.W3.T.copy()  # [num_classes, 32]

    tflite_bytes = build_tflite_model(
        W1_tflite, model.b1,
        W2_tflite, model.b2,
        W3_tflite, model.b3
    )

    # ─── Step 7: Save files ───────────────────────────────────
    import json
    weights_data = {
        "labels": classes.tolist(),
        "W1": model.W1.tolist(),
        "b1": model.b1.tolist(),
        "W2": model.W2.tolist(),
        "b2": model.b2.tolist(),
        "W3": model.W3.tolist(),
        "b3": model.b3.tolist(),
    }
    with open("sign_weights.json", "w") as f:
        json.dump(weights_data, f)
    print("💾  Saved weights to sign_weights.json")

    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_bytes)
    print(f"💾  Saved model to {TFLITE_MODEL_PATH} ({len(tflite_bytes) / 1024:.2f} KB)")

    with open(LABELS_PATH, 'w') as f:
        for c in classes:
            f.write(f"{c}\n")
    print(f"💾  Saved labels to {LABELS_PATH}")

    # Verify the TFLite file starts with the magic bytes
    with open(TFLITE_MODEL_PATH, 'rb') as f:
        magic = f.read(4)
    # Note: flatbuffers prepends the file_identifier at offset 4
    print(f"\n✅  TFLite file identifier check: {'PASS' if b'TFL3' in open(TFLITE_MODEL_PATH, 'rb').read(8) else 'CHECK MANUALLY'}")

    print(f"\n🎉  Conversion complete!")
    print(f"   Model: {TFLITE_MODEL_PATH} ({os.path.getsize(TFLITE_MODEL_PATH)} bytes)")
    print(f"   Labels: {LABELS_PATH}")
    print(f"   Classes: {classes.tolist()}")


if __name__ == "__main__":
    main()
