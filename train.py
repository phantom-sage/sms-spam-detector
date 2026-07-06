import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model_utils import (
    tokenize, build_vocab, build_word_to_index, build_X,
    predict, loss, gradient, accuracy, train_test_split, sigmoid,
)

# ── Config ───────────────────────────────────────────────────────────────────

DATA_PATH  = "spam.csv"
MODEL_DIR  = "model"
LR         = 0.1
EPOCHS     = 10_000
TOL        = 1e-6

# ── Load dataset ─────────────────────────────────────────────────────────────

sms = pd.read_csv(DATA_PATH, encoding="latin-1")
sms = sms[["v1", "v2"]]

# ── Clean & tokenize ─────────────────────────────────────────────────────────

sms["tokens"] = sms["v2"].apply(tokenize)

assert sms["v2"].isna().sum() == 0, "NaN values found in v2 — check the dataset"

# ── Build vocabulary & feature matrix ────────────────────────────────────────

vocab        = build_vocab(sms["tokens"])
word_to_index = build_word_to_index(vocab)

print(f"Vocabulary size: {len(vocab)}")

X = build_X(sms["tokens"], word_to_index)

# ── Encode labels ─────────────────────────────────────────────────────────────

label_names       = sorted(set(sms["v1"]))          # ['ham', 'spam']
label_to_int      = {name: i for i, name in enumerate(label_names)}
y                 = np.array([label_to_int[l] for l in sms["v1"]])

print(f"Label mapping: {label_to_int}")

# ── Train / test split ────────────────────────────────────────────────────────

X_train, y_train, X_test, y_test = train_test_split(X, y)

# ── Training loop ─────────────────────────────────────────────────────────────

def train_loop(X, y, w, epochs=EPOCHS, tol=TOL, lr=LR, plot=True):
    hist       = []
    prev_loss  = float("inf")
    bar_width  = 30

    for epoch in range(epochs):
        g = gradient(X, y, w)
        w = w - lr * g
        l = loss(X, y, w)
        hist.append(l)

        # progress bar
        pct      = (epoch + 1) / epochs
        filled   = int(bar_width * pct)
        bar      = "█" * filled + "░" * (bar_width - filled)
        print(f"\r  [{bar}] {pct*100:5.1f}%  epoch {epoch+1}/{epochs}  loss {l:.6f}", end="", flush=True)

        if abs(prev_loss - l) < tol:
            print(f"\n  Converged at epoch {epoch + 1}")
            break
        prev_loss = l
    else:
        print()  # newline after full run

    if plot:
        plt.plot(hist)
        plt.xlabel("epoch")
        plt.ylabel("MSE")
        plt.title("Training Loss")
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, "loss_curve.png"))
        plt.show()

    return w, hist[-1]


w = np.zeros(X.shape[-1])
w, final_loss = train_loop(X_train, y_train, w)
print(f"Final training loss: {final_loss:.6f}")

# ── Evaluate ──────────────────────────────────────────────────────────────────

y_hat = predict(X_test, w)
acc   = accuracy((y_hat >= 0.5).astype(int), y_test)
print(f"Test accuracy: {acc * 100:.2f}%")

# ── Save model weights ────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)

np.save(os.path.join(MODEL_DIR, "weights.npy"), w)
np.save(os.path.join(MODEL_DIR, "vocab.npy"), np.array(vocab))

print(f"Weights saved to {MODEL_DIR}/weights.npy")
print(f"Vocab saved  to  {MODEL_DIR}/vocab.npy")

# ── Test with real-world messages ─────────────────────────────────────────────

def predict_message(message: str) -> tuple[str, float]:
    from model_utils import tokenize, message_to_vector
    tokens = tokenize(message)
    x      = message_to_vector(tokens, word_to_index)  # includes bias at x[0]
    prob   = float(sigmoid(x @ w))
    label  = "spam" if prob >= 0.5 else "ham"
    return label, prob


test_messages = [
    "Congratulations! You have won a free ticket. Call now to claim your prize!",
    "Hey, are we still meeting for lunch tomorrow?",
    "WINNER!! As a valued network customer you have been selected to receive a 900 prize reward!",
    "Can you pick up some milk on your way home?",
]

print("\n── Real-world predictions ──")
for msg in test_messages:
    label, prob = predict_message(msg)
    print(f"[{label.upper()}] ({prob:.2%}) — {msg[:60]}")
