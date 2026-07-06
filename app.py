import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from model_utils import tokenize, build_word_to_index, message_to_vector, sigmoid

app = Flask(__name__)

# ── Load model once at startup ────────────────────────────────────────────────

MODEL_DIR = "model"

w     = np.load(os.path.join(MODEL_DIR, "weights.npy"))
vocab = np.load(os.path.join(MODEL_DIR, "vocab.npy"), allow_pickle=True).tolist()
word_to_index = build_word_to_index(vocab)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "empty message"}), 400

    tokens = tokenize(message)
    x      = message_to_vector(tokens, word_to_index)
    prob   = float(sigmoid(x @ w))
    label  = "SPAM" if prob >= 0.5 else "HAM"

    return jsonify({
        "label": label,
        "probability": round(prob * 100, 2),
    })


if __name__ == "__main__":
    app.run(debug=True)
