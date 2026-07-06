import re
import numpy as np


# ── Text cleaning ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Apply the same cleaning pipeline used during training."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\b\d+\b", "", text)
    return text


def tokenize(text: str) -> list[str]:
    """Clean and split text into tokens."""
    return clean_text(text).split()


# ── Vocabulary ───────────────────────────────────────────────────────────────

def build_vocab(token_series) -> list[str]:
    """Build sorted vocabulary from a series of token lists (alpha-only words)."""
    return sorted(set(
        word
        for tokens in token_series
        for word in tokens
        if word[0].isalpha()
    ))


def build_word_to_index(vocab: list[str]) -> dict[str, int]:
    return {word: i for i, word in enumerate(vocab)}


# ── Feature vector ───────────────────────────────────────────────────────────

def message_to_vector(tokens: list[str], word_to_index: dict[str, int]) -> np.ndarray:
    """Convert a token list into a BoW feature vector with a leading bias term of 1."""
    x = np.zeros(len(word_to_index) + 1)
    x[0] = 1.0  # bias
    for word in tokens:
        if word in word_to_index:
            x[word_to_index[word] + 1] += 1
    return x


def build_X(token_series, word_to_index: dict[str, int]) -> np.ndarray:
    """Build the full feature matrix with a leading bias column of ones."""
    n = len(token_series)
    vocab_size = len(word_to_index)
    X = np.zeros((n, vocab_size + 1), dtype=int)
    X[:, 0] = 1  # bias column
    for i, tokens in enumerate(token_series):
        for word in tokens:
            if word in word_to_index:
                X[i, word_to_index[word] + 1] += 1
    return X


# ── Model functions ──────────────────────────────────────────────────────────

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return sigmoid(X @ w)


def loss(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    e = predict(X, w) - y
    return float(np.mean(e ** 2))


def gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    n = len(y)
    e = predict(X, w) - y
    return (2 / n) * (X.T @ e)


def accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(pred == true))


# ── Train/test split ─────────────────────────────────────────────────────────

def train_test_split(X: np.ndarray, y: np.ndarray,
                     test_frac: float = 0.25, seed: int = 0):
    r = np.random.default_rng(seed)
    idx = r.permutation(len(X))
    n_test = int(len(X) * test_frac)
    test, train = idx[:n_test], idx[n_test:]
    return X[train], y[train], X[test], y[test]
