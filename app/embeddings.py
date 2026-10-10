import numpy as np

DIM = 384
_model = None


def _get():
    global _model
    if _model is None:  # lazy: importing the app (and running tests) never downloads the model
        from fastembed import TextEmbedding
        _model = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _model


def embed(texts: list[str]) -> np.ndarray:
    v = np.array(list(_get().embed(texts)), dtype="float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)
