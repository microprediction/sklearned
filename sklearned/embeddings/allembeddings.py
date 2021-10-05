from sklearned.embeddings.kerasmodels import KERAS_EMBEDDINGS

EMEDDINGS = KERAS_EMBEDDINGS


def embedding_from_name(name):
    valid = [f for f in KERAS_EMBEDDINGS if f.__name__ == name]
    return valid[0] if len(valid) == 1 else None