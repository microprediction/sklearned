from sklearned.embeddings.kerasmodels import KERAS_EMBEDDINGS
from sklearned.embeddings.keraslstm import KERAS_LSTM_MODELS
from sklearned.embeddings.kerastcn import KERAS_TCN_MODELS
from sklearned.embeddings.kerascnn import KERAS_CNN_MODELS

EMBEDDINGS = KERAS_EMBEDDINGS + KERAS_LSTM_MODELS + KERAS_TCN_MODELS + KERAS_CNN_MODELS


def embedding_from_name(name):
    valid = [f for f in EMBEDDINGS if f.__name__ == name]
    return valid[0] if len(valid) == 1 else None