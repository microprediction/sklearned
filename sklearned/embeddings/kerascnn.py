from sklearned.embeddings.transforms import to_log_space_1d, choice_from_dict
from tensorflow import keras
from sklearned.embeddings.optimizerembeddings import keras_optimizer_from_name, keras_optimizer_name
from sklearned.embeddings.lossembeddings import mostly_mse

# be aware of https://github.com/philipperemy/keras-cnn/issues/211


def keras_cnn_shallow_10(us, n_inputs: int):
    layer_choices = {1: 30}
    return keras_cnn_factory(us=us, n_inputs=n_inputs, layer_choices=layer_choices, filters_low=4, filters_high=64)


def keras_cnn_deep_14(us, n_inputs: int):
    layer_choices = {2: 30, 3: 10, 4: 5, 5: 1}
    return keras_cnn_factory(us=us, n_inputs=n_inputs, layer_choices=layer_choices, filters_low=2, filters_high=16)


def keras_cnn_factory(us, n_inputs: int, layer_choices: dict, filters_low: int, filters_high: int):
    """ Maps cube onto model and search params """
    search_params = {'epochs': int(to_log_space_1d(us[0], low=5, high=5000)),
                     'patience': int(to_log_space_1d(us[1], low=1, high=10)),
                     'jiggle_fraction': us[2] ** 2}

    n_search_params = len(search_params)
    n_cnn_layers = choice_from_dict(us[n_search_params], layer_choices)
    learning_rate = to_log_space_1d(us[n_search_params + 1], low=0.000001, high=0.001)
    opt_name = keras_optimizer_name(us[n_search_params + 2])
    dropout_rate = to_log_space_1d(us[n_search_params + 3], low=0.00001, high=0.2)
    keras_optimizer = keras_optimizer_from_name(opt_name=opt_name, learning_rate=learning_rate)
    info = {'keras_optimizer': opt_name, 'learning_rate': learning_rate}
    loss = mostly_mse(us[n_search_params + 4])
    offset = n_search_params + 5

    model = keras.Sequential()
    # First layer is permutation, so as to align with conventions for the dense networks
    model.add(keras.layers.Permute((2, 1), input_shape=(1, n_inputs)))

    # n_inputs = timesteps
    layer_ndx = 0
    n_filters = int(to_log_space_1d(us[1 * layer_ndx + offset], low=filters_low, high=filters_high))
    model.add(keras.layers.Conv1D(filters=n_filters, input_shape=(n_inputs, 1), kernel_size=3))

    for layer_ndx in range(1, n_cnn_layers):
        n_filters = int(to_log_space_1d(us[1 * layer_ndx + offset], low=filters_low, high=filters_high))
        model.add(keras.layers.Conv1D(filters=n_filters, input_shape=(n_inputs, 1), kernel_size=3))

    # Tails
    layer_ndx = n_cnn_layers
    n_units = int(to_log_space_1d(us[1 * layer_ndx + offset], low=10, high=200))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(n_units))
    model.add(keras.layers.Dense(1, activation='linear'))

    model.compile(loss=loss, optimizer=keras_optimizer)
    return model, search_params, info


KERAS_CNN_MODELS = [keras_cnn_shallow_10, keras_cnn_deep_14]

if __name__ == '__main__':
    import numpy as np

    for _ in range(20):
        x = np.random.randn(1000, 1, 20)
        model, search_params, info = keras_cnn_deep_14(us=list(np.random.rand(13, 1)), n_inputs=20)
        print(model.summary())
        y = model(x)
        print(np.shape(y))
