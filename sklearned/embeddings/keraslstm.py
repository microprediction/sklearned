
from sklearned.embeddings.transforms import to_log_space_1d, choice_from_dict
from tensorflow import keras
from sklearned.embeddings.activationembeddings import mostly_linear, mostly_swish
from sklearned.embeddings.optimizerembeddings import keras_optimizer_from_name, keras_optimizer_name
from sklearned.embeddings.lossembeddings import mostly_mse


def keras_lstm_shallow_11(us, n_inputs:int):
    layer_choices = {1: 30, 2:20}
    return keras_lstm_factory(us=us,n_inputs=n_inputs, layer_choices=layer_choices, units_low=2, units_high=6)


def keras_lstm_deep_14(us, n_inputs:int):
    layer_choices = {3:10, 4:5, 5:2}
    return keras_lstm_factory(us=us,n_inputs=n_inputs, layer_choices=layer_choices, units_low=2, units_high=32)


def keras_lstm_factory(us, n_inputs:int, layer_choices:dict, units_low=4, units_high=128):
    """ Maps cube onto model and search params """
    # Inspired by https://towardsdatascience.com/using-lstms-to-forecast-time-series-4ab688386b1f
    # But see https://shiva-verma.medium.com/understanding-input-and-output-shape-in-lstm-keras-c501ee95c65e
    # Or https://towardsdatascience.com/3-steps-to-forecast-time-series-lstm-with-tensorflow-keras-ba88c6f05237
    # THIS IS NOT WORKING CURRENTLY on M1
    # Maybe ... https://stackoverflow.com/questions/66373169/tensorflow-2-object-detection-api-numpy-version-errors/66486051#66486051
    search_params = {'epochs':int(to_log_space_1d(us[0], low=5, high=5000)),
                     'patience':int(to_log_space_1d(us[1], low=1, high=10)),
                     'jiggle_fraction':us[2]**2}

    n_search_params = len(search_params)
    n_lstm_layers = choice_from_dict(us[n_search_params], layer_choices)
    learning_rate = to_log_space_1d( us[n_search_params+1], low=0.000001, high=0.001)
    opt_name = keras_optimizer_name(us[n_search_params+2])
    dropout_rate = to_log_space_1d(us[n_search_params+3], low=0.00001, high=0.2)
    keras_optimizer = keras_optimizer_from_name(opt_name=opt_name, learning_rate=learning_rate)
    info = {'keras_optimizer':opt_name,'learning_rate':learning_rate}
    loss = mostly_mse( us[n_search_params + 4])
    offset = n_search_params+5

    model = keras.Sequential()
    # First layer is permutation, so as to align with conventions for the dense networks
    model.add(keras.layers.Permute((2, 1), input_shape=(1, n_inputs)))

    # n_inputs = timesteps
    layer_ndx = 0
    n_units = int(to_log_space_1d(us[1 * layer_ndx + offset], low=units_low, high=units_high))
    model.add(keras.layers.LSTM(units=n_units, input_shape=(1,n_inputs), return_sequences=True))
    model.add(keras.layers.Dropout(dropout_rate))

    # Intermediate layers
    for layer_ndx in range(1,n_lstm_layers-1):
        n_units = int(to_log_space_1d(us[1*layer_ndx+offset], low=units_low, high=units_high))
        model.add(keras.layers.LSTM(units=n_units, return_sequences=True))
        model.add(keras.layers.Dropout(dropout_rate))
    # Last layer
    layer_ndx = n_lstm_layers
    n_units = int(to_log_space_1d(us[1 * layer_ndx + offset], low=units_low, high=units_high))
    model.add(keras.layers.LSTM(units=n_units, return_sequences=False))

    # Dense
    model.add(keras.layers.Dense(1,activation='linear'))


    model.compile(loss=loss, optimizer=keras_optimizer)
    return model, search_params, info

KERAS_LSTM_MODELS = [keras_lstm_deep_14,keras_lstm_shallow_11]


if __name__=='__main__':
    import numpy as np
    for _ in range(20):
        x = np.random.randn(1000,1,20)
        model, search_params, info = keras_lstm_shallow_11(us=list(np.random.rand(11,1)),n_inputs=20)
        print(model.summary())
        y = model(x)
        print(np.shape(y))
