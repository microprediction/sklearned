
from sklearned.embeddings.transforms import to_log_space_1d, choice_from_dict
from tensorflow import keras
from sklearned.embeddings.activationembeddings import mostly_linear, mostly_swish
from sklearned.embeddings.optimizerembeddings import keras_optimizer_from_name, keras_optimizer_name
from sklearned.embeddings.lossembeddings import mostly_mse



def keras_lstm_11(us, n_inputs:int):
    """ Maps cube onto model and search params """
    # Inspired by https://towardsdatascience.com/using-lstms-to-forecast-time-series-4ab688386b1f
    # But see https://shiva-verma.medium.com/understanding-input-and-output-shape-in-lstm-keras-c501ee95c65e
    # Or https://towardsdatascience.com/3-steps-to-forecast-time-series-lstm-with-tensorflow-keras-ba88c6f05237
    # THIS IS NOT WORKING CURRENTLY
    # Maybe ... https://stackoverflow.com/questions/66373169/tensorflow-2-object-detection-api-numpy-version-errors/66486051#66486051
    search_params = {'epochs':int(to_log_space_1d(us[0], low=50, high=20000)),
                     'patience':int(to_log_space_1d(us[1], low=5, high=500)),
                     'jiggle_fraction':us[2]**2}

    n_search_params = len(search_params)
    n_lstm_layers = choice_from_dict(us[n_search_params], {1: 10, 2:20, 3:20, 4:10, 5:10, 6:5})
    learning_rate = to_log_space_1d( us[n_search_params+1], low=0.000001, high=0.001)
    opt_name = keras_optimizer_name(us[n_search_params + 2])
    dropout_rate = to_log_space_1d(us[n_search_params + 3], low=0.00001, high=1.0)
    keras_optimizer = keras_optimizer_from_name(opt_name=opt_name, learning_rate=learning_rate)
    info = {'keras_optimizer':opt_name,'learning_rate':learning_rate}
    loss = mostly_mse( us[n_search_params + 4])
    offset = n_search_params+5

    model = keras.Sequential()
    for layer_ndx in range(n_lstm_layers-1):
        n_units = int(to_log_space_1d(us[1*layer_ndx+offset], low=4, high=128))
        model.add(keras.layers.LSTM(units=n_units, input_shape=(1, n_inputs), return_sequences=True))
        model.add(keras.layers.Dropout(dropout_rate))
    # Last layer
    layer_ndx = n_lstm_layers
    n_units = int(to_log_space_1d(us[1 * layer_ndx + offset], low=4, high=128))
    model.add(keras.layers.LSTM(units=n_units, input_shape=(1, n_inputs), return_sequences=False))
    # Dense
    model.add(keras.layers.Dense(1,activation='linear'))
    model.compile(loss=loss, optimizer=keras_optimizer)
    return model, search_params, info