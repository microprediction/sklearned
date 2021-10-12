
# Mappings from the cube to a tuple (model, search_kwargs)
from sklearned.embeddings.transforms import to_log_space_1d, choice_from_dict
from tensorflow import keras
from sklearned.embeddings.activationembeddings import mostly_linear
from sklearned.embeddings.optimizerembeddings import keras_optimizer_from_name, keras_optimizer_name
from sklearned.embeddings.lossembeddings import mostly_mse


def keras_mostly_linear_27(us, n_inputs:int):
    """ Maps cube onto model and search params """
    search_params = {'epochs':int(to_log_space_1d(us[0], low=50, high=5000)),
                     'patience':int(to_log_space_1d(us[1], low=5, high=150)),
                     'jiggle_fraction':us[2]}

    n_search_params = len(search_params)
    n_layers = choice_from_dict(us[n_search_params], {1:5, 2: 10, 3: 20, 4: 5, 5: 5})
    learning_rate = to_log_space_1d( us[n_search_params+1], low=0.00001, high=0.1)
    opt_name = keras_optimizer_name(us[n_search_params + 2])
    keras_optimizer = keras_optimizer_from_name(opt_name=opt_name, learning_rate=learning_rate)
    info = {'keras_optimizer':opt_name,'learning_rate':learning_rate}
    loss = mostly_mse( us[n_search_params + 3])
    offset = n_search_params+4

    model = keras.Sequential()
    for layer_ndx in range(n_layers):
        n_units = int(to_log_space_1d(us[4*layer_ndx+offset], low=2, high=128))
        activation = mostly_linear( us[4*layer_ndx+offset] )
        kernel_size = us[4*layer_ndx+offset]
        bias_size = us[4 * layer_ndx + offset]
        kernel_initializer_0 = keras.initializers.RandomUniform(minval=-kernel_size, maxval=kernel_size, seed=None)
        bias_initializer_0 = keras.initializers.RandomUniform(minval=-bias_size, maxval=bias_size, seed=None)
        model.add(keras.layers.Dense(n_units, activation=activation, input_shape=(1, n_inputs),
                                     kernel_initializer=kernel_initializer_0,
                                     bias_initializer=bias_initializer_0))
    model.add(keras.layers.Dense(1,activation='linear'))
    model.compile(loss=loss, optimizer=keras_optimizer)
    return model, search_params, info


def keras_linear(us, n_inputs:int):
    min_kernel = us[0]
    max_kernel = us[0] + us[1] + 0.001
    bias_size = us[2]
    layers_0 = int( to_log_space_1d( us[3], low=8, high=128 ) )
    layers_1 = int( to_log_space_1d( us[4], low=2, high=128) )
    layers_2 = int( to_log_space_1d( us[5], low=2, high=16) )
    learning_rate = to_log_space_1d( us[6], low=0.00001, high=0.05)
    epochs = int( to_log_space_1d( us[7], low=50, high=1000 ))  # 5000
    patience = int( to_log_space_1d( us[8], low=5, high=50))    # 5
    jiggle_fraction = us[9]

    search_params = {'epochs': epochs,
                    'patience': patience,
                    'jiggle_fraction': jiggle_fraction,
                    }

    def build_linear_model(n_inputs):
        model = keras.Sequential()
        kernel_initializer_0 = keras.initializers.RandomUniform(minval=min_kernel, maxval=max_kernel, seed=None)
        bias_initializer_0 = keras.initializers.RandomUniform(minval=-bias_size, maxval=bias_size, seed=None)
        model.add(keras.layers.Dense(layers_0, activation="linear", input_shape=(1, n_inputs),
                                                             kernel_initializer=kernel_initializer_0,
                                                             bias_initializer=bias_initializer_0))
        model.add(keras.layers.Dense(layers_1, activation='linear'))
        model.add(keras.layers.Dense(layers_2, activation="linear"))  # selu
        model.add(keras.layers.Dense(1, activation="linear"))
        model.compile(loss='mse', optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate))
        return model

    model = build_linear_model(n_inputs=n_inputs)

    return model, search_params


KERAS_EMBEDDINGS = [keras_linear, keras_mostly_linear_27]