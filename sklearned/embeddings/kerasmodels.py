
# Mappings from the cube to a tuple (model, search_kwargs)
from sklearned.embeddings.transforms import to_log_space_1d, choice_from_dict
from tensorflow import keras
from sklearned.embeddings.activationembeddings import mostly_linear, mostly_swish
from sklearned.embeddings.optimizerembeddings import keras_optimizer_from_name, keras_optimizer_name
from sklearned.embeddings.lossembeddings import mostly_mse


def keras_jiggly_7(us,n_inputs:int):
    search_params = {'epochs': 500,
                     'patience': 6,
                     'jiggle_fraction': 0.5}
    learning_rate = to_log_space_1d(us[0],low=0.0001,high=0.001)
    info = {'keras_optimizer':"Adamax",'learning_rate':learning_rate}
    loss = 'mse'
    unit0 = int(to_log_space_1d(us[1],low=1.5,high=20))
    unts = [unit0,5,3,5,16,1]
    activs = ['linear','relu','relu','gelu','swish','linear']
    model = keras.Sequential()
    keras_optimizer = keras_optimizer_from_name('Adamax', learning_rate=info['learning_rate'])

    for layer_ndx,(n_units,activation) in enumerate(zip(unts,activs)):
        if layer_ndx==len(unts)-1:
            # Last layer
            model.add(keras.layers.Dense(1))
        else:
            bias_size = us[layer_ndx + 2]
            bias_initializer_0 = keras.initializers.RandomUniform(minval=-bias_size, maxval=bias_size, seed=None)
            if layer_ndx==0:
                model.add(keras.layers.Dense(n_units, activation=activation, input_shape = (1,n_inputs),
                                             bias_initializer=bias_initializer_0))
            else:
                model.add(keras.layers.Dense(n_units, activation=activation,
                                             bias_initializer=bias_initializer_0))
    model.compile(loss=loss, optimizer=keras_optimizer)
    return model, search_params, info


def keras_jiggly_10(us,n_inputs:int):
    search_params = {'epochs': 5000,
                     'patience': 6,
                     'jiggle_fraction': 0.5}
    info = {'keras_optimizer':"Adamax",'learning_rate':0.000345}
    loss = 'mse'
    unts = [2,5,3,5,16,1]
    activs = ['linear','relu','relu','gelu','swish','linear']
    model = keras.Sequential()
    keras_optimizer = keras_optimizer_from_name('Adamax', learning_rate=info['learning_rate'])

    for layer_ndx,(n_units,activation) in enumerate(zip(unts,activs)):
        if layer_ndx==len(unts)-1:
            # Last layer
            model.add(keras.layers.Dense(1))
        else:
            kernel_size = us[2 * layer_ndx]
            bias_size = us[2 * layer_ndx + 1]
            kernel_initializer_0 = keras.initializers.RandomUniform(minval=-kernel_size, maxval=kernel_size, seed=None)
            bias_initializer_0 = keras.initializers.RandomUniform(minval=-bias_size, maxval=bias_size, seed=None)
            if layer_ndx==0:
                model.add(keras.layers.Dense(n_units, activation=activation, input_shape = (1,n_inputs),
                                             kernel_initializer=kernel_initializer_0,
                                             bias_initializer=bias_initializer_0))
            else:
                model.add(keras.layers.Dense(n_units, activation=activation,
                                             kernel_initializer=kernel_initializer_0,
                                             bias_initializer=bias_initializer_0))
    model.compile(loss=loss, optimizer=keras_optimizer)
    return model, search_params, info




def keras_mostly_linear_27(us, n_inputs:int):
    """ Maps cube onto model and search params """
    search_params = {'epochs':int(to_log_space_1d(us[0], low=50, high=5000)),
                     'patience':int(to_log_space_1d(us[1], low=5, high=150)),
                     'jiggle_fraction':us[2]}

    n_search_params = len(search_params)
    n_layers = choice_from_dict(us[n_search_params], {1:5, 2: 10, 3: 20, 4: 5, 5: 5})
    learning_rate = to_log_space_1d( us[n_search_params+1], low=0.000001, high=0.001)
    opt_name = keras_optimizer_name(us[n_search_params + 2])
    keras_optimizer = keras_optimizer_from_name(opt_name=opt_name, learning_rate=learning_rate)
    info = {'keras_optimizer':opt_name,'learning_rate':learning_rate}
    loss = mostly_mse( us[n_search_params + 3])
    offset = n_search_params+4

    model = keras.Sequential()
    for layer_ndx in range(n_layers):
        n_units = int(to_log_space_1d(us[4*layer_ndx+offset], low=2, high=128))
        activation = mostly_linear( us[4*layer_ndx+offset+1] )
        kernel_size = us[4*layer_ndx+offset+2]
        bias_size = us[4 * layer_ndx + offset+3]
        kernel_initializer_0 = keras.initializers.RandomUniform(minval=-kernel_size, maxval=kernel_size, seed=None)
        bias_initializer_0 = keras.initializers.RandomUniform(minval=-bias_size, maxval=bias_size, seed=None)
        model.add(keras.layers.Dense(n_units, activation=activation, input_shape=(1, n_inputs),
                                     kernel_initializer=kernel_initializer_0,
                                     bias_initializer=bias_initializer_0))
    model.add(keras.layers.Dense(1,activation='linear'))
    model.compile(loss=loss, optimizer=keras_optimizer)
    return model, search_params, info


def keras_fast_swish_28(us, n_inputs:int):
    """ Maps cube onto model and search params """
    search_params = {'epochs':int(to_log_space_1d(us[0], low=50, high=500)),
                     'patience':int(to_log_space_1d(us[1], low=2, high=10)),
                     'jiggle_fraction':us[2]**2}

    n_search_params = len(search_params)
    n_layers = choice_from_dict(us[n_search_params], {3: 20, 4: 10, 5: 10, 6:10, 7:10 })
    learning_rate = to_log_space_1d( us[n_search_params+1], low=0.000001, high=0.001)
    opt_name = keras_optimizer_name(us[n_search_params + 2])
    keras_optimizer = keras_optimizer_from_name(opt_name=opt_name, learning_rate=learning_rate)
    info = {'keras_optimizer':opt_name,'learning_rate':learning_rate}
    loss = mostly_mse( us[n_search_params + 3])
    offset = n_search_params+4

    model = keras.Sequential()
    for layer_ndx in range(n_layers):
        n_units = int(to_log_space_1d(us[3*layer_ndx+offset], low=2, high=128))
        activation = mostly_swish( us[3*layer_ndx+offset+1] )
        kernel_size = us[3*layer_ndx+offset+2]
        kernel_initializer_0 = keras.initializers.RandomUniform(minval=-kernel_size, maxval=kernel_size, seed=None)
        model.add(keras.layers.Dense(n_units, activation=activation, input_shape=(1, n_inputs),
                                     kernel_initializer=kernel_initializer_0))
        model.add(keras.layers.Dropout(0.05))
    model.add(keras.layers.Dense(1,activation='linear'))
    model.compile(loss=loss, optimizer=keras_optimizer)
    return model, search_params, info




def keras_deeper_swish_17(us, n_inputs:int):
    """ Maps cube onto model and search params """
    search_params = {'epochs':int(to_log_space_1d(us[0], low=10, high=100)),
                     'patience':int(to_log_space_1d(us[1], low=1, high=10)),
                     'jiggle_fraction':us[2]**2}

    n_search_params = len(search_params)
    n_layers = choice_from_dict(us[n_search_params], {5: 10, 6:10, 7:10, 8:10, 9:10, 10:10 })
    learning_rate = to_log_space_1d( us[n_search_params+1], low=0.000001, high=0.001)
    opt_name = keras_optimizer_name(us[n_search_params + 2])
    keras_optimizer = keras_optimizer_from_name(opt_name=opt_name, learning_rate=learning_rate)
    info = {'keras_optimizer':opt_name,'learning_rate':learning_rate}
    loss = mostly_mse( us[n_search_params + 3])
    offset = n_search_params+4

    model = keras.Sequential()
    for layer_ndx in range(n_layers):
        n_units = int(to_log_space_1d(us[1*layer_ndx+offset], low=2, high=128))
        model.add(keras.layers.Dense(n_units, input_shape=(1, n_inputs)))
        model.add(keras.layers.Dropout(0.1))
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





KERAS_EMBEDDINGS = [ keras_jiggly_7, keras_jiggly_10, keras_mostly_linear_27, keras_fast_swish_28, keras_deeper_swish_17]


if __name__=='__main__':
    import numpy as np
    for _ in range(10):
        x = np.random.randn(1000,1,160)
        model, search_params, info = keras_jiggly_7(us=list(np.random.rand(7,1)),n_inputs=160)
        print(model.summary())
        y = model(x)
        print(np.shape(y))