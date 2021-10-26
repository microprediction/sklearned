
from sklearned.embeddings.transforms import to_log_space_1d, choice_from_dict
from tensorflow import keras
from sklearned.embeddings.optimizerembeddings import keras_optimizer_from_name, keras_optimizer_name
from sklearned.embeddings.lossembeddings import mostly_mse
from tcn import TCN
# be aware of https://github.com/philipperemy/keras-tcn/issues/211



def keras_tcn_shallow_10(us, n_inputs:int):
    layer_choices = {1: 30}
    return keras_tcn_factory(us=us,n_inputs=n_inputs, layer_choices=layer_choices, filters_low=4, filters_high=64)


def keras_tcn_deep_14(us, n_inputs:int):
    layer_choices = {2: 30, 3:10, 4:5}
    return keras_tcn_factory(us=us,n_inputs=n_inputs, layer_choices=layer_choices, filters_low=2, filters_high=16)



def keras_tcn_factory(us, n_inputs:int, layer_choices:dict, filters_low:int, filters_high:int):
    """ Maps cube onto model and search params """
    search_params = {'epochs':int(to_log_space_1d(us[0], low=5, high=5000)),
                     'patience':int(to_log_space_1d(us[1], low=1, high=10)),
                     'jiggle_fraction':us[2]**2}

    n_search_params = len(search_params)
    n_tcn_layers = choice_from_dict(us[n_search_params], layer_choices)
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
    return_sequences = n_tcn_layers>1
    n_filters = int(to_log_space_1d(us[1 * layer_ndx + offset], low=filters_low, high=filters_high))
    model.add(TCN(input_shape=(n_inputs,1), nb_filters=n_filters, dropout_rate=dropout_rate,return_sequences=return_sequences))

    for layer_ndx in range(1,n_tcn_layers-1):
        return_sequences = True
        n_filters = int(to_log_space_1d(us[1 * layer_ndx + offset], low=filters_low, high=filters_high))
        model.add(TCN(nb_filters=n_filters, dropout_rate=dropout_rate, return_sequences=return_sequences))

    if n_tcn_layers>1:
        layer_ndx = 1
        return_sequences = False
        n_filters = int(to_log_space_1d(us[1 * layer_ndx + offset], low=filters_low, high=filters_high))
        model.add(TCN(nb_filters=n_filters, dropout_rate=dropout_rate,
                      return_sequences=return_sequences))

    # Dense
    model.add(keras.layers.Dense(1,activation='linear'))


    model.compile(loss=loss, optimizer=keras_optimizer)
    return model, search_params, info

KERAS_TCN_MODELS = [keras_tcn_shallow_10]


if __name__=='__main__':
    import numpy as np
    for _ in range(20):
        x = np.random.randn(1000,1,20)
        model, search_params, info = keras_tcn_shallow_10(us=list(np.random.rand(10,1)),n_inputs=20)
        print(model.summary())
        y = model(x)
        print(np.shape(y))
