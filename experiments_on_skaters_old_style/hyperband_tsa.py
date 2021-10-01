import keras_tuner as kt
from tensorflow import keras, ones
from sklearned.tasks.skatertasks import cached_surrogate_data
from pprint import pprint
from functools import partial

SLUGGISH_MOVING_AVERAGE_BEST = {'test_error': 0.04533515537923001,
                             'train_error': 0.011145835494209172,
                             'val_error': 0.024883833225884753}

def tunertrain(d, n_input, epochs):

    def build_model(hp, n_inputs):
        model = keras.Sequential()
        learning_rate = hp.Choice('learni   ng_rate',[0.00001, 0.0001,0.001, 0.005, 0.01, 0.02, 0.04, 0.07, 0.1])
        optimizer_name = hp.Choice('optimizer_name',['adam','adagrad','rmsprop','sgd'])
        if optimizer_name == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'adagrad':
            optimizer = keras.optimizers.Adagrad(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            raise ValueError("missing case")

        # Characteristics of layers
        n_layers = hp.Choice('num_layers',[1,2,3,4])
        for layer_ndx in range(n_layers):
            units = hp.Choice('units_'+str(layer_ndx),[2,4,8,16,32,64,128])
            activation = hp.Choice('activation_'+str(layer_ndx),['softsign','linear','tanh','selu','elu','relu'])
            kernel_init = hp.Choice('kernel_'+str(layer_ndx),[0.001,0.01,0.1,1.,10.])
            bias_init = hp.Choice('bias_' + str(layer_ndx), [0.001, 0.01, 0.1, 1., 10.])
            bias_offset = hp.Choice('bias_' + str(layer_ndx), [-0.1, -0.01, 0., 0.01, 0.1])
            layer_kwargs = dict(units = units,
                                activation = activation,
                                kernel_initializer=keras.initializers.RandomUniform(minval=-kernel_init, maxval=kernel_init, seed=None),
                                bias_initializer = keras.initializers.RandomUniform(minval=-bias_init+bias_offset, maxval=bias_init+bias_offset, seed=None))
            if layer_ndx==1:
               layer_kwargs['input_shape']=(1, n_inputs)
            model.add(keras.layers.Dense(**layer_kwargs))
        model.add(keras.layers.Dense(1, activation='linear'))
        dropout = hp.Choice('dropout',[0.0,0.05,0.2])
        if dropout>0:
            model.add(keras.layers.Dropout(rate=dropout))

        model.compile(loss='mse',optimizer=optimizer)
        return model

    p_build = partial(build_model, n_inputs=n_input)
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=250)
    tuner = kt.Hyperband(
        p_build,
        objective='val_loss',
        overwrite=True,
        max_epochs=2500)

    from sklearned.augment.affine import affine, jiggle
    aug_X, aug_y = affine(X=d['x_train'], y=d['y_train'], s=[0.95, 0.975, 0.99, 1.0, 1.01, 1.025, 1.05])
    jiggle_X = jiggle(aug_X, jiggle_fraction=0.1)

    tuner.search(jiggle_X, aug_y, epochs=epochs,
                 validation_data=(d['x_val'], d['y_val']),callbacks = [callback])
    print(tuner.results_summary())
    best_model = tuner.get_best_models()[0]
    return best_model


def summarize_model(d, model):
    y_test_hat = model.predict(d['x_test'])
    test_error = float(keras.metrics.mean_squared_error(y_test_hat[:, 0, 0], d['y_test'][:, 0]))
    y_val_hat = model.predict(d['x_val'])
    val_error = float(keras.metrics.mean_squared_error(y_val_hat[:, 0, 0], d['y_val'][:, 0]))
    y_train_hat = model.predict(d['x_train'])
    train_error = float(keras.metrics.mean_squared_error(y_train_hat[:, 0, 0], d['y_train'][:, 0]))
    summary = {"train_error": train_error,
            "val_error": val_error,
            "test_error": test_error}
    return summary

if __name__=='__main__':
   n_inputs = 80
   epochs = 500
   d = cached_surrogate_data(fname='tsa_p2_d0_q0', k=1, n_real=60, n_samples=150, n_warm=100, n_input=n_inputs)
   best_model = tunertrain(d=d,n_input=n_inputs, epochs=epochs)
   summary = summarize_model(d=d, model=best_model)
   ratio = summary['test_error'] / SLUGGISH_MOVING_AVERAGE_BEST['test_error']
   summary['test_error_ratio'] = ratio
   pprint(summary)