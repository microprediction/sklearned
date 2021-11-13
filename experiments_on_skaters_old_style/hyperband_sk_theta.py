import keras_tuner as kt
from tensorflow import keras, ones
from sklearned.challenging.surrogatedata import cached_skater_surrogate_data
from pprint import pprint
from functools import partial

SLUGGISH_MOVING_AVERAGE_BEST = {'test_error': 0.04533515537923001,
                             'train_error': 0.011145835494209172,
                             'val_error': 0.024883833225884753}

def tunertrain(d, n_input, epochs):

    def build_model(hp, n_inputs):
        model = keras.Sequential()
        learning_rate = hp.Choice('learning_rate',[0.00001, 0.0001,0.001, 0.005, 0.01, 0.02, 0.04, 0.07, 0.1])
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

        kernel_0_min = hp.Choice('kernel_0_min',[-0.01,0.01,0.1])
        kernel_0_up = hp.Choice('kernel_0_up',[0.001,0.01,0.01,0.1])
        bias_0_min = hp.Choice('bias_0_min', [-1.,-0.1,-0.01,0.0, 0.01, 0.1,1.0])
        bias_0_up = hp.Choice('bias_0_up', [0.01, 0.1, 0.2, 1.])
        kernel_initializer_0 = keras.initializers.RandomUniform(minval=kernel_0_min, maxval=kernel_0_min+kernel_0_up, seed=None)
        bias_initializer_0 = keras.initializers.RandomUniform(minval=bias_0_min, maxval=bias_0_min+bias_0_up, seed=None)
        # First layer
        model.add(keras.layers.Dense(
            units=hp.Choice('units_0', [8, 16, 32, 64, 92, 128, 156, 256]),
            activation=hp.Choice('activation_0',
                                 ['linear','relu','tanh','sigmoid','selu','elu','softsign','swish']),
            input_shape=(1, n_inputs),
            kernel_initializer=kernel_initializer_0,
            bias_initializer=bias_initializer_0))
        # Second layer
        model.add(keras.layers.Dense(
                        units = hp.Choice('units_1', [8,16,32,64]),
                        activation=hp.Choice('activation_1',['linear','relu','tanh','sigmoid','selu','elu','softsign','swish']) ))
        # Third layer
        model.add( keras.layers.Dense(
                        units=hp.Choice('units_2', [2,4,8,16,32]),
                        activation=hp.Choice('activation_2',['linear','relu','tanh','sigmoid','selu','elu','softsign'])))
        # Final layer
        model.add(keras.layers.Dense(1, activation='linear'))
        model.compile(loss='mse',optimizer=optimizer)
        return model

    p_build = partial(build_model, n_inputs=n_input)
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=250)
    tuner = kt.Hyperband(
        p_build,
        objective='val_loss',
        overwrite=True,
        max_epochs=2500)

    tuner.search(d['x_train'], d['y_train'], epochs=epochs,
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
   n_inputs = 60
   epochs = 500
   d = cached_skater_surrogate_data(skater_name='thinking_slow_and_fast', k=1, n_samples=150, n_warm=100, n_input = n_inputs,
                                       verbose = False, include_strs = None, exclude_str = '~')
   best_model = tunertrain(d=d,n_input=n_inputs, epochs=epochs)
   summary = summarize_model(d=d, model=best_model)
   ratio = summary['test_error'] / SLUGGISH_MOVING_AVERAGE_BEST['test_error']
   summary['test_error_ratio'] = ratio
   pprint(summary)