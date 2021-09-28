import keras_tuner as kt
from tensorflow import keras, ones
from sklearned.tasks.lastvalue import last_value_task
from pprint import pprint
from functools import partial


def tunertrain(d, n_input, epochs=50):

    def build_model(hp, n_input):
        model = keras.Sequential()
        model.add(keras.layers.Dense(
            hp.Choice('units', [8, 16, 24, 32, 64]),activation=hp.Choice('activation_1',['linear','relu']), input_shape=(1,n_input) ))
        model.add(keras.layers.Dense(8, activation=hp.Choice('activation_2',['linear','relu'])))
        model.add(keras.layers.Dense(1, activation='linear'))
        model.compile(loss='mse')
        return model

    p_build = partial(build_model, n_input=n_input)
    tuner = kt.Hyperband(
        p_build,
        objective='val_loss',
        overwrite=True,
        max_epochs=100)

    tuner.search(d['x_train'], d['y_train'], epochs=epochs, validation_data=(d['x_val'], d['y_val']))
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
   n_input = 20
   d = last_value_task(n_input=20, n_train=1000)
   best_model = tunertrain(d=d,n_input=n_input)
   summary = summarize_model(d=d, model=best_model)
   pprint(summary)