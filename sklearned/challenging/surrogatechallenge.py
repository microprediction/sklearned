import numpy as np
from tensorflow import keras
from sklearned.challenging.surrogatedata import cached_skater_surrogate_data
from sklearned.augment.affine import affine, jiggle
from sklearned.challenging.surrogateio import read_model_champion_metrics, save_champion_metrics, save_champion_model, save_champion_onnx, save_champion_weights
from pprint import pprint
import os
from sklearned.wherami import CHAMPION_METRICS_PATH, CHAMPION_MODELS_PATH, CHAMPION_WEIGHTS_PATH, CHAMPION_ONNX_PATH


# Utilities belonging elsewhere

DREADFUL = 1000000
POOR_METRICS = {'test_error': DREADFUL,
                'train_error': DREADFUL,
                'val_error': DREADFUL}


def challenge(model, skater_name: str, epochs=200, jiggle_fraction=0.1, symmetries=None,
              k=1, n_real=60, n_samples=150, n_warm=100, n_input=80, patience=50, with_metrics=False, verbose=2):
    """
           See how a model architecture performs against the champ

    :param model:
    :param skater_name:        tsa_p2_d0_q0 or similar
    :param epochs:             Maximum number of training epochs
    :param jiggle_fraction:    Fraction of data points in first half of time-series to jiggle
    :param symmetries:         List of implied symmetries (translations, if scalar)
    :param k                   Number of steps to forecast ahead
    :param n_real              Number of live time-series to use
    :param n_samples           Number of sample points to use per time-series
    :param n_warm              Length of history used in warm-up
    :param n_input             Number of lags to use (length of input vector to neural network model)
    :param patience            Keras early stopping patience parameter
    :return:
    """
    os.makedirs(CHAMPION_METRICS_PATH, exist_ok=True)
    os.makedirs(CHAMPION_MODELS_PATH, exist_ok=True)
    os.makedirs(CHAMPION_WEIGHTS_PATH, exist_ok=True)
    os.makedirs(CHAMPION_ONNX_PATH, exist_ok=True)

    print('Champion data')
    champion_metrics = read_model_champion_metrics(skater_name=skater_name, k=k, n_input=n_input)
    pprint(champion_metrics)

    print('Surrogate data ')
    d = cached_skater_surrogate_data(skater_name=skater_name, k=1, n_samples=150, n_warm=290, n_input=80)
    if symmetries is None:
        symmetries = [0.95, 0.975, 0.99, 1.0, 1.01, 1.025, 1.05]
    aug_X, aug_y = affine(X=d['x_train'], y=d['y_train'], s=symmetries)
    jiggle_X = jiggle(aug_X, jiggle_fraction=jiggle_fraction)

    print('Training')
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
    model.fit(x=jiggle_X, y=aug_y, epochs=epochs, verbose=verbose, callbacks=[callback])

    y_test_hat = model.predict(d['x_test'])
    test_error = float(keras.metrics.mean_squared_error(y_test_hat[:, 0, 0], d['y_test'][:, 0]))
    y_val_hat = model.predict(d['x_val'])
    val_error = float(keras.metrics.mean_squared_error(y_val_hat[:, 0, 0], d['y_val'][:, 0]))
    y_train_hat = model.predict(d['x_train'])
    train_error = float(keras.metrics.mean_squared_error(y_train_hat[:, 0, 0], d['y_train'][:, 0]))

    # Innovations relative to last value
    dy_surrogate = list(y_test_hat[:, 0, 0] - d['x_test'][:, 0, -1])
    dy_model = list(d['y_test'][:, 0] - d['x_test'][:, 0, -1])
    rho = np.corrcoef(x=dy_surrogate, y=dy_model)[0][1]

    challenger_metrics = {"train_error": train_error / d['y_train_typical'],
                          "val_error": val_error / d['y_val_typical'],
                          "test_error": test_error / d['y_test_typical'],
                          "rho": rho}

    test_error_ratio = challenger_metrics['test_error'] / champion_metrics['test_error']
    pprint(challenger_metrics)
    pprint('Test error ratio to champion is ' + str(test_error_ratio))
    if test_error_ratio < 0.95:
        print('You won the challenge ... saving new champion metrics')
        save_champion_metrics(metrics=challenger_metrics, skater_name=skater_name, k=k, n_input=n_input)
        save_champion_model(model=model, skater_name=skater_name, k=k, n_input=n_input)
        save_champion_weights(model=model, skater_name=skater_name, k=k, n_input=n_input)
        save_champion_onnx(model=model, skater_name=skater_name, k=k, n_input=n_input)

    if with_metrics:
        return model, challenger_metrics, test_error_ratio
    else:
        return model


if __name__ == '__main__':
    skater_name = 'elo_fastest_residual_balanced_ensemble'


    def build_model(n_inputs):
        model = keras.Sequential()
        kernel_initializer_0 = keras.initializers.RandomUniform(minval=0.01, maxval=0.02, seed=None)
        bias_initializer_0 = keras.initializers.RandomUniform(minval=0.01, maxval=0.21, seed=None)
        model.add(keras.layers.Dense(80, activation="linear", input_shape=(1, n_inputs),
                                     kernel_initializer=kernel_initializer_0,
                                     bias_initializer=bias_initializer_0))
        model.add(keras.layers.Dense(16, activation='softsign'))
        model.add(keras.layers.Dense(2, activation="tanh"))  # selu
        model.add(keras.layers.Dense(1, activation="linear"))
        optimizer = keras.optimizers.Adagrad(learning_rate=0.005)
        model.compile(loss='mse', optimizer=optimizer)
        model.compile(loss='mse', optimizer=optimizer)
        return model


    n_inputs = 80
    skater_name = 'tsa_p2_d0_q0'
    model = build_model(n_inputs)
    challenge(model=model, skater_name=skater_name, epochs=500)
