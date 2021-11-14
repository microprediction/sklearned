import numpy as np
from tensorflow import keras
from sklearned.challenging.surrogatedata import cached_skater_surrogate_data
from sklearned.augment.affine import affine, jiggle
from sklearned.challenging.surrogateio import read_champion_metrics, save_champion_metrics, save_champion_model,\
    save_champion_onnx, save_champion_weights, save_champion_info, save_champion_tensorflow
from pprint import pprint
import os
from sklearned.wherami import CHAMPION_METRICS_PATH, CHAMPION_MODELS_PATH, CHAMPION_WEIGHTS_PATH, CHAMPION_ONNX_PATH, CHAMPION_INFO_PATH
from sklearned.augment.cleaning import remove_surrogate_outliers
from sklearned.challenging.dimensional import squeeze_out_middle

# Utilities belonging elsewhere

DREADFUL = 1000000
POOR_METRICS = {'test_error': DREADFUL,
                'train_error': DREADFUL,
                'val_error': DREADFUL}


def challenge(model, skater_name: str, info:dict, epochs=200, jiggle_fraction=0.1, symmetries=None,
              k=1, n_real=60, n_samples=150, n_warm=100, n_input=80, patience=50, with_metrics=False, verbose=2):
    """
           See how a model architecture performs against the champ

    :param model:
    :param skater_name:        tsa_p2_d0_q0 or similar
    :param info                dict of stuff to save if challenge is successful
    :param epochs:             Maximum number of training epochs
    :param jiggle_fraction:    Fraction of data points in first half of time-series to jiggle
    :param symmetries:         List of implied symmetries (translations, if scalar)
    :param k                   Number of steps to forecast ahead
    :param n_real              Number of live time-series to use
    :param n_samples           Number of sample points to use per time-series
    :param n_warm              Length of history used in warm-up
    :param n_input             Number of lags to use (length of input vector to neural network model)
    :param patience            Keras early stopping patience parameter
    :param search_params       dict that will be saved if the challenge is successful
    :return:
    """
    search_params = {'epochs':epochs,
                     'patience':patience,
                     'jiggle_fraction':jiggle_fraction}
    info.update(search_params)

    os.makedirs(CHAMPION_METRICS_PATH, exist_ok=True)
    os.makedirs(CHAMPION_MODELS_PATH, exist_ok=True)
    os.makedirs(CHAMPION_WEIGHTS_PATH, exist_ok=True)
    os.makedirs(CHAMPION_ONNX_PATH, exist_ok=True)
    os.makedirs(CHAMPION_INFO_PATH, exist_ok=True)

    print('Champion data')
    champion_metrics = read_champion_metrics(skater_name=skater_name, k=k, n_input=n_input)
    pprint(champion_metrics)

    print('Surrogate data ')
    d = cached_skater_surrogate_data(skater_name=skater_name, k=k, n_samples=150, n_warm=290, n_input=n_input)
    d = remove_surrogate_outliers(d)
    if symmetries is None:
        symmetries = [0.995, 1.0, 1.001]
    aug_X, aug_y = affine(X=d['x_train'], y=d['y_train'], s=symmetries)
    jiggle_X = jiggle(aug_X, jiggle_fraction=jiggle_fraction)

    print('Training')
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
    model.fit(x=jiggle_X, y=aug_y, epochs=epochs, verbose=verbose, callbacks=[callback], use_multiprocessing=False, workers=1)

    from sklearned.challenging.savemaybe import assess_and_maybe_save
    challenger_metrics, test_error_ratio = assess_and_maybe_save(model, info, d, champion_metrics, skater_name, k, n_input)

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
