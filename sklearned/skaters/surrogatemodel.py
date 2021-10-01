from tensorflow import keras
from sklearned.skaters.surrogatedata import cached_skater_surrogate_data
from sklearned.augment.affine import affine, jiggle
from pprint import pprint
import os
from sklearned.wherami import CHAMPION_METRICS_PATH
import json

# Utilities belonging elsewhere

DREADFUL = 1000000
POOR_METRICS =  {'test_error': DREADFUL,
                 'train_error': DREADFUL,
                 'val_error': DREADFUL}

def model_champion_file(skater_name, k:int, n_input:int):
    return CHAMPION_METRICS_PATH + os.path.sep + skater_name + '_'+str(k)+'_'+str(n_input)+'.json'


def read_model_champion_metrics(**kwargs):
    try:
        with open(model_champion_file(**kwargs),'r') as fp:
            metrics = json.load(fp)
    except FileNotFoundError:
        metrics = POOR_METRICS
    return metrics


def save_model_champion_metrics(metrics:dict,**kwargs):
    with open(model_champion_file(**kwargs),'w') as fp:
        json.dump(metrics, fp)


def challenge(model, skater_name:str, epochs=200, jiggle_fraction=0.1, symmetries=None,
              k=1, n_real=60, n_samples=150, n_warm=100, n_input=80, patience=50):
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

    print('Champion data')
    champion_metrics = read_model_champion_metrics(skater_name=skater_name, k=k, n_input=n_input)
    pprint(champion_metrics)

    print('Surrogate data ')
    d = cached_skater_surrogate_data(skater_name=skater_name, k=1, n_real=60, n_samples=150, n_warm=100, n_input=80)
    if symmetries is None:
        symmetries=[0.95, 0.975, 0.99, 1.0, 1.01, 1.025, 1.05]
    aug_X, aug_y = affine(X=d['x_train'], y=d['y_train'], s=symmetries)
    jiggle_X = jiggle(aug_X, jiggle_fraction=jiggle_fraction)

    print('Training')
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
    model.fit(x=jiggle_X, y=aug_y, epochs=epochs, verbose=1, callbacks=[callback])

    y_test_hat = model.predict(d['x_test'])
    test_error = float(keras.metrics.mean_squared_error(y_test_hat[:, 0, 0], d['y_test'][:, 0]))
    y_val_hat = model.predict(d['x_val'])
    val_error = float(keras.metrics.mean_squared_error(y_val_hat[:, 0, 0], d['y_val'][:, 0]))
    y_train_hat = model.predict(d['x_train'])
    train_error = float(keras.metrics.mean_squared_error(y_train_hat[:, 0, 0], d['y_train'][:, 0]))
    challenger_metrics = {"train_error": train_error / d['y_train_typical'],
                           "val_error": val_error / d['y_val_typical'],
                           "test_error": test_error / d['y_test_typical']}

    test_error_ratio = challenger_metrics['test_error'] / champion_metrics['test_error']
    pprint(challenger_metrics)
    pprint('Test error ratio to champion is '+str(test_error_ratio))
    if test_error_ratio<0.95:
        print('You won the challenge ... saving new champion metrics')
        save_model_champion_metrics(skater_name=skater_name, k=k, n_input=n_input, metrics=challenger_metrics)

    return model



if __name__=='__main__':
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