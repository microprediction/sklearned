from tensorflow import keras
from sklearned.tasks.skatertasks import cached_surrogate_data
from pprint import pprint
import numpy as np


TSA_P1_D0_Q0_BEST =  {'test_error': 0.02411468745504141,
 'train_error': 0.10159907059382978,
 'val_error': 0.11688733289815269}

TSA_P2_D0_Q0_BEST = {'test_error': 0.028349160046629577,
                    'train_error': 0.01061077033422497,
                    'val_error': 0.02898961924191815}


def build_tsa_p2_d0_q0_challenger_model(n_inputs):
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



def build_tsa_p2_d0_q0_champion_model(n_inputs):
    model = keras.Sequential()
    kernel_initializer_0 = keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)
    bias_initializer_0 = keras.initializers.RandomUniform(minval=-1, maxval=1, seed=None)
    model.add(keras.layers.Dense(32, activation="relu", input_shape=(1, n_inputs),
                                 kernel_initializer=kernel_initializer_0,
                                 bias_initializer=bias_initializer_0))
    model.add(keras.layers.Dense(32, activation='linear'))
    model.add(keras.layers.Dense(8, activation="linear"))
    model.add(keras.layers.Dense(1, activation="linear"))
    optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(loss='mse', optimizer=optimizer)
    return model

from sklearned.augment.affine import affine, jiggle


def ktrain(d:dict, epochs=200):
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=1000)
    model = build_tsa_p2_d0_q0_challenger_model(n_inputs=d['n_input'])
    aug_X, aug_y = affine(X=d['x_train'],y=d['y_train'],s=[0.95,0.975,0.99,1.0,1.01,1.025,1.05])
    jiggle_X = jiggle(aug_X, jiggle_fraction=0.1)
    model.fit(x=jiggle_X, y=aug_y, epochs=epochs, verbose=1, callbacks=[callback])
    #model.fit(x=d['x_train'], y=d['y_train'], epochs=epochs, verbose=1)
    y_test_hat = model.predict(d['x_test'])
    test_error = float(keras.metrics.mean_squared_error(y_test_hat[:,0,0], d['y_test'][:,0]))
    y_val_hat = model.predict(d['x_val'])
    val_error = float(keras.metrics.mean_squared_error(y_val_hat[:,0,0], d['y_val'][:,0]))
    y_train_hat = model.predict(d['x_train'])
    train_error = float(keras.metrics.mean_squared_error(y_train_hat[:,0,0], d['y_train'][:,0]))
    summary = {"train_error": train_error/d['y_train_typical'],
                "val_error": val_error/d['y_val_typical'],
                "test_error": test_error/d['y_test_typical']}
    return summary


def compare_to_previous():
   d = cached_surrogate_data(fname='tsa_p2_d0_q0', k=1, n_real=60, n_samples=150, n_warm = 100, n_input=80)
   summary = ktrain(d=d, epochs=50000)
   ratio = summary['test_error'] / TSA_P2_D0_Q0_BEST['test_error']
   summary['test_error_ratio'] = ratio
   return summary



if __name__=='__main__':
   pprint(compare_to_previous())