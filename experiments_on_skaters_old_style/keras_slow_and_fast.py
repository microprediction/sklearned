from tensorflow import keras
from sklearned.tasks.skatertasks import cached_surrogate_data
from pprint import pprint
import numpy as np

# loss displayed ~ 0.0022
SLOW_AND_FAST_BEST =  {'test_error': 0.006492504701224025,
                         'train_error': 0.011541599365815437,
                         'val_error': 0.028317418094659496}



def build_champion_model(n_inputs):
  model = keras.Sequential()
  kernel_initializer_0 = keras.initializers.RandomUniform(minval=0.01, maxval=0.011, seed=None)
  bias_initializer_0 = keras.initializers.RandomUniform(minval=0.01, maxval=0.011, seed=None)
  model.add(keras.layers.Dense(64, activation="elu",input_shape=(1, n_inputs),
            kernel_initializer=kernel_initializer_0,
            bias_initializer=bias_initializer_0))
  model.add(keras.layers.Dense(64, activation='elu'))
  model.add(keras.layers.Dense(4, activation="linear"))  # selu
  model.add(keras.layers.Dense(1, activation="linear"))
  optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
  model.compile(loss='mse',optimizer=optimizer)
  return model


def build_challenger_model(n_inputs):
  model = keras.Sequential()
  kernel_initializer_0 = keras.initializers.RandomUniform(minval=0.01, maxval=0.011, seed=None)
  bias_initializer_0 = keras.initializers.RandomUniform(minval=0.01, maxval=0.011, seed=None)
  model.add(keras.layers.Dense(64, activation="elu",input_shape=(1, n_inputs),
            kernel_initializer=kernel_initializer_0,
            bias_initializer=bias_initializer_0))
  model.add(keras.layers.Dense(64, activation='elu'))
  model.add(keras.layers.Dense(4, activation="linear"))  # selu
  model.add(keras.layers.Dense(1, activation="linear"))
  optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
  model.compile(loss='mse',optimizer=optimizer)
  return model





def ktrain(d:dict, epochs=200):
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=25)
    model = build_challenger_model(n_inputs=d['n_input'])
    model.fit(x=d['x_train'], y=d['y_train'], epochs=epochs, verbose=1, callbacks=[callback])
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
   d = cached_surrogate_data(fname='thinking_slow_and_fast', k=1, n_real=50, n_samples=150, n_warm = 100, n_input=80)
   summary = ktrain(d=d, epochs=5000)
   ratio = summary['test_error'] / SLOW_AND_FAST_BEST['test_error']
   summary['test_error_ratio'] = ratio
   return summary



if __name__=='__main__':
   pprint(compare_to_previous())