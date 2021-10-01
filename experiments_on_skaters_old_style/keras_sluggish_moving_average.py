from tensorflow import keras
from sklearned.tasks.skatertasks import cached_surrogate_data
from pprint import pprint
import numpy as np

# loss displayed ~ 0.0022
SLUGGISH_MOVING_AVERAGE_BEST =  {'test_error': 0.02999787349626533,
                                 'train_error': 0.022068321680294486,
                                 'val_error': 0.025155224595028527}



def build_sluggish_moving_average_champion_model(n_inputs):
  model = keras.Sequential()
  kernel_initializer_0 = keras.initializers.RandomUniform(minval=0.01, maxval=0.02, seed=None)
  bias_initializer_0 = keras.initializers.RandomUniform(minval=0.01, maxval=0.21, seed=None)
  model.add(keras.layers.Dense(80, activation="linear",input_shape=(1, n_inputs),
            kernel_initializer=kernel_initializer_0,
            bias_initializer=bias_initializer_0))
  model.add(keras.layers.Dense(16, activation='softsign'))
  model.add(keras.layers.Dense(2, activation="tanh"))  # selu
  model.add(keras.layers.Dense(1, activation="linear"))
  optimizer = keras.optimizers.Adagrad(learning_rate=0.005)
  model.compile(loss='mse',optimizer=optimizer)
  return model


def build_sluggish_moving_average_challenger_model_overfitting(n_inputs):
  model = keras.Sequential()
  kernel_initializer_0 = keras.initializers.RandomUniform(minval=0.1, maxval=0.11, seed=None)
  bias_initializer_0 = keras.initializers.RandomUniform(minval=-0.01, maxval=0, seed=None)
  model.add(keras.layers.Dense(16, activation="linear",input_shape=(1, n_inputs),
            kernel_initializer=kernel_initializer_0,
            bias_initializer=bias_initializer_0))
  model.add(keras.layers.Dense(8, activation='softsign'))
  model.add(keras.layers.Dense(2, activation="exponential"))
  model.add(keras.layers.Dense(1, activation="linear"))
  optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
  model.compile(loss='mse',optimizer=optimizer)
  return model



def ktrain(d:dict, epochs=200):
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=25)
    model = build_sluggish_moving_average_challenger_model_overfitting(n_inputs=d['n_input'])
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
   d = cached_surrogate_data(fname='sluggish_moving_average', k=1, n_real=50, n_samples=150, n_warm = 100, n_input=80)
   summary = ktrain(d=d, epochs=1000)
   ratio = summary['test_error'] / SLUGGISH_MOVING_AVERAGE_BEST['test_error']
   summary['test_error_ratio'] = ratio
   return summary



if __name__=='__main__':
   pprint(compare_to_previous())