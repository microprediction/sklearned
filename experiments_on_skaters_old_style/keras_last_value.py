from tensorflow import keras, ones
from sklearned.tasks.lastvalue import last_value_task
from pprint import pprint
import numpy as np


def build_last_value_champion_model(n_inputs):
  model = keras.Sequential()
  model.add(keras.layers.Dense(8, activation="linear", input_shape=(1,n_inputs)))
  model.add(keras.layers.Dense(6, activation="relu"))
  model.add(keras.layers.Dense(8, activation="linear"))
  model.add(keras.layers.Dense(1, activation="linear"))
  model.compile(loss='mse')
  #y = model(ones((1, n_inputs)))
  return model


LAST_VALUE_BEST =  {'test_error': 7.042708119756874e-05,
                                 'train_error': 6.619203063910744e-05,
                                 'val_error': 7.373535828612822e-05}

def build_last_value_challenger_model(n_inputs):
    """
    """
    model = keras.Sequential()
    model.add(keras.layers.Dense(6, activation="linear", input_shape=(1,n_inputs)))
    model.add(keras.layers.Dense(8, activation="relu"))
    model.add(keras.layers.Dense(12, activation="linear"))
    model.add(keras.layers.Dense(1, activation="linear"))
    model.compile(loss='mse')
    #y = model(ones((1, n_inputs)))
    return model


def ktrain(d:dict, epochs=50):
    model = build_last_value_challenger_model(n_inputs=d['n_input'])
    model.fit(x=d['x_train'], y=d['y_train'], epochs=epochs, verbose=0)
    y_test_hat = model.predict(d['x_test'])
    test_error = float(keras.metrics.mean_squared_error(y_test_hat[:,0,0], d['y_test'][:,0]))
    y_val_hat = model.predict(d['x_val'])
    val_error = float(keras.metrics.mean_squared_error(y_val_hat[:,0,0], d['y_val'][:,0]))
    y_train_hat = model.predict(d['x_train'])
    train_error = float(keras.metrics.mean_squared_error(y_train_hat[:,0,0], d['y_train'][:,0]))
    summary = {"train_error": train_error,
                "val_error": val_error,
                "test_error": test_error}
    return summary

if __name__=='__main__':
   d = last_value_task(n_input=20, n_train=1000)
   summary = ktrain(d=d, epochs=50)
   ratio = summary['test_error']/LAST_VALUE_BEST['test_error']
   summary['test_error_ratio']=ratio
   pprint(summary)