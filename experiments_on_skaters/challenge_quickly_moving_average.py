from sklearned.challenging.surrogatechallenge import challenge
from tensorflow import keras
import os


def build_challenger_model(n_inputs):
    model = keras.Sequential()
    kernel_initializer_0 = keras.initializers.RandomUniform(minval=0.01, maxval=0.1, seed=None)
    bias_initializer_0 = keras.initializers.RandomUniform(minval=-0.2, maxval=0.2, seed=None)
    model.add(keras.layers.Dense(80, activation="linear", input_shape=(1, n_inputs),
                                 kernel_initializer=kernel_initializer_0,
                                 bias_initializer=bias_initializer_0))
    model.add(keras.layers.Dense(16, activation='linear'))
    model.add(keras.layers.Dense(2, activation="linear"))  # selu
    model.add(keras.layers.Dense(1, activation="linear"))
    optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(loss='mse', optimizer=optimizer)
    return model


if __name__=='__main__':
    skater_name = __file__.split(os.path.sep)[-1].replace('challenge_','').replace('.py','')
    print(skater_name)
    model = build_challenger_model(n_inputs=80)
    challenge(model=model, skater_name=skater_name, epochs=5, patience=25)