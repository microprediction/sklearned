from sklearned.skaters.surrogatemodel import challenge
from tensorflow import  keras
import os


def build_challenger_model(n_inputs):
    model = keras.Sequential()
    kernel_initializer_0 = keras.initializers.RandomUniform(minval=-1, maxval=1, seed=None)
    bias_initializer_0 = keras.initializers.RandomUniform(minval=-0.2, maxval=0.21, seed=None)
    model.add(keras.layers.Dense(80, activation="relu", input_shape=(1, n_inputs),
                                 kernel_initializer=kernel_initializer_0,
                                 bias_initializer=bias_initializer_0))
    model.add(keras.layers.Dense(16, activation='linear'))
    model.add(keras.layers.Dense(2, activation="tanh"))  # selu
    model.add(keras.layers.Dense(1, activation="linear"))
    optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(loss='mse', optimizer=optimizer)
    return model


if __name__=='__main__':
    skater_name = __file__.split(os.path.sep)[-1].replace('challenge_','').replace('.py','')
    print(skater_name)
    model = build_challenger_model(n_inputs=80)
    challenge(model=model, skater_name=skater_name, epochs=500)