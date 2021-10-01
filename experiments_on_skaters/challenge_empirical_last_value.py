from sklearned.skaters.surrogatemodel import challenge
from tensorflow import  keras
import os


def build_challenger_model(n_inputs):
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation="linear", input_shape=(1, n_inputs)))
    model.add(keras.layers.Dense(6, activation="linear"))
    model.add(keras.layers.Dense(8, activation="linear"))
    model.add(keras.layers.Dense(1, activation="linear"))
    optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
    model.add(keras.layers.Dropout(0.01))
    model.compile(loss='mse', optimizer=optimizer)
    # y = model(ones((1, n_inputs)))
    return model

suggestion = """
def build_last_value_champion_model(n_inputs):
  model = keras.Sequential()
  model.add(keras.layers.Dense(8, activation="linear", input_shape=(1,n_inputs)))
  model.add(keras.layers.Dense(6, activation="relu"))
  model.add(keras.layers.Dense(8, activation="linear"))
  model.add(keras.layers.Dense(1, activation="linear"))
  model.compile(loss='mse')
  #y = model(ones((1, n_inputs)))
  return model"""

champ = """
     
"""


if __name__=='__main__':
    skater_name = __file__.split(os.path.sep)[-1].replace('challenge_','').replace('.py','')
    print(skater_name)
    model = build_challenger_model(n_inputs=80)
    challenge(model=model, skater_name=skater_name, epochs=5000,  jiggle_fraction=0.05)