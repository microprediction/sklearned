from sklearned.challenging.surrogatechallenge import challenge
from tensorflow import keras
from sklearned.challenging.transforms import to_log_space_1d


def humpday_challenge(optimizer, skater_name:str, k:int, n_input:int, n_trials:int):

    def objective(us:[float])->float:
        min_kernel = us[0]
        max_kernel = us[1]
        bias_size = us[2]
        layers_0 = int( to_log_space_1d( us[3], low=8, high=512 ) )
        layers_1 = int( to_log_space_1d( us[4], low=2, high=256) )
        layers_2 = int( to_log_space_1d( us[5], low=2, high=36) )
        learning_rate = to_log_space_1d( us[6], low=0.000001, high=0.1)
        epochs = int( to_log_space_1d( us[7], low=100, high=5000 ))
        patience = int( to_log_space_1d( us[8], low=5, high=500))
        jiggle_fraction = us[9]


        def build_linear_model(n_inputs):
            model = keras.Sequential()
            kernel_initializer_0 = keras.initializers.RandomUniform(minval=min_kernel, maxval=max_kernel, seed=None)
            bias_initializer_0 = keras.initializers.RandomUniform(minval=-bias_size, maxval=bias_size, seed=None)
            model.add(keras.layers.Dense(layers_0, activation="linear", input_shape=(1, n_inputs),
                                         kernel_initializer=kernel_initializer_0,
                                         bias_initializer=bias_initializer_0))
            model.add(keras.layers.Dense(layers_1, activation='linear'))
            model.add(keras.layers.Dense(layers_2, activation="linear"))  # selu
            model.add(keras.layers.Dense(1, activation="linear"))
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
            model.compile(loss='mse', optimizer=optimizer)
            return model

        model = build_linear_model(n_inputs=n_input)

        model, metrics = challenge(model=model, skater_name=skater_name, epochs=epochs, jiggle_fraction=jiggle_fraction, symmetries=None,
              k=k, n_real=60, n_samples=150, n_warm=100, n_input=n_input, patience=patience, with_metrics=True)
        return metrics['test_error']

    n_dim = 10
    return optimizer(objective=objective, n_trials=n_trials, n_dim=n_dim)

if __name__=='__main__':
    from humpday.optimizers.nevergradcube import nevergrad_ngopt8_cube
    skater_name = 'tsa_precision_combined_ensemble'
    k = 1
    n_input = 80
    n_trials = 5
    humpday_challenge(optimizer=nevergrad_ngopt8_cube, skater_name=skater_name, k=1, n_input=n_input, n_trials =n_trials )