from sklearned.challenging.surrogatechallenge import challenge
from tensorflow import keras
from sklearned.embeddings.transforms import to_log_space_1d
from pprint import pprint
from sklearned.embeddings.allembeddings import embedding_from_name


def humpday_challenge(optimizer, embedding_name:str, skater_name: str, k: int, n_input: int, n_trials: int):
    """
    :param optimizer:        On hypercube in style of humpday
    :param embedding_name:   Must end in _27 or _9 or whatever to indicate dimension
    :param skater_name:      Anything from timemachines.skaters
    :param k:                Number of steps ahead
    :param n_input:          Number of lags to use as input
    :param n_trials:         Number of trials (outer optimization)
    :return:  best_val, best_x, feval_count
    """

    n_dim = int( embedding_name.split('_')[-1])
    embedding = embedding_from_name(embedding_name)

    def objective(us: [float]) -> float:
        model, user_search_params = embedding(us, n_input)
        n_dim = int(embedding_name.split('_')[-1])

        print('  ')
        print(model.summary())
        search_params = {'epochs':2000,'patience':50,'jiggle_fraction':0.1,'symmetries':None}
        search_params.update(user_search_params)
        pprint(search_params)
        model, metrics, test_error_ratio = challenge(model=model, skater_name=skater_name, k=k, n_real=60, n_samples=150, n_warm=100, n_input=n_input,
                                                     with_metrics=True, verbose=1, **search_params)
        return metrics['test_error']

    # Run outer optim and interpret results
    best_test, best_u, feval_count = optimizer(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=True)
    best_model, best_search_params = embedding(us=best_u, n_inputs=n_input)
    return best_test, best_model, best_search_params




def humpday_challenge_fixed_example(optimizer, skater_name:str, k:int, n_input:int, n_trials:int):
    """
    :param optimizer:        On hypercube in style of humpday
    :param skater_name:      Anything from timemachines.skaters
    :param k:                Number of steps ahead
    :param n_input:          Number of lags to use as input
    :param n_trials:         Number of trials (outer optimization)
    :return:  best_val, best_x, feval_count
    """

    def objective(us:[float])->float:
        min_kernel = us[0]
        max_kernel = us[0] + us[1] + 0.001
        bias_size = us[2]
        layers_0 = int( to_log_space_1d( us[3], low=8, high=128 ) )
        layers_1 = int( to_log_space_1d( us[4], low=2, high=32) )
        layers_2 = int( to_log_space_1d( us[5], low=2, high=16) )
        learning_rate = to_log_space_1d( us[6], low=0.00001, high=0.05)
        epochs = int( to_log_space_1d( us[7], low=50, high=1000 ))  # 5000
        patience = int( to_log_space_1d( us[8], low=5, high=50))    # 5
        jiggle_fraction = us[9]

        next_try = {'learning_rate': learning_rate,
                'epochs': epochs,
                'patience': patience,
                'jiggle_fraction': jiggle_fraction,
                'min_kernal': min_kernel,
                'max_kernel': max_kernel,
                'bias_size': bias_size
                }
        print('  ')
        pprint(next_try)

        def build_linear_model(n_inputs):
            model = keras.Sequential()
            kernel_initializer_0 = keras.initializers.RandomUniform(minval=min_kernel, maxval=max_kernel, seed=None)
            bias_initializer_0 = keras.initializers.RandomUniform(minval=-bias_size, maxval=bias_size, seed=None)
            model.add(keras.layers.Dense(layers_0, activation="relu", input_shape=(1, n_inputs),
                                         kernel_initializer=kernel_initializer_0,
                                         bias_initializer=bias_initializer_0))
            model.add(keras.layers.Dense(layers_1, activation='relu'))
            model.add(keras.layers.Dense(layers_2, activation="tanh"))  # selu
            model.add(keras.layers.Dense(1, activation="linear"))
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
            model.compile(loss='mse', optimizer=optimizer)
            return model

        model = build_linear_model(n_inputs=n_input)

        model, metrics, test_error_ratio = challenge(model=model, skater_name=skater_name, epochs=epochs, jiggle_fraction=jiggle_fraction, symmetries=None,
              k=k, n_real=60, n_samples=150, n_warm=100, n_input=n_input, patience=patience, with_metrics=True, verbose=1)

        return metrics['test_error']

    # Run outer optim and interpret results
    n_dim = 10
    best_val, best_u, feval_count = optimizer(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=True)
    interpretation = {'min_kernel':best_u[0],
                      'max_kernel':best_u[0] + best_u[1] + 0.001,
                       'bias_size':best_u[2],
                       'layers_0':int(to_log_space_1d(best_u[3], low=8, high=512)),
                       'layers_1':int(to_log_space_1d(best_u[4], low=2, high=256)),
                       'layers_2':int(to_log_space_1d(best_u[5], low=2, high=36)),
                       'learning_rate':to_log_space_1d(best_u[6], low=0.000001, high=0.1),
                       'epochs': int(to_log_space_1d(best_u[7], low=100, high=5000)),
                       'patience':int(to_log_space_1d(best_u[8], low=5, high=500)),
                   'jiggle_fraction':best_u[9],
                      'test_error':best_val,
                       'fevaL_count':feval_count}
    return interpretation


if __name__=='__main__':
    from humpday.optimizers.nevergradcube import nevergrad_ngopt8_cube
    skater_name = 'tsa_precision_combined_ensemble'
    k = 1
    n_input = 80
    n_trials = 5
    interpretation = humpday_challenge_fixed_example(optimizer=nevergrad_ngopt8_cube, skater_name=skater_name, k=1, n_input=n_input, n_trials =n_trials)
    pprint(interpretation)