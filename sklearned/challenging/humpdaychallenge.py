from sklearned.challenging.surrogatechallenge import challenge
from pprint import pprint
from sklearned.embeddings.allembeddings import embedding_from_name


def humpday_challenge(global_optimizer, embedding_name:str, skater_name: str, k: int, n_input: int, n_trials: int):
    """
        Search space of models, search and augmentation strategies

    :param global_optimizer:        Global optimizer On hypercube in style of humpday
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
        model, user_search_params, info = embedding(us, n_input)
        info['global_optimizer'] = global_optimizer.__name__
        info['embedding_name']=embedding_name

        print('  ')
        print(model.summary())
        search_params = {'epochs':2000,'patience':50,'jiggle_fraction':0.1,'symmetries':None}
        search_params.update(user_search_params)
        pprint(search_params)
        model, metrics, test_error_ratio = challenge(model=model, skater_name=skater_name, k=k, n_real=60, n_samples=150, n_warm=100, n_input=n_input,
                                                     with_metrics=True, verbose=1, info=info, **search_params)
        return metrics['test_error']

    # Run outer optim and interpret results
    best_test, best_u, feval_count = global_optimizer(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=True)
    best_model, best_search_params = embedding(us=best_u, n_inputs=n_input)
    return best_test, best_model, best_search_params




