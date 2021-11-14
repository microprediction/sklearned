from sklearned.incremental.increasingchallenge import increasing_challenge
from pprint import pprint
from sklearned.embeddings.allembeddings import embedding_from_name
import numpy as np


def pre_pad_zero(wghts, n_lags):
    """ Enlarge layer weights with zeros
    :param wghts:
    :param n_lags:
    :return:
    """
    the_existing_weights = wghts[0]
    the_existing_activations = wghts[1]
    n_lags_existing, n_units = np.shape(the_existing_weights)  # Is this right?
    the_new_weights = np.zeros(shape=(n_lags,n_units))
    the_new_weights[-n_lags_existing:,:] = the_existing_weights
    the_new_weights = [ the_new_weights, the_existing_activations ]
    return the_new_weights



def humpday_increasing_challenge(global_optimizer, embedding_name:str, skater_name: str, k: int, n_input: int, n_trials: int):
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
    if embedding is None:
        raise Exception('Could not instantiate '+embedding_name)


    def objective(us: [float]) -> float:
        prev_weights = None
        best_test_error_ratio = 10000
        for n_lags in [3,5,8,13,21,34,n_input]:
            model, user_search_params, info = embedding(us, n_lags)
            if prev_weights is not None:
                # Warm start using previous weights
                for i, prev_w in enumerate(prev_weights):
                    if i==0:
                        # Resize first layer
                        model.layers[i].set_weights(pre_pad_zero(wghts=prev_w,n_lags=n_lags))
                    else:
                        model.layers[i].set_weights(prev_w)

            info['global_optimizer'] = global_optimizer.__name__
            info['embedding_name']=embedding_name
            info['n_lags']=n_lags
            print('  ')
            print(model.summary())
            search_params = {'epochs':2,'patience':1,'jiggle_fraction':0.1,'symmetries':None}
            search_params.update(user_search_params)
            pprint(search_params)
            # We use n_input in the surrogate data cache but n_lags are pulled out
            model, metrics, test_error_ratio = increasing_challenge(model=model, skater_name=skater_name, k=k, n_real=60, n_samples=150, n_warm=100, n_input=n_input, n_lags=n_lags,
                                                         with_metrics=True, verbose=1, info=info, **search_params)
            metrics['n_lags']=n_lags
            pprint(metrics)
            prev_weights = [ lyr.get_weights() for lyr in model.layers ]
            best_test_error_ratio = min(best_test_error_ratio, test_error_ratio)

        return best_test_error_ratio

    # Run outer optim and interpret results
    best_test, best_u, feval_count = global_optimizer(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=True)
    best_model, best_search_params = embedding(us=best_u, n_inputs=n_input)
    return best_test, best_model, best_search_params


if __name__=='__main__':
    wghts =   [ np.ndarray(shape=(2,2)),
                np.array([-0.9,1.0]) ],

    pre_pad_zero(wghts=wghts[0], n_lags=15)

