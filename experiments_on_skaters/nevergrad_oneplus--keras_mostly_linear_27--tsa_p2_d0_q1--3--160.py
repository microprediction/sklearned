from sklearned.challenging.humpdaychallenge import humpday_challenge
from humpday.optimizers.alloptimizers import optimizer_from_name
import os
from pprint import pprint
if __name__=='__main__':
    this_file = __file__.split(os.path.sep)[-1]
    optimizer_name = this_file.split('--')[0]+'_cube'
    skater_name = this_file.split('--')[2]
    embedding_name = this_file.split('--')[1]
    k = int(this_file.split('--')[3])
    n_input = int(this_file.split('--')[4].replace('.py',''))
    n_dim = this_file.split
    optimizer = optimizer_from_name(optimizer_name)
    print(optimizer.__name__)
    n_trials = 5000
    best_test, best_model, best_search_params = humpday_challenge(global_optimizer=optimizer, embedding_name=embedding_name, skater_name=skater_name, k=k, n_input=n_input, n_trials =n_trials)
    print(best_test)
    print(best_model.summary())
    pprint(best_search_params)

