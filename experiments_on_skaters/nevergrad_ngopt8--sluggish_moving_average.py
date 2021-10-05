from sklearned.challenging.humpdaychallenge import humpday_challenge_fixed_example
from humpday.optimizers.alloptimizers import optimizer_from_name
import os
from pprint import pprint
if __name__=='__main__':
    this_file = __file__.split(os.path.sep)[-1]
    optimizer_name = this_file.split('--')[0]+'_cube'
    optimizer = optimizer_from_name(optimizer_name)
    print(optimizer.__name__)
    skater_name = this_file.split('--')[1].replace('.py', '')
    k = 1
    n_input = 80
    n_trials = 1500
    pprint(humpday_challenge_fixed_example(optimizer=optimizer, skater_name=skater_name, k=1, n_input=n_input, n_trials =n_trials))
