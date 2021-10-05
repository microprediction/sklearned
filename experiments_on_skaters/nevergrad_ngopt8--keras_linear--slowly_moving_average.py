from sklearned.challenging.humpdaychallenge import humpday_challenge
from humpday.optimizers.alloptimizers import optimizer_from_name
from sklearned.embeddings.allembeddings import embedding_from_name
import os
from pprint import pprint
if __name__=='__main__':
    this_file = __file__.split(os.path.sep)[-1]
    optimizer_name = this_file.split('--')[0]+'_cube'
    embedding_name = this_file.split('--')[1]
    optimizer = optimizer_from_name(optimizer_name)
    embedding = embedding_from_name(embedding_name)
    print(optimizer.__name__)
    skater_name = this_file.split('--')[2].replace('.py', '')
    k = 1
    n_input = 80
    n_trials = 1500
    pprint(humpday_challenge(optimizer=optimizer, embedding_name=embedding, skater_name=skater_name, k=1, n_input=n_input, n_trials =n_trials))
