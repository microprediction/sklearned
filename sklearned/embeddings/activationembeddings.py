import numpy as np
from sklearned.embeddings.transforms import choice_from_dict


MOSTLY_LINEAR = {'linear' :20,
               'relu' :20,
               'gelu' :5,
               'hard_sigmoid' :3,
               'sigmoid' :3,
               'softmax' :3,
               'softplus' :3,
               'softsign' :2,
               'swish' :20,
               'tanh' :2,
               'exponential' :1}


MOSTLY_SWISH = MOSTLY_LINEAR
MOSTLY_SWISH['swish'] = 100


def mostly_linear(u:float):
    return choice_from_dict(u=u, weighting=MOSTLY_LINEAR)


def mostly_swish(u:float):
    return choice_from_dict(u=u, weighting=MOSTLY_SWISH)

if __name__=='__main__':
    print(mostly_linear(u=0.999))