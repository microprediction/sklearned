import numpy as np
from sklearned.embeddings.transforms import choice_from_dict


MOSTLY_MSE = {'mse' :20,
               'mae' :5}


def mostly_mse(u:float):
    return choice_from_dict(u=u, weighting=MOSTLY_MSE)


if __name__=='__main__':
    print(mostly_mse(u=0.999))