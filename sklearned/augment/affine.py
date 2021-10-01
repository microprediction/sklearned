import numpy as np
import math
import random


def jiggle(X,jiggle_fraction=0.05):
    d = np.shape(X)[2]
    n  = np.shape(X)[0]
    assert np.shape(X)[1]==1   # TODO: generalize
    d_half = int(math.ceil(d/2))
    n_jiggle = max(int(d_half*jiggle_fraction*n),5)
    for _ in range(n_jiggle):
        rand_n = random.choice(range(n))
        rand_j = random.choice(range(d_half))
        jiggle_amount = d_half/(rand_j+d_half)*np.random.exponential()
        X[rand_n,0,rand_j] = jiggle_amount*X[rand_n,0,rand_j]
    return X



def affine(X,y, s:[float]):
    """
    :param X:      n x 1 x b
    :param y:      n x 1
    :param s:      scale factors
    :return:
    """
    n = len(y)
    X_copies = [ si*X for si in s ]
    aug_X = np.concatenate(X_copies)
    y_copies = [ si*y for si in s]
    aug_y = np.concatenate(y_copies)
    return aug_X, aug_y




if __name__=='__main__':
    X = np.random.randn(5,1,3)
    y = X[:,:,-1]
    print(np.shape(y))
    X_, y_ = affine(X,y,s=[1,0.01])
    print(X_)
    print(np.shape(X_))
    print('y...')
    print(y_)
    print(np.shape(y_))

