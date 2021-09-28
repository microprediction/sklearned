import numpy as np
from joblib import Memory

memory = Memory("surrogate_data")


def surrogate_data(fname, k=1, n_real=50, n_samples=50, n_warm = 200, n_input=20, verbose=False,
                               include_str='electricity',exclude_str='~'):
    from timemachines.skaters.localskaters import local_skater_from_name
    from timemachines.skatertools.data.live import random_surrogate_data
    f = local_skater_from_name(fname)
    assert n_samples+n_warm<=250
    x_train, y_train, y_train_true = random_surrogate_data(f=f,k=k, n_real=n_real, n_samples=n_samples, n_warm = n_warm,
                                                     n_input=n_input, verbose=False,
                                                     include_str='electricity',exclude_str='~')
    print('got training data')
    x_val, y_val, y_val_true = random_surrogate_data(f=f, k=k, n_real=n_real, n_samples=n_samples, n_warm=2*n_warm+n_samples,
                                                     n_input=n_input, verbose=False,
                                                     include_str='electricity', exclude_str='~')
    print('got val data')
    x_test, y_test, y_test_true = random_surrogate_data(f=f, k=k, n_real=n_real, n_samples=n_samples, n_warm=3*n_warm+2*n_samples,
                                                     n_input=n_input, verbose=False,
                                                     include_str='electricity', exclude_str='~')
    print('got test data')
    # Include the typical prediction error, which is a useful scale
    from sklearn.metrics import mean_squared_error
    y_train_typical = mean_squared_error(y_train_true,y_train, squared=False)
    y_val_typical = mean_squared_error(y_val_true, y_val, squared=False)
    y_test_typical = mean_squared_error(y_test_true, y_test, squared=False)
    return dict(n_input=n_input, n_output=k, n_train=n_real*n_samples,
                x_train=np.transpose(x_train, axes=[0,2,1]), y_train=y_train,
                x_val=np.transpose(x_val, axes=[0,2,1]), y_val=y_val, y_val_true=y_val_true,
                x_test=np.transpose(x_test, axes=[0,2,1]), y_test=y_test, y_test_true=y_test_true,
                y_train_typical=y_train_typical, y_test_typical=y_test_typical, y_val_typical=y_val_typical)

cached_surrogate_data = memory.cache(surrogate_data)


if __name__ == '__main__':
    try:
        import microprediction
        import timemachines
    except:
        raise('pip install microprediction')
    from timemachines.skaters.elo.eloensembles import elo_faster_residual_balanced_ensemble
    d = cached_surrogate_data(fname='precision_ema_ensemble',n_real=50, n_warm=200, n_samples=50)

