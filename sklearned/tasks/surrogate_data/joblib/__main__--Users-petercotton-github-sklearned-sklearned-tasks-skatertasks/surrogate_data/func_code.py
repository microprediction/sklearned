# first line: 7
def surrogate_data(fname, k=1, n_real=50, n_samples=50, n_warm = 200, n_input=20, verbose=False,
                               include_str='electricity',exclude_str='~'):
    from timemachines.skaters.localskaters import local_skater_from_name
    from timemachines.skatertools.data.live import random_surrogate_data
    f = local_skater_from_name(fname)
    assert n_samples+n_warm<=250
    x_train, y_train, y_train_true = random_surrogate_data(f=f,k=k, n_real=n_real, n_samples=n_samples, n_warm = n_warm,
                                                     n_input=n_input, verbose=False,
                                                     include_str='electricity',exclude_str='~')

    x_val, y_val, y_val_true = random_surrogate_data(f=f, k=k, n_real=n_real, n_samples=n_samples, n_warm=2*n_warm+n_samples,
                                                     n_input=n_input, verbose=False,
                                                     include_str='electricity', exclude_str='~')
    x_test, y_test, y_test_true = random_surrogate_data(f=f, k=k, n_real=n_real, n_samples=n_samples, n_warm=3*n_warm+2*n_samples,
                                                     n_input=n_input, verbose=False,
                                                     include_str='electricity', exclude_str='~')
    return dict(n_input=n_input, n_output=k, n_train=n_real*n_samples,
                x_train=np.transpose(x_train, axes=[0,2,1]), y_train=y_train,
                x_val=np.transpose(x_val, axes=[0,2,1]), y_val=y_val, y_val_true=y_val_true,
                x_test=np.transpose(x_test, axes=[0,2,1]), y_test=y_test, y_test_true=y_test_true)
