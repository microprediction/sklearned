import numpy as np
from joblib import Memory
from sklearned.wherami import SURROGATE_DATA_CACHE, LIVE_DATA_CACHE
from sklearn.metrics import mean_squared_error
from microprediction.reader import MicroReader
from typing import List


live_data_memory = Memory(LIVE_DATA_CACHE)
surrogate_data_memory = Memory(SURROGATE_DATA_CACHE)


def get_multiple_streams(sub_sample=2, include_strs=None)->List[List[float]]:
    """  """
    if include_strs is None:
        include_strs = ['hospital','electricity','airport','volume','emoji','three_body','helicopter','noaa']
    mr = MicroReader()
    streams = mr.get_stream_names()
    acceptable = [s for s in streams if any( [incl in s for incl in include_strs]) and not '~' in s]
    ys = list()
    for nm in acceptable:
        try:
            lagged_values, lagged_times = mr.get_lagged_values_and_times(name=nm, count=2000)
            y, t = list(reversed(lagged_values)), list(reversed(lagged_times))
            if 'hospital' in nm:
                y_sub = y[::sub_sample]
            else:
                y_sub = y
            y_scale = np.mean([ abs(yi) for yi in y_sub[:100]])+1
            if len(y_sub)>=750:
                y_scaled = [ yi/y_scale for yi in y_sub]
                ys.append(y_scaled)
                print(nm)
            else:
                print(nm+' too short')
        except:
            print(nm+' exception')
    print(len(ys))
    return ys

cached_get_stream_data = live_data_memory.cache(get_multiple_streams)


def subsampled_surrogate_data(f, k, n_warm, n_input, n_samples, verbose=False,
                              include_strs=None, exclude_str='~', sub_sample=2):
    """ Create dataset of skater input (abbreviated to last n_input lags) and output
    :param f:   skater
    :param k:   step ahead
    :param n_real:   number of distinct real time series to use
    :param n_input:  number of lags to use in training
    :param n_warm:   number of data points to use to warm up skater before harvesting predictions
    :return: x_train, y_train, y_true
    """
    real_data = cached_get_stream_data(sub_sample=sub_sample, include_strs=include_strs)
    n_real = len(real_data)

    n_train = n_real * n_samples
    x_train = np.zeros(shape=(n_train, n_input, 1))
    y_train = np.zeros(shape=(n_train, 1))
    x_stds = [1.0]
    y_true = list()

    if verbose:
        print(np.shape(x_train))
    for ndx_real in range(n_real):
        # Get real data
        y_real = real_data[ndx_real]
        if verbose:
            print(y_real[:5])
            print((ndx_real, n_real))
            print(np.mean(x_stds))
        # Warm the model up
        y_real_warm = y_real[:n_warm]
        s = {}
        for y in y_real_warm:
            x, x_std, s = f(y, k=k, s=s)
        # Now create examples of k-step ahead forecasts
        y_real_harvest = y_real[n_warm:n_warm + n_samples]
        for j2, y in enumerate(y_real_harvest):
            x, x_std, s = f(y, k=k, s=s)
            xk = x[k - 1]
            j = ndx_real * n_samples + j2
            yj_true = y_real[n_warm + j2 + k]
            y_true.append(yj_true)
            y_train[j, 0] = xk
            x_stds.append(x_std[-k])
            reverse_input_data = y_real[n_warm + j2 - n_input + 1:n_warm + j2 + 1]
            input_data = reverse_input_data
            # assert input_data[0]==y
            for l in range(n_input):
                x_train[j, l, 0] = input_data[l]

    return x_train, y_train, y_true


def skater_surrogate_data(skater_name:str, k, n_samples, n_warm, n_input:int=80,
                          verbose=False, include_strs=None, exclude_str='~'):
    from timemachines.skaters.localskaters import local_skater_from_name
    f = local_skater_from_name(skater_name)
    if f is None:
        raise ValueError('Cannot instantiate '+skater_name)
    assert 3*n_samples+n_warm<=750
    x_train, y_train, y_train_true = subsampled_surrogate_data(f=f,k=k,  n_samples=n_samples, n_warm = n_warm,
                                                     n_input=n_input, verbose=False,
                                                     include_strs=include_strs,exclude_str=exclude_str)
    print('got training data')
    x_val, y_val, y_val_true = subsampled_surrogate_data(f=f, k=k,  n_samples=n_samples, n_warm=n_warm+n_samples,
                                                     n_input=n_input, verbose=False,
                                                     include_strs=include_strs, exclude_str=exclude_str)
    print('got val data')
    x_test, y_test, y_test_true = subsampled_surrogate_data(f=f, k=k, n_samples=n_samples, n_warm=n_warm+2*n_samples,
                                                     n_input=n_input, verbose=False,
                                                     include_strs=include_strs, exclude_str=exclude_str)
    print('got test data')
    # Include the typical prediction error, which is a useful scale
    n_real = np.shape(x_train)[0]
    y_train_typical = mean_squared_error(y_train_true,y_train, squared=False)
    y_val_typical = mean_squared_error(y_val_true, y_val, squared=False)
    y_test_typical = mean_squared_error(y_test_true, y_test, squared=False)
    return dict(n_input=n_input, n_output=k, n_train=n_real*n_samples,
                x_train=np.transpose(x_train, axes=[0,2,1]), y_train=y_train,
                x_val=np.transpose(x_val, axes=[0,2,1]), y_val=y_val, y_val_true=y_val_true,
                x_test=np.transpose(x_test, axes=[0,2,1]), y_test=y_test, y_test_true=y_test_true,
                y_train_typical=y_train_typical, y_test_typical=y_test_typical, y_val_typical=y_val_typical)


cached_skater_surrogate_data = surrogate_data_memory.cache(skater_surrogate_data)



if __name__ == '__main__':
    try:
        import microprediction
        import timemachines
    except:
        raise('pip install microprediction')
    from timemachines.skaters.elo.eloensembles import elo_faster_residual_balanced_ensemble
    d = cached_skater_surrogate_data(skater_name='precision_ema_ensemble', k=1,  n_warm=120, n_samples=50)

