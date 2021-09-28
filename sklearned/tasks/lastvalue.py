import numpy as np


def last_value_task(n_input=20, n_train=1000) -> dict:
    """ Example of a simple [ float ] -> flaot data set
        The task is learning the last value in a vector
    :param n_input:      Dimension of input
    :param n_train:      Number of training samples
    :return:       d['y_test'] is (1000, 1)
                   d['x_test'] is (1000, 1, n_input)
    """
    n_output = 1
    x_train = np.random.randn(n_train, 1, n_input)
    y_train = x_train[:, :, n_input-1]
    x_val = np.random.randn(n_train, 1, n_input)
    y_val = x_val[:, :, n_input-1]
    x_test = np.random.randn(n_train, 1,n_input)
    y_test = x_test[:, :, n_input-1]
    return dict(n_input=n_input,n_output=n_output, n_train=n_train, x_train=x_train, x_test=x_test, x_val=x_val, y_train=y_train, y_test=y_test, y_val=y_val )


def last_value_task_transposed(n_input=20, n_train=1000) -> dict:
    """ Example of a simple [ float ] -> flaot data set
        The task is learning the last value in a vector
    :param n_input:      Dimension of input
    :param n_train:      Number of training samples
    :return:
    """
    n_output = 1
    x_train = np.random.randn(n_train, n_input, 1)
    y_train = x_train[:, n_input - 1,:]
    x_val = np.random.randn(n_train, n_input,1)
    y_val = x_val[:, n_input - 1,:]
    x_test = np.random.randn(n_train, n_input,1)
    y_test = x_test[:, n_input-1, :]
    return dict(n_input=n_input,n_output=n_output, n_train=n_train, x_train=x_train, x_test=x_test, x_val=x_val, y_train=y_train, y_test=y_test, y_val=y_val )


