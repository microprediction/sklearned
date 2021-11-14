import numpy as np
from tensorflow import keras
from sklearned.challenging.surrogateio import save_champion_metrics, save_champion_model,\
    save_champion_onnx, save_champion_weights, save_champion_info, save_champion_tensorflow
from pprint import pprint
from sklearned.challenging.dimensional import squeeze_out_middle


def assess_and_maybe_save(model, info, d, champion_metrics, skater_name, k, n_input, n_lags=None):
    if n_lags is not None:
       x_test = d['x_test'][:,:,-n_lags:]
       x_val = d['x_val'][:, :, -n_lags:]
       x_train = d['x_train'][:,:,-n_lags:]
    else:
       x_test = d['x_test']
       x_val = d['x_val']
       x_train = d['x_train']

    y_test_hat = squeeze_out_middle(model(x_test))
    test_error = float(keras.metrics.mean_squared_error(y_test_hat[:, 0], d['y_test'][:, 0]))
    y_val_hat = squeeze_out_middle(model(x_val))
    val_error = float(keras.metrics.mean_squared_error(y_val_hat[:, 0], d['y_val'][:, 0]))
    y_train_hat = squeeze_out_middle(model(x_train))
    train_error = float(keras.metrics.mean_squared_error(y_train_hat[:, 0], d['y_train'][:, 0]))

    # Innovations relative to last value
    dy_surrogate = list(y_test_hat[:, 0] - x_test[:, 0, -1])
    dy_model = list(d['y_test'][:, 0] - x_test[:, 0, -1])
    rho = np.corrcoef(x=dy_surrogate, y=dy_model)[0][1]

    challenger_metrics = {"train_error": train_error / d['y_train_typical'],
                          "val_error": val_error / d['y_val_typical'],
                          "test_error": test_error / d['y_test_typical'],
                          "rho": rho}

    test_error_ratio = challenger_metrics['test_error'] / champion_metrics['test_error']
    pprint(challenger_metrics)
    pprint('Test error ratio to champion is ' + str(test_error_ratio) )
    if n_lags is not None:
        print(' ... using '+str(n_lags)+' lags. ')
    if test_error_ratio < 0.95:
        print('You won the challenge ... saving new champion metrics, model, weights, onnx and search params')
        save_champion_metrics(metrics=challenger_metrics, skater_name=skater_name, k=k, n_input=n_input)
        save_champion_model(model=model, skater_name=skater_name, k=k, n_input=n_input)
        save_champion_weights(model=model, skater_name=skater_name, k=k, n_input=n_input)
        save_champion_onnx(model=model, skater_name=skater_name, k=k, n_input=n_input)
        save_champion_info(info=info, skater_name=skater_name, k=k, n_input=n_input)
        save_champion_tensorflow(model=model, skater_name=skater_name, k=k, n_input=n_input)

    return challenger_metrics, test_error_ratio