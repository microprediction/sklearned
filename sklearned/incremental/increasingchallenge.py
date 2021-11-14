from tensorflow import keras
from sklearned.challenging.surrogatedata import cached_skater_surrogate_data
from sklearned.augment.affine import affine, jiggle
from sklearned.challenging.surrogateio import read_champion_metrics, save_champion_metrics, save_champion_model,\
    save_champion_onnx, save_champion_weights, save_champion_info, save_champion_tensorflow
from pprint import pprint
import os
from sklearned.wherami import CHAMPION_METRICS_PATH, CHAMPION_MODELS_PATH, CHAMPION_WEIGHTS_PATH, CHAMPION_ONNX_PATH, CHAMPION_INFO_PATH
from sklearned.augment.cleaning import remove_surrogate_outliers

# Utilities belonging elsewhere

DREADFUL = 1000000
POOR_METRICS = {'test_error': DREADFUL,
                'train_error': DREADFUL,
                'val_error': DREADFUL}


def increasing_challenge(model, skater_name: str, info: dict, n_lags, epochs=2, jiggle_fraction=0.1, symmetries=None,
                  k=1, n_real=60, n_samples=150, n_warm=100, n_input=80, patience=50, with_metrics=False, verbose=0):

    search_params = {'epochs' :epochs,
                     'patience' :patience,
                     'jiggle_fraction' :jiggle_fraction}
    info.update(search_params)

    os.makedirs(CHAMPION_METRICS_PATH, exist_ok=True)
    os.makedirs(CHAMPION_MODELS_PATH, exist_ok=True)
    os.makedirs(CHAMPION_WEIGHTS_PATH, exist_ok=True)
    os.makedirs(CHAMPION_ONNX_PATH, exist_ok=True)
    os.makedirs(CHAMPION_INFO_PATH, exist_ok=True)

    print('Champion data')
    champion_metrics = read_champion_metrics(skater_name=skater_name, k=k, n_input=n_input)
    pprint(champion_metrics)

    print('Surrogate data ')
    d = cached_skater_surrogate_data(skater_name=skater_name, k=k, n_samples=150, n_warm=290, n_input=n_input)
    d = remove_surrogate_outliers(d)
    if symmetries is None:
        symmetries = [0.995, 1.0, 1.001]
    aug_X, aug_y = affine(X=d['x_train'], y=d['y_train'], s=symmetries)
    jiggle_X = jiggle(aug_X, jiggle_fraction=jiggle_fraction)

    resized_jiggle_X = jiggle_X[:, :, -n_lags:]

    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
    model.fit(x=resized_jiggle_X, y=aug_y, epochs=epochs, verbose=verbose, callbacks=[callback], use_multiprocessing=False, workers=1)

    from sklearned.challenging.savemaybe import assess_and_maybe_save
    challenger_metrics, test_error_ratio = assess_and_maybe_save(model=model, info=info, d=d, champion_metrics=champion_metrics,
                                                                 skater_name=skater_name, k=k, n_input=n_input, n_lags=n_lags)

    if with_metrics:
        return model, challenger_metrics, test_error_ratio
    else:
        return model