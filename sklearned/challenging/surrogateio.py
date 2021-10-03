import os
from sklearned.wherami import CHAMPION_METRICS_PATH, CHAMPION_MODELS_PATH, CHAMPION_WEIGHTS_PATH, CHAMPION_ONNX_PATH
from tf2onnx.keras2onnx_api import convert_keras
import json
# Utilities belonging elsewhere

DREADFUL = 1000000
POOR_METRICS = {'test_error': DREADFUL,
                'train_error': DREADFUL,
                'val_error': DREADFUL}


def champion_stem(skater_name, k: int, n_input: int):
    return skater_name + '_' + str(k) + '_' + str(n_input)


def champion_metrics_file(skater_name, k: int, n_input: int):
    return CHAMPION_METRICS_PATH + os.path.sep + champion_stem(skater_name=skater_name, k=k, n_input=n_input) + '.json'


def champion_weights_file(skater_name, k: int, n_input: int):
    return CHAMPION_WEIGHTS_PATH + os.path.sep + champion_stem(skater_name=skater_name, k=k, n_input=n_input) + '.h5'


def champion_model_file(skater_name, k: int, n_input: int):
    return CHAMPION_MODELS_PATH + os.path.sep + champion_stem(skater_name=skater_name, k=k, n_input=n_input) + '.json'


def champion_onnx_file(skater_name, k: int, n_input: int):
    return CHAMPION_ONNX_PATH + os.path.sep + champion_stem(skater_name=skater_name, k=k, n_input=n_input) + '.onnx'


def read_model_champion_metrics(**kwargs):
    try:
        with open(champion_metrics_file(**kwargs), 'r') as fp:
            metrics = json.load(fp)
    except FileNotFoundError:
        metrics = POOR_METRICS
    return metrics


def save_champion_metrics(metrics: dict, **kwargs):
    with open(champion_metrics_file(**kwargs), 'w') as fp:
        json.dump(metrics, fp)


def save_champion_model(model, **kwargs):
    model_as_json = model.to_json()
    mf = champion_model_file(**kwargs)
    with open(mf, 'w') as fp:
        fp.write(model_as_json)


def save_champion_weights(model, **kwargs):
    wf = champion_weights_file(**kwargs)
    return model.save_weights(wf)


def save_champion_onnx(model, **kwargs):
    # in accordance with https://github.com/onnx/onnx/pull/541 I think
    # But https://github.com/onnx/tensorflow-onnx/issues/1734
    of = champion_onnx_file(**kwargs)
    onnx_model = convert_keras(model=model, name='example')
    onnx_model_as_byte_string = onnx_model.SerializeToString()
    with open(of, 'wb') as fp:
        fp.write(onnx_model_as_byte_string)


def load_champion_model(**kwargs):
    from tensorflow.keras.models import model_from_json
    mf = champion_model_file(**kwargs)
    with open(mf, 'rt') as fp:
        model_as_json = fp.read()
    model = model_from_json(model_as_json)
    return model


def load_champion_model_with_weights(**kwargs):
    raise ValueError('https://github.com/keras-team/keras/issues/14265')
    model = load_champion_model(**kwargs)
    wf = champion_weights_file(**kwargs)
    model.compile()
    model.load_weights(wf)
    return model




if __name__ == '__main__':
    model = load_champion_model(skater_name='quick_balanced_ema_ensemble',k=1,n_input=80)
    print(model.summary())
    import onnxruntime as nxrun
    import numpy as np
    of = champion_onnx_file(skater_name='quick_balanced_ema_ensemble',k=1,n_input=80)
    sess = nxrun.InferenceSession(of)
    print("The model expects input shape: ", sess.get_inputs()[0].shape)


