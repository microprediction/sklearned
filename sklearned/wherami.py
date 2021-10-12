import os
import pathlib

ROOT_PATH = str(pathlib.Path(os.path.dirname(os.path.realpath(__file__))).parent)
CHAMPION_METRICS_PATH = ROOT_PATH + os.path.sep + 'champion_metrics'
CHAMPION_MODELS_PATH = ROOT_PATH + os.path.sep + 'champion_models'
CHAMPION_WEIGHTS_PATH = ROOT_PATH + os.path.sep + 'champion_weights'
CHAMPION_ONNX_PATH = ROOT_PATH + os.path.sep + 'champion_onnx'
SURROGATE_DATA_CACHE = ROOT_PATH + os.path.sep + 'surrogate_data_cache'
CHAMPION_INFO_PATH = ROOT_PATH + os.path.sep + 'champion_info'

LIVE_DATA_CACHE = ROOT_PATH + os.path.sep + 'live_data_cache'



if __name__=='__main__':
    print('champion path is ' + CHAMPION_METRICS_PATH)


