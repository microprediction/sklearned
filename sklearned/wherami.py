import os
import pathlib

ROOT_PATH = str(pathlib.Path(os.path.dirname(os.path.realpath(__file__))).parent)
CHAMPION_METRICS_PATH = ROOT_PATH + os.path.sep + 'champion_model_metrics'
SURROGATE_DATA_CACHE = ROOT_PATH + os.path.sep + 'surrogate_data_cache'

if __name__=='__main__':
    print('champion path is ' + CHAMPION_METRICS_PATH)


