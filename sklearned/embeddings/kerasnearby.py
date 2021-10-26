
from sklearned.challenging.surrogateio import read_champion_info, load_champion_model
from sklearned.embeddings.optimizerembeddings import keras_optimizer_from_name
from sklearned.embeddings.transforms import to_log_space_1d


def keras_nearby_quick_2(us:[float], n_input:int):
    """ Varies learning_rate and jiggle_fraction """
    return keras_nearby(us=us, n_input=n_input, skater_name='quick_aggressive_ema_ensemble', k=1,  fixed_search_params=['epochs', 'patience'])


def keras_nearby_quick_3(us:[float], n_input:int):
    """ Varies learning_rate and jiggle_fraction and patience """
    return keras_nearby(us=us, n_input=n_input, skater_name='quick_aggressive_ema_ensemble', k=1, fixed_search_params=['epochs'])



def keras_nearby(us:[float], n_input:int, skater_name:str, k:int, fixed_search_params:[str]=None):
    """
       fixed_search_params : list with 'epochs','patience','jiggle_fraction', 'learning_rate' optional
    """
    model = load_champion_model(skater_name=skater_name,k=k,n_input=n_input)
    info = read_champion_info(skater_name=skater_name,k=k,n_input=n_input)

    if 'learning_rate' in fixed_search_params:
        learning_rate_scaling = 1
    else:
        learning_rate_scaling = to_log_space_1d(u=us[0], low=0.1, high=10.0)
    learning_rate = info['learning_rate'] * learning_rate_scaling
    keras_optimizer = keras_optimizer_from_name(info['keras_optimizer'], learning_rate=learning_rate)

    model.compile(loss='mse', optimizer=keras_optimizer)

    DEFAULT_LOW  = {'epochs':500,'patience':3,'jiggle_fraction':0.001}
    DEFAULT_HIGH = {'epochs':5000, 'patience': 30, 'jiggle_fraction': 0.2}
    AS_INT = ['epochs','patience']

    fixed_search_params = dict() if fixed_search_params is None else fixed_search_params
    u_ndx = 1
    search_params = dict()
    for thing in ['epochs','patience','jiggle_fraction']:
        if thing in fixed_search_params:
            search_params[thing]=info[thing]
        else:
            low = DEFAULT_LOW[thing]
            high = DEFAULT_HIGH[thing]
            number = to_log_space_1d(us[u_ndx], low=low, high=high)
            if thing in AS_INT:
                number = int(number)
            u_ndx+=1
            search_params[thing]=number

    return model, search_params, info




KERAS_NEARBY = [keras_nearby_quick_2, keras_nearby_quick_3]



if __name__=='__main__':
    import numpy as np
    n_input = 160
    for _ in range(10):
        x = np.random.randn(1000,1,n_input)
        model, search_params, info = keras_nearby_quick_2(us=list(np.random.rand(2,1)),n_input=n_input)
        print(model.summary())
        y = model(x)
        print(np.shape(y))