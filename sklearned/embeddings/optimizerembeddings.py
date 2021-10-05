from sklearned.embeddings.transforms import choice_from_dict
from tensorflow import keras

OPTIMIZER_WEIGHTS = {'SGD' :10,
                   'RMSprop' :20,
                   'Adam' :5,
                   'Adadelta' :3,
                   'Adagrad' :3,
                   'Adamax' :3,
                   'Nadam' :3,
                   'Ftrl' :2}


def keras_optimizer_name(u:float):
    return choice_from_dict(u=u, weighting=OPTIMIZER_WEIGHTS)


def keras_optimizer(u:float, learning_rate):
    opt_name = keras_optimizer_name(u)
    if opt_name == 'SGD':
        return keras.optimizers.SGD(learning_rate=learning_rate)
    elif opt_name == 'RMSprop':
        return keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif opt_name == 'Adam':
        return keras.optimizers.Adam(learning_rate=learning_rate)
    elif opt_name == 'Adagrad':
        return keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif opt_name == 'Adamax':
        return  keras.optimizers.Adamax(learning_rate=learning_rate)
    elif opt_name == 'Nadam':
        return keras.optimizers.Nadam(learning_rate=learning_rate)
    elif opt_name =='Ftrl':
        return keras.optimizers.Ftrl(learning_rate=learning_rate)
    else:
        raise ValueError()

if __name__=='__main__':
    print(keras_optimizer_name(u=0.999))
    print(keras_optimizer(u=0.4, learning_rate=0.1))