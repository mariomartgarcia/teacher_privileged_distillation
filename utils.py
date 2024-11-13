import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, Activation, Input, Concatenate, Lambda
import numpy as np
import pandas as pd
import tensorflow as tf




def feat_correlation(X, y):
    """Selects as the privileged variable the most correlated one (w.r.t the class)

    Args:
        X (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """
    df = pd.concat([X, y], axis = 1)
    val1 = sorted(list(np.abs(df.corr(method = 'spearman')).iloc[-1]), reverse = True)[1]
    indice1 = list(np.abs(df.corr(method = 'spearman')).iloc[-1]).index(val1)
    pi_features = [X.columns[indice1]][0]
    return pi_features




def nn_binary_clasification(dim, lay, activatio = 'relu'):
    kernel_init = tf.keras.initializers.RandomNormal(mean=0., stddev=0.01, seed = 0)
    bias_init = tf.keras.initializers.RandomNormal(mean=0., stddev=0.01, seed = 0)
    model = keras.Sequential()
    model.add(Input(shape=(dim,)))
    if lay:        
        for units in lay:
            model.add(Dense(units, activation=activation, kernel_initializer = kernel_init, bias_initializer = bias_init))

    model.add(Dense(1, activation='sigmoid', kernel_initializer = kernel_init, bias_initializer = bias_init))

    return model

