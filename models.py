# %%
def model_SimpleRNN():
    
    # X : shape(m, Tx, encoding)
    
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    pitch_range = 49
    beats_range = 51
    batch_size = 32
    
    # X = layers.Input(shape=(None, pitch_range+beats_range),
    #                  batch_size = batch_size)
    
    X = layers.Input(shape=(batch_size, pitch_range+beats_range+1))
    
    A, _ = layers.SimpleRNN(units = 100,
                            activation = 'tanh',
                            # stateful = True,
                            return_state = True,
                            return_sequences = True)(X)
    
    Y = layers.Dense(pitch_range + beats_range + 1)(A)
    
    model = keras.Model(inputs = X,
                        outputs = Y)
    
    return model
    

    
# %%
# def model_coursera(batch_size = 32, n_a = 50, n_values = 101):
    
#     from tensorflow.keras.layers import Reshape, LSTM, Dense, Input, Lambda
#     from tensorflow.keras.models import Model
    
#     Tx = batch_size

#     X = Input(shape=(Tx, n_values))
    
#     reshapor = Reshape((1, n_values))
#     LSTM_cell = LSTM(n_a, return_state = True)
#     densor = Dense(n_values, activation='softmax')
    
#     a0 = Input(shape=(n_a,), name='a0')
#     c0 = Input(shape=(n_a,), name='c0')
#     a = a0
#     c = c0
    
#     outputs = []
    
#     for t in range(Tx):
#         x = Lambda(lambda x: X[:,t,:])(X)
#         x = reshapor(x)
#         a, _, c = LSTM_cell(inputs=x, initial_state=[a, c])
#         out = densor(a)
#         outputs.append(out)
        
#     model = Model(inputs=[X, a0, c0], outputs=outputs)
    
#     return model

