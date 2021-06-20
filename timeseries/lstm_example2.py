import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

x_train = tf.random.normal([5358, 300, 54])
y_train = tf.random.normal([5358, 1])

input_layer = Input(shape=(300, 54))
lstm = LSTM(100)(input_layer)
dense1 = Dense(20, activation='relu')(lstm)
dense2 = Dense(1, activation='sigmoid')(dense1)

model = Model(inputs=input_layer, outputs=dense2)
model.compile("adam", loss='binary_crossentropy')
model.fit(x_train, y_train, batch_size=512)