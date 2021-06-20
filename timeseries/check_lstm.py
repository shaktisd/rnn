from tensorflow.keras.layers import Embedding
import tensorflow as tf
import numpy as np

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=555, output_dim= 100, input_length=10))
#The model will take as input an integer matrix of size vocab size = 555 (batch , input_length)
# output = input_length , output_dim
#model.add(tf.keras.layers.LSTM(25))

input_array = np.random.randint(555, size=(1, 10))
output_array = model.predict(input_array)
print(output_array.shape)

# inputs: A 3D tensor with shape [batch, timesteps, feature].
inputs = tf.random.normal([32, 10, 8])
lstm = tf.keras.layers.LSTM(4)
output = lstm(inputs)
print(output.shape)
