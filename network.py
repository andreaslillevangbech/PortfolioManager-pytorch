import tensorflow as tf
from tf import keras

inputs = keras.Input(shape=(784,))
x = keras.layers.Dense(64, activation="relu")(inputs)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs = inputs, outputs = outputs, name = "Policy")

print(model.summary())

