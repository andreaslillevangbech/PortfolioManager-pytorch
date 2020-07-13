import tensorflow as tf
from tensorflow import keras

# Inputs to the network should be in a config file
# Could be kept in a separate file
config = {}

def get_model():
    inputshape = (config['feature_no'], config['coin_no'], config['window_size'])

    inputs = keras.Input(shape=(784,))
    x = keras.layers.Dense(64, activation="relu")(inputs)
    outputs = keras.layers.Dense(10)(x)

    return keras.Model(inputs = inputs, outputs = outputs, name = "Policy")

model = get_model()
print(model.summary())

