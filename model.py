#!/usr/bin/env python
from src.network import CNN
import numpy as np
import tensorflow as tf

model = CNN()

X = np.random.randn(100, 11,50,3)
w = np.random.randn(100, 11, 1, 1)
with tf.GradientTape() as tape:
    y = model([X,w])

print([var.name for var in tape.watched_variables()])
grads = tape.gradient(y, model.trainable_variables)
print(grads)
# tf.keras.utils.plot_model(model, "CNN.png", show_shapes=True)
