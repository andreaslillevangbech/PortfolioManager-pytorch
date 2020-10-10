#!/usr/bin/env python
from src.network import CNN
from config import config
import numpy as np
import tensorflow as tf

model = CNN(feature_number=3, rows=11, columns=50, layers=config["layers"])
def loss(y, output):
        #r_t = log(mu_t * y_t dot w_{t-1})
        input_no = tf.shape(y)[0]
        future_price = tf.concat([tf.ones([input_no, 1]), y[:, 0, :]], 1) # Add cash price (always 1)
        future_w = (future_price * output) / tf.reduce_sum(future_price * output, axis=1)[:, None]
        pv_vector = tf.reduce_sum(output * future_price, axis=1) *\
                           (tf.concat([tf.ones(1), pure_pc(output, input_no, future_w)], 0))
        
        return -tf.reduce_mean(tf.math.log(pv_vector))

def pure_pc(output, input_no, future_w):
        c = 0.0025
        w_t = future_w[:input_no-1] 
        w_t1 = output[1:input_no]
        mu = 1 - tf.reduce_sum(tf.math.abs(w_t1[:, 1:]-w_t[:, 1:]), axis=1)*c
        return mu

opt = tf.compat.v1.train.AdamOptimizer()

X = tf.random.uniform((100, 11, 50, 3), minval=0.0001, maxval=0.2)
w = tf.random.uniform((100, 11, 1, 1), minval=0.5, maxval=1.5)
w = w/tf.reduce_sum(w)
y = tf.random.uniform((100, 3, 11), minval=0.5, maxval=1.5)
x = [X,w]

with tf.GradientTape() as tape:
    y_hat = model(x)
    total = loss(y, y_hat) + tf.math.add_n(model.losses)

print([var.name for var in tape.watched_variables()])
print("Variables: ", [var for var in tape.watched_variables()])
print("Weigts: " , [var for var in model.trainable_variables])
grads = tape.gradient(y, model.trainable_variables)
print('\n')
print('gradients')
for grad in grads:
    print(grad)
print('\n')
print('output: ', y)

from src.network import CNN_Keras
from config import config
import numpy as np
import tensorflow as tf
PVM = np.ones(shape=(1000,11))
PVM = PVM / PVM.sum(axis=1)[:,None]
model = CNN_Keras(features=3, rows=11, cols=50, layers=config["layers"], PVM_array=PVM)
X = tf.random.uniform((100, 11, 50, 3), minval=0.0001, maxval=0.2)
y = tf.random.uniform((100, 3, 11), minval=0.5, maxval=1.5)
idx = np.arange(112,212)
x = [X,idx]

