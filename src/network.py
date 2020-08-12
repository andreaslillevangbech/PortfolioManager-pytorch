import tensorflow as tf
from tensorflow import keras
import numpy as np

from config import config

# Cols are time and rows are assets
# input size is 11x50x3
# Remember: Channels last

class CNN(tf.keras.Model):
    
    def __init__(self, rows = 11, cols = 50, features = 3, batch_size=None):
        super(CNN, self).__init__()
        
        self.tensor_shape = (rows, cols, features)
        self.batch_size = batch_size
        
        self.conv1 = tf.keras.layers.Conv2D(
                            filters = 2, 
                            kernel_size = (1,3), 
                            padding='valid', 
                            activation='relu',
                            name = 'conv1'
                        )    
        
        self.conv2 =  keras.layers.Conv2D(
                            filters = 20, 
                            kernel_size = (1, cols-2), 
                            activation="relu", 
                            name = 'conv2'
                        )
        self.votes = keras.layers.Conv2D(1, (1,1), name = 'votes')
        self.b = tf.Variable(tf.zeros((1, 1), dtype=tf.float32), trainable=True)
        self.softmax = tf.keras.layers.Activation('softmax')

    def call(self, inputs):
        x = self.conv1(inputs[0])
        x = self.conv2(x)
        x = tf.concat((x, inputs[1]), axis=3)
        #x = keras.layers.Concatenate(axis=3)([x, inputs[1]])
        x = self.votes(x)
        x = tf.squeeze(x)
        cash_bias = tf.tile(self.b, [tf.shape(x)[0], 1])
        x = tf.concat((cash_bias, x), axis = -1)
        x = self.softmax(x)
        return x

if __name__=='__main__':
    model = CNN()

    X = np.random.randn(100, 11,50,3)
    w = np.random.randn(100, 11, 1, 1)
    with tf.GradientTape() as tape:
        y = model([X,w])

    print([var.name for var in tape.watched_variables()])
    grads = tape.gradient(y, model.trainable_variables)
    print(grads)
    # keras.utils.plot_model(model, "CNN.png", show_shapes=True)

