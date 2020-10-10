import os
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import summary_ops_v2
from config import config

layers = config["layers"]


class CNN(tf.Module):
    name=''
    
    def __init__(self, rows = 11, cols = 50, features = 3, layers=None, name = 'GraphTest'):
        super(CNN, self).__init__()

        self.name=name
        
        filter_shape = layers[0]["filter_shape"]

        self.conv1 = tf.keras.layers.Conv2D(
                            filters = layers[0]["filter_number"], 
                            kernel_size = filter_shape, 
                            kernel_initializer= tf.keras.initializers.truncated_normal,
                            padding='valid', 
                            activation='relu',
                            name = 'conv1'
                        )    
        
        self.conv2 =  tf.keras.layers.Conv2D(
                            filters = layers[1]["filter_number"], 
                            kernel_size = (1, cols-(filter_shape[1]-1)),
                            kernel_initializer= tf.keras.initializers.truncated_normal,
                            kernel_regularizer=tf.keras.regularizers.L2(
                                                l2=layers[1]["weight_decay"]),
                            activation="relu", 
                            name = 'conv2'
                        )

        self.votes = tf.keras.layers.Conv2D(
                                    1, 
                                    (1,1), 
                                    kernel_initializer= tf.keras.initializers.truncated_normal,
                                    kernel_regularizer=tf.keras.regularizers.L2(
                                    l2=layers[2]["weight_decay"]), 
                                    name = 'votes'
                        )

        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(1,1), dtype="float32"), 
            trainable=True,
            name = 'btc_bias'
        )

    @tf.function
    def __call__(self, inputs):
        x = self.conv1(inputs[0])
        x = self.conv2(x)
        x = tf.concat((x, inputs[1]), axis=3)
        x = self.votes(x)
        x = tf.squeeze(x)
        cash_bias = tf.tile(self.b, [tf.shape(x)[0], 1])
        x = tf.concat((cash_bias, x), axis = -1)
        x = tf.nn.softmax(x)
        return x
 
model = CNN(layers=layers)

cwd=os.getcwd()
rel_log_dir='model_graph'
logs_dir=os.path.join(cwd,rel_log_dir)
# Create a file_writer for the graph
graph_writer = tf.summary.create_file_writer(logdir=logs_dir)

a = tf.random.normal((100, 11, 50,3))
b = tf.random.normal((100, 11, 1, 1))
inputs = [a,b]

# Write the graph
with graph_writer.as_default():
    graph=model.__call__.get_concrete_function(inputs).graph
    summary_ops_v2.graph(graph.as_graph_def())
graph_writer.close()

