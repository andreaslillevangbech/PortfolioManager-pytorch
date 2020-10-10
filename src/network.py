import tensorflow as tf
import numpy as np

# Cols are time and rows are assets
# input size is 11x50x3
# Channels last

# Version without Keras
class CNN(tf.Module):

    def __init__(self, feature_number, rows, columns, layers, name="CNN"):
        super(CNN, self).__init__(name=name)

        self.layers=layers
        self.losses = []
        self.initializer = tf.keras.initializers.TruncatedNormal()

        filter_shape = layers[0]["filter_shape"]
        features = [feature_number]
        filters_out = layers[0]["filter_number"]
        kernel_shape = filter_shape + features + [filters_out]

        self.kernel_1 = tf.Variable(self.initializer(shape=kernel_shape), name = "conv1")
        self.bias_1 = tf.Variable(tf.zeros([filters_out]), name = "bias1")
        self.losses.append(self.layers[0]["weight_decay"]*tf.nn.l2_loss(self.kernel_1))

        self.is_built = False

        self.kernel_votes = tf.Variable(self.initializer(shape=(1,1, self.layers[1]["filter_number"]+1, 1)), name="votes")
        self.bias_votes = tf.Variable(tf.zeros([1]), name="biasVotes")
        self.losses.append(self.layers[2]["weight_decay"]*tf.nn.l2_loss(self.kernel_votes))

        self.b = tf.Variable(
            initial_value=tf.zeros(shape=(1,1), dtype="float32"), 
            trainable=True,
            name = 'btc_bias'
        )

    def __call__(self, inputs):
        x = tf.nn.conv2d(inputs[0], self.kernel_1, strides=1, padding='VALID')
        x = tf.nn.bias_add(x, self.bias_1)
        x = tf.nn.relu(x)

        if not self.is_built:
            filter_shape = [1, tf.shape(x)[2]]
            features = [tf.shape(x)[3]]
            filters_out = self.layers[1]["filter_number"]
            kernel_shape = filter_shape + features + [filters_out]
            self.kernel_2 = tf.Variable(self.initializer(shape=kernel_shape), name = "conv2")
            self.bias_2 = tf.Variable(tf.zeros([filters_out]), name = "bias2")
            self.losses.append(self.layers[1]["weight_decay"]*tf.nn.l2_loss(self.kernel_2)) 
            self.is_built = True
        
        x = tf.nn.conv2d(x, self.kernel_2, strides=1, padding='VALID')
        x = tf.nn.bias_add(x, self.bias_2)    
        x = tf.nn.relu(x)
        x = tf.concat((x, inputs[1]), axis=3)
        x = tf.nn.conv2d(x, self.kernel_votes, strides=1, padding='VALID')
        x = tf.nn.bias_add(x, self.bias_votes)
        x = tf.squeeze(x)
        cash_bias = tf.tile(self.b, [tf.shape(x)[0], 1])
        x = tf.concat((cash_bias, x), axis = -1)
        return tf.nn.softmax(x)

        


class CNN_Keras(tf.keras.Model):

    def __init__(self, features, rows, cols, layers=None, PVM_array=None):
        super(CNN_Keras, self).__init__()

        self.PVM = tf.Variable(PVM_array, name='PVM', trainable=False, dtype="float32")

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
                            padding='valid',
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

        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs):
        x = self.conv1(inputs[0])
        x = self.conv2(x)
        idx = tf.squeeze(inputs[1])
        last_w = tf.gather(self.PVM, idx-1, axis=0)
        last_w = last_w[:,:,tf.newaxis, tf.newaxis]
        # last_w = tf.reshape(last_w, [tf.shape(last_w)[0], tf.shape(last_w)[1], 1, 1])
        x = tf.concat((x, last_w), axis=3)
        x = self.votes(x)
        x = tf.squeeze(x)
        cash_bias = tf.tile(self.b, [tf.shape(x)[0], 1])
        x = tf.concat((cash_bias, x), axis = -1)
        x = self.softmax(x)
        f = tf.squeeze(inputs[1][0])
        t = tf.squeeze(inputs[1][-1]+1)
        self.PVM[f:t, :].assign(x[:, 1:])
        return x
