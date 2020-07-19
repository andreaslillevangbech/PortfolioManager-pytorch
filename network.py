import tensorflow as tf
from tensorflow import keras

from config import config

# Cols are time and rows are assets

def CNN(batch_size, rows, cols, feature_no):

    X = keras.Input(shape=(rows, cols, feature_no), batch_size=batch_size)
    w = keras.Input(shape=(rows, 1, 1), batch_size=batch_size)

    feature_1 = keras.layers.Conv2D(2, (1,3), padding='valid', activation="relu", name = 'conv1')(X)
    feature_2 = keras.layers.Conv2D(20, (1, feature_1.shape[2]), activation="relu", name = 'conv2')(feature_1) 
    
    # Concat the w as an extra filter
    feature_2 = keras.layers.Concatenate(axis=3)([w, feature_2])
    feature_3 = keras.layers.Conv2D(1, (1,1), name = 'votes')(feature_2)

    # Add a cash bias
    feature_3 = feature_3[:,:,0,0]
    with_bias = CashBias()(feature_3)

    outputs = keras.layers.Activation('softmax')(with_bias)

    return keras.Model(inputs = [X, w], outputs = outputs, name = "Policy")

class CashBias(keras.layers.Layer):
    def __init__(self):
        super(CashBias, self).__init__()
    
    def build(self, input_shape):
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
                initial_value=b_init(shape=(1, 1), dtype="float32"),
                trainable=True
                )
        self.b = tf.tile(self.b, [input_shape[0], 1])

    def call(self, inputs):
        return keras.layers.Concatenate(axis=1)([self.b, inputs])

if __name__=='__main__':
    model = CNN(config['batch_size'],
            config['input_shape']['coins'],
            config['input_shape']['window_size'],
            config['input_shape']['feature_no'] )
    print(model.summary())
    # keras.utils.plot_model(model, "CNN.png", show_shapes=True)



# Reward function
# mean( ln ( mu_t * (y_t dot w_{t-1} ) ) )
# y_t is the relative price vector. Closing price over open price for period t
# mu_t is the cost of adjusting portfolio.
#def loss():
    #return tf.reduce_mean( mu_t * 
