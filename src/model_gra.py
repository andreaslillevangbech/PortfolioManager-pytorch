import tensorflow as tf

@tf.function
def model(X, b):
    x = tf.keras.layers.Conv2D(10, [1,])(X[0])
    x = tf.keras.layers.Conv2D(1, (1,1,))(x)

    x = tf.squeeze(x)
    cash = tf.tile(b, [tf.shape(x)[0], 1])
    x = tf.concat((cash, x), axis=-1)
    x = tf.nn.softmax(x)
    return x

b = tf.Variable(tf.zeros((1,1)), trainable=True)
X = np.random.randn(100, 11,50,3)
w = np.random.randn(100, 11, 1, 1)