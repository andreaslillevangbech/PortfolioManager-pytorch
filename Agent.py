import tensorflow as tf
from tensorflow import keras

from .network import CNN

class Agent:
    def __init__(self):
        self.model = CNN()
        self.pv = tf.zeros(coin.no)

# Training is done in mini batches. For a full test period, [0, t_f], parameters are updated in batches.
# 
