import tensorflow as tf
import numpy as np 
import pandas as pd

from src.network import CNN

class Agent:

    def __init__(self, config, restore_dir=None):
        
        self.train_config = config['training']        
        self.batch_size = self.train_config['batch_size']
        self.input_no = self.batch_size
        
        self.input_config = config['input']
        self.coin_no =self.input_config['coin_no']
        self.window_size = self.input_config['window_size']
        self.global_period = self.input_config["global_period"]
        self.feature_no = self.input_config['feature_no']
        
        self.commission_ratio = config['trading']["trading_consumption"]
        
        self.pv_vector = None
        
        self.model = CNN(
            self.coin_no, 
            self.window_size,
            self.feature_no,
            config['training']['batch_size']
        )

        if restore_dir:
            self.model.load_weights(restore_dir)

        
        # NOTE: Can be customized
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        
    #@tf.function
    def train_step(self, batch):
        
        w = batch['last_w']
        w = tf.reshape(w, [w.shape[0], w.shape[1], 1, 1] )
        X = tf.transpose(batch['X'], [0, 2, 3, 1])   # (coins, time, features) that is, channels last. How tf likes it
        y = batch['y']
        self.input_no = y.shape[0]
                
        with tf.GradientTape() as tape:
                output = self.model([X, w])
                                
                # Compute negative reward
                loss = self.loss(y, output)

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        # Save the model output in PVM
        batch['setw'](output[:, 1:].numpy())


    def loss(self, y, output):
        #r_t = log(mu_t * y_t dot w_{t-1})
        self.future_price = tf.concat([tf.ones([self.input_no, 1]), y[:, 0, :]], 1) # Add cash price (always 1)
        self.future_w = (self.future_price * output) / tf.reduce_sum(self.future_price * output, axis=1)[:, None]
        self.pv_vector = tf.reduce_sum(output * self.future_price, axis=1) *\
                           (tf.concat([tf.ones(1), self.pure_pc(output)], 0))
        
        
        return -tf.reduce_mean(tf.math.log(self.pv_vector))
        
        
    # consumption vector (on each periods)
    def pure_pc(self, output):
        c = self.commission_ratio
        w_t = self.future_w[:self.input_no-1]  # rebalanced
        w_t1 = output[1:self.input_no]
        mu = 1 - tf.reduce_sum(tf.math.abs(w_t1[:, 1:]-w_t[:, 1:]), axis=1)*c
        """
        mu = 1-3*c+c**2

        def recurse(mu0):
            factor1 = 1/(1 - c*w_t1[:, 0])
            if isinstance(mu0, float):
                mu0 = mu0
            else:
                mu0 = mu0[:, None]
            factor2 = 1 - c*w_t[:, 0] - (2*c - c**2)*tf.reduce_sum(
                tf.nn.relu(w_t[:, 1:] - mu0 * w_t1[:, 1:]), axis=1)
            return factor1*factor2

        for i in range(20):
            mu = recurse(mu)
        """
        return mu


    def evaluate(self, batch):
        w = batch['last_w']
        w = tf.reshape(w, [w.shape[0], w.shape[1], 1, 1] )
        X = tf.transpose(batch['X'], [0, 2, 3, 1])   # (coins, time, features) that is, channels last. How tf likes it
        y = batch['y'] 
        self.input_no = y.shape[0]

        output = self.model([X, w])
        loss = self.loss(y, output)

        self.portfolio_value = tf.reduce_prod(self.pv_vector)
        self.mean = tf.reduce_mean(self.pv_vector)
        self.log_mean = tf.reduce_mean(tf.math.log(self.pv_vector))
        self.standard_deviation = tf.math.sqrt(tf.reduce_mean((self.pv_vector - self.mean) ** 2))
        self.sharp_ratio = (self.mean - 1) / self.standard_deviation

        self.log_mean_free = tf.math.reduce_mean(tf.math.log(tf.reduce_sum(output * self.future_price,
                                                                   axis=1)))

        return self.pv_vector, loss, output


    def call_model(self, history, prev_w):
        assert isinstance(history, np.ndarray),\
            "the history should be a numpy array, not %s" % type(history)
        assert not np.any(np.isnan(prev_w))
        assert not np.any(np.isnan(history))

        return self.model([history, prev_w])

    def recycle(self):
        self.model = CNN(
            self.coin_no, 
            self.window_size,
            self.feature_no,
            self.train_config['batch_size']
        )