import tensorflow as tf
import numpy as np 
import pandas as pd

from src.network import CNN
from src.data.replayBuffer import ReplayBuffer

class Agent:

    def __init__(self, config):
        
        self.train_config = config['training']        
        self.batch_size = self.train_config['batch_size']
        self.buffer_bias = self.train_config['buffer_bias']
        
        self.input_config = config['input']
        self.coin_no =self.input_config['coin_no']
        self.window_size = self.input_config['window_size']
        self.global_period = self.input_config["global_period"]
        self.feature_no = self.input_config['feature_no']
        
        # self.no_periods should be equal to len(global data matrix)
        self.no_periods = 150
        
        self.commission_ratio = config['trading']["trading_consumption"]
        
        #Just make something random
        self.global_data = tf.random.uniform(shape = (self.feature_no, self.coin_no, self.no_periods))
        
        PVM = np.ones((self.global_data.shape[2], self.global_data.shape[1]), dtype='float32')/self.coin_no
        self.PVM = pd.DataFrame(PVM)
        
    #    # Notice this part is made with pandas.panel
    #          # portfolio vector memory, [time, assets]
    #         self.__PVM = pd.DataFrame(index=self.__global_data.minor_axis,
    #                                   columns=self.__global_data.major_axis)
    #         self.__PVM = self.__PVM.fillna(1.0 / self.__coin_no)

        
        self.pv_vector = None
        
        
        self.model = CNN(
            self.coin_no, 
            self.window_size,
            self.feature_no,
            config['training']['batch_size']
        )
        
        self.divide_data(config['input']['test_portion']) # This gives the indekses of the training and test data
        
        # This needs to be written such that it gets arguments from config, like sample bias (geo dist)
        end_index = self._train_ind[-1]
        self.__replay_buffer = ReplayBuffer(start_index=self._train_ind[0],
                                               end_index=end_index,
                                               sample_bias=self.buffer_bias,
                                               batch_size=self.batch_size,
                                               coin_no=self.coin_no)
    #@tf.function
    def train_step(self, batch):
        
        w = batch['last_w']
        w = tf.reshape(w, [w.shape[0], w.shape[1], 1, 1] )
        X = tf.transpose(batch['X'], [0, 2, 3, 1])
        y = batch['y']
                
        with tf.GradientTape() as tape:
                output = self.model([X, w])
                                
                # Compute negative reward
                loss = self.loss(y, output)

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        # Save the model output in PVM
        #batch['setw'](w[:, 1:])
        self.PVM.iloc[self.indexs, :] = output[:, 1:].numpy()
        
        return loss

    def train(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        #loss_metric = -tf.keras.metrics.Mean()
        
        for step in range(self.train_config['steps']):
            
            batch = self.next_batch()

            # Do a train step
            loss_value = self.train_step(batch)
            
             # You can write a custom metric here. See tf.org Keras -> Train and Evaluate
            #loss_metric(loss)
            portfolio_value = tf.reduce_prod(self.pv_vector)
            mean = tf.reduce_mean(self.pv_vector)
            standard_deviation = tf.math.sqrt(tf.reduce_mean((self.pv_vector - mean) ** 2))
            sharp_ratio = (mean - 1) / standard_deviation
            
            # Log every 200 batches.
            if step % 200 == 0:
                print('Step %2d: loss=%2.5f, cumval=%.1f' %
                        (step, -loss_value, portfolio_value))           

                # You can add a log between steps here
                # Both manually and with tensorboard

    def log_between_steps(self, step):
        pass




            

    def next_batch(self):
        """
        @:return: the next batch of training sample. The sample is a dictionary
        with key "X"(input data); "y"(future relative price); "last_w" a numpy array
        with shape [batch_size, assets]; "w" a list of numpy arrays list length is
        batch_size
        """
        batch = self.pack_samples([exp.state_index for exp in self.__replay_buffer.next_experience_batch()])
        return batch

    def pack_samples(self, indexs):
        self.indexs = indexs
        indexs = np.array(indexs)
        last_w = self.PVM.values[indexs-1, :]

        def setw(w):                      # Notice that this function is defined in terms of the specifik indexs
            self.PVM.iloc[indexs, :] = w    
        M = [self.get_submatrix(index) for index in indexs]   # For each state_index in the batch, get a input tensor
        M = np.array(M, dtype='float32')
        X = M[:, :, :, :-1]    # X_t tensor
        y = M[:, :, :, -1] / M[:, 0, None, :, -2]     # y_{t+1} obtained by dividing all features by prev close price
        return {"X": X, "y": y, "last_w": last_w, "setw": setw}

    # volume in y is the volume in next access period
    def get_submatrix(self, ind):
        return self.global_data[:, :, ind-(self.window_size):ind+1]

    def get_test_set(self):
        return self.__pack_samples(self.test_indices)

    def get_training_set(self):
        return self.__pack_samples(self._train_ind[:-self._window_size])

    def divide_data(self, test_portion, portion_reversed = False):
        train_portion = 1 - test_portion
        s = float(train_portion + test_portion)
        if portion_reversed:
            portions = np.array([test_portion]) / s
            portion_split = (portions * self.no_periods).astype(int)
            indices = np.arange(self.no_periods)
            self._test_ind, self._train_ind = np.split(indices, portion_split)
        else:
            portions = np.array([train_portion]) / s
            portion_split = (portions * self.no_periods).astype(int)
            indices = np.arange(self.no_periods)
            self._train_ind, self._test_ind = np.split(indices, portion_split)

        self._train_ind = self._train_ind[(self.window_size):-1]
        # NOTE(zhengyao): change the logic here in order to fit both
        # reversed and normal version
        self._train_ind = list(self._train_ind)
        self._num_train_samples = len(self._train_ind)
        self._num_test_samples = len(self._test_ind)



        
    #get a loss function, which is minus the reward function
    def loss(self, y, output):
        #r_t = log(mu_t * y_t dot w_{t-1})
        
        self.future_price = tf.concat([tf.ones([16, 1]), y[:, 0, :]], 1)
        self.future_w = (self.future_price * output) / tf.reduce_sum(self.future_price * output, axis=1)[:, None]
        self.pv_vector = tf.reduce_sum(output * self.future_price, axis=1) *\
                           (tf.concat([tf.ones(1), self.pure_pc(output)], axis=0))
        
        
        return -tf.reduce_mean(tf.math.log(self.pv_vector))
        
        
        
        
    # consumption vector (on each periods)
    def pure_pc(self, output):
        c = self.commission_ratio
        w_t = self.future_w[:self.batch_size-1]  # rebalanced
        w_t1 = output[1:self.batch_size]
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


if __name__=='__main__':
    from config import config
    agent = Agent(config)
    print('Training model straigt from agent.py file', '\n')
    agent.train()

