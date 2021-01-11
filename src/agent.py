import torch
import numpy as np
import pandas as pd

from src.network import CNN

class Agent:

    def __init__(self, config, time_index=None, coins=None, restore_dir=None, device="cpu"):

        self.config = config
        self.train_config = config['training']
        self.learning_rate = self.train_config['learning_rate']

        self.input_config = config['input']
        self.coin_no =self.input_config['coin_no']
        self.window_size = self.input_config['window_size']
        self.feature_no = self.input_config['feature_no']

        self.commission_ratio = config['trading']["trading_consumption"]

        self.model = CNN(
            self.feature_no,
            self.coin_no,
            self.window_size,
            config["layers"],
            device = device
        )
        
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(dev)

        if restore_dir:
            self.model.load_state_dict(torch.load(restore_dir))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train_step(self, X, w, y, setw):
        output = self.model(X, w)
        loss = self.loss(output, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #NOTE: ADD regu loss

        # Update weights in PVM
        setw(output[:, 1:].detach().numpy())

    def loss(self, output, y):
        #r_t = log(mu_t * y_t dot w_{t-1})
        input_no = y.shape[0]
        future_price = torch.cat([torch.ones((input_no, 1)), y[:, 0, :]], dim=1) # Add cash price (always 1)
        future_w = (future_price * output) / torch.sum(future_price * output, dim=1)[:, None]
        pv_vector = torch.sum(output * future_price, dim=1) *\
                           (torch.cat([torch.ones(1), self.pure_pc(output, input_no, future_w)], dim=0))

        return -torch.mean(torch.log(pv_vector))

    def pure_pc(self, output, input_no, future_w):
        c = self.commission_ratio
        w_t = future_w[:input_no-1]  # rebalanced
        w_t1 = output[1:input_no]
        mu = 1 - torch.sum(torch.abs(w_t1[:, 1:]-w_t[:, 1:]), dim=1)*c
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


    def test_step(self, X, w, y):
        output = self.model(X, w)
        loss = self.loss(output, y)
        return loss, output

    def evaluate(self, X, w, y):
        loss, output = self.test_step(X, w, y)

        input_no = y.shape[0]
        future_price = torch.cat([torch.ones((input_no, 1)), y[:, 0, :]], dim=1) # Add cash price (always 1)
        future_w = (future_price * output) / torch.sum(future_price * output, dim=1)[:, None]
        self.pv_vector = torch.sum(output * future_price, dim=1) *\
                           (torch.cat([torch.ones(1), self.pure_pc(output, input_no, future_w)], dim=0))

        self.portfolio_value = torch.prod(self.pv_vector)
        self.mean = torch.mean(self.pv_vector)
        self.log_mean = torch.mean(torch.log(self.pv_vector))
        self.standard_deviation = torch.sqrt(torch.mean((self.pv_vector - self.mean) ** 2))
        self.sharp_ratio = (self.mean - 1) / self.standard_deviation

        self.log_mean_free = torch.mean(torch.log(torch.sum(output * future_price,
                                                                   dim=1)))

        return self.pv_vector, loss, output


    def call_model(self, history, prev_w):
        assert isinstance(history, np.ndarray),\
            "the history should be a numpy array, not %s" % type(history)
        assert not np.any(np.isnan(prev_w))
        assert not np.any(np.isnan(history))

        return self.model(history, prev_w)

