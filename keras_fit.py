#!/usr/bin/env python

import logging
from src.kerastrainer import KerasTrainer
from config import config
import os

train_dir = "keras_train"
if not os.path.exists("./" + train_dir): 
    os.makedirs("./" + train_dir)

log_file_dir = './' + train_dir + '/keras_logs'
logfile_level=logging.DEBUG 
console_level=logging.INFO

if log_file_dir:
    logging.basicConfig(filename=log_file_dir,
                        level=logfile_level)
    console = logging.StreamHandler()
    console.setLevel(console_level)
    logging.getLogger().addHandler(console)
        
KerasTrainer(config).keras_fit()