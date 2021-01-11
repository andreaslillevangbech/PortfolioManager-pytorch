from src.trainer import Trainer
from config import config
import pandas as pd

tra = Trainer(config)
df = tra.test_set['X']
df = df[0,0,:,:]
df = pd.DataFrame(df)
df.to_csv("/Users/andreasbech/torch.csv")

