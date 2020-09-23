from src.config import config
from src.agent import Agent

agent = Agent(config)

batch = agent.matrix.next_batch()
