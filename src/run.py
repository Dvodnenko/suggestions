import numpy as np

from .agent import Agent


init_preferences = np.zeros(5) + 1/np.sqrt(5)
agent = Agent(init_preferences, 0.1, 0.8)

while True:
    agent.step()
