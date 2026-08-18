import numpy as np

from .agent import Agent, Environment


env = Environment(10, np.random.normal(0, 1, 10))
agent = Agent(
    env,
    np.zeros(10) + 1,
    np.zeros(10) + 1,
)
