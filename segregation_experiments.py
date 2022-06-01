#%%
import matplotlib.pyplot as plt
import os
import pandas as pd

from segregation_simulation import Environment
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.realpath(__file__)))

#%%

a_class = pd.read_csv(os.path.join("data", "a.csv"))
b_class = pd.read_csv(os.path.join("data", "b.csv"))
# %%

env = Environment(
    num_firms = 100,
    capacities = 150,
    alphas = 1,
    betas = 1,
    num_group_a = len(a_class),
    num_group_b = len(b_class),
    gammas_a = a_class["x"].values,
    gammas_b = b_class["x"].values
)

env.random_assignment()
env.plot_network()

#%%
env.simulate(100)
env.plot_network()
# %%

