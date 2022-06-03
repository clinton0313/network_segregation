#%%
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

from copy import deepcopy
from segregation_simulation import Environment
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.realpath(__file__)))


def run_experiment(rounds: int, file: str, seed: int = 0, **env_args):

    if not os.path.isfile(file):
        data = pd.DataFrame(
            columns = (
                ["seed", "environment", "rounds", "a_segregation", "b_segregation"] 
                + list(env_args.keys())
            )
        )
        with open(file, "wb") as outfile:
            pickle.dump(data, outfile)

    with open(file, "rb") as infile:
        data = pickle.load(infile)
    env = Environment(**env_args)
    env.random_assignment()
    a_segregation, b_segregation = env.simulate(rounds)

    data.loc[len(data.index)] = [seed, deepcopy(env), rounds, a_segregation, b_segregation] + list(env_args.values())

    with open(file, "wb") as outfile:
        pickle.dump(data, outfile)



a_class = pd.read_csv(os.path.join("data", "a.csv"))
b_class = pd.read_csv(os.path.join("data", "b.csv"))
# %%

#Standard experiment not firm racism. 
exp1 = {
    "num_firms": 100,
    "capacities": 150,
    "alphas": 1,
    "betas": 1,
    "num_group_a": len(a_class),
    "num_group_b": len(b_class),
    "gammas_a": a_class["x"].values,
    "gammas_b": b_class["x"].values,
}

exp2 = deepcopy(exp1)
exp2.update({
    "alphas": 0.5,
    "betas": 0.5
})

gamma_std_a = np.var(a_class["x"].values) ** 0.5
gamma_std_b = np.var(b_class["x"].values) ** 0.5

pop_prop_a = len(a_class) / (len(a_class) + len(b_class))
pop_prop_b = len(b_class) / (len(a_class) + len(b_class))

#gammas and betas equal to proportion of population. Fair preferences.

gamma_a_range = np.linspace(0.05, pop_prop_a, 10)
gamma_b_range = np.linspace(0.05, pop_prop_b, 10)

exps = [
    {
        "num_firms": 100,
        "capacities": 150,
        "alphas": alpha,
        "betas": beta,
        "num_group_a": len(a_class),
        "num_group_b": len(b_class),
        "gammas_a": np.random.normal(gamma_a, gamma_std_a, len(a_class)),
        "gammas_b": np.random.normal(gamma_b, gamma_std_b, len(b_class)),
    }
    for (alpha, beta), gamma_a, gamma_b
    in itertools.product(zip([0.5, 1], [0.5, 1]), gamma_a_range, gamma_b_range)
]

exps.insert(0, exp1)
exps.insert(1, exp2)


# %%

for env_args, seed in tqdm(itertools.product(exps, range(10))):
    run_experiment(200, "segregation_experiments.pkl", seed, **env_args)