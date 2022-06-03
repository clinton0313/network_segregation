#%%
from typing import Optional
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

os.chdir(os.path.dirname(os.path.realpath(__file__)))
matplotlib.rcParams.update({
    "figure.figsize": (12, 12),
    "axes.spines.top": False,
    "axes.spines.right": False
})
matplotlib.use("tkagg")

def unpack_segregation(segregation: list, index: int) -> np.ndarray:
    unpacked = np.array([seg[index] for seg in segregation])
    return unpacked

def parse_results(res: pd.DataFrame) -> pd.DataFrame:
    res["a_isolation"] = res["a_segregation"].apply(lambda x: unpack_segregation(x, 0))
    res["a_dissimilarity"] = res["a_segregation"].apply(lambda x: unpack_segregation(x, 1))
    res["b_isolation"] = res["b_segregation"].apply(lambda x: unpack_segregation(x, 0))
    res["b_dissimilarity"] = res["b_segregation"].apply(lambda x: unpack_segregation(x, 1))

    res["gamma_a_mean"] = res["gammas_a"].apply(lambda x: np.mean(x))
    res["gamma_b_mean"] = res["gammas_b"].apply(lambda x: np.mean(x))

    res = res.drop(
        columns=[
            "environment", 
            "a_segregation",
            "b_segregation",
            "gammas_a",
            "gammas_b"
        ]
    )

    groupby = [
        col 
        for col in res.columns 
        if col not in [
            "seed", 
            "a_isolation", 
            "a_dissimilarity", 
            "b_isolation", 
            "b_dissimilarity"
        ]
    ]

    res = res.groupby(groupby).agg(
        {
            "a_isolation": lambda x: np.mean(np.stack(x, axis=1), axis=1),
            "a_dissimilarity": lambda x: np.mean(np.stack(x, axis=1), axis=1),
            "b_isolation": lambda x: np.mean(np.stack(x, axis=1), axis=1),
            "b_dissimilarity": lambda x: np.mean(np.stack(x, axis=1), axis=1)

        }
    )

    res["a_final_isolation"] = res["a_isolation"].apply(max)
    res["a_final_dissimilarity"] = res["a_dissimilarity"].apply(max)
    res["b_final_isolation"] = res["b_isolation"].apply(max)
    res["b_final_dissimilarity"] = res["b_dissimilarity"].apply(max)


    res = res.reset_index()

    return res


def plot_segregations(record: pd.Series, savepath: Optional[str] = None) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.plot(
        range(record.rounds + 1), 
        record.a_isolation, 
        color="tab:blue",
        label="Group A Isolation"
    )
    ax.plot(
        range(record.rounds + 1), 
        record.a_dissimilarity, 
        color="tab:blue",
        label="Group A Dissimilarity",
        linestyle="dashed"
    )
    ax.plot(
        range(record.rounds + 1), 
        record.b_isolation, 
        color="tab:red",
        label="Group B Isolation"
    )
    ax.plot(
        range(record.rounds + 1), 
        record.b_dissimilarity, 
        color="tab:red",
        label="Group B Dissimilarity",
        linestyle="dashed"
    )
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 20)
    ax.legend()
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Segregation")
    ax.set_title(
        f"Alphas: {record.alphas}; Betas: {record.betas}; Gamma A "
        f"{str(round(record.gamma_a_mean, 3)).replace('.', '_')}; Gamma "
        f"B {str(round(record.gamma_b_mean, 3)).replace('.', '_')}"
    )

    if savepath is not None:
        fig.savefig(
            os.path.join(
                savepath, 
                f"a_{record.alphas}_b_{record.betas}_ga_"
                f"{str(round(record.gamma_a_mean, 3)).replace('.', '_')}_gb_"
                f"{str(round(record.gamma_b_mean, 3)).replace('.', '_')}.png"
            ),
            facecolor="white",
            transparent=False
        )
        plt.close(fig)
        fig.clear()
    else:
        return fig

#%%

res = pd.read_pickle("segregation_experiments2.pkl")
parsed = parse_results(res)
parsed.apply(lambda x: plot_segregations(x, savepath="figs"), axis=1)

# %%
