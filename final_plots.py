#%%

import matplotlib
from matplotlib import tight_layout
import matplotlib.style
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from plot_segregation import parse_results
from typing import Optional

os.chdir(os.path.dirname(os.path.realpath(__file__)))

matplotlib.use("tkagg")
matplotlib.style.use("seaborn-bright")
matplotlib.rcParams.update({
    "figure.figsize": (12, 12),
    "axes.spines.top": False,
    "axes.spines.right": False
})


employer = pd.read_pickle("employer_experiments.pkl")
employee = pd.read_pickle("employee_experiments.pkl")


employer = parse_results(employer)
employee = parse_results(employee)


#%%

def plot_employer(res: pd.DataFrame, group: str = "alphas", savepath: Optional[str] = None) -> plt.Figure:
    
    data = res[res[group]== 1]
    other_group = "alphas" if group == "betas" else "betas"
    other_group_title = other_group.replace("s", "").capitalize()
    group_letter = list(group)[0]
    other_group_letter = list(group)[0]

    fig, ax = plt.subplots()
    
    for value in np.linspace(0.1, 0.5, 5):
        record = data[data[other_group] == value]
        x = range(record.rounds.values[0] + 1)
        ax.plot(
            x, 
            record[f"{group_letter}_isolation"].values[0], 
            color="tab:blue",
            label=f"Isolation with {other_group_title} = {round(value, 1)}",
            alpha=2*value
        )
    for value in np.linspace(0.1, 0.5, 5):
        record = data[data[other_group] == value]
        x = range(record.rounds.values[0] + 1)
        ax.plot(
            x, 
            record[f"{group_letter}_dissimilarity"].values[0], 
            color="tab:red",
            label=f"Dissimilarity with {other_group_title} = {round(value, 1)}",
            alpha=2*value
        )

    ax.set_ylim(0.2, 1.01)
    ax.set_xlim(0, 35)
    
    ax.legend()
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Segregation")
    ax.set_title(f"Segregation of Group {list(group.capitalize())[0]}")

    if savepath is not None:
        fig.savefig(
            os.path.join(savepath, f"changing_{other_group}_segregation.png"),
            facecolor="white",
            transparent=False
        )
        plt.close(fig)
        fig.clear()
    else:
        return fig

# %%

# plot_employer(employer, "alphas", "final_figs")
# plot_employer(employer, "betas", "final_figs")
# %%

def find_closest(value: float, a: np.ndarray) -> float:
    distance = np.abs(a - value)
    return a[np.argmin(distance)]

def plot_final_segregation(
    res: pd.DataFrame, 
    group: str = "alphas", 
    savepath: Optional[str] = None,
    **plot_kwargs
) -> plt.Figure:

    other_group = "alphas" if group == "betas" else "betas"
    other_group_title = other_group.replace("s", "").capitalize()
    fig, axes = plt.subplots(1, 2, figsize=(24, 12), tight_layout=True)

    group_letter = list(group)[0]
    other_group_letter = list(other_group)[0]
    
    res[f"Gamma {other_group_letter.capitalize()} Mean"] = res[f"gamma_{other_group_letter}_mean"]

    sns.scatterplot(
        res[f"gamma_{group_letter}_mean"], 
        res[f"{group_letter}_final_isolation"],
        hue=res[f"Gamma {other_group_letter.capitalize()} Mean"],
        ax=axes[0],
        label=f"Isolation of Group {group_letter.capitalize()}",
        **plot_kwargs
    )

    sns.scatterplot(
        res[f"gamma_{group_letter}_mean"], 
        res[f"{group_letter}_final_dissimilarity"],
        hue=res[f"Gamma {other_group_letter.capitalize()} Mean"],
        ax=axes[1],
        label=f"Dissimilarity of Group {group_letter.capitalize()}",
        **plot_kwargs
    )

    axes[0].set_ylim(0.3, 1.2)
    axes[1].set_ylim(0.3, 1)
    survey_mean = 0.4918 if group == "betas" else 0.5879
    group_prop = 244/(811 + 244) if group == "betas" else 811/(244 + 811)

    for ax in axes:
        ax.axvline(survey_mean, linestyle="dashed", color="tab:red")
        ax.axvline(group_prop, linestyle="dashed", color="black")
        ax.set_xlabel("")
        ax.set_ylabel("")
    
    axes[0].set_title(f"Final Isolation of Group {group_letter.capitalize()}")
    axes[1].set_title(f"Final Dissimilarity of Group {group_letter.capitalize()}")

    


    fig.supxlabel(f"Group {group_letter.capitalize()} Gamma")
    fig.supylabel("Final Segregation")


    if savepath is not None:
        fig.savefig(
            os.path.join(savepath, f"{group_letter}_final_segregation.png"),
            facecolor="white",
            transparent=False
        )
        plt.close(fig)
        fig.clear()
    else:
        return fig

plot_final_segregation(employee, "alphas", "final_figs", s=200)
plot_final_segregation(employee, "betas", "final_figs", s=200)
# %%