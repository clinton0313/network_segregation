#%%
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import warnings

from tqdm import tqdm
from typing import Any, Sequence

class Agent():
    def __init__(self, id: int, gamma: float, group: int, firm = None):
        self.id = id
        self.gamma = gamma
        self.group = group
        self.firm = firm
    
    def wants_to_leave(self) -> bool:
        if self.firm is None:
            return True
        elif self.firm.employee_fraction(self.group) < self.gamma:
            return True
        else:
            return False

    def choose_firm(self, firms: list):
        #How to tackle no firms to move to
        candidate_firms = [
            firm 
            for firm in firms
            if firm.capacity < firm.total_agents()
            and firm.employee_fraction(self.group) >= self.gamma
        ]
        if len(candidate_firms) == 0:
            return self.firm
        else:
            return random.choice(candidate_firms)

class Firm():
    def __init__(self, id: int, alpha: float, beta: float, capacity: int):
        self.id = id
        self.alpha = alpha
        self.beta = beta
        self.capacity = capacity
        self.agents = []
    
    def add_agent(self, agent: Agent):
        if len(self.agents) < self.capacity:
            self.agents.append(agent)
        else:
            raise RuntimeError(f"Firm was full and could not add employee.")

    def hire(self, agent: Agent):
        if (
            (agent.group == 0 and np.random.uniform() < self.alpha)
            or (agent.group == 1 and np.random.uniform() < self.beta)
        ):
            self.add_agent(agent)
            agent.firm = self

    def total_agents(self) -> int:
        return len(self.agents)
    
    def group_count(self, group: int) -> float:
        agent_count = 0
        for agent in self.agents:
            if agent.group == group:
                agent_count += 0
        return agent_count

    def group_fraction(self, group: int) -> float:
        agent_count = self.group_count(group)
        group_fraction =  agent_count / len(self.agents)
        return group_fraction

class Environment():
    def __init__(
        self,
        num_firms: int,
        capacities: Sequence,
        alphas: Sequence,
        betas: Sequence,
        num_group_a: int,
        num_group_b: int,
        gammas_a: Sequence,
        gammas_b: Sequence,
    ):

        self.num_firms = num_firms
        self.num_group_a = num_group_a
        self.num_group_b = num_group_b
        
        self.capacities = self._aslist(capacities, self.num_firms)
        self.alphas = self._aslist(alphas, self.num_firms)
        self.betas = self._aslist(betas, self.num_firms)
        self.gammas_a = self._aslist(gammas_a, self.num_group_a)
        self.gammas_b = self._aslist(gammas_b, self.num_group_b)


        self.ids = list(range(self.num_firms + self.num_group_a + self.num_group_b))
        self.firms = [
            Firm(id, self.alphas[i], self.betas[i], self.capacities[i]) 
            for id, i in zip(self.ids[:self.num_firms], range(self.num_firms))
        ]
        self.agents = self._get_agents(self.gammas_a, self.gammas_b)
        self.fig = None
        self.graph = None

    def _aslist(self, x: Any, n: int) -> list:
        if isinstance(x, int) or isinstance(x, float):
            return [x for _ in range(n)]
        else:
            return x

    def _get_agents(self, 
        gammas_a: Sequence,
        gammas_b: Sequence
    ) -> tuple:

        agents = [
            Agent(id, gammas_a[i], group = 0) 
            for id, i in zip(
                self.ids[self.num_firms:self.num_group_a + self.num_firms], 
                range(self.num_group_a)
            )
        ]
        agents.extend(
            [
                Agent(id, gammas_b[i], group = 1) 
                for id, i in zip(
                    self.ids[self.num_group_a + self.num_firms:],
                    range(self.num_group_b)
                )
            ]
        )
        return tuple(agents)

    def get_player(self, id):
        if id < 0 or id >= len(self.agents) + len(self.firms):
            raise ValueError(f"There are only {len(self.agents) + len(self.firms)} players in the game. Tried to get id {id}")
        if id < len(self.firms):
            return self.firms[id]
        else:
            return self.agents[id - len(self.firms)]

    def segregation(self, firm_id: int, group: int = 1, type: str = "isolation") -> float:
        
        if group != 0 and group != 1:
            raise ValueError(f"Can only use group 0 or 1, got {group}")
            

        if firm_id < self.num_firms:
            group_total = self.num_group_a if group == 0 else self.num_group_b
            group_count = self.firms[firm_id].group_count(group)
            if self.firms[firm_id].total_agents() == 0:
                warnings.warn(
                    f"Firm {firm_id} has 0 agents. Segregation is set to 0 for this firm", 
                    RuntimeWarning
                )
                segregation = 0
            elif type == "isolation":
                segregation = (
                    group_count ** 2
                    / group_total
                    / len(self.agents)
                )
            elif type == "dissimilarity":
                segregation = (
                    1/2 * (
                            group_count/group_total
                            - (self.firms[firm_id].total_agents() - group_count) / (len(self.agents) - group_total)
                        )
                )
            else:
                raise ValueError(f"Can only calculate isolation or dissimilarity, got {type}")
        
        else:
            raise ValueError(f"Firm {firm_id} is not a firm between 0 and {len(self.firms)}")
        
        return segregation

    def total_segregation(self, group: int = 1, type: str = "isolation") -> float:
        
        total_segregation = [self.segregation(firm, group, type) for firm in range(self.num_firms)]
        return sum(total_segregation)

    def assign_firms(self, assignments: dict):

        for i, k in assignments.items():
            self.agents[i].firm = self.firms[k]
            self.firms[k].add_agent(self.agents[i])

    def simulate(self, rounds: int, shuffle: bool = True, verbose: bool = False):
        
        if shuffle:
            indices = np.random.permutation(len(self.agents))
        else:
            indices = list(range(len(self.agents)))

        for _ in tqdm(range(rounds), disable = not verbose):
            for i in indices:
                if self.agents[i].wants_to_leave():
                    new_firm = self.agents[i].choose_firm(self.firms)
                    
                    if isinstance(new_firm, Firm):
                        new_firm.hire(self.agents[i])
    
    def get_graph(self, **segregation_kwargs) -> nx.Graph:

        g = nx.Graph()
        for firm in self.firms:
            if len(firm.agents) == 0:
                g.add_node(firm.id)
        for agent in self.agents:
            if agent.firm is not None:
                g.add_edge((agent.id, agent.firm.id))
            else:
                g.add_node(agent.id)

        
        self.graph = g
        self.update_graph_attributes(**segregation_kwargs)

        return self.graph

    def update_graph_attributes(self, **segregation_kwargs):

        for node in self.graph.nodes():
            player = self.get_player(node)
            if isinstance(player, Agent):
                node_attr = {
                    "type": "agent",
                    "group": player.group,
                    "gamma": player.gamma,
                }
            elif isinstance(player, Firm):
                node_attr = {
                    "type": "firm",
                    "alpha": player.alpha,
                    "beta": player.beta,
                    "segregation": self.segregation(node, **segregation_kwargs)
                }
            nx.set_node_attributes(self.graph, {node: node_attr})

    def plot_network(
        self, 
        new_fig: bool = True, 
        firm_color: str = "black",
        a_color: str = "tab:blue",
        b_color: str = "tab:red",
        firm_attrs: bool = True,
        agent_attrs: bool = True,
        **plot_kwargs
    ):
        
        if new_fig:
            self.fig, self.ax = plt.subplots(figsize = (12, 12))
        
        if self.graph is None:
            self.get_graph()
        
        nx.draw_kamada_kawai(self.graph, ax=self.ax)

        return self.fig

K = 20
c = 20

env = Environment(
    num_firms = K,
    capacities = c,
    alphas = [0.4 for _ in range(int(K/2))] + [0.9 for _ in range(int(K/2))],
    betas = 1,
    num_group_a = 80,
    num_group_b = 20,
    gammas_a = 0.2,
    gammas_b = 0.5,
)

#%%
env.plot_network()
env.fig

#%%
env.simulate(100)
env.plot_network()
env.fig
# %%
