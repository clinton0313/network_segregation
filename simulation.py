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
    """Single agent of the simulation"""
    def __init__(self, id: int, gamma: float, group: int, firm = None):
        """Agents of the simulation.

        :param id: Unique node id. 
        :param gamma: Tolerance threshold for minimum fraction of similar colleagues
        :param group: Group type agent is a part of. (A) 0 for minoirty, (B) 1 for majority. 
        :param firm: Firm instance that this agent is a part of, defaults to None
        """
        self.id = id
        self.gamma = gamma
        self.group = group
        self.firm = firm
    
    def wants_to_leave(self) -> bool:
        """Whether agent wants to leave current firm or not."""
        if self.firm is None:
            return True
        elif self.firm.group_fraction(self.group) < self.gamma:
            return True
        else:
            return False

    def choose_firm(self, firms: list):
        """Of a list of potential Firm instances, pick one to move to.
        If no available firms matching criteria, returns None."""
        #How to tackle no firms to move to
        candidate_firms = [
            firm 
            for firm in firms
            if firm.capacity > firm.total_agents()
            and firm.group_fraction(self.group) >= self.gamma
        ]
        if len(candidate_firms) == 0:
            return None
        else:
            return random.choice(candidate_firms)

class Firm():
    """Class of firms in the simulation"""
    def __init__(self, id: int, alpha: float, beta: float, capacity: int):
        """Firm class.

        :param id: Unique node id of the graph.
        :param alpha: Probability of hiring agent of group A (0), the minority class
        :param beta: Probability of hiring agent of group B (1), the majority class
        :param capacity: Maximum capacity of the firm. 
        """
        self.id = id
        self.alpha = alpha
        self.beta = beta
        self.capacity = capacity
        self.agents = []
    
    def add_agent(self, agent: Agent):
        """Adds an agent to the firm."""
        if len(self.agents) < self.capacity:
            self.agents.append(agent)
        else:
            raise RuntimeError(f"Firm was full and could not add employee.")

    def hire(self, agent: Agent):
        """Hires an agent based on alpha and beta probabilities."""
        if (
            (agent.group == 0 and np.random.uniform() < self.alpha)
            or (agent.group == 1 and np.random.uniform() < self.beta)
        ):
            self.add_agent(agent)
            agent.firm = self

    def total_agents(self) -> int:
        return len(self.agents)
    
    def group_count(self, group: int) -> float:
        """Number of agents at this firm in the `group` type"""
        agent_count = 0
        for agent in self.agents:
            if agent.group == group:
                agent_count += 1
        return agent_count

    def group_fraction(self, group: int) -> float:
        """Fraction of agents at this firm that are of group type `group`"""
        if len(self.agents) == 0:
            return 1
        else:
            agent_count = self.group_count(group)
            group_fraction =  agent_count / len(self.agents)
            return group_fraction

class Environment():
    """Simulation environment.
    Attributes:
        ids: Node ids of the firms, group A agents, group B agents in that order.
        firms: list of fhe firm instances. 
        agents: list of the agent instances.
        fig: Currently plotted figure.
        graph: Currently instantiated graph. 
    """
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
        """Simulation Environment Class. 

        :param num_firms: Total number of firms. 
        :param capacities: Capacities of the firms either as a single int or list
        :param alphas: Alpha probabilities of the firms. Single float or list. 
        :param betas: Beta probabilities of the firms. Single Float or list. 
        :param num_group_a: Number of agents in group A. 
        :param num_group_b: Number of agents in group B. 
        :param gammas_a: Float or list of gammas for group A. 
        :param gammas_b: Float or list of gammas for group B. 
        """

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
        self.ax = None
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
        """Gets the class instance with the given id."""
        if id < 0 or id >= len(self.agents) + len(self.firms):
            raise ValueError(f"There are only {len(self.agents) + len(self.firms)} players in the game. Tried to get id {id}")
        if id < len(self.firms):
            return self.firms[id]
        else:
            return self.agents[id - len(self.firms)]

    def segregation(self, firm_id: int, group: int = 1, type: str = "isolation") -> float:
        """Computes the segregation measure.

        :param firm_id: Firm id to compute the measure for.
        :param group: Group to compute segregation for, defaults to 1
        :param type: Type of segregation to compute. Either 'isolation' or
            'dissimilarity', defaults to "isolation"
        :return: Segregation measure. 
        """
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
        """Gets the total segregation measure for a group in the current environment"""
        total_segregation = [self.segregation(firm, group, type) for firm in range(self.num_firms)]
        return sum(total_segregation)

    def assign_firms(self, assignments: dict):
        """Assigns agents to firms arbitrarily using a dictionary of agent: firm assignments."""
        for i, k in assignments.items():
            self.agents[i].firm = self.firms[k]
            self.firms[k].add_agent(self.agents[i])

    def simulate(self, rounds: int, shuffle: bool = True, verbose: bool = False):
        """Run the main simulation.

        :param rounds: Number of rounds to run for. 
        :param shuffle: Shuffle the order of agents turns every round, defaults to True
        :param verbose: Shows progress bar if true., defaults to False
        """

        for _ in tqdm(range(rounds), disable = not verbose):
            if shuffle:
                indices = np.random.permutation(len(self.agents))
            else:
                indices = list(range(len(self.agents)))
            for i in indices:
                if self.agents[i].wants_to_leave():
                    new_firm = self.agents[i].choose_firm(self.firms)
                    
                    if isinstance(new_firm, Firm):
                        new_firm.hire(self.agents[i])
    
    def get_graph(self, **segregation_kwargs) -> nx.Graph:
        """Gets the underlying graph and saves as a class attribute."""
        g = nx.Graph()
        for firm in self.firms:
            if len(firm.agents) == 0:
                g.add_node(firm.id)
        for agent in self.agents:
            if agent.firm is not None:
                g.add_edge(agent.id, agent.firm.id)
            else:
                g.add_node(agent.id)

        
        self.graph = g
        self.update_graph_attributes(**segregation_kwargs)

        return self.graph

    def update_graph_attributes(self, **segregation_kwargs):
        """Updates the current class' attributes with node attributes. """
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
        bipartite: bool = False,
        label_y_offset: float = 0.1,
        label_x_offset: float = 0,
        **plot_kwargs
    ):
    
        if new_fig:
            self.fig, self.ax = plt.subplots(figsize = (12, 12))
        

        self.get_graph()
        
        node_colors = []
        for node in self.graph.nodes:
            if node < self.num_firms:
                node_colors.append(firm_color)
            elif node < self.num_firms + self.num_group_a:
                node_colors.append(a_color)
            else:
                node_colors.append(b_color)

        if bipartite:
            pos = nx.drawing.bipartite_layout(
                self.graph, 
                range(self.num_firms), 
                align="horizontal"
            )
        else:
            pos = nx.drawing.spring_layout(self.graph)


        for firm in self.firms:
            x, y = pos[firm.id]
            plt.text(
                x + label_x_offset, 
                y + label_y_offset, 
                s = f'{self.segregation(firm.id):2g}'
            )

        nx.draw(
            self.graph, 
            pos=pos, 
            ax=self.ax, 
            node_color= node_colors, 
            **plot_kwargs
        )

        plt.text(
            0.8,
            0.05,
            s = (
                f"Total Segregation\n"
                f"Group A: {self.total_segregation(0):2g}\n"
                f"Group B: {self.total_segregation(1):2g}"
            ),
            fontsize=12,
            transform=self.ax.transAxes
        )

        return self.fig

K = 20
c = 20

env = Environment(
    num_firms = K,
    capacities = c,
    alphas = [0.8 for _ in range(int(K/2))] + [0.9 for _ in range(int(K/2))],
    betas = 1,
    num_group_a = 80,
    num_group_b = 20,
    gammas_a = 0.2,
    gammas_b = 0.5,
)


#%%
env.simulate(100)
env.plot_network()
env.fig
# %%
