import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


class Node:
    """A base Node class for nodes used in the Graph class"""

    def __init__(self, label):
        self.label = label

    def __str__(self):
        return self.label

    def __eq__(self, other):
        return str(self) == str(other)

    def __lt__(self, other):
        return str(self) < str(other)

    def __hash__(self):
        return hash(self.label)

    def __repr__(self):
        return repr(self.label)


class Edge:
    def __init__(self, v1: Node, v2: Node, w=1, label='', block_chance=0.0):
        """edge created lexicographically by its vertices names"""
        if v2.label < v1.label:
            v1, v2 = v2, v1
        self.label = label
        self.v1 = v1
        self.v2 = v2
        self.w = w
        self.blocked = None
        self.block_chance = block_chance

    def get(self):
        return self.v1, self.v2, self.w

    def contains(self, v):
        return v in [self.v1, self.v2]

    def __eq__(self, other):
        return self.v1 == other.v1 and self.v2 == other.v2

    def __lt__(self, other):
        return self.label < other.label
        # return (self.v1, self.v2) < (other.v1, other.v2)

    def __str__(self):
        return self.label
        # return str((self.v1.label, self.v2.label, self.w))

    def __repr__(self):
        return '({},{})'.format(self.v1.label, self.v2.label)

    def __hash__(self):
        return hash((self.v1, self.v2))


class Graph:
    """Graph with blockable edges"""

    def __init__(self, V: List[Node]=[], E: List[Edge]=[]):
        self.pos = None  # used to maintain vertices position in visualization
        self.n_vertices = 0
        self.V: Dict[Node, List[Node]] = {}
        self.Adj: Dict[Tuple[Node, Node], Edge] = {}
        self.init(V, E)

    def init(self, V: List[Node], E: List[Edge]):
        """initialize graph with list of nodes and a list of edges"""
        for v in V: self.add_vertex(v)
        for e in E: self.add_edge(e)

    def add_vertex(self, v):
        if v in self.V:
            raise Exception("{} already exists in V".format(v))
        self.V[v] = set([])
        self.n_vertices += 1

    def get_edge(self, v1, v2):
        return self.Adj.get((v1, v2))

    def edge_exists_check(self, v1, v2, expected: bool):
        if (v1 not in self.V) or (v2 not in self.V):
            raise Exception("{} or {} are not in V".format(v1, v2))
        edge_exists = self.get_edge(v1, v2)
        if edge_exists and not expected:
            raise Exception("({},{}) already exists in E".format(v1, v2))
        elif not edge_exists and expected:
            raise Exception("({},{}) doesn't exist in E".format(v1, v2))

    def add_edge(self, e: Edge):
        v1 = e.v1
        v2 = e.v2
        self.edge_exists_check(v1, v2, expected=False)
        self.V[v1].add(v2)
        self.V[v2].add(v1)
        self.Adj[v1, v2] = e
        self.Adj[v2, v1] = e

    def neighbours(self, u):
        return self.V[u]

    def get_vertices(self):
        return self.V.keys()

    def get_edges(self):
        return list(set(self.Adj.values()))

    def display(self, graph_id=0):
        V = self.get_vertices()
        G = nx.Graph()
        G.add_nodes_from(V)
        G.add_weighted_edges_from([e.get() for e in self.Adj.values() if not e.blocked])
        node_labels = {v: v.describe() for v in G.nodes()}
        if G.number_of_nodes() == 0:
            return
        if self.pos is None:
            # save node position to maintain the same graph layout throughout simulations
            self.pos = nx.spring_layout(G, scale=25)
        edge_labels = {}
        for k, e in self.Adj.items():
            block_chance = '({})'.format(e.block_chance) if e.block_chance > 0 else ''
            block_state = ''
            if e.block_chance > 0:
                block_state = '[Y]' if e.blocked else '[N]' if e.blocked is False else '[?]'
            edge_labels[k] = '{},w={}'.format(e.label, e.w) + block_chance + block_state

        nx.draw_networkx_edge_labels(G, self.pos, edge_labels=edge_labels, rotate=False, font_size=6)

        nx.draw(G, self.pos, node_size=1700, with_labels=False)
        nx.draw_networkx_labels(G, self.pos, node_labels, font_size=7.5, font_weight='bold')
        plt.margins(0.2)
        plt.legend([], title=graph_id, loc='upper center')
        plt.show()

    def get_evac_nodes(self):
        return sorted([v for v in self.get_vertices() if v.n_people_initial > 0])

    def get_blockable_edges(self):
        return sorted([e for e in self.get_edges() if e.block_chance > 0])

    def get_max_deadline(self):
        return max([v.deadline for v in self.get_vertices()])



class EvacuateNode(Node):
    """Represents a node with people that are waiting for evacuation"""
    def __init__(self, label, deadline: int, n_people=0):
        super().__init__(label)
        self.deadline = deadline
        self.n_people = n_people
        self.n_people_initial = n_people
        self.evacuated = (n_people == 0)
        self.agents = set([])

    def is_shelter(self):
        return False

    def summary(self):
        return '{}\n(D{}|P{}/{})'.format(self.label, self.deadline, self.n_people, self.n_people_initial)

    def describe(self):
        return self.summary() + '\n' + '\n'.join([agent.summary() for agent in self.agents])


class ShelterNode(EvacuateNode):
    """Represents a node with a shelter"""
    def is_shelter(self):
        return True

    def summary(self):
        return '{}\n(D{})'.format(self.label, self.deadline)

    def describe(self):
        return 'Shelter\n' + super().describe()
