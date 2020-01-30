from graph import Node, Edge, Graph, EvacuateNode
from typing import List, Set, TypeVar
from copy import copy as shallow_copy
AgentType = TypeVar('Agent')


class SmartGraph(Graph):
    """A variation of a graph that accounts for edge and node deadlines when running dijkstra"""

    def __init__(self, V: List[Node]=[], E: List[Edge]=[], env=None):
        """:param env: the enclosing environment in which the graph "lives". Used to access the environment's time."""
        super().__init__(V, E)
        self.env = env

    def is_blocked(self, u, v):
        e = self.get_edge(u, v)
        return e.blocked or self.env.time + e.w > e.deadline


class State:
    def __init__(self,
                 agent: AgentType,
                 agent_state: AgentType,
                 require_evac_nodes: Set[EvacuateNode],
                 blocked_edges: Set[Edge]):
        """creates a new state. Inherits env and agent data, unless overwritten"""
        self.agent = agent
        self.agent_state = agent_state
        self.require_evac_nodes = require_evac_nodes
        self.blocked_edges = blocked_edges

    def is_goal(self):
        return self.agent_state.terminated

    def describe(self):
        print("State: [{:<30}Evac:{}|Blocked:{}]"
              .format(self.agent.summary(), self.require_evac_nodes, self.blocked_edges))


class Environment:
    def __init__(self, G):
        self.time = 0
        self.G: SmartGraph = G
        self.agents: List[AgentType] = []
        self.require_evac_nodes: Set[EvacuateNode] = self.init_required_evac_nodes()
        self.blocked_edges: Set[Edge] = set([])
        self.agent_actions = {}

    def tick(self):
        self.time += 1
        self.execute_agent_actions()

    def all_terminated(self):
        return all([agent.terminated for agent in self.agents])

    def init_required_evac_nodes(self):
        return set([v for v in self.G.get_vertices() if (not v.is_shelter() and v.n_people > 0)])

    def get_blocked_edges(self):
        return set([e for e in self.G.get_edges() if self.G.is_blocked(e.v1, e.v2)]) # shallow_copy(self.blocked_edges)

    def get_require_evac_nodes(self):
        return shallow_copy(self.require_evac_nodes)

    def get_state(self, agent: AgentType):
        return State(
            agent,
            agent.get_agent_state(),
            self.get_require_evac_nodes(),
            self.get_blocked_edges()
        )
