from graph import Graph, Edge, Node, EvacuateNode
from typing import Dict, List, Tuple, TypeVar, Union
import itertools
from numpy import random

UNKNOWN = None
StateT = TypeVar('State')
Action = Union['TERMINATE', EvacuateNode]
Ternary = Union[bool, UNKNOWN]


def rnd(frac):
    return round(frac, 4)


def d(v):
    return '?' if v is UNKNOWN else str(v)[0]


def zip2(keys, values, description=''):
    return description + '[' + ', '.join(['{}:{}'.format(k, d(v)) for k, v in zip(keys, values)]) + ']'


def get_all_options(domain, n):
    return itertools.product(domain, repeat=n)


class State:
    STATES: Dict[StateT, StateT] = {}
    G: Graph
    edge_to_index: Dict[Edge, int]
    node_to_index: Dict[Node, int]


    @staticmethod
    def register_state(state):
        State.STATES[state] = state

    @staticmethod
    def get_state(state):
        return State.STATES[state]

    @staticmethod
    def get_initial_state(loc):
        n_blockable = len(State.G.get_blockable_edges())
        n_saveable = len(State.G.get_evac_nodes())
        p = (False,) * n_saveable
        e = (UNKNOWN,) * n_blockable
        return State.get_state((loc, p, e, p, p, 0, False))

    @staticmethod
    def init_state_space(G: Graph):
        State.G = G
        evac_nodes = G.get_evac_nodes()
        blockable_edges = G.get_blockable_edges()
        State.node_to_index = {v: i for i, v in enumerate(evac_nodes)}
        State.edge_to_index = {e: i for i, e in enumerate(blockable_edges)}
        max_deadline = G.get_max_deadline() + 1
        n_evacuate_nodes = len(evac_nodes)
        n_blockable_edges = len(blockable_edges)
        # order matters. For each simulation tick in reverse order, first generate the states where the agent is terminated, then the ones where it is active
        #TODO: prune impossible states
        for t in reversed(range(max_deadline)):
            for term in [True, False]:
                for loc in G.get_vertices():
                    for evacuees_state in get_all_options([True, False], n_evacuate_nodes):
                        for block_states in get_all_options([True, False, UNKNOWN], n_blockable_edges):
                            for carrying_state in get_all_options([True, False], n_evacuate_nodes):
                                for saved_state in get_all_options([True, False], n_evacuate_nodes):
                                    s = State(loc, evacuees_state, block_states, carrying_state, saved_state, t, term)
                                    State.register_state(s)

    def reward(self):
        if not self.terminated:
            return 0
        total = 0
        for saved, v in zip(self.saved_state, self.G.get_evac_nodes()):
            if saved:
                total += v.n_people
        return total

    def is_reachable(self, v):
        e = self.G.get_edge(self.loc, v)
        return self.time + e.w <= v.deadline

    def is_reachable_shelter(self, v):
        return self.is_reachable(v) and v.is_shelter()

    def result(self, action: Action):
        if action == 'TERMINATE':
            return self.transitions['TERMINATE'][1]
        dest = action
        possible_results = self.transitions[dest]
        if len(possible_results) == 1:
            return possible_results[0][1]
        # randomize successor state with the given blockage distribution
        probabilities, results = zip(*possible_results)
        return random.choice(results, p=probabilities)

    def __init__(self,
                 loc: EvacuateNode,
                 evacuees_state:    Tuple[bool],
                 block_states:      Tuple[Ternary],
                 carrying_state:    Tuple[bool],
                 saved_state:       Tuple[bool],
                 time: int,
                 terminated: bool):
        self.loc = loc
        self.evacuees_state = evacuees_state
        self.block_states = block_states
        self.carrying_state = carrying_state
        self.saved_state = saved_state
        self.time = time
        self.terminated = terminated
        # a dict of actions and their possible result states and their probabilities for occurring
        self.transitions: Dict[Action, List[Tuple[float, StateT]]] = self.get_transitions()
        # find the best move from this state to maximize expected utility
        self.expected_utility = '?'
        self.best_option, self.expected_utility = self.get_best_action()

    def __iter__(self):
        yield self.loc
        yield self.evacuees_state
        yield self.block_states
        yield self.carrying_state
        yield self.saved_state
        yield self.time
        yield self.terminated

    def __eq__(self, other):
        return tuple(self) == tuple(other)

    def __repr__(self):
        blockable = State.G.get_blockable_edges()
        saveable = State.G.get_evac_nodes()

        return repr((self.loc,
                    zip2(saveable,  self.evacuees_state, 'P:'),
                    zip2(blockable, self.block_states,   'B:'),
                    zip2(saveable,  self.carrying_state, 'C:'),
                    zip2(saveable,  self.saved_state,    'S:'),
                    't=' + str(self.time),
                    'T:' + d(self.terminated),
                    'EU=' + str(rnd(self.expected_utility))))

    def __hash__(self):
        return hash(tuple(self))

    def get_transition_evacuees_state(self, v):
        current = self.evacuees_state
        # v can be evacuated
        if v in State.node_to_index:
            new_state = list(current)
            # traversing to v means we picked up the people in it (now or before)
            node_state_tuple_idx = State.node_to_index[v]
            new_state[node_state_tuple_idx] = True  # True: people in node were picked up
            return tuple(new_state)
        return current

    def get_transition_carrying_state(self, v):
        current = self.carrying_state
        # v can be evacuated
        if v in State.node_to_index:
            new_state = list(current)
            # traversing to v means we are carrying the people in it, unless previously saved or arriving at a shelter
            node_state_tuple_idx = State.node_to_index[v]
            previously_saved = self.saved_state[node_state_tuple_idx]
            new_state[node_state_tuple_idx] = (not self.is_reachable_shelter(v)) and (not previously_saved)  # True: carrying the people initially in node
            return tuple(new_state)
        return current

    def get_transition_saved_state(self, v: EvacuateNode):
        # a node is saved (True) if it was previously saved or if we arrive a shelter while carrying its people.
        if self.is_reachable_shelter(v):
            return tuple(saved or carrying for saved, carrying in zip(self.saved_state, self.carrying_state))
        return self.saved_state

    def path_unblocked(self, v):
        u = self.loc
        e: Edge = self.G.get_edge(u, v)
        idx = State.edge_to_index.get(e)
        if idx is None:
            # edge cannot be blocked
            return True
        edge_blocked = self.block_states[idx]
        if edge_blocked == UNKNOWN:
            # state cannot be reached, so it's a don'tcare
            # raise('block state undetermined before traversal!')
            return False
        return edge_blocked is False

    def get_transition_blocked_state(self, v: EvacuateNode):
        new_state_possibilities = []
        current = list(self.block_states)
        determined_indices = []  # which edges in the blocked state tuple are now determined after traversing to v.
        for e, idx in State.edge_to_index.items():
            if e.contains(v) and self.block_states[idx] == UNKNOWN:
                determined_indices.append((idx, e.block_chance))
        if len(determined_indices) == 0:
            # deterministic successor state - no change in knowledge about edge blockage
            return [(1, self.block_states)]
        # get probability of blocked/ unblocked state combinations for each road connected to v
        for possible_assignment in get_all_options([True,False], len(determined_indices)):
            prob = 1
            option = current[:]
            for edge_idx_p, edge_blocked in zip(determined_indices, possible_assignment):
                idx, p = edge_idx_p
                option[idx] = edge_blocked
                if edge_blocked:
                    prob *= p
                else:
                    prob *= (1-p)
            new_state_possibilities.append((prob, tuple(option)))
        return new_state_possibilities

    def terminated_transition(self):
        return State.get_state((self.loc, self.evacuees_state, self.block_states, self.carrying_state, self.saved_state, self.time, True))

    def get_transitions(self):
        transitions = {}
        if self.terminated:
            return transitions
        u = self.loc
        transitions['TERMINATE'] = [(1, self.terminated_transition())]
        # for every unblocked road to neighbour, add all the possible transitions and their probabilities
        for v in self.G.neighbours(u):
            if self.path_unblocked(v):
                if self.is_reachable(v):
                    e = self.G.get_edge(u, v)
                    transitions[v] = []
                    # get probability of blocked/ unblocked state combinations for each road connected to v
                    for prob, block_state_possibility in self.get_transition_blocked_state(v):
                        key = (v,
                               self.get_transition_evacuees_state(v),
                               block_state_possibility,
                               self.get_transition_carrying_state(v),
                               self.get_transition_saved_state(v),
                               self.time + e.w,
                               False)
                        result_state = State.get_state(key)
                        transitions[v].append((prob, result_state))
                else:
                    # if cannot reach destination before deadline, terminate.
                    transitions[v] = transitions['TERMINATE']
        return transitions

    def get_best_action(self):
        current_reward = self.reward()  # reward for the current state
        best_score = current_reward
        best_action = None
        for action, possible_outcomes in sorted(self.transitions.items()):  # prefer TERMINATE ops
            expected_action_utility = current_reward
            for p, state in possible_outcomes:
                expected_action_utility += p * state.expected_utility
            if best_score < expected_action_utility:
                best_score = expected_action_utility
                best_action = action
        # if repr(self) == "('V3', 'P:[V3:T, V4:F]', \"B:[('V3', 'V4', 3):?]\", 'C:[V3:F, V4:T]', 'S:[V3:F, V4:F]', 't=0', 'T:F', 'EU=?')":
        #     print('t')
        # print(self, best_action, best_score)
        return best_action, best_score

