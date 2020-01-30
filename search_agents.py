from graph import Graph, Edge, Node, EvacuateNode
from typing import Dict, List, Tuple, TypeVar, Union
import itertools
from numpy import random

UNKNOWN = None
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


class BeliefStateSpace:
    def __init__(self, G: Graph):
        self.G = G
        self.STATES: Dict[State, State] = {}
        self.evac_nodes = G.get_evac_nodes()
        self.blockable_edges = G.get_blockable_edges()
        n_evacuate_nodes = len(self.evac_nodes)
        n_blockable_edges = len(self.blockable_edges)
        self.node_to_index = {v: i for i, v in enumerate(self.evac_nodes)}
        self.edge_to_index = {e: i for i, e in enumerate(self.blockable_edges)}
        max_deadline = G.get_max_deadline() + 1
        # order matters. For each simulation tick in reverse order, first generate the states where the agent is terminated, then the ones where it is active
        #TODO: prune impossible states
        for t in reversed(range(max_deadline)):
            for term in [True, False]:
                for loc in G.get_vertices():
                    for evacuees_state in get_all_options([True, False], n_evacuate_nodes):
                        for block_states in get_all_options([True, False, UNKNOWN], n_blockable_edges):
                            for carrying_state in get_all_options([True, False], n_evacuate_nodes):
                                for saved_state in get_all_options([True, False], n_evacuate_nodes):
                                    s = State(self, loc, evacuees_state, block_states, carrying_state, saved_state, t, term)
                                    self.register_state(s)

    def register_state(self, state):
        self.STATES[state] = state

    def get_state(self, state):
        return self.STATES[state]

    def get_initial_state(self, loc):
        n_blockable = len(self.blockable_edges)
        n_saveable  = len(self.evac_nodes)
        p = (False,) * n_saveable
        e = (UNKNOWN,) * n_blockable
        return self.get_state((loc, p, e, p, p, 0, False))


class State:
    def __init__(self,
                 state_space: BeliefStateSpace,
                 loc: EvacuateNode,
                 evacuees_state:    Tuple[bool],
                 block_states:      Tuple[Ternary],
                 carrying_state:    Tuple[bool],
                 saved_state:       Tuple[bool],
                 time: int,
                 terminated: bool):
        self.state_space = state_space
        self.loc = loc
        self.evacuees_state = evacuees_state
        self.block_states = block_states
        self.carrying_state = carrying_state
        self.saved_state = saved_state
        self.time = time
        self.terminated = terminated

        # a dict of actions and their possible get_action_result states and their probabilities for occurring
        self.expected_utility = float('-0.1')
        self.transitions: Dict[Action, List[Tuple[float, State]]] = self.get_transitions()
        # find the best move from this state to maximize expected utility
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
        blockable = self.state_space.G.get_blockable_edges()
        saveable = self.state_space.G.get_evac_nodes()

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

    # Methods used to update the graph
    def get_total_num_people(self, sub_state):
        """get the total number of people picked up/carried/saved ( depending on the sub-state)"""
        total = 0
        for to_add, v in zip(sub_state, self.state_space.evac_nodes):
            if to_add:
                total += v.n_people_initial
        return total

    def get_total_saved(self):
        return self.get_total_num_people(self.saved_state)

    def get_total_carrying(self):
        return self.get_total_num_people(self.carrying_state)

    def update_graph(self):
        for v, i in self.state_space.node_to_index.items():
            v.n_people = 0 if self.evacuees_state[i] is True else v.n_people_initial
        for e, i in self.state_space.edge_to_index.items():
            e.blocked = self.block_states[i]
    #

    def get_action_result(self, action: Action):
        if action == 'TERMINATE':
            return self.transitions['TERMINATE'][0][1]
        dest = action
        possible_results = self.transitions[dest]
        if len(possible_results) == 1:
            return possible_results[0][1]
        # randomize successor state with the given blockage distribution
        probabilities, results = zip(*possible_results)
        return random.choice(results, p=probabilities)

    def is_reachable(self, v):
        e = self.state_space.G.get_edge(self.loc, v)
        return self.time + e.w <= v.deadline

    def is_reachable_shelter(self, v):
        return self.is_reachable(v) and v.is_shelter()

    def get_transition_evacuees_state(self, v):
        current = self.evacuees_state
        # v can be evacuated
        if v in self.state_space.node_to_index:
            new_state = list(current)
            # traversing to v means we picked up the people in it (now or before)
            node_state_tuple_idx = self.state_space.node_to_index[v]
            new_state[node_state_tuple_idx] = True  # True: people in node were picked up
            return tuple(new_state)
        return current

    def get_transition_carrying_state(self, v):
        current = self.carrying_state
        # arriving at a shelter drops off all currently carried evacuees
        if self.is_reachable_shelter(v):
            return (False,) * len(self.state_space.evac_nodes)
        # if v can be evacuated
        if v in self.state_space.node_to_index:
            new_state = list(current)
            # traversing to v means we are carrying the people in it, unless previously saved or arriving at a shelter
            node_state_tuple_idx = self.state_space.node_to_index[v]
            previously_saved = self.saved_state[node_state_tuple_idx]
            new_state[node_state_tuple_idx] = not previously_saved  # True: carrying the people initially in node
            return tuple(new_state)
        return current

    def get_transition_saved_state(self, v: EvacuateNode):
        # a node is saved (True) if it was previously saved or if we arrive a shelter while carrying its people.
        if self.is_reachable_shelter(v):
            return tuple(saved or carrying for saved, carrying in zip(self.saved_state, self.carrying_state))
        return self.saved_state

    def path_unblocked(self, v):
        u = self.loc
        e: Edge = self.state_space.G.get_edge(u, v)
        idx = self.state_space.edge_to_index.get(e)
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
        for e, idx in self.state_space.edge_to_index.items():
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
        return self.state_space.get_state((self.loc, self.evacuees_state, self.block_states, self.carrying_state, self.saved_state, self.time, True))

    def get_transitions(self):
        transitions = {}
        if self.terminated:
            return transitions
        u = self.loc
        transitions['TERMINATE'] = [(1, self.terminated_transition())]
        # for every unblocked road to neighbour, add all the possible transitions and their probabilities
        for v in self.state_space.G.neighbours(u):
            if self.path_unblocked(v):
                if self.is_reachable(v):
                    e = self.state_space.G.get_edge(u, v)
                    transitions[v] = []
                    new_evacuees_state = self.get_transition_evacuees_state(v)

                    if repr(self) == "('V5', 'P:[V3:T, V4:T]', 'B:[E3:F]', 'C:[V3:F, V4:F]', 'S:[V3:T, V4:T]', 't=7', 'T:F', 'EU=-0.1')":
                        print(d(False))
                    new_carrying_state = self.get_transition_carrying_state(v)
                    new_saved_state = self.get_transition_saved_state(v)
                    # get probability of blocked/ unblocked state combinations for each road connected to v
                    for prob, block_state_possibility in self.get_transition_blocked_state(v):
                        key = (v,
                               new_evacuees_state,
                               block_state_possibility,
                               new_carrying_state,
                               new_saved_state,
                               self.time + e.w,
                               False)
                        result_state = self.state_space.get_state(key)
                        transitions[v].append((prob, result_state))
                else:
                    # if cannot reach destination before deadline, terminate.
                    transitions[v] = transitions['TERMINATE']
        return transitions

    def reward(self):
        if not self.terminated:
            return 0
        return self.get_total_saved()

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
        return best_action, best_score

