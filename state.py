from graph import Graph, Edge, EvacuateNode
from typing import Dict, List, Tuple, Union
from numpy import random
import itertools
import tree

CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
base = len(CHARS)
TERMINATE = None
UNKNOWN = None
DECIMALS = 4
Action = Union[EvacuateNode, TERMINATE]
Ternary = Union[bool, UNKNOWN]

# Auxiliary methods #


def to_base(n, base):
    return "0" if not n else to_base(n // base, base).lstrip("0") + CHARS[n % base]


def ID(v):
    """appends a unique ID to string - used to generate unique tree nodes"""
    try:
        ID.ID += 1
    except AttributeError:
        ID.ID = 1
    return '{}|{}'.format(v, to_base(ID.ID, base))


def vectorize(func):
    """allows a function to operate on a single element or a list/tuple"""
    def each(f, args):
        return [f(arg) for arg in args]

    def inner(*args):
        if type(*args) in [list, tuple]:
            output = each(func, *args)
        else:
            output = func(*args)
        return output
    return inner


@vectorize
def d(v):
    """shorthand for boolean and UNKKNOWN: True=T,False=F,UNKNOWN=?"""
    return '?' if v is None else str(v)[0]


def zip2(keys, values, description=''):
    """merges 2 lists to a string with an optional description"""
    return description + '[' + ', '.join(['{}:{}'.format(k, v) for k, v in zip(keys, d(values))]) + ']'


def get_all_options(domain, n):
    """all combinations of {domain items}^n"""
    return itertools.product(domain, repeat=n)


class State:
    def __init__(self,
                 state_space,
                 loc: EvacuateNode,
                 pickup_state:    Tuple[bool],
                 block_states:      Tuple[Ternary],
                 carrying_state:    Tuple[bool],
                 saved_state:       Tuple[bool],
                 time: int,
                 terminated: bool):
        self.state_space: BeliefStateSpace = state_space  # containing belief state space
        self.loc = loc
        self.pickup_state = pickup_state
        self.block_states = block_states
        self.carrying_state = carrying_state
        self.saved_state = saved_state
        self.time = time
        self.terminated = terminated

        # a dict of actions and their possible result states, with their probabilities for occurring
        self.transitions: Dict[Action, List[Tuple[float, State]]] = self.get_transitions()
        # find the best move from this state to maximize expected utility
        self.best_option, self.expected_utility = self.get_best_action()

    def __iter__(self):
        """used to represent each state as a tuple"""
        yield self.loc
        yield self.pickup_state
        yield self.block_states
        yield self.carrying_state
        yield self.saved_state
        yield self.time
        yield self.terminated

    def __eq__(self, other):
        return tuple(self) == tuple(other)

    def __repr__(self):
        blockable = self.state_space.blockable_edges
        saveable = self.state_space.evac_nodes

        return repr((self.loc,
                    zip2(saveable,  self.pickup_state,   'P:'),
                    zip2(blockable, self.block_states,   'B:'),
                    zip2(saveable,  self.carrying_state, 'C:'),
                    zip2(saveable,  self.saved_state,    'S:'),
                    't=' + str(self.time),
                    'T:' + d(self.terminated),
                    'EU=' + str(round(self.expected_utility, DECIMALS))))

    def __hash__(self):
        return hash(tuple(self))

    def compact(self):
        """state representation for policy tree"""
        return '({})'.format(','.join(str(e) for e in
                    [self.loc,
                    d(self.pickup_state),
                    d(self.block_states),
                    d(self.carrying_state),
                    d(self.saved_state),
                    self.time,
                    d(self.terminated),
                    str(round(self.expected_utility, 2))]).replace(' ', ''))

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
            v.n_people = 0 if self.pickup_state[i] is True else v.n_people_initial
        for e, i in self.state_space.edge_to_index.items():
            e.blocked = self.block_states[i]
    #

    def get_action_result(self, action: Action):
        """produces a real world state after taking the action"""
        if action is TERMINATE:
            return self.transitions[TERMINATE][0][1]
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

    def get_transition_pickup_state(self, v):
        current = self.pickup_state
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
            return tuple((saved or carrying) for saved, carrying in zip(self.saved_state, self.carrying_state))
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
            # state cannot be reached, so it's a don't-care
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
        for possible_assignment in get_all_options([True, False], len(determined_indices)):
            prob = 1
            option = current[:]
            # calculate the overall probability of a specific outcome by multiplying blocked/unblocked probabilities
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
        return self.state_space.get_state((self.loc, self.pickup_state, self.block_states, self.carrying_state, self.saved_state, self.time, True))

    def get_transitions(self):
        transitions = {}
        if self.terminated:
            return transitions
        u = self.loc
        transitions[TERMINATE] = [(1, self.terminated_transition())]
        # for every unblocked road to neighbour, add all the possible transitions and their probabilities
        for v in self.state_space.G.neighbours(u):
            if self.path_unblocked(v):
                if self.is_reachable(v):
                    e = self.state_space.G.get_edge(u, v)
                    transitions[v] = []
                    new_pickup_state = self.get_transition_pickup_state(v)
                    new_carrying_state = self.get_transition_carrying_state(v)
                    new_saved_state = self.get_transition_saved_state(v)
                    # get probability of blocked/ unblocked state combinations for each road connected to v
                    for prob, block_state_possibility in self.get_transition_blocked_state(v):
                        key = (v,
                               new_pickup_state,
                               block_state_possibility,
                               new_carrying_state,
                               new_saved_state,
                               self.time + e.w,
                               False)
                        result_state = self.state_space.get_state(key)
                        transitions[v].append((prob, result_state))
                else:
                    # if cannot reach destination before deadline, terminate.
                    transitions[v] = transitions[TERMINATE]
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

    def describe(self):
        print(self)
        for action, results in self.transitions.items():
            action_expected_util = sum([p * st.expected_utility for p, st in results])
            print('\tEU[{}]={:.2f}'.format(action or 'TERMINATE', action_expected_util))


class BeliefStateSpace:
    """This class represents an agent's belief-state space and is used to devise and follow a policy
    given a physical initial state i.e. graph"""

    def __init__(self, G: Graph):
        self.G = G
        self.STATES: Dict[State, State] = {}
        self.evac_nodes = G.get_evac_nodes()
        self.blockable_edges = G.get_blockable_edges()
        self.node_to_index = {v: i for i, v in enumerate(self.evac_nodes)}
        self.edge_to_index = {e: i for i, e in enumerate(self.blockable_edges)}

        n_evacuate_nodes = len(self.evac_nodes)
        n_blockable_edges = len(self.blockable_edges)
        max_deadline = self.G.get_max_deadline() + 1

        # Creating all the states in the State Space: order matters!
        # For each time unit up to max deadline (in REVERSE order):
        # first generate the states where the agent is terminated, then the ones where it is active.
        # This is similar to dynamic programming cache - the transitions from each state result in states
        # that have already been created, with their expected utility calculated

        def is_legal_state(pickup_state, carrying_state, saved_state, block_states):
            illegal_false_pickup = any(
                [((not pu) and (c or s)) for pu, c, s in zip(pickup_state, carrying_state, saved_state)])
            illegal_unknown_edge = any([(e.contains(loc) and (e_blocked is UNKNOWN)) for e, e_blocked in
                                        zip(self.blockable_edges, block_states)])
            return not (illegal_false_pickup or illegal_unknown_edge)

        for t in reversed(range(max_deadline)):  # for each time unit in reverse order
            for term in [True, False]:  # first terminated states, then active states
                for loc in G.get_vertices():
                    for pickup_state in get_all_options([True, False], n_evacuate_nodes):
                        for block_states in get_all_options([True, False, UNKNOWN], n_blockable_edges):
                            for carrying_state in get_all_options([True, False], n_evacuate_nodes):
                                for saved_state in get_all_options([True, False], n_evacuate_nodes):
                                    # skip illegal states
                                    if is_legal_state(pickup_state, carrying_state, saved_state, block_states):
                                        s = State(self, loc, pickup_state, block_states, carrying_state, saved_state, t,
                                                  term)
                                        self.register_state(s)

    def register_state(self, state):
        self.STATES[state] = state

    def get_state(self, state):
        return self.STATES[state]

    def get_initial_state(self, loc):
        n_blockable = len(self.blockable_edges)
        n_saveable = len(self.evac_nodes)
        p = (False,) * n_saveable
        e = (UNKNOWN,) * n_blockable
        return self.get_state((loc, p, e, p, p, 0, False))

    def display_policy(self, initial_state: State):
        V, E = [], []
        root = ID(initial_state.compact())
        Q = [(initial_state, root)]
        L = {}
        B = set([])
        while len(Q) > 0:
            state, state_node = Q.pop()
            for action, results in state.transitions.items():
                action_expected_util = sum([p * st.expected_utility for p, st in results])
                a = ID((action or 'T', round(action_expected_util, 2)))
                V.append(a)
                E.append((state_node, a))
                if action == state.best_option:
                    B.add(a)
                else:  # comment this else block to view all reachable states
                    continue
                for p, result in results:
                    s = ID(result.compact())
                    Q.append((result, s))
                    V.append(s)
                    e = (a, s)
                    E.append(e)
                    L[e] = round(p, 2)
        tree.display_tree(root, V, E, L, B)
