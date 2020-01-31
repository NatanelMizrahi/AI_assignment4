from graph import EvacuateNode, Graph
from state import BeliefStateSpace, State


class Agent:
    def __init__(self, name, start_loc: EvacuateNode, G: Graph):
        self.loc: EvacuateNode = start_loc
        self.loc.agents.add(self)
        self.name = name
        self.n_saved = 0
        self.n_carrying = 0
        self.terminated = False
        self.time = 0
        self.G = G
        self.state: State = None
        self.state_space: BeliefStateSpace = None
        print("Agent {} created in {}".format(self.name, start_loc))

    def apply_state(self):
        self.terminated = self.state.terminated
        self.time = self.state.time
        self.n_carrying = self.state.get_total_carrying()
        self.n_saved = self.state.get_total_saved()
        self.loc.agents.remove(self)
        self.loc = self.state.loc
        self.loc.agents.add(self)
        self.state.update_graph()

    def create_policy(self):
        self.state_space = BeliefStateSpace(self.G)
        self.state = self.state_space.get_initial_state(self.loc)
        self.state_space.display_policy(self.state)


    def apply_policy(self):
        while self.state.best_option is not None:
            best_action = self.state.best_option
            self.state = self.state.get_action_result(best_action)
            print(self.state)
            self.apply_state()
            self.G.display('T={}'.format(self.state.time))

    def summary(self):
        terminate_string = '[${}]'.format(self.n_saved) if self.terminated else ''
        return '{0.name}|{0.loc}|S{0.n_saved}|C{0.n_carrying}'.format(self) + terminate_string
