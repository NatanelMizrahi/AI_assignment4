from environment import Environment
from graph import EvacuateNode


class Agent:
    def __init__(self, name, start_loc: EvacuateNode):
        self.loc: EvacuateNode = start_loc
        self.actions_seq = []
        self.name = name
        self.n_saved = 0
        self.penalty = 0
        self.n_carrying = 0
        self.terminated = False
        self.time = 0
        self.state = None

        print("{}({}) created in {}".format(self.name, self.__class__.__name__, start_loc))

    def get_score(self):
        return self.n_saved

    def is_reachable(self, env: Environment, v: EvacuateNode, verbose=False):
        """returns True iff transit to node v can be finished within v's deadline AND (u,v) is not blocked"""
        e = env.G.get_edge(self.loc, v)
        if env.G.is_blocked(e.v1, e.v2):
            if verbose:
                print('edge ({},{}) is blocked.'.format(e.v1, e.v2))
            return False
        if self.time + e.w > v.deadline:
            if verbose:
                print('cannot reach {} from {} before deadline. time={} e.w={} deadline={}'
                      .format(v.label, self.loc, self.time, e.w, v.deadline))
            return False
        return True

    def goto_end_time(self, env, v):
        return self.time + env.G.get_edge(self.loc, v).w

    def traverse(self, env: Environment, v: EvacuateNode):
        self.time = self.goto_end_time(env, v)
        self.loc = v
        self.try_evacuate(env, v)

    def terminate(self):
        self.terminated = True

    def try_evacuate(self, env: Environment, v: EvacuateNode):
        if self.terminated:
            return
        if v.is_shelter():
            if self.n_carrying > 0:
                # debug('Dropped off {.n_carrying} people'.format(self))
                self.n_saved += self.n_carrying
                self.n_carrying = 0
        elif not v.evacuated:
            # debug('Picked up {} people'.format(v.n_people))
            self.n_carrying += v.n_people
            v.evacuated = True
            v.n_people = 0
            env.require_evac_nodes.remove(v)


    def summary(self):
        terminate_string = '[${}]'.format(self.get_score()) if self.terminated else ''
        return '{0.name}|{0.loc}|S{0.n_saved}|C{0.n_carrying}{0.goto_str}|T{0.time:.2f}'.format(self) + terminate_string

    def __hash__(self):
        return hash(repr(self))
