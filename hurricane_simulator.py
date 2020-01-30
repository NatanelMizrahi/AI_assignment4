import re
import argparse
from agent import Agent
from graph import Edge, Graph, ShelterNode, EvacuateNode
from environment import Environment
from search_agents import State


class Simulator:
    """Hurricane evacuation simulator"""

    def __init__(self, graph_path):
        self.agent = None
        self.G: Graph = self.parse_graph(graph_path)
        self.env: Environment = Environment(self.G)

    def parse_graph(self, path):
        """Parse and create graph from tests file, syntax same as in assignment instructions"""
        num_v_pattern = re.compile("#N\s+(\d+)")
        start_pattern = re.compile("#Start\s+(\d+)")
        shelter_vertex_pattern = re.compile("#(V\d+)\s+D(\d+)\s+S")
        person_vertex_pattern  = re.compile("#(V\d+)\s+D(\d+)\s+P(\d+)")
        edge_pattern = re.compile("#(E\d+)\s+(\d+)\s+(\d+)\s+W(\d+)\s*(?:B(0\.[0-9]+))?")

        shelter_nodes = []
        person_nodes = []
        name_2_node = {}
        n_vertices = 0
        E = []
        start_loc = None
        max_deadline = 0

        with open(path, 'r') as f:
            for line in f.readlines():
                if not line.startswith('#'):
                    continue

                match = num_v_pattern.match(line)
                if match:
                    n_vertices = int(match.group(1))

                match = start_pattern.match(line)
                if match:
                    node_idx = match.groups()[0]
                    start_loc = name_2_node['V' + node_idx]

                match = shelter_vertex_pattern.match(line)
                if match:
                    name, deadline = match.groups()
                    new_node = ShelterNode(name, int(deadline))
                    shelter_nodes.append(new_node)
                    name_2_node[new_node.label] = new_node
                    max_deadline = max(max_deadline, int(deadline))

                match = person_vertex_pattern.match(line)
                if match:
                    name, deadline, n_people = match.groups()
                    new_node = EvacuateNode(name, int(deadline), int(n_people))
                    person_nodes.append(new_node)
                    name_2_node[new_node.label] = new_node
                    max_deadline = max(max_deadline, int(deadline))

                match = edge_pattern.match(line)
                if match:
                    name, v1_name, v2_name, weight, block_chance = match.groups()
                    print(block_chance)
                    block_chance = 0.0 if (block_chance is None) else float(block_chance)
                    v1 = name_2_node['V'+v1_name]
                    v2 = name_2_node['V'+v2_name]
                    E.append(Edge(v1, v2, int(weight), name, block_chance))

        self.agent = Agent('A1', start_loc)
        V = person_nodes + shelter_nodes

        if n_vertices != len(V):
            raise Exception("Error: |V| != N")
        return Graph(V, E)

    def run_simulation(self):
        self.env.G.display()
        State.init_state_space(self.G)
        s: State = State.get_initial_state(self.agent.loc)
        print(s)
        while s.best_option is not None:
            s = s.result(s.best_option)
            print(s)
        # self.agent.create_policy(self.G)
        # self.agent.follow_policy()
        # print('** STARTING SIMULATION **')
        # while not self.agent.terminated:
        #     tick = self.env.time
        #     print('\nT={}'.format(tick))
        #     for agent in self.env.agents:
        #         self.env.G.display('T={}: {}'.format(tick, agent.name))
        #         self.agent.act(self.env)
        #     self.env.tick()
        # self.env.G.display('Final State: T=' + str(self.env.time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
            The Hurricane Evacuation Problem - Decision-making under uncertainty
            example: python3 hurricane_simulator.py --graph_path tests/basic.config --seed 3''')
    parser.add_argument('-g', '--graph_path', default='tests/test.config', help='path to graph initial configuration')
    parser.add_argument('-s', '--seed', default=42, type=int, help='random seed for the sampling. Used for debugging')
    args = parser.parse_args()
    Simulator(args.graph_path).run_simulation()
