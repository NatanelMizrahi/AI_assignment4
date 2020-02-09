# Introduction to Artificial Inteligence
## Programming assignment 4 - Decision-making under uncertainty
### Hurricane Evacuation Problem
#### How to run
```
python3 hurricane_simulator.py -h
usage: python3 hurricane_simulator.py [-h] [-g GRAPH_PATH]

The Hurricane Evacuation Problem - Decision-making under uncertainty example:
python3 hurricane_simulator.py --graph_path tests/basic.config

optional arguments:
  -h, --help            show this help message and exit
  -g GRAPH_PATH, --graph_path GRAPH_PATH
                        path to graph initial configuration
```

#### Goals
Sequential decision making under uncertainty using belief-state MDP for decision-making: the Hurricane Evacuation problem. (This is an obscure variant of the Canadian traveler problem.)

#### Hurricane Evacuation Decision Problem - Domain Description
The domain description is similar to that described in assignment 1, except that again we do not know the locations of the blocakges. For simplicity, however, we will assume that the blockages occur independently, with a known given probability. They are revealed with certainty when the agent reaches a neigbouring vertex. We will also assumes that the number of evacuees at each vertex is known, and is always less than 5.

Thus, in the current problem to be solved, we are given a weighted undirected graph, where each edge has a known probability of being blocked. These distributions are jointly independent. The agent's only actions are traveling between vertices and terminating. Traversal times are the weight of the edges. Also for simplicity, we will assume only one agent, starting at Start, and only one shelter at vertex Shelter, The problem is to find a policy that saves (in expectation) as many people as possible.

The graph can be provided in a manner similar to previous assignments, for example:
```
#V5    ; number of vertices n in graph (from 1 to n)

#V3 D7 P2  ; Vertex 3, deadline 7, has 2 evacuees
#V4 D5 P1  ; Vertex 4, deadline 5, has 1 evacuee

etc.

#E1 1 2 W3   ; Edge from vertex 1 to vertex 2, weight 3
#E2 2 3 W2   ; Edge from vertex 2 to vertex 3, weight 2
#E3 3 4 W3 B0.3  ; Edge from vertex 3 to vertex 4, weight 3, probability of blockage 0.3
#E4 4 5 W1   ; Edge from vertex 4 to vertex 5, weight 1
#E5 2 4 W4   ; Edge from vertex 2 to vertex 4, weight 4
#Start 1
#Shelter 5
```
The start and goal (shelter) vertices, should be determined via some form of user input (in the file like the above example, or by querying the user). For example, in the above graph the start vertex is 1, and the goal (shelter) vertex is 5.

#### Solution method
The Canadian traveller problem is known to be PSPACE-complete (and it is likely that the uncertain Hurricane Evacuation problem is also PSPACE-complete) so you will be required to solve only very small instances. We will require that the entire belief space be stored in memory explicitly, and thus impose a limit of at most 10 vertices with and at most 10 possible blockages. Your program should initialize belief space value functions and use a form of value iteration (discussed in class) to compute the value function for the belief states. Maintain the optimal action during the value iteration for each belief state, so that you have the optimal policy at convergence.

#### Requirements
Your program should read the data, including the parameters (start and shelter vertices). You should construct the policy, and present it in some way. Provide at least the following types of output:

1. A full printout of the value of each belief-state, and the optimal action in that belief state, if it exists. (Print something that indicates so if this state is irregular, e.g. if it is unreachable).
2. Run a sequence of simulations. That is, generate a graph instance (blockage locations) according to the distributions, and run the agent through this graph based on the (optimal) policy computed by your algorithm. Display the graph instance and sequence of actions. Allow the user to run additional simulations, for a newly generated instance in each case.
#### Deliverables
1. Source code and executable files of programs.
2. Explanation of the method employed in your algorithm.
3. Non-trivial example runs on at least 2 scenarios, including the input and output.
4. Submit makefile and/or short description on how to run your program.
