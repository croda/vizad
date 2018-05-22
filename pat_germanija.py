from Python_prebaruvanje_vo_konechen_graph_final import *
Pocetok = input()
Kraj = input()

class GraphProblem(Problem):
    """The problem of searching a graph from one node to another."""

    def __init__(self, initial, goal, graph):
        Problem.__init__(self, initial, goal)
        self.graph = graph

    def actions(self, A):
        """The actions at a graph node are just its neighbors."""
        return list(self.graph.get(A).keys())

    def result(self, state, action):
        """The result of going to a neighbor is just that neighbor."""
        return action

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A, B) or infinity)

    def h(self, node):
        """h function is straight-line distance from a node's state to goal."""
        prob = GraphProblem(node.state, self.goal, self.graph)
        temp = breadth_first_graph_search(prob).solve()
        passedTowns = len(temp) - 1
        return passedTowns


graph = UndirectedGraph(dict(
    Frankfurt=dict(Mannheim=85, Wurzburg=217, Kassel=173),
    Kassel=dict(Munchen=502),
    Munchen=dict(Nurnberg=167, Augsburg=84),
    Augsburg=dict(Karlsruhe=250),
    Karlsruhe=dict(Mannheim=80),
    Wurzburg=dict(Erfurt=186, Nurnberg=103),
    Nurnberg=dict(Stuttgart=183)
))

graph_problem = GraphProblem(Pocetok, Kraj, graph)
answer = astar_search(graph_problem).path_cost
print(answer)
