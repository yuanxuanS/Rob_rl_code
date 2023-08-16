import networkx as nx
import logging
# class Graph:
#     def __init__(self):

# 生成图的方法
# g = nx.erdos_renyi_graph(n=3, p=0.5)
#
# print(list(g.nodes))
# print(list(g.edges))
# print(nx.adjacency_matrix(g))   # (i, j) weight


class Graph_IM:
    # 生成一张图, 返回图的性质
    def __init__(self, nodes, edges_p, seed=0):
        self.graph = nx.erdos_renyi_graph(n=nodes, p=edges_p, seed=seed)
        self.graph_name = None

        self.node_degree_lst = None
        self.sorted_ndegree_lst = None
        self.gener_node_degree_lst()
        

    def gener_node_degree_lst(self):
        self.node_degree_lst = [val for (node, val) in sorted(self.graph.degree(weight="weight"), key=lambda pair: pair[0])]
        logging.info(f"node degree lst (by index) is \n {self.node_degree_lst}")
        self.sorted_ndegree_lst = [val for (node, val) in sorted(self.graph.degree(weight="weight"), key=lambda pair: pair[1])]
        logging.info(f"sorted degree lst is \n {self.sorted_ndegree_lst}")

    @property
    def node(self):
        return nx.number_of_nodes(self.graph)       # number int

    @property
    def nodes(self):
        return self.graph.nodes()

    @property
    def edges(self):
        # list of tuples (start node, end node)
        return self.graph.edges()

    @property
    def adj_matrix(self):
        return nx.adjacency_matrix(self.graph).todense()
