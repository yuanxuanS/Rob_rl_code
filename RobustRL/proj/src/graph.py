import networkx as nx
import logging
import numpy as np


def softmax(x):
    max = np.max(x, axis=1, keepdims=True)  # returns max of each row and keeps same dims
    e_x = np.exp(x - max)  # subtracts each row with its max value
    sum = np.sum(e_x, axis=1, keepdims=True)  # returns sum of each row and keeps same dims
    f_x = e_x / sum
    return f_x


class Graph_IM:
    # 生成一张图, 返回图的性质
    def __init__(self, nodes, edges_p, seed=0):
        self.graph = nx.erdos_renyi_graph(n=nodes, p=edges_p, seed=seed)
        self.graph_name = None

        self.node_degree_lst = None
        self.sorted_ndegree_lst = None
        logging.info(f"initialize graph class")
        # self.gener_node_degree_lst()
        _ = self.adj_degree_matrix_sfm()        # 生成 self.adm
        

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

    def adj_degree_matrix_sfm(self):
        self.gener_node_degree_lst()
        self.degree_1v = np.array(self.node_degree_lst)
        # print(self.degree_1v, type(self.degree_1v), self.degree_1v.shape)
        self.ones_1v = np.ones((self.node, 1))
        # print(self.ones_1v.transpose(), self.ones_1v.transpose().shape)
        self.adm = self.degree_1v * self.ones_1v        # n, n
        # print(self.adm.shape)
        # zero_vec = -9e15*np.ones((self.node, self.node))
        # self.adm = np.where(self.adj_matrix>0, self.adm, zero_vec)
        # print("before", self.adm)
        # print(f"adj {np.array(self.adj_matrix)}")
        self.adm *= np.array(self.adj_matrix)       # element-wise multiply
        print("before", self.adm)

        self.adm /= self.node
        logging.debug("after", self.adm)
        print("after", self.adm)

        # self.adm = softmax(self.adm)
        # print(self.adm)

        return self.adm


# test
# nnode = 10
# edge_p = 0.8
# g = Graph_IM(nnode, edge_p)
# g.adj_degree_matrix_sfm()