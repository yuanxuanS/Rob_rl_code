import numpy as np
import networkx as nx
from graph import Graph_IM
def generate_node_feature(graph, dimension):
    # 返回n个节点特征，特征维度为d
    # 返回 n*d np.array
    '''

    :param graph: Graph.graph 图, networkx
    :param dimension:
    :return: node_features, numpy array, 2 dimens, n * dimension
    '''
    n = graph.node
    # node_features = np.zeros((n, dimension))
    node_features = np.random.rand(n, dimension)        # [0, 1]
    # print(node_features)
    return node_features


def generate_edge_features(node_features, graph):
    '''

    :param node_features: numpy array, 2 dimens
    :param graph: Graph_IM instance
    :return: edge_features, nested list, edge_number * (2*d) [[], [], ...]
    '''

    def gen_edge_fea(u, v):
        # print(node_features[u])   # 索引后为一维
        cat_fea = np.concatenate((node_features[u], node_features[v]))
        return list(cat_fea)

    g = graph.graph
    edge_features = []
    for start_node, end_node in graph.edges:     # 每个节点的邻节点
        print(f"two nodes is {start_node}, {end_node}")

        edge_fea = gen_edge_fea(start_node, end_node)
        edge_features.append(edge_fea)
            # print(edge_features)

    return edge_features


# generate_node_feature(graph, 3)