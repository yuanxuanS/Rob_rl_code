import time
# starttime = time.strftime("%Y-%m-%d_%H:%M:%S")
# print("start time", starttime, starttime[:13])

import numpy as np
# tmp = np.random.normal(loc=0.5, scale=0.5, size=(1, 2*3))      # normal distribution

import matplotlib.pyplot as plt

# mu = 0.5
# for sigma in [3]:
#     s = np.random.normal(mu, sigma, 1000)
#     count, bins, ignored = plt.hist(s, 30, density=True)
#     plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
#                np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
#          linewidth=2, color='r')
#     plt.savefig("./other/test_normal"+str(sigma)+".jpg")
# plt.show()


# print("before", tmp)
# tmp = np.clip(tmp, a_min=0, a_max=1)
# print(tmp)
import torch
# criterion = torch.nn.MSELoss(reduction='mean')
# tmp = torch.Tensor([[1, 2, 3]])
# tmp = torch.concat((tmp, torch.Tensor([[4]])), 1)     # 注意tensor有几维
# print(tmp)
# tmp_ = torch.Tensor([1])
# print(criterion(tmp, tmp_))

# from pyrwr.rwr import RWR
import pandas as pd
# input_graph = "./g.tsv"
# rwr = RWR()
# graph_type = 'undirected'
# rwr.read_graph(input_graph, graph_type)
# c = 0.3
# epsilon = 1e-6
# max_iters = 100
# r = rwr.compute(1, c, epsilon, max_iters)
# print(r)


# 没看懂是怎么p的传参是怎么传的，把传播概率传进来就没问题了。
from graph import Graph_IM
from IC import runIC_repeat
import random
def gener_node_features(node_nbr, node_dim, feat_nbr, normal_mean):
    n_feat_dic = {}
    for f in range(feat_nbr):
        seed = f
        np.random.seed(seed)
        # tmp = np.random.normal(loc=normal_mean, scale=3, size=(node_nbr, node_dim))
        # n_feat_dic[f] = np.clip(tmp, a_min=0, a_max=normal_mean+0.1)  # 0-1
        n_feat_dic[f] = np.ones([node_nbr, node_dim]) * normal_mean

    return n_feat_dic


def generate_edge_features(node_features, G):
    '''

    :param node_features: numpy array, 2 dimen
    :return: edge_features, nested list, edge_number * (2*d) [[], [], ...]
    '''
    edge_features = []

    def gen_edge_fea(u, v):
        # print(node_features[u])   # 索引后为一维
        cat_fea = np.concatenate((node_features[u], node_features[v]))  #
        # print(f"edge feature dim is {cat_fea.ndim}")
        return list(cat_fea)

    print(f"graph edges :{G.edges}")
    for start_node, end_node in G.edges:  # 每个节点的邻节点

        edge_fea = gen_edge_fea(start_node, end_node)

        edge_features.append(edge_fea)
        # print(edge_features)
    return edge_features

def gener_z(node_dim, z_nbr, z_mean):
    z_dic = {}
    for z_i in range(z_nbr):
        seed = z_i
        np.random.seed(seed)
        # z_dic[z_i] = np.random.rand(1, 2 * node_dim)        # uniform distribution
        # tmp = np.random.normal(loc=0.5, scale=3, size=(1, 2*node_dim))
        # z_dic[z_i] = np.clip(tmp, a_min=0, a_max=1)
        z_dic[z_i] = np.ones([1, 2*node_dim]) * z_mean

    return z_dic


def greedy(G, k):
    """
    input
    G: the graph you input
    k: number of nodes in influence maximization set, which equals budget size

    output
    influence maximization set
    spread of each node
    """
    S, spread = [], []
    # S为seed set, spread代表每个seed的传染节点个数
    for _ in range(k):
        spread_mem, node_mem = -1, -1
        for i in set(range(int(G.node))) - set(S): # set函数是做一个集合，里面不能包含重复元素，里面接受一个list做参数
            s, _ = runIC_repeat(G.graph, S + [i])
            if s > spread_mem: # 遍历找到spread最广的节点
                # print(f"larger spread is {len(s)}, node {i}")
                spread_mem = s
                node_mem = i
        S.append(node_mem)
        spread.append(spread_mem)
    return S, spread


# Code Sample:
# node_nbr = 50
# budget=2
# node_edge_p = 0.1
# seed=0
# G = Graph_IM(nodes=node_nbr, edges_p=node_edge_p, seed=seed)
#
# feat = gener_node_features(node_nbr, 3, 1, 0.5)[0]
# print(f"feature pool: one {feat}")
# edge_features = generate_edge_features(feat, G)
# print(f"edge feature is\n {edge_features}")
# z = gener_z(3, 1, 0.5)[0]
# print(f"hyper param is\n {z}")
# multi = edge_features * z
# print(f"after multi is\n {multi}")
# propagate_p_list = multi.mean(axis=1)  #
# print(f"propa p is \n {propagate_p_list}")
# print(f"edge number is {len(propagate_p_list)}")
# # 构造权重矩阵
# idx = 0
# for start_node, end_node in G.edges:  # 每个节点的邻节点
#     # self.propagate_p_matrix[start_node, end_node] = propagate_p_list[idx]
#     G.graph.add_edge(start_node, end_node, weight=propagate_p_list[idx])
#     idx += 1
#
# # print(self.propagate_p_matrix)
# # print(f"{self.print_tag} propagate probability initialized done!")
#
# print(f"initial graph weighted")
# G.gener_node_degree_lst()
#
# S1, spread1 = greedy(G, budget)
# print(S1, spread1)


# test random
# seed = 10
# random.seed(seed)  # Python random module.
# np.random.seed(seed)  # Numpy module.

def t_rand():

    print(f"random now :{random.random()}")
    print(f"np random now :{np.random.rand()}")


from utils import process_config
# process_config(1, 2,3)