import numpy as np
import torch
import networkx as nx
import time
import matplotlib.pyplot as plt
import logging
from utils import draw_distri_hist

from IC import runIC_repeat


class Environment(object):

    def __init__(self, graph_pool, node_feat_pool, z_pool, budget):
        ## 图
        self.graphs = graph_pool
        self.g_id = None
        self.graph_greedy_dict = {}

        self.node_features_pool = node_feat_pool
        self.feat_id = None
        self.z_pool = z_pool
        self.z_id = None
        # print(self.A)
        self.G = None
        self.node_features = None
        self.z = None
        self.path = None
        # self.propagate_p_matrix = None
        ## 超参模型

        self.edge_features = []  # 通过 generate_edge_feature 拼接生成

        ## 网络上的迭代

        self.budget = budget
        self.state = None
        self.done = False

        # print(f"state dim is {self.state.ndim}")
        ##  test
        self.print_tag = "ENV---"

    def init_graph(self, g_id):
        self.g_id = g_id
        self.G = self.graphs[g_id]  # a Graph class
        self.g = self.G.graph  # a graph
        self.N = self.G.node  # 节点个数， int

        self.adj_matrix = self.G.adj_matrix  # 邻接矩阵，n,n

        ## 图上的传播
        # self.propagate_p_matrix = np.zeros([self.N, self.N])  # 边上的传播概率

    def init_n_feat(self, ft_id):
        '''
        初始化节点特征、环境超参z、图的传播概率、node state
        :return:
        '''
        self.feat_id = ft_id
        self.node_features = self.node_features_pool[ft_id]
        self.node_feat_dimension = len(self.node_features[0])  # 节点特征维度

    def init_hyper(self, hyper_id):
        self.z_id = hyper_id
        self.z = self.z_pool[hyper_id]
        # 通过超参z初始化传播概率矩阵
        # self.propagate_p_matrix = self.hyper_model()
        self.hyper_model()

    def reset(self):
        # episode状态记录相关的
        self.init_state()
        self.done = False

    def init_state(self):
        self.state = np.zeros((1, self.N), dtype=int)  # 二维（1，N）0/1向量，节点加入set对应下标，=1
        # print(f"{self.print_tag} seed set state initialized done!")

    def get_seed_state(self):
        # print(f'{self.print_tag} current state is {self.state}')
        # available_action_mask = np.array([1] * self.G.cur_n)
        feasible_action = [idx for idx in range(len(self.state[0])) if self.state[0][idx] == 0]
        # print(f"{self.print_tag} feasible actions are {feasible_action}")
        return self.state, feasible_action

    def get_z_state(self):
        return self.z

    def transition(self, action_node):
        # action_node为选择的一个节点，为节点的index，根据action更新state
        self.state[0, action_node] = 1
        return self.state

    def step_seed(self, i, main_action):
        '''
        计算reward并且进行state update, reward-marginal contribution
        :param action: node下标
        :return: reward， next_state
        '''

        # compute reward as marginal contribution of a node
        ## 计算marginal contribution的一种方式
        # 根据state统计seeds个数
        seeds_set = [v for v in range(self.N) if self.state[0][v] == 1]
        # print("seeds is {}".format(seeds))
        # 蒙特卡洛得到influence reward
        ccwn_st = time.time()
        influence_without, std_without = self.run_cascade(seeds=seeds_set)
        ccwn_ed = time.time()

        print(f"time of repeat cascade is {ccwn_ed - ccwn_st}")
        logging.debug(f"simulation: seed set- {seeds_set}, mean is {influence_without}, std is {std_without}")
        seeds_set.append(main_action)  ################# main_action 是tensor

        ccw_st = time.time()
        influence_with, std_with = self.run_cascade(seeds=seeds_set)
        ccw_ed = time.time()
        print(f"time of IC, more 1 node, is {ccw_ed - ccw_st}")
        logging.debug(f"simulation: seed set- {seeds_set}, mean is {influence_with}, std is {std_with}")

        self.reward = (influence_with - influence_without)
        # 归一化，×100%
        self.reward = self.reward / self.N

        # update next_state and done
        # main agent
        next_state = self.transition(main_action).copy()  # copy保证返回的next_state不会随着内部self.state的变化而改变

        if i == self.budget - 1:
            self.done = True
        return next_state, self.reward, self.done

    def step_hyper(self, z_action_pair_lst):
        # z_action: torch.FloatTensor
        action_nosoftmax, z_action = z_action_pair_lst
        # 先类型转换
        if not isinstance(z_action, np.ndarray):
            if isinstance(z_action, torch.FloatTensor):
                self.z = z_action.numpy()

        self.hyper_model()
        # print(f"{self.print_tag} after add {self.z}")
        return self.z

    def run_cascade(self, seeds):
        # reward, _ = runIC_repeat(self.g, seeds, p=self.propagate_p_matrix)
        reward, std = runIC_repeat(self.g, seeds, p=None)

        return reward, std

    def generate_edge_features(self):
        '''

        :param node_features: numpy array, 2 dimen
        :return: edge_features, nested list, edge_number * (2*d) [[], [], ...]
        '''
        self.edge_features = []

        def gen_edge_fea(u, v):
            # print(node_features[u])   # 索引后为一维
            cat_fea = np.concatenate((self.node_features[u], self.node_features[v]))  #
            # print(f"edge feature dim is {cat_fea.ndim}")
            return list(cat_fea)

        for start_node, end_node in self.G.edges:  # 每个节点的邻节点

            edge_fea = gen_edge_fea(start_node, end_node)

            self.edge_features.append(edge_fea)
            # print(edge_features)
        return self.edge_features

    def draw_graph(self):
        print("draw graph! ")
        elarge = [(u, v) for (u, v, d) in self.g.edges(data=True) if d["weight"] > 0.5]
        esmall = [(u, v) for (u, v, d) in self.g.edges(data=True) if d["weight"] <= 0.5]
        pos = nx.spring_layout(self.g, seed=7)  # positions for all nodes - seed for reproducibility

        # nodes
        nx.draw_networkx_nodes(self.g, pos, node_size=80)

        # edges
        nx.draw_networkx_edges(self.g, pos, edgelist=elarge, width=3)
        nx.draw_networkx_edges(
            self.g, pos, edgelist=esmall, width=3, alpha=0.5, edge_color="b", style="dashed"
        )

        # node labels
        nx.draw_networkx_labels(self.g, pos, font_size=5, font_family="sans-serif")
        # edge weight labels
        # edge_labels = nx.get_edge_attributes(self.g, "weight")
        edge_labels = dict([((u, v,), f"{d['weight']:.2f}") for u, v, d in self.g.edges(data=True)])
        nx.draw_networkx_edge_labels(self.g, pos, edge_labels, font_size=3)

        plt.figure(1, figsize=(800, 800), dpi=100)
        ax = plt.gca()
        ax.margins()
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(self.path + "_graph network")
        plt.close()

    def hyper_model(self):
        '''
        产生传播概率, 初始化graph 的edge weight
        :param edge_features:
        :param z:   [1, ]
        # :return: self.propagate_p_matrix, n*n array
        '''
        self.edge_features = self.generate_edge_features()
        # print(f"edge feature is\n {self.edge_features}")
        # print(f"hyper param is\n {self.z}")
        multi = self.edge_features * self.z
        # print(f"after multi is\n {multi}")
        propagate_p_list = multi.mean(axis=1)  #
        # print(f"propa p is \n {propagate_p_list}")
        # print(f"edge number is {len(propagate_p_list)}")
        draw_distri_hist(propagate_p_list, self.path, "propagate_prob")
        # 构造权重矩阵
        idx = 0
        for start_node, end_node in self.G.edges:  # 每个节点的邻节点
            # self.propagate_p_matrix[start_node, end_node] = propagate_p_list[idx]
            self.g.add_edge(start_node, end_node, weight=propagate_p_list[idx])
            idx += 1

        # print(self.propagate_p_matrix)
        print(f"{self.print_tag} propagate probability initialized done!")

        print(f"initial graph weighted")
        self.G.gener_node_degree_lst()
        # self.draw_graph()
        # return self.propagate_p_matrix

    def greedy_solution(self):
        # get greedy solution in current graph
        """
            input
            G: the graph you input
            k: number of nodes in influence maximization set, which equals budget size

            output
            influence maximization set
            spread of each node
            """
        if self.g_id in self.graph_greedy_dict.keys():
            if self.feat_id in self.graph_greedy_dict[self.g_id].keys():
                if self.z_id in self.graph_greedy_dict[self.g_id][self.feat_id].keys():
                    logging.debug(f"graph {self.g_id}, feat {self.feat_id}, z {self.z_id} in dict, no recompute it")
                    S, spread = self.graph_greedy_dict[self.g_id][self.feat_id][self.z_id]
                    return S, spread

        S, spread = [], []
        # S为seed set, spread代表每个seed的传染节点个数
        for _ in range(self.budget):
            spread_mem, node_mem = -1, -1
            for i in set(range(int(self.G.node))) - set(S):  # set函数是做一个集合，里面不能包含重复元素，里面接受一个list做参数
                s, _ = runIC_repeat(self.G.graph, S + [i])
                if s > spread_mem:  # 遍历找到spread最广的节点
                    # print(f"larger spread is {len(s)}, node {i}")
                    spread_mem = s
                    node_mem = i
            S.append(node_mem)
            spread.append(spread_mem)
        spread = [s / self.N for s in spread]

        if self.g_id not in self.graph_greedy_dict.keys():
            self.graph_greedy_dict[self.g_id] = {}

        if self.feat_id not in self.graph_greedy_dict[self.g_id].keys():
            self.graph_greedy_dict[self.g_id][self.feat_id] = {}

        if self.z_id not in self.graph_greedy_dict[self.g_id][self.feat_id].keys():
            self.graph_greedy_dict[self.g_id][self.feat_id][self.z_id] = [S, spread]
        return S, spread