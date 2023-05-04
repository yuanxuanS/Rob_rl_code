from models import GAT
import torch
import numpy as np


class DQAgent:
    def __init__(self, graph, model_name, init_epsilon):
        self.graph = graph  # a graph, Graph_IM instance
        self.model_name = model_name

        # policy
        self.curr_epsilon = init_epsilon

        self.memory = []
        if self.model_name == 'GAT_QN':
            # args
            features_dim = 100
            hidden_dim = 4
            dropout = 0.6
            alpha = 0.2  # leakyReLU的alpha
            nhead = 1
            self.model = GAT(nfeat=features_dim, nhid=hidden_dim, nout=1, alpha=alpha, nheads=nhead)

    def reset(self):
        '''
        获取图的节点--action和邻接矩阵
        :param graph: Graph_IM instance
        :return:
        '''

        self.nodes = self.graph.node
        self.adj = self.graph.adj_matrix
        if not isinstance(self.adj, torch.FloatTensor):
            self.adj = torch.from_numpy(self.adj)
            self.adj = self.adj.type(torch.FloatTensor)

        self.last_action = 0
        self.last_observation = torch.zeros(self.nodes, 1, dtype=torch.float)       # 暂时和env的state一样
        print(f"observation is {self.last_observation}")
        self.last_reward = -0.01

    def act(self, observation, feasible_action):
        '''
        epsilon-greedy policy
        :param observation: state [1, n] 0/1值
        :param feasible_action: action list
        :return:
        '''
        if self.curr_epsilon > np.random.rand():
            action = np.random.choice(feasible_action)
            # print(f"action is {action}")
        else:
            # GAT, 输入所有节点特征， 图的结构关系-邻接矩阵，
            q_a = self.model(xv, self.adj)
