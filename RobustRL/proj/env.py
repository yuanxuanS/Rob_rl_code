import numpy as np
import torch

from IC import runIC_repeat
class Environment(object):

    def __init__(self, graph_pool, node_feat_pool, z_pool, budget):
        ## 图
        self.graphs = graph_pool
        self.node_features_pool = node_feat_pool
        self.z_pool = z_pool
        # print(self.A)
        self.G = None
        self.node_features = None
        self.z = None
        self.propagate_p_matrix = None
        ## 超参模型

        self.edge_features = []     # 通过 generate_edge_feature 拼接生成

        ## 网络上的迭代

        self.budget = budget
        self.state = None
        self.done = False

        # print(f"state dim is {self.state.ndim}")
        ##  test
        self.print_tag = "ENV---"
    def init_graph(self, g_id):

        self.G = self.graphs[g_id]  # a Graph class
        self.g = self.G.graph  # a graph
        self.N = self.G.node  # 节点个数， int

        self.adj_matrix = self.G.adj_matrix  # 邻接矩阵，n,n

        ## 图上的传播
        self.propagate_p_matrix = np.zeros([self.N, self.N])  # 边上的传播概率

    def init_n_feat(self, ft_id):
        '''
        初始化节点特征、环境超参z、图的传播概率、node state
        :return:
        '''
        self.node_features = self.node_features_pool[ft_id]
        self.node_feat_dimension = len(self.node_features[0])  # 节点特征维度


    def init_hyper(self, hyper_id):
        self.z = self.z_pool[hyper_id]
        # 通过超参z初始化传播概率矩阵
        self.propagate_p_matrix = self.hyper_model()


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
        seeds_set = [v for v in range(self.N) if self.state[0][v]==1]
        # print("seeds is {}".format(seeds))
        # 蒙特卡洛得到influence reward
        influence_without = self.run_cascade(seeds=seeds_set)
        seeds_set.append(main_action)       ################# main_action 是tensor
        influence_with = self.run_cascade(seeds=seeds_set)
        self.reward = (influence_with - influence_without)
        # 归一化，×100%
        # self.reward = self.reward / self.N * 100


    # update next_state and done
        # main agent
        next_state = self.transition(main_action).copy()     # copy保证返回的next_state不会随着内部self.state的变化而改变

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
        reward, _ = runIC_repeat(self.g, seeds, p=self.propagate_p_matrix)
        return reward

    def generate_node_feature(self):
        # 返回n个节点特征，特征维度为d
        # 返回 n*d np.array
        '''

        :param node_feat_dimension:
        :return: node_features, numpy array, 2 dimens, n * node_feat_dimension
        '''

        # node_features = np.zeros((n, node_feat_dimension))
        node_features = np.random.rand(self.N, self.node_feat_dimension)  # 0-1
        # print(f"{self.print_tag} node feature initialized done!")
        # print(self.node_features)
        return node_features

    def generate_edge_features(self):
        '''

        :param node_features: numpy array, 2 dimen
        :return: edge_features, nested list, edge_number * (2*d) [[], [], ...]
        '''

        def gen_edge_fea(u, v):
            # print(node_features[u])   # 索引后为一维
            cat_fea = np.concatenate((self.node_features[u], self.node_features[v]))        #
            # print(f"edge feature dim is {cat_fea.ndim}")
            return list(cat_fea)



        for start_node, end_node in self.G.edges:  # 每个节点的邻节点

            edge_fea = gen_edge_fea(start_node, end_node)

            self.edge_features.append(edge_fea)
            # print(edge_features)
        return self.edge_features



    def hyper_model(self):
        '''
        产生传播概率
        :param edge_features:
        :param z:   [1, ]
        :return: self.propagate_p_matrix, n*n array
        '''
        self.edge_features = self.generate_edge_features()
        multi = self.edge_features * self.z
        propagate_p_list = multi.mean(axis=1)      # 0-1


        # 构造权重矩阵
        idx = 0
        for start_node, end_node in self.G.edges:  # 每个节点的邻节点
            self.propagate_p_matrix[start_node, end_node] = propagate_p_list[idx]
            idx += 1

        # print(self.propagate_p_matrix)
        print(f"{self.print_tag} propagate probability initialized done!")
        return self.propagate_p_matrix