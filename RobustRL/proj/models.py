import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Word2Vec
import networkx as nx
import random
import numpy as np
from layers import GraphAttentionLayer




# hidden_reg = 0
# embed_dim = 5
# len_pre_pooling = 0
# len_post_pooling = 0
# T = 3
# s2v = S2V_QN(reg_hidden=hidden_reg, embed_dim=embed_dim,
#              len_pre_pooling=len_pre_pooling, len_post_pooling=len_post_pooling, T=T)


# n = 6
# g = nx.erdos_renyi_graph(n=n, p=0.5)
# len_pre = 0
# len_post = 0
# embed_dim = 4
# window_size = 2
# num_paths = 1
# path_length = 3
# T = 2
#
#
# w2v = W2V_QN(g, len_pre, len_post, embed_dim, window_size, num_paths, path_length, T)
# x = g.nodes()
# x = torch.Tensor(x)
# x = x.reshape(-1, 1)        # [n, 1]
# print(f"nodes are {x}")
# adj = nx.adjacency_matrix(g).todense()
# adj = torch.from_numpy(adj).to(torch.float32)       # [n, n]
# print(f"adj is {adj}")
# mu_init = None
# w2v(x, adj, mu_init)

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, alpha, nheads, mergeZ, mergeState, use_cuda, device):
        """Dense version of GAT."""
        super(GAT, self).__init__()

        self.use_cuda = use_cuda
        self.device = device
        self.nfeat = nfeat
        self.nhid = nhid

        self.nout = nout
        self.alpha = alpha
        self.mergeZ = mergeZ
        self.mergeState = mergeState
        # k多头，有k个头就有k个层。
        # nhid 就是输出的特征大小。 因为GAT的隐藏层hid会输出特征
        # print(f"nhid is {nhid}")
        # print(f"stack multi GAT layer")
        self.attentions = [GraphAttentionLayer(self.nfeat, self.nhid, alpha=self.alpha, concat=True, mergeZ=self.mergeZ, node_dim=self.nfeat) for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            # print(f"第{i}个layer, {str(attention)}")
            self.add_module('attention_{}'.format(i), attention)

        # 会把多头的结果拼接起来，所以是nhid * nheads， 输出n个类
        #################
        # print(f"add final attention layer")
        self.out_att = GraphAttentionLayer(self.nhid * nheads, self.nout, alpha=self.alpha, concat=False, mergeZ=self.mergeZ, node_dim=self.nfeat)

        # seed set feature theta
        self.theta = nn.Parameter(torch.empty(1, self.nfeat))  # 大小和每个节点的feature向量一样
        nn.init.uniform_(self.theta, a=0, b=1)  # 均匀分布

        # test
        self.print_tag = "Models ---"

    def forward(self, x, adj, observation, z=None):

        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        if not isinstance(adj, torch.Tensor):
            adj = torch.Tensor(adj)
        #融合seed set信息
        if self.mergeState:
            seed_set = [idx for idx in range(len(observation[0])) if observation[0][idx] == 1]
            sdset_mask = torch.zeros([x.size()[0], x.size()[1]])
            sdset_mask[seed_set] += 1.
            if self.use_cuda:
                sdset_mask = sdset_mask.to(self.device)

            # print(f"use cuda {self.device} | mask type {sdset_mask.device} | theta type {self.theta.device}")
            sdset_mask = sdset_mask * self.theta
            x = x + sdset_mask
            # print(f"seed set mask is {sdset_mask}")
            # print(f"x after seed set mask is {x}")


        ## x, features [n, feature_size]


        x = torch.cat([att(x, adj, z) for att in self.attentions], dim=1)

        x = F.elu(self.out_att(x, adj, z))


        result = F.log_softmax(x, dim=0)    # 不是分类问题，应该是纵向softmax
        return result


# features_dim = 10
# hidden_dim = 4
# dropout = 0.
# alpha = 0.2     # leakyReLU的alpha
# nhead = 1
# model = GAT(nfeat=features_dim, nhid=hidden_dim, nout=1, dropout=dropout, alpha=alpha, nheads=nhead)
# # test
# graph = Graph_IM(nodes=10, edges_p=0.5)
# adj_matrix = graph.adj_matrix
# # print(f"graph adj matrix {graph.adj_matrix}")
# adj_matrix = torch.Tensor(adj_matrix)
# xv = generate_node_feature(graph, features_dim)
# print(f"node feature vector size {xv}")     # [0-100]
# xv = torch.Tensor(xv)       # torch的输入必须是tensor
# y = model(xv, adj_matrix)
#
# print(f"get y from GAT is {y}")     # [n, 1]
# action = y.argmax()     # node index, int
# print(f"max node idx is {action}")