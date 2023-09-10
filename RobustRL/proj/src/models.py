import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Word2Vec
import networkx as nx
import random
import numpy as np
from layers import GraphAttentionLayer
import logging
from graph import Graph_IM
from generate_node_feature import generate_node_feature
import math


class S2V_QN(torch.nn.Module):
    def __init__(self, reg_hidden, embed_dim, len_pre_pooling, len_post_pooling, T):

        super(S2V_QN, self).__init__()
        self.reg_hidden = reg_hidden
        self.embed_dim = embed_dim
        self.T = T
        self.len_pre_pooling = len_pre_pooling
        self.len_post_pooling = len_post_pooling
        self.mu_1 = torch.nn.Linear(1, embed_dim)
        #self.mu_1 = torch.nn.Parameter(torch.Tensor(1, embed_dim))
        #torch.nn.init.normal_(self.mu_1, mean=0, std=0.01)
        self.mu_2 = torch.nn.Linear(embed_dim, embed_dim)
        #self.mu_2 = torch.nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        #torch.nn.init.normal_(self.mu_2, mean=0, std=0.01)

        if self.len_pre_pooling > 0:
            self.list_pre_pooling = []
            for i in range(self.len_pre_pooling):
                pre_lin = torch.nn.Linear(embed_dim, embed_dim)
                torch.nn.init.normal_(pre_lin.weight, mean=0, std=0.01)
                self.list_pre_pooling.append(pre_lin)

        if self.len_post_pooling > 0:
            self.list_post_pooling = []
            for i in range(self.len_post_pooling):
                pre_lin = torch.nn.Linear(embed_dim, embed_dim)
                torch.nn.init.normal_(pre_lin.weight, mean=0, std=0.01)
                self.list_post_pooling.append(pre_lin)

        self.q_1 = torch.nn.Linear(embed_dim, embed_dim,bias=True)
        torch.nn.init.normal_(self.q_1.weight, mean=0, std=0.01)
        self.q_2 = torch.nn.Linear(embed_dim, embed_dim,bias=True)
        torch.nn.init.normal_(self.q_2.weight, mean=0, std=0.01)
        if self.reg_hidden > 0:
            self.q_reg = torch.nn.Linear(2 * embed_dim, self.reg_hidden,bias=True)
            torch.nn.init.normal_(self.q_reg.weight, mean=0, std=0.01)
            self.q = torch.nn.Linear(self.reg_hidden, 1,bias=True)
        else:
            self.q = torch.nn.Linear(2 * embed_dim, 1,bias=True)
        torch.nn.init.normal_(self.q.weight, mean=0, std=0.01)

    def forward(self, xv, adj):

        minibatch_size = xv.shape[0]
        nbr_node = xv.shape[1]


        for t in range(self.T):
            if t == 0:
                mu_1 = self.mu_1(xv)
                #mu_1 = torch.matmul(xv, self.mu_1)
                # mu_2 = self.mu_2(torch.matmul(adj, mu_init))
                # mu = torch.add(mu_1, mu_2).clamp(0)
                mu = mu_1.clamp(0)


            else:
                mu_1 = self.mu_1(xv)
                #mu_1 = torch.matmul(xv, self.mu_1)

                # before pooling:
                if self.len_pre_pooling > 0:
                    for i in range(self.len_pre_pooling):
                        mu = self.list_pre_pooling[i](mu).clamp(0)

                mu_pool = torch.matmul(adj, mu)

                # after pooling
                if self.len_post_pooling > 0:
                    for i in range(self.len_post_pooling):
                        mu_pool = self.list_post_pooling[i](mu_pool).clamp(0)

                mu_2_ = self.mu_2(mu_pool)
                #mu_2_ = torch.matmul(self.mu_2, mu_pool.transpose(1, 2))
                #mu_2_ = mu_2_.transpose(1, 2)
                mu = torch.add(mu_1, mu_2_).clamp(0)

        # q_1 = self.q_1(torch.sum( mu,dim=1).reshape(minibatch_size,1,self.embed_dim).expand(minibatch_size,nbr_node,self.embed_dim))
        xv = xv.transpose(1, 2)
        q_1 = self.q_1(torch.matmul(xv, mu))
        q_1_ = q_1.clone()
        q_1_ = q_1_.expand(minibatch_size, nbr_node, self.embed_dim)
        ####
        # mat = xv.reshape(minibatch_size, nbr_node).type(torch.ByteTensor)
        # mat = torch.ones(minibatch_size, nbr_node).type(torch.ByteTensor) - mat
        # res = torch.zeros(minibatch_size, nbr_node, nbr_node)
        # res.as_strided(mat.size(), [res.stride(0), res.size(2) + 1]).copy_(mat)
        # mu_ = mu.transpose(1, 2)
        # mu_y = torch.matmul(mu_, res)
        # mu_y = mu_y.transpose(1, 2)
        q_2 = self.q_2(mu)
        q_ = torch.cat((q_1_, q_2), dim=-1)
        if self.reg_hidden > 0:
            q_reg = self.q_reg(q_).clamp(0)
            q = self.q(q_reg)
        else:
            q_=q_.clamp(0)
            q = self.q(q_)
        return q


# hidden_reg = 0
# embed_dim = 5
# len_pre_pooling = 0
# len_post_pooling = 0
# T = 2
# s2v = S2V_QN(reg_hidden=hidden_reg, embed_dim=embed_dim,
#              len_pre_pooling=len_pre_pooling, len_post_pooling=len_post_pooling, T=T)

class attentions(nn.Module):
    def __init__(self, layers, nfeat, nhid, alpha, concat, mergeZ, node_dim, method="base"):
        super(attentions, self).__init__()

        self.nlayer = layers
        self.nfeat = nfeat
        self.nhid_tuple = nhid
        self.alpha = alpha
        self.mergeZ = mergeZ
        self.node_dim = node_dim


        assert self.nlayer == len(self.nhid_tuple), "layer number not equals to hidden number"

        self.attention = []
        for i in range(self.nlayer):
            out_dim = self.nhid_tuple[i]
            if i == 0:
                in_dim = self.nfeat

            else:
                in_dim = self.nhid_tuple[i-1]
            layer = GraphAttentionLayer(in_dim, out_dim, alpha=self.alpha, concat=concat, mergeZ=self.mergeZ,
                                        node_dim=self.node_dim, method=method)
            self.attention.append(layer)

    def forward(self, x, adj, s_mat, z=None):
        h = x
        for t in range(self.nlayer):
            h = self.attention[t](h, adj, s_mat, z)
            # print(f" layer {t} -- size {h.size()} ")
            # print(f"h {h}")

        return h

# test
# layer = 2
# nfeat = 3
# nhid = (8,4,)
# alpha = 0.2
# merge = False
# node_dim = 1
# atten = attentions(layer, nfeat, nhid, alpha, merge, node_dim)
#
# node = 5
# x = torch.ones(node, nfeat)
# x[1, :] *= 20
# adj = torch.ones(node, node)
# h_re = atten(x, adj, None)
# print(f"final feature size {h_re.size()}")

class degreeNN(nn.Module):
    def __init__(self, n_hidden, lr=None):
        super(degreeNN, self).__init__()

        self.n_hidden = n_hidden
        # hidden 1
        self.W = nn.Parameter(torch.zeros(size=(2, self.n_hidden)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.W0 = nn.Parameter(torch.zeros(size=(1, self.n_hidden)))
        nn.init.xavier_uniform_(self.W0.data, gain=1.414)
        self.alpha1 = 0.2
        self.hid_activation = nn.LeakyReLU(self.alpha1)
        # hidden 2
        self.W1 = nn.Parameter(torch.zeros(size=(self.n_hidden, self.n_hidden * 2)))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.W01 = nn.Parameter(torch.zeros(size=(1, self.n_hidden * 2)))
        nn.init.xavier_uniform_(self.W01.data, gain=1.414)
        self.hid_activation1 = nn.LeakyReLU(self.alpha1)
        # hidden 3
        self.W2 = nn.Parameter(torch.zeros(size=(self.n_hidden *2, self.n_hidden)))
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)
        self.W02 = nn.Parameter(torch.zeros(size=(1, self.n_hidden)))
        nn.init.xavier_uniform_(self.W02.data, gain=1.414)
        self.hid_activation2 = nn.LeakyReLU(self.alpha1)
        # output layer
        self.V = nn.Parameter(torch.zeros(size=(self.n_hidden, 1)))
        nn.init.xavier_uniform_(self.V.data, gain=1.414)
        self.V0 = nn.Parameter(torch.zeros(size=(1, 1)))
        nn.init.xavier_uniform_(self.V0.data, gain=1.414)
        self.out_activation = nn.LeakyReLU(self.alpha1)

    def forward(self, x_feat, x_deg):
        x_ = torch.cat((x_feat, x_deg), dim=1)       # [N, 2]
        # print(f"x_ shape {x_.shape}")
        hid_input = torch.mm(x_, self.W) + self.W0    # [N, hid]
        # print(f"W1 x_ shape {hid_input.shape}")     # [N, hid]
        hid_output = self.hid_activation(hid_input)
        # print(f"hidden output shape {hid_output.shape}")     # [N, hid]
        #
        #hidden 2
        hid_input1 = torch.mm(hid_output, self.W1) + self.W01  # [N, hid]
        # print(f"h1 shape {hid_input1.shape}")     # [N, hid]
        hid_output1 = self.hid_activation1(hid_input1)

        # hidden 3
        hid_input2 = torch.mm(hid_output1, self.W2) + self.W02  # [N, hid]
        # print(f"h1 shape {hid_input2.shape}")  # [N, hid]
        hid_output2 = self.hid_activation2(hid_input2)

        # out
        out_input = torch.mm(hid_output2, self.V) + self.V0    # [N, 1]
        # print(f"output: x_ shape {out_input.shape}")     # [N, 1]
        out_output = self.out_activation(out_input)
        # print(f"output shape {out_output.shape}")     # [N, 1]
        return out_output

# test
# hid = 5
# node = 3
# dnn = degreeNN(hid)
# x1 = torch.ones((node, 1))
# x2 = torch.ones((node, 1)) * 2.
# y = dnn(x1, x2)
# print(y)
class GAT_degree(nn.Module):
    def __init__(self, layer_tuple, nfeat, nhid_tuple, alpha, nheads, mergeZ, mergeState, use_cuda, device, method):
        super(GAT_degree, self).__init__()

        self.gat = GAT(layer_tuple, nfeat, nhid_tuple, alpha, nheads, mergeZ, mergeState, use_cuda, device, method)

        self.model_v = "v2"
        if self.model_v == "v1":
            self.W_e = nn.Parameter(torch.empty(1, 1))  # 大小和每个节点的feature向量一样
            nn.init.xavier_uniform_(self.W_e.data, gain=1.414)
            # degree feature
            self.W_s = nn.Parameter(torch.empty(1, 1))  # 大小和每个节点的feature向量一样
            nn.init.xavier_uniform_(self.W_s.data, gain=1.414)
        elif self.model_v == "v2":
            self.hid = 10
            self.dnn = degreeNN(self.hid)


    def forward(self, x, adj, observation, s_mat, z=None):
        h_ = self.gat(x, adj, observation, s_mat, z)
        # print(f"input  feature vector is \n {h_}")

        # logging.debug(f"number of ndoe is {nbr}, s is \n {s_mat}")
        if not isinstance(s_mat, torch.Tensor):
            s_mat = torch.Tensor(s_mat)
        d_v = s_mat.sum(1, keepdim=True).clone()
        # logging.debug(f"input vector about degree is \n {d_v}")
        # print(f"input vector about degree is \n {d_v}")

        if self.model_v == "v1":
            h_e = torch.mm(h_, abs(self.W_e))
            h_s = torch.mm(d_v, abs(self.W_s))
            result = h_e + h_s
        elif self.model_v == "v2":
            result = self.dnn(h_, d_v)


        return result

class GAT(nn.Module):
    def __init__(self, layer_tuple, nfeat, nhid_tuple, alpha, nheads, mergeZ, mergeState, use_cuda, device, method):
        """Dense version of GAT."""
        super(GAT, self).__init__()

        self.use_cuda = use_cuda
        self.device = device
        self.nfeat = nfeat
        self.nhid_tuple, self.out_nhid_tuple = nhid_tuple


        self.alpha = alpha
        self.mergeZ = mergeZ
        self.mergeState = mergeState
        self.method = method
        # k多头，有k个头就有k个层。
        # nhid 就是输出的特征大小。 因为GAT的隐藏层hid会输出特征
        # print(f"nhid is {nhid}")
        self.nlayer, self.out_layer = layer_tuple      # (atten layer, out atten layer-1, ) if outatten is 1 layer: nhead*outfeat, 1


        self.attentions = [attentions(self.nlayer, self.nfeat, self.nhid_tuple, alpha=self.alpha, concat=True, mergeZ=self.mergeZ, node_dim=self.nfeat, method=method) for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            # print(f"第{i}个layer, {str(attention)}")
            self.add_module('attention_{}'.format(i), attention)

        # 会把多头的结果拼接起来，所以是nhid * nheads， 输出n个类
        #################
        # print(f"add final attention layer")
        # self.out_att = GraphAttentionLayer(self.nhid[-1] * nheads, self.nout, alpha=self.alpha, concat=False, mergeZ=self.mergeZ, node_dim=self.nfeat)
        self.out_att = attentions(self.out_layer, self.nhid_tuple[-1]*nheads , self.out_nhid_tuple, self.alpha, False, self.mergeZ, self.nfeat, method=method)

        # seed set feature theta
        self.theta = nn.Parameter(torch.empty(1, self.nfeat))  # 大小和每个节点的feature向量一样
        nn.init.uniform_(self.theta, a=0, b=1)  # 均匀分布



        # test
        self.print_tag = "Models ---"


    def forward(self, x, adj, observation, s_mat, z=None):

        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        if not isinstance(adj, torch.Tensor):
            adj = torch.Tensor(adj)
        if self.method == "aggre_degree":
            if not isinstance(s_mat, torch.Tensor):
                s_mat = torch.Tensor(s_mat)
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

        # logging.debug(f"input x is \n {x}")
        x = torch.cat([att(x, adj, s_mat, z) for att in self.attentions], dim=1)
        # logging.debug(f"after attention layer,  x size is \n {x.size()}")
        # print(f"------ after attention layer,  x size is \n {x.size()}")
        temp = self.out_att(x, adj, s_mat, z)
        # logging.debug(f"before out elu \n {temp}")
        result = F.elu(temp)

        # logging.debug(f"after out atten,  x size is \n {result.size()}")
        # print(f"------ after out atten,  x size is \n {result.size()}")


        # result = F.log_softmax(x, dim=0)    # 不是分类问题，应该是纵向softmax
        # logging.debug(f"after log softmax, x is \n {x}")

        return result


layer = (2, 2)
features_dim = 3
hidden_dim = ((8,3,), (16, 1))
alpha = 0.2     # leakyReLU的alpha
nhead = 2
model = GAT_degree(layer, nfeat=features_dim, nhid_tuple=hidden_dim, alpha=alpha, nheads=nhead,
            mergeZ=False, mergeState=False, use_cuda=False, device=False, method="base")
# # test
graph = Graph_IM(nodes=10, edges_p=0.5)
adj_matrix = graph.adj_matrix
# # print(f"graph adj matrix {graph.adj_matrix}")
adj_matrix = torch.Tensor(adj_matrix)
xv = generate_node_feature(graph, features_dim)
# # print(f"node feature vector size {xv.size()}")     # [0-100]
xv = torch.Tensor(xv)       # torch的输入必须是tensor

s_mat = graph.adm
y = model(xv, adj_matrix, None, s_mat, None)
print(f"y size {y.size()}")
# #
# print(f"get y from GAT is {y}")     # [n, 1]
