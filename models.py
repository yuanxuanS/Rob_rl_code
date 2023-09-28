import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import random
import numpy as np
from layers import GraphAttentionLayer, GraphAttentionLayer_struc, GraphAttentionLayer_origin
import logging
from graph import Graph_IM
from generate_node_feature import generate_node_feature
import math


class GAT_origin(nn.Module):
    def __init__(self, nfeat=1, nhid=8, nclass=2, gat_out_dim=32, node_num=8, dropout=0, alpha=0.2, nheads=8,
                 layer_type=None, args=None):
        """
        Dense version of GAT for Q learning.

        nfeat: dimension of featuers
        nhid: Number of hidden units.
        nheads: Number of head attentions.
        dropout: Dropout rate (1 - keep probability).
        alpha: Alpha for the leaky_relu.

        nclass: 没用到
        gat_out_dim: out dim of GAT
        """
        self.args = args
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.gat_out_dim = gat_out_dim
        self.node_num = node_num
        self.dropout = dropout
        self.alpha = alpha
        self.nheads = nheads
        self.layer_type = layer_type

        super().__init__()

        if self.layer_type == 'default':
            layer_nets = GraphAttentionLayer_origin
        elif self.layer_type == 'cc_m':
            layer_nets = GraphAttentionLayerCC

        self.attentions = [layer_nets(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True, args=args) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = layer_nets(nhid * nheads, gat_out_dim, dropout=dropout, alpha=alpha, concat=False, args=args)

        # q net: 展开为一维，分别映射，所以是node_num * get_out_dim
        self.q = torch.nn.Linear(node_num * gat_out_dim, node_num, bias=True)

    def forward(self, xv, adj, mask=None):
        """
        xv: batch_size x N_nodes x Dim_features
        adj: batch_size x N_nodes x N_nodes
        """
        batch_size = xv.shape[0]

        x = xv  # .view(-1, xv.shape[-1])  # TODO use [batch_size x N_nodes, Dim_features]
        adj = adj  # .view(-1, adj.shape[-1])

        outs = []
        for i in range(batch_size):
            out = [att(x[i, ...], adj[i, ...]) for att in self.attentions]
            out = torch.cat(out, dim=1)
            out = F.elu(self.out_att(out, adj[i, ...]))
            outs.append(out)

        # print(f"before stack , size {len(outs), len(outs[0][1])}, outs \n {outs}")
        outs = torch.stack(outs, dim=0)     # [bs, node_num, nheads]
        # print(f"after stack , size {outs.size()}, outs \n {outs}")

        q = self.q(outs.view(batch_size, -1))       # 如果batch_size=1, [1, ode_num, gat_out_dim] —— [1, node_num, 1]
        # print(f"after nn : {q.size()}")
        q = q[..., None]
        # print(f"q size {q.size()}")
        return q



# layer = (2, 2)
# features_dim = 3
# hidden_dim = 8
# alpha = 0.2     # leakyReLU的alpha
# nhead = 2
# node_nbr = 10
# graph = Graph_IM(nodes=node_nbr, edges_p=0.5)

# nfeat_s = node_nbr
# hidden_dim_s = ((8,3,), (16, 4))
# model = GAT_origin(features_dim, hidden_dim, 1, 2, node_nbr, 0, alpha=alpha, nheads=nhead,
#             layer_type="default")
# # # test

# adj_matrix = graph.adj_matrix
# print(f"graph adj matrix {graph.adj_matrix}")
# adj_matrix = torch.Tensor(adj_matrix)
# xv = generate_node_feature(graph, features_dim)
# xv = torch.Tensor(xv)       # torch的输入必须是tensor

# xv = torch.ones((node_nbr, features_dim))
# # # print(f"node feature vector size {xv.size()}")     # [0-100]
# xv = xv.unsqueeze(0)
# print(f"xv size {xv.size()}, xv \n {xv}")
# s_mat = graph.adm
# y = model(xv, adj_matrix)
# print(f"y size {y.size()}, y \n {y}")
# y = torch.squeeze(y, dim=0)
# print(f"y size {y.size()}, y \n {y}")

#

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

class attentions_struc(nn.Module):
    def __init__(self, layers, nfeat, nhid, nfeat_s, nhid_s, alpha, concat, mergeZ, node_dim, method="base"):
        super(attentions_struc, self).__init__()

        self.nlayer = layers
        self.nfeat = nfeat
        self.nhid_tuple = nhid
        self.nfeat_s = nfeat_s
        self.nhid_s_tuple = nhid_s
        self.alpha = alpha
        self.mergeZ = mergeZ
        self.node_dim = node_dim



        assert self.nlayer == len(self.nhid_s_tuple), "layer number not equals to hidden number"

        self.attention = []
        for i in range(self.nlayer):
            # 和GAT 区别

            # GAT多层卷积，h不变，所以输入输出维度不变
            in_dim = self.nfeat
            # h输出维度始终用传入的第一个值
            out_dim = self.nhid_tuple[0]

            out_dim_s = self.nhid_s_tuple[i]
            if i == 0:
                in_dim_s = self.nfeat_s

            else:
                in_dim_s = self.nhid_s_tuple[i - 1]
            layer = GraphAttentionLayer_struc(in_dim, out_dim, in_dim_s, out_dim_s, self.alpha, concat=concat, mergeZ=self.mergeZ,
                                        node_dim=self.node_dim, method=method)
            self.attention.append(layer)

    def forward(self, x, adj, s_mat, z=None):
        h = x
        h_struc = s_mat.clone()
        for t in range(self.nlayer):
            h_struc = self.attention[t](h, adj, h_struc, z)
            # print(f" layer {t} -- size {h.size()} ")
            # print(f"h {h}")

        return h_struc

# test
# layer = 2
# nfeat = 3
# nhid = (8,4,)
# alpha = 0.2
# merge = False
# node_dim = 1
# node = 5
# nfeat_s = node
# nhid_s = (7, node)
# atten = attentions_struc(layer, nfeat, nhid, nfeat_s, nhid_s, alpha, True, merge, node_dim, "base")
# # #
# #
# x = torch.ones(node, nfeat)
# x[1, :] *= 20
# adj = torch.ones(node, node)
# x_struc = torch.ones(node, node) * 2.1
# h_re = atten(x, adj, x_struc, None)
# print(f"final feature size {h_re.size()}")

class MLP(nn.Module):
    def __init__(self, nlayer, n_hidden, input_dim, out_dim):
        super(MLP, self).__init__()

        self.nlayer = nlayer
        self.n_hidden = n_hidden
        # hidden 1
        self.W = nn.Parameter(torch.zeros(size=(input_dim, self.n_hidden)))
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
        self.W2 = nn.Parameter(torch.zeros(size=(self.n_hidden *2, self.n_hidden * 2)))
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)
        self.W02 = nn.Parameter(torch.zeros(size=(1, self.n_hidden * 2)))
        nn.init.xavier_uniform_(self.W02.data, gain=1.414)
        self.hid_activation2 = nn.LeakyReLU(self.alpha1)

        if nlayer == 6:
            # hidden 4
            self.W3 = nn.Parameter(torch.zeros(size=(self.n_hidden * 2, self.n_hidden * 2)))
            nn.init.xavier_uniform_(self.W3.data, gain=1.414)
            self.W03 = nn.Parameter(torch.zeros(size=(1, self.n_hidden *2)))
            nn.init.xavier_uniform_(self.W03.data, gain=1.414)
            self.hid_activation3 = nn.LeakyReLU(self.alpha1)
            # hidden 5
            self.W4 = nn.Parameter(torch.zeros(size=(self.n_hidden * 2, self.n_hidden * 2)))
            nn.init.xavier_uniform_(self.W4.data, gain=1.414)
            self.W04 = nn.Parameter(torch.zeros(size=(1, self.n_hidden *2)))
            nn.init.xavier_uniform_(self.W04.data, gain=1.414)
            self.hid_activation4 = nn.LeakyReLU(self.alpha1)
            # hidden 6
            self.W5 = nn.Parameter(torch.zeros(size=(self.n_hidden * 2, self.n_hidden)))
            nn.init.xavier_uniform_(self.W5.data, gain=1.414)
            self.W05 = nn.Parameter(torch.zeros(size=(1, self.n_hidden)))
            nn.init.xavier_uniform_(self.W05.data, gain=1.414)
            self.hid_activation5 = nn.LeakyReLU(self.alpha1)

            # output layer
            self.V = nn.Parameter(torch.zeros(size=(self.n_hidden, 1)))
            nn.init.xavier_uniform_(self.V.data, gain=1.414)
        elif nlayer == 3:
            # output layer
            self.V = nn.Parameter(torch.zeros(size=(self.n_hidden * 2, out_dim)))
            nn.init.xavier_uniform_(self.V.data, gain=1.414)

        self.V0 = nn.Parameter(torch.zeros(size=(1, out_dim)))
        nn.init.xavier_uniform_(self.V0.data, gain=1.414)
        self.out_activation = nn.LeakyReLU(self.alpha1)

    def forward(self, x_feat):
        x_ = x_feat
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

        if self.nlayer == 6:
            # hidden 4
            hid_input3 = torch.mm(hid_output2, self.W3) + self.W03  # [N, hid]
            # print(f"h1 shape {hid_input2.shape}")  # [N, hid]
            hid_output3 = self.hid_activation3(hid_input3)

            # hidden 5
            hid_input4 = torch.mm(hid_output3, self.W4) + self.W04  # [N, hid]
            # print(f"h1 shape {hid_input2.shape}")  # [N, hid]
            hid_output4 = self.hid_activation4(hid_input4)

            # hidden 6
            hid_input5 = torch.mm(hid_output4, self.W5) + self.W05  # [N, hid]
            # print(f"h1 shape {hid_input2.shape}")  # [N, hid]
            hid_output5 = self.hid_activation5(hid_input5)

            # out
            out_input = torch.mm(hid_output5, self.V) + self.V0    # [N, 1]
        elif self.nlayer == 3:
            # out
            out_input = torch.mm(hid_output2, self.V) + self.V0  # [N, 1]
        # print(f"output: x_ shape {out_input.shape}")     # [N, 1]
        out_output = self.out_activation(out_input)
        # print(f"output shape {out_output.shape}")     # [N, 1]
        return out_output

# test
# hid = 5
# node = 3
# dnn = MLP(3, hid, 1, 2)
# x1 = torch.ones((node, 1))
# y = dnn(x1)
# print(y)

class aggreMLP(nn.Module):
    def __init__(self, nlayer, n_hidden, input_dim, out_dim, lr=None):
        super(aggreMLP, self).__init__()

        self.nlayer = nlayer
        self.n_hidden = n_hidden
        # hidden 1
        self.W = nn.Parameter(torch.zeros(size=(input_dim, self.n_hidden)))
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
        self.W2 = nn.Parameter(torch.zeros(size=(self.n_hidden *2, self.n_hidden * 2)))
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)
        self.W02 = nn.Parameter(torch.zeros(size=(1, self.n_hidden * 2)))
        nn.init.xavier_uniform_(self.W02.data, gain=1.414)
        self.hid_activation2 = nn.LeakyReLU(self.alpha1)

        if nlayer == 6:
            # hidden 4
            self.W3 = nn.Parameter(torch.zeros(size=(self.n_hidden * 2, self.n_hidden * 2)))
            nn.init.xavier_uniform_(self.W3.data, gain=1.414)
            self.W03 = nn.Parameter(torch.zeros(size=(1, self.n_hidden *2)))
            nn.init.xavier_uniform_(self.W03.data, gain=1.414)
            self.hid_activation3 = nn.LeakyReLU(self.alpha1)
            # hidden 5
            self.W4 = nn.Parameter(torch.zeros(size=(self.n_hidden * 2, self.n_hidden * 2)))
            nn.init.xavier_uniform_(self.W4.data, gain=1.414)
            self.W04 = nn.Parameter(torch.zeros(size=(1, self.n_hidden *2)))
            nn.init.xavier_uniform_(self.W04.data, gain=1.414)
            self.hid_activation4 = nn.LeakyReLU(self.alpha1)
            # hidden 6
            self.W5 = nn.Parameter(torch.zeros(size=(self.n_hidden * 2, self.n_hidden)))
            nn.init.xavier_uniform_(self.W5.data, gain=1.414)
            self.W05 = nn.Parameter(torch.zeros(size=(1, self.n_hidden)))
            nn.init.xavier_uniform_(self.W05.data, gain=1.414)
            self.hid_activation5 = nn.LeakyReLU(self.alpha1)

            # output layer
            self.V = nn.Parameter(torch.zeros(size=(self.n_hidden, 1)))
            nn.init.xavier_uniform_(self.V.data, gain=1.414)
        elif nlayer == 3:
            # output layer
            self.V = nn.Parameter(torch.zeros(size=(self.n_hidden * 2, out_dim)))
            nn.init.xavier_uniform_(self.V.data, gain=1.414)

        self.V0 = nn.Parameter(torch.zeros(size=(1, out_dim)))
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

        if self.nlayer == 6:
            # hidden 4
            hid_input3 = torch.mm(hid_output2, self.W3) + self.W03  # [N, hid]
            # print(f"h1 shape {hid_input2.shape}")  # [N, hid]
            hid_output3 = self.hid_activation3(hid_input3)

            # hidden 5
            hid_input4 = torch.mm(hid_output3, self.W4) + self.W04  # [N, hid]
            # print(f"h1 shape {hid_input2.shape}")  # [N, hid]
            hid_output4 = self.hid_activation4(hid_input4)

            # hidden 6
            hid_input5 = torch.mm(hid_output4, self.W5) + self.W05  # [N, hid]
            # print(f"h1 shape {hid_input2.shape}")  # [N, hid]
            hid_output5 = self.hid_activation5(hid_input5)

            # out
            out_input = torch.mm(hid_output5, self.V) + self.V0    # [N, 1]
        elif self.nlayer == 3:
            # out
            out_input = torch.mm(hid_output2, self.V) + self.V0  # [N, 1]
        # print(f"output: x_ shape {out_input.shape}")     # [N, 1]
        out_output = self.out_activation(out_input)
        # print(f"output shape {out_output.shape}")     # [N, 1]
        return out_output


# test
# hid = 5
# node = 3
# dnn = aggreMLP(hid)
# x1 = torch.ones((node, 1))
# x2 = torch.ones((node, 1)) * 2.
# y = dnn(x1, x2)
# print(y)
class GAT_degree2(nn.Module):
    def __init__(self, layer_tuple, nfeat, nhid_tuple, alpha, nheads, mergeZ, mergeState, use_cuda, device, method):
        super(GAT_degree2, self).__init__()

        self.gat = GAT(layer_tuple, nfeat, nhid_tuple, alpha, nheads, mergeZ, mergeState, use_cuda, device, method)
        # Q^
        self.pdim = nhid_tuple[1][-1]
        self.theta_6 = nn.Parameter(torch.empty(self.pdim, self.pdim))  # 大小和每个节点的feature向量一样
        nn.init.xavier_uniform_(self.theta_6.data, gain=1.414)
        self.theta_7 = nn.Parameter(torch.empty(self.pdim, self.pdim))  # 大小和每个节点的feature向量一样
        nn.init.xavier_uniform_(self.theta_7.data, gain=1.414)
        self.theta_5 = nn.Parameter(torch.empty(1, self.pdim * 2))  # 大小和每个节点的feature向量一样
        nn.init.xavier_uniform_(self.theta_5.data, gain=1.414)

    def forward(self, x, adj, observation, s_mat, z=None):
        h_ = self.gat(x, adj, observation, s_mat, z)        # [n, p]
        # print(f"input  feature vector is \n {h_}")

        # logging.debug(f"number of ndoe is {nbr}, s is \n {s_mat}")
        if not isinstance(s_mat, torch.Tensor):
            s_mat = torch.Tensor(s_mat)

        # print(f"initial vector \n{h_} \n adj is \n{adj}")
        h_neighbor_sum = torch.mm(adj, h_)      # get neighbor vector, [n, n] * [n, p]——[n, p]
        # print(f"neighbor size {h_neighbor_sum.size()} is \n{h_neighbor_sum}, ")
        ngbr_v = torch.mm(h_neighbor_sum, self.theta_6)       # [n, p]*[p, p] ——[n, p]
        # print(f"neighbor vector size {ngbr_v.size()}")

        self_v = torch.mm(h_, self.theta_7)     # [n, p]
        # print(f"self vector size {self_v.size()}")
        cat_v = torch.cat((ngbr_v, self_v), 1)      # [n, 2p]
        # print(f"cat vector size {cat_v.size()}, defore relu \n {cat_v}")

        cat_v = cat_v.clamp(0)      # relu
        # print(f"cat vector size {cat_v.size()}, after relu \n {cat_v}")
        result = torch.mm(cat_v, self.theta_5.T)        # [n, 1]
        # print(f"result size {cat_v.size()}, result \n {result}")

        return result


class GAT_degree(nn.Module):
    def __init__(self, version_n, node_nbr, layer_tuple, nfeat, nhid_tuple, nfeat_s, nhid_s_tuple, alpha, nheads, mergeZ, mergeState, use_cuda, device, method):
        super(GAT_degree, self).__init__()

        self.gat = GAT(layer_tuple, nfeat, nhid_tuple, alpha, nheads, mergeZ, mergeState, use_cuda, device, method)
        self.model_v = ""
        self.gat_struc = None
        self.ann = None
        self.mlp = None
        if version_n == "v2":
            self.model_v = "v3"
        elif version_n == "v3":
            self.model_v = "v4"
        else:
            logging.error(f"wrong version !")

        if self.model_v == "v1":
            self.W_e = nn.Parameter(torch.empty(1, 1))  # 大小和每个节点的feature向量一样
            nn.init.xavier_uniform_(self.W_e.data, gain=1.414)
            # degree feature
            self.W_s = nn.Parameter(torch.empty(1, 1))  # 大小和每个节点的feature向量一样
            nn.init.xavier_uniform_(self.W_s.data, gain=1.414)
        elif self.model_v == "v2":
            # MLP 输入 h_和d_v
            self.hid = 10
            input_dims = 2
            self.ann = aggreMLP(6, self.hid, input_dims, 1)
        elif self.model_v == "v3":
            assert nfeat_s == node_nbr, "structure GAT, number of input structure feature dims not equals to node number"

            self.gat_struc = GAT_struc(layer_tuple, nfeat, nhid_tuple, nfeat_s, nhid_s_tuple, alpha, nheads, mergeZ, mergeState, use_cuda, device, method)
            self.hid = 10
            self.input_dims = nhid_tuple[1][-1] + nhid_s_tuple[1][-1]
            self.mlp = aggreMLP(3, self.hid, self.input_dims, 1)
        elif self.model_v == "v4":
            # structure feature dim = node_nbr
            self.gat_struc = GAT(layer_tuple, node_nbr, nhid_tuple, alpha, nheads, mergeZ, mergeState, use_cuda, device, method)
            self.hid = 10
            self.input_dims = nhid_tuple[1][-1] * 2      # 两个GAT的输出维度相同
            self.mlp = aggreMLP(3, self.hid, self.input_dims, 1)

    def forward(self, x, adj, observation, s_mat, z=None):
        h_ = self.gat(x, adj, observation, s_mat, z)
        # print(f"input  feature vector is \n {h_}")

        # logging.debug(f"number of ndoe is {nbr}, s is \n {s_mat}")
        if not isinstance(s_mat, torch.Tensor):
            s_mat = torch.Tensor(s_mat)
        # logging.debug(f"input vector about degree is \n {d_v}")
        # print(f"input vector about degree is \n {d_v}")

        if self.model_v == "v1":
            h_e = torch.mm(h_, abs(self.W_e))
            d_v = s_mat.sum(1, keepdim=True).clone()

            h_s = torch.mm(d_v, abs(self.W_s))
            result = h_e + h_s
        elif self.model_v == "v2":
            d_v = s_mat.sum(1, keepdim=True).clone()

            result = self.ann(h_, d_v)
        elif self.model_v == "v3":
            # 一个GAT， 一个GAT_struc
            h_struc = self.gat_struc(x, adj, observation, s_mat, z)
            result = self.mlp(h_, h_struc)
        elif self.model_v == "v4":
            # 两个GAT相同，只是输入的h向量不同
            h_struc = self.gat_struc(s_mat, adj, observation, s_mat, z)
            result = self.mlp(h_, h_struc)

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


    def forward(self, x, adj, observation, s_mat=None, z=None):

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

        # print(f"input x size {x.size()}  \n {x}")
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

class GAT_struc(nn.Module):
    def __init__(self, layer_tuple, nfeat, nhid_tuple, nfeat_s, nhid_s_tuple, alpha, nheads, mergeZ, mergeState, use_cuda, device, method):
        super(GAT_struc, self).__init__()

        self.use_cuda = use_cuda
        self.device = device
        self.nfeat = nfeat
        self.nhid_tuple, self.out_nhid_tuple = nhid_tuple
        self.nfeat_s = nfeat_s
        self.nhid_s_tuple, self.out_nhid_s_tuple = nhid_s_tuple


        self.alpha = alpha
        self.mergeZ = mergeZ
        self.mergeState = mergeState
        self.method = method
        # k多头，有k个头就有k个层。
        # nhid 就是输出的特征大小。 因为GAT的隐藏层hid会输出特征
        # print(f"nhid is {nhid}")
        self.nlayer, self.out_layer = layer_tuple      # (atten layer, out atten layer-1, ) if outatten is 1 layer: nhead*outfeat, 1


        self.attentions_struc = [attentions_struc(self.nlayer, self.nfeat, self.nhid_tuple, self.nfeat_s, self.nhid_s_tuple, alpha=self.alpha, concat=True, mergeZ=self.mergeZ, node_dim=self.nfeat, method=method) for _ in range(nheads)]

        for i, attention in enumerate(self.attentions_struc):
            # print(f"第{i}个layer, {str(attention)}")
            self.add_module('attention_{}'.format(i), attention)

        # GAT 区别：h_struc把多头的结果拼接起来，所以是nhid * nheads; h不拼接
        #################
        # print(f"add final attention layer")
        # self.out_att = GraphAttentionLayer(self.nhid[-1] * nheads, self.nout, alpha=self.alpha, concat=False, mergeZ=self.mergeZ, node_dim=self.nfeat)
        self.out_att = attentions_struc(self.out_layer, self.nfeat, self.out_nhid_tuple, self.nhid_s_tuple[-1]*nheads, self.out_nhid_s_tuple, self.alpha, False, self.mergeZ, self.nfeat, method=method)

        # seed set feature theta
        self.theta = nn.Parameter(torch.empty(1, self.nfeat))  # 大小和每个节点的feature向量一样
        nn.init.uniform_(self.theta, a=0, b=1)  # 均匀分布



        # test
        self.print_tag = "Models ---"


    def forward(self, x, adj, observation, x_struc, z=None):

        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        if not isinstance(adj, torch.Tensor):
            adj = torch.Tensor(adj)
        if self.method == "aggre_degree":
            if not isinstance(x_struc, torch.Tensor):
                s_mat = torch.Tensor(x_struc)
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
        x_struc = torch.cat([att(x, adj, x_struc, z) for att in self.attentions_struc], dim=1)
        # logging.debug(f"after attention layer,  x size is \n {x.size()}")
        # print(f"------ after attention layer,  x struc size is \n {x_struc.size()}")
        temp = self.out_att(x, adj, x_struc, z)
        # logging.debug(f"before out elu \n {temp}")
        result = F.elu(temp)

        # logging.debug(f"after out atten,  x size is \n {result.size()}")
        # print(f"------ after out atten,  x size is \n {result.size()}")


        # result = F.log_softmax(x, dim=0)    # 不是分类问题，应该是纵向softmax
        # logging.debug(f"after log softmax, x is \n {x}")

        return result

# layer = (2, 2)
# features_dim = 3
# hidden_dim = ((8,3,), (16, 5))
# alpha = 0.2     # leakyReLU的alpha
# nhead = 2
# node_nbr = 10
# graph = Graph_IM(nodes=node_nbr, edges_p=0.5)
#
# nfeat_s = node_nbr
# hidden_dim_s = ((8,3,), (16, 4))
# model = GAT_degree("v2", node_nbr, layer, nfeat=features_dim, nhid_tuple=hidden_dim, nfeat_s=nfeat_s, nhid_s_tuple=hidden_dim_s,alpha=alpha, nheads=nhead,
#             mergeZ=False, mergeState=False, use_cuda=False, device=False, method="base")
# # # # test
#
# adj_matrix = graph.adj_matrix
# # # print(f"graph adj matrix {graph.adj_matrix}")
# adj_matrix = torch.Tensor(adj_matrix)
# xv = generate_node_feature(graph, features_dim)
# # # print(f"node feature vector size {xv.size()}")     # [0-100]
# xv = torch.Tensor(xv)       # torch的输入必须是tensor
#
# s_mat = graph.adm
# y = model(xv, adj_matrix, None, s_mat, None)
# print(f"y size {y.size()}")

# print(f"get y from GAT is {y}")     # [n, 1]

class GAT_MLP(nn.Module):
    def __init__(self, mlp_layers, nhid, layer_tuple, nfeat, nhid_tuple, alpha, nheads, mergeZ, mergeState, use_cuda, device, method):
        super(GAT_MLP, self).__init__()

        self.gat = GAT(layer_tuple, nfeat, nhid_tuple, alpha, nheads, mergeZ, mergeState, use_cuda, device, method)
        self.hid = nhid
        self.input_dims = nhid_tuple[1][-1]       
        self.mlp = MLP(mlp_layers, self.hid, self.input_dims, 1)

    def forward(self, x, adj, observation, z=None):
        h_ = self.gat(x, adj, observation, None, z)
        h = self.mlp(h_)

        return h

# layer = (2, 2)
# features_dim = 3
# hidden_dim = ((8,3,), (16, 5))
# alpha = 0.2     # leakyReLU的alpha
# nhead = 2
# node_nbr = 10
# graph = Graph_IM(nodes=node_nbr, edges_p=0.5)

# mlp_layer = 3
# mlp_hid = 10
# model = GAT_MLP(mlp_layer, mlp_hid, layer, nfeat=features_dim, nhid_tuple=hidden_dim, 
            # alpha=alpha, nheads=nhead,
            # mergeZ=False, mergeState=False, use_cuda=False, device=False, method="base")
# # # # test
#
# adj_matrix = graph.adj_matrix
# # # print(f"graph adj matrix {graph.adj_matrix}")
# adj_matrix = torch.Tensor(adj_matrix)
# xv = generate_node_feature(graph, features_dim)
# xv = torch.Tensor(xv)       # torch的输入必须是tensor
# print(f"node feature vector size {xv.size()}")     # [0-100]
#
# y = model(xv, adj_matrix, None, None)
# print(f"y size {y.size()}")

# print(f"get y from GAT is {y}")     # [n, 1]