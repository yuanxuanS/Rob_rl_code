import random
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary
from models import GAT, attentions, GAT_degree, GAT_degree2, aggreMLP, GAT_origin
from layers import GraphAttentionLayer
import copy
import numpy as np
import utils
import seed
import logging

seed = seed.get_value()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
def _init_fn(worker_id):
    np.random.seed(int(seed))

class GATValueNet_d(GAT_degree):
    def __init__(self, nnmodel, node_num, layer_tuple, nfeat, nhid_tuple, hid_s_dim_tuple, alpha, nheads, mergeZ, observe_state, use_cuda, device, method):
        super(GATValueNet_d, self).__init__(nnmodel, node_num, layer_tuple, nfeat, nhid_tuple, node_num, hid_s_dim_tuple,
                                            alpha, nheads, mergeZ, observe_state, use_cuda, device, method)  # super找到当前类继承的父类，并对父类属性进行初始化，父类这里部分的参数为上一行给的参数。子类也得到父类的成员变量，后面可以直接用，不用再定义 self.nfeat = nfeat

        # 映射为一个scalar
        self.map_W = torch.nn.Parameter(torch.empty(node_num, 1))
        torch.nn.init.xavier_uniform_(self.map_W.data, gain=1.414)

        self.print_tag = "Models --- PPO ValueNet"

    def forward(self, x_feat, adj, observation=None, s_mat=None, z=None):
        x = copy.deepcopy(x_feat)
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        if not isinstance(adj, torch.Tensor):
            adj = torch.Tensor(adj)

        if not isinstance(s_mat, torch.Tensor):
            s_mat = torch.Tensor(s_mat)
        ## x, features [n, feature_size]
        h_ = self.gat(x, adj, observation, s_mat, z)

        if self.model_v == "v3":
            # 一个GAT， 一个GAT_struc
            h_struc = self.gat_struc(x, adj, observation, s_mat, z)
            result = self.mlp(h_, h_struc)

        elif self.model_v == "v4":
            # 两个GAT相同，只是输入的h向量不同
            h_struc = self.gat_struc(s_mat, adj, observation, s_mat, z)
            result = self.mlp(h_, h_struc)

        # 映射为一个标量
        result = torch.mm(result.T, self.map_W)     # [dim, n] * [n, 1] = []
        return result

class GATPolicyNet_d(GAT_degree):
    def __init__(self, nnmodel, node_num, layer_tuple, nfeat, nhid_tuple, hid_s_dim_tuple, alpha, nheads, mergeZ, observe_state, use_cuda, device, method):
        super(GATPolicyNet_d, self).__init__(nnmodel, node_num, layer_tuple, nfeat, nhid_tuple, node_num, hid_s_dim_tuple, alpha,
                                          nheads, mergeZ, observe_state, use_cuda, device, method)  # super找到当前类继承的父类，并对父类属性进行初始化，父类这里部分的参数为上一行给的参数。子类也得到父类的成员变量，后面可以直接用，不用再定义 self.nfeat = nfeat

        self.out_mlp_mu = aggreMLP(3, self.hid, self.input_dims, 2*nfeat)

        self.out_mlp_std = aggreMLP(3, self.hid, self.input_dims, 2*nfeat)

        # 映射为想要的维度， 输入 (n* 2*node_feat_dim).T
        self.mu_W = torch.nn.Parameter(torch.empty(node_num, 1))  ## W
        torch.nn.init.xavier_normal_(self.mu_W.data, gain=1.414)
        self.std_W = torch.nn.Parameter(torch.empty(node_num, 1))  ## W
        torch.nn.init.xavier_uniform_(self.std_W.data, gain=1.414)

        self.print_tag = " PPO PolicyNet ---"

    def forward(self, x_feat, adj, observation=None, s_mat=None, z=None, sv_x=False):
        x = copy.deepcopy(x_feat)

        if sv_x:
            writeTxT(filename1, "this is a forward ----------")
            writeTxT(filename1, "[0] after copy")
            writeTxT(filename1, x)

        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)

        if sv_x:
            writeTxT(filename1, "[1] after type trans")
            writeTxT(filename1, x)

        if not isinstance(adj, torch.Tensor):
            adj = torch.Tensor(adj)


        if not isinstance(s_mat, torch.Tensor):
            s_mat = torch.Tensor(s_mat)

        h_ = self.gat(x, adj, observation, s_mat, z)
        if self.model_v == "v3":
            # 一个GAT， 一个GAT_struc
            h_struc = self.gat_struc(x, adj, observation, s_mat, z)
            result = self.mlp(h_, h_struc)
        elif self.model_v == "v4":
            # 两个GAT相同，只是输入的h向量不同
            h_struc = self.gat_struc(s_mat, adj, observation, s_mat, z)
            result = self.mlp(h_, h_struc)

        if sv_x:
            writeTxT(filename1, "[2] after add seed set")
            writeTxT(filename1, x)

        ## x, features [n, feature_size]

        if sv_x:
            writeTxT(filename1, "[3] after dropout")
            writeTxT(filename1, x)


        if sv_x:
            writeTxT(filename1, "[4] after cat")
            writeTxT(filename1, x)


        if sv_x:
            writeTxT(filename1, "[5] after dropout2")
            writeTxT(filename1, x)
        # print(f"{self.print_tag} Policy net -- forward -- after dropout x is {x}")

        # 通过W映射为需要的z大小

        mu_map = torch.mm(result.T, self.mu_W)

        # if sv_x:
        #     writeTxT(filename1, "[6] get mu_map")
        #     writeTxT(filename1, x)
        std_map = torch.mm(result.T, self.std_W)

        # if sv_x:
        #     writeTxT(filename1, "[7] get std_map")
        #     writeTxT(filename1, x)
        # print(f"{self.print_tag} after W map mu {mu_map} std {std_map}")
        x_mu = 2.0 * torch.tanh(mu_map)

        # if sv_x:
        #     writeTxT(filename1, "[8] get x_mu")
        #     writeTxT(filename1, x)
        x_std = F.softplus(std_map)    # >0, relu的平滑形式

        # if sv_x:
        #     writeTxT(filename1, "[9] get x_std")
        #     writeTxT(filename1, x)
        # print(f"{self.print_tag} after tanh mu {x_mu} softplus std {x_std}")

        # print(f"{self.print_tag} -- sigma \n\t\t\t{x_std}")
        return x_mu.T, x_std.T          # [2*node_feat_dim, 1]

        # mu = self.out_att_mu(x, adj, z)
        # std = self.out_att_std(x, adj, z)[0].view(1, -1)
        # x_mu = 2.0 * torch.tanh(mu)
        # x_std = F.softplus(std)
        # return mu, std

class GATValueNet(GAT):
    def __init__(self, node_num, layer_tuple, nfeat, nhid_tuple, alpha, nheads, mergeZ, observe_state, use_cuda, device, method):
        super(GATValueNet, self).__init__(layer_tuple, nfeat, nhid_tuple, alpha,
                                          nheads, mergeZ, observe_state, use_cuda, device, method)  # super找到当前类继承的父类，并对父类属性进行初始化，父类这里部分的参数为上一行给的参数。子类也得到父类的成员变量，后面可以直接用，不用再定义 self.nfeat = nfeat

        # 映射为一个scalar
        self.map_W = torch.nn.Parameter(torch.empty(node_num, 1))
        torch.nn.init.xavier_uniform_(self.map_W.data, gain=1.414)

        self.print_tag = "Models --- PPO ValueNet"

    def forward(self, x_feat, adj, observation=None, s_mat=None, z=None):
        x = copy.deepcopy(x_feat)
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        if not isinstance(adj, torch.Tensor):
            adj = torch.Tensor(adj)
        if self.method == "aggre_degree":
            if not isinstance(s_mat, torch.Tensor):
                s_mat = torch.Tensor(s_mat)
        ## x, features [n, feature_size]

        if self.mergeState:
            seed_set = [idx for idx in range(len(observation[0])) if observation[0][idx] == 1]
            sdset_mask = torch.zeros([x.size()[0], x.size()[1]])
            sdset_mask[seed_set] += 1.
            if self.use_cuda:
                sdset_mask = sdset_mask.to(self.device)
            sdset_mask = sdset_mask * self.theta
            x = x + sdset_mask


        x = torch.cat([att(x, adj, s_mat, z) for att in self.attentions], dim=1)
        #
        x = F.elu(self.out_att(x, adj, s_mat, z))

        result = F.log_softmax(x, dim=0)    # 不是分类问题，应该是纵向softmax

        # 映射为一个标量
        result = torch.mm(result.T, self.map_W)
        return result


class GATPolicyNet(GAT):
    def __init__(self, node_num, layer_tuple, nfeat, nhid_tuple, alpha, nheads, mergeZ, observe_state, use_cuda, device, method):
        super(GATPolicyNet, self).__init__(layer_tuple, nfeat, nhid_tuple, alpha,
                                          nheads, mergeZ, observe_state, use_cuda, device, method)  # super找到当前类继承的父类，并对父类属性进行初始化，父类这里部分的参数为上一行给的参数。子类也得到父类的成员变量，后面可以直接用，不用再定义 self.nfeat = nfeat

        self.out_att_mu = self.out_att

        self.out_att_std = attentions(self.out_layer, self.nhid_tuple[-1]*nheads , self.out_nhid_tuple, self.alpha, False, self.mergeZ, node_num, method)

        # 映射为想要的维度， 输入 (n, 2*node_feat_dim).T
        self.mu_W = torch.nn.Parameter(torch.empty(node_num, 1))  ## W
        torch.nn.init.xavier_normal_(self.mu_W.data, gain=1.414)
        self.std_W = torch.nn.Parameter(torch.empty(node_num, 1))  ## W
        torch.nn.init.xavier_uniform_(self.std_W.data, gain=1.414)

        self.print_tag = " PPO PolicyNet ---"

    def forward(self, x_feat, adj, observation=None, s_mat=None, z=None, sv_x=False):
        x = copy.deepcopy(x_feat)

        if sv_x:
            writeTxT(filename1, "this is a forward ----------")
            writeTxT(filename1, "[0] after copy")
            writeTxT(filename1, x)

        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)

        if sv_x:
            writeTxT(filename1, "[1] after type trans")
            writeTxT(filename1, x)

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
            sdset_mask = sdset_mask * self.theta
            x = x + sdset_mask
            # print(f"seed set mask is {sdset_mask}")
            # print(f"x after seed set mask is {x}")

        if sv_x:
            writeTxT(filename1, "[2] after add seed set")
            writeTxT(filename1, x)

        ## x, features [n, feature_size]

        if sv_x:
            writeTxT(filename1, "[3] after dropout")
            writeTxT(filename1, x)

        x = torch.cat([att(x, adj, s_mat, z) for att in self.attentions], dim=1)

        if sv_x:
            writeTxT(filename1, "[4] after cat")
            writeTxT(filename1, x)


        if sv_x:
            writeTxT(filename1, "[5] after dropout2")
            writeTxT(filename1, x)
        # print(f"{self.print_tag} Policy net -- forward -- after dropout x is {x}")

        # 通过W映射为需要的z大小
        # print(f"{self.print_tag} before W map x {x}")
        # mu_map = self.mu_W(self.out_att_mu(x, adj, z).T)
        # std_map = self.std_W(self.out_att_std(x, adj, z).T)
        mu_map = torch.mm(self.out_att_mu(x, adj, s_mat, z).T, self.mu_W)

        if sv_x:
            writeTxT(filename1, "[6] get mu_map")
            writeTxT(filename1, x)
        std_map = torch.mm(self.out_att_std(x, adj, s_mat, z).T, self.std_W)

        if sv_x:
            writeTxT(filename1, "[7] get std_map")
            writeTxT(filename1, x)
        # print(f"{self.print_tag} after W map mu {mu_map} std {std_map}")
        x_mu = 2.0 * torch.tanh(mu_map)

        if sv_x:
            writeTxT(filename1, "[8] get x_mu")
            writeTxT(filename1, x)
        x_std = F.softplus(std_map)    # >0, relu的平滑形式

        if sv_x:
            writeTxT(filename1, "[9] get x_std")
            writeTxT(filename1, x)
        # print(f"{self.print_tag} after tanh mu {x_mu} softplus std {x_std}")

        # print(f"{self.print_tag} -- sigma \n\t\t\t{x_std}")
        return x_mu.T, x_std.T          # [2*node_feat_dim, 1]

        # mu = self.out_att_mu(x, adj, z)
        # std = self.out_att_std(x, adj, z)[0].view(1, -1)
        # x_mu = 2.0 * torch.tanh(mu)
        # x_std = F.softplus(std)
        # return mu, std

class GATValueNet_d2(GAT_degree2):
    def __init__(self, node_num, layer_tuple, nfeat, nhid_tuple, alpha, nheads, mergeZ, observe_state, use_cuda, device, method):
        super(GATValueNet_d2, self).__init__(layer_tuple, nfeat, nhid_tuple, alpha,
                                          nheads, mergeZ, observe_state, use_cuda, device, method)  # super找到当前类继承的父类，并对父类属性进行初始化，父类这里部分的参数为上一行给的参数。子类也得到父类的成员变量，后面可以直接用，不用再定义 self.nfeat = nfeat

        # 映射为一个scalar
        self.map_W = torch.nn.Parameter(torch.empty(node_num, 1))
        torch.nn.init.xavier_uniform_(self.map_W.data, gain=1.414)

        self.print_tag = "Models --- PPO ValueNet"

    def forward(self, x_feat, adj, observation=None, s_mat=None, z=None):
        x = copy.deepcopy(x_feat)
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        if not isinstance(adj, torch.Tensor):
            adj = torch.Tensor(adj)
        if not isinstance(s_mat, torch.Tensor):
            s_mat = torch.Tensor(s_mat)
        ## x, features [n, feature_size]

        h_ = self.gat(x, adj, observation, s_mat, z)

        # print(f"initial vector \n{h_} \n adj is \n{adj}")
        h_neighbor_sum = torch.mm(adj, h_)  # get neighbor vector, [n, n] * [n, p]——[n, p]
        # print(f"neighbor size {h_neighbor_sum.size()} is \n{h_neighbor_sum}, ")
        ngbr_v = torch.mm(h_neighbor_sum, self.theta_6)  # [n, p]*[p, p] ——[n, p]
        # print(f"neighbor vector size {ngbr_v.size()}")

        self_v = torch.mm(h_, self.theta_7)  # [n, p]
        # print(f"self vector size {self_v.size()}")
        cat_v = torch.cat((ngbr_v, self_v), 1)  # [n, 2p]
        # print(f"cat vector size {cat_v.size()}, defore relu \n {cat_v}")

        cat_v = cat_v.clamp(0)  # relu
        # print(f"cat vector size {cat_v.size()}, after relu \n {cat_v}")
        result = torch.mm(cat_v, self.theta_5.T)  # [n, 1]
        # print(f"result size {cat_v.size()}, result \n {result}")
        # 映射为一个标量
        result = torch.mm(result.T, self.map_W)  # [1, n] * [n, 1] = []
        return result

class GATPolicyNet_d2(GAT_degree2):
    def __init__(self, node_num, layer_tuple, nfeat, nhid_tuple, alpha, nheads, mergeZ, observe_state, use_cuda, device, method):
        super(GATPolicyNet_d2, self).__init__(layer_tuple, nfeat, nhid_tuple, alpha,
                                          nheads, mergeZ, observe_state, use_cuda, device, method)  # super找到当前类继承的父类，并对父类属性进行初始化，父类这里部分的参数为上一行给的参数。子类也得到父类的成员变量，后面可以直接用，不用再定义 self.nfeat = nfeat

        self.theta_5 = nn.Parameter(torch.empty(2*nfeat, self.pdim * 2))  # 大小和每个节点的feature向量一样
        nn.init.xavier_uniform_(self.theta_5.data, gain=1.414)

        # 映射为想要的维度， 输入 (n, 2*node_feat_dim).T
        self.mu_W = torch.nn.Parameter(torch.empty(node_num, 1))  ## W
        torch.nn.init.xavier_normal_(self.mu_W.data, gain=1.414)
        self.std_W = torch.nn.Parameter(torch.empty(node_num, 1))  ## W
        torch.nn.init.xavier_uniform_(self.std_W.data, gain=1.414)

        self.print_tag = " PPO PolicyNet ---"

    def forward(self, x_feat, adj, observation=None, s_mat=None, z=None, sv_x=False):
        x = copy.deepcopy(x_feat)

        if sv_x:
            writeTxT(filename1, "this is a forward ----------")
            writeTxT(filename1, "[0] after copy")
            writeTxT(filename1, x)

        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)

        if sv_x:
            writeTxT(filename1, "[1] after type trans")
            writeTxT(filename1, x)

        if not isinstance(adj, torch.Tensor):
            adj = torch.Tensor(adj)

        if not isinstance(s_mat, torch.Tensor):
            s_mat = torch.Tensor(s_mat)


        if sv_x:
            writeTxT(filename1, "[2] after add seed set")
            writeTxT(filename1, x)

        ## x, features [n, feature_size]

        if sv_x:
            writeTxT(filename1, "[3] after dropout")
            writeTxT(filename1, x)


        if sv_x:
            writeTxT(filename1, "[4] after cat")
            writeTxT(filename1, x)


        if sv_x:
            writeTxT(filename1, "[5] after dropout2")
            writeTxT(filename1, x)
        # print(f"{self.print_tag} Policy net -- forward -- after dropout x is {x}")

        # 通过W映射为需要的z大小
        h_ = self.gat(x, adj, observation, s_mat, z)        # [n, p]
        # print(f"initial vector \n{h_} \n adj is \n{adj}")
        h_neighbor_sum = torch.mm(adj, h_)  # get neighbor vector, [n, n] * [n, p]——[n, p]
        # print(f"neighbor size {h_neighbor_sum.size()} is \n{h_neighbor_sum}, ")
        ngbr_v = torch.mm(h_neighbor_sum, self.theta_6)  # [n, p]*[p, p] ——[n, p]
        # print(f"neighbor vector size {ngbr_v.size()}")

        self_v = torch.mm(h_, self.theta_7)  # [n, p]
        # print(f"self vector size {self_v.size()}")
        cat_v = torch.cat((ngbr_v, self_v), 1)  # [n, 2p]
        # print(f"cat vector size {cat_v.size()}, defore relu \n {cat_v}")

        cat_v = cat_v.clamp(0)  # relu
        # print(f"cat vector size {cat_v.size()}, after relu \n {cat_v}")
        result = torch.mm(cat_v, self.theta_5.T)  # [n, 2*feat]
        mu_map = torch.mm(result.T, self.mu_W)

        if sv_x:
            writeTxT(filename1, "[6] get mu_map")
            writeTxT(filename1, x)
        std_map = torch.mm(result.T, self.std_W)

        if sv_x:
            writeTxT(filename1, "[7] get std_map")
            writeTxT(filename1, x)
        # print(f"{self.print_tag} after W map mu {mu_map} std {std_map}")
        x_mu = 2.0 * torch.tanh(mu_map)

        if sv_x:
            writeTxT(filename1, "[8] get x_mu")
            writeTxT(filename1, x)
        x_std = F.softplus(std_map)    # >0, relu的平滑形式

        if sv_x:
            writeTxT(filename1, "[9] get x_std")
            writeTxT(filename1, x)
        # print(f"{self.print_tag} after tanh mu {x_mu} softplus std {x_std}")

        # print(f"{self.print_tag} -- sigma \n\t\t\t{x_std}")
        return x_mu.T, x_std.T          # [2*node_feat_dim, 1]

        # mu = self.out_att_mu(x, adj, z)
        # std = self.out_att_std(x, adj, z)[0].view(1, -1)
        # x_mu = 2.0 * torch.tanh(mu)
        # x_std = F.softplus(std)
        # return mu, std




class PPOContinuousAgent:
    def __init__(self, graph_pool, node_feature_pool, hyper_pool,
                 nature_setting,
                 node_nbr, node_dim,
                 lmbda, eps, epochs, use_cuda, device):
        self.print_tag = "PPO Agent---"
        self.use_cuda = use_cuda
        self.merge_z = nature_setting["canObserve_hyper"]
        self.device = device

        # necessary env info
        self.graphs = graph_pool
        self.node_nbr = node_nbr
        self.node_features_pool = node_feature_pool
        self.hyper_pool = hyper_pool

        self.actor_lr = nature_setting["actor_lr"]
        self.critic_lr = nature_setting["critic_lr"]

        self.node_features = None
        self.node_features_dims = node_dim
        self.z = None
        self.observe_state = nature_setting["canObserve_state"]

        self.policy_dis = nature_setting["PolicyDisName"]  # “Beta” or "Gauss"
        self.norm_name = nature_setting["PolicyNormName"]  # norm method when Gaussian distribution, "sigmoid" or "softmax"
        self.model_name = nature_setting["agent_method"]
        self.method = nature_setting["GAT_mtd"]

        self.graph = None
        self.adj = None


        # buffer
        self.memory = {'states':[], 'actions':[], 'rewards':[]}

        self.gamma = nature_setting["gamma"]
        self.lmbda = lmbda
        self.eps = eps
        self.epochs = epochs

        if self.model_name == 'GAT_PPO':

            alpha = nature_setting["alpha"]  # leakyReLU的alpha
            nhead = nature_setting["nheads"]
            layer_tp = (nature_setting["GAT_atten_layer"], nature_setting["GAT_out_atten_layer"])
            hid_dim_tp = (nature_setting["GAT_hid_dim"], nature_setting["GAT_out_hid_dim"])
            hid_s_dim_tp = (nature_setting["GAT_s_hid_dim"], nature_setting["GAT_s_out_hid_dim"])

            self.nnmodel = nature_setting["nnVersion"]
            if self.nnmodel == "v2" or self.nnmodel == "v3":
                logging.debug(f"nnmodel {self.nnmodel}")

                self.critic = GATValueNet_d(self.nnmodel, self.node_nbr, layer_tp, self.node_features_dims, hid_dim_tp, hid_s_dim_tp,
                                          alpha,
                                          nhead, mergeZ=self.merge_z,
                                          observe_state=self.observe_state, use_cuda=self.use_cuda, device=self.device,
                                          method=self.method)


                self.actor = GATPolicyNet_d(self.nnmodel, self.node_nbr, layer_tp, self.node_features_dims, hid_dim_tp, hid_s_dim_tp,
                                          alpha,
                                          nhead, mergeZ=self.merge_z,
                                          observe_state=self.observe_state, use_cuda=self.use_cuda, device=self.device,
                                          method=self.method)  # 从n个中随意选一个分布
            elif self.nnmodel == "v1":
                self.critic = GATValueNet_d2(self.node_nbr, layer_tp, self.node_features_dims, hid_dim_tp,
                                            alpha,
                                            nhead, mergeZ=self.merge_z,
                                            observe_state=self.observe_state, use_cuda=self.use_cuda,
                                            device=self.device,
                                            method=self.method)

                self.actor = GATPolicyNet_d2(self.node_nbr, layer_tp, self.node_features_dims, hid_dim_tp,
                                            alpha,
                                            nhead, mergeZ=self.merge_z,
                                            observe_state=self.observe_state, use_cuda=self.use_cuda,
                                            device=self.device,
                                            method=self.method)  # 从n个中随意选一个分布
            elif (self.nnmodel == "v4") or (self.nnmodel == "v01") or (self.nnmodel == "v0")or (self.nnmodel == "v5"):

                self.critic = GATValueNet(self.node_nbr, layer_tp, self.node_features_dims + self.node_nbr, hid_dim_tp, alpha,
                                          nhead, mergeZ=self.merge_z,
                                          observe_state=self.observe_state, use_cuda=self.use_cuda, device=self.device, method=self.method)

                hidden_dim_list = [list(inner_tuple) for inner_tuple in hid_dim_tp]
                out_atten_list = hidden_dim_list[-1]
                # Modify the list
                out_atten_list[-1] = 2 * self.node_features_dims

                # convert to tuple
                hid_dim_tp2 = (tuple(hidden_dim_list[0]), tuple(out_atten_list))

                assert hid_dim_tp2[-1][-1] == 2 * self.node_features_dims, "out feature of nature policy model must be 2xnode feat"
                self.actor = GATPolicyNet(self.node_nbr, layer_tp, self.node_features_dims + self.node_nbr, hid_dim_tp2, alpha,
                                          nhead, mergeZ=self.merge_z, observe_state=self.observe_state, use_cuda=self.use_cuda, device=self.device, method=self.method)  # 从n个中随意选一个分布


            
            if self.use_cuda:
                self.actor.to(self.device)
                self.critic.to(self.device)

            print("PPO- actor architecture")
            # summary(self.actor, ((self.node_nbr, self.node_features_dims), (self.node_nbr, self.node_nbr)))
            # print(self.actor)
            print("PPO- critic architecture")
            # summary(self.critic, ((self.node_nbr, self.node_features_dims), (self.node_nbr, self.node_nbr)))
            # print(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    def init_graph(self, g_id):

        self.graph = self.graphs[g_id]
        self.adj = torch.Tensor(self.graph.adj_matrix)
        self.s_mat = self.graph.adm

    def init_n_feat(self, ft_id):
        self.node_features = torch.Tensor(self.node_features_pool[ft_id])


    def init_hyper(self, hyper_id):
        self.z = torch.Tensor(self.hyper_pool[hyper_id])

    def reset(self):


        # print(f"{self.print_tag} agent reset done!")
        pass
    def act(self, observation=None):
        if not isinstance(self.s_mat, torch.Tensor):
            self.s_mat = torch.Tensor(self.s_mat)

        if not isinstance(self.node_features, torch.Tensor):
            self.node_features = torch.Tensor(self.node_features)

        input_node_feat = copy.deepcopy(self.node_features)
        if self.nnmodel == "v4":
            input_node_feat = torch.concat((input_node_feat, self.s_mat), 1)

        mu, sigma = self.actor(input_node_feat.to(self.device), self.adj.to(self.device),
                               torch.Tensor(observation).to(self.device), self.s_mat, z=self.z.to(self.device))
        if self.use_cuda:
            mu = mu.cpu()
            sigma = sigma.cpu()
        if self.policy_dis == "Gauss":
            sigma = torch.sigmoid(sigma)        # Gauss std bounds to 1
            action_dist = torch.distributions.Normal(mu, sigma)     # object, mu sigma维度必须相同，是该位置上的分布参数
            action_nosoftmax = action_dist.sample()  # 每个值从分布中采样
            action = self.norm(action_nosoftmax)        # norm to 0-1
        elif self.policy_dis == "Beta":
            alpha = F.softplus(mu) + 1.
            beta = F.softplus(sigma) + 1.       # alpha and beta need to be larger than 1
            action_dist = torch.distributions.beta.Beta(alpha, beta)      # alpha, beta
            action_nosoftmax = action_dist.sample()
            action = action_nosoftmax
        else:
            # print(f"{self.print_tag} wrong dis str")
            pass
        # print(f"{self.print_tag} -- action before \n\t\t{action_nosoftmax}\n after norm {action}")
        return [action_nosoftmax, action]

    def norm(self, data):
        if self.norm_name == "softmax":
            withNorm = F.softmax(data, dim=1)      # 归一化到0-1, # tensor, 二维
        elif self.norm_name == "sigmoid":
            withNorm = torch.sigmoid(data)
        return withNorm
    def remember(self, state, action_pair, reward):

        self.memory['states'].append(state)
        self.memory['actions'].append(action_pair)
        self.memory['rewards'].append(reward)

    def update(self):
        # print(f"{self.print_tag} memory is {self.memory}")
        states = self.memory['states'][-1]
        if not isinstance(states, torch.Tensor):
            states = torch.from_numpy(states)
        print(f"{self.print_tag} state is {states}")

        actions_pair = self.memory['actions'][-1]    # [action_nosoftmax, action]
        actions_nosoftmax, actions = actions_pair
        print(f"{self.print_tag} actions is {actions_pair} and action nosoftmax {actions_nosoftmax} after softmax{actions}")

        rewards = self.memory['rewards'][-1]
        # rewards = (reward_ + 8.0) / 8.0
        if not isinstance(rewards, torch.Tensor):
            if isinstance(rewards, np.ndarray):
                rewards = torch.from_numpy(rewards)
            elif isinstance(rewards, float):
                rewards  = torch.FloatTensor([rewards])
        print(f"{self.print_tag} rewards is {rewards}")


        # 时序差分target

        # td_target = rewards + self.gamma * self.critic(self.node_features, self.graph.adj_matrix, next_states, add_state=False)
        td_target = rewards
        td_target = td_target.to(self.device)
        states = states.to(self.device)
        self.z = self.z.to(self.device)

        if not isinstance(self.node_features, torch.Tensor):
            self.node_features = torch.Tensor(self.node_features)

        input_node_feat = copy.deepcopy(self.node_features)
        if self.nnmodel == "v4":
            input_node_feat = torch.concat((input_node_feat, self.s_mat), 1)

        td_delta = td_target - self.critic(input_node_feat.to(self.device), self.adj.to(self.device),
                                           states, self.s_mat, z=self.z)
        if self.use_cuda:
            td_delta = td_delta.cpu()
        print(f"{self.print_tag} td delta is {td_delta}")
        advantage = utils.compute_advantage(self.gamma, self.lmbda, td_delta)       # one-step，相当于就是td_delta
        print(f"{self.print_tag} advantage is {advantage}")


        mu, std = self.actor(input_node_feat.to(self.device), self.adj.to(self.device),
                             states.to(self.device), self.s_mat, z=self.z.to(self.device))
        if self.use_cuda:
            mu = mu.cpu()
            std = std.cpu()

        if self.policy_dis == "Gauss":
            std = torch.sigmoid(std)
            action_dists = torch.distributions.Normal(mu.detach(), std.detach())        # 这里计算出mu std，后面更新用到这个结果，比如更新网络，不再因为mu std更新的网络的参数
            old_log_probs = action_dists.log_prob(actions_nosoftmax)      # 该action值在分布中对应的概率的log值，还原概率加上.exp()
        elif self.policy_dis == "Beta":
            alpha = F.softplus(mu) + 1.
            beta = F.softplus(std) + 1.  # alpha and beta need to be larger than 1
            action_dists = torch.distributions.beta.Beta(alpha.detach(), beta.detach())
            old_log_probs = action_dists.log_prob(actions_nosoftmax)

        for e in range(self.epochs):
            # print(f"{self.print_tag} updating --- epoch {e}")
            mu, std = self.actor(input_node_feat.to(self.device), self.adj.to(self.device),
                                 states.to(self.device), self.s_mat, z=self.z.to(self.device))
            if self.use_cuda:
                mu = mu.cpu()
                std = std.cpu()
            # print(f"{self.print_tag} updating-- mu {mu} std {std}")
            if self.policy_dis == "Gauss":
                std = torch.sigmoid(std)
                action_dists = torch.distributions.Normal(mu, std)
                log_probs = action_dists.log_prob(actions_nosoftmax)
            elif self.policy_dis == "Beta":
                alpha = F.softplus(mu) + 1.
                beta = F.softplus(std) + 1.  # alpha and beta need to be larger than 1
                action_dists = torch.distributions.beta.Beta(alpha, beta)
                log_probs = action_dists.log_prob(actions_nosoftmax)
            else:
                # print(f"{self.print_tag} wrong dis str")
                pass
            # print(f"{self.print_tag} updating -- log probs is {log_probs}")
            ratio = torch.exp(log_probs - old_log_probs)
            # print(f"{self.print_tag} updating -- exp-ratio {ratio}")


            # surr1 = ratio * advantage
            # surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            # actor_loss = torch.mean(-torch.min(surr1, surr2))

            surr1_ratio = ratio
            surr2_ratio = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
            torch.autograd.set_detect_anomaly(True)
            # with torch.autograd.detect_anomaly():
            print(f"{self.print_tag} updating -- surr1 -ratio {surr1_ratio}")
            print(f"{self.print_tag} updating -- surr2 -ratio {surr2_ratio}")

            actor_loss = torch.mean(-torch.min(surr1_ratio, surr2_ratio) * advantage)
            critic_loss = torch.mean(F.mse_loss(self.critic(input_node_feat.to(self.device),
                                                            self.adj.to(self.device),
                                                            states.to(self.device), self.s_mat,
                                                            z=self.z.to(self.device)), td_target.detach()))
            # print(f"{self.print_tag} updating --- in epoch {e} actor loss {actor_loss} critic loss {critic_loss}")

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            # for name, parms in self.actor.named_parameters():
            #     print('-->name:', name)
            #     print('-->para:', parms)
            #     print('-->grad_requirs:', parms.requires_grad)
            #     print('-->grad_value:', parms.grad)
            #     print("===")

            actor_loss.backward()
            # for name, parms in self.actor.named_parameters():
            #     if parms.grad is not None and torch.isnan(parms.grad).any():
            #         print(f"nan found")
            #         print('-->name:', name)
            #         print('-->grad_value:', parms.grad)
            #         print("===")
            #         raise SystemExit
            critic_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.step()

        return actor_loss, critic_loss


# # env
# T = 1
# sub_T = 4
# budget = 4  #
# graph = Graph_IM(nodes=10, edges_p=0.5)
# dimensions = 3
# env = Environment(T, budget, graph, dimensions)
# env.reset()
# # -- test --
# nhid = dimensions       # 中间layer的输出大小必须和节点特征维度相同才能融合z
# alpha = 0.2  # leakyReLU的alpha
# nhead = 1
# mergeZ = False
# obs_state = False

# # write to txt
# filename1 = '.\\test.txt'

def writeTxT(file, data):
    f = open(file, 'a')
    f.write(str(data))
    f.close()


# -- policy test and debug --
# net= GATPolicyNet(env.N, env.node_feat_dimension, nhid, 2*env.node_feat_dimension, 0.2, nhead, mergeZ, obs_state)
# mu, std = net(env.node_features, env.adj_matrix, None, env.z)
# print(mu, std)
# #
# mu1, std1 = net(env.node_features, env.adj_matrix, None, env.z)
# print(mu1, std1)
#
#
# action_dist = torch.distributions.Normal(mu, std)     # ??
# print(action_dist)
# action = action_dist.sample()  # ??
# print(f" before {action}")
# action = F.softmax(action, dim=1)
# print(f" after {action}")
# old_log_probs = action_dist.log_prob(action)
# print(old_log_probs)

# value net
# vnet = GATValueNet(env.N, env.node_feat_dimension, nhid, 0.6, 0.2, nhead, mergeZ, obs_state)
# value = vnet(env.node_features, env.adj_matrix, None, env.z)
# print(value)

# PPO
# actor_lr = 1e-3
# critic_lr = 1e-3
# lr = [actor_lr, critic_lr]
#
# norm = "sigmoid"
# dis = "Gauss"       #"Beta"
# observe_state = False
# gamma = 0.98
# lmbda = 0.95
# epochs = 50
# eps = 0.2
# agent = PPOContinuousAgent(env, lr, 'GAT_PPO', dis, norm, observe_state, gamma, lmbda, eps, epochs)
# agent.reset()

# print(f"model structure  {agent.actor}")
# - act -
# action = net.act(None)
# print(action)
# # - update -
# nature_state, _ = env.get_seed_state()
# print(f"before {env.z}")
# z_action_pair_lst = agent.act(nature_state)
# # print(f"get action from agent {z_action}")
# z_new = env.step_hyper(z_action_pair_lst)
# # print(f"after {env.z}")
# agent.remember(nature_state, z_action_pair_lst, 1.)
# actor_loss, critic_loss = agent.update()
# print(f"actor loss {actor_loss} critic loss {critic_loss}")
