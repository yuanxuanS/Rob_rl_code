import random
import os

import torch
import torch.nn.functional as F
from models import GAT
from layers import GraphAttentionLayer
import copy
import numpy as np
import utils
from graph import Graph_IM
from env import Environment

seed = 10
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

class GATValueNet(GAT):
    def __init__(self, node_num, nfeat, nhid, dropout, alpha, nheads, mergeZ, observe_state):
        super(GATValueNet, self).__init__(nfeat, nhid, 1, dropout, alpha, nheads, mergeZ, observe_state)  # super找到当前类继承的父类，并对父类属性进行初始化，父类这里部分的参数为上一行给的参数。子类也得到父类的成员变量，后面可以直接用，不用再定义 self.nfeat = nfeat

        # 映射为一个scalar
        self.map_W = torch.nn.Parameter(torch.empty(node_num, 1))
        torch.nn.init.xavier_uniform_(self.map_W.data, gain=1.414)

        self.print_tag = "Models --- PPO ValueNet"

    def forward(self, x_feat, adj, observation=None, z=None):
        x = copy.deepcopy(x_feat)
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        if not isinstance(adj, torch.Tensor):
            adj = torch.Tensor(adj)

        ## x, features [n, feature_size]

        if self.mergeState:
            seed_set = [idx for idx in range(len(observation[0])) if observation[0][idx] == 1]
            sdset_mask = torch.zeros([x.size()[0], x.size()[1]])
            sdset_mask[seed_set] += 1.
            sdset_mask = sdset_mask * self.theta
            x = x + sdset_mask

        x = F.dropout(x, self.dropout, training=self.training)
        # print(f"{self.print_tag} x size  {x.size()[0]}")
        # print(f"after dropout x is {x}")
        x = torch.cat([att(x, adj, z) for att in self.attentions], dim=1)
        # print(f"after concat multi attention: {x.size()}")      # nhid = 8, 拼起来是8 * 8=64
        # print(f"after multi attention concat is {x}")
        x = F.dropout(x, self.dropout, training=self.training)

        #
        x = F.elu(self.out_att(x, adj, z))
        # print(f"{self.print_tag} final x  {x}")
        # result = F.log_softmax(x, dim=1)

        result = F.log_softmax(x, dim=0)    # 不是分类问题，应该是纵向softmax

        # 映射为一个标量
        # result = self.map_W(result.T)
        result = torch.mm(result.T, self.map_W)
        return result


class GATPolicyNet(GAT):
    def __init__(self, node_num, nfeat, nhid, nout, dropout, alpha, nheads, mergeZ, observe_state):
        super(GATPolicyNet, self).__init__(nfeat, nhid, nout, dropout, alpha,
                                          nheads, mergeZ, observe_state)  # super找到当前类继承的父类，并对父类属性进行初始化，父类这里部分的参数为上一行给的参数。子类也得到父类的成员变量，后面可以直接用，不用再定义 self.nfeat = nfeat

        self.out_att_mu = self.out_att
        self.out_att_std = GraphAttentionLayer(self.nhid * nheads, self.nout, dropout=self.dropout, alpha=self.alpha, concat=False, mergeZ=False)   # 只有layer的out_feat = 节点特征维度才能融合（z size才和a相同），所以mergeZ=False

        # 映射为想要的维度， 输入 (n* 2*node_feat_dim).T
        self.mu_W = torch.nn.Parameter(torch.empty(node_num, 1))  ## W
        torch.nn.init.xavier_normal(self.mu_W.data, gain=1.414)
        self.std_W = torch.nn.Parameter(torch.empty(node_num, 1))  ## W
        torch.nn.init.xavier_uniform_(self.std_W.data, gain=1.414)

        self.print_tag = "Models --- PPO PolicyNet"

    def forward(self, x_feat, adj, observation=None, z=None, sv_x=False):
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
        #融合seed set信息
        if self.mergeState:
            seed_set = [idx for idx in range(len(observation[0])) if observation[0][idx] == 1]
            sdset_mask = torch.zeros([x.size()[0], x.size()[1]])
            sdset_mask[seed_set] += 1.
            sdset_mask = sdset_mask * self.theta
            x = x + sdset_mask
            # print(f"seed set mask is {sdset_mask}")
            # print(f"x after seed set mask is {x}")

        if sv_x:
            writeTxT(filename1, "[2] after add seed set")
            writeTxT(filename1, x)

        ## x, features [n, feature_size]
        x = F.dropout(x, self.dropout, training=self.training)

        if sv_x:
            writeTxT(filename1, "[3] after dropout")
            writeTxT(filename1, x)
        # print(f"{self.print_tag} x size  {x.size()[0]}")
        # print(f"{self.print_tag} Policy net -- forward --  after dropout x is {x}")
        x = torch.cat([att(x, adj, z) for att in self.attentions], dim=1)

        if sv_x:
            writeTxT(filename1, "[4] after cat")
            writeTxT(filename1, x)
        # print(f"after concat multi attention: {x.size()}")      # nhid = 8, 拼起来是8 * 8=64
        # print(f"{self.print_tag} Policy net -- forward -- after multi attention concat is {x}")
        x = F.dropout(x, self.dropout, training=self.training)

        if sv_x:
            writeTxT(filename1, "[5] after dropout2")
            writeTxT(filename1, x)
        # print(f"{self.print_tag} Policy net -- forward -- after dropout x is {x}")

        # 通过W映射为需要的z大小
        # print(f"{self.print_tag} before W map x {x}")
        # mu_map = self.mu_W(self.out_att_mu(x, adj, z).T)
        # std_map = self.std_W(self.out_att_std(x, adj, z).T)
        mu_map = torch.mm(self.out_att_mu(x, adj, z).T, self.mu_W)

        if sv_x:
            writeTxT(filename1, "[6] get mu_map")
            writeTxT(filename1, x)
        std_map = torch.mm(self.out_att_std(x, adj, z).T, self.std_W)

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
        return x_mu.T, x_std.T          # [2*node_feat_dim, 1]

        # mu = self.out_att_mu(x, adj, z)
        # std = self.out_att_std(x, adj, z)[0].view(1, -1)
        # x_mu = 2.0 * torch.tanh(mu)
        # x_std = F.softplus(std)
        # return mu, std



class PPOContinuousAgent:
    def __init__(self, env, lr, model_name, observe_state,
                 gamma, lmbda, eps, epochs):
        self.print_tag = "PPO Agent---"
        self.env = env
        self.graph = self.env.G
        self.actor_lr, self.critic_lr = lr      # [actor_lr, critic_lr]
        self.node_features_dims = self.env.node_feat_dimension
        self.node_features = self.env.node_features
        self.z = self.env.z
        self.observe_state = observe_state

        self.model_name = model_name
        if self.model_name == 'GAT_PPO':
            nhid = self.node_features_dims
            dropout = 0.6
            alpha = 0.2  # leakyReLU的alpha
            nhead = 1
            self.actor = GATPolicyNet(self.graph.node, self.node_features_dims, nhid, 2*self.node_features_dims, dropout, alpha, nhead, mergeZ=True, observe_state=self.observe_state)  # 从n个中随意选一个分布
            self.critic = GATValueNet(self.graph.node, self.node_features_dims, nhid,  dropout, alpha, nhead, mergeZ=True, observe_state=self.observe_state)

        # buffer
        self.memory = {}


        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.epochs = epochs
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = self.critic_lr)

    def reset(self):

        self.memory = {'states':[], 'actions':[], 'rewards':[]}
        print(f"{self.print_tag} agent reset done!")

    def act(self, observation=None):

        mu, sigma = self.actor(self.node_features, self.graph.adj_matrix, observation, z=self.z)
        action_dist = torch.distributions.Normal(mu, sigma)     # object, mu sigma维度必须相同，是该位置上的分布参数
        action_nosoftmax = action_dist.sample()  # 每个值从分布中采样
        action = F.softmax(action_nosoftmax, dim=1)      # 归一化到0-1, # tensor, 二维
        return [action_nosoftmax, action]

    def remember(self, state, action_pair, reward):

        self.memory['states'].append(state)
        self.memory['actions'].append(action_pair)
        self.memory['rewards'].append(reward)

    def update(self):
        print(f"{self.print_tag} memory is {self.memory}")
        states = self.memory['states'][0]
        if not isinstance(states, torch.Tensor):
            states = torch.from_numpy(states)
        print(f"{self.print_tag} state is {states}")

        actions_pair = self.memory['actions'][0]    # [action_nosoftmax, action]
        actions_nosoftmax, actions = actions_pair
        print(f"{self.print_tag} actions is {actions_pair} and action nosoftmax {actions_nosoftmax} after softmax{actions}")

        rewards = self.memory['rewards'][0]
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
        td_delta = td_target - self.critic(self.node_features, self.graph.adj_matrix, states, z=self.z)
        # print(f"{self.print_tag} td delta is {td_delta}")
        advantage = utils.compute_advantage(self.gamma, self.lmbda, td_delta)       # one-step，相当于就是td_delta
        # print(f"{self.print_tag} advantage is {advantage}")
        mu, std = self.actor(self.node_features, self.graph.adj_matrix, states, z=self.z)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())        # 这里计算出mu std，后面更新用到这个结果，比如更新网络，不再因为mu std更新的网络的参数
        old_log_probs = action_dists.log_prob(actions_nosoftmax)      # 该action值在分布中对应的概率的log值，还原概率加上.exp()

        for e in range(self.epochs):
            print(f"{self.print_tag} updating --- epoch {e}")
            mu, std = self.actor(self.node_features, self.graph.adj_matrix, states, z=self.z)
            print(f"{self.print_tag} updating-- mu {mu} std {std}")
            action_dists = torch.distributions.Normal(mu, std)
            # tmp_distri_action = action_dists.log_prob(actions_nosoftmax).exp()
            # print(f"{self.print_tag} updating-- action 在新分布中采样 {tmp_distri_action} ")
            log_probs = action_dists.log_prob(actions_nosoftmax)
            print(f"{self.print_tag} updating -- 仅log {log_probs} old log {old_log_probs}")
            ratio = torch.exp(log_probs - old_log_probs)
            print(f"{self.print_tag} updating -- exp-ratio {ratio}")
            # print(f"{self.print_tag} ratio is {ratio}")

            # surr1 = ratio * advantage
            # surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            # actor_loss = torch.mean(-torch.min(surr1, surr2))

            surr1_ratio = ratio
            surr2_ratio = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
            torch.autograd.set_detect_anomaly(True)
            with torch.autograd.detect_anomaly():
                actor_loss = torch.mean(-torch.min(surr1_ratio, surr2_ratio)) * advantage
                critic_loss = torch.mean(F.mse_loss(self.critic(self.node_features, self.graph.adj_matrix, states, z=self.z), td_target.detach()))
                print(f"{self.print_tag} updating --- in epoch {e} actor loss {actor_loss} critic loss {critic_loss}")

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
                print(f"{self.print_tag} updating --- after loss step W {self.actor.attentions[0].W}")
                # for name, parms in self.actor.named_parameters():
                #     print('-->name:', name)
                #     print('-->para:', parms)
                #     print('-->grad_requirs:', parms.requires_grad)
                #     print('-->grad_value:', parms.grad)
                #     print("===")


                self.critic_optimizer.step()

        return actor_loss, critic_loss


# env
T = 1
sub_T = 4
budget = 4  #
graph = Graph_IM(nodes=10, edges_p=0.5)
dimensions = 3
env = Environment(T, sub_T, budget, graph, dimensions)
env.reset()
# -- test --
nhid = dimensions       # 中间layer的输出大小必须和节点特征维度相同才能融合z
dropout = 0.6
alpha = 0.2  # leakyReLU的alpha
nhead = 1
mergeZ = True
obs_state = False

# write to txt
filename1 = '.\\test.txt'

def writeTxT(file, data):
    f = open(file, 'a')
    f.write(str(data))
    f.close()


# -- policy --
net= GATPolicyNet(env.N, env.node_feat_dimension, nhid, 2*env.node_feat_dimension, 0.6, 0.2, nhead, mergeZ, obs_state)
mu, std = net(env.node_features, env.adj_matrix, None, env.z)
print(mu, std)
#
mu1, std1 = net(env.node_features, env.adj_matrix, None, env.z)
print(mu1, std1)


action_dist = torch.distributions.Normal(mu, std)     # ??
print(action_dist)
action = action_dist.sample()  # ??
print(f" before {action}")
action = F.softmax(action, dim=1)
print(f" after {action}")
old_log_probs = action_dist.log_prob(action)
print(old_log_probs)

# value net
# vnet = GATValueNet(env.N, env.node_feat_dimension, nhid, 0.6, 0.2, nhead, mergeZ, obs_state)
# value = vnet(env.node_features, env.adj_matrix, None, env.z)
# print(value)

# PPO
actor_lr = 1e-5
critic_lr = 1e-5
lr = [actor_lr, critic_lr]

observe_state = False
gamma = 0.98
lmbda = 0.95
epochs = 500
eps = 0.2
# agent = PPOContinuousAgent(env, lr, 'GAT_PPO', observe_state, gamma, lmbda, eps, epochs)
# agent.reset()

# print(f"model structure  {agent.actor}")
# - act -
# action = net.act(None)
# print(action)
# - update -
# nature_state, _ = env.get_seed_state()
# print(f"before {env.z}")
# z_action_pair_lst = agent.act(nature_state)
# print(f"get action from agent {z_action}")
# z_new = env.step_hyper(z_action_pair_lst)
# print(f"after {env.z}")
# agent.remember(nature_state, z_action_pair_lst, 1.)
# actor_loss, critic_loss = agent.update()
# print(f"actor loss {actor_loss} critic loss {critic_loss}")
