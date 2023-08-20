from models import GAT
import torch
import numpy as np
import random
import copy
import seed
import logging
from graphviz import Digraph
from torchviz import make_dot, make_dot_from_trace
seed = seed.get_value()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class DQAgent:
    def __init__(self, graph_pool, node_feat_pool, hyper_pool, model_name,
                 main_setting,
                 node_dim,
                 init_epsilon, train_batch, update_target_steps, use_cuda, device):

        self.use_cuda = use_cuda
        self.merge_z = main_setting["observe_z"]
        self.device = device
        self.graphs = graph_pool
        self.node_feat_pool = node_feat_pool
        self.hyper_pool = hyper_pool

        self.graph = None
        self.adj = None
        self.z = None
        self.model_name = model_name

        self.node_features = None
        self.node_features_dims = node_dim
        # policy
        self.curr_epsilon = init_epsilon
        self.policy_model = None
        self.target_model = None

        # buffer
        self.memory = []
        self.buffer_max = 500       # equal to global iterations

        # train args
        self.basic_batch_size = train_batch
        self.train_batch_size = train_batch     # 训练网络需要的样本数量

        self.gamma = main_setting["gamma"]
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.copy_model_steps = update_target_steps
        self.lr = main_setting["lr"]

        self.path = None

        if self.model_name == 'GAT_QN':
            hidden_dim = main_setting["hidden_dims"]

            alpha = main_setting["alpha"]  # leakyReLU的alpha
            nhead = main_setting["nheads"]

            self.policy_model = GAT(nfeat=self.node_features_dims, nhid=hidden_dim, nout=1, alpha=alpha,
                                    nheads=nhead, mergeZ=self.merge_z, mergeState=True, use_cuda=self.use_cuda, device=self.device)
            self.target_model = GAT(nfeat=self.node_features_dims, nhid=hidden_dim, nout=1, alpha=alpha,
                                    nheads=nhead, mergeZ=self.merge_z, mergeState=True, use_cuda=self.use_cuda, device=self.device)
            if self.use_cuda:
                self.policy_model.to(self.device)
                self.target_model.to(self.device)


            self.optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=self.lr)
            
        elif self.model_name == "random":
            self.policy_model = None
            self.target_model = None



        # test
        self.print_tag = "DQN Agent---"


    def init_graph(self, g_id):
        self.graph = self.graphs[g_id]  # a graph, Graph_IM instance
        self.adj = torch.Tensor(self.graph.adj_matrix)

    def init_n_feat(self, ft_id):

        self.node_features = torch.Tensor(self.node_feat_pool[ft_id])




    def init_hyper(self, hyper_id):
        self.z = torch.Tensor(self.hyper_pool[hyper_id])

    def reset(self):
        self.iter_step = 1
        # print(f"{self.print_tag} agent reset done!")

    def forward_hook(self, module, input, output):
        logging.debug('forward')
        logging.debug(output.shape)
        # logging.debug(output[0][0][0])

    def backward_hook(self, module, grad_in, grad_out):
        logging.debug('backward')
        logging.debug(grad_in.shape)
        logging.debug(grad_in)
        logging.debug(grad_out.shape)
        logging.debug(grad_out)

    def hook(self, grad):
        logging.debug("tensor grad:", grad)

    def act(self, observation, feasible_action, mode):
        # policy
        if self.model_name == 'GAT_QN':

            if self.curr_epsilon > np.random.rand() and mode != "valid":
                action = np.random.choice(feasible_action)
                # print(f"action is {action}")
            else:
                # GAT, 输入所有节点特征， 图的结构关系-邻接矩阵，
                # node_features 融合state
                # print(f"{self.print_tag} adj_matrix is {self.graph.adj_matrix}")
                input_node_feat = copy.deepcopy(self.node_features)


                q_a = self.policy_model(input_node_feat.to(self.device), self.adj.to(self.device),
                                        torch.Tensor(observation).to(self.device), z=self.z.to(self.device))
                if self.use_cuda:
                    q_a = q_a.cpu()
                infeasible_action = [k for k in range(self.graph.node) if k not in feasible_action]
                # print(f"{self.print_tag} infeasible action is {infeasible_action}")

                q_a[infeasible_action] = -9e15

                logging.debug(f"get model params")
                for name, param in self.policy_model.named_parameters():
                    logging.debug(f"this layer: {name}, params: {param}")

                logging.debug(f"act, final q_a is {q_a}")
                action = q_a.argmax()

        elif self.model_name == "random":
            action = np.random.choice(feasible_action)

        if not isinstance(action, int):
            action = int(action)
        # ？ return action.item()
        return action

    def remember(self, sample_lst):
        '''

        :param sample_lst: [state, action, reward, next state]
        :return:
        '''
        self.memory.append(sample_lst)


    def get_sample(self):
        self.train_batch_size = int((1 + len(self.memory) / self.buffer_max) * self.basic_batch_size)

        if len(self.memory) > self.train_batch_size:

            batch = random.sample(self.memory, self.train_batch_size)
            # print(f"{self.print_tag} batch is {batch}")
            # print(f" zip is {list(zip(*batch))}")
            state_batch = list(list(zip(*batch))[0])
            # print(f"state batch is {state_batch}")
            action_batch = list(list(zip(*batch))[1])
            reward_batch = list(list(zip(*batch))[2])
            next_state_batch = list(list(zip(*batch))[3])
            feasible_batch = list(list(zip(*batch))[4])
        else:
            batch = []
        return batch
        # return state_batch, action_batch, reward_batch, next_state_batch

    def update(self, i):
        # 采样batch更新policy_model
            # 从memory中采样
        batch = self.get_sample()
        if not batch:
            # print(f"{self.print_tag} no enough sample and no update")
            return 0.
        else:
            logging.debug(f"batch length is {len(batch)}")
        # check gradients
        self.hs = []
        # for name, module in self.policy_model.named_children():
        #     logging.debug(f"add hook")
        #     h = module.register_backward_hook(self.backward_hook)
        #     self.hs.append(h)

        losses = torch.tensor(0.)
        for transition in batch:
            state, action, reward, next_state, feasible_a, done, g_id, ft_id, hyper_id = transition
            # 用目标网络计算目标值y
            graph = self.graphs[g_id]
            adj = torch.Tensor(graph.adj_matrix)
            node_feature = torch.Tensor(self.node_feat_pool[ft_id])
            hyper = torch.Tensor(self.hyper_pool[hyper_id])
            q_without_mask = self.target_model(node_feature.to(self.device), adj.to(self.device),
                                                torch.Tensor(next_state).to(self.device), z=hyper.to(self.device))
            infeasible_action = [k for k in range(self.graph.node) if k not in feasible_a]
            # print(f"{self.print_tag} infeasible action is {infeasible_action}")

            q_without_mask[infeasible_action] = -9e15

            target = reward + (1 - done) * self.gamma * q_without_mask.max()
            if not isinstance(target, torch.Tensor):
                target = torch.Tensor([target])
            # print(f"{self.print_tag} calculated target q is {target}")
            # 用行为网络计算当前值q
            q_a = self.policy_model(node_feature.to(self.device), adj.to(self.device),
                                    torch.Tensor(state).to(self.device), z=hyper.to(self.device))

            # g_b = make_dot(q_a)
            # g_b.render(
            #     filename="all q value",
            #     directory=self.path + "/logdir",
            #     format="png"
            # )
            
            q = q_a[action]
            # h = q.register_hook(self.hook)
            # self.hs.append(h)


            # g_a = make_dot(q)
            # g_a.render(
            #     filename="action selected",
            #     directory=self.path + "/logdir",
            #     format="png"
            # )

            # logging.debug(f" q , requires_grad {q.requires_grad},")
            # logging.debug(f" target , requires_grad {target.requires_grad}")

            losses  = losses + self.criterion(q, target)
        losses_a = make_dot(losses)
        losses_a.render(
            filename="losses",
            directory=self.path + "/logdir",
            format="png"
        )
        logging.debug(f"losses is {losses}")
        # loss = torch.mean(torch.tensor(losses, requires_grad=True))
        loss = losses / len(batch)
        loss_a = make_dot(loss)
        loss_a.render(
            filename="loss",
            directory=self.path + "/logdir",
            format="png"
        )
        # h = loss.register_hook(self.hook)
        # self.hs.append(h)
        # logging.debug(f" loss , requires_grad {loss.requires_grad}, grad {loss.grad}")
        # print(f"{self.print_tag} update losses are {losses} and loss is {loss}")




        # 梯度更新
        self.loss = loss
        self.optimizer.zero_grad()
        # logging.debug(f"before backward")
        # for name, param in self.policy_model.named_parameters():
        #     logging.debug(f"this layer: {name}, required grad: {param.requires_grad}, gradients: {param.grad}")

        loss.backward()
        logging.debug(f"after backward, loss grad {loss.grad}")
        # for name, param in self.policy_model.named_parameters():
        #     logging.debug(f"this layer: {name}, required grad: {param.requires_grad}, gradients: {param.grad}")
        self.optimizer.step()

        for h in self.hs:
            h.remove()

        # 每 C step，更新目标网络 = 当前的行为网络
        if i % self.copy_model_steps == 0:
            with torch.no_grad():
                self.target_model.load_state_dict(self.policy_model.state_dict())  # ？？ 是这样用吗
        return self.loss



