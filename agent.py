from models import GAT, GAT_degree, GAT_degree2, GAT_origin, GAT_MLP
import torch
import numpy as np
import random
import copy
import seed
import logging
from graphviz import Digraph
from torchviz import make_dot, make_dot_from_trace
from torch.profiler import profile, record_function, ProfilerActivity
from utils import test_memory
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
    def __init__(self, graph_pool, nodes, node_feat_pool, hyper_pool, model_name,
                 main_setting,
                 node_dim,
                 init_epsilon, train_batch, update_target_steps, use_cuda, device):

        self.use_cuda = use_cuda
        self.merge_z = main_setting["observe_z"]
        self.device = device
        self.graphs = graph_pool
        self.node_feat_pool = node_feat_pool
        self.hyper_pool = hyper_pool

        self.nodes = nodes
        self.graph = None
        self.adj = None
        self.z = None
        self.model_name = model_name

        self.node_features = None
        self.node_features_dims = node_dim
        # policy
        self.algor = main_setting["rl_algor"]
        self.method = main_setting["GAT_mtd"]
        self.curr_epsilon = init_epsilon
        self.policy_model = None
        self.target_model = None

        # buffer
        self.memory = []
        self.buffer_max = 500  # equal to global iterations

        # train args
        self.basic_batch_size = train_batch
        self.train_batch_size = train_batch  # 训练网络需要的样本数量

        self.gamma = main_setting["gamma"]
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.copy_model_steps = update_target_steps
        self.lr = main_setting["lr"]

        self.path = None
        self.test_mem =True

        if self.model_name == 'GAT_QN':

            alpha = main_setting["alpha"]  # leakyReLU的alpha
            nhead = main_setting["nheads"]

            layer_tp = (main_setting["GAT_atten_layer"], main_setting["GAT_out_atten_layer"])
            hid_dim_tp = (main_setting["GAT_hid_dim"], main_setting["GAT_out_hid_dim"])
            hid_s_dim_tp = (main_setting["GAT_s_hid_dim"], main_setting["GAT_s_out_hid_dim"])

            self.nnmodel = main_setting["nnVersion"]
            logging.debug(f"nnmodel {self.nnmodel}")
            if self.nnmodel == "v2" or self.nnmodel == "v3":
                out_nhid_tp = hid_dim_tp[-1]
                # assert out_nhid_tp[-1] == 1, "out feature of agent model must be one"
                # logging.debug(f"nnmodel {self.nnmodel}")
                self.policy_model = GAT_degree(self.nnmodel, self.nodes, layer_tp, nfeat=self.node_features_dims,
                                               nhid_tuple=hid_dim_tp,
                                               nfeat_s=self.nodes, nhid_s_tuple=hid_s_dim_tp,
                                               alpha=alpha, nheads=nhead, mergeZ=self.merge_z, mergeState=True,
                                               use_cuda=self.use_cuda, device=self.device, method=self.method)
                self.target_model = GAT_degree(self.nnmodel, self.nodes, layer_tp, nfeat=self.node_features_dims,
                                               nhid_tuple=hid_dim_tp,
                                               nfeat_s=self.nodes, nhid_s_tuple=hid_s_dim_tp,
                                               alpha=alpha, nheads=nhead, mergeZ=self.merge_z, mergeState=True,
                                               use_cuda=self.use_cuda, device=self.device, method=self.method)
            elif self.nnmodel == "v1":
                # GAT输出维度不需要是1
                self.policy_model = GAT_degree2(layer_tp, nfeat=self.node_features_dims, nhid_tuple=hid_dim_tp,
                                                alpha=alpha, nheads=nhead, mergeZ=self.merge_z, mergeState=True,
                                                use_cuda=self.use_cuda, device=self.device, method=self.method)
                self.target_model = GAT_degree2(layer_tp, nfeat=self.node_features_dims, nhid_tuple=hid_dim_tp,
                                                alpha=alpha, nheads=nhead, mergeZ=self.merge_z, mergeState=True,
                                                use_cuda=self.use_cuda, device=self.device, method=self.method)
            elif self.nnmodel == "v4":
                self.policy_model = GAT(layer_tp, self.node_features_dims + self.nodes, hid_dim_tp,
                                        alpha, nhead, self.merge_z, True,
                                        self.use_cuda, self.device, method=self.method)
                self.target_model = GAT(layer_tp, self.node_features_dims + self.nodes, hid_dim_tp,
                                        alpha, nhead, self.merge_z, True,
                                        self.use_cuda, self.device, method=self.method)
            elif self.nnmodel == "v01":
                mlp_layer = 1
                mlp_hid = 10
                self.policy_model = GAT_MLP(mlp_layer, mlp_hid, layer_tp, self.node_features_dims,
                                    hid_dim_tp, alpha, nhead, self.merge_z, True, self.use_cuda, self.device, self.method)
                self.target_model = GAT_MLP(mlp_layer, mlp_hid, layer_tp, self.node_features_dims,
                                    hid_dim_tp, alpha, nhead, self.merge_z, True, self.use_cuda, self.device, self.method)

            elif self.nnmodel == "v0":
                self.policy_model = GAT_origin(1, 8, 0, 32, self.nodes, 0, alpha=alpha, nheads=nhead,
                        layer_type="default")
                self.target_model = GAT_origin(1, 8, 0, 32, self.nodes, 0, alpha=alpha,
                                               nheads=nhead, layer_type="default")


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
        self.s_mat = self.graph.adm

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

    @torch.no_grad()
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
                # input_node_feat = copy.deepcopy(self.node_features)
                if self.test_mem:
                    test_memory()
                input_node_feat = self.node_features

                if not isinstance(input_node_feat, torch.Tensor):
                    input_node_feat = torch.Tensor(input_node_feat)
                if not isinstance(self.s_mat, torch.Tensor):
                    self.s_mat = torch.Tensor(self.s_mat)
                logging.debug(f" get policy ")

                if self.test_mem:
                    test_memory()

                if self.nnmodel == "v4":
                    input_node_feat = torch.concat((input_node_feat, self.s_mat), 1)
                if self.nnmodel == "v0":
                    input_node_feat = observation.T
                    logging.debug(f"input node feature size of v0 is {input_node_feat}")

                if self.test_mem:
                    test_memory()

                # with profile(activities=[ProfilerActivity.CPU],
                #             profile_memory=True, record_shapes=True) as prof:
                if self.nnmodel == "v01":
                    q_a = self.policy_model(input_node_feat.to(self.device), self.adj.to(self.device),
                                            torch.Tensor(observation).to(self.device), None,
                                            z=self.z.to(self.device))
                    logging.debug(f"v01 q size is {q_a.size()}")
                elif self.nnmodel == "v0":
                    input_node_feat = input_node_feat[None, ...]
                    if not isinstance(input_node_feat, torch.Tensor):
                        input_node_feat = torch.Tensor(input_node_feat)
                    q_a = self.policy_model(input_node_feat.to(self.device), self.adj.to(self.device))
                else:

                    q_a = self.policy_model(input_node_feat.to(self.device), self.adj.to(self.device),
                                            torch.Tensor(observation).to(self.device), self.s_mat,
                                            z=self.z.to(self.device))
                logging.debug(f" get policy done , action")
                if self.test_mem:
                    test_memory()
                # print(prof.key_averages().table(sort_by="self_cpu_memory_usage"))

                
                if self.use_cuda:
                    q_a = q_a.cpu()
                if self.nnmodel == "v0":
                    q_a = torch.squeeze(q_a, dim=0)

                if self.test_mem:
                    test_memory()

                infeasible_action = [k for k in range(self.graph.node) if k not in feasible_action]
                # print(f"{self.print_tag} infeasible action is {infeasible_action}")

                q_a[infeasible_action] = -9e15

                # logging.debug(f"get model params")
                # for name, param in self.policy_model.named_parameters():
                #     logging.debug(f"this layer: {name}, params: {param}")

                # logging.debug(f"act, final q_a is {q_a}")
                action = q_a.argmax()

                if self.test_mem:
                    test_memory()

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

        # self.train_batch_size = int((1 + len(self.memory) / self.buffer_max) * self.basic_batch_size)
        self.train_batch_size = self.basic_batch_size

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
        if self.test_mem:
            test_memory()   

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

        # losses = torch.tensor(0.)

        
        i = 0
        losses = 0.
        if self.test_mem:
            logging.debug(f"before batches")
            test_memory()
        for transition in batch:

            
            state, action, reward, next_state, feasible_a, done, g_id, ft_id, hyper_id = transition
            # 用目标网络计算目标值y
            graph = self.graphs[g_id]
            adj = torch.Tensor(graph.adj_matrix)
            node_feature = torch.Tensor(self.node_feat_pool[ft_id])
            hyper = torch.Tensor(self.hyper_pool[hyper_id])

            if self.test_mem:
                logging.debug(f"p1")
                test_memory()

            if not isinstance(self.s_mat, torch.Tensor):
                self.s_mat = torch.Tensor(self.s_mat)
            
            if self.nnmodel == "v0":
                node_feature = state.T
                # logging.debug(f" node feature size of v0 is {node_feature.size()}")
            
            if self.test_mem:
                logging.debug(f"p2")
                test_memory()

            if self.nnmodel == "v0":
                if not isinstance(node_feature, torch.Tensor):
                    node_feature = torch.Tensor(node_feature)
                node_feature = node_feature[None, ...]
                q_a = self.policy_model(node_feature.to(self.device), adj.to(self.device))
            else:
                if self.nnmodel == "v4":
                    logging.debug(f"ndoe feature size {node_feature.size()}, s_mat size {self.s_mat.size()}")
                    node_feature = torch.concat((node_feature, self.s_mat), 1)

                    q_a = self.policy_model(node_feature.to(self.device), adj.to(self.device),
                                            torch.Tensor(state).to(self.device), None,
                                            z=hyper.to(self.device))
                else:
                    q_a = self.policy_model(node_feature.to(self.device), adj.to(self.device),
                                            torch.Tensor(state).to(self.device), self.s_mat.to(self.device),
                                            z=hyper.to(self.device))
            
            if self.test_mem:
                logging.debug(f"p3")
                test_memory()

            if self.nnmodel == "v0": # 该网络输出第一个维度是batchsizw
                
                q_a = torch.squeeze(q_a, dim=0)
                logging.debug(f"v01 q size is {q_a.size()}")
            

            q = q_a[action]

            if self.test_mem:
                logging.debug(f"p4")
                test_memory()

            if self.nnmodel == "v0":
                next_s = next_state.T
                if not isinstance(next_s, torch.Tensor):
                    next_s = torch.Tensor(next_s)
                next_s = next_s[None, ...]
                q_target = self.target_model(next_s.to(self.device), adj.to(self.device))
            else:
                if self.nnmodel == "v01":
                    q_target = self.target_model(node_feature.to(self.device), self.adj.to(self.device),
                                            torch.Tensor(next_state).to(self.device), None,
                                            z=hyper.to(self.device))
                else:
                    q_target = self.target_model(node_feature.to(self.device), self.adj.to(self.device),
                                        torch.Tensor(next_state).to(self.device), self.s_mat.to(self.device),
                                        z=hyper.to(self.device))
            
            if self.test_mem:
                logging.debug(f"p5")
                test_memory()
            
            if self.nnmodel == "v0":
                q_target = torch.squeeze(q_target, dim=0)
                logging.debug(f"v0 q target size is {q_target.size()}")
            
            # logging.debug(f"target nn is {q_target}")
            infeasible_action = [k for k in range(self.graph.node) if k not in feasible_a]
            # print(f"{self.print_tag} infeasible action is {infeasible_action}")

            if self.test_mem:
                logging.debug(f"p6")
                test_memory()

            if self.algor == "DQN":
                q_target[infeasible_action] = -9e15
                target = reward + (1 - done) * self.gamma * q_target.detach().max()
            elif self.algor == "DDQN":
                # logging.debug(f"q target, max value is {q_target.max()}")
                # logging.debug(f"q a, before mask: \n{q_a}")
                q_a_tmp = q_a.clone()  # 深拷贝，不能改变原值
                q_a_tmp[infeasible_action] = -9e15
                # logging.debug(f"q a, after mask: \n{q_a}")
                policy_max_action = q_a_tmp.argmax()
                # logging.debug(f"policy max action is {policy_max_action}")
                # logging.debug(f"q_target[max] is {q_target[policy_max_action]}")
                target = reward + (1 - done) * self.gamma * q_target[policy_max_action].detach()
                # logging.debug(f"target value is {target}")
            
            if self.test_mem:
                logging.debug(f"p7")
                test_memory()

            if not isinstance(target, torch.Tensor):
                target = torch.Tensor([target])
            # print(f"{self.print_tag} calculated target q is {target}")
            # 用行为网络计算当前值q

            # g_b = make_dot(q_a)
            # g_b.render(
            #     filename="all q value",
            #     directory=self.path + "/logdir",
            #     format="png"
            # )

            # h = q.register_hook(self.hook)
            # self.hs.append(h)
            if i == 0:

                # logging.debug(f"q is {q}")
                # logging.debug(f"target is {target}")
                qs = torch.Tensor([q])
                ts = torch.Tensor([target])
            else:
                qs = torch.concat((qs, q), 0)
                ts = torch.concat((ts, target), 0)
            # loss = self.criterion(q, target)
            if self.test_mem:
                logging.debug(f"p8")
                test_memory()
            i += 1
        # losses_a = make_dot(losses)
        # losses_a.render(
        #     filename="losses",
        #     directory=self.path + "/logdir",
        #     format="png"
        # )
        # logging.debug(f"q batch is {qs}")
        # logging.debug(f"targets is {ts}")
        # loss = torch.mean(torch.tensor(losses, requires_grad=True))
        # loss = losses / len(batch)
        # loss_a = make_dot(loss)
        # loss_a.render(
        #     filename="loss",
        #     directory=self.path + "/logdir",
        #     format="png"
        # )
        # h = loss.register_hook(self.hook)
        # self.hs.append(h)
        # logging.debug(f" loss , requires_grad {loss.requires_grad}, grad {loss.grad}")
        # print(f"{self.print_tag} update losses are {losses} and loss is {loss}")

        # torch.autograd.set_detect_anomaly(True)
        # 梯度更新
        # logging.debug(f"this update: q is {qs}, target is {ts}")
        if self.test_mem:
            logging.debug(f"after batches")
            test_memory()

        loss = self.criterion(qs, ts)
        logging.debug(f"loss is {losses}")
        self.loss = loss.item()
        self.optimizer.zero_grad()
        # logging.debug(f"before backward")
        # for name, param in self.policy_model.named_parameters():
        #     logging.debug(f"this layer: {name}, required grad: {param.requires_grad}, gradients: {param.grad}")

        loss.backward()

        if self.test_mem:
            test_memory()
        # logging.debug(f"after backward")
        for name, param in self.policy_model.named_parameters():
            # logging.debug(f"this layer: {name}, required grad: {param.requires_grad}, gradients: {param.grad}")
            pass
        self.optimizer.step()

        if self.test_mem:
            test_memory()

        for h in self.hs:
            h.remove()

        # 每 C step，更新目标网络 = 当前的行为网络
        if i % self.copy_model_steps == 0:
            # logging.debug(f"reload target model, policy model params: {self.policy_model.state_dict()}\n target model param: {self.target_model.state_dict()}")
            with torch.no_grad():
                self.target_model.load_state_dict(self.policy_model.state_dict())  #
            # logging.debug(f"after reload, target model params: {self.target_model.state_dict()}")

        if self.test_mem:
            test_memory()
        del qs, ts, loss
        return self.loss



