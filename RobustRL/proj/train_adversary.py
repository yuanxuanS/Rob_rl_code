
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from env import Environment
from graph import Graph_IM
from agent import DQAgent
from PPO_nature import PPOContinuousAgent
import sys
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import networkx as nx
import random
from runner import Runner
# sys.stdout = open(os.devnull, 'w')

log_dir_global = "log_2"
def load_graph(graph_nbr_train, node_nbr, node_edge_p, logdir, logtime):
    graph_dic = {}
    for graph_ in range(graph_nbr_train):
        seed = graph_
        G = Graph_IM(nodes=node_nbr, edges_p=node_edge_p, seed=seed)
        graph_dic[graph_] = G
        graph_dic[graph_].graph_name = str(graph_)
        # degree
        degree = dict(nx.degree(G.graph))
        avg_degree = sum(degree.values()) / G.node
        print(f"Graph {graph_}: average degree {avg_degree}")
        # degree histogram
        x = list(range(max(degree.values())+1))
        y = [i / sum(nx.degree_histogram(G.graph)) for i in nx.degree_histogram(G.graph)]
        plt.bar(x, y, color="blue")
        plt.xlabel("$k$")
        plt.ylabel("$p_k$")
        plt.savefig("./"+log_dir_global+"/"+logdir+"/graphs/"+logtime+"_"+str(graph_)+".png")

    # print('train graphs in total: ', len(graph_dic))
    return graph_dic

def gener_node_features(node_nbr, node_dim, feat_nbr):
    n_feat_dic = {}
    for f in range(feat_nbr):
        seed = f
        np.random.seed(seed)
        tmp = np.random.normal(loc=0.5, scale=3, size=(node_nbr, node_dim))

        n_feat_dic[f] = np.clip(tmp, a_min=0, a_max=1)  # 0-1
    return n_feat_dic

def gener_z(node_dim, z_nbr):
    z_dic = {}
    for z_i in range(z_nbr):
        seed = z_i
        np.random.seed(seed)
        # z_dic[z_i] = np.random.rand(1, 2 * node_dim)        # uniform distribution
        tmp = np.random.normal(loc=0.5, scale=3, size=(1, 2*node_dim))
        z_dic[z_i] = np.clip(tmp, a_min=0, a_max=1)
    return z_dic


parser = argparse.ArgumentParser()
# log setting
parser.add_argument("--logdir", type=str, default="")
parser.add_argument("--logtime", type=str, default="")
# algor setting
parser.add_argument("--graph-pool-nbr", type=int, default=1)
parser.add_argument("--train-graph-nbr", type=int, default=1)
parser.add_argument("--valid-graph-nbr", type=int, default=1)
parser.add_argument("--nodes", type=int, default=100)
parser.add_argument("--edge-p", type=float, default=0.5)
parser.add_argument("--node-feat-dims", type=int, default=3)
parser.add_argument("--feat-pool-nbr", type=int, default=1)
parser.add_argument("--z-pool-nbr", type=int, default=1)
parser.add_argument("--hyper-way", type=str, default="random")      # random or nature_rl, hyperparams generation way
parser.add_argument("--budget", type=int, default=4)
parser.add_argument("--train-episodes", type=int, default=10)       # total training episodes
parser.add_argument("--valid-episodes", type=int, default=10)       # episodes in every validation
parser.add_argument("--valid-every", type=int, default=10)      # valid every train episode
parser.add_argument("--use-cuda", type=bool, default=False)
parser.add_argument("--with-nature", type=bool, default=False)
parser.add_argument("--observe-z", type=bool, default=False)
parser.add_argument("--main-method", type=str, default="rl")
args = parser.parse_args()

# env : pools
env_setting = {"graph_pool_n": args.graph_pool_nbr,  # number of  graphs in pool
               "train_graph_nbr": args.train_graph_nbr,  # number of training graphs in pool
               "valid_graph_nbr": args.valid_graph_nbr,  # number of validation graphs in pool
               "feat_pool_n": args.feat_pool_nbr,  # number of trained features in pool
               "node_feat_dims": args.node_feat_dims,
               "z_pool_n": args.z_pool_nbr,  # number of trained z in pool
               "hyper_way": args.hyper_way,
               "nodes": args.nodes,  # number of graph nodes
               "edge_p": args.edge_p,
               "budgets": args.budget
               }

# main 's training settings
main_setting = {
    "agent_method": args.main_method,  # rl
    "observe_z": args.observe_z
}

# other training setting
training_setting = {
    "with_nature": args.with_nature,
    "train_episodes": args.train_episodes,
    "valid_episodes": args.valid_episodes,
    "hyper_way": args.hyper_way,
    "valid_every": args.valid_every

}

# gpu

use_cuda = args.use_cuda
device_setting = {
    "gpu_flag": torch.cuda.is_available(),
    "use_cuda": args.use_cuda,
    "device": torch.device("cuda:0" if (torch.cuda.is_available() and use_cuda) else "cpu")
}



graph_pool = load_graph(env_setting["graph_pool_n"], env_setting["nodes"], env_setting["edge_p"], args.logdir, args.logtime)
node_feat_pool = gener_node_features(env_setting["nodes"], env_setting["node_feat_dims"], env_setting["feat_pool_n"])
z_pool = gener_z(env_setting["node_feat_dims"], env_setting["z_pool_n"])
propagate_p = 0.7

# nature
actor_lr = 1e-5
critic_lr = 1e-3
nature_lr = [actor_lr, critic_lr]
PolicyDisName = "Beta"
PolicyNormName = "sigmoid"
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
canObserveState = False
canObserveHyper_nature = args.observe_z




cascade = None
epsilon = 0.3
batch_size = 16
update_target_steps = 5       # copy policy_model -> target model
main_lr = 1e-3





# initialize
env = Environment(graph_pool, node_feat_pool, z_pool, env_setting["budgets"])      #
nature_agent = PPOContinuousAgent(graph_pool, node_feat_pool, z_pool, nature_lr, 'GAT_PPO',
                                  env_setting["nodes"], env_setting["node_feat_dims"],
                                  PolicyDisName, PolicyNormName,
                                  canObserveState, gamma,
                                lmbda, eps, epochs, device_setting["use_cuda"], canObserveHyper_nature,
                                  device_setting["device"])

if main_setting['agent_method'] == 'rl':
    model_name = 'GAT_QN'
elif main_setting['agent_method'] == 'random':
    model_name = 'random'

main_agent = DQAgent(graph_pool, node_feat_pool, z_pool,
                     main_lr, model_name,
                     env_setting["node_feat_dims"], epsilon, batch_size, update_target_steps,
                     device_setting["use_cuda"], main_setting['observe_z'], device_setting["device"])

st = time.time()


# record infor

def record_infor(env_setting, training_setting, main_setting, device_setting):
    print(f"Environment ---")
    print(f"graph pool :{env_setting['graph_pool_n']}, connect edge: {env_setting['edge_p']}")
    print(f"feature_pool :{env_setting['feat_pool_n']}, node_feature dimensions :{env_setting['node_feat_dims']}")
    print(f"hyper_pool : {env_setting['z_pool_n']}")
    print(f"\tgraph size: {env_setting['nodes']} nodes, {env_setting['budgets']} budgets")

    print(f"Main Agent ---")
    print(f"main agent: {main_setting['agent_method']}")
    print(f"agent get hyper: {main_setting['observe_z']}")

    print(f"Training --- ")
    print(f"with nature adversary： {training_setting['with_nature']}")
    print(f"hyper generation way: by {training_setting['hyper_way']}")
    print(f"training episodes: {training_setting['train_episodes']}")
    print(f"validation episodes: {training_setting['valid_episodes']}")
    print(f"valid every {training_setting['valid_every']} training episodes")

    print(f"Device ---")
    print(f"GPU is {device_setting['gpu_flag']} and {torch.cuda.device_count()} gpus")
    print(f"use cuda: {device_setting['use_cuda']}")
    print(f"get device {device_setting['device']}")

record_infor(env_setting, training_setting, main_setting, device_setting)

# tensorboard
starttime = time.strftime("%Y-%m-%d_%H:%M:%S")
print(f"start time {starttime}")
writer = SummaryWriter("./" + log_dir_global + "/" + args.logdir + "/" + starttime[:19] + " -" + str(
            env_setting["nodes"]) + " nodes -" + str(env_setting["budgets"]) + " budgets")





# training
runner = Runner(env, env_setting, main_agent, main_setting, nature_agent,
                training_setting, device_setting, writer)
runner.train()

writer.close()
print(f"run time {time.time() - st}")
# plt.figure()
# plt.plot(range(len(y_cumulative_reward)), y_cumulative_reward)
# plt.title(f"reward every episode, {node_nbr} nodes, {T} round, {budget} budget,\n run time {time.time() - st}, end time {time.time()}", fontsize=6)
# plt.savefig("reward")
#
# 
# plt.figure()
# plt.plot(range(len(y_cumulative_reward)), nature_actor_loss)
# plt.title("actor loss of nature agent every episode")
# plt.savefig("act_loss")
# #
# plt.figure()
# plt.plot(range(len(y_cumulative_reward)), nature_critic_loss)
# plt.title("critic loss of nature agent every episode")
# plt.savefig("cri_loss")
#
# plt.show()
# plt.figure()
# plt.plot(range(len(main_loss)), main_loss)
# plt.title("loss of main agent every step")
# plt.savefig("main_loss per step.png")
#
# plt.figure()
# plt.plot(range(global_iter), main_loss_episode)
# plt.title("mean loss of main agent every episode")
# plt.savefig("mean_loss per episode.png")
# print(f"{list(main_agent.policy_model.named_children())}")
# print(f"third {next(main_agent.policy_model.parameters()).device}")

