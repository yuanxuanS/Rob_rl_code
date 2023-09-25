
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
import seed
import multiprocessing
from multiprocessing import Manager
import logging
import ast


log_dir_global = "log_5"

parser = argparse.ArgumentParser()
# log setting
parser.add_argument("--logdir", type=str, default="")
parser.add_argument("--logtime", type=str, default="")
# algor setting
parser.add_argument("--graph-pool-nbr", type=int, default=1)
parser.add_argument("--train-graph-nbr", type=int, default=1)
parser.add_argument("--valid-graph-nbr", type=int, default=4)
parser.add_argument("--nodes", type=int, default=100)
parser.add_argument("--budget", type=int, default=4)

parser.add_argument("--valid-with-nature", type=bool, default=False)      # random or rl_nature, hyperparams generation way in validation
parser.add_argument("--edge-p", type=float, default=0.1)
parser.add_argument("--train-episodes", type=int, default=10)       # total training episodes
parser.add_argument("--valid-episodes", type=int, default=10)       # episodes in every validation

parser.add_argument("--with-nature", type=bool, default=False)

parser.add_argument("--batch-size", type=int, default=1)

parser.add_argument("--GAT-heads", type=int, default=1)
parser.add_argument("--GAT-atten-layer", type=int, default=1)
parser.add_argument("--GAT-out-atten-layer", type=int, default=1)
parser.add_argument("--GAT-hid-dim", type=str, default="")
parser.add_argument("--GAT-out-hid-dim", type=str, default="")
parser.add_argument("--GAT-s-hid-dim", type=str, default="")
parser.add_argument("--GAT-s-out-hid-dim", type=str, default="")

parser.add_argument("--alpha", type=float, default=0.2)

parser.add_argument("--seed-nbr", type=int, default=3)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--rl-algor", type=str, default="DQN")
parser.add_argument("--nnVersion", type=str, default="v1")

#
parser.add_argument("--node-feat-dims", type=int, default=3)
parser.add_argument("--feat-pool-nbr", type=int, default=1)
parser.add_argument("--z-pool-nbr", type=int, default=1)

parser.add_argument("--valid-every", type=int, default=10)      # valid every train episode
parser.add_argument("--use-cuda", type=bool, default=False)

parser.add_argument("--observe-z", type=bool, default=False)
parser.add_argument("--main-method", type=str, default="rl")

args = parser.parse_args()

def setup_logger(path):
    # # 创建一个控制台处理器，将日志输出到控制台
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.DEBUG)
    # console_handler.setFormatter(formatter)
    #
    # # 创建一个文件处理器，将日志记录到文件中
    file_handler = logging.FileHandler(path)
    # 创建一个格式器，定义日志输出格式，包含时间、进程ID、日志名称、日志级别和消息内容
    formatter = logging.Formatter('%(asctime)s - %(process)d - %(levelname)s - %(filename)s - %(funcName)s - %(message)s')
    file_handler.setFormatter(formatter)

    log = logging.getLogger()
    for hdlr in log.handlers[:]:
        log.removeHandler(hdlr)
    log.addHandler(file_handler)
    log.setLevel(logging.DEBUG)

    return log
# sys.stdout = open(os.devnull, 'w')

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
        # plt.bar(x, y, color="blue")
        # plt.xlabel("$k$")
        # plt.ylabel("$p_k$")
        # plt.savefig("./"+log_dir_global+"/"+logdir+"/graphs/"+logtime+"_"+str(graph_)+".png")

    # print('train graphs in total: ', len(graph_dic))
    return graph_dic

def gener_node_features(node_nbr, node_dim, feat_nbr, normal_mean):
    n_feat_dic = {}
    for f in range(feat_nbr):
        seed = f
        np.random.seed(seed)
        # tmp = np.random.normal(loc=normal_mean, scale=3, size=(node_nbr, node_dim))
        # n_feat_dic[f] = np.clip(tmp, a_min=0, a_max=normal_mean+0.1)  # 0-1
        n_feat_dic[f] = np.ones([node_nbr, node_dim]) * normal_mean

    return n_feat_dic

def gener_z(node_dim, z_nbr, z_mean):
    z_dic = {}
    for z_i in range(z_nbr):
        seed = z_i
        np.random.seed(seed)
        # z_dic[z_i] = np.random.rand(1, 2 * node_dim)        # uniform distribution
        # tmp = np.random.normal(loc=0.5, scale=3, size=(1, 2*node_dim))
        # z_dic[z_i] = np.clip(tmp, a_min=0, a_max=1)
        z_dic[z_i] = np.ones([1, 2*node_dim]) * z_mean

    return z_dic




# env : pools
env_setting = {"graph_pool_n": args.graph_pool_nbr,  # number of  graphs in pool
               "train_graph_nbr": args.train_graph_nbr,  # number of training graphs in pool
               "valid_graph_nbr": args.valid_graph_nbr,  # number of validation graphs in pool
               "feat_pool_n": args.feat_pool_nbr,  # number of trained features in pool
               "node_feat_dims": args.node_feat_dims,
               "node_feat_normal_mean": 0.3,
               "z_pool_n": args.z_pool_nbr,  # number of trained z in pool
               "z_mean": 0.5,
               "nodes": args.nodes,  # number of graph nodes
               "edge_p": args.edge_p,
               "budgets": args.budget
               }

# main agent 's training settings
main_setting = {
    "agent_method": args.main_method,  # rl
    "observe_z": args.observe_z,
    "batch_size": args.batch_size,
    "nheads": args.GAT_heads,
    "alpha": args.alpha,
    "gamma": args.gamma,
    "lr": args.lr,
    "rl_algor": args.rl_algor,      # "DQN"
    "nnVersion": args.nnVersion,
    "GAT_mtd": "base",  # "aggre_degree", # "base': original GAT attention
    "GAT_atten_layer": args.GAT_atten_layer,           # equal to number of GAT_hid_dim
    "GAT_out_atten_layer": args.GAT_out_atten_layer,
    "GAT_hid_dim": ast.literal_eval(args.GAT_hid_dim),
    "GAT_out_hid_dim": ast.literal_eval(args.GAT_out_hid_dim), # final one must be 1
    "GAT_s_hid_dim": ast.literal_eval(args.GAT_s_hid_dim),
    "GAT_s_out_hid_dim": ast.literal_eval(args.GAT_s_out_hid_dim),
}

nature_out_atten_dim = ast.literal_eval(args.GAT_out_hid_dim)

# nature's settings
nature_setting = {
    "agent_method": 'GAT_PPO',
    "PolicyDisName": "Beta",
    "PolicyNormName": "sigmoid",
    "canObserve_hyper": args.observe_z,
    "canObserve_state": False,
    "nheads": args.GAT_heads,
    "nnVersion": args.nnVersion,
    "GAT_mtd": "base",  # "base': original GAT attention
    "GAT_atten_layer": args.GAT_atten_layer,           # equal to number of GAT_hid_dim
    "GAT_out_atten_layer": args.GAT_out_atten_layer,
    "GAT_hid_dim": ast.literal_eval(args.GAT_hid_dim),
    "GAT_out_hid_dim": nature_out_atten_dim,            # final one must be 1
    "GAT_s_hid_dim": ast.literal_eval(args.GAT_s_hid_dim),
    "GAT_s_out_hid_dim": ast.literal_eval(args.GAT_s_out_hid_dim),
    "alpha": args.alpha,
    "gamma": args.gamma,     # = main agent gamma
    "actor_lr": args.lr,
    "critic_lr": args.lr
}

# other training setting
training_setting = {
    "with_nature": args.with_nature,
    "train_episodes": args.train_episodes,
    "valid_episodes": args.valid_episodes,
    "valid_every": args.valid_every

}

valid_setting = {
    "with_nature": args.valid_with_nature
}

# gpu

use_cuda = args.use_cuda
device_setting = {
    "gpu_flag": torch.cuda.is_available(),
    "use_cuda": args.use_cuda,
    "device": torch.device("cuda:0" if (torch.cuda.is_available() and use_cuda) else "cpu")
}


gen_grh_st = time.time()
graph_pool = load_graph(env_setting["graph_pool_n"], env_setting["nodes"], env_setting["edge_p"], args.logdir, args.logtime)
node_feat_pool = gener_node_features(env_setting["nodes"], env_setting["node_feat_dims"], env_setting["feat_pool_n"], env_setting["node_feat_normal_mean"])
z_pool = gener_z(env_setting["node_feat_dims"], env_setting["z_pool_n"], env_setting["z_mean"])
gen_grh_ed = time.time()
print(f"time of generate graph: {gen_grh_ed - gen_grh_st}")
propagate_p = 0.7

# nature

lmbda = 0.95
epochs = 10
eps = 0.2




# main agent
cascade = None
epsilon = 0.3
update_target_steps = 5       # copy policy_model -> target model


def run_one_seed(logger, lock, this_seed, seed_per_g_dict):
    logger.info(f"this seed is {this_seed}, pid is {os.getpid()}")
    # initialize
    env = Environment(graph_pool, node_feat_pool, z_pool, env_setting["budgets"])  #
    nature_agent = PPOContinuousAgent(graph_pool, node_feat_pool, z_pool,
                                      nature_setting,
                                      env_setting["nodes"], env_setting["node_feat_dims"],
                                      lmbda, eps, epochs, device_setting["use_cuda"],
                                      device_setting["device"])


    if main_setting['agent_method'] == 'rl':
        model_name = 'GAT_QN'
    elif main_setting['agent_method'] == 'random':
        model_name = 'random'

    main_agent = DQAgent(graph_pool, args.nodes, node_feat_pool, z_pool,
                         model_name,
                         main_setting,
                         env_setting["node_feat_dims"],
                         epsilon, main_setting["batch_size"], update_target_steps,
                         device_setting["use_cuda"],
                         device_setting["device"])

    st = time.time()

    # record infor

    def record_infor(env_setting, training_setting, valid_setting, main_setting, device_setting):
        logger.info(f"Environment ---")
        for key, value in env_setting.items():
            logger.info(f" {key} : {value}")

        logger.info(f"Main Agent ---")
        for key, value in main_setting.items():
            logger.info(f" {key} : {value}")

        logger.info(f"Nature Agent ---")
        for key, value in nature_setting.items():
            logger.info(f" {key} : {value}")

        logger.info(f"Training --- ")
        for key, value in training_setting.items():
            logger.info(f" {key} : {value}")

        logger.info(f"Validation --- ")
        for key, value in valid_setting.items():
            logger.info(f" {key} : {value}")

        logger.info(f"Device ---")
        for key, value in device_setting.items():
            logger.info(f" {key} : {value}")

    record_infor(env_setting, training_setting, valid_setting, main_setting, device_setting)

    # tensorboard
    starttime = time.strftime("%Y-%m-%d_%H:%M:%S")
    logger.info(f"start time {starttime}")
    img_str = starttime[:19] + " -" + \
              str(env_setting["nodes"]) + " nodes -" \
              + str(env_setting["budgets"]) + " budgets -" \
              + str(this_seed) + " seed "
    writer = SummaryWriter("../pscr/" + log_dir_global + "/" + args.logdir + "/" + img_str)
    writer_base = SummaryWriter("../pscr/" + log_dir_global + "/" + args.logdir + "/" + img_str + "_base")

    # graph save dir
    env.path = "../pscr/" + log_dir_global + "/" + args.logdir + "/graphs/" + img_str
    # training
    runner = Runner(env, env_setting, main_agent, main_setting, nature_agent,
                    training_setting, valid_setting, device_setting, writer, writer_base)
    runner.path = "../pscr/" + log_dir_global + "/" + args.logdir + "/graphs/" + img_str
    main_agent.path = "../pscr/" + log_dir_global + "/" + args.logdir
    runner.train()

    returns = runner.final_valid()  # in all graphs，


    lock.acquire()
    for r in range(env_setting["valid_graph_nbr"]):
        record_per_seed = []
        record_per_seed.append(returns[r])  # return in all graph of a seed
        record_per_seed.append(this_seed)  # this seed
        record_per_seed.append(img_str)  # this run name of this seed
        logger.info(f"current seed list is \n{record_per_seed}")
        value = seed_per_g_dict[r]
        print(f"value is {value}")
        value.append(record_per_seed)
        print(f"value after append is {value}")
        seed_per_g_dict.update({r:value})
        logger.info(f"current dict is \n {seed_per_g_dict}")

    lock.release()

    logger.info(f"run time {time.time() - st}")

    writer.close()


# iter with different seed
manager = Manager()
seed_record_per_g = manager.dict()    # [ [[return, seed, 'name str'], [],...],

for r in range(env_setting["valid_graph_nbr"]):
    seed_record_per_g[r] = []

whole_st = time.time()




processes = []
lock = multiprocessing.Lock()
for i in range(args.seed_nbr):
    seed.gener_seed()

    this_seed = seed.get_value()
    print(f"seed is {this_seed}")

    # 设置全局日志配置
    path = "../pscr/" + log_dir_global + "/" + args.logdir + "/logdir/" + args.logtime + "_seed_" + str(this_seed) + ".log"
    logger = setup_logger(path)

    p = multiprocessing.Process(target=run_one_seed,
                                args=(logger, lock, this_seed, seed_record_per_g))
    processes.append(p)
    p.start()

# 等待所有进程执行完毕
for p in processes:
    p.join()



print(f"whole seed dict:\n {seed_record_per_g}")

starttime = time.strftime("%Y-%m-%d_%H:%M:%S")
print(f"plot reward-percentile time is {starttime[:19]}")
writer2 = SummaryWriter("../pscr/" + log_dir_global + "/" + args.logdir + "/reward-percentile/" + starttime[:19])

for r in range(env_setting["valid_graph_nbr"]):
    multi_seed_g = seed_record_per_g[r]
    print(f"before sort {multi_seed_g}")
    multi_seed_g.sort(key=lambda x: int(x[0]))      # x = 其中的一个[reward, seed, 'str']
    print(f"after sort {multi_seed_g}")

    x = []
    y = []
    for i in range(len(multi_seed_g)):
        x.append((i+1) / len(multi_seed_g))
        y.append(multi_seed_g[i][0])
        writer2.add_scalar(f'percentile——reward/graph:{r}/seed number={args.seed_nbr} ', multi_seed_g[i][0],
                           int((i+1) / len(multi_seed_g) * 100))

    print(f" x is {x} \ny is {y}")


    # plt.figure()
    # plt.plot(x, y)
    # plt.title("percentile——reward")
    # plt.savefig("./"+log_dir_global+"/"+args.logdir+"/graphs/"+args.logtime+"_"+str(r)+".png")


writer2.close()
print(f"whole run time: {time.time() - whole_st}")

