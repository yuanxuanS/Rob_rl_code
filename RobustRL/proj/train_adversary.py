
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

def setup_logger():
    # 创建一个logger对象
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)

    # 创建一个格式器，定义日志输出格式，包含时间、进程ID、日志名称、日志级别和消息内容
    formatter = logging.Formatter('%(asctime)s - %(process)d  - %(name)s - %(levelname)s - %(message)s')

    # # 创建一个控制台处理器，将日志输出到控制台
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.DEBUG)
    # console_handler.setFormatter(formatter)
    #
    # # 创建一个文件处理器，将日志记录到文件中
    # file_handler = logging.FileHandler('app.log')
    # file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(formatter)

    # 将处理器添加到logger对象
    # logger.addHandler(console_handler)
    # logger.addHandler(file_handler)

    return logger
# sys.stdout = open(os.devnull, 'w')

log_dir_global = "log_3"
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
parser.add_argument("--valid-graph-nbr", type=int, default=4)
parser.add_argument("--nodes", type=int, default=100)
parser.add_argument("--edge-p", type=float, default=0.5)
parser.add_argument("--node-feat-dims", type=int, default=3)
parser.add_argument("--feat-pool-nbr", type=int, default=1)
parser.add_argument("--z-pool-nbr", type=int, default=1)
parser.add_argument("--hyper-way", type=str, default="random")      # random or rl_nature, hyperparams generation way
parser.add_argument("--budget", type=int, default=4)
parser.add_argument("--train-episodes", type=int, default=10)       # total training episodes
parser.add_argument("--valid-episodes", type=int, default=10)       # episodes in every validation
parser.add_argument("--valid-every", type=int, default=10)      # valid every train episode
parser.add_argument("--use-cuda", type=bool, default=False)
parser.add_argument("--with-nature", type=bool, default=False)
parser.add_argument("--observe-z", type=bool, default=False)
parser.add_argument("--main-method", type=str, default="rl")
parser.add_argument("--GAT-heads", type=int, default=1)
parser.add_argument("--hidden-dims", type=int, default=4)
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument("--seed-nbr", type=int, default=3)

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
    "observe_z": args.observe_z,
    "nheads": args.GAT_heads,
    "hidden_dims": args.hidden_dims,
    "alpha": args.alpha
}

# nature's settings
nature_setting = {
    "agent_method": 'GAT_PPO',
    "canObserve_hyper": args.observe_z,
    "canObserve_state": False,
    "nheads": args.GAT_heads,
    "hidden_dims": args.hidden_dims,
    "alpha": args.alpha
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




# main agent
cascade = None
epsilon = 0.3
batch_size = 16
update_target_steps = 5       # copy policy_model -> target model
main_lr = 1e-3


def run_one_seed(logger, lock, this_seed, seed_per_g_dict):
    logger.info(f"this seed is {this_seed}")
    # initialize
    env = Environment(graph_pool, node_feat_pool, z_pool, env_setting["budgets"])  #
    nature_agent = PPOContinuousAgent(graph_pool, node_feat_pool, z_pool, nature_lr, nature_setting["agent_method"],
                                      nature_setting["alpha"], nature_setting["nheads"], nature_setting['hidden_dims'],
                                      env_setting["nodes"], env_setting["node_feat_dims"],
                                      PolicyDisName, PolicyNormName,
                                      nature_setting["canObserve_state"], gamma,
                                      lmbda, eps, epochs, device_setting["use_cuda"],
                                      nature_setting["canObserve_hyper"],
                                      device_setting["device"])

    if main_setting['agent_method'] == 'rl':
        model_name = 'GAT_QN'
    elif main_setting['agent_method'] == 'random':
        model_name = 'random'

    main_agent = DQAgent(graph_pool, node_feat_pool, z_pool,
                         main_lr, model_name, main_setting['alpha'], main_setting["nheads"],
                         env_setting["node_feat_dims"], main_setting["hidden_dims"],
                         epsilon, batch_size, update_target_steps,
                         device_setting["use_cuda"], main_setting['observe_z'], device_setting["device"])

    st = time.time()

    # record infor

    def record_infor(env_setting, training_setting, main_setting, device_setting):
        logger.info(f"Environment ---")
        logger.info(f"graph pool :{env_setting['graph_pool_n']}, connect edge: {env_setting['edge_p']}")
        logger.info(f"feature_pool :{env_setting['feat_pool_n']}, node_feature dimensions :{env_setting['node_feat_dims']}")
        logger.info(f"hyper_pool : {env_setting['z_pool_n']}")
        logger.info(f"\tgraph size: {env_setting['nodes']} nodes, {env_setting['budgets']} budgets")

        logger.info(f"Main Agent ---")
        logger.info(f"main agent: {main_setting['agent_method']}")
        logger.info(f"agent get hyper: {main_setting['observe_z']}")
        logger.info(f"agent GAT nheads: {main_setting['nheads']}")
        logger.info(f"agent GAT hidden dims :{main_setting['hidden_dims']}")
        logger.info(f"agent GAT , alpha: {main_setting['alpha']}")

        logger.info(f"Nature Agent ---")
        logger.info(f"nature agent: {nature_setting['agent_method']}")
        logger.info(f"nature get hyper: {nature_setting['canObserve_hyper']}")
        logger.info(f"nature GAT nheads: {nature_setting['nheads']}")
        logger.info(f"nature GAT hidden dims :{nature_setting['hidden_dims']}")
        logger.info(f"nature GAT , alpha: {nature_setting['alpha']}")

        logger.info(f"Training --- ")
        logger.info(f"with nature adversary： {training_setting['with_nature']}")
        logger.info(f"hyper generation way: by {training_setting['hyper_way']}")
        logger.info(f"training episodes: {training_setting['train_episodes']}")
        logger.info(f"validation episodes: {training_setting['valid_episodes']}")
        logger.info(f"valid every {training_setting['valid_every']} training episodes")

        logger.info(f"Device ---")
        logger.info(f"GPU is {device_setting['gpu_flag']} and {torch.cuda.device_count()} gpus")
        logger.info(f"use cuda: {device_setting['use_cuda']}")
        logger.info(f"get device {device_setting['device']}")

    record_infor(env_setting, training_setting, main_setting, device_setting)

    # tensorboard
    starttime = time.strftime("%Y-%m-%d_%H:%M:%S")
    logger.info(f"start time {starttime}")
    img_str = starttime[:19] + " -" + \
              str(env_setting["nodes"]) + " nodes -" \
              + str(env_setting["budgets"]) + " budgets -" \
              + str(this_seed) + " seed "
    writer = SummaryWriter("./" + log_dir_global + "/" + args.logdir + "/" + img_str)

    # training
    runner = Runner(env, env_setting, main_agent, main_setting, nature_agent,
                    training_setting, device_setting, writer)
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

# 设置全局日志配置
logger = setup_logger()


processes = []
lock = multiprocessing.Lock()
for i in range(args.seed_nbr):
    seed.gener_seed()

    this_seed = seed.get_value()
    logging.info(f"seed is {this_seed}")


    p = multiprocessing.Process(target=run_one_seed,
                                args=(logger, lock, this_seed, seed_record_per_g))
    processes.append(p)
    p.start()

# 等待所有进程执行完毕
for p in processes:
    p.join()



logging.info(f"whole seed dict:\n {seed_record_per_g}")

starttime = time.strftime("%Y-%m-%d_%H:%M:%S")
logging.info(f"plot reward-percentile time is {starttime[:19]}")
writer2 = SummaryWriter("./" + log_dir_global + "/" + args.logdir + "/reward-percentile/" + starttime[:19])

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
logging.info(f"whole run time: {time.time() - whole_st}")

# 结束日志处理器
logging.shutdown()