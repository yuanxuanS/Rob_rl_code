
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
# sys.stdout = open(os.devnull, 'w')


def load_graph(graph_nbr_train, node_nbr, logdir, logtime):
    graph_dic = {}
    for graph_ in range(graph_nbr_train):
        seed = graph_
        G = Graph_IM(nodes=node_nbr, edges_p=0.08, seed=seed)
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
        plt.savefig("./log/"+logdir+"/"+logtime+"_"+str(graph_)+".png")

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
parser.add_argument("--nodes", type=int, default=100)
parser.add_argument("--node-feat-dims", type=int, default=3)
parser.add_argument("--feat-pool-nbr", type=int, default=1)
parser.add_argument("--z-pool-nbr", type=int, default=1)
parser.add_argument("--budget", type=int, default=4)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--use-cuda", type=bool, default=False)
parser.add_argument("--with-nature", type=bool, default=False)
parser.add_argument("--observe-z", type=bool, default=False)
parser.add_argument("--main-method", type=str, default="rl")
args = parser.parse_args()


# env : pools
env_setting = {"graph_pool_n": args.graph_pool_nbr,     # number of trained graphs in pool
               "feat_pool_n": args.feat_pool_nbr,        # number of trained features in pool
               "node_feat_dims": args.node_feat_dims,
                "z_pool_n": args.z_pool_nbr,       # number of trained z in pool
               "nodes": args.nodes,       # number of graph nodes
                "budgets": args.budget
               }


graph_pool = load_graph(env_setting["graph_pool_n"], env_setting["nodes"], args.logdir, args.logtime)
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

# main 's training settings
main_setting = {
    "agent_method": args.main_method,       # rl
    "observe_z": args.observe_z
}


cascade = None
epsilon = 0.3
batch_size = 16
update_target_steps = 5       # copy policy_model -> target model
main_lr = 1e-3


# other training setting
training_setting = {
    "with_nature": args.with_nature,
    "episodes": args.episodes
}

# gpu

use_cuda = args.use_cuda

device_setting = {
    "gpu_flag": torch.cuda.is_available(),
    "use_cuda": args.use_cuda,
    "device": torch.device("cuda:0" if (torch.cuda.is_available() and use_cuda) else "cpu")
}

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

# tensorboard
starttime = time.strftime("%Y-%m-%d_%H:%M:%S")
writer = SummaryWriter("./log/"+args.logdir+"/"+starttime[:19]+" -"+str(env_setting["nodes"])+" nodes -"+str(env_setting["budgets"])+" budgets")

# record infor

def record_infor(env_setting, training_setting, main_setting, device_setting):
    print(f"Environment ---")
    print(f"graph pool :{env_setting['graph_pool_n']}")
    print(f"feature_pool :{env_setting['feat_pool_n']}, node_feature dimensions :{env_setting['node_feat_dims']}")
    print(f"hyper_pool : {env_setting['z_pool_n']}")
    print(f"\tgraph size: {env_setting['nodes']} nodes, {env_setting['budgets']} budgets")

    print(f"Main Agent ---")
    print(f"main agent: {main_setting['agent_method']}")
    print(f"agent get hyper: {main_setting['observe_z']}")

    print(f"Training --- ")
    print(f"with nature adversary： {training_setting['with_nature']}")
    print(f"training episodes: {training_setting['episodes']}")

    print(f"Device ---")
    print(f"GPU is {device_setting['gpu_flag']} and {torch.cuda.device_count()} gpus")
    print(f"use cuda: {device_setting['use_cuda']}")
    print(f"get device {device_setting['device']}")

record_infor(env_setting, training_setting, main_setting, device_setting)

print(f"start time {starttime}")

y_cumulative_reward = []
nature_critic_loss = []
nature_actor_loss = []
main_loss = []
main_loss_episode = []
global_iter = 0

for g_id, graph in graph_pool.items():

    env.init_graph(g_id)
    nature_agent.init_graph(g_id)
    main_agent.init_graph(g_id)
    print(f"----------- training in graph {g_id}")
    for ft_id, feat in node_feat_pool.items():
        env.init_n_feat(ft_id)
        nature_agent.init_n_feat(ft_id)
        main_agent.init_n_feat(ft_id)

        for hyper_id, hyper in z_pool.items():
            env.init_hyper(hyper_id)
            nature_agent.init_hyper(hyper_id)
            main_agent.init_hyper(hyper_id)

            # print(f"second {next(main_agent.policy_model.parameters()).device}")

            # train


            for episode in range(training_setting["episodes"]):       #  one-step, adversary is a bandit
                global_iter += 1
                print(f"this is -- {global_iter} iteration")
                
                env.reset()
                if training_setting["with_nature"]:
                    nature_agent.reset()

                    nature_state, _ = env.get_seed_state()
                    z_action_pair_lst = nature_agent.act(nature_state)
                    z_new = env.step_hyper(z_action_pair_lst)

                
                # main agent
                main_agent.reset()
                cumul_reward = 0.

                sub_reward = []
                sub_loss = 0
                for i in range(env.budget):
                    # print(f"---------- sub step {i}")
                    state, feasible_action = env.get_seed_state()     # [1, N]
                    action = main_agent.act(state, feasible_action)
                    # print(f"action is {action} ")
                    next_state, reward, done = env.step_seed(i, action)
                    # print(f"get reward is {reward}")

                    # add to buffer
                    if main_setting["agent_method"] == 'rl':
                        sample = [state, action, reward, next_state, done, g_id, ft_id, hyper_id]
                        main_agent.remember(sample)
                    elif main_setting["agent_method"] == 'random':
                        pass

                    cumul_reward += reward
                    sub_reward.append(reward)

                    # get sample and update the main model, GAT
                    if main_setting["agent_method"] == "rl":
                        loss = main_agent.update(i)
                        main_loss.append(loss)
                        sub_loss += loss
                    elif main_setting["agent_method"] == "random":
                        pass
                    # print(f"loss is {loss}")

                y_cumulative_reward.append(cumul_reward)
                if main_setting["agent_method"] == "rl":
                    main_loss_episode.append(sub_loss / env.budget)
                # print(f"cumulative reward is {cumul_reward}")
                # plot
                # plt.plot(range(env.budget), sub_reward)
                # plt.title("reward per step")
                # plt.show()

                if training_setting["with_nature"]:
                    # nature agent
                    nature_agent.remember(nature_state, z_action_pair_lst, -cumul_reward)
                    # get a trajectory and update the nature model
                    act_loss_nature, cri_loss_nature = nature_agent.update()
                # print(f"actor loss {act_loss_nature} critic loss {cri_loss_nature}")
                # nature_critic_loss.append(cri_loss_nature.item())
                # nature_actor_loss.append(act_loss_nature.item())

                writer.add_scalar(f'main/GPU={device_setting["use_cuda"]}/nature={training_setting["with_nature"]}/cumulative reward per episode', cumul_reward, global_iter)
                if main_setting["agent_method"] == "rl":
                    writer.add_scalar(f'main/GPU={device_setting["use_cuda"]}/nature={training_setting["with_nature"]}/mean loss ', sub_loss / env.budget, global_iter)
                if training_setting["with_nature"]:
                    writer.add_scalar(f'nature/GPU={device_setting["use_cuda"]}/actor loss ', act_loss_nature.item(), global_iter)
                    writer.add_scalar(f'nature/GPU={device_setting["use_cuda"]}/critic loss ', cri_loss_nature.item(), global_iter)


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

