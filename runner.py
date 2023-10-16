import random
import numpy as np
import utils
import logging
import time
import psutil
import sys
from torch.profiler import profile, record_function, ProfilerActivity
logging.basicConfig(level=logging.INFO,
                    format='(%(threadName)-10s) %(message)s',
                    )


class Runner:
    def __init__(self, environment, env_setting, agent, main_setting, nature,
                 training_setting, valid_setting, device_setting, writer, writer_base):
        self.environment = environment
        self.env_setting = env_setting
        self.agent = agent
        self.main_setting = main_setting
        self.nature = nature

        self.training_setting = training_setting
        self.valid_setting = valid_setting
        self.device_setting = device_setting
        self.writer = writer
        self.writer_base = writer_base
        self.path = None

        # statistic
        self.cumu_reward = []

        self.nature_critic_loss = []
        self.nature_actor_loss = []
        self.main_loss = []

        self.check_epi_step = 50
        self.global_iter = 0
        self.global_steps = 0


    def valid(self):
        episode_accumulated_rewards = np.empty((self.env_setting["valid_graph_nbr"], self.training_setting["valid_episodes"]))
        for r in range(self.env_setting["valid_graph_nbr"]):
            for episode in range(self.training_setting["valid_episodes"]):
                # in valid set
                g_id = self.env_setting["graph_pool_n"] - self.env_setting["valid_graph_nbr"] + r
                self.environment.init_graph(g_id)
                self.nature.init_graph(g_id)
                self.agent.init_graph(g_id)
                logging.info(f"-"*10 + f" valid : {episode} round in {r} valid graph, graph-{g_id}")

                # sample node feature
                ft_id = 0
                self.environment.init_n_feat(ft_id)
                self.nature.init_n_feat(ft_id)
                self.agent.init_n_feat(ft_id)

                # sample z: random
                hyper_id = 0
                self.environment.init_hyper(hyper_id)
                self.nature.init_hyper(hyper_id)
                self.agent.init_hyper(hyper_id)

                #
                self.environment.reset()
                if self.training_setting["with_nature"]:
                    self.nature.reset()
                    if self.valid_setting["with_nature"]:
                        logging.info("generate hyperparams by nature in valid")
                        nature_state, _ = self.environment.get_seed_state()
                        z_action_pair_lst = self.nature.act(nature_state)
                        z_new = self.environment.step_hyper(z_action_pair_lst)
                    else:
                        logging.info("generate hyperparams by random in valid")

                # main agent
                self.agent.reset()
                cumul_reward = 0.

                sub_reward = []
                sub_loss = 0

                for i in range(self.environment.budget):
                    # print(f"---------- sub step {i}")
                    state, feasible_action = self.environment.get_seed_state()  # [1, N]
                    action = self.agent.act(state, feasible_action, 0, "valid")
                    logging.info(f"main agent action is {action} ")
                    next_state, reward, done = self.environment.step_seed(i, action, "valid")
                    logging.info(f"get reward is {reward}")


                    cumul_reward += reward
                    sub_reward.append(reward)

                # record result
                episode_accumulated_rewards[r, episode] = cumul_reward

        return episode_accumulated_rewards

    def test_memory(self):
        # logging.debug(f"use memory: {psutil.memory_percent()}")
        logging.debug(f"use vitual memory: {psutil.virtual_memory()}")

    def train(self):

        self.test_mem = True
        self.test_time = True
        for episode in range(self.training_setting["train_episodes"]):  # one-step, adversary is a bandit
            
            # validation
            if self.global_iter % self.check_epi_step == 0:
                g_episode_returns = self.valid()     
                g_mean_returns = np.mean(g_episode_returns, axis=1)

                for r in range(self.env_setting["valid_graph_nbr"]):
                    g_id = self.env_setting["graph_pool_n"] - self.env_setting["valid_graph_nbr"] + r
                    self.writer.add_scalar(f"valid in graph: {g_id}/ with nature: {self.valid_setting['with_nature']}", g_mean_returns[r], self.global_iter)


            self.global_iter += 1


            if self.test_time:
                epi_st = time.time()
            # sample graph
            g_id = random.randint(0, self.env_setting["train_graph_nbr"] - 1)        # [ ]
            self.environment.init_graph(g_id)
            self.nature.init_graph(g_id)
            self.agent.init_graph(g_id)
            logging.info(f"-----------this is -- {self.global_iter} iteration,  training in graph {g_id}")

            if self.test_mem:
                self.test_memory()
                # logging.debug(f"before interact, memory size {sys.getsizeof(self.agent.memory)}")

            # sample node feature
            ft_id = 0
            self.environment.init_n_feat(ft_id)
            self.nature.init_n_feat(ft_id)
            self.agent.init_n_feat(ft_id)

            # sample z
            hyper_id = 0
            self.environment.init_hyper(hyper_id)
            self.nature.init_hyper(hyper_id)
            self.agent.init_hyper(hyper_id)

            #
            self.environment.reset()
            # logging.debug(f"Now reset env, state is \n {self.environment.state}")

            if self.test_time:
                print(f"initial time is {time.time() - epi_st}")

            if self.training_setting["with_nature"]:
                logging.info(f"current train with nature")
                self.nature.reset()
                nature_state, _ = self.environment.get_seed_state()
                logging.info(f"seed state is, and send it to adversary: \n {nature_state}")
                z_action_pair_lst = self.nature.act(nature_state)
                logging.info(f"adversary return: \n {z_action_pair_lst}")
                logging.info(f"before adversary do, env hyperparameter is \n {self.environment.z}")
                z_new = self.environment.step_hyper(z_action_pair_lst)
                logging.info(f"updating env hyperparameter, current hyperparam is \n {self.environment.z}")

            # after adversary change environment z, get best solution
            if self.test_time:
                grd_st = time.time()
            best_s, best_reward = self.environment.greedy_solution()

            if self.test_time:
                grd_ed = time.time()
                print(f"time of greedy computation is {grd_ed - grd_st}")

                agent_st = time.time()
            logging.info(f"best seed set is {best_s}, best reward is {best_reward}")
            # main agent
            self.agent.reset()
            cumul_reward = 0.

            sub_reward = []
            sub_loss = 0
            logging.info(f"before main agent act")

            for i in range(self.environment.budget):
                logging.info(f"---------- sub step {i}, global steps {self.global_steps}")
                self.global_steps += 1
    
                if self.test_time:
                    this_st = time.time()
                if self.test_mem:
                    self.test_memory()

                state, feasible_action = self.environment.get_seed_state()  # [1, N]
                if self.test_mem:
                    self.test_memory()

                logging.info(f"return state")
                infeasible_action = [k for k in range(self.agent.graph.node) if k not in feasible_action]
                logging.info(f"return infeasible action")

                if self.test_mem:
                    self.test_memory()

                if self.test_time:
                    print(f"time before act in a episode {time.time() - this_st}")
                    act_st = time.time()

                action = self.agent.act(state, feasible_action, self.global_steps, "train")

                if self.test_mem:
                    self.test_memory()

                if self.test_time:
                    act_ed = time.time()
                    print(f"time of agent act is {act_ed - act_st}")
                    step_st = time.time()

                logging.info(f"curr seed set is {infeasible_action}, main agent action is {action}, its degree is {self.environment.G.node_degree_lst[action]}")

                
                next_state, reward, done = self.environment.step_seed(i, action, "train")
                logging.info(f"get reward is {reward}")

                if self.test_mem:
                    self.test_memory()

                if self.test_time:
                    step_ed = time.time()
                    print(f"time of env step(simulation) is {step_ed - step_st}")
                    last_st = time.time()

                feasible_action.remove(action)

                if self.test_mem:
                    self.test_memory()

                # add to buffer
                if self.main_setting["agent_method"] == 'rl':
                    logging.info(f"main agent method is reinforcement learning")
                    sample = [state, action, reward, next_state, feasible_action, done, g_id, ft_id, hyper_id]
                    # logging.info(f"sample to remember: \n state is {state} \n action is {action}\n  \
                    # reward is {reward} \n next_state is {next_state} \n feasible action is {feasible_action}\n \
                    # done is {done} ")
                    self.agent.remember(sample)
                elif self.main_setting["agent_method"] == 'random':
                    pass
                
                if self.test_mem:
                    self.test_memory()

                cumul_reward += reward
                sub_reward.append(reward)

                if self.test_mem:
                    self.test_memory()

                # # get sample and update the main model, GAT
                # with profile(activities=[ProfilerActivity.CPU],
                #             profile_memory=True, record_shapes=True) as prof:
                if self.main_setting["agent_method"] == "rl":
                    logging.info(f"now update the main agent")
                    if self.test_time:
                        upd_st = time.time()
                    loss = self.agent.update(self.global_iter)
                    
                    if self.test_mem:
                        self.test_memory()

                    if self.test_time:
                        upd_ed = time.time()
                        print(f"time of update is {upd_ed - upd_st}")
                    logging.info(f"after main agent update, get loss :{loss}")
                    self.main_loss.append(loss)
                    sub_loss += loss

                    
                elif self.main_setting["agent_method"] == "random":
                    pass
                if self.test_mem:
                    self.test_memory()
                    # logging.debug(f"after interact, memory size {utils.get_size(self.agent.memory)}")

                # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

                if self.test_time:
                    print(f"time of last after update is {time.time() - last_st}")
            
            if self.test_time:
                print(f"agent part time is {time.time() - agent_st}")
                record_st = time.time()

            logging.info(f"after budget selection, main agent's return is {cumul_reward}, overall loss is {sub_loss}")
            self.cumu_reward.append(cumul_reward)

            if self.test_mem:
                self.test_memory()
            # print(f"cumulative reward is {cumul_reward}")
            # plot
            # plt.plot(range(self.environment.budget), sub_reward)
            # plt.title("reward per step")
            # plt.show()

            if self.training_setting["with_nature"]:
                # nature agent
                self.nature.remember(nature_state, z_action_pair_lst, -cumul_reward)
                # get a trajectory and update the nature model
                act_loss_nature, cri_loss_nature = self.nature.update()
                logging.info(f"adversary, actor loss {act_loss_nature} critic loss {cri_loss_nature}")
            # self.nature_critic_loss.append(cri_loss_nature.item())
            # self.nature_actor_loss.append(act_loss_nature.item())

            self.writer.add_scalar(
                f'main/GPU={self.device_setting["use_cuda"]}/nature={self.training_setting["with_nature"]}/cumulative reward per episode',
                cumul_reward, self.global_iter)
            self.writer_base.add_scalar(
                f'main/GPU={self.device_setting["use_cuda"]}/nature={self.training_setting["with_nature"]}/cumulative reward per episode',
                best_reward[-1], self.global_iter)
            
            

            if self.main_setting["agent_method"] == "rl":
                self.writer.add_scalar(
                    f'main/GPU={self.device_setting["use_cuda"]}/nature={self.training_setting["with_nature"]}/mean loss ',
                    sub_loss / self.environment.budget, self.global_iter)
            if self.training_setting["with_nature"]:
                self.writer.add_scalar(f'nature/GPU={self.device_setting["use_cuda"]}/actor loss ', act_loss_nature.item(),
                                  self.global_iter)
                self.writer.add_scalar(f'nature/GPU={self.device_setting["use_cuda"]}/critic loss ', cri_loss_nature.item(),
                                  self.global_iter)
            
            if self.test_time:
                print(f"last record in tensorboard time is {time.time() - record_st}")
                print(f"time of an training episode is {time.time() - epi_st}")
            
            if self.test_mem:
                logging.debug(f"tensorboard writer part:")
                self.test_memory()
                # logging.debug(f"agent size: {sys.getsizeof(self.agent)}; env size:{sys.getsizeof(self.environment)}; ")
        # utils.draw_distri_hist(self.cumu_reward, self.path, "cumu_reward")

    def final_valid(self):
        # validation at the end

        logging.info("-"*10+ f" validation from inter: {self.global_iter}")
        episode_accumulated_rewards = self.valid()      # valid graph nbr， 20
        logging.info(episode_accumulated_rewards)
        average_accumulated_rewards = np.mean(episode_accumulated_rewards, axis=1)      # 5, 1dims
        for r in range(self.env_setting["valid_graph_nbr"]):
            g_id = self.env_setting["graph_pool_n"] - self.env_setting["valid_graph_nbr"] + r
            self.writer.add_scalar(f"valid in graph: {g_id}/ with nature: {self.valid_setting['with_nature']}", average_accumulated_rewards[r], self.global_iter)

        return average_accumulated_rewards