import torch
import matplotlib.pyplot as plt
import time
import sys
import os
import psutil
import logging
from pathlib import Path
import yaml
def test_memory():
    # logging.debug(f"use memory: {psutil.memory_percent()}")
    logging.debug(f"use vitual memory: {psutil.virtual_memory()}")



def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

# t = torch.tensor([[3.2836]])
# compute_advantage(0.9, 0.9, t)

def draw_distri_hist(data_lst, sav_path, name):
    print(f"--- return histogram --- {name}")
    print(f"max is {max(data_lst)}, min is {min(data_lst)}")
    plt.hist(data_lst)
    plt.title(f"histogram of {name}")
    plt.savefig(sav_path + "_" + name)
    plt.close()


def get_size(obj, seen=None):
    """递归计算对象及其嵌套对象的内存大小"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    if isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(get_size(item, seen) for item in obj)
    elif isinstance(obj, dict):
        size += sum(get_size(k, seen) + get_size(v, seen) for k, v in obj.items())
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__slots__'):
        size += sum(get_size(getattr(obj, slot), seen) for slot in obj.__slots__)

    return size


def _load_model_config(config_dict, params):
    # load the model configuration from model_config.yaml
    # check the command line args to see if in the model_config
    
    # check if model.name is passed
    model_name = _get_basic_config(params, arg_name='model.name', must_in=False)
    if model_name is False:
        model_name = config_dict['model']['name']

    model_args_init = load_model_config()[model_name]    
    # read the params and update the model_args_init, then write the data
    for _i, _v in enumerate(params):
        if _v.startswith('model.') and not _v.startswith('model.name'):
            config = _v.split('.')[1].split('=')[0]
            if config not in model_args_init:
                raise KeyError(f'{_v} is not in the config of model: {model_name}')
            try:
                model_args_init[config] = eval(_v.split('=')[1])
            except:
                model_args_init[config] = _v.split('=')[1]

    config_dict['model'].update(model_args_init)
    return config_dict



def process_config(params, params_bak, logger):
    # Get the defaults from task_default.yaml
    path = Path(os.path.dirname(__file__))
    # print(f"{path}")
    with open(os.path.join(path, "task_default.yaml"), "r") as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "task_default.yaml error: {}".format(exc)
    
    '''
    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config")
    alg_config = _get_config(params, "--config")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    results_path = _get_basic_config(params_bak, other_params=params, arg_name='--results-dir')
    task_name = _get_basic_config(params_bak, arg_name='--config')
    # Save to disk by default for sacred
    #save_path = os.path.join(results_path,
    #                         f'{config_dict["real_graph_name"]}_{config_dict["graph_type"]}', 
    #                         task_name)

    real_graph_name = _get_basic_config(params_bak, arg_name='real_graph_name', must_in=False)
    real_graph_name = config_dict["real_graph_name"] if real_graph_name == False else real_graph_name
    
    reward_alphas = [_get_basic_config(params_bak, arg_name='reward_alpha_r1', must_in=False),
                     _get_basic_config(params_bak, arg_name='reward_alpha_r2', must_in=False),
                     _get_basic_config(params_bak, arg_name='reward_alpha_r3', must_in=False)]
    
    activate_threshold_k = _get_basic_config(params_bak, arg_name='activate_threshold_k', must_in=False)
    activate_threshold_k = config_dict["activate_threshold_k"] if activate_threshold_k == False else activate_threshold_k
    # for rl4im_cc
    if 'reward_alpha_r1' in alg_config:
        _r1, _r2, _r3 = alg_config["reward_alpha_r1"] if reward_alphas[0] == False else reward_alphas[0], \
                        alg_config["reward_alpha_r2"] if reward_alphas[1] == False else reward_alphas[1], \
                        alg_config["reward_alpha_r3"] if reward_alphas[2] == False else reward_alphas[2]
        save_path = os.path.join(results_path,
                                f'{real_graph_name}_ccK-{activate_threshold_k}',
                                f'{task_name}_alpha-{float(_r1)}-{float(_r2)}-{float(_r3)}')
    else:
        save_path = os.path.join(results_path,
                                f'{real_graph_name}',
                                task_name)

    logger.info(f"Saving to FileStorageObserver in {save_path}.")
    file_obs_path = save_path

    # save the results to config_dict and can be used in args
    config_dict['local_results_path'] = file_obs_path

    # load model config
    config_dict = _load_model_config(config_dict, params_bak)
    # load active learning (disabled)
    #config_dict = _load_active_learning_config(config_dict, params_bak)
    '''
    return config_dict
    