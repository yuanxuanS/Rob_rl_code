#!/usr/bin/env bash
# bash run.sh 100 4 100 3 1037
# set -vx
graph_type="powerlaw" #"erdos_renyi"  #
nodes=$1
budget=$2
episode=$3
seeds=$4
runtime=$5
edge_p=0.05

graph_pn=1   # 106
g_train_nbr=1   # graph-pool-nbr >= train-graph-nbr, graph-pool-nbr >= valid-graph-nbr
g_valid_nbr=1


valid_with_nature=False   # valid with nature?
valid_episodes=3

glb_logfile="log_5"
logdir="10_28"
main_method="rl"    # rl
buffer_type="per_td_return" # ["er", "per_td", "per_return", "per_td_return"]
with_nature=False

rl_algor="DDQN"
nn_version="v01"
# hyper parameter of GAT
nheads=(8)
atten_layer=1
hid_dim='(8,)'  # must write , it is a tuple
hid_dim_s='(8,)'
out_atten_layer=1
out_hid_dim='(32,)'  # final one must be 1 if nn model == v1 or v3
out_hid_dim_s='(32,)'

alphas=(0.2)
gammas=(0.99)
lrs=(1e-5)
batch_size=32

use_decay=0
init_epsilon=0.2
final_epsilon=0.01
epsilon_decay_steps=5000

target_start_update_t=0
target_update_interval=20

for nhead in ${nheads[@]}
do
  for alpha in ${alphas[@]}
  do
    for gamma in ${gammas[@]}
    do
      for lr in ${lrs[@]}
      do
        if [ $with_nature = "True" ]; then
          COMMAND="python3 -u train_adversary.py --nodes $nodes --budget $budget
          --graph-type $graph_type
          --graph-pool-nbr $graph_pn --train-graph-nbr $g_train_nbr --valid-graph-nbr $g_valid_nbr
           --edge-p $edge_p --main-method $main_method
          --train-episodes $episode --valid-episodes $valid_episodes
          --with-nature $with_nature
          --rl-algor $rl_algor
          --buffer-type $buffer_type
          --nnVersion $nn_version
          --batch-size $batch_size
          --GAT-heads $nhead
          --GAT-atten-layer $atten_layer
          --GAT-out-atten-layer $out_atten_layer
          --GAT-hid-dim '$hid_dim'
          --GAT-s-hid-dim '$hid_dim_s'
          --GAT-out-hid-dim '$out_hid_dim'
          --GAT-s-out-hid-dim '$out_hid_dim_s'
          --alpha $alpha
          --use-decay $use_decay
          --init-epsilon $init_epsilon
          --final-epsilon $final_epsilon
          --epsilon-decay-steps $epsilon_decay_steps
          --target-start-update-t $target_start_update_t
          --target-update-interval $target_update_interval
          --logdir $logdir --logtime $runtime --seed-nbr $seeds --gamma $gamma --lr $lr >../pscr/$glb_logfile/$logdir/logdir/${runtime}_n.txt 2>../pscr/$glb_logfile/$logdir/logdir/${runtime}_n_error.txt &"
        else
          COMMAND="python3 -u train_adversary.py --nodes $nodes --budget $budget
          --graph-type $graph_type
          --graph-pool-nbr $graph_pn --train-graph-nbr $g_train_nbr --valid-graph-nbr $g_valid_nbr
          --valid-with-nature $valid_with_nature --edge-p $edge_p --main-method $main_method
          --train-episodes $episode --valid-episodes $valid_episodes
          --rl-algor $rl_algor
          --buffer-type $buffer_type
          --nnVersion $nn_version
          --batch-size $batch_size
          --GAT-heads $nhead
          --GAT-atten-layer $atten_layer
          --GAT-out-atten-layer $out_atten_layer
          --GAT-hid-dim '$hid_dim'
          --GAT-s-hid-dim '$hid_dim_s'
          --GAT-out-hid-dim '$out_hid_dim'
          --GAT-s-out-hid-dim '$out_hid_dim_s'
          --alpha $alpha
          --use-decay $use_decay
          --init-epsilon $init_epsilon
          --final-epsilon $final_epsilon
          --epsilon-decay-steps $epsilon_decay_steps
          --target-start-update-t $target_start_update_t
          --target-update-interval $target_update_interval
          --logdir $logdir --logtime $runtime --seed-nbr $seeds --gamma $gamma --lr $lr >../pscr/$glb_logfile/$logdir/logdir/$runtime.txt 2>../pscr/$glb_logfile/$logdir/logdir/${runtime}_error.txt &"
        fi
        echo $COMMAND
        eval ${COMMAND}
        let runtime++
        sleep 2
      done
    done
  done
done

wait