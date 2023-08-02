#!/usr/bin/env bash
# bash run.sh 100 4 100 3 1037
nodes=$1
budget=$2
episode=$3
seeds=$4
runtime=$5
edge_p=0.1

graph_pn=106   # 106
g_train_nbr=100   # graph-pool-nbr >= train-graph-nbr + valid-graph-nbr
g_valid_nbr=4


hyper_way="rl_nature"   # random  rl_nature
valid_episodes=20

logdir="7_31"
with_nature=True

# hyper parameter of GAT
nheads=(8)
hidden_dims=(16)
alphas=(0.2)


for nhead in ${nheads[@]}
do
  for hid_dim in ${hidden_dims[@]}
  do
    for alpha in ${alphas[@]}
    do
      if [ $with_nature = "True" ]; then
        COMMAND="python3 -u train_adversary.py --nodes $nodes --budget $budget
        --graph-pool-nbr $graph_pn --train-graph-nbr $g_train_nbr --valid-graph-nbr $g_valid_nbr
        --hyper-way $hyper_way --edge-p $edge_p
        --train-episodes $episode --valid-episodes $valid_episodes
         --with-nature $with_nature
         --GAT-heads $nhead --hidden-dims $hid_dim --alpha $alpha
        --logdir $logdir --logtime $runtime --seed-nbr $seeds >./log_3/$logdir/${runtime}_n.txt 2>./log_3/$logdir/${runtime}_n_error.txt &"
      else
        COMMAND="python3 -u train_adversary.py --nodes $nodes --budget $budget
        --graph-pool-nbr $graph_pn --train-graph-nbr $g_train_nbr --valid-graph-nbr $g_valid_nbr
        --hyper-way $hyper_way --edge-p $edge_p
        --train-episodes $episode --valid-episodes $valid_episodes
        --GAT-heads $nhead --hidden-dims $hid_dim --alpha $alpha
        --logdir $logdir --logtime $runtime --seed-nbr $seeds >./log_3/$logdir/$runtime.txt 2>./log_3/$logdir/${runtime}_error.txt &"
      fi

      echo $COMMAND
      eval ${COMMAND}
      let runtime++
      sleep 2
    done
  done
done
wait
