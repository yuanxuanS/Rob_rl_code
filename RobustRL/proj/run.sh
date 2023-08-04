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


valid_with_nature=False   # valid with nature?
valid_episodes=20

logdir="8_3"
with_nature=False

# hyper parameter of GAT
nheads=(8)
hidden_dims=(16)
alphas=(0.2)
gammas=(0.01 0.1 0.5 0.99)
lrs=(1e-3 1e-4 5e-5 1e-5)

for nhead in ${nheads[@]}
do
  for hid_dim in ${hidden_dims[@]}
  do
    for alpha in ${alphas[@]}
    do
      for gamma in ${gammas[@]}
      do
        for lr in ${lrs[@]}
        do
          if [ $with_nature = "True" ]; then
            COMMAND="python3 -u train_adversary.py --nodes $nodes --budget $budget
            --graph-pool-nbr $graph_pn --train-graph-nbr $g_train_nbr --valid-graph-nbr $g_valid_nbr
             --edge-p $edge_p
            --train-episodes $episode --valid-episodes $valid_episodes
             --with-nature $with_nature
             --GAT-heads $nhead --hidden-dims $hid_dim --alpha $alpha
            --logdir $logdir --logtime $runtime --seed-nbr $seeds --gamma $gamma --lr $lr >./log_3/$logdir/${runtime}_n.txt 2>./log_3/$logdir/${runtime}_n_error.txt &"
          else
            COMMAND="python3 -u train_adversary.py --nodes $nodes --budget $budget
            --graph-pool-nbr $graph_pn --train-graph-nbr $g_train_nbr --valid-graph-nbr $g_valid_nbr
            --valid-with-nature $valid_with_nature --edge-p $edge_p
            --train-episodes $episode --valid-episodes $valid_episodes
            --GAT-heads $nhead --hidden-dims $hid_dim --alpha $alpha
            --logdir $logdir --logtime $runtime --seed-nbr $seeds --gamma $gamma --lr $lr >./log_3/$logdir/$runtime.txt 2>./log_3/$logdir/${runtime}_error.txt &"
          fi

          echo $COMMAND
          eval ${COMMAND}
          let runtime++
          sleep 2
        done
      done
    done
  done
done

wait
