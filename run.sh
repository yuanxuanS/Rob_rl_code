#!/usr/bin/env bash
# bash run.sh 100 4 100 3 1037
set -vx
nodes=$1
budget=$2
episode=$3
seeds=$4
runtime=$5
edge_p=0.1

graph_pn=2   # 106
g_train_nbr=1   # graph-pool-nbr >= train-graph-nbr + valid-graph-nbr
g_valid_nbr=1


valid_with_nature=False   # valid with nature?
valid_episodes=1

glb_logfile="log_5"
logdir="9_25"
main_method="rl"    # rl
with_nature=False

rl_algor="DDQN"
nn_version="v01"
# hyper parameter of GAT
nheads=(8)
atten_layer=1
hid_dim='(16, )'
hid_dim_s='(16, )'
out_atten_layer=1
out_hid_dim='(1,)'  # final one must be 1 if nn model == v1 or v3
out_hid_dim_s='(1,)'

alphas=(0.2)
gammas=(0.99)
lrs=(1e-4)
batch_size=8

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
          --graph-pool-nbr $graph_pn --train-graph-nbr $g_train_nbr --valid-graph-nbr $g_valid_nbr
           --edge-p $edge_p --main-method $main_method
          --train-episodes $episode --valid-episodes $valid_episodes
          --with-nature $with_nature
          --rl-algor $rl_algor
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
          --logdir $logdir --logtime $runtime --seed-nbr $seeds --gamma $gamma --lr $lr >../pscr/$glb_logfile/$logdir/logdir/${runtime}_n.txt 2>../pscr/$glb_logfile/$logdir/logdir/${runtime}_n_error.txt &"
        else
          COMMAND="python3 -u train_adversary.py --nodes $nodes --budget $budget
          --graph-pool-nbr $graph_pn --train-graph-nbr $g_train_nbr --valid-graph-nbr $g_valid_nbr
          --valid-with-nature $valid_with_nature --edge-p $edge_p --main-method $main_method
          --train-episodes $episode --valid-episodes $valid_episodes
          --rl-algor $rl_algor
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