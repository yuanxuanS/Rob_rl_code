#!/usr/bin/env bash
# bash run.sh 100 4 100 1037
nodes=$1
budget=$2
episode=$3
runtime=$4
edge_p=0.2

graph_pn=200
g_train_nbr=100   # graph-pool-nbr >= train-graph-nbr + valid-graph-nbr
g_valid_nbr=5


hyper_way="random"
valid_episodes=20

logdir="7_4"
with_nature=True

if [ $with_nature = "True" ]; then
  COMMAND="nohup python3 -u train_adversary.py --nodes $nodes --budget $budget
  --graph-pool-nbr $graph_pn --train-graph-nbr $g_train_nbr --valid-graph-nbr $g_valid_nbr
  --hyper-way $hyper_way --edge-p $edge_p
  --train-episodes $episode --valid-episodes $valid_episodes
   --with-nature $with_nature
  --logdir $logdir --logtime $runtime >./log_2/$logdir/${runtime}_n.log 2>&1 &"
else
  COMMAND="nohup python3 -u train_adversary.py --nodes $nodes --budget $budget
  --graph-pool-nbr $graph_pn --train-graph-nbr $g_train_nbr --valid-graph-nbr $g_valid_nbr
  --hyper-way $hyper_way --edge-p $edge_p
  --train-episodes $episode --valid-episodes $valid_episodes
  --logdir $logdir --logtime $runtime >./log_2/$logdir/$runtime.log 2>&1 &"
fi
echo $COMMAND
eval ${COMMAND}