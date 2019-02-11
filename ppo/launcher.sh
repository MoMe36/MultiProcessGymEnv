#!/bin/bash 


launch_training()
{

	env_name=$1
	seed=$2
	frames=$3
	run_id=$4

	python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 \
	--value-loss-coef 1 --num-processes 8 --num-steps 128 \
	--num-mini-batch 4 --vis-interval 1 --log-interval 1 \
	--run_id $run_id --seed $seed --env-name $env_name --num-frames $frames 
}



env="BipedalWalker-v2"
# env="Pendulum-v0"
seed=0
frames=$((10*(10**6)))
run_id="test01"

launch_training $env $seed $frames $run_id


