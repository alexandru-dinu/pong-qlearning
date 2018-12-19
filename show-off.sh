#!/bin/bash

python src/main.py \
	--canvas_size 32 24 \
	--paddle_length 7 \
	--velocity 1 \
	--epsilon 0.03 \
	--learning_rate 0.7 \
	--discount 0.99 \
	--train_episodes 50000 \
	--eval_episodes 10 \
	--eval_every 200 \
	--max_iter 1000 \
	--agent_strategy eps_greedy \
	--opponent_strategy almost_perfect --alpha 0.8 \
	--load --filename saved_qs/32_24_7_003_07_50k_eg_ap_08.p
