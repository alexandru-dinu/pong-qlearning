# pong-qlearning

```
src
├── game.py     # game logic (using pygame)
├── main.py     # parse args and run
├── utils.py    # useful stuff
└── world.py    # main module: Q-Learning + game interactions
```

Usage:

```
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
```

To train with custom parameters, omit the `--load --filename <file>` args.

You can also use `--plot_scores` to plot train and eval scores, after training.

Using `--filename <file>` (without `--load`) will save final Q-values to the specified file,
as a pickle dictionary.

## Results

### red opponent has 100% chance of moving perfectly
![1](https://i.imgur.com/zI52aqp.gif)

### red opponent has 80% chance of moving perfectly
![2](https://i.imgur.com/PwuOn7x.gif)
