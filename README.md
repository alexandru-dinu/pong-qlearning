# Pong Q-learning

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Simple [Q-learning](https://en.wikipedia.org/wiki/Q-learning) agent for [pong](https://en.wikipedia.org/wiki/Pong).

## Structure

```
src
├── game.py  # game logic (using pygame)
├── main.py  # parse args and run
└── world.py # main module: Q-learning + game interactions
```

## Usage

```bash
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
	--load --filename q-values/32_24_7_003_07_50k_eg_ap_08.p
```

To train with custom parameters, omit the `--load --filename <file>` args.

You can also use `--plot_scores` to plot train and eval scores, after training.

Using `--filename <file>` (without `--load`) will save final Q-values to the specified file,
as a pickle dictionary.

Pickle file naming convention:
```
<w>_<h>_<paddle_len>_<eps>_<lr>_<train_episodes>_<agent_strategy>_<opponent_strategy>
```

## Results

### red opponent has 100% chance of moving perfectly
![1](https://i.imgur.com/zI52aqp.gif)

### red opponent has 80% chance of moving perfectly
![2](https://i.imgur.com/PwuOn7x.gif)
