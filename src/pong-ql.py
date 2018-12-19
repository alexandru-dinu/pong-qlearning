# Dinu Alexandru, 342C4

import numpy as np
import pygame
import argparse
import pickle
from matplotlib import pyplot as plt

X, Y = 0, 1
AGENT, OPPONENT, BALL = 0, 1, 2
WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480

MOVE_KEYS = {
	pygame.K_UP: (OPPONENT, -1),
	pygame.K_DOWN: (OPPONENT, 1),
	pygame.K_w: (AGENT, -1),
	pygame.K_s: (AGENT, 1)
}

DIRECTIONS = {
	'top_left': [-1, -1],
	'top_right': [1, -1],
	'bot_left': [-1, 1],
	'bot_right': [1, 1]
}


REWARDS = {
	'lose': -1,
	'win': 1,
	'default': 0
}
ACTIONS = ["UP", "DOWN", "STAY"]

# how paddle's y coord moves, depending on the action
ACTION_EFFECTS = {
	"UP": -1,
	"DOWN": 1,
	"STAY": 0
}


def random_choice(lst):
	r = np.random.randint(len(lst))
	return lst[r]


class Ball:
	VELOCITY = 1

	def __init__(self, pos):
		# set initial direction to a random one
		self.direction = DIRECTIONS[random_choice(list(DIRECTIONS.keys()))]

		self.init_pos = pos

		# underlying rectangle representing the ball
		# the position is given by Rect members (.x, .y)
		self.rect = pygame.Rect(pos[X], pos[Y], 1, 1)

	def draw(self, canvas):
		pygame.draw.rect(canvas, pygame.Color("white"), self.rect)

	def set_position(self, nx, ny):
		self.rect.x = nx
		self.rect.y = ny

	def get_position(self):
		return self.rect.x, self.rect.y

	def in_paddle_area(self, paddle):
		py = paddle.get_position()
		return py <= self.rect.y <= py + Paddle.LENGTH

	def reset(self):
		self.direction = DIRECTIONS[random_choice(list(DIRECTIONS.keys()))]
		self.rect = pygame.Rect(self.init_pos[X], self.init_pos[Y], 1, 1)


class Paddle:
	VELOCITY = 1
	LENGTH = 5

	def __init__(self, pos):
		self.init_pos = pos
		self.direction = 0

		# position is given by rect member (.x, .y)
		self.rect = pygame.Rect(pos[X], pos[Y], 1, Paddle.LENGTH)

	def set_position(self, ny):
		self.rect.y = ny

	def get_position(self):
		return self.rect.y

	def draw(self, canvas):
		pygame.draw.rect(canvas, pygame.Color("green"), self.rect)

	def reset(self):
		self.direction = 0
		self.rect = pygame.Rect(self.init_pos[X], self.init_pos[Y], 1, Paddle.LENGTH)


class World:
	def __init__(self, args):
		# all cmd line args
		self.args = args

		self.canvas_width, self.canvas_height = args.canvas_size

		pygame.init()

		# big screen (so we can actually see the objects)
		self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
		# where the drawing takes place (small surface)
		self.canvas = pygame.surface.Surface((self.canvas_width, self.canvas_height))

		pygame.display.set_caption('Q-Learning Pong')

		font = pygame.font.SysFont('Arial', 12)
		self.text_params = font.render(
			'Testing fonts',
			True, pygame.Color("white")
		)

		self.text_params = [
			font.render("canvas_size = {}x{}".format(self.canvas_width, self.canvas_height),
						True, pygame.Color("white")),
			font.render("paddle_length = {}".format(Paddle.LENGTH),
						True, pygame.Color("white")),
			font.render("eps = {} / lr = {} / gamma = {} / max_iter = {}".format(
				self.args.epsilon, self.args.learning_rate, self.args.discount, self.args.max_iter),
				True, pygame.Color("white")),
			font.render("train_episodes = {}".format(self.args.train_episodes),
						True, pygame.Color("white")),
			font.render("agent_strategy = {}".format(self.args.agent_strategy),
						True, pygame.Color("white")),
			font.render("opponent_strategy = {}".format(self.args.opponent_strategy),
						True, pygame.Color("white")),
			font.render("almost_perfect_alpha = {}".format(self.args.alpha),
						True, pygame.Color("white"))
		]

		# init ball
		ball_pos = (self.canvas_width // 2, self.canvas_height // 2)
		self.ball = Ball(ball_pos)

		# init paddles
		paddle_y = (self.canvas_height - Paddle.LENGTH) // 2
		p1 = (0, paddle_y)
		p2 = (self.canvas_width - 1, paddle_y)

		self.paddles = [Paddle(p1), Paddle(p2)]

	def handle_ball_movement(self, cx, cy):
		# cx, cy = self.ball.rect.x, self.ball.rect.y
		nx = cx + self.ball.direction[X] * Ball.VELOCITY
		ny = cy + self.ball.direction[Y] * Ball.VELOCITY

		# check paddles
		in_p1_area = self.ball.in_paddle_area(self.paddles[AGENT])
		in_p2_area = self.ball.in_paddle_area(self.paddles[OPPONENT])

		touching_p1 = nx <= 1
		touching_p2 = nx >= self.canvas_width - 2  # -1 paddle -1 left

		# clamp and change dir
		if in_p1_area and touching_p1:
			nx = 1
			self.ball.direction[X] *= -1
		if in_p2_area and touching_p2:
			nx = self.canvas_width - 2
			self.ball.direction[X] *= -1

		# check walls 1 = WALL THICKNESS
		hit_top = ny <= 1
		hit_bottom = ny >= self.canvas_height - 1

		if hit_top:
			ny = 1
			self.ball.direction[Y] *= -1
		if hit_bottom:
			ny = self.canvas_height - 1
			self.ball.direction[Y] *= -1

		# return (without setting) the new position
		return nx, ny

	def handle_paddle_movement(self, paddle, direction):
		ny = paddle.rect.y + direction * Paddle.VELOCITY

		# don't go off the screen
		ny = max(ny, 0)
		ny = min(ny, self.canvas_height - Paddle.LENGTH)

		return ny

	def game_reset(self):
		self.ball.reset()

		for paddle in self.paddles:
			paddle.reset()

	# Q-LEARNING
	def get_eval_strategies(self):
		if self.args.agent_strategy == "almost_perfect":
			agent_eval_strategy = "almost_perfect"
		elif self.args.agent_strategy in ["greedy", "eps_greedy"]:
			agent_eval_strategy = "greedy"
		else:
			agent_eval_strategy = "random"

		opponent_eval_strategy = self.args.opponent_strategy

		return agent_eval_strategy, opponent_eval_strategy

	def final_show(self, Q):
		score = num_iter = 0

		agent_eval_strategy, opponent_eval_strategy = self.get_eval_strategies()

		state = self.get_initial_state()

		game_sleep = 1000 // self.args.fps

		text_offset = 15

		while not self.is_final_state(state, num_iter):
			p_actions = self.get_legal_actions(AGENT, state)
			o_actions = self.get_legal_actions(OPPONENT, state)

			p_act = self.choose_action_by_strategy(
				AGENT, agent_eval_strategy, Q, state, p_actions
			)
			o_act = self.choose_action_by_strategy(
				OPPONENT, opponent_eval_strategy, Q, state, o_actions
			)

			# -------------------------------------------------------
			py, oy, (bx, by) = state
			self.ball.set_position(bx, by)
			self.paddles[AGENT].set_position(py)
			self.paddles[OPPONENT].set_position(oy)

			self.canvas.fill(pygame.Color("black"))
			self.ball.draw(self.canvas)

			for paddle in self.paddles:
				paddle.draw(self.canvas)

			self.window.blit(
				pygame.transform.scale(self.canvas, (WINDOW_WIDTH, WINDOW_HEIGHT)),
				(0, 0)
			)

			for i, tp in enumerate(self.text_params):
				self.window.blit(tp, (WINDOW_WIDTH // 2 - 100, text_offset * (i + 1)))

			pygame.display.update()

			pygame.time.delay(game_sleep)
			# -------------------------------------------------------

			state, reward = self.apply_actions(state, p_act, o_act, num_iter)
			score += reward
			num_iter += 1

		print(self.get_reward_description(state, num_iter))

	def get_initial_state(self):
		self.game_reset()

		state = (
			self.paddles[AGENT].get_position(),
			self.paddles[OPPONENT].get_position(),
			self.ball.get_position()
		)

		return state

	def get_state(self):
		state = (
			self.paddles[AGENT].get_position(),
			self.paddles[OPPONENT].get_position(),
			self.ball.get_position()
		)

		return state

	def make_q_state(self, state):
		_state = (state[AGENT], state[BALL])

		return _state

	def is_final_state(self, state, num_iter):
		bx, _ = state[BALL]

		if (num_iter >= self.args.max_iter) or (bx <= 0) or (bx >= self.canvas_width):
			return True

		return False

	def get_reward_description(self, state, num_iter):
		bx, _ = state[BALL]

		if num_iter >= self.args.max_iter:
			return "default"
		if bx <= 0:
			return "lose"
		if bx >= self.canvas_width:
			return "win"

		return "default"

	def get_legal_actions(self, player_idx, state):
		player_y = state[player_idx]
		legal_actions = ["STAY"]

		if player_y >= Paddle.VELOCITY:
			legal_actions.append("UP")
		if player_y + Paddle.LENGTH <= self.canvas_height - Paddle.VELOCITY:
			legal_actions.append("DOWN")

		return legal_actions

	# returns next_state, reward
	# state = (my paddle pos, adv paddle pas, ball pos)
	def apply_actions(self, state, action_agent, action_opponent, num_iter):
		reward = 0

		# move ball
		bx, by = state[BALL]
		nx, ny = self.handle_ball_movement(bx, by)
		next_ball = (nx, ny)
		self.ball.set_position(nx, ny)

		# move paddles according to action
		agent_y = self.handle_paddle_movement(
			self.paddles[AGENT], ACTION_EFFECTS[action_agent]
		)
		opponent_y = self.handle_paddle_movement(
			self.paddles[OPPONENT], ACTION_EFFECTS[action_opponent]
		)

		self.paddles[AGENT].set_position(agent_y)
		self.paddles[OPPONENT].set_position(opponent_y)

		next_state = (agent_y, opponent_y, next_ball)

		if self.is_final_state(next_state, num_iter):
			rd = self.get_reward_description(next_state, num_iter)
			reward = REWARDS.get(rd, 0)

		return next_state, reward

	def epsilon_greedy(self, Q, state, legal_actions):
		if self.args.choose_unexplored_first:
			unexplored = [a for a in legal_actions if (state, a) not in Q]

			if unexplored:
				return random_choice(unexplored)

		if np.random.uniform(0, 1) <= self.args.epsilon:
			return random_choice(legal_actions)
		else:
			return self.get_best_action(Q, state, legal_actions)

	# state = (agent_y, ball_pos)
	def get_best_action(self, Q, state, legal_actions):
		if self.args.mirror_actions:
			py, oy, (bx, by) = state
			bx = self.canvas_width - 1 - bx

			# spoof state
			state = (oy, py, (bx, by))
		# ---------

		# store in Q my part of the state
		my_state = self.make_q_state(state)
		v = {a: Q.get((my_state, a), 0) for a in legal_actions}

		best_act = max(v.keys(), key=(lambda k: v[k]))

		return best_act

	def get_perfect_action(self, player_idx, state):
		paddle_y = state[player_idx]
		_, by = state[BALL]

		legal_actions = self.get_legal_actions(player_idx, state)

		# follow ball down
		if by > paddle_y and "DOWN" in legal_actions:
			return "DOWN"

		# follow ball up
		if by < paddle_y and "UP" in legal_actions:
			return "UP"

		if by == paddle_y:
			# ball goes up
			if self.ball.direction[Y] == -1:
				return "UP"
			# ball goes down
			if self.ball.direction[Y] == 1:
				return "DOWN"

		return "STAY"

	def get_almost_perfect_action(self, player_idx, state, legal_actions):
		if np.random.uniform(0, 1) <= self.args.alpha:
			return self.get_perfect_action(player_idx, state)
		else:
			return random_choice(legal_actions)

	def choose_action_by_strategy(self, player_idx, strategy, Q, state, actions):
		if strategy == "greedy":
			return self.get_best_action(Q, state, actions)
		if strategy == "eps_greedy":
			return self.epsilon_greedy(Q, state, actions)
		if strategy == "almost_perfect":
			return self.get_almost_perfect_action(player_idx, state, actions)

		# strategy == random
		return random_choice(actions)

	def play_game(self, Q):
		score = num_iter = 0

		agent_eval_strategy, opponent_eval_strategy = self.get_eval_strategies()

		state = self.get_initial_state()

		while not self.is_final_state(state, num_iter):
			p_actions = self.get_legal_actions(AGENT, state)
			o_actions = self.get_legal_actions(OPPONENT, state)

			p_act = self.choose_action_by_strategy(
				AGENT, agent_eval_strategy, Q, state, p_actions
			)
			o_act = self.choose_action_by_strategy(
				OPPONENT, opponent_eval_strategy, Q, state, o_actions
			)

			state, reward = self.apply_actions(state, p_act, o_act, num_iter)
			score += reward
			num_iter += 1

		return score

	def plot_scores(self, train_scores, eval_scores, num_states):
		plt.xlabel("Episode")
		plt.ylabel("Average score")

		# train
		plt.plot(
			np.linspace(1, self.args.train_episodes, self.args.train_episodes),
			np.convolve(train_scores, [0.05] * 20, "same"),
			linewidth=1.0, color="blue"
		)

		# eval
		plt.plot(
			np.linspace(self.args.eval_every, self.args.train_episodes, len(eval_scores)),
			eval_scores, linewidth=2.0, color="red"
		)

		plt.show()

		plt.xlabel("Episode")
		plt.ylabel("size of Q")

		plt.plot(
			np.linspace(self.args.eval_every, self.args.train_episodes, len(num_states)),
			num_states, linewidth=1.0, color="green"
		)

		plt.show()

	def qlearning(self):
		Q = {}

		alpha = self.args.learning_rate
		gamma = self.args.discount

		train_scores = []
		eval_scores = []
		num_states = []

		stats = {
			'win': 0,
			'lose': 0,
			'default': 0
		}

		for train_ep in range(1, self.args.train_episodes + 1):
			score = 0
			num_iter = 0
			state = self.get_initial_state()

			while not self.is_final_state(state, num_iter):
				# choose one of the legal actions
				actions_agent = self.get_legal_actions(AGENT, state)
				actions_opponent = self.get_legal_actions(OPPONENT, state)

				# choose actions for both players, depending on strategy
				action_agent = self.choose_action_by_strategy(
					AGENT, self.args.agent_strategy, Q, state, actions_agent
				)

				action_opponent = self.choose_action_by_strategy(
					OPPONENT, self.args.opponent_strategy, Q, state, actions_opponent
				)

				# apply actions, collect reward for agent and switch to next state
				state_next, reward = self.apply_actions(
					state, action_agent, action_opponent, num_iter
				)
				score += reward

				# update Q value
				actions_next = self.get_legal_actions(AGENT, state_next)
				action_best = self.get_best_action(Q, state_next, actions_next)

				# store in Q my part of the state
				my_state = self.make_q_state(state)
				my_state_next = self.make_q_state(state_next)
				qmax = Q.get((my_state_next, action_best), 0)

				Q[(my_state, action_agent)] = \
					(1 - alpha) * Q.get((my_state, action_agent), 0) + \
					alpha * (reward + gamma * qmax)

				# move to next state
				state = state_next
				num_iter += 1
			# end episode loop

			# record stats for current episode
			rd = self.get_reward_description(state, num_iter)
			stats[rd] += 1

			print("Episode %6d / %6d" % (train_ep, self.args.train_episodes))
			train_scores.append(score)
			num_states.append(len(Q))

			# evaluate policy
			if train_ep % self.args.eval_every == 0:
				# TODO: eval play
				# self.final_show(Q)
				# TODO

				avg_score = .0

				for i in range(args.eval_episodes):
					score = self.play_game(Q)
					avg_score += score

				eval_scores.append(avg_score / self.args.eval_episodes)
		# end for each training episode

		print(stats)
		# plot scores if needed
		if args.plot_scores:
			self.plot_scores(train_scores, eval_scores, num_states)

		return Q

	# Q-LEARNING ---------------------------------------------------------------


def main(args):
	Ball.VELOCITY = args.velocity
	Paddle.VELOCITY = args.velocity
	Paddle.LENGTH = args.paddle_length

	world = World(args)

	# load saved weights
	if args.load:
		Q = pickle.load(open(args.filename, "rb"))
		world.final_show(Q)
	
	# train
	else:
		Q = world.qlearning()
		print("Training done!")

		# save weights if needed
		if args.filename != "":
			pickle.dump(Q, open(args.filename, "wb"))
			print("Saved Q in ", args.filename)

		# show final game if needed
		if args.final_show:
			input("Press any key to play game...")
			world.final_show(Q)


def parse_args():
	argparser = argparse.ArgumentParser()

	# game params
	argparser.add_argument(
		'--canvas_size', dest='canvas_size', nargs=2, default=(32, 24), type=int
	)
	argparser.add_argument(
		'--paddle_length', dest='paddle_length', default=7, type=int
	)
	argparser.add_argument(
		'--velocity', dest='velocity', default=1, type=int
	)
	argparser.add_argument(
		'--filename', dest='filename', default="", type=str
	)
	argparser.add_argument(
		'--load', dest='load', action="store_true"
	)

	# qlearning params
	argparser.add_argument(
		'--max_iter', dest='max_iter', default=1000, type=int
	)
	argparser.add_argument(
		'--learning_rate', dest='learning_rate', default=0.3, type=float
	)
	argparser.add_argument(
		'--discount', dest='discount', default=0.99, type=float
	)
	argparser.add_argument(
		'--epsilon', dest='epsilon', default=0.05, type=float
	)
	argparser.add_argument(
		'--alpha', dest='alpha', default=1.0, type=float
	)
	argparser.add_argument(
		'--train_episodes', dest='train_episodes', default=1000, type=int
	)
	argparser.add_argument(
		'--eval_episodes', dest='eval_episodes', default=10, type=int
	)
	argparser.add_argument(
		'--eval_every', dest='eval_every', default=10, type=int
	)
	argparser.add_argument(
		'--plot', dest='plot_scores', action="store_true"
	)
	argparser.add_argument(
		'--final_show', dest='final_show', action="store_true"
	)
	argparser.add_argument(
		'--fps', dest='fps', default=20, type=int
	)

	# training strategies
	argparser.add_argument(
		'--agent_strategy', dest='agent_strategy', default="eps_greedy", type=str
	)
	argparser.add_argument(
		'--choose_unexplored_first', dest='choose_unexplored_first', action="store_true"
	)
	argparser.add_argument(
		'--opponent_strategy', dest='opponent_strategy', default="random", type=str
	)
	argparser.add_argument(
		'--mirror', dest='mirror_actions', action="store_true"
	)

	return argparser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	main(args)
