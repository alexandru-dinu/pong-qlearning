import matplotlib.pyplot as plt
import numpy as np

from game import *

AGENT, OPPONENT, BALL = 0, 1, 2
WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480

MOVE_KEYS = {
	pygame.K_UP  : (OPPONENT, -1),
	pygame.K_DOWN: (OPPONENT, 1),
	pygame.K_w   : (AGENT, -1),
	pygame.K_s   : (AGENT, 1)
}

DIRECTIONS = {
	'top_left' : [-1, -1],
	'top_right': [1, -1],
	'bot_left' : [-1, 1],
	'bot_right': [1, 1]
}

REWARDS = {
	'lose'   : -1,
	'win'    : 1,
	'default': 0
}

# how paddle's y coord moves, depending on the action
ACTIONS = {
	"UP"  : -1,
	"DOWN": 1,
	"STAY": 0
}


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

		pygame.display.set_caption('Pong Q-Learning')

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
		self.ball = Ball((self.canvas_width // 2, self.canvas_height // 2))

		# init paddles
		paddle_y = (self.canvas_height - Paddle.LENGTH) // 2
		self.paddles = [
			Paddle((0, paddle_y), "green"), Paddle((self.canvas_width - 1, paddle_y), "red")
		]

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

	def get_state(self, initial=False):
		if initial:
			self.game_reset()

		return (
			self.paddles[AGENT].get_position(),
			self.paddles[OPPONENT].get_position(),
			self.ball.get_position()
		)

	@staticmethod
	def make_q_state(state):
		return state[AGENT], state[BALL]

	def is_final_state(self, state, num_iter):
		bx, _ = state[BALL]

		return (num_iter >= self.args.max_iter) or (bx <= 0) or (bx >= self.canvas_width)

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
			self.paddles[AGENT], ACTIONS[action_agent]
		)
		opponent_y = self.handle_paddle_movement(
			self.paddles[OPPONENT], ACTIONS[action_opponent]
		)

		self.paddles[AGENT].set_position(agent_y)
		self.paddles[OPPONENT].set_position(opponent_y)

		next_state = (agent_y, opponent_y, next_ball)

		if self.is_final_state(next_state, num_iter):
			r = self.get_reward_description(next_state, num_iter)
			reward = REWARDS.get(r, 0)

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

		# store in Q my part of the state
		my_state = self.make_q_state(state)
		v = {a: Q.get((my_state, a), 0) for a in legal_actions}

		# best action
		return max(v.keys(), key=(lambda k: v[k]))

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

		state = self.get_state(initial=True)

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
		plt.ylabel("Q size")

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
			'win'    : 0,
			'lose'   : 0,
			'default': 0
		}

		for train_ep in range(1, self.args.train_episodes + 1):
			score = 0
			num_iter = 0
			state = self.get_state(initial=True)

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
			r = self.get_reward_description(state, num_iter)
			stats[r] += 1

			print(
				"[ Episode %6d / %6d ]" % (train_ep, self.args.train_episodes),
				"[ stats:", stats, "]",
				"[ w/l: %.5f ]" % (np.inf if stats['lose'] == 0 else round(stats['win'] / stats['lose'], 5))
			)
			train_scores.append(score)
			num_states.append(len(Q))

			# evaluate policy
			if train_ep % self.args.eval_every == 0:
				avg_score = .0

				for i in range(self.args.eval_episodes):
					score = self.play_game(Q)
					avg_score += score

				eval_scores.append(avg_score / self.args.eval_episodes)
		# end for each training episode

		# plot scores if needed
		if self.args.plot_scores:
			self.plot_scores(train_scores, eval_scores, num_states)

		return Q

	def final_show(self, Q):
		score = num_iter = 0

		agent_eval_strategy, opponent_eval_strategy = self.get_eval_strategies()

		state = self.get_state(initial=True)

		game_sleep = 1000 // self.args.fps

		text_offset = 15

		while not self.is_final_state(state, num_iter):
			p_act = self.choose_action_by_strategy(
				AGENT, agent_eval_strategy, Q, state, self.get_legal_actions(AGENT, state)
			)
			o_act = self.choose_action_by_strategy(
				OPPONENT, opponent_eval_strategy, Q, state, self.get_legal_actions(OPPONENT, state)
			)

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

			state, reward = self.apply_actions(state, p_act, o_act, num_iter)
			score += reward
			num_iter += 1

		print(self.get_reward_description(state, num_iter))
