import argparse
import pickle

from world import Ball, Paddle, World


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

        # save Q-values if needed
        if args.filename is not None:
            pickle.dump(Q, open(args.filename, "wb"))
            print("Saved Q in ", args.filename)

        # show final game if needed
        if args.final_show:
            input("Press any key to play game...")
            world.final_show(Q)


def parse_args():
    parser = argparse.ArgumentParser()

    # game params
    parser.add_argument("--canvas_size", nargs=2, default=(32, 24), type=int)
    parser.add_argument("--paddle_length", default=7, type=int)
    parser.add_argument("--velocity", default=1, type=int)
    parser.add_argument("--filename", type=str)
    parser.add_argument("--load", action="store_true")

    # qlearning params
    parser.add_argument("--max_iter", default=1000, type=int)
    parser.add_argument("--learning_rate", default=0.3, type=float)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--epsilon", default=0.05, type=float)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--train_episodes", default=1000, type=int)
    parser.add_argument("--eval_episodes", default=10, type=int)
    parser.add_argument("--eval_every", default=10, type=int)
    parser.add_argument("--plot_scores", action="store_true")
    parser.add_argument("--final_show", action="store_true")
    parser.add_argument("--fps", default=20, type=int)

    # training strategies
    parser.add_argument("--agent_strategy", default="eps_greedy", type=str)
    parser.add_argument("--choose_unexplored_first", action="store_true")
    parser.add_argument("--opponent_strategy", default="random", type=str)
    parser.add_argument("--mirror_actions", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
