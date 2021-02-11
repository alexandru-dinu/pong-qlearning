import argparse
import pickle

from world import *


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
        "--canvas_size", dest="canvas_size", nargs=2, default=(32, 24), type=int
    )
    argparser.add_argument("--paddle_length", dest="paddle_length", default=7, type=int)
    argparser.add_argument("--velocity", dest="velocity", default=1, type=int)
    argparser.add_argument("--filename", dest="filename", default="", type=str)
    argparser.add_argument("--load", dest="load", action="store_true")

    # qlearning params
    argparser.add_argument("--max_iter", dest="max_iter", default=1000, type=int)
    argparser.add_argument(
        "--learning_rate", dest="learning_rate", default=0.3, type=float
    )
    argparser.add_argument("--discount", dest="discount", default=0.99, type=float)
    argparser.add_argument("--epsilon", dest="epsilon", default=0.05, type=float)
    argparser.add_argument("--alpha", dest="alpha", default=1.0, type=float)
    argparser.add_argument(
        "--train_episodes", dest="train_episodes", default=1000, type=int
    )
    argparser.add_argument(
        "--eval_episodes", dest="eval_episodes", default=10, type=int
    )
    argparser.add_argument("--eval_every", dest="eval_every", default=10, type=int)
    argparser.add_argument("--plot", dest="plot_scores", action="store_true")
    argparser.add_argument("--final_show", dest="final_show", action="store_true")
    argparser.add_argument("--fps", dest="fps", default=20, type=int)

    # training strategies
    argparser.add_argument(
        "--agent_strategy", dest="agent_strategy", default="eps_greedy", type=str
    )
    argparser.add_argument(
        "--choose_unexplored_first", dest="choose_unexplored_first", action="store_true"
    )
    argparser.add_argument(
        "--opponent_strategy", dest="opponent_strategy", default="random", type=str
    )
    argparser.add_argument("--mirror", dest="mirror_actions", action="store_true")

    return argparser.parse_args()


if __name__ == "__main__":
    main(parse_args())
