import os

from src.altdreamer.dreamer import Dreamer
from src.enviroment import Shape, make_env

os.environ["MUJOCO_GL"] = "egl"

import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def main(config_file):
    agent = Dreamer(
            Shape()[0], True, Shape()[1], SummaryWriter(), "cuda"
    )
    env = make_env()
    agent.train(env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="dmc-walker-walk.yml",
        help="config file to run(default: dmc-walker-walk.yml)",
    )
    main(parser.parse_args().config)