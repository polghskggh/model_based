import time

from src.singletons.hyperparameters import Args


def save_name():
    args = Args().args
    run_name = f"{args.seed}_{time.asctime(time.localtime(time.time())).replace('  ', ' ').replace(' ', '_')}"
    return f"{args.env}/{args.algorihtm}/{args.env}/{run_name}"
