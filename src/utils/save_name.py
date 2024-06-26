import time

from src.singletons.hyperparameters import Args


def save_name():
    args = Args().args
    run_name = f"{args.seed}_{time.asctime(time.localtime(time.time())).replace('  ', ' ').replace(' ', '_')}"
    type_name = "hybrid" if Args().args.hybrid_learning else "not_hybrid"
    dones_name = "dones" if Args().args.predict_dones else "no_dones"
    agent_count = Args().args.num_agents

    return f"{args.env}/{args.algorithm}/{type_name}_{agent_count}_{dones_name}_{run_name}"
