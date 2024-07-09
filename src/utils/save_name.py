import time

from src.singletons.hyperparameters import Args


def save_name():
    args = Args().args
    run_name = f"{args.seed}_{time.asctime(time.localtime(time.time())).replace('  ', ' ').replace(' ', '_')}"
    type_name = "hybrid" if Args().args.hybrid_learning else "not_hybrid"
    dones_name = "dones" if Args().args.predict_dones else "no_dones"
    dropout_name = "dropout" if Args().args.dropout else "no_dropout"
    simplified_obs_name = "simplified_obs" if Args().args.simplified_obs else "no_simplified_obs"
    initial_updates = Args().args.initial_updates
    agent_count = Args().args.num_agents

    return (f"{args.env}/{args.algorithm}/{type_name}_agents:{agent_count}_initsteps:{initial_updates}_{dropout_name}"
            f"_{dones_name}_{simplified_obs_name}_{run_name}")
