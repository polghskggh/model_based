from src.singletons.hyperparameters import Args


def linear_schedule(count):
    frac = 1.0 - (count // Args().args.num_minibatches) / Args().args.num_episodes
    return Args().args.learning_rate * frac
