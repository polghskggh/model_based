from src.singletons.hyperparameters import Args


def linear_schedule(count):
    args = Args().args
    fraction = 1.0 - (count // (args.num_minibatches * args.num_epochs)) / args.num_updates
    return args.learning_rate * fraction
