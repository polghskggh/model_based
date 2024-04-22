hyperparameters = {
    'world': {
        'deterministic_lr': 0.0001,
        'stochastic_lr': 0.0001,
        'lstm_lr': 0.0001,
        'kl_loss_weight': 2,
        'batch_size': 32,
    },
    'ddpg': {
        'batch_size': 100,
        'batches_per_update': 5,
        'discount_factor': 0.95,
        'polyak': 0.995,
    },
    'dqn': {
        'batch_size': 100,
        'batches_per_update': 5,
        'discount_factor': 0.95,
        'polyak': 0.995,
    },
    'ppo': {
        'discount_factor': 0.95,
        'lambda': 0.7,
        'clip_threshold': 0.2,
        'sequence_length': 10,
    },
    'rng': {
        'dropout': 0,
        'normal': 0,
        'carry': 0,
        'params': 0,
        'action': 0,
    }
}
