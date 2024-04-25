hyperparameters = {
    'world': {
        'deterministic_lr': 0.0001,
        'stochastic_lr': 0.0001,
        'lstm_lr': 0.0001,
        'kl_loss_weight': 2,
        'batch_size': 32,
        'data_size': 1000,
        'rollout_length': 10,
        'frame_stack': 4,
    },
    'dqn': {
        'batch_size': 100,
        'batches_per_update': 5,
        'discount_factor': 0.95,
        'polyak': 0.995,
    },
    'ppo': {
        'batch_size': 100,
        'discount_factor': 0.95,
        'lambda': 0.7,
        'clip_threshold': 0.2,
        'sequence_length': 1,
    },
    'rng': {
        'dropout': 0,
        'normal': 0,
        'carry': 0,
        'params': 0,
        'action': 0,
    },
    'agent': {
        'start_steps': 200,
        'update_every': 50,
    },
    'mixing_coefficients': {
        'kl_loss': 0.8,
    },
    'save_path': '/tmp/flax_ckpt',
    'max_episode_length': 1000,
}
