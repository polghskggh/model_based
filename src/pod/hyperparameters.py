hyperparameters = {
    'simple': {
        'deterministic_lr': 0.0001,
        'stochastic_lr': 0.0001,
        'lstm_lr': 0.0001,
        'kl_loss_weight': 2,
        'batch_size': 5,
        'data_size': 10,
        'rollout_length': 3,
        'kl_loss': 0.8,
        'pixel_reward': 0.5,
        'parallel_agents': 2,
        'agent_updates': 4
    },
    'dqn': {
        'batch_size': 100,
        'batches_per_update': 5,
        'discount_factor': 0.95,
        'target_update_period': 10,
        'start_steps': 100,
        'update_every': 50,
    },
    'ppo': {
        'batch_size': 300,
        'number_of_trajectories': 5,
        'discount_factor': 0.99,
        'lambda': 0.97,
        'clip_threshold': 0.2,
        'trajectory_length': 512
    },
    'rng': {
        'dropout': 0,
        'normal': 0,
        'carry': 0,
        'params': 0,
        'action': 0,
    },
    'frame_stack': 4,
    'save_path': '/tmp/flax_ckpt',
    'max_episode_length': 512,
    # 'max_episode_length': 5,
}
