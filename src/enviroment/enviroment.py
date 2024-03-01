import gymnasium as gym


def setup_env() -> gym.Env:
    return gym.make("ALE/Breakout-v5", render_mode="human")
