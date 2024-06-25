import jax.numpy as jnp
import numpy as np

from src.singletons.hyperparameters import Args


def test_world_model(agent, env, world_model):
    simple = Args().args.algorithnm == "simple"
    dreamer = Args().args.algorithnm == "dreamer"
    actions = jnp.zeros((Args().args.num_envs, ))
    for i in range(50):
        action = agent.select_action()
        observation, reward, terminated, truncated, infos = env.step(action)
        agent.receive_reward(reward)
        agent.receive_state(observation)
        dones = terminated | truncated
        agent.receive_done(dones)

    #
    #     if simple:
    #         image
    #     np.save(f"realf{i}.npy", )
    # np.save("bf2.npy", frames[:, :, :, 1])
    # np.save("bf3.npy", frames[:, :, :, 2])
    # np.save("bf4.npy", frames[:, :, :, 3])
    # print(frames[0, 0, 0, 0])
    # frames = jnp.roll(frames, -1, axis=-1)
    # frames = frames.at[:, :, :, -1:].set(next_frame)
    # np.save("f1.npy", frames[:, :, :, 0])
    # np.save("f2.npy", frames[:, :, :, 1])
    # np.save("f3.npy", frames[:, :, :, 2])
    # np.save("f4.npy", frames[:, :, :, 3])

def test():
    #test_baseline()
    #test_mario_env()
    # test_ppo()
    #test_dqn()
    #test_stochastic_autoencoder()
    #test_mario_make()
    test_frame_stack()


if __name__ == '__main__':
    test()