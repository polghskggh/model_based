import gymnasium as gym
import jax.numpy as jnp
from jax import lax

from src.agent.agent import Agent
from src.enviroment import Shape
from src.modelfree import interact, model_free_train_loop
from src.pod.storage import TransitionStorage, store
from src.singletons.hyperparameters import Args
from src.singletons.step_traceker import StepTracker
from src.singletons.writer import Writer
from src.worldmodel.worldmodelinterface import WorldModelInterface


def sample_env(storage, agent, envs):
    observation, _ = envs.reset()
    agent.receive_state(observation)
    writer = Writer().writer
    args = Args().args

    returns = jnp.zeros(args.num_agents)
    for step in range(args.trajectory_length):
        StepTracker().increment(args.num_agents)
        reward, done, _ = interact(agent, envs, False)
        returns += reward
        observations, actions, rewards, next_observations = agent.last_transition()
        # TODO: remove hardcoded 9
        next_observations = lax.slice_in_dim(next_observations, 9, None, -1)
        storage = store(storage, slice(step * args.num_agents, (step + 1) * args.num_agents), observations=observations,
                        actions=actions, rewards=rewards, next_observation=next_observations)

    for ret in returns:
        writer.add_scalar("charts/episodic_return", ret, int(StepTracker()))

    return storage


def model_based_train_loop(agent: Agent, world_model: WorldModelInterface, env: gym.Env):
    data_size = Args().args.batch_size * Args().args.num_epochs
    storage: TransitionStorage = TransitionStorage(observations=jnp.zeros((data_size,) + Shape()[0]),
                                                   actions=jnp.zeros((data_size)),
                                                   rewards=jnp.zeros((data_size)),
                                                   next_observation=jnp.zeros((data_size, Shape()[0][0], Shape()[0][1],
                                                                       Shape()[0][2] // Args().args.frame_stack)))

    sample_env(storage, agent, env)
    world_model.update(storage)
    model_free_train_loop(agent, world_model)

