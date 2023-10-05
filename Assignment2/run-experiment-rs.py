import argparse
import gymnasium as gym
import importlib.util
import matplotlib.pyplot as plt
from tqdm import tqdm
from operator import add
import numpy as np
import scipy.stats as stats
import math

parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object",
                    default="/Users/lisasamuelsson/Documents/Chalmers/AdvancedTopics/advanced_ml_assignment/q-learning.py")
parser.add_argument("--env", type=str, help="Environment",
                    default="FrozenLake-v1")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location('Agent', args.agentfile)
agentfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentfile)

try:
    env = gym.make(args.env, is_slippery=True)
    print("Loaded ", args.env)
    print()
except:
    file_name, env_name = args.env.split(":")
    gym.envs.register(
        id=env_name + "-v0",
        entry_point=args.env,
    )
    env = gym.make(env_name + "-v0")
    print("Loaded", args.env)


action_dim = env.action_space.n
state_dim = env.observation_space.n

episodes = 10000
agents = 5


def run_agent():
    agent = agentfile.Agent(state_dim, action_dim)
    rewards = []
    mean_rewards = []

    observation = env.reset()

    for _ in tqdm(range(episodes)):
        episode_rewards = 0

        # env.render()

        # your agent here (this currently takes random actions)
        action = agent.act(observation)
        observation, reward, done, truncated, info = env.step(action)
        agent.observe(observation, reward, done)

        rewards.append(reward)
        mean_rewards.append(np.mean(rewards))

    return mean_rewards


reward_matrix = np.zeros((agents, episodes))

rewards_for_agents = np.zeros(episodes)

for i in range(agents):
    agent_rewards = run_agent()

    reward_matrix[i] = agent_rewards

mean_vector = np.mean(reward_matrix, axis=0)
std_vector = np.std(reward_matrix, axis=0)

confidence_level = 0.95

z = stats.norm.ppf((1 + confidence_level) / 2)

n = agents
confidence_interval = z * (std_vector / math.sqrt(n))

lower_bound = mean_vector - confidence_interval
upper_bound = mean_vector + confidence_interval

plt.plot(range(episodes), mean_vector, label='Mean of all agents')
plt.legend()

plt.fill_between(range(episodes), lower_bound, upper_bound,
                 alpha=0.2)
plt.xlabel("Episodes")
plt.ylabel("Average Rewards")
plt.title("Average Rewards with 95% Confidence Intervals")
plt.show()
