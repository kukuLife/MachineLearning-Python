import gym
from DuelingDQNPrioritizedReplay import DuelingDQNPrioritizedReplay

env = gym.make('LunarLander-v2')
env.seed(1)

#define parameters which nedded to be passed to DuelingDQNClass or control the all learning process
N_A = env.action_space.n
N_S = env.observation_space.shape[0]
MEMORY_CAPACITY = 50000
TARGET_REP_ITER = 2000
MAX_EPISONDES = 900
E_GREEDY = 0.95
E_INCREMENT = 0.00001
GAMMA = 0.99
LR = 0.0001
BATCH_SIZE = 32
HIDDEN = [400, 400]
RENDER = True

RL = DuelingDQNPrioritizedReplay(
    n_actions=N_A, n_features=N_S, learning_rate=LR, e_greedy=E_GREEDY, reward_decay=GAMMA,
    memory_size=MEMORY_CAPACITY, e_greedy_increment=E_INCREMENT)

total_steps = 0

for i in range(MAX_EPISONDES):
    s = env.reset()
    while True:
        if total_steps >= MEMORY_CAPACITY : env.render()
        a = RL.choose_action(s)
        s_, r, done,_ = env.step(a)

        if done:
            break

        s = s_
        total_steps += 1