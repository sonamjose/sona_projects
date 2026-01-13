import gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

# Hyperparameters
BUFFER_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 0.001
EPSILON = 0.1
TARGET_UPDATE = 1000
EPISODES = 1000
TEST_EPISODES = 100

# Define the Q-Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize environment
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Initialize networks
model = DQN(state_size, action_size)
target_model = DQN(state_size, action_size)
target_model.load_state_dict(model.state_dict())
target_model.eval()

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# Replay buffer
replay_buffer = deque(maxlen=BUFFER_SIZE)

# Trackers
rewards_list = []
losses = []


for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0

    for t in range(1000):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Epsilon-greedy action selection
        if random.random() < EPSILON:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.append((state, action, reward, next_state, float(done)))

        state = next_state
        total_reward += reward

        if len(replay_buffer) > BATCH_SIZE:
            batch = random.sample(replay_buffer, BATCH_SIZE)
            states, actions, rewards_batch, next_states, dones = zip(*batch)

            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards_batch = torch.FloatTensor(rewards_batch).unsqueeze(1)
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(dones).unsqueeze(1)

            q_values = model(states).gather(1, actions)
            with torch.no_grad():
                next_q_values = target_model(next_states).max(1)[0].unsqueeze(1)
                target_q = rewards_batch + GAMMA * next_q_values * (1 - dones)

            loss = criterion(q_values, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if done:
            break

    rewards_list.append(total_reward)

    # Update target network
    if episode % 10 == 0:
        target_model.load_state_dict(model.state_dict())

    print(f"Episode {episode + 1}: Reward = {total_reward}")

# --- Plotting ---
# 1. MSE Loss over episodes
plt.figure()
plt.plot(losses)
plt.xlabel('Training Steps')
plt.ylabel('MSE Loss')
plt.title('DQN Training Loss')
plt.savefig('dqn_loss.png')

# 2. Reward and moving average
window = 50
moving_avg = [np.mean(rewards_list[max(0, i - window):(i + 1)]) for i in range(len(rewards_list))]
plt.figure()
plt.plot(rewards_list, label='Reward')
plt.plot(moving_avg, label='Moving Average')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.legend()
plt.title('Episode Rewards')
plt.savefig('dqn_rewards.png')



# --- Testing the trained model ---
positions = []
angles = []
state, _ = env.reset()
for _ in range(200):
    with torch.no_grad():
        action = model(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
    next_state, _, done, _, _ = env.step(action)
    positions.append(state[0])
    angles.append(state[2])
    state = next_state
    if done:
        break

plt.figure()
plt.plot(positions, label='Cart Position')
plt.plot(angles, label='Pole Angle')
plt.xlabel('Time Step')
plt.legend()
plt.title('Cart Position and Pole Angle Over Time')
plt.savefig('dqn_position_angle.png')

# --- Test reward distribution over 100 episodes ---
test_rewards = []
for _ in range(TEST_EPISODES):
    state, _ = env.reset()
    total = 0
    for _ in range(1000):
        with torch.no_grad():
            action = model(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
        state, reward, done, _, _ = env.step(action)
        total += reward
        if done:
            break
    test_rewards.append(total)

plt.figure()
plt.boxplot(test_rewards)
plt.ylabel('Total Reward')
plt.title('DQN Test Performance (100 Episodes)')
plt.savefig('dqn_boxplot.png')

env.close()
print("Training and testing complete. Plots saved.")
