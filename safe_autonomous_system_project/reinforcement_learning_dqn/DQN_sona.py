import gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import pandas as pd
import os

# Hyperparameters
BUFFER_SIZE = 100000
#BATCH_SIZE = 64
#GAMMA = 0.99
#LR = 0.001
EPSILON = 0.1
TARGET_UPDATE = 10
EPISODES = 1000
TEST_EPISODES = 100
# other parameters asre passed as arguments
device = torch.device("cpu")

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




#Replay buffer : to store experience
# Implemented using the deque 
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    

# Epsilon-greedy policy, this step select the best action based on the max  value
def select_action(state, policy_net, epsilon, action_size):
    if random.random() < epsilon:
        return random.randrange(action_size)
    else:
        with torch.no_grad():
            return policy_net(torch.FloatTensor(state)).argmax().item()
        



# Training dqn

def train_dqn(env,LR, BATCH_SIZE, GAMMA):
    # 4: position, velocity, pole angle, pole angular velocity (from gym doc)
    state_size = env.observation_space.shape[0]
    # 2 actions: Left and Right
    action_size = env.action_space.n

    # Initialize policy and target models
    # the traget network will be a copy of the policy model
    policy_model = DQN(state_size, action_size).to(device)
    target_model = DQN(state_size, action_size).to(device)
    target_model.load_state_dict(policy_model.state_dict())  

    # # adding optizer = Adam choosen because it have adaptive learning, works well with noisy gradiant 
    optimizer = optim.Adam(policy_model.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    loss_fn = nn.MSELoss()

    rewards_list = []
    episode_losses = []
    

    # starting the episode loop, and resetiing the env
    for episode in range(EPISODES):
        state, _ = env.reset()
        done = False
        total_reward = 0
        episode_loss = 0
        step = 0

        # we use here the greedy policy to choose the best action
        # next state, reward all get upadted based on the action we take 
        # Save the experience in the replay buffer
        # Update the reward
        while not done:
            # Epsilon-greedy action selection
            action = select_action(state, policy_model, EPSILON, action_size)
            next_state, reward, done, _, _ = env.step(action)

            #print(f"Step {step + 1}: Action: {action}, State: {next_state}, Reward: {reward}, Done: {done}")
            step += 1

            replay_buffer.push((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            
            # Check there are enough samples in the replay buffer
            #sampling the transitions and converting to tensors
            #Compute the predicted q values based on policy model
            # compute the next state q values with target model and using the bellmans eqn compute the target value
            if len(replay_buffer) >= BATCH_SIZE:
                
                batch = replay_buffer.sample(BATCH_SIZE)
                states, actions, rewards_, next_states, dones = zip(*batch)

                states = torch.FloatTensor(np.array(states)).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards_ = torch.FloatTensor(rewards_).to(device)
                next_states = torch.FloatTensor(np.array(next_states)).to(device)
                dones = torch.FloatTensor(dones).to(device)

           
                q_values = policy_model(states).gather(1, actions).squeeze()

                
                next_q_values = target_model(next_states).max(1)[0]
                targets = rewards_ + (1 - dones) * GAMMA * next_q_values

                
                loss = loss_fn(q_values, targets.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                episode_loss += loss.item()  # Add loss for this step

            if step >= 1000:
                break

        
        

        rewards_list.append(total_reward)
        episode_losses.append(episode_loss)
        

        
        if episode % TARGET_UPDATE == 0:
            target_model.load_state_dict(policy_model.state_dict())

        print(f"Episode {episode}: Total Reward = {total_reward}")

    return policy_model, rewards_list, episode_losses


# MAIN
env = gym.make("CartPole-v1", render_mode=None)
# basic model which is in the main question, done it passing parameters, so I can resue same for finding the best model 
model1, rewards_list , episode_losses = train_dqn(env,LR=0.01, BATCH_SIZE=64, GAMMA=0.99)




# Subquestion 3 -finding the best setting


#  experimentation
results_summary = []

# Hyperparameters to test (low, medium, high)
#learning_rates = [0.0001, 0.001, 0.01]  
#batch_sizes = [32, 64, 128]  
#gamma_values = [0.9, 0.99, 0.999]  



# Define the selected 9 combinations of hyperparameters due to time limit
selected_combinations =selected_combinations = [
    
    {'LR': 0.0001, 'BATCH_SIZE': 128, 'GAMMA': 0.999},
    {'LR': 0.001, 'BATCH_SIZE': 64, 'GAMMA': 0.99},
    {'LR': 0.01, 'BATCH_SIZE': 32, 'GAMMA': 0.9}, 
    {'LR': 0.001, 'BATCH_SIZE': 128, 'GAMMA': 0.9},
    {'LR': 0.01, 'BATCH_SIZE': 64, 'GAMMA': 0.99},
    {'LR': 0.0001, 'BATCH_SIZE': 32, 'GAMMA': 0.99},
    {'LR': 0.01, 'BATCH_SIZE': 128, 'GAMMA': 0.999},
    {'LR': 0.001, 'BATCH_SIZE': 32, 'GAMMA': 0.999},
    {'LR': 0.0001, 'BATCH_SIZE': 64, 'GAMMA': 0.9}
]

results_summary = []

for combination in selected_combinations:
    lr = combination['LR']
    batch_size = combination['BATCH_SIZE']
    gamma = combination['GAMMA']
    
    print(f"Training with LR={lr}, BATCH_SIZE={batch_size}, GAMMA={gamma}")
    
    model, rewards_list, episode_losses = train_dqn(env, LR=lr, BATCH_SIZE=batch_size, GAMMA=gamma)
    
    last_100_avg_reward = np.mean(rewards_list[-100:]) if len(rewards_list) >= 100 else np.mean(rewards_list)
    max_reward = max(rewards_list)

    results_summary.append({
        'Learning Rate': lr,
        'Batch Size': batch_size,
        'Gamma': gamma,
        'Avg Reward (Last 100)': last_100_avg_reward,
        'Max Reward': max_reward
    })

    # Plot MSE loss over episodes 
    plt.figure(figsize=(10, 5))
    plt.plot(episode_losses)
    plt.title(f"MSE Training Loss Over Episodes (LR={lr}, BATCH_SIZE={batch_size}, GAMMA={gamma})")
    plt.xlabel("Episode")
    plt.ylabel("MSE Loss")
    plt.savefig(f'MSE_Loss_LR{lr}_BS{batch_size}_GAMMA{gamma}.png')
    plt.close()  

    # Plot reward 
    window = 50
    moving_avg = [np.mean(rewards_list[max(0, i - window):(i + 1)]) for i in range(len(rewards_list))]
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_list, label='Reward')
    plt.plot(moving_avg, label='Moving Average')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.title(f'Episode Rewards (LR={lr}, BATCH_SIZE={batch_size}, GAMMA={gamma})')
    plt.savefig(f'moving_avg_LR{lr}_BS{batch_size}_GAMMA{gamma}.png')
    plt.close()

# Save results to CSV
df = pd.DataFrame(results_summary)
df.to_csv("dqn_hyperparameter_results.csv", index=False)
print("Results saved to dqn_hyperparameter_results.csv")


#Sub question 1 : retrain on the best setting and plotting the MSE loss over episode
#Retrain 
#The  best setting I got from training 9 models isLR =0.001, Batch size=64 , gamma=0.99
print("\nRetraining with best hyperparameters...")
best_model, best_rewards_list, best_episode_losses = train_dqn(
    env,
    LR=0.001,
    BATCH_SIZE=64,
    GAMMA=0.99
)
plt.figure(figsize=(10, 5))
plt.plot(best_episode_losses)
plt.title(f"MSE Training Loss Over Episodes (Best Setting)")
plt.xlabel("Episode")
plt.ylabel("MSE Loss")
plt.savefig("MSE_Loss_BestSetting.png")



# Subquestion 2: Raw reward and moving aveare over window of 50
window = 50
moving_avg = [np.mean(best_rewards_list[max(0, i - window):(i + 1)]) for i in range(len(best_rewards_list))]
plt.figure(figsize=(10, 5))
plt.plot(best_rewards_list, label='Reward')
plt.plot(moving_avg, label='Moving Average', linestyle='--')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.legend()
plt.title(f'Episode Rewards (Best Setting)')
plt.savefig("Rewards_BestSetting.png")




# saving the model
#save_path = r"C:\Users\sonam\OneDrive\Desktop\Autonomous_spring _2025\hw3\best_dqn_model.pth"
#os.makedirs(os.path.dirname(save_path), exist_ok=True)
#torch.save(best_model.state_dict(), save_path)
#print(f" Model saved to: {save_path}")

# SUBQUESTION 4
# max step is set to 1000
print("Running Subquestion 4: Plotting position and angle over time...")

state, _ = env.reset()
done = False
positions = []
angles = []

max_steps = 1000  
for step in range(max_steps):
    action = select_action(state, best_model, epsilon=0.0, action_size=env.action_space.n)
    next_state, reward, done, _, _ = env.step(action)

    positions.append(state[0])  
    angles.append(np.degrees(state[2]))  

    state = next_state

    if done:
        break

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(positions, label='Position')
plt.title("Cart Position Over Time")
plt.xlabel("Time Step")
plt.ylabel("Position")

plt.subplot(1, 2, 2)
plt.plot(angles, label='Angle', color='orange')
plt.title("Pole Angle Over Time")
plt.xlabel("Time Step")
plt.ylabel("Angle (degrees)")

plt.tight_layout()
plt.savefig("Position_Angle_Over_Time.png")


print("Saved plot: Position_Angle_Over_Time.png")

print("\nRunning Subquestion 5: Testing for 100 episodes...")

# subquestion 5
# max step is fixed as 1000, test episode as 100
test_rewards = []
max_steps = 1000  

for episode in range(100):
    print(f"Running Episode {episode + 1}...")

    state, _ = env.reset()
    done = False
    total_reward = 0

    for step in range(max_steps):
        action = select_action(state, best_model, epsilon=0.0, action_size=env.action_space.n)
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        state = next_state

        if done:
            break

    test_rewards.append(total_reward)
    print(f"Episode {episode + 1} → Total Reward: {total_reward}")

plt.figure(figsize=(8, 6))
plt.boxplot(test_rewards, vert=True, patch_artist=True)
plt.title("Box & Whisker Plot of Test Episode Rewards (ε = 0)")
plt.ylabel("Total Reward per Episode")
plt.savefig("BoxWhisker_TestRewards.png")
print("Saved plot: BoxWhisker_TestRewards.png")





