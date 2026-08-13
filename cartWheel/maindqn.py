import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import random


env = gym.make("CartPole-v1")

epsilon = 0.1
alpha = 0.05
gamma = 0.9

class DQN(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)
    
model = DQN()
state = torch.tensor(
    [0.0, 0.1, 0.02, -0.1],
    dtype=torch.float32
)
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
criterion = nn.MSELoss()

NUM_EPISODES = 5000

cart_positions = []
cart_velocities = []
pole_angles = []
pole_velocities = []
rewards = []

for episode in range(NUM_EPISODES):

    obs, info = env.reset()
    state = torch.tensor(obs, dtype=torch.float32)
    total_reward = 0
    
    while True:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = torch.argmax(model(state)).item()
        curQ = model(state)[action]
        newObs, reward, terminated, truncated, info = env.step(action)
        newObs = torch.tensor(newObs, dtype=torch.float32)
    
        reward = reward - 2 * abs(newObs[2])

        done = terminated or truncated
        with torch.no_grad():
            predQ = model(newObs)
            target = reward + gamma*torch.max(predQ)
        loss = criterion(curQ, target)
        state = newObs
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward += reward

        if terminated or truncated:
            break
    rewards.append(total_reward)
    if episode % 50 == 0:
        print(
            f"Episode {episode} | "
            f"Reward: {total_reward}"
        )
    if episode % 100 == 0:
        print(
            f"Episode {episode} | "
            f"Avg Reward: {np.mean(rewards[-100:]):.2f}"
        )
    epsilon = max(0.1, epsilon * 0.999)

print("\nTraining Complete\n")

env = gym.make("CartPole-v1", render_mode="human")

epsilon = 0

obs, info = env.reset()
state = torch.tensor(obs, dtype=torch.float32)

print("\nTesting Learned Policy\n")


while True:
    
    action = torch.argmax(model(state)).item()  

    obs, reward, terminated, truncated, info = env.step(action)

    print(
        f"State: {obs} | "
        f"Action: {action}"
    )
    state = torch.tensor(obs, dtype=torch.float32)
    if terminated or truncated:
        print("Episode Ended")
        break


env.close()