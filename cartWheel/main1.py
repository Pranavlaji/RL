import gymnasium as gym
import numpy as np
import random

env = gym.make("CartPole-v1")

epsilon = 0.1
alpha = 0.05
gamma = 0.9

Q = np.zeros((10, 10, 10, 10, 2))


def bucketize(value, min_val, max_val, num_buckets):
    normalized = (value - min_val) / (max_val - min_val)

    bucket = int(normalized * num_buckets)

    bucket = max(0, bucket)
    bucket = min(bucket, num_buckets - 1)

    return bucket


def get_state(obs):
    cart_pos = bucketize(obs[0], -1.0, 1.0, 10)
    cart_vel = bucketize(obs[1], -4.0, 2.0, 10)
    pole_angle = bucketize(obs[2], -0.51, 0.51, 10)
    pole_vel = bucketize(obs[3], -3.5, 5.5, 10)

    return (
        cart_pos,
        cart_vel,
        pole_angle,
        pole_vel
    )


def choose_action(state):

    if random.random() < epsilon:
        return random.randint(0, 1)

    return np.argmax(Q[state])


def updateQ(curState, nextState, action, reward, done):

    if done:
        target = reward
    else:
        target = reward + gamma * np.max(Q[nextState])

    Q[curState][action] += (
        alpha * (target - Q[curState][action])
    )


NUM_EPISODES = 5000

cart_positions = []
cart_velocities = []
pole_angles = []
pole_velocities = []

for episode in range(NUM_EPISODES):

    obs, info = env.reset()
    cart_positions.append(obs[0])
    cart_velocities.append(obs[1])
    pole_angles.append(obs[2])
    pole_velocities.append(obs[3])   

    state = get_state(obs)

    total_reward = 0

    while True:

        action = choose_action(state)

        newObs, reward, terminated, truncated, info = env.step(action)
        cart_positions.append(newObs[0])
        cart_velocities.append(newObs[1])
        pole_angles.append(newObs[2])
        pole_velocities.append(newObs[3])
        reward = reward - 2 * abs(newObs[2])

        nextState = get_state(newObs)

        done = terminated or truncated

        updateQ(
            state,
            nextState,
            action,
            reward,
            done
        )
        state = nextState

        total_reward += reward

        if terminated or truncated:
            break

    if episode % 50 == 0:
        print(
            f"Episode {episode} | "
            f"Reward: {total_reward}"
        )
    epsilon = max(0.1, epsilon * 0.999)

print("\nTraining Complete\n")

print("\nObserved Ranges:")
print("Cart Position :", min(cart_positions), max(cart_positions))
print("Cart Velocity :", min(cart_velocities), max(cart_velocities))
print("Pole Angle    :", min(pole_angles), max(pole_angles))
print("Pole Velocity :", min(pole_velocities), max(pole_velocities))
env = gym.make("CartPole-v1", render_mode="human")

epsilon = 0

obs, info = env.reset()

state = get_state(obs)

print("\nTesting Learned Policy\n")


while True:

    action = choose_action(state)

    obs, reward, terminated, truncated, info = env.step(action)

    state = get_state(obs)

    print(
        f"State: {state} | "
        f"Action: {action}"
    )

    if terminated or truncated:
        print("Episode Ended")
        break

env.close()