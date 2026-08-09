import gymnasium as gym
import numpy as np
import random

env = gym.make("CartPole-v1", render_mode="human")

obs, info = env.reset()

epsilon = 0.1
alpha = 0.9
gamma = 0.9

print("Initial observation:", obs)

Q = np.zeros((10,10,10,10,2))

def choose_action(state):
    p1 ,p2, p3, p4 = state
    if random.random() < epsilon:
        return random.randint(0, 1)

    return np.argmax(Q[p1,p2,p3,p4])

def poleAngle(polAngleVal,polAngleMin_val,polAngleMax_val,polAngleNum_buckets):
        normalized = (
            polAngleVal - polAngleMin_val
        ) / (
            polAngleMax_val - polAngleMin_val
        )

        bucket = int(
            normalized * polAngleNum_buckets
        )
        return bucket

def cartPosition(cartPosVal,cartPosMin_val,cartPosMax_val,cartPosNum_buckets):
        normalized = (
            cartPosVal - cartPosMin_val
        ) / (
            cartPosMax_val - cartPosMin_val
        )

        bucket = int(
            normalized * cartPosNum_buckets
        )
        return bucket

def cartVelocity(cartVelVal,cartVelMin_val,cartVelMax_val,cartVelNum_buckets):
        normalized = (
            cartVelVal - cartVelMin_val
        ) / (
            cartVelMax_val - cartVelMin_val
        )

        bucket = int(
            normalized * cartVelNum_buckets
        )
        return bucket

def poleVelocity(polVelVal,polVelMin_val,polVelMax_val,polVelNum_buckets):
        normalized = (
            polVelVal - polVelMin_val
        ) / (
            polVelMax_val - polVelMin_val
        )

        bucket = int(
            normalized * polVelNum_buckets
        )
        return bucket

def updateQ(curState,nextState,action,reward):
  
    best_next_q = np.max(Q[nextState])

    Q[curState, action] = (
        Q[curState, action]
        + alpha * (
            reward
            + gamma * best_next_q
            - Q[curState, action]
        )
    )


obs = [poleAngle(obs[2],-0.2,0.2,9),
poleVelocity(obs[3],-0.10,0.10,9),
cartVelocity(obs[1],-1.0,1.0,9),
cartPosition(obs[0],-1.0,1.0,9)]

for step in range(20):
    state = obs
    action = choose_action(state)

    newObs, reward, terminated, truncated, info = env.step(action)

    newState = [poleAngle(newObs[2],-0.2,0.2,9),
    poleVelocity(newObs[3],-0.10,0.10,9),
    cartVelocity(newObs[1],-1.0,1.0,9),
    cartPosition(newObs[0],-1.0,1.0,9)]
  
    updateQ(state,newState,action,reward)
    state = newState
    print(
        f"Step {step+1} | "
        f"Obs: {obs} | "
        f"Action: {action} |" 
        f"Reward: {reward}"
    )
    if terminated or truncated:
        print("Episode ended")
        break

env.close()