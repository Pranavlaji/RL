import random
import numpy as np

GRID_SIZE = 4

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

alpha = 0.9
gamma = 0.9
epsilon = 0.1

agent_pos = [0, 0]
goal_pos = [3, 3]

obstacles = [
    [1, 1],
    [2, 2]
]

Q = np.zeros((16, 4))

action_names = {
    UP: "UP",
    DOWN: "DOWN",
    LEFT: "LEFT",
    RIGHT: "RIGHT"
}

def render():
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):

            if [r, c] == agent_pos:
                print("A", end=" ")

            elif [r, c] == goal_pos:
                print("G", end=" ")

            elif [r, c] in obstacles:
                print("X", end=" ")

            else:
                print(".", end=" ")

        print()

    print("-" * 15)
    
def state_to_index(pos):
    row, col = pos
    return row * GRID_SIZE + col


def choose_action(pos):
    state = state_to_index(pos)

    if random.random() < epsilon:
        return random.randint(0, 3)

    return np.argmax(Q[state])


def update_q(current_pos, action, next_pos, reward):
    state = state_to_index(current_pos)
    next_state = state_to_index(next_pos)

    best_next_q = np.max(Q[next_state])

    Q[state, action] = (
        Q[state, action]
        + alpha * (
            reward
            + gamma * best_next_q
            - Q[state, action]
        )
    )


def step(action):
    global agent_pos

    current_pos = agent_pos.copy()

    row, col = current_pos
    new_row, new_col = row, col

    if action == UP:
        new_row -= 1
    elif action == DOWN:
        new_row += 1
    elif action == LEFT:
        new_col -= 1
    elif action == RIGHT:
        new_col += 1

    # Wall
    if (
        new_row < 0 or
        new_row >= GRID_SIZE or
        new_col < 0 or
        new_col >= GRID_SIZE
    ):
        reward = -1
        next_pos = current_pos

        update_q(current_pos, action, next_pos, reward)

        return next_pos, reward, False

    # Obstacle
    if [new_row, new_col] in obstacles:
        reward = -10
        next_pos = current_pos

        update_q(current_pos, action, next_pos, reward)

        return next_pos, reward, False

    # Valid move
    next_pos = [new_row, new_col]
    agent_pos = next_pos

    if next_pos == goal_pos:
        reward = 100

        update_q(current_pos, action, next_pos, reward)

        return next_pos, reward, True

    reward = -1

    update_q(current_pos, action, next_pos, reward)

    return next_pos, reward, False


NUM_EPISODES = 1000
MAX_STEPS = 100

for episode in range(NUM_EPISODES):

    agent_pos = [0, 0]
    total_reward = 0

    for step_num in range(MAX_STEPS):

        action_taken = choose_action(agent_pos)

        state, reward, done = step(action_taken)

        total_reward += reward

        if done:
            break

    if episode % 100 == 0:
        print(
            f"Episode {episode} | "
            f"Reward: {total_reward}"
        )

print("\nTraining complete.\n")
print(Q)

epsilon = 0

agent_pos = [0, 0]

print("Testing Learned Policy:\n")
render()

for step_num in range(20):

    action_taken = choose_action(agent_pos)

    state, reward, done = step(action_taken)

    print(
        f"Step {step_num + 1} | "
        f"Action: {action_names[action_taken]} | "
        f"State: {state} | "
        f"Reward: {reward}"
    )

    render()

    if done:
        print("Goal reached!")
        break