
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# ==========================================
# CHAPTER 15: REINFORCEMENT LEARNING
# ==========================================

ROOT = Path(__file__).resolve().parent

# STEP 1: CREATE ENVIRONMENT

env = gym.make("FrozenLake-v1", is_slippery=True)

states = env.observation_space.n
actions = env.action_space.n

print("\nCHAPTER 15: Q-LEARNING")
print("States:", states)
print("Actions:", actions)

# STEP 2: CREATE Q-TABLE

q_table = np.zeros((states, actions))

print("\nInitial Q-Table:")
print(q_table)

# STEP 3: TRAINING SETTINGS

episodes = 30000
learning_rate = 0.8
discount_factor = 0.95

epsilon = 1.0
epsilon_decay = 0.9997
epsilon_min = 0.01

rng = np.random.default_rng(42)

rewards = []

# STEP 4: TRAIN THE AGENT

print("\nTraining started...")

for episode in range(episodes):

    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:

        # Exploration or exploitation
        if rng.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(q_table[state]))

        new_state, reward, terminated, truncated, info = (
            env.step(action)
        )

        done = terminated or truncated

        # Bellman update
        future_reward = (
            0 if done
            else np.max(q_table[new_state])
        )

        old_value = q_table[state, action]

        q_table[state, action] = old_value + learning_rate * (
            reward
            + discount_factor * future_reward
            - old_value
        )

        state = new_state
        total_reward += reward

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

    rewards.append(total_reward)

    if (episode + 1) % 5000 == 0:
        print(f"Completed {episode + 1} episodes")

print("\nTRAINING COMPLETE")

# STEP 5: DISPLAY Q-TABLE

print("\nFinal Q-Table:")
print(np.round(q_table, 3))

np.savetxt(
    ROOT / "q_table.csv",
    q_table,
    delimiter=",",
    fmt="%.6f"
)

np.save(ROOT / "q_table.npy", q_table)

# STEP 6: DISPLAY LEARNED POLICY

directions = ["←", "↓", "→", "↑"]

# Standard FrozenLake map
layout = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]

print("\nLEARNED POLICY")

for row in range(4):

    symbols = []

    for col in range(4):

        state = row * 4 + col
        tile = layout[row][col]

        if tile in ("H", "G"):
            symbols.append(tile)
        else:
            best_action = np.argmax(q_table[state])
            symbols.append(directions[best_action])

    print("  ".join(symbols))

# STEP 7: TEST THE AGENT

test_episodes = 100
successes = 0

for episode in range(test_episodes):

    state, info = env.reset()
    done = False

    while not done:

        action = int(np.argmax(q_table[state]))

        new_state, reward, terminated, truncated, info = (
            env.step(action)
        )

        if reward == 1:
            successes += 1

        state = new_state
        done = terminated or truncated

success_rate = successes / test_episodes * 100

print("\nTEST RESULTS")
print("Test episodes:", test_episodes)
print("Successful episodes:", successes)
print(f"Success rate: {success_rate:.1f}%")

# STEP 8: PLOT LEARNING PROGRESS

window = 500

moving_average = np.convolve(
    rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(10, 5))
plt.plot(moving_average)

plt.title("Q-Learning Progress")
plt.xlabel("Training Episode")
plt.ylabel("Average Reward (500 Episodes)")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    ROOT / "learning_progress.png",
    dpi=300
)

plt.show()
plt.close()

# STEP 9: VISUALIZE THE Q-TABLE

plt.figure(figsize=(8, 6))
plt.imshow(q_table, aspect="auto", cmap="viridis")

plt.colorbar(label="Q-Value")
plt.xlabel("Action")
plt.ylabel("State")
plt.title("Learned Q-Table")

plt.xticks(
    range(4),
    ["LEFT", "DOWN", "RIGHT", "UP"]
)

plt.yticks(range(16))
plt.tight_layout()

plt.savefig(
    ROOT / "q_table_heatmap.png",
    dpi=300
)

plt.show()
plt.close()

env.close()

print("\nPROJECT COMPLETE")
print("Files saved to:", ROOT)
