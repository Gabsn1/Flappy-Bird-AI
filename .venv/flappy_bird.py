import os
import random
from collections import deque

import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# ==========================================
# 1. ZUSTAND
# ==========================================
# FlappyBird-v0 nutzt standardmaessig use_lidar=True und liefert dann
# 180 Lidar-Werte. 
#
# Mit use_lidar=False bekommen wir die 12 einfachen, normalisierten Features:
# 0-8: Roehrenpositionen, 9: bird_y, 10: velocity_y, 11: rotation.
#
# Fuer das Netz rechnen wir daraus zusaetzliche Ziel-Features aus. Dadurch muss
# die KI nicht selbst aus top/bottom erst die Lochmitte "erraten".
RAW_INPUT_DIM = 12
PIPE_COUNT = 3
ENGINEERED_PER_PIPE = 4
INPUT_DIM = RAW_INPUT_DIM + PIPE_COUNT * ENGINEERED_PER_PIPE + 3
OUTPUT_DIM = 2  # 0 = nichts tun, 1 = flappen


def pipe_values(state, pipe_index):
    base = pipe_index * 3
    pipe_x = float(state[base])
    pipe_top_y = float(state[base + 1])
    pipe_bottom_y = float(state[base + 2])
    gap_center_y = (pipe_top_y + pipe_bottom_y) * 0.5
    return pipe_x, pipe_top_y, pipe_bottom_y, gap_center_y


def next_pipe_index(state):
    """Findet die wichtigste Roehre: die naechste, die vor dem Vogel liegt."""
    candidates = []
    for pipe_index in range(PIPE_COUNT):
        pipe_x, _, _, _ = pipe_values(state, pipe_index)
        if pipe_x >= -0.05:
            candidates.append((pipe_x, pipe_index))

    if not candidates:
        return 0

    return min(candidates, key=lambda item: item[0])[1]


def extract_features(state):
    """Erweitert die Env-Features um klare Zielsignale zur Lochmitte."""
    raw = np.asarray(state, dtype=np.float32)
    bird_y = float(raw[9])
    velocity_y = float(raw[10])

    engineered = []
    for pipe_index in range(PIPE_COUNT):
        pipe_x, _, _, gap_center_y = pipe_values(raw, pipe_index)
        gap_delta = bird_y - gap_center_y
        distance_weight = 1.0 - min(max(pipe_x, 0.0), 1.0)
        engineered.extend(
            [
                gap_center_y,
                gap_delta,
                abs(gap_delta),
                gap_delta * distance_weight,
            ]
        )

    next_index = next_pipe_index(raw)
    next_x, _, _, next_center_y = pipe_values(raw, next_index)
    next_gap_delta = bird_y - next_center_y

    return np.asarray(
        [
            *raw,
            *engineered,
            next_x,
            next_gap_delta,
            velocity_y * next_gap_delta,
        ],
        dtype=np.float32,
    )


def gap_error(state):
    """Abstand des Vogels zur Mitte der naechsten Roehrenluecke."""
    raw = state[:RAW_INPUT_DIM]
    pipe_index = next_pipe_index(raw)
    _, _, _, gap_center_y = pipe_values(raw, pipe_index)
    bird_y = raw[9]
    return float(abs(bird_y - gap_center_y))


def shape_reward(env_reward, state, next_state, done):
    """Kleine Zusatzbelohnung, damit der Agent nicht nur 'nicht sterben' lernt."""
    if done:
        return -10.0

    reward = float(env_reward)
    raw_next_state = next_state[:RAW_INPUT_DIM]

    old_gap_error = gap_error(state)
    new_gap_error = gap_error(next_state)
    velocity_y = abs(float(raw_next_state[10]))
    bird_y = float(raw_next_state[9])
    pipe_index = next_pipe_index(raw_next_state)
    pipe_x, _, _, _ = pipe_values(raw_next_state, pipe_index)
    pipe_close_weight = 1.0 - min(max(pipe_x, 0.0), 1.0)

    # Je naeher die Pipe ist, desto wichtiger wird die exakte Lueckenmitte.
    reward += (old_gap_error - new_gap_error) * (2.0 + 4.0 * pipe_close_weight)
    reward += max(0.0, 0.20 - new_gap_error) * (0.3 + pipe_close_weight)
    reward -= new_gap_error * 0.05 * pipe_close_weight
    reward -= velocity_y * 0.01

    # Verhindert, dass "immer flappen" oder "nie flappen" lange attraktiv bleibt.
    if bird_y < 0.05 or bird_y > 0.78:
        reward -= 0.5

    return reward


# ==========================================
# 2. NEURONALES NETZ
# ==========================================
def create_q_model(input_dim, output_dim):
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(output_dim, activation="linear"),
        ]
    )
    return model


# ==========================================
# 3. REPLAY BUFFER
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ==========================================
# 4. HYPERPARAMETER
# ==========================================
BATCH_SIZE = int(os.getenv("FLAPPY_BATCH_SIZE", "64"))
GAMMA = 0.99
LR = 0.00025
BUFFER_SIZE = 50000
MIN_BUFFER_SIZE = int(os.getenv("FLAPPY_MIN_BUFFER", "2000"))

NUM_EPISODES = int(os.getenv("FLAPPY_EPISODES", "2000"))
TRAIN_EVERY_STEPS = 4
TARGET_UPDATE_STEPS = 1000
MAX_STEPS_PER_EPISODE = int(os.getenv("FLAPPY_MAX_STEPS", "5000"))

EPS_START = float(os.getenv("FLAPPY_EPS_START", "1.0"))
EPS_MIN = float(os.getenv("FLAPPY_EPS_MIN", "0.05"))
EPS_DECAY_EPISODES = float(os.getenv("FLAPPY_EPS_DECAY_EPISODES", "250"))
RANDOM_FLAP_CHANCE = float(os.getenv("FLAPPY_RANDOM_FLAP_CHANCE", "0.35"))

# Fuer schnelles Lernen bleibt das Fenster beim Training aus.
# Zum Zuschauen nach dem Training unten RENDER_AFTER_TRAINING auf True setzen.
RENDER_TRAINING = os.getenv("FLAPPY_RENDER_TRAINING", "0") == "1"
RENDER_AFTER_TRAINING = os.getenv("FLAPPY_RENDER_DEMO", "1") == "1"
SAVE_MODEL = os.getenv("FLAPPY_SAVE_MODEL", "1") == "1"
MODEL_PATH = os.getenv("FLAPPY_MODEL", "flappy_bird_dqn_gap.keras")


# ==========================================
# 5. INITIALISIERUNG
# ==========================================
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

render_mode = "human" if RENDER_TRAINING else None
env = gym.make(
    "FlappyBird-v0",
    render_mode=render_mode,
    use_lidar=False,
    audio_on=False,
    disable_env_checker=True,
)

q_net = create_q_model(INPUT_DIM, OUTPUT_DIM)
target_net = create_q_model(INPUT_DIM, OUTPUT_DIM)
target_net.set_weights(q_net.get_weights())

optimizer = tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=1.0)
loss_fn = tf.keras.losses.Huber()
buffer = ReplayBuffer(BUFFER_SIZE)
global_step = 0


def epsilon_by_episode(episode):
    """Faellt am Anfang schnell und flacht dann gegen EPS_MIN ab."""
    progress = max(0, episode - 1)
    return EPS_MIN + (EPS_START - EPS_MIN) * np.exp(
        -progress / EPS_DECAY_EPISODES
    )


@tf.function
def train_step(states, actions, rewards, next_states, dones):
    """Ein Double-DQN Trainingsschritt."""
    next_actions = tf.argmax(q_net(next_states, training=False), axis=1)
    next_q_target_all = target_net(next_states, training=False)
    next_action_mask = tf.one_hot(next_actions, OUTPUT_DIM)
    next_q = tf.reduce_sum(next_q_target_all * next_action_mask, axis=1)
    target_q = rewards + GAMMA * next_q * (1.0 - dones)

    with tf.GradientTape() as tape:
        q_values = q_net(states, training=True)
        action_mask = tf.one_hot(actions, OUTPUT_DIM)
        current_q = tf.reduce_sum(q_values * action_mask, axis=1)
        loss = loss_fn(target_q, current_q)

    grads = tape.gradient(loss, q_net.trainable_variables)
    optimizer.apply_gradients(zip(grads, q_net.trainable_variables))
    return loss


def choose_action(state, epsilon_value):
    if random.random() < epsilon_value:
        return 1 if random.random() < RANDOM_FLAP_CHANCE else 0

    q_values = q_net(np.expand_dims(state, axis=0), training=False)
    return int(tf.argmax(q_values[0]).numpy())


# ==========================================
# 6. TRAINING
# ==========================================
print("Starte DQN-Training mit Ziel-Features zur Roehrenmitte...")

for episode in range(1, NUM_EPISODES + 1):
    raw_state, info = env.reset()
    state = extract_features(raw_state)
    epsilon = epsilon_by_episode(episode)
    episode_reward = 0.0
    current_score = 0
    losses = []

    for step_in_episode in range(MAX_STEPS_PER_EPISODE):
        global_step += 1
        action = choose_action(state, epsilon)

        next_raw_state, env_reward, terminated, truncated, info = env.step(action)
        next_state = extract_features(next_raw_state)
        done = terminated or truncated
        reward = shape_reward(env_reward, state, next_state, done)

        if info["score"] > current_score:
            current_score = info["score"]
            reward += 5.0

        buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        if len(buffer) >= MIN_BUFFER_SIZE and global_step % TRAIN_EVERY_STEPS == 0:
            batch = buffer.sample(BATCH_SIZE)
            loss = train_step(*batch)
            losses.append(float(loss.numpy()))

        if global_step % TARGET_UPDATE_STEPS == 0:
            target_net.set_weights(q_net.get_weights())

        if done:
            break

    avg_loss = float(np.mean(losses)) if losses else 0.0

    print(
        f"Episode {episode:4d} | Score {current_score:2d} | "
        f"Reward {episode_reward:7.2f} | Epsilon {epsilon:.3f} | "
        f"Buffer {len(buffer):5d} | Loss {avg_loss:.4f}"
    )

env.close()

if SAVE_MODEL:
    q_net.save(MODEL_PATH)
    print(f"Modell gespeichert: {MODEL_PATH}")


# ==========================================
# 7. DEMO NACH DEM TRAINING
# ==========================================
if RENDER_AFTER_TRAINING:
    demo_env = gym.make(
        "FlappyBird-v0",
        render_mode="human",
        use_lidar=False,
        audio_on=False,
        disable_env_checker=True,
    )

    for demo_episode in range(3):
        raw_state, info = demo_env.reset()
        state = extract_features(raw_state)
        done = False

        while not done:
            action = choose_action(state, 0.0)
            raw_state, _, terminated, truncated, info = demo_env.step(action)
            state = extract_features(raw_state)
            done = terminated or truncated

        print(f"Demo {demo_episode + 1}: Score {info['score']}")

    demo_env.close()
