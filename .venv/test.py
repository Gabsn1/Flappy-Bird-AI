import gymnasium as gym
import flappy_bird_gymnasium
import tensorflow as tf
import numpy as np

# Gleiche Feature-Funktion wie im Trainingsskript
def extract_features(state):
    bird_y = state[0]
    pipe_x_dist = state[2]
    pipe_top_y_dist = state[3]
    pipe_bottom_y_dist = state[4]
    return np.array([bird_y, pipe_x_dist, pipe_top_y_dist, pipe_bottom_y_dist], dtype=np.float32)

# 1. Umgebung mit GUI starten
env = gym.make("FlappyBird-v0", render_mode="human", disable_env_checker=True)

# 2. Trainiertes Modell laden
model = tf.keras.models.load_model("flappy_bird_dqn.keras")

# 3. Spiel spielen lassen
state, info = env.reset()
done = False

while not done:
    features = extract_features(state)
    state_input = np.expand_dims(features, axis=0)
    
    # Das Modell sagt die Q-Werte vorher (ohne Zufall/Epsilon!)
    q_values = model(state_input, training=False)
    action = np.argmax(q_values.numpy()[0])
    
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()