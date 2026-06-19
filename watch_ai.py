import os
from pathlib import Path

import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np
import tensorflow as tf


INPUT_DIM = 12
MODEL_PATH = Path(os.getenv("FLAPPY_MODEL", "flappy_bird_dqn_v1.keras"))
EPISODES = int(os.getenv("FLAPPY_WATCH_EPISODES", "10"))
RENDER = os.getenv("FLAPPY_RENDER", "1") == "1"


def extract_features(state):
    return np.asarray(state, dtype=np.float32)


def choose_action(model, state):
    q_values = model(np.expand_dims(state, axis=0), training=False)
    return int(tf.argmax(q_values[0]).numpy())


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modell nicht gefunden: {MODEL_PATH}. "
            "Trainiere zuerst mit .venv/flappy_bird.py."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    render_mode = "human" if RENDER else None
    env = gym.make(
        "FlappyBird-v0",
        render_mode=render_mode,
        use_lidar=False,
        audio_on=False,
        disable_env_checker=True,
    )

    print(f"Zeige KI aus Modell: {MODEL_PATH}")

    try:
        for episode in range(1, EPISODES + 1):
            raw_state, info = env.reset()
            state = extract_features(raw_state)
            done = False

            while not done:
                action = choose_action(model, state)
                raw_state, _, terminated, truncated, info = env.step(action)
                state = extract_features(raw_state)
                done = terminated or truncated

            print(f"Runde {episode:2d} | Score: {info['score']}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
