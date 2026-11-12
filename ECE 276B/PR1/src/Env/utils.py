import os
import numpy as np
import gymnasium as gym
import pickle
import matplotlib.pyplot as plt
import imageio
import random
from enum import Enum
from minigrid.core.world_object import Goal, Key, Door

from Config import cfgG

class Act(Enum):
    MF = 1  # Move Forward
    TL = 2  # Turn Left
    TR = 3  # Turn Right
    PK = 4  # Pick Key
    UD = 5  # Unlock Door

def step_cost(action):
    """
    Template (stage cost): map each action index to a nonnegative scalar.

    - Keep this consistent with the stage cost l(x, u) in your report and DP
    - Non-uniform costs (e.g., different prices for turn vs. forward vs. unlock)
    - Invalid or no-op steps: decide in your formulation whether they incur cost
    """
    # STUDENT: placeholder only (uniform zero cost). Remove when implementing.
    cost_dict = {
        Act.MF: 3,
        Act.TL: 1,
        Act.TR: 1,
        Act.PK: 2,
        Act.UD: 5,
    }
    return cost_dict[action]


def step(env, action):
    """
    Take Action
    ----------------------------------
    actions:
        0 # Move forward (MF)
        1 # Turn left (TL)
        2 # Turn right (TR)
        3 # Pickup the key (PK)
        4 # Unlock the door (UD)
    """
    actions = {
        Act.MF: env.unwrapped.actions.forward,
        Act.TL: env.unwrapped.actions.left,
        Act.TR: env.unwrapped.actions.right,
        Act.PK: env.unwrapped.actions.pickup,
        Act.UD: env.unwrapped.actions.toggle,
    }

    (obs, reward, terminated, truncated, info) = env.step(actions[action])
    return step_cost(action), terminated


def native_actions_to_act(env, actions):
    ua = env.unwrapped.actions
    mapping = {
        ua.forward: Act.MF,
        ua.left: Act.TL,
        ua.right: Act.TR,
        ua.pickup: Act.PK,
        ua.toggle: Act.UD,
    }
    return [mapping[a] for a in actions]


def generate_random_env(seed, task):
    """
    Generate a random environment for testing
    -----------------------------------------
    seed:
        A Positive Integer,
        the same seed always produces the same environment
    task:
        'MiniGrid-DoorKey-5x5-v0'
        'MiniGrid-DoorKey-6x6-v0'
        'MiniGrid-DoorKey-8x8-v0'
    """
    if seed < 0:
        seed = np.random.randint(50)
    env = gym.make(task, render_mode="rgb_array")
    env.reset(seed=seed)
    return env


def load_env(path):
    """
    Load Environments
    ---------------------------------------------
    Returns:
        gym-environment, info
    """
    with open(path, "rb") as f:
        env = pickle.load(f)

    info = {
        "height": env.unwrapped.height, 
        "width": env.unwrapped.width, 
        "init_agent_pos": env.unwrapped.agent_pos, 
        "init_agent_dir": env.unwrapped.dir_vec,
    }

    door_cells = []
    for i in range(env.unwrapped.height):
        for j in range(env.unwrapped.width):
            cell = env.unwrapped.grid.get(j, i)
            if isinstance(cell, Key):
                info["key_pos"] = np.array([j, i])
            elif isinstance(cell, Door):
                door_cells.append((j, i))
            elif isinstance(cell, Goal):
                info["goal_pos"] = np.array([j, i])
    door_cells.sort(key=lambda p: (p[1], p[0]))
    if door_cells:
        info["door_positions"] = [np.array(p) for p in door_cells]
        info["door_pos"] = np.array(door_cells[0])

    return env, info


def load_random_env(env_folder):
    """
    Load a random DoorKey environment
    ---------------------------------------------
    Returns:
        gym-environment, info
    """
    env_list = [os.path.join(env_folder, env_file) for env_file in os.listdir(env_folder) if env_file.endswith(".env")]
    env_path = random.choice(env_list)
    with open(env_path, "rb") as f:
        env = pickle.load(f)

    info = {
        "height": env.unwrapped.height,
        "width": env.unwrapped.width,
        "init_agent_pos": env.unwrapped.agent_pos,
        "init_agent_dir": env.unwrapped.dir_vec,
        "door_pos": [],
        "door_open": [],
    }

    grid = env.unwrapped.grid
    door_cells = []
    for i in range(env.unwrapped.height):
        for j in range(env.unwrapped.width):
            cell = grid.get(j, i)
            if isinstance(cell, Key):
                info["key_pos"] = np.array([j, i])
            elif isinstance(cell, Door):
                door_cells.append((j, i, cell.is_open))
            elif isinstance(cell, Goal):
                info["goal_pos"] = np.array([j, i])
    door_cells.sort(key=lambda t: (t[1], t[0]))
    for j, i, is_open in door_cells:
        info["door_pos"].append(np.array([j, i]))
        info["door_open"].append(bool(is_open))
    if door_cells:
        info["door_positions"] = [np.array([j, i]) for j, i, _ in door_cells]

    return env, info, env_path


def save_env(env, path):
    with open(path, "wb") as f:
        pickle.dump(env, f)


def plot_env(env):
    """
    Plot current environment
    ----------------------------------
    """
    img = env.render()
    plt.figure()
    plt.imshow(img)
    plt.show()


def draw_gif_from_seq(seq, env, file_name="doorkey"):
    """
    Save gif with a given action sequence
    ----------------------------------------
    seq:
        Action sequence, e.g [0,0,0,0] or [MF, MF, MF, MF]

    env:
        The doorkey environment
    """
    path = cfgG.out_dir / "gif" / f"{file_name}.gif"
    with imageio.get_writer(path, mode="I", duration=0.8) as writer:
        img = env.render()
        writer.append_data(img)
        for act in seq:
            step(env, act)
            img = env.render()
            writer.append_data(img)
    print(f"GIF is written to {path}")
    return
