# moon_lander_pipeline_final_v2.py
import os
import torch
import random
from collections import deque
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import time
import csv
import copy

print(torch.cuda.is_available())   # should print True
print(torch.cuda.get_device_name(0))

==============================================================================

============================= CONFIGURATION ==================================

==============================================================================

--- Training Settings ---
SKIP_TRAINING = False
GENERATE_SNAPSHOTS_DURING_TRAINING = True

--- HYPERPARAMETERS (TUNED FOR STABILITY) ---
NUM_EPISODES = 1500
BATCH_SIZE = 128
GAMMA = 0.99
LEARNING_RATE = 1e-3
MEMORY_SIZE = 50000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.996   # --- MODIFIED: Slower decay to encourage more exploration
TARGET_UPDATE_FREQ = 10 # --- NEW: How often to update the stable target network

--- File Paths & Intervals ---
MODEL_CHECKPOINT_PATH = "checkpoints/dqn_lunar_lander_stable.pth"
SNAPSHOT_INTERVAL = NUM_EPISODES // 15

==============================================================================

========================== ENVIRONMENT & MODEL ===============================

==============================================================================

class LunarEnv:
    """A custom, enhanced Lunar Lander environment."""
    def __init__(self):
        self.gravity = 1.62
        self.max_thrust = 10.0
        self.dt = 0.1
        self.max_steps = 1000
        self.landing_zone_width = 10.0
        self.reset()

    def reset(self):
        self.x = random.uniform(-20, 20)
        self.y = 100.0
        self.vx = random.uniform(-5, 5)
        self.vy = random.uniform(-5, 0)
        self.fuel = 100.0
        self.done = False
        self.steps = 0
        self.path = [(self.x, self.y)]
        return self._get_state()

    def _get_state(self):
        return np.array([self.x, self.y, self.vx, self.vy, self.fuel], dtype=np.float32)

    def step(self, action):
        thrust_x_unit, thrust_y_unit = action
        thrust_magnitude = (abs(thrust_x_unit) + abs(thrust_y_unit)) * self.dt

        if self.fuel > thrust_magnitude:
            self.fuel -= thrust_magnitude
            thrust_x = thrust_x_unit * self.max_thrust
            thrust_y = (thrust_y_unit + 1) * 0.5 * self.max_thrust
        else:
            thrust_x, thrust_y = 0.0, 0.0
            self.fuel = 0.0

        ax, ay = thrust_x, thrust_y - self.gravity
        self.vx += ax * self.dt; self.vy += ay * self.dt
        self.x += self.vx * self.dt; self.y += self.vy * self.dt
        self.steps += 1
        self.path.append((self.x, self.y))

        reward = 0
        reward -= 0.1 * (abs(self.x) / self.landing_zone_width + np.sqrt(self.vx**2 + self.vy**2))
        reward -= 0.2 * thrust_magnitude

        if self.y <= 0:
            self.y = 0
            self.done = True
            speed = np.sqrt(self.vx**2 + self.vy**2)
            if speed < 2.0 and abs(self.x) < self.landing_zone_width / 2:
                reward += 200.0
            else:
                reward -= 100.0
        elif self.steps >= self.max_steps or self.fuel <= 0 or abs(self.x) > 150:
            self.done = True
            reward -= 100.0

        return self._get_state(), reward, self.done

    def get_path(self): return np.array(self.path)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x): return self.net(x)

==============================================================================

========================== HELPERS & VISUALS =================================

==============================================================================

def ensure_dirs():
    """Create all necessary output directories."""
    for dirname in ["snapshots", "charts", "animation", "checkpoints"]:
        os.makedirs(dirname, exist_ok=True)

def save_training_charts(reward_history):
    """Saves raw and smoothed reward charts from training."""
    filename = "charts/training_rewards.png"
    plt.figure(figsize=(12, 6))
    plt.plot(reward_history, label="Reward per episode", alpha=0.6, color='deepskyblue')
    if len(reward_history) >= 100:
        smooth = pd.Series(reward_history).rolling(100, min_periods=20).mean()
        plt.plot(smooth, label="Smoothed Reward (100-ep window)", color='orangered', linewidth=2)
    plt.xlabel("Episode"); plt.ylabel("Total Reward"); plt.title("Training Progression")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"📊 Training charts saved to '{filename}'")

def save_clustered_chart(metrics, path_len):
    """Saves a bar chart of final metrics for a successful landing."""
    filename = "charts/successful_landing_metrics.png"
    labels = ['Final Horizontal Speed', 'Final Vertical Speed', 'Remaining Fuel', 'Path Length (steps)']
    values = [abs(metrics['vx']), abs(metrics['vy']), metrics['fuel'], path_len]
    plt.figure(figsize=(10, 7))
    bars = plt.bar(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.ylabel("Values"); plt.title("Metrics for Successful Landing")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.2f}', va='bottom', ha='center')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"📊 Clustered metrics chart saved to '{filename}'")

def draw_frame(path_coords, text_info, crashed=False):
    """Creates a single, visually appealing frame of the simulation."""
    W, H = 800, 600
    img = Image.new("RGB", (W, H), color="#00001a")
    draw = ImageDraw.Draw(img)
    for _ in range(150):
        sx, sy = random.randint(0, W - 1), random.randint(0, H - 1)
        draw.ellipse((sx, sy, sx + 1, sy + 1), fill="white")

    surface_y = H - 50
    draw.rectangle([0, surface_y, W, H], fill="#606060")
    draw.line([0, surface_y, W, surface_y], fill="#c0c0c0")

    # Landing pad
    pad_width_px = int((10.0 / 100) * W * 0.9)
    draw.rectangle([(W/2 - pad_width_px, surface_y-2), (W/2 + pad_width_px, surface_y)], fill="yellow")

    def world_to_img(x, y):
        ix = int(W / 2 + (x / 50.0) * W * 0.45)
        iy = int(surface_y - (y / 120.0) * (H - 100))
        return ix, iy

    if len(path_coords) > 1:
        draw.line([world_to_img(x, y) for x, y in path_coords], fill="#00ffff", width=2)

    lander_x, lander_y = path_coords[-1]
    ix, iy = world_to_img(lander_x, lander_y)

    if crashed:
        draw.ellipse((ix - 20, iy - 20, ix + 20, iy + 20), fill="#ff8c00")
        draw.ellipse((ix - 15, iy - 15, ix + 15, iy + 15), fill="#ffff00")
    else:
        lander_shape = [(ix, iy - 10), (ix - 6, iy + 7), (ix + 6, iy + 7)]
        draw.polygon(lander_shape, fill="#c0c0c0", outline="white")

    try: font = ImageFont.truetype("arial.ttf", 16)
    except IOError: font = ImageFont.load_default()
    draw.text((10, 10), text_info, font=font, fill="white")
    return np.array(img)

==============================================================================

============================= MAIN PIPELINE ==================================

==============================================================================

def main():
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = LunarEnv()
    state_dim = 5
    action_map = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
    action_dim = len(action_map)

    policy_net = DQN(state_dim, action_dim).to(device)
    # --- NEW: Create the separate, stable target network ---
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval() # Target network is only for inference

    if not SKIP_TRAINING:
        print("\n🚀 Starting training (stabilized version)...")
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
        memory = deque(maxlen=MEMORY_SIZE)
        reward_history = []
        epsilon = EPSILON_START
        
        with open('progression_log.csv', 'w', newline='') as f:
            progression_writer = csv.writer(f)
            progression_writer.writerow(['reward', 'length', 'time_s'])
            start_time = time.time()

            for episode in range(NUM_EPISODES):
                state = env.reset()
                total_reward = 0
                done = False

                while not done:
                    s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action_idx = random.randrange(action_dim) if random.random() < epsilon \
                        else torch.argmax(policy_net(s_t)).item()
                    
                    action = action_map[action_idx]
                    next_state, reward, done = env.step(action)
                    memory.append((state, action_idx, reward, next_state, done))
                    state, total_reward = next_state, total_reward + reward

                    if len(memory) > BATCH_SIZE:
                        batch = random.sample(memory, BATCH_SIZE)
                        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
                        states_b = torch.FloatTensor(states).to(device)
                        actions_b = torch.LongTensor(actions).unsqueeze(1).to(device)
                        rewards_b = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                        next_states_b = torch.FloatTensor(next_states).to(device)
                        dones_b = torch.FloatTensor(dones).unsqueeze(1).to(device)

                        q_values = policy_net(states_b).gather(1, actions_b)
                        with torch.no_grad():
                            # --- MODIFIED: Get next Q-values from the stable target_net ---
                            next_q = target_net(next_states_b).max(1)[0].unsqueeze(1)
                            target_q = rewards_b + GAMMA * next_q * (1 - dones_b)
                        
                        loss = nn.functional.smooth_l1_loss(q_values, target_q)
                        optimizer.zero_grad(); loss.backward(); optimizer.step()

                # --- NEW: Periodically update the target network ---
                if episode % TARGET_UPDATE_FREQ == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                reward_history.append(total_reward)
                epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
                progression_writer.writerow([f'{total_reward:.2f}', env.steps, f'{time.time() - start_time:.2f}'])

                if (episode + 1) % 100 == 0:
                    # --- MODIFIED: Log the average reward for a better trend view ---
                    avg_reward = np.mean(reward_history[-100:])
                    print(f"Episode {episode+1}/{NUM_EPISODES} | Avg Reward (last 100): {avg_reward:.2f} | Eps: {epsilon:.3f}")
                
                if GENERATE_SNAPSHOTS_DURING_TRAINING and episode % SNAPSHOT_INTERVAL == 0:
                    info = f"Training Episode: {episode}\nReward: {total_reward:.2f}"
                    frame_img = draw_frame(env.get_path(), info)
                    imageio.imwrite(f"snapshots/training_ep_{episode:05d}.png", frame_img)
        
        print("\n✅ Training complete.")
        torch.save(policy_net.state_dict(), MODEL_CHECKPOINT_PATH)
        print(f"💾 Model saved to '{MODEL_CHECKPOINT_PATH}'")
        print("1️⃣ Progression chart saved to 'progression_log.csv'")
        save_training_charts(reward_history) # Output 2
    else:
        # Load model logic remains the same
        print(f"⏩ Skipping training, loading model from '{MODEL_CHECKPOINT_PATH}'...")
        if os.path.exists(MODEL_CHECKPOINT_PATH):
            policy_net.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=device))
        else:
            print("❌ Error: Model checkpoint not found. Cannot proceed."); return

    # -------------------------- EVALUATION PHASE (No changes needed) ------------------------------
    print("\n🎬 Running final evaluation to generate animation...")
    policy_net.eval()
    state = env.reset()
    frames, done = [], False

    while not done:
        s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad(): action_idx = torch.argmax(policy_net(s_t)).item()
        state, _, done = env.step(action_map[action_idx])
        
        info = (f"Evaluation Run\nPos: ({env.x:.2f}, {env.y:.2f}) | Vel: ({env.vx:.2f}, {env.vy:.2f})\n"
                f"Fuel: {env.fuel:.2f} | Steps: {env.steps}")
        frames.append(draw_frame(env.get_path(), info))

    final_speed = np.sqrt(env.vx**2 + env.vy**2)
    success = final_speed < 2.0 and abs(env.x) < env.landing_zone_width / 2 and env.y == 0

    if success:
        print("🏆 LANDING SUCCESSFUL!")
        anim_path = "animation/perfect_landing.gif"
        imageio.mimsave(anim_path, frames, duration=50, loop=0)
        print(f"5️⃣ Perfect landing animation saved to '{anim_path}'")
        metrics = {'vx': env.vx, 'vy': env.vy, 'fuel': env.fuel}
        save_clustered_chart(metrics, path_len=len(env.get_path())) # Output 3
    else:
        print("💥 LANDING FAILED.")
        anim_path = "animation/failed_landing.gif"
        imageio.mimsave(anim_path, frames, duration=50, loop=0)
        print(f"⚠️ Failed landing animation saved to '{anim_path}'")

    print("\n4️⃣ Training snapshots (if enabled) are in the 'snapshots/' folder.")
    print("✅ Pipeline finished.")

if __name__ == "__main__":
    main()
