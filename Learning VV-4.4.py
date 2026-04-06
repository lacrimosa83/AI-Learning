"""
============================================
Curiosity-Driven Exploration Framework
VERSION 4.4 - 高难度验证（从V4.3冠军模型迁移）
============================================
核心策略：
1. 从 V4.3 冠军模型开始（165.4 分）
2. 切换到 V4.2 高难度环境（12静态+4动态障碍物）
3. 继续训练 2000 episodes
4. 验证信心是否可迁移到高难度

预期目标：140-160 分（原始 V4.2 最佳 121 分）
============================================
"""

import os
import sys
import math
import time
import warnings

warnings.filterwarnings('ignore')

import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime


# ========== GPU 配置 ==========
def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("=" * 60)
        print("GPU 配置信息")
        print("=" * 60)
        print(f"✅ 使用 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print("=" * 60)
    else:
        device = torch.device('cpu')
        print("⚠️ 使用 CPU 模式")
    return device


DEVICE = setup_device()

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True


# ============================================
# V4.2 高难度环境（复现）
# ============================================

class DynamicObstacle:
    def __init__(self, x, y, radius, speed=0.3):
        self.initial_x = x
        self.initial_y = y
        self.x = x
        self.y = y
        self.radius = radius
        self.vx = random.uniform(-speed, speed)
        self.vy = random.uniform(-speed, speed)
        self.speed = speed

    def update(self, world_size):
        self.x += self.vx * 0.1
        self.y += self.vy * 0.1

        half_size = world_size / 2
        if self.x > half_size - self.radius:
            self.x = half_size - self.radius - (self.x - (half_size - self.radius))
            self.vx = -self.vx
        elif self.x < -half_size + self.radius:
            self.x = -half_size + self.radius - (self.x + half_size - self.radius)
            self.vx = -self.vx

        if self.y > half_size - self.radius:
            self.y = half_size - self.radius - (self.y - (half_size - self.radius))
            self.vy = -self.vy
        elif self.y < -half_size + self.radius:
            self.y = -half_size + self.radius - (self.y + half_size - self.radius)
            self.vy = -self.vy

    def reset(self):
        self.x = self.initial_x
        self.y = self.initial_y
        self.vx = random.uniform(-self.speed, self.speed)
        self.vy = random.uniform(-self.speed, self.speed)


class World2DHighDifficulty:
    """
    V4.2 高难度环境（复现）
    12 个静态障碍物 + 4 个动态障碍物
    """

    def __init__(self, world_size=12.0, friction=0.98,
                 reward_zones=None, static_obstacles=None,
                 dynamic_obstacles=None,
                 noise_std=0.08, max_steps=200,
                 collision_penalty=1.0):
        self.world_size = world_size
        self.friction = friction
        self.max_steps = max_steps
        self.collision_penalty = collision_penalty

        # 奖励区域：4个，半径 0.4
        if reward_zones is None:
            self.reward_zones = [
                {'x': 4.0, 'y': 4.0, 'radius': 0.4, 'reward': 1.0},
                {'x': -3.0, 'y': 3.0, 'radius': 0.4, 'reward': 1.0},
                {'x': 2.0, 'y': -3.0, 'radius': 0.4, 'reward': 1.0},
                {'x': -4.0, 'y': -2.0, 'radius': 0.4, 'reward': 1.0},
            ]
        else:
            self.reward_zones = reward_zones

        # 静态障碍物：12个
        if static_obstacles is None:
            self.static_obstacles = [
                {'x': 0, 'y': 0, 'radius': 0.8},
                {'x': 2.5, 'y': 2.0, 'radius': 0.6},
                {'x': -2.0, 'y': 1.5, 'radius': 0.6},
                {'x': 1.0, 'y': -2.0, 'radius': 0.5},
                {'x': -1.5, 'y': -1.5, 'radius': 0.5},
                {'x': 3.5, 'y': -0.5, 'radius': 0.4},
                {'x': -3.0, 'y': -0.5, 'radius': 0.4},
                {'x': 0, 'y': 3.0, 'radius': 0.5},
                {'x': 0, 'y': -3.0, 'radius': 0.5},
                {'x': 3.0, 'y': -3.0, 'radius': 0.4},
                {'x': -3.5, 'y': 1.0, 'radius': 0.4},
                {'x': 1.5, 'y': 3.5, 'radius': 0.4},
            ]
        else:
            self.static_obstacles = static_obstacles

        # 动态障碍物：4个
        if dynamic_obstacles is None:
            self.dynamic_obstacles = [
                DynamicObstacle(2.0, 1.0, 0.4, 0.3),
                DynamicObstacle(-1.0, -2.0, 0.4, 0.25),
                DynamicObstacle(0, 2.5, 0.35, 0.35),
                DynamicObstacle(-2.5, -1.0, 0.35, 0.3),
            ]
        else:
            self.dynamic_obstacles = dynamic_obstacles

        self.noise_std = noise_std
        self.reset()

    def reset(self, extreme=True):
        corners = [
            (-self.world_size / 2 + 0.5, -self.world_size / 2 + 0.5),
            (-self.world_size / 2 + 0.5, self.world_size / 2 - 0.5),
            (self.world_size / 2 - 0.5, -self.world_size / 2 + 0.5),
            (self.world_size / 2 - 0.5, self.world_size / 2 - 0.5),
        ]
        self.x, self.y = random.choice(corners)
        self.vx = random.uniform(-2.0, 2.0)
        self.vy = random.uniform(-2.0, 2.0)

        while self._check_collision(self.x, self.y, check_dynamic=False):
            self.x = np.random.uniform(-self.world_size / 2, self.world_size / 2)
            self.y = np.random.uniform(-self.world_size / 2, self.world_size / 2)

        for obs in self.dynamic_obstacles:
            obs.reset()

        self.step_count = 0
        self.collision_count = 0
        return self._get_state()

    def _get_state(self):
        return np.array([self.x, self.y, self.vx, self.vy])

    def _check_collision(self, x, y, check_dynamic=True):
        for obs in self.static_obstacles:
            dx = x - obs['x']
            dy = y - obs['y']
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < obs['radius']:
                return True

        if check_dynamic:
            for obs in self.dynamic_obstacles:
                dx = x - obs.x
                dy = y - obs.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < obs.radius:
                    return True

        return False

    def _apply_collision_response(self):
        self.vx = -self.vx * 0.4
        self.vy = -self.vy * 0.4

        for obs in self.static_obstacles:
            dx = self.x - obs['x']
            dy = self.y - obs['y']
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < obs['radius']:
                if dist > 0:
                    push_x = dx / dist
                    push_y = dy / dist
                else:
                    push_x, push_y = 1.0, 0.0
                self.x = obs['x'] + push_x * (obs['radius'] + 0.1)
                self.y = obs['y'] + push_y * (obs['radius'] + 0.1)

        for obs in self.dynamic_obstacles:
            dx = self.x - obs.x
            dy = self.y - obs.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < obs.radius:
                if dist > 0:
                    push_x = dx / dist
                    push_y = dy / dist
                else:
                    push_x, push_y = 1.0, 0.0
                self.x = obs.x + push_x * (obs.radius + 0.1)
                self.y = obs.y + push_y * (obs.radius + 0.1)

        self.collision_count += 1

    def step(self, action):
        force_x = np.clip(action[0], -1.0, 1.0)
        force_y = np.clip(action[1], -1.0, 1.0)

        self.vx += force_x * 0.12
        self.vy += force_y * 0.12

        self.x += self.vx * 0.1
        self.y += self.vy * 0.1
        self.vx *= self.friction
        self.vy *= self.friction

        half_size = self.world_size / 2
        if self.x > half_size:
            self.x = half_size - (self.x - half_size)
            self.vx = -self.vx * 0.6
        elif self.x < -half_size:
            self.x = -half_size - (self.x + half_size)
            self.vx = -self.vx * 0.6

        if self.y > half_size:
            self.y = half_size - (self.y - half_size)
            self.vy = -self.vy * 0.6
        elif self.y < -half_size:
            self.y = -half_size - (self.y + half_size)
            self.vy = -self.vy * 0.6

        for obs in self.dynamic_obstacles:
            obs.update(self.world_size)

        collision = False
        if self._check_collision(self.x, self.y):
            self._apply_collision_response()
            collision = True

        self.x += np.random.randn() * self.noise_std
        self.y += np.random.randn() * self.noise_std
        self.vx += np.random.randn() * self.noise_std * 0.1
        self.vy += np.random.randn() * self.noise_std * 0.1

        reward = 0.0
        for zone in self.reward_zones:
            dx = self.x - zone['x']
            dy = self.y - zone['y']
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < zone['radius']:
                reward = zone['reward']
                break

        if collision:
            reward -= self.collision_penalty

        self.step_count += 1
        done = self.step_count >= self.max_steps

        return self._get_state(), reward, done

    def get_collision_rate(self):
        if self.step_count > 0:
            return self.collision_count / self.step_count
        return 0.0


# ============================================
# 好奇心机制模块（与 V4.2/V4.3 相同）
# ============================================

class PredictionErrorCuriosity(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=256, lr=0.0005):
        super().__init__()
        self.dynamics_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        ).to(DEVICE)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999)
        self.loss_history = deque(maxlen=100)

    def forward(self, state, action, next_state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        input_tensor = torch.cat([state, action], dim=-1)
        pred_next = self.dynamics_model(input_tensor)
        error = F.mse_loss(pred_next, next_state, reduction='mean')
        return torch.tanh(error), error.item()

    def update(self, state, action, next_state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        self.optimizer.zero_grad()
        pred = self.dynamics_model(torch.cat([state, action], -1))
        loss = F.mse_loss(pred, next_state)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        self.loss_history.append(loss.item())
        return loss.item()

    def step_scheduler(self):
        self.scheduler.step()


class NoveltyCuriosity2D:
    def __init__(self, state_dim=4, memory_size=15000, novelty_decay=0.998):
        self.memory = deque(maxlen=memory_size)
        self.novelty_decay = novelty_decay
        self.exploration_count = 0

    def compute_novelty(self, state):
        if len(self.memory) == 0:
            self.memory.append(state.copy())
            return 1.0

        min_dist = min(np.linalg.norm(state - mem) for mem in self.memory)
        novelty = min(1.0, min_dist / 5.0)
        novelty *= (self.novelty_decay ** (self.exploration_count / 100))

        if random.random() < 0.05:
            self.memory.append(state.copy())

        return novelty

    def record_exploration(self):
        self.exploration_count += 1

    def reset_memory(self, keep_ratio=0.7):
        if len(self.memory) > 100:
            old_len = len(self.memory)
            keep_count = int(old_len * keep_ratio)
            self.memory = deque(list(self.memory)[-keep_count:], maxlen=self.memory.maxlen)
            print(f"   [Novelty重置] 清除了 {old_len - keep_count} 条旧记忆，保留 {keep_count} 条")
            return old_len - keep_count
        return 0


class RNDCuriosity2D(nn.Module):
    def __init__(self, state_dim=4, embedding_dim=128, hidden_dim=256, lr=0.0005):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        ).to(DEVICE)
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        ).to(DEVICE)
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999)
        self.register_buffer('mean', torch.zeros(1, device=DEVICE))
        self.register_buffer('std', torch.ones(1, device=DEVICE))
        self.register_buffer('count', torch.zeros(1, device=DEVICE))
        self.loss_history = deque(maxlen=100)
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            target = self.target(state)
        pred = self.predictor(state)
        error = F.mse_loss(pred, target, reduction='none').mean(-1)
        if self.count > 0:
            error = (error - self.mean) / (self.std + 1e-8)
        return torch.clamp(error, 0, 1).squeeze(), error.mean().item()

    def update(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        self.optimizer.zero_grad()
        target = self.target(state).detach()
        pred = self.predictor(state)
        loss = F.mse_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
        self.optimizer.step()
        with torch.no_grad():
            error = F.mse_loss(pred, target, reduction='none').mean(-1)
            self.count += len(error)
            delta = error - self.mean
            self.mean += delta.sum() / self.count
            self.std += (delta * (error - self.mean)).sum() / self.count
        self.loss_history.append(loss.item())
        return loss.item()

    def step_scheduler(self):
        self.scheduler.step()


# ============================================
# 信用分配系统
# ============================================

class ContributionTracker:
    def __init__(self, num_mechanisms, gamma=0.95):
        self.num = num_mechanisms
        self.gamma = gamma
        self.reset()

    def reset(self):
        self.weights = []
        self.values = []
        self.rewards = []

    def record(self, weights, values, reward):
        self.weights.append(weights.copy())
        self.values.append(values.copy())
        self.rewards.append(reward)

    def compute_credit(self):
        if not self.weights:
            return np.zeros(self.num)
        T = len(self.weights)
        contrib = np.zeros((T, self.num))
        for t in range(T):
            total = np.sum(self.weights[t] * self.values[t])
            if total > 0:
                contrib[t] = self.weights[t] * self.values[t] / total
            else:
                contrib[t] = np.ones(self.num) / self.num
        discounts = np.array([self.gamma ** (T - 1 - t) for t in range(T)])
        if discounts.sum() > 0:
            discounts /= discounts.sum()
        rewards = np.array(self.rewards)
        credit = np.zeros(self.num)
        for t in range(T):
            credit += contrib[t] * rewards[t] * discounts[t]
        return credit


# ============================================
# 自适应好奇心系统（V4.4）
# ============================================

@dataclass
class CuriosityConfigV44:
    state_dim: int = 4
    action_dim: int = 2
    num_mechanisms: int = 3
    hidden_dim: int = 256
    rnd_embedding_dim: int = 128
    learning_rate: float = 0.002  # 降低学习率，微调
    weight_update_freq: int = 10
    credit_gamma: float = 0.95
    save_freq: int = 100
    checkpoint_dir: str = "checkpoints_v44"
    min_weight: float = 0.10
    novelty_reset_freq: int = 500
    exploration_bonus: float = 0.05  # 降低探索保底
    weight_reset_threshold: float = 0.95
    high_score_threshold: float = 130  # 高难度下降低阈值
    lock_duration: int = 80


class AdaptiveCuriositySystemV44:
    def __init__(self, config: CuriosityConfigV44):
        self.config = config
        self.num = config.num_mechanisms

        self.mechanisms = {
            'prediction_error': PredictionErrorCuriosity(
                config.state_dim, config.action_dim, config.hidden_dim, lr=0.0005),
            'novelty': NoveltyCuriosity2D(config.state_dim),
            'rnd': RNDCuriosity2D(config.state_dim, config.rnd_embedding_dim, lr=0.0005)
        }

        self.weights_logits = nn.Parameter(torch.zeros(config.num_mechanisms, device=DEVICE))
        self.weights_optimizer = optim.Adam([self.weights_logits], lr=config.learning_rate)

        self.tracker = ContributionTracker(config.num_mechanisms, config.credit_gamma)

        self.history = {
            'episode_returns': [], 'weights_history': [], 'curiosity_history': [],
            'pred_errors': [], 'rnd_errors': [], 'novelty_values': [],
            'timestamps': [], 'episode_times': [], 'mechanism_credits': [],
            'explore_rates': [], 'collision_rates': []
        }

        self.step_count = 0
        self.episode_count = 0
        self.start_time = time.time()
        self._current_curiosity = None
        self._current_weights = None

        self.best_return = 0
        self.best_episode = 0
        self.high_score_lock_counter = 0

        self.champion_path = os.path.join(config.checkpoint_dir, "champion_model_v44.pth")
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    @property
    def weights(self):
        return F.softmax(self.weights_logits, dim=0).detach().cpu().numpy()

    def compute_curiosity(self, state, action, next_state, episode_progress=0):
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        next_t = torch.tensor(next_state, dtype=torch.float32, device=DEVICE)
        action_t = torch.tensor(action, dtype=torch.float32, device=DEVICE)

        pred_c, pred_e = self.mechanisms['prediction_error'](state_t, action_t, next_t)
        novelty = self.mechanisms['novelty'].compute_novelty(state)
        rnd_c, rnd_e = self.mechanisms['rnd'](state_t)

        values = np.array([pred_c.item(), novelty, rnd_c.item()])
        weights = self.weights
        weighted_sum = np.sum(weights * values)

        exploration_bonus = self.config.exploration_bonus * (1 - episode_progress) * 0.5
        total_curiosity = weighted_sum + exploration_bonus

        self._current_curiosity = values
        self._current_weights = weights

        return total_curiosity, {
            'prediction_error': pred_c.item(), 'novelty': novelty,
            'rnd': rnd_c.item(), 'pred_error': pred_e, 'rnd_error': rnd_e
        }

    def record_step(self, state, action, reward, next_state):
        if self._current_curiosity is not None:
            self.tracker.record(self._current_weights, self._current_curiosity, reward)

        self.step_count += 1
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        next_t = torch.tensor(next_state, dtype=torch.float32, device=DEVICE)
        action_t = torch.tensor(action, dtype=torch.float32, device=DEVICE)

        self.mechanisms['prediction_error'].update(state_t, action_t, next_t)
        self.mechanisms['rnd'].update(state_t)
        self.mechanisms['novelty'].record_exploration()

        if self.step_count % self.config.weight_update_freq == 0:
            self._update_weights()

    def end_episode(self, episode_return, episode_time, details=None, collision_rate=0):
        credit = self.tracker.compute_credit()
        self._update_weights_from_credit(credit)

        self.history['episode_returns'].append(episode_return)
        self.history['weights_history'].append(self.weights.copy())
        self.history['episode_times'].append(episode_time)
        self.history['timestamps'].append(time.time() - self.start_time)
        self.history['mechanism_credits'].append(credit)
        self.history['collision_rates'].append(collision_rate)

        if details:
            self.history['pred_errors'].append(details.get('pred_error', 0))
            self.history['rnd_errors'].append(details.get('rnd_error', 0))
            self.history['novelty_values'].append(details.get('novelty', 0))

        self.tracker.reset()
        self.episode_count += 1

        if episode_return >= self.config.high_score_threshold:
            self.high_score_lock_counter = self.config.lock_duration
            print(f"   🔒 高分锁定激活！({episode_return:.1f} >= {self.config.high_score_threshold})")
        elif self.high_score_lock_counter > 0:
            self.high_score_lock_counter -= 1

        if self.episode_count % self.config.novelty_reset_freq == 0:
            self.mechanisms['novelty'].reset_memory(keep_ratio=0.7)

        if episode_return > self.best_return:
            self.best_return = episode_return
            self.best_episode = self.episode_count
            self._save_champion()
            print(f"   🏆 新纪录！Episode {self.episode_count}: {episode_return:.2f} 分")

        if self.episode_count % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.episode_count % self.config.save_freq == 0:
            self.save_checkpoint()

        return episode_return

    def is_high_score_lock_active(self):
        return self.high_score_lock_counter > 0

    def _update_weights(self):
        """更新权重 - 基于近期信用"""
        if len(self.history['episode_returns']) < 5:
            return
        recent_credits = np.mean(self.history['mechanism_credits'][-5:], axis=0)
        self._update_weights_from_credit(recent_credits)

    def _update_weights_from_credit(self, credit):
        """根据信用更新权重"""
        credit_sum = credit.sum()
        if credit_sum > 0:
            target = credit / credit_sum
        else:
            target = np.ones(self.num) / self.num

        target = np.maximum(target, self.config.min_weight)
        target = target / target.sum()

        target_logits = np.log(target + 1e-8)
        target_t = torch.tensor(target_logits, dtype=torch.float32, device=DEVICE)

        self.weights_optimizer.zero_grad()
        loss = F.mse_loss(self.weights_logits, target_t)
        loss.backward()
        self.weights_optimizer.step()

    def _save_champion(self):
        champion_checkpoint = {
            'episode': self.episode_count,
            'return': self.best_return,
            'weights_logits': self.weights_logits.detach().cpu(),
            'pred_error_state': self.mechanisms['prediction_error'].state_dict(),
            'novelty_memory': list(self.mechanisms['novelty'].memory),
            'novelty_count': self.mechanisms['novelty'].exploration_count,
            'rnd_predictor_state': self.mechanisms['rnd'].predictor.state_dict(),
            'rnd_mean': self.mechanisms['rnd'].mean.cpu(),
            'rnd_std': self.mechanisms['rnd'].std.cpu(),
            'rnd_count': self.mechanisms['rnd'].count.cpu(),
            'timestamp': datetime.now().isoformat()
        }
        torch.save(champion_checkpoint, self.champion_path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.episode_count = checkpoint['episode']
        self.step_count = checkpoint['step']
        self.weights_logits.data = checkpoint['weights_logits'].to(DEVICE)
        self.weights_optimizer.load_state_dict(checkpoint['weights_optimizer'])
        self.history = checkpoint['history']

        if 'explore_rates' not in self.history:
            self.history['explore_rates'] = []
        if 'collision_rates' not in self.history:
            self.history['collision_rates'] = []

        self.best_return = checkpoint.get('best_return', 0)
        self.best_episode = checkpoint.get('best_episode', 0)
        self.high_score_lock_counter = checkpoint.get('high_score_lock_counter', 0)

        self.mechanisms['novelty'].memory = deque(checkpoint['novelty_memory'], maxlen=15000)
        self.mechanisms['novelty'].exploration_count = checkpoint['novelty_count']
        self.mechanisms['prediction_error'].load_state_dict(checkpoint['pred_error_state'])
        self.mechanisms['prediction_error'].optimizer.load_state_dict(checkpoint['pred_error_optimizer'])
        self.mechanisms['rnd'].predictor.load_state_dict(checkpoint['rnd_predictor_state'])
        self.mechanisms['rnd'].optimizer.load_state_dict(checkpoint['rnd_optimizer'])
        self.mechanisms['rnd'].mean = checkpoint['rnd_mean'].to(DEVICE)
        self.mechanisms['rnd'].std = checkpoint['rnd_std'].to(DEVICE)
        self.mechanisms['rnd'].count = checkpoint['rnd_count'].to(DEVICE)

        print(f"✅ 从检查点恢复: Episode {self.episode_count}")
        print(f"🏆 历史最佳: Episode {self.best_episode}, Return {self.best_return:.2f}")
        return self.episode_count

    def load_v43_champion(self):
        """专门加载 V4.3 冠军模型"""
        v43_champion_path = "checkpoints_v43/champion_model_v43.pth"
        if os.path.exists(v43_champion_path):
            checkpoint = torch.load(v43_champion_path, map_location='cpu', weights_only=False)
            self.weights_logits.data = checkpoint['weights_logits'].to(DEVICE)
            self.mechanisms['prediction_error'].load_state_dict(checkpoint['pred_error_state'])
            self.mechanisms['novelty'].memory = deque(checkpoint['novelty_memory'], maxlen=15000)
            self.mechanisms['novelty'].exploration_count = checkpoint['novelty_count']
            self.mechanisms['rnd'].predictor.load_state_dict(checkpoint['rnd_predictor_state'])
            self.mechanisms['rnd'].mean = checkpoint['rnd_mean'].to(DEVICE)
            self.mechanisms['rnd'].std = checkpoint['rnd_std'].to(DEVICE)
            self.mechanisms['rnd'].count = checkpoint['rnd_count'].to(DEVICE)
            self.best_return = checkpoint.get('return', 0)
            self.best_episode = checkpoint.get('episode', 0)
            print(f"🏆 V4.3 冠军模型已加载 (Episode {self.best_episode}, Return {self.best_return:.2f})")
            return True
        else:
            print("⚠️ 未找到 V4.3 冠军模型")
            return False

    def save_checkpoint(self):
        checkpoint = {
            'episode': self.episode_count,
            'step': self.step_count,
            'weights_logits': self.weights_logits.detach().cpu(),
            'weights_optimizer': self.weights_optimizer.state_dict(),
            'history': self.history,
            'best_return': self.best_return,
            'best_episode': self.best_episode,
            'high_score_lock_counter': self.high_score_lock_counter,
            'novelty_memory': list(self.mechanisms['novelty'].memory),
            'novelty_count': self.mechanisms['novelty'].exploration_count,
            'pred_error_state': self.mechanisms['prediction_error'].state_dict(),
            'pred_error_optimizer': self.mechanisms['prediction_error'].optimizer.state_dict(),
            'rnd_predictor_state': self.mechanisms['rnd'].predictor.state_dict(),
            'rnd_optimizer': self.mechanisms['rnd'].optimizer.state_dict(),
            'rnd_mean': self.mechanisms['rnd'].mean.cpu(),
            'rnd_std': self.mechanisms['rnd'].std.cpu(),
            'rnd_count': self.mechanisms['rnd'].count.cpu(),
        }
        path = f"{self.config.checkpoint_dir}/checkpoint_ep{self.episode_count}.pth"
        torch.save(checkpoint, path)
        print(f"   💾 检查点已保存: {path}")

    def get_stats(self):
        recent_returns = self.history['episode_returns'][-20:] if self.history['episode_returns'] else []
        return {
            'weights': self.weights.tolist(),
            'episode': self.episode_count,
            'step': self.step_count,
            'avg_return': np.mean(recent_returns) if recent_returns else 0,
            'total_time': time.time() - self.start_time,
            'best_return': self.best_return,
            'best_episode': self.best_episode
        }


# ============================================
# 可视化工具
# ============================================

class Visualizer2D:
    def __init__(self):
        self.fig = None
        self.axes = None
        self.initialized = False

    def setup(self):
        plt.close('all')
        self.fig = plt.figure(figsize=(16, 10))
        gs = self.fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        self.axes = {
            'trajectory': self.fig.add_subplot(gs[0, :2]),
            'returns': self.fig.add_subplot(gs[0, 2]),
            'weights': self.fig.add_subplot(gs[1, 0]),
            'curiosity': self.fig.add_subplot(gs[1, 1]),
            'credit': self.fig.add_subplot(gs[1, 2])
        }

        self.fig.suptitle('2D Exploration - V4.4 (高难度验证，从V4.3迁移)', fontsize=14, fontweight='bold')
        self.initialized = True
        plt.ion()
        plt.show()

    def update_trajectory(self, ax, positions, reward_zones, static_obstacles, dynamic_obstacles, current_pos=None):
        ax.clear()

        for obs in static_obstacles:
            circle = Circle((obs['x'], obs['y']), obs['radius'], color='gray', alpha=0.5)
            ax.add_patch(circle)

        for obs in dynamic_obstacles:
            circle = Circle((obs.x, obs.y), obs.radius, color='orange', alpha=0.5)
            ax.add_patch(circle)

        for zone in reward_zones:
            circle = Circle((zone['x'], zone['y']), zone['radius'], color='green', alpha=0.3)
            ax.add_patch(circle)
            ax.text(zone['x'], zone['y'], f'{zone["reward"]}', ha='center', va='center', fontsize=8)

        if len(positions) > 0:
            positions = np.array(positions)
            ax.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.5, linewidth=1)
            ax.scatter(positions[0, 0], positions[0, 1], c='green', s=50, marker='o', label='Start')
            ax.scatter(positions[-1, 0], positions[-1, 1], c='red', s=80, marker='o', label='Current')

        ax.set_xlim(-7, 7)
        ax.set_ylim(-7, 7)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('2D Trajectory with High Difficulty Obstacles')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_aspect('equal')

    def update(self, system, env, positions, current_state=None):
        if not self.initialized:
            self.setup()

        stats = system.get_stats()
        history = system.history
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        labels = ['PredErr', 'Novelty', 'RND']

        self.update_trajectory(self.axes['trajectory'], positions,
                               env.reward_zones, env.static_obstacles,
                               env.dynamic_obstacles, current_state)

        ax = self.axes['returns']
        ax.clear()
        if history['episode_returns']:
            returns = history['episode_returns']
            ax.plot(returns, alpha=0.5)
            if len(returns) >= 20:
                smooth = np.convolve(returns, np.ones(20) / 20, mode='valid')
                ax.plot(range(19, len(returns)), smooth, 'r-', linewidth=2)
            if stats['best_episode'] > 0:
                ax.scatter(stats['best_episode'] - 1, stats['best_return'],
                           c='gold', s=100, marker='*', zorder=5)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Return')
            ax.set_title(f'Returns (Avg: {stats["avg_return"]:.1f}, Best: {stats["best_return"]:.1f})')
            ax.grid(True, alpha=0.3)

        ax = self.axes['weights']
        ax.clear()
        if history['weights_history']:
            wh = np.array(history['weights_history'])
            for i, label in enumerate(labels):
                ax.plot(wh[:, i], label=label, color=colors[i])
            ax.set_xlabel('Episode')
            ax.set_ylabel('Weight')
            ax.set_title('Weights Evolution')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)

        ax = self.axes['curiosity']
        ax.clear()
        if history.get('novelty_values'):
            ax.plot(history['novelty_values'], label='Novelty', color=colors[1], alpha=0.7)
        if history.get('pred_errors'):
            ax.plot(history['pred_errors'], label='Pred Error', color=colors[0], alpha=0.7)
        if history.get('rnd_errors'):
            ax.plot(history['rnd_errors'], label='RND Error', color=colors[2], alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Value')
        ax.set_title('Curiosity Signals')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = self.axes['credit']
        ax.clear()
        if history.get('mechanism_credits') and len(history['mechanism_credits']) > 0:
            recent = np.mean(history['mechanism_credits'][-10:], axis=0)
            ax.bar(labels, recent, color=colors)
            ax.set_ylabel('Credit')
            ax.set_title('Recent Credit Distribution')

        plt.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.initialized = False


# ============================================
# 训练函数（V4.4 - 从V4.3冠军模型迁移）
# ============================================

def train_v44(num_episodes: int = 2000,
              render: bool = True,
              save_freq: int = 100,
              verbose_freq: int = 20):
    """V4.4 训练 - 从 V4.3 冠军模型迁移到高难度"""
    print("=" * 60)
    print("V4.4 HIGH DIFFICULTY VALIDATION")
    print("=" * 60)
    print("策略:")
    print("  ✅ 从 V4.3 冠军模型开始 (165.4 分)")
    print("  ✅ 切换到 V4.2 高难度环境")
    print("  ✅ 验证信心是否可迁移")
    print("=" * 60)

    config = CuriosityConfigV44()
    env = World2DHighDifficulty()
    system = AdaptiveCuriositySystemV44(config)
    visualizer = Visualizer2D() if render else None

    # 关键：加载 V4.3 冠军模型
    print("\n📥 加载 V4.3 冠军模型...")
    if not system.load_v43_champion():
        print("❌ 无法加载 V4.3 冠军模型，训练终止")
        return None

    print(f"\n目标 Episodes: {num_episodes}")
    print(f"设备: {DEVICE}")
    print(f"高分锁定阈值: {config.high_score_threshold}")
    print(f"V4.2 高难度环境: 12静态 + 4动态障碍物")
    print("-" * 60)

    total_start = time.time()

    # 探索率参数（从低探索率开始）
    explore_rate = 0.15
    explore_rate_min = 0.08
    explore_rate_max = 0.25
    explore_smoothing = 0.98

    for episode in range(num_episodes):
        state = env.reset(extreme=True)
        episode_return = 0
        episode_steps = 0
        episode_start = time.time()
        episode_details = {'pred_error': 0, 'rnd_error': 0, 'novelty': 0}

        positions = [(state[0], state[1])] if episode % 10 == 0 else None

        episode_progress = episode / num_episodes

        # 动态调整探索率
        if episode >= 50 and not system.is_high_score_lock_active():
            recent_returns = system.history['episode_returns'][-50:] if system.history['episode_returns'] else [0]
            recent_avg = np.mean(recent_returns)

            if recent_avg > 120:
                target_rate = max(explore_rate_min, explore_rate * 0.97)
            elif recent_avg > 100:
                target_rate = max(explore_rate_min, explore_rate * 0.98)
            elif recent_avg > 80:
                target_rate = max(explore_rate_min, explore_rate * 0.99)
            elif recent_avg < 40:
                target_rate = min(explore_rate_max, explore_rate * 1.03)
            else:
                target_rate = explore_rate
        else:
            target_rate = explore_rate

        if system.is_high_score_lock_active():
            target_rate = explore_rate_min

        explore_rate = explore_smoothing * explore_rate + (1 - explore_smoothing) * target_rate
        explore_rate = np.clip(explore_rate, explore_rate_min, explore_rate_max)

        system.history['explore_rates'].append(explore_rate)

        for step in range(env.max_steps):
            if random.random() < explore_rate:
                action = np.random.uniform(-1.0, 1.0, size=2)
            else:
                nearest_zone = min(env.reward_zones,
                                   key=lambda z: ((state[0] - z['x']) ** 2 + (state[1] - z['y']) ** 2) ** 0.5)
                dx = nearest_zone['x'] - state[0]
                dy = nearest_zone['y'] - state[1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 0.1:
                    action = np.array([np.clip(dx / dist, -1.0, 1.0), np.clip(dy / dist, -1.0, 1.0)])
                else:
                    action = np.zeros(2)

            next_state, reward, done = env.step(action)
            curiosity, details = system.compute_curiosity(state, action, next_state, episode_progress)
            system.record_step(state, action, reward, next_state)

            episode_return += reward
            episode_steps += 1
            episode_details['pred_error'] += details.get('pred_error', 0)
            episode_details['rnd_error'] += details.get('rnd_error', 0)
            episode_details['novelty'] += details.get('novelty', 0)

            state = next_state

            if positions is not None:
                positions.append((state[0], state[1]))

            if done:
                break

        for k in episode_details:
            episode_details[k] /= max(episode_steps, 1)

        collision_rate = env.get_collision_rate()
        episode_time = time.time() - episode_start
        system.end_episode(episode_return, episode_time, episode_details, collision_rate)

        if visualizer and episode % 5 == 0 and positions:
            visualizer.update(system, env, positions, state)
            plt.pause(0.01)

        if (episode + 1) % verbose_freq == 0:
            stats = system.get_stats()
            elapsed = time.time() - total_start
            weights = system.weights
            lock_status = "🔒" if system.is_high_score_lock_active() else "  "
            recent_avg = np.mean(system.history['episode_returns'][-50:]) if system.history['episode_returns'] else 0
            print(f"Ep {episode + 1:4d}/{num_episodes} | "
                  f"Return: {episode_return:7.2f} | "
                  f"Recent Avg: {recent_avg:7.2f} | "
                  f"Best: {stats['best_return']:.1f} {lock_status} | "
                  f"Weights: [{weights[0]:.2f},{weights[1]:.2f},{weights[2]:.2f}] | "
                  f"Explore: {explore_rate:.3f} | "
                  f"Coll: {collision_rate:.2f} | "
                  f"Time: {elapsed / 60:.1f}min")

        if (episode + 1) % 100 == 0 and torch.cuda.is_available():
            print(f"   📊 GPU: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    total_time = time.time() - total_start
    print("-" * 60)
    print(f"训练完成！总时间: {total_time / 60:.1f} 分钟")
    print(f"总 Episodes: {system.episode_count}")

    if system.history['episode_returns']:
        recent = system.history['episode_returns'][-100:] if len(system.history['episode_returns']) > 100 else \
        system.history['episode_returns']
        print(f"平均 Episode 奖励 (最后100): {np.mean(recent):.2f}")
        print(f"最佳 Episode 奖励: {system.best_return:.2f} (Episode {system.best_episode})")

    system.save_checkpoint()
    print(f"🏆 冠军模型已保存至: {system.champion_path}")

    if visualizer:
        visualizer.close()

    return system


# ============================================
# 结果分析（保存到本地）
# ============================================

def analyze_and_save_results(system: AdaptiveCuriositySystemV44, save_path: str = "v44_analysis.png"):
    """分析训练结果并保存图片到本地"""
    history = system.history

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. 奖励曲线
    returns = history['episode_returns']
    axes[0, 0].plot(returns, alpha=0.5, linewidth=0.8)
    if len(returns) >= 50:
        smooth = np.convolve(returns, np.ones(50) / 50, mode='valid')
        axes[0, 0].plot(range(49, len(returns)), smooth, 'r-', linewidth=2, label='Moving Avg (50)')
    if system.best_episode > 0:
        axes[0, 0].scatter(system.best_episode - 1, system.best_return,
                           c='gold', s=150, marker='*', zorder=5, label=f'Best: {system.best_return}')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].set_title(f'V4.4 Episode Returns (Best: {system.best_return:.1f})')
    axes[0, 0].legend(loc='lower right', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 权重演化
    weights = np.array(history['weights_history'])
    for i, label in enumerate(['Prediction Error', 'Novelty', 'RND']):
        axes[0, 1].plot(weights[:, i], label=label, linewidth=1.5)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Weight')
    axes[0, 1].set_title('Weights Evolution')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)

    # 3. 探索率演化
    explore_rates = history.get('explore_rates', [])
    if explore_rates:
        axes[0, 2].plot(explore_rates, 'b-', alpha=0.7, linewidth=1.5)
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Explore Rate')
        axes[0, 2].set_title('Explore Rate Evolution')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim(0, 0.6)

    # 4. 好奇心信号
    if history.get('novelty_values'):
        axes[1, 0].plot(history['novelty_values'], label='Novelty', alpha=0.7, linewidth=1.5)
    if history.get('pred_errors'):
        axes[1, 0].plot(history['pred_errors'], label='Pred Error', alpha=0.7, linewidth=1.5)
    if history.get('rnd_errors'):
        axes[1, 0].plot(history['rnd_errors'], label='RND Error', alpha=0.7, linewidth=1.5)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('Curiosity Signals')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # 5. 奖励分布直方图
    axes[1, 1].hist(returns, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1, 1].axvline(np.mean(returns), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(returns):.1f}')
    axes[1, 1].axvline(system.best_return, color='gold', linestyle='--',
                       linewidth=2, label=f'Best: {system.best_return:.1f}')
    axes[1, 1].set_xlabel('Return')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].legend(fontsize=8)

    # 6. 最终权重饼图
    final_weights = system.weights
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    labels = ['Prediction Error', 'Novelty', 'RND']
    axes[1, 2].pie(final_weights, labels=labels, colors=colors,
                   autopct='%1.1f%%', startangle=90)
    axes[1, 2].set_title(f'Final Weights (min={system.config.min_weight:.0%})')

    plt.suptitle(f'V4.4 High Difficulty Validation - Best: {system.best_return:.1f} points', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 分析图片已保存: {save_path}")
    plt.show()

    # 打印统计报告
    print("\n" + "=" * 60)
    print("训练统计报告 (V4.4)")
    print("=" * 60)
    print(f"总 Episodes: {len(returns)}")
    print(f"平均奖励: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"最佳奖励: {system.best_return:.2f} (Episode {system.best_episode})")
    print(f"最终权重: [{final_weights[0]:.3f}, {final_weights[1]:.3f}, {final_weights[2]:.3f}]")
    if explore_rates:
        print(f"平均探索率: {np.mean(explore_rates):.3f}")
    print("=" * 60)

    # 保存统计报告
    report_path = save_path.replace('.png', '_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("V4.4 训练统计报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"总 Episodes: {len(returns)}\n")
        f.write(f"平均奖励: {np.mean(returns):.2f} ± {np.std(returns):.2f}\n")
        f.write(f"最佳奖励: {system.best_return:.2f} (Episode {system.best_episode})\n")
        f.write(f"最终权重: [{final_weights[0]:.3f}, {final_weights[1]:.3f}, {final_weights[2]:.3f}]\n")
        if explore_rates:
            f.write(f"平均探索率: {np.mean(explore_rates):.3f}\n")
        f.write("=" * 60 + "\n")
    print(f"📄 统计报告已保存: {report_path}")


# ============================================
# 主程序
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🌙 V4.4 高难度验证训练")
    print("=" * 60)
    print("策略:")
    print("  - 从 V4.3 冠军模型开始 (165.4 分)")
    print("  - 切换到 V4.2 高难度环境")
    print("  - 验证信心是否可迁移")
    print("  - 训练量: 2000 episodes")
    print("=" * 60)

    # 运行训练
    system = train_v44(
        num_episodes=2000,
        render=True,
        save_freq=100,
        verbose_freq=20
    )

    if system:
        # 分析并保存结果
        print("\n" + "=" * 60)
        print("生成分析报告...")
        print("=" * 60)
        analyze_and_save_results(system, save_path="v44_analysis.png")

        print("\n✨ V4.4 训练完成！信心迁移验证完毕！✨")
    else:
        print("\n❌ V4.4 训练失败，请检查 V4.3 冠军模型是否存在")