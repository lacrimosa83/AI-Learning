"""
============================================
Curiosity-Driven Exploration Framework
VERSION 4.0 - 2D环境 + 连续动作空间
============================================
核心改进：
1. 2D 空间 (x, y 坐标)
2. 连续动作空间 (力的大小和方向)
3. 圆形奖励区域
4. 更复杂的物理引擎
5. 可视化升级 (2D轨迹图)
============================================
"""

import os
import sys
import math
import time
import warnings

warnings.filterwarnings('ignore')

# ========== 设置 matplotlib 后端 ==========
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

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

# 设置随机种子
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================
# 2D 环境模拟器
# ============================================

class World2D:
    """
    2D 连续空间环境
    状态: [x, y, vx, vy] (4维)
    动作: [force_x, force_y] (连续，范围 -1 到 1)
    """

    def __init__(self, world_size=10.0, friction=0.98,
                 reward_zones=None, noise_std=0.05,
                 max_steps=200):
        self.world_size = world_size
        self.friction = friction
        self.max_steps = max_steps

        # 奖励区域: 每个区域是 (x, y, radius, reward)
        if reward_zones is None:
            self.reward_zones = [
                {'x': 3.0, 'y': 3.0, 'radius': 0.8, 'reward': 1.0},
                {'x': -2.0, 'y': 4.0, 'radius': 0.8, 'reward': 1.0},
                {'x': 5.0, 'y': -1.0, 'radius': 0.8, 'reward': 1.0},
                {'x': -4.0, 'y': -3.0, 'radius': 0.8, 'reward': 1.0},
                {'x': 0.0, 'y': 0.0, 'radius': 0.6, 'reward': 0.5},  # 中心小奖励
            ]
        else:
            self.reward_zones = reward_zones

        self.noise_std = noise_std
        self.reset()

    def reset(self, extreme=False):
        """重置环境"""
        if extreme:
            # 极端起始位置（角落）
            self.x = random.choice([-self.world_size / 2 + 0.5, self.world_size / 2 - 0.5])
            self.y = random.choice([-self.world_size / 2 + 0.5, self.world_size / 2 - 0.5])
            self.vx = random.uniform(-3.0, 3.0)
            self.vy = random.uniform(-3.0, 3.0)
        else:
            self.x = np.random.uniform(-self.world_size / 2, self.world_size / 2)
            self.y = np.random.uniform(-self.world_size / 2, self.world_size / 2)
            self.vx = np.random.uniform(-2.0, 2.0)
            self.vy = np.random.uniform(-2.0, 2.0)

        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        """返回状态 [x, y, vx, vy]"""
        return np.array([self.x, self.y, self.vx, self.vy])

    def set_difficulty(self, noise_std=None, reward_radius=None):
        """动态调整难度"""
        if noise_std is not None:
            self.noise_std = noise_std
        if reward_radius is not None:
            for zone in self.reward_zones:
                zone['radius'] = reward_radius

    def step(self, action):
        """
        执行动作
        action: [force_x, force_y] 范围 -1 到 1
        """
        # 应用力
        force_x = np.clip(action[0], -1.0, 1.0)
        force_y = np.clip(action[1], -1.0, 1.0)

        self.vx += force_x * 0.2  # 力的大小
        self.vy += force_y * 0.2

        # 物理更新
        self.x += self.vx * 0.1
        self.y += self.vy * 0.1
        self.vx *= self.friction
        self.vy *= self.friction

        # 边界处理（弹性碰撞）
        half_size = self.world_size / 2
        if self.x > half_size:
            self.x = half_size - (self.x - half_size)
            self.vx = -self.vx * 0.8
        elif self.x < -half_size:
            self.x = -half_size - (self.x + half_size)
            self.vx = -self.vx * 0.8

        if self.y > half_size:
            self.y = half_size - (self.y - half_size)
            self.vy = -self.vy * 0.8
        elif self.y < -half_size:
            self.y = -half_size - (self.y + half_size)
            self.vy = -self.vy * 0.8

        # 添加噪声
        self.x += np.random.randn() * self.noise_std
        self.y += np.random.randn() * self.noise_std
        self.vx += np.random.randn() * self.noise_std * 0.1
        self.vy += np.random.randn() * self.noise_std * 0.1

        # 计算奖励
        reward = 0.0
        for zone in self.reward_zones:
            dx = self.x - zone['x']
            dy = self.y - zone['y']
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < zone['radius']:
                reward = zone['reward']
                break

        self.step_count += 1
        done = self.step_count >= self.max_steps

        return self._get_state(), reward, done


# ============================================
# 好奇心机制模块（V4.0 连续动作版）
# ============================================

class PredictionErrorCuriosity(nn.Module):
    """基于预测误差的好奇心 - 支持连续动作"""

    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128, lr=0.0005):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.dynamics_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        ).to(DEVICE)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.998)
        self.loss_history = deque(maxlen=100)

    def forward(self, state, action, next_state):
        """计算好奇心值"""
        if state.dim() == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        input_tensor = torch.cat([state, action], dim=-1)
        pred_next = self.dynamics_model(input_tensor)

        prediction_error = F.mse_loss(pred_next, next_state, reduction='mean')
        curiosity = torch.tanh(prediction_error)

        return curiosity, prediction_error.item()

    def update(self, state, action, next_state):
        """更新动力学模型"""
        if state.dim() == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        self.optimizer.zero_grad()
        input_tensor = torch.cat([state, action], dim=-1)
        pred = self.dynamics_model(input_tensor)
        loss = F.mse_loss(pred, next_state)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        self.loss_history.append(loss.item())
        return loss.item()

    def step_scheduler(self):
        self.scheduler.step()


class NoveltyCuriosity2D:
    """基于状态新颖性的好奇心 - 2D 版本"""

    def __init__(self, state_dim=4, memory_size=10000, novelty_decay=0.998):
        self.memory = deque(maxlen=memory_size)
        self.novelty_decay = novelty_decay
        self.exploration_count = 0
        self.state_dim = state_dim

    def compute_novelty(self, state):
        """计算状态新颖性"""
        if len(self.memory) == 0:
            self.memory.append(state.copy())
            return 1.0

        # 计算到最近邻的距离（欧氏距离）
        min_dist = min(np.linalg.norm(state - mem) for mem in self.memory)
        novelty = min(1.0, min_dist / 5.0)  # 2D 空间距离范围更大
        novelty *= (self.novelty_decay ** (self.exploration_count / 100))

        # 随机采样存储
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
    """随机网络蒸馏 - 2D 版本"""

    def __init__(self, state_dim=4, embedding_dim=128, hidden_dim=256, lr=0.0005):
        super().__init__()

        self.target = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        ).to(DEVICE)

        self.predictor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        ).to(DEVICE)

        self.optimizer = optim.Adam(self.predictor.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.998)

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
# 信用分配系统（连续动作版）
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
# 自适应好奇心系统（V4.0）
# ============================================

@dataclass
class CuriosityConfigV4:
    state_dim: int = 4
    action_dim: int = 2
    num_mechanisms: int = 3
    hidden_dim: int = 128
    rnd_embedding_dim: int = 128
    learning_rate: float = 0.003
    weight_update_freq: int = 20
    credit_gamma: float = 0.95
    save_freq: int = 50
    checkpoint_dir: str = "checkpoints_v4"
    min_weight: float = 0.10
    novelty_reset_freq: int = 200
    exploration_bonus: float = 0.05
    weight_reset_threshold: float = 0.95
    high_score_threshold: float = 150
    lock_duration: int = 50


class AdaptiveCuriositySystemV4:
    """V4.0 自适应好奇心系统 - 支持连续动作"""

    def __init__(self, config: CuriosityConfigV4):
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
            'explore_rates': []
        }

        self.step_count = 0
        self.episode_count = 0
        self.start_time = time.time()
        self._current_curiosity = None
        self._current_weights = None

        self.best_return = 0
        self.best_episode = 0
        self.high_score_lock_counter = 0

        self.champion_path = os.path.join(config.checkpoint_dir, "champion_model_v4.pth")
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
            'rnd': rnd_c.item(), 'pred_error': pred_e, 'rnd_error': rnd_e,
            'exploration_bonus': exploration_bonus
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

    def end_episode(self, episode_return, episode_time, details=None):
        credit = self.tracker.compute_credit()
        self._update_weights_from_credit(credit)

        self.history['episode_returns'].append(episode_return)
        self.history['weights_history'].append(self.weights.copy())
        self.history['episode_times'].append(episode_time)
        self.history['timestamps'].append(time.time() - self.start_time)
        self.history['mechanism_credits'].append(credit)

        if details:
            self.history['pred_errors'].append(details.get('pred_error', 0))
            self.history['rnd_errors'].append(details.get('rnd_error', 0))
            self.history['novelty_values'].append(details.get('novelty', 0))

        self.tracker.reset()
        self.episode_count += 1

        # 高分锁定
        if episode_return >= self.config.high_score_threshold:
            self.high_score_lock_counter = self.config.lock_duration
            print(f"   🔒 高分锁定激活！({episode_return} >= {self.config.high_score_threshold})")
            print(f"      接下来 {self.config.lock_duration} episodes 保持低探索率")
        elif self.high_score_lock_counter > 0:
            self.high_score_lock_counter -= 1

        # 定期重置 Novelty 记忆
        if self.episode_count % self.config.novelty_reset_freq == 0:
            self.mechanisms['novelty'].reset_memory(keep_ratio=0.7)

        # 冠军模型保存
        if episode_return > self.best_return:
            self.best_return = episode_return
            self.best_episode = self.episode_count
            self._save_champion()
            print(f"   🏆 新纪录！Episode {self.episode_count}: {episode_return:.2f} 分")

        # 定期清理显存
        if self.episode_count % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.episode_count % self.config.save_freq == 0:
            self.save_checkpoint()

        return episode_return

    def is_high_score_lock_active(self):
        return self.high_score_lock_counter > 0

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

    def _update_weights(self):
        if len(self.history['episode_returns']) < 5:
            return
        recent_credits = np.mean(self.history['mechanism_credits'][-5:], axis=0)
        self._update_weights_from_credit(recent_credits)

    def _update_weights_from_credit(self, credit):
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
            'best_episode': self.best_episode,
            'weights_history': self.history['weights_history'][-100:] if self.history['weights_history'] else []
        }


# ============================================
# 2D 可视化工具
# ============================================

class Visualizer2D:
    """2D 环境可视化"""

    def __init__(self):
        self.fig = None
        self.axes = None
        self.initialized = False

    def setup(self):
        plt.close('all')
        self.fig = plt.figure(figsize=(16, 10))

        # 2x3 子图布局
        gs = self.fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        self.axes = {
            'trajectory': self.fig.add_subplot(gs[0, :2]),  # 轨迹图占2列
            'returns': self.fig.add_subplot(gs[0, 2]),
            'weights': self.fig.add_subplot(gs[1, 0]),
            'curiosity': self.fig.add_subplot(gs[1, 1]),
            'credit': self.fig.add_subplot(gs[1, 2])
        }

        self.fig.suptitle('2D Curiosity-Driven Exploration - V4.0', fontsize=14, fontweight='bold')
        self.initialized = True
        plt.ion()
        plt.show()

    def update_trajectory(self, ax, positions, reward_zones, current_pos=None):
        """更新轨迹图"""
        ax.clear()

        # 绘制奖励区域
        for zone in reward_zones:
            circle = Circle((zone['x'], zone['y']), zone['radius'],
                            color='green', alpha=0.3)
            ax.add_patch(circle)
            ax.text(zone['x'], zone['y'], f'{zone["reward"]}',
                    ha='center', va='center', fontsize=8)

        # 绘制轨迹
        if len(positions) > 0:
            positions = np.array(positions)
            ax.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.5, linewidth=1)
            ax.scatter(positions[0, 0], positions[0, 1], c='green', s=50, marker='o', label='Start')
            ax.scatter(positions[-1, 0], positions[-1, 1], c='red', s=80, marker='o', label='Current')

        # 绘制当前速度向量
        if current_pos is not None and len(positions) > 1:
            if len(positions) >= 2:
                dx = positions[-1][0] - positions[-2][0]
                dy = positions[-1][1] - positions[-2][1]
                ax.arrow(positions[-1][0], positions[-1][1], dx, dy,
                         head_width=0.2, head_length=0.2, fc='red', ec='red', alpha=0.7)

        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('2D Trajectory')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_aspect('equal')

    def update(self, system, env, positions, current_state=None):
        if not self.initialized:
            self.setup()

        if self.fig is None:
            return

        stats = system.get_stats()
        history = system.history
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        labels = ['PredErr', 'Novelty', 'RND']

        # 2D 轨迹图
        self.update_trajectory(self.axes['trajectory'], positions,
                               env.reward_zones, current_state)

        # 奖励曲线
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

        # 权重演化
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

        # 好奇心信号
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

        # 信用分配
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
# 训练函数（V4.0）
# ============================================

def train_v4(num_episodes: int = 500,
             render: bool = True,
             save_freq: int = 50,
             verbose_freq: int = 20):
    """V4.0 训练函数"""
    print("=" * 60)
    print("2D CURIOSITY-DRIVEN EXPLORATION - VERSION 4.0")
    print("=" * 60)
    print("新特性:")
    print("  ✅ 2D 空间 (x, y 坐标)")
    print("  ✅ 连续动作空间 (力的大小和方向)")
    print("  ✅ 圆形奖励区域")
    print("  ✅ 2D 轨迹可视化")
    print("=" * 60)

    config = CuriosityConfigV4()
    env = World2D()
    system = AdaptiveCuriositySystemV4(config)
    visualizer = Visualizer2D() if render else None

    print(f"\n目标 Episodes: {num_episodes}")
    print(f"设备: {DEVICE}")
    print(f"状态维度: {config.state_dim} (x, y, vx, vy)")
    print(f"动作维度: {config.action_dim} (force_x, force_y)")
    print(f"奖励区域: {len(env.reward_zones)} 个")
    print("-" * 60)

    total_start = time.time()

    # 探索率参数（连续动作需要更高探索率）
    explore_rate = 0.30  # 初始探索率
    explore_rate_min = 0.10
    explore_rate_max = 0.50
    explore_smoothing = 0.97

    for episode in range(num_episodes):
        state = env.reset(extreme=(episode > 100))
        episode_return = 0
        episode_steps = 0
        episode_start = time.time()
        episode_details = {'pred_error': 0, 'rnd_error': 0, 'novelty': 0}

        # 记录轨迹用于可视化
        positions = [(state[0], state[1])]

        episode_progress = episode / num_episodes

        # 动态调整探索率
        if episode >= 20 and not system.is_high_score_lock_active():
            recent_returns = system.history['episode_returns'][-20:] if system.history['episode_returns'] else [0]
            recent_avg = np.mean(recent_returns)

            if recent_avg > 100:
                target_rate = max(explore_rate_min, explore_rate * 0.98)
            elif recent_avg > 80:
                target_rate = max(explore_rate_min, explore_rate * 0.99)
            elif recent_avg < 40:
                target_rate = min(explore_rate_max, explore_rate * 1.05)
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
            # 动作选择
            if random.random() < explore_rate:
                # 随机探索
                action = np.random.uniform(-1.0, 1.0, size=2)
            else:
                # 贪心策略：向最近的奖励区域移动
                nearest_zone = min(env.reward_zones,
                                   key=lambda z: ((state[0] - z['x']) ** 2 + (state[1] - z['y']) ** 2) ** 0.5)
                dx = nearest_zone['x'] - state[0]
                dy = nearest_zone['y'] - state[1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 0.1:
                    action = np.array([np.clip(dx / dist, -1, 1), np.clip(dy / dist, -1, 1)])
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
            positions.append((state[0], state[1]))

            if done:
                break

        # 计算平均值
        for k in episode_details:
            episode_details[k] /= max(episode_steps, 1)

        episode_time = time.time() - episode_start
        system.end_episode(episode_return, episode_time, episode_details)

        # 更新可视化
        if visualizer and episode % 5 == 0:
            visualizer.update(system, env, positions, state)
            plt.pause(0.01)

        # 输出进度
        if (episode + 1) % verbose_freq == 0:
            stats = system.get_stats()
            elapsed = time.time() - total_start
            weights = system.weights
            lock_status = "🔒" if system.is_high_score_lock_active() else "  "
            print(f"Ep {episode + 1:4d}/{num_episodes} | "
                  f"Return: {episode_return:6.2f} | "
                  f"Avg Return: {stats['avg_return']:6.2f} | "
                  f"Weights: [{weights[0]:.2f}, {weights[1]:.2f}, {weights[2]:.2f}] | "
                  f"Explore: {explore_rate:.3f} | "
                  f"Best: {stats['best_return']:.1f} {lock_status} | "
                  f"Time: {elapsed / 60:.1f}min")

        # 每 50 episodes 打印统计
        if (episode + 1) % 50 == 0:
            stats = system.get_stats()
            print(f"   📊 统计: Avg Return={stats['avg_return']:.1f}, "
                  f"Best={stats['best_return']:.1f}, Explore={explore_rate:.3f}")

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
# 主程序
# ============================================

if __name__ == "__main__":
    try:
        from PyQt5 import QtCore

        print("✅ PyQt5 已安装")
    except ImportError:
        print("⚠️ PyQt5 未安装，可视化将使用 Agg 后端")
        matplotlib.use('Agg')

    # 运行 V4.0 训练
    system = train_v4(
        num_episodes=500,
        render=True,
        save_freq=50,
        verbose_freq=20
    )

    print("\n✨ V4.0 训练完成！开始探索 2D 世界！✨")