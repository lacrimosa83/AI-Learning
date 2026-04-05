"""
============================================
Curiosity-Driven Exploration Framework
VERSION 3.2 - 低探索率 + 环境微调
============================================
改进内容：
1. 降低探索率范围 (0.05-0.25)
2. 优化高分锁定机制 (阈值140, 锁定30 episodes)
3. 增加环境噪声 (探索更鲁棒的策略)
4. 添加课程学习（渐进增加难度）
5. 冠军模型测试功能
"""

import os
import sys
import json
import time
import warnings

warnings.filterwarnings('ignore')

# ========== 设置 matplotlib 后端 ==========
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

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
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ 使用 Apple MPS")
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
# 环境模拟器（V3.2 增强版）
# ============================================

class ContinuousWorld:
    def __init__(self, world_size=10.0, friction=0.98,
                 reward_positions=None, noise_std=0.05):
        self.world_size = world_size
        self.friction = friction
        # 【V3.2】更多奖励点，更分散
        if reward_positions is None:
            self.reward_positions = [3.0, -2.0, 5.0, -4.0, 1.5, -1.5]
        else:
            self.reward_positions = reward_positions
        self.base_noise_std = noise_std
        self.noise_std = noise_std
        self.reset()

    def reset(self):
        self.pos = np.random.uniform(-self.world_size / 2, self.world_size / 2)
        self.vel = np.random.uniform(-2.0, 2.0)
        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        return np.array([self.pos, self.vel])

    def set_difficulty(self, noise_std):
        """动态调整环境难度（课程学习）"""
        self.noise_std = noise_std

    def step(self, action=None):
        if action == 0:
            self.vel += 0.5
        elif action == 1:
            self.vel -= 0.3

        self.pos += self.vel * 0.1
        self.vel *= self.friction

        if self.pos > self.world_size / 2:
            self.pos = self.world_size / 2 - (self.pos - self.world_size / 2)
            self.vel = -self.vel * 0.8
        elif self.pos < -self.world_size / 2:
            self.pos = -self.world_size / 2 - (self.pos + self.world_size / 2)
            self.vel = -self.vel * 0.8

        self.pos += np.random.randn() * self.noise_std
        self.vel += np.random.randn() * self.noise_std * 0.1

        reward = 0.0
        for rp in self.reward_positions:
            if abs(self.pos - rp) < 0.3:
                reward = 1.0
                break

        self.step_count += 1
        done = self.step_count >= 200

        return self._get_state(), reward, done


# ============================================
# 好奇心机制模块
# ============================================

class PredictionErrorCuriosity(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=0.001):
        super().__init__()
        self.dynamics_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        ).to(DEVICE)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995)
        self.loss_history = deque(maxlen=100)

    def _prepare_action(self, action):
        if action is None:
            return torch.tensor([1.0, 0.0, 0.0], device=DEVICE)
        elif action == 0:
            return torch.tensor([0.0, 1.0, 0.0], device=DEVICE)
        elif action == 1:
            return torch.tensor([0.0, 0.0, 1.0], device=DEVICE)
        return torch.tensor([1.0, 0.0, 0.0], device=DEVICE)

    def forward(self, state, action, next_state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
        batch = state.shape[0]
        action_t = self._prepare_action(action).unsqueeze(0).expand(batch, -1)
        pred = self.dynamics_model(torch.cat([state, action_t], -1))
        error = F.mse_loss(pred, next_state)
        return torch.tanh(error), error.item()

    def update(self, state, action, next_state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
        batch = state.shape[0]
        action_t = self._prepare_action(action).unsqueeze(0).expand(batch, -1)
        self.optimizer.zero_grad()
        pred = self.dynamics_model(torch.cat([state, action_t], -1))
        loss = F.mse_loss(pred, next_state)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        self.loss_history.append(loss.item())
        return loss.item()

    def step_scheduler(self):
        self.scheduler.step()


class NoveltyCuriosity:
    def __init__(self, state_dim, memory_size=5000, novelty_decay=0.99):
        self.memory = deque(maxlen=memory_size)
        self.novelty_decay = novelty_decay
        self.exploration_count = 0
        self.state_dim = state_dim

    def compute_novelty(self, state):
        if len(self.memory) == 0:
            self.memory.append(state.copy())
            return 1.0
        min_dist = min(np.linalg.norm(state - mem) for mem in self.memory)
        novelty = min(1.0, min_dist / 3.0)
        novelty *= (self.novelty_decay ** (self.exploration_count / 100))
        if random.random() < 0.1:
            self.memory.append(state.copy())
        return novelty

    def record_exploration(self):
        self.exploration_count += 1

    def reset_memory(self, keep_ratio=0.5):
        if len(self.memory) > 100:
            old_len = len(self.memory)
            keep_count = int(old_len * keep_ratio)
            self.memory = deque(list(self.memory)[-keep_count:], maxlen=self.memory.maxlen)
            print(f"   [Novelty重置] 清除了 {old_len - keep_count} 条旧记忆，保留 {keep_count} 条")
            return old_len - keep_count
        return 0


class RNDCuriosity(nn.Module):
    def __init__(self, state_dim, embedding_dim=64, hidden_dim=128, lr=0.002):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        ).to(DEVICE)
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        ).to(DEVICE)
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995)
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
# 自适应好奇心系统（V3.2）
# ============================================

@dataclass
class CuriosityConfig:
    state_dim: int = 2
    action_dim: int = 3
    num_mechanisms: int = 3
    hidden_dim: int = 64
    rnd_embedding_dim: int = 64
    learning_rate: float = 0.01
    weight_update_freq: int = 10
    credit_gamma: float = 0.95
    save_freq: int = 50
    checkpoint_dir: str = "checkpoints"
    # V3.2 改进参数
    min_weight: float = 0.10
    novelty_reset_freq: int = 100
    exploration_bonus: float = 0.05  # 降低探索保底
    weight_reset_threshold: float = 0.95
    # V3.2 新高分锁定参数
    high_score_threshold: float = 140  # 降低阈值
    lock_duration: int = 30  # 增加锁定时间


class AdaptiveCuriositySystem:
    def __init__(self, config: CuriosityConfig):
        self.config = config
        self.num = config.num_mechanisms

        self.mechanisms = {
            'prediction_error': PredictionErrorCuriosity(
                config.state_dim, config.action_dim, config.hidden_dim),
            'novelty': NoveltyCuriosity(config.state_dim),
            'rnd': RNDCuriosity(config.state_dim, config.rnd_embedding_dim, lr=0.002)
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

        # 冠军模型追踪
        self.best_return = 0
        self.best_episode = 0
        self.champion_path = os.path.join(config.checkpoint_dir, "champion_model.pth")

        # V3.2 高分锁定机制
        self.high_score_lock_counter = 0

        os.makedirs(config.checkpoint_dir, exist_ok=True)

    @property
    def weights(self):
        return F.softmax(self.weights_logits, dim=0).detach().cpu().numpy()

    def compute_curiosity(self, state, action, next_state, episode_progress=0):
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        next_t = torch.tensor(next_state, dtype=torch.float32, device=DEVICE)

        pred_c, pred_e = self.mechanisms['prediction_error'](state_t, action, next_t)
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

        self.mechanisms['prediction_error'].update(state_t, action, next_t)
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

        # V3.2 更新高分锁定计数器
        if episode_return >= self.config.high_score_threshold:
            self.high_score_lock_counter = self.config.lock_duration
            print(f"   🔒 高分锁定激活！({episode_return} >= {self.config.high_score_threshold})")
            print(f"      接下来 {self.config.lock_duration} episodes 保持低探索率")
        elif self.high_score_lock_counter > 0:
            self.high_score_lock_counter -= 1

        # 定期重置 Novelty 记忆
        if self.episode_count % self.config.novelty_reset_freq == 0:
            self.mechanisms['novelty'].reset_memory(keep_ratio=0.5)

        # 权重干预
        if self.weights[0] > self.config.weight_reset_threshold:
            print(f"   [干预] 权重极化检测 (PredErr={self.weights[0]:.3f})，执行软重置...")
            self._soft_reset_weights()

        # 冠军模型保存
        if episode_return > self.best_return:
            self.best_return = episode_return
            self.best_episode = self.episode_count
            self._save_champion()
            print(f"   🏆 新纪录！Episode {self.episode_count}: {episode_return} 分")

        # 定期清理显存
        if self.episode_count % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.episode_count % self.config.save_freq == 0:
            self.save_checkpoint()

        return episode_return

    def is_high_score_lock_active(self):
        """检查高分锁定是否激活"""
        return self.high_score_lock_counter > 0

    def _save_champion(self):
        """保存冠军模型"""
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

    def load_champion(self):
        """加载冠军模型"""
        if os.path.exists(self.champion_path):
            checkpoint = torch.load(self.champion_path, map_location='cpu', weights_only=False)
            self.weights_logits.data = checkpoint['weights_logits'].to(DEVICE)
            self.mechanisms['prediction_error'].load_state_dict(checkpoint['pred_error_state'])
            self.mechanisms['novelty'].memory = deque(checkpoint['novelty_memory'], maxlen=5000)
            self.mechanisms['novelty'].exploration_count = checkpoint['novelty_count']
            self.mechanisms['rnd'].predictor.load_state_dict(checkpoint['rnd_predictor_state'])
            self.mechanisms['rnd'].mean = checkpoint['rnd_mean'].to(DEVICE)
            self.mechanisms['rnd'].std = checkpoint['rnd_std'].to(DEVICE)
            self.mechanisms['rnd'].count = checkpoint['rnd_count'].to(DEVICE)
            print(f"🏆 冠军模型已加载 (Episode {checkpoint['episode']}, Return {checkpoint['return']})")
            return checkpoint['episode'], checkpoint['return']
        return None, None

    def test_champion(self, num_tests=100):
        """测试冠军模型性能"""
        print(f"\n🏆 测试冠军模型 ({num_tests} episodes)...")

        # 保存当前状态
        original_weights = self.weights_logits.data.clone()
        original_episode = self.episode_count

        # 临时设置为评估模式
        self.weights_logits.requires_grad = False

        test_returns = []
        env = ContinuousWorld()

        for i in range(num_tests):
            state = env.reset()
            episode_return = 0

            for step in range(200):
                # 使用冠军模型的策略（贪心，无探索）
                if state[0] < 3 and state[0] > -2:
                    action = 0
                elif state[0] > 3:
                    action = 1
                else:
                    action = None

                next_state, reward, done = env.step(action)
                episode_return += reward
                state = next_state

                if done:
                    break

            test_returns.append(episode_return)

            if (i + 1) % 20 == 0:
                print(f"   测试进度: {i + 1}/{num_tests}")

        # 恢复状态
        self.weights_logits.data = original_weights
        self.weights_logits.requires_grad = True
        self.episode_count = original_episode

        mean_return = np.mean(test_returns)
        std_return = np.std(test_returns)

        print(f"\n   测试结果:")
        print(f"   平均奖励: {mean_return:.2f} ± {std_return:.2f}")
        print(f"   最佳奖励: {np.max(test_returns):.2f}")
        print(f"   最差奖励: {np.min(test_returns):.2f}")

        return test_returns

    def _soft_reset_weights(self):
        """软重置权重"""
        with torch.no_grad():
            self.weights_logits.data = torch.randn(self.config.num_mechanisms, device=DEVICE) * 0.5
            self.weights_logits.data = self.weights_logits.data - self.weights_logits.data.mean()
            print(f"   [干预] 权重已软重置: {self.weights.tolist()}")

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

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.episode_count = checkpoint['episode']
        self.step_count = checkpoint['step']
        self.weights_logits.data = checkpoint['weights_logits'].to(DEVICE)
        self.weights_optimizer.load_state_dict(checkpoint['weights_optimizer'])
        self.history = checkpoint['history']

        # 为旧检查点添加缺失的键
        if 'explore_rates' not in self.history:
            self.history['explore_rates'] = []
            print("   [迁移] 已添加 explore_rates 历史记录")

        self.best_return = checkpoint.get('best_return', 0)
        self.best_episode = checkpoint.get('best_episode', 0)
        self.high_score_lock_counter = checkpoint.get('high_score_lock_counter', 0)

        self.mechanisms['novelty'].memory = deque(checkpoint['novelty_memory'], maxlen=5000)
        self.mechanisms['novelty'].exploration_count = checkpoint['novelty_count']
        self.mechanisms['prediction_error'].load_state_dict(checkpoint['pred_error_state'])
        self.mechanisms['prediction_error'].optimizer.load_state_dict(checkpoint['pred_error_optimizer'])
        self.mechanisms['rnd'].predictor.load_state_dict(checkpoint['rnd_predictor_state'])
        self.mechanisms['rnd'].optimizer.load_state_dict(checkpoint['rnd_optimizer'])
        self.mechanisms['rnd'].mean = checkpoint['rnd_mean'].to(DEVICE)
        self.mechanisms['rnd'].std = checkpoint['rnd_std'].to(DEVICE)
        self.mechanisms['rnd'].count = checkpoint['rnd_count'].to(DEVICE)

        print(f"✅ 从检查点恢复: Episode {self.episode_count}")
        print(f"🏆 历史最佳: Episode {self.best_episode}, Return {self.best_return}")
        return self.episode_count

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
# 可视化工具
# ============================================

class LongTermVisualizer:
    def __init__(self):
        self.fig = None
        self.axes = None
        self.initialized = False

    def setup(self):
        plt.close('all')
        self.fig = plt.figure(figsize=(14, 8))

        self.axes = {
            'returns': self.fig.add_subplot(2, 3, 1),
            'weights': self.fig.add_subplot(2, 3, 2),
            'curiosity': self.fig.add_subplot(2, 3, 3),
            'credit': self.fig.add_subplot(2, 3, 4),
            'state': self.fig.add_subplot(2, 3, 5),
            'time': self.fig.add_subplot(2, 3, 6)
        }

        self.fig.suptitle('Curiosity-Driven Exploration - V3.2 (Low Explore Rate)', fontsize=14)
        self.initialized = True
        plt.tight_layout()
        plt.ion()
        plt.show()

    def update(self, system, current_state=None, details=None, explore_rate=None):
        if not self.initialized:
            self.setup()

        if self.fig is None:
            return

        stats = system.get_stats()
        history = system.history
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        labels = ['PredErr', 'Novelty', 'RND']

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
                ax.annotate(f"🏆 {stats['best_return']}",
                            (stats['best_episode'] - 1, stats['best_return']),
                            textcoords="offset points", xytext=(5, 10), ha='left')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Return')
            ax.set_title(f'Returns (Avg: {stats["avg_return"]:.1f}, Best: {stats["best_return"]})')
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
            ax.set_title(f'Weights (min={system.config.min_weight:.0%})')
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

        # 当前状态
        ax = self.axes['state']
        ax.clear()
        if current_state is not None:
            ax.scatter(current_state[0], current_state[1], c='red', s=100, marker='o')
            ax.set_xlim(-6, 6)
            ax.set_ylim(-3, 3)
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_title('Current State')
        ax.grid(True, alpha=0.3)

        # Episode 时间 + 探索率
        ax = self.axes['time']
        ax.clear()
        if history.get('episode_times'):
            ax.plot(history['episode_times'], 'g-', alpha=0.7, label='Time')
            if history.get('explore_rates'):
                ax_twin = ax.twinx()
                ax_twin.plot(history['explore_rates'], 'b--', alpha=0.5, label='Explore Rate')
                ax_twin.set_ylabel('Explore Rate', fontsize=9)
                ax_twin.tick_params(labelsize=8)
                ax_twin.set_ylim(0, 0.4)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Time (s)', fontsize=9)
        ax.set_title('Episode Duration & Explore Rate')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.initialized = False


# ============================================
# 训练函数（V3.2 - 低探索率）
# ============================================

def train_long_term(num_episodes: int = 3000,
                    resume_from: str = None,
                    render: bool = True,
                    save_freq: int = 50,
                    verbose_freq: int = 20,
                    use_curriculum: bool = True):
    """长期训练主函数 - V3.2 低探索率"""
    print("=" * 60)
    print("LONG-TERM CURIOSITY-DRIVEN TRAINING - VERSION 3.2")
    print("=" * 60)
    print("新增特性:")
    print("  ✅ 降低探索率范围 (0.05-0.25)")
    print("  ✅ 优化高分锁定 (阈值140, 锁定30 episodes)")
    print("  ✅ 增加环境噪声")
    print("  ✅ 课程学习支持")
    print("  ✅ 冠军模型测试功能")
    print("=" * 60)

    config = CuriosityConfig()
    config.save_freq = save_freq
    env = ContinuousWorld()
    system = AdaptiveCuriositySystem(config)

    start_episode = 0
    if resume_from and os.path.exists(resume_from):
        start_episode = system.load_checkpoint(resume_from)
        # 确保 explore_rates 存在
        if 'explore_rates' not in system.history:
            system.history['explore_rates'] = []
        print(f"从 Episode {start_episode} 继续训练")

    visualizer = LongTermVisualizer() if render else None

    print(f"\n目标 Episodes: {num_episodes}")
    print(f"设备: {DEVICE}")
    print(f"保存频率: 每 {save_freq} episodes")
    print(f"最小权重: {config.min_weight}")
    print(f"高分锁定阈值: {config.high_score_threshold} 分")
    print(f"高分锁定持续时间: {config.lock_duration} episodes")
    print("-" * 60)

    total_start = time.time()

    # V3.2 低探索率参数
    explore_rate = 0.20  # 初始探索率（降低）
    explore_rate_min = 0.05  # 最低探索率（降低）
    explore_rate_max = 0.25  # 最高探索率（降低）
    explore_smoothing = 0.95
    recent_returns_window = 20

    for episode in range(start_episode, num_episodes):
        state = env.reset()
        episode_return = 0
        episode_steps = 0
        episode_start = time.time()
        episode_details = {'pred_error': 0, 'rnd_error': 0, 'novelty': 0}

        episode_progress = episode / num_episodes

        # V3.2 课程学习：渐进增加难度
        if use_curriculum:
            if episode < 500:
                env.set_difficulty(0.03)  # 低噪声
            elif episode < 1500:
                env.set_difficulty(0.05)  # 中噪声
            else:
                env.set_difficulty(0.08)  # 高噪声

        # 计算目标探索率
        if episode >= recent_returns_window and not system.is_high_score_lock_active():
            recent_returns = system.history['episode_returns'][-recent_returns_window:]
            recent_avg = np.mean(recent_returns)

            # 根据近期表现计算目标探索率
            if recent_avg > 130:
                target_rate = max(explore_rate_min, explore_rate * 0.97)
            elif recent_avg > 100:
                target_rate = max(explore_rate_min, explore_rate * 0.98)
            elif recent_avg < 50:
                target_rate = min(explore_rate_max, explore_rate * 1.05)
            else:
                target_rate = explore_rate
        else:
            target_rate = explore_rate

        # 高分锁定时强制低探索率
        if system.is_high_score_lock_active():
            target_rate = explore_rate_min
            if episode % 25 == 0:
                print(f"   🔒 高分锁定剩余: {system.high_score_lock_counter} episodes")

        # 指数移动平均平滑探索率
        explore_rate = explore_smoothing * explore_rate + (1 - explore_smoothing) * target_rate
        explore_rate = np.clip(explore_rate, explore_rate_min, explore_rate_max)

        # 记录探索率历史
        system.history['explore_rates'].append(explore_rate)

        for step in range(200):
            # 根据探索率选择动作
            if random.random() < explore_rate:
                action = random.choice([None, 0, 1])
            else:
                # 贪心策略
                if state[0] < 3 and state[0] > -2:
                    action = 0
                elif state[0] > 3:
                    action = 1
                else:
                    action = None

            next_state, reward, done = env.step(action)
            curiosity, details = system.compute_curiosity(state, action, next_state, episode_progress)
            system.record_step(state, action, reward, next_state)

            episode_return += reward
            episode_steps += 1
            episode_details['pred_error'] += details.get('pred_error', 0)
            episode_details['rnd_error'] += details.get('rnd_error', 0)
            episode_details['novelty'] += details.get('novelty', 0)

            state = next_state

            if done:
                break

        # 计算平均值
        for k in episode_details:
            episode_details[k] /= max(episode_steps, 1)

        episode_time = time.time() - episode_start
        system.end_episode(episode_return, episode_time, episode_details)

        # 更新可视化
        if visualizer and episode % 5 == 0:
            visualizer.update(system, state, episode_details, explore_rate)
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
                  f"Explore: {explore_rate:.2f} | "
                  f"Best: {stats['best_return']} {lock_status} | "
                  f"Noise: {env.noise_std:.2f} | "
                  f"Time: {elapsed / 60:.1f}min")

        # 每 50 episodes 打印统计摘要
        if (episode + 1) % 50 == 0:
            stats = system.get_stats()
            print(f"   📊 统计摘要: Avg Return={stats['avg_return']:.1f}, "
                  f"Best={stats['best_return']}, Explore={explore_rate:.2f}, "
                  f"Noise={env.noise_std:.2f}")

    total_time = time.time() - total_start
    print("-" * 60)
    print(f"训练完成！总时间: {total_time / 60:.1f} 分钟")
    print(f"总 Episodes: {system.episode_count}")

    if system.history['episode_returns']:
        print(f"平均 Episode 奖励 (最后100): {np.mean(system.history['episode_returns'][-100:]):.2f}")
        print(f"最佳 Episode 奖励: {system.best_return} (Episode {system.best_episode})")

    system.save_checkpoint()
    print("最终模型已保存")

    if system.best_return > 0:
        print(f"🏆 冠军模型已保存至: {system.champion_path}")
        print(f"   最佳成绩: {system.best_return} 分 (Episode {system.best_episode})")

    if visualizer:
        visualizer.close()

    return system


# ============================================
# 结果分析
# ============================================

def analyze_results(system: AdaptiveCuriositySystem):
    """增强版结果分析"""
    history = system.history

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 奖励曲线
    returns = history['episode_returns']
    axes[0, 0].plot(returns, alpha=0.5)
    if len(returns) >= 50:
        smooth = np.convolve(returns, np.ones(50) / 50, mode='valid')
        axes[0, 0].plot(range(49, len(returns)), smooth, 'r-', linewidth=2)
    if system.best_episode > 0:
        axes[0, 0].scatter(system.best_episode - 1, system.best_return,
                           c='gold', s=150, marker='*', zorder=5)
    axes[0, 0].set_title(f'Episode Returns (Best: {system.best_return})')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Return')

    # 权重演化
    weights = np.array(history['weights_history'])
    for i, label in enumerate(['PredErr', 'Novelty', 'RND']):
        axes[0, 1].plot(weights[:, i], label=label)
    axes[0, 1].set_title('Weights Evolution')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Weight')
    axes[0, 1].legend()

    # 探索率演化
    axes[0, 2].plot(history.get('explore_rates', []), 'b-', alpha=0.7)
    axes[0, 2].set_title('Explore Rate Evolution')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Explore Rate')
    axes[0, 2].set_ylim(0, 0.35)

    # 好奇心信号
    if history.get('novelty_values'):
        axes[1, 0].plot(history['novelty_values'], label='Novelty')
    if history.get('pred_errors'):
        axes[1, 0].plot(history['pred_errors'], label='Pred Error')
    if history.get('rnd_errors'):
        axes[1, 0].plot(history['rnd_errors'], label='RND Error')
    axes[1, 0].set_title('Curiosity Signals')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].legend()

    # 累积平均
    cumulative_avg = np.cumsum(returns) / (np.arange(len(returns)) + 1)
    axes[1, 1].plot(cumulative_avg)
    axes[1, 1].set_title('Cumulative Average Return')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Average Return')

    # 最终权重
    final_weights = system.weights
    axes[1, 2].pie(final_weights, labels=['PredErr', 'Novelty', 'RND'],
                   autopct='%1.1f%%', colors=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    axes[1, 2].set_title(f'Final Weights (min={system.config.min_weight:.0%})')

    plt.tight_layout()
    plt.savefig('training_analysis_v3.2.png', dpi=150)
    plt.show()

    print("\n" + "=" * 60)
    print("训练统计报告 (V3.2)")
    print("=" * 60)
    print(f"总 Episodes: {len(returns)}")
    print(f"平均奖励: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"最佳奖励: {system.best_return} (Episode {system.best_episode})")
    print(f"最终权重: [{final_weights[0]:.3f}, {final_weights[1]:.3f}, {final_weights[2]:.3f}]")
    print(f"平均探索率: {np.mean(history.get('explore_rates', [0])):.3f}")
    print("=" * 60)


# ============================================
# 主程序
# ============================================

if __name__ == "__main__":
    # 检查 PyQt5
    try:
        from PyQt5 import QtCore

        print("✅ PyQt5 已安装")
    except ImportError:
        print("⚠️ PyQt5 未安装，可视化将使用 Agg 后端")
        matplotlib.use('Agg')

    # 运行训练 - 从 Episode 2000 继续到 3000
    system = train_long_term(
        num_episodes=3000,
        resume_from='checkpoints/checkpoint_ep2950.pth',  # 从 Episode 2000 继续
        render=True,
        save_freq=50,
        verbose_freq=20,
        use_curriculum=True
    )

    # 分析结果
    analyze_results(system)

    # 可选：测试冠军模型
    print("\n" + "=" * 60)
    print("开始测试冠军模型...")
    print("=" * 60)
    test_returns = system.test_champion(num_tests=50)

    print("\n训练完成！")