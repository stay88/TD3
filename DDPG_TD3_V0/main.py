import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import TD3
import torch as th
import os
from stable_baselines3.common.monitor import Monitor

from logging_utils import build_callbacks

# 从tools.py导入必要的函数
from tools import generate_nakagami_channel, generate_RIS_from_phase, rate_uk, AMDEP, precoder_normalization

# 配置参数
K = 3  # 用户数量
M_t = 3  # 天线数
N = 8  # RIS维度
power_total = 12.0
noise_variance = 1e-9  # 噪声方差
self_interference_cof = 0.0  # 自干扰系数
Lambda = 0.2  # 用于AMDEP约束 (1 - Lambda = 0.8)

# 距离参数（根据您的文档）
d_RU = 10.0  # RIS到用户距离
d_JR = 15.0  # Jammer到RIS距离
d_AR = 20.0  # Alice到RIS距离
d_RW = 12.0  # RIS到Willie距离

# 距离衰减模型（L(d) = d^(-2.5)）
def calculate_distance_loss(d):
    return d**(-2.5)

# 计算L3和L4（用于AMDEP）
L3 = calculate_distance_loss(d_RW) * calculate_distance_loss(d_JR)
L4 = calculate_distance_loss(d_RW) * calculate_distance_loss(d_AR)

# 联合优化环境
class JointOptimizationEnv(gym.Env):
    def __init__(self, K, M_t, N, power_total, noise_variance, self_interference_cof, L3, L4):
        super(JointOptimizationEnv, self).__init__()
        self.K = K
        self.M_t = M_t
        self.N = N
        self.power_total = power_total
        self.noise_variance = noise_variance
        self.self_interference_cof = self_interference_cof
        self.L3 = L3
        self.L4 = L4
        
        # 动作空间: [RIS phases (N), alpha_c (1), power_jammer (1)]
        # 全部归一化到 [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(N + 2,), dtype=np.float32)
        
        # 状态空间: [RIS phases (N), alpha_c (1), power_jammer (1)]
        # 全部归一化到 [0, 1]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(N + 2,), dtype=np.float32)
        
        # 生成随机信道（固定，假设在一个优化周期内信道不变）
        self.h_AR = generate_nakagami_channel(N, M_t, 2, 1)  # Alice-RIS信道
        self.h_JR = generate_nakagami_channel(N, 1, 2, 1)    # Jammer-RIS信道
        self.h_RUk = generate_nakagami_channel(N, K, 2, 1)   # RIS-User信道
        
        # 距离损失系数 (K, 1)
        self.distance_loss_cof_aru = np.array([calculate_distance_loss(d_RU)] * K).reshape(-1, 1)
        self.distance_loss_cof_jr = np.array([calculate_distance_loss(d_RU)] * K).reshape(-1, 1)
        
        # 初始状态
        self.ris_phase = np.random.uniform(0, 2 * np.pi, self.N)
        self.alpha_c = np.random.uniform(0.1, 0.9)
        self.power_jammer = np.random.uniform(0, self.power_total)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 初始化状态
        self.ris_phase = np.random.uniform(0, 2 * np.pi, self.N)
        self.alpha_c = np.random.uniform(0.1, 0.9)
        self.power_jammer = np.random.uniform(0, self.power_total)
        self.distance_loss_cof_aru = np.array([calculate_distance_loss(d_RU)] * self.K).reshape(-1, 1)
        self.distance_loss_cof_jru = np.array([calculate_distance_loss(d_RU)] * self.K).reshape(-1, 1)
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        # 将状态归一化并拼接
        obs_ris = self.ris_phase / (2 * np.pi)
        obs_alpha = np.array([self.alpha_c])
        obs_pj = np.array([self.power_jammer / self.power_total])
        return np.concatenate([obs_ris, obs_alpha, obs_pj]).astype(np.float32)
    
    def step(self, action):
        # 动作映射: [-1, 1] -> 实际范围
        # RIS相位增量 [-pi, pi]
        ris_inc = action[:self.N] * np.pi
        self.ris_phase = np.mod(self.ris_phase + ris_inc, 2 * np.pi)
        
        # alpha_c 映射到 [0.01, 0.99]
        self.alpha_c = np.clip((action[self.N] + 1) / 2, 0.01, 0.99)
        
        # power_jammer 映射到 [0, power_Jammer_max]
        self.power_jammer = np.clip((action[self.N + 1] + 1) / 2 * self.power_total, 0, self.power_total)
        power_alice_cur = self.power_total - self.power_jammer
        
        # 计算速率和AMDEP
        self.ris = generate_RIS_from_phase(self.ris_phase)
        H = self.h_RUk.T.conj() @ self.ris.T.conj() @ self.h_AR
        H = H.T  # (M_t, K)
        p_c, p_k = precoder_normalization(H)
        if not hasattr(self, "distance_loss_cof_aru"):
            self.distance_loss_cof_aru = np.array([calculate_distance_loss(d_RU)] * self.K).reshape(-1, 1)
        if not hasattr(self, "distance_loss_cof_jru"):
            self.distance_loss_cof_jru = np.array([calculate_distance_loss(d_RU)] * self.K).reshape(-1, 1)
        
        # 计算速率
        sinr_common, sinr_private = rate_uk(
            power_common_cof=self.alpha_c,
            power_private_cof=np.array([(1 - self.alpha_c) / self.K] * self.K),
            power_alice=power_alice_cur,
            power_jammer=self.power_jammer,
            channal_ruk=self.h_RUk,
            channal_ar=self.h_AR,
            channal_jr=self.h_JR,
            ris=self.ris,
            distance_loss_cof_aru=self.distance_loss_cof_aru,
            distance_loss_cof_jru=self.distance_loss_cof_jru,
            self_interference_cof=self.self_interference_cof,
            noise_variance=self.noise_variance,
            user_number=self.K,
            procoder_common=np.expand_dims(p_c, axis=1),
            procoder_private=p_k
        )
        
        # 计算总速率: R = log2(1+min(sinr_c)) + sum(log2(1+sinr_p))
        common_rate = np.log2(1 + np.min(sinr_common))
        private_rates = np.log2(1 + sinr_private)
        total_rate = common_rate + np.sum(private_rates)
        
        # 计算AMDEP
        amdep = AMDEP(self.K, self.power_jammer, power_alice_cur, self.alpha_c, self.L3, self.L4)
        
        # 奖励设计: 
        # 1. 基础奖励为总速率
        # 2. 如果不满足 AMDEP >= 0.8, 施加惩罚
        reward = total_rate
        if amdep < 0.8:
            reward -= 5.0 * (0.8 - amdep)  # 比例惩罚
            reward -= 2.0  # 阶梯惩罚
        reward = max(reward, 0.0)
        penalized_reward = reward
        
        # 统计功率系数与功率细项
        coeff_jammer = self.power_jammer / self.power_total
        coeff_alice_total = power_alice_cur / self.power_total
        power_common = self.alpha_c * power_alice_cur
        power_private_total = (1 - self.alpha_c) * power_alice_cur
        power_private_per_user = power_private_total / self.K
        coeff_common_total = power_common / self.power_total
        coeff_private_total = power_private_total / self.power_total
        coeff_private_per_user = power_private_per_user / self.power_total
            
        done = False
        truncated = False
        info = {
            'total_rate': total_rate,
            'amdep': amdep,
            'alpha_c': self.alpha_c,
            'power_jammer': self.power_jammer,
            'power_alice': power_alice_cur,
            'power_common': power_common,
            'power_private_total': power_private_total,
            'power_private_per_user': power_private_per_user,
            'coeff_common_total': coeff_common_total,
            'coeff_private_total': coeff_private_total,
            'coeff_private_per_user': coeff_private_per_user,
            'coeff_jammer': coeff_jammer,
            'coeff_alice_total': coeff_alice_total,
            'common_rate': common_rate,
            'private_rate_sum': np.sum(private_rates),
            'penalized_reward': penalized_reward
        }
        
        return self._get_obs(), reward, done, truncated, info

device_str = os.environ.get('SB3_DEVICE')
if device_str is None:
    device_str = 'cuda:2' if th.cuda.is_available() else 'cpu'

env = JointOptimizationEnv(K, M_t, N, power_total, noise_variance, self_interference_cof, L3, L4)
env = Monitor(env)
env = gym.wrappers.TimeLimit(env, max_episode_steps=200)

# 创建TD3模型（双Q Critic）
n_actions = env.action_space.shape[-1]
# 使用分维度 OU 噪声，并在训练过程中逐步减小强度
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
phase_sigma = 0.3
power_sigma = 0.05
min_sigma_ratio = 0.1
ou_mean = np.zeros(n_actions)
initial_sigma = np.concatenate([phase_sigma * np.ones(N), np.array([power_sigma, power_sigma])])
ou_theta = 0.15
action_noise = OrnsteinUhlenbeckActionNoise(mean=ou_mean, sigma=initial_sigma.copy(), theta=ou_theta)

class NoiseDecayCallback(BaseCallback):
    def __init__(self, action_noise, initial_sigma, min_sigma_ratio=0.1):
        super().__init__()
        self.action_noise = action_noise
        self.initial_sigma = initial_sigma
        self.min_sigma = initial_sigma * min_sigma_ratio

    def _on_step(self) -> bool:
        progress_remaining = self.model._current_progress_remaining
        sigma = self.min_sigma + (self.initial_sigma - self.min_sigma) * progress_remaining
        self.action_noise.sigma = sigma
        return True

# 设置网络结构和超参数
if device_str.startswith('cuda') and th.cuda.is_available():
    _idx = th.device(device_str).index
    _name = th.cuda.get_device_name(_idx if _idx is not None else 0)
    print(f"使用显卡: {device_str} - {_name} / Using GPU: {device_str} - {_name}")
else:
    print("使用CPU / Using CPU")

policy_kwargs = dict(net_arch=[400, 300], activation_fn=th.nn.ReLU)

def linear_decay_schedule(initial_lr, min_lr_ratio=0.1):
    min_lr = initial_lr * min_lr_ratio
    def lr_schedule(progress_remaining):
        return min_lr + (initial_lr - min_lr) * progress_remaining
    return lr_schedule

model = TD3(
    "MlpPolicy",  # 使用多层感知机策略与价值网络
    env,  # 训练环境
    action_noise=action_noise,  # 连续动作探索噪声
    verbose=0,  # 关闭训练过程的控制台日志
    learning_rate=linear_decay_schedule(5e-5, min_lr_ratio=0.05),  # 学习率随训练进度衰减以稳定后期比如 0.1 表示训练末期学习率降到初始的 10%。
    buffer_size=100000,  # 经验回放容量，越大越稳定但占用内存更多
    learning_starts=10,  # 采样若干步后再开始更新，避免冷启动不稳定
    batch_size=128,  # 每次更新的采样批量，影响梯度噪声大小
    tau=0.001,  # 目标网络软更新系数，越小越稳定
    gamma=0.99,  # 折扣因子，权衡短期与长期回报
    policy_kwargs=policy_kwargs,  # 策略与价值网络结构配置
    tensorboard_log="./ddpg_joint_tensorboard/",  # 训练指标输出路径
    device=device_str,  # 训练设备选择
    policy_delay=2,  # 延迟更新Actor，降低双Q训练抖动
    target_policy_noise=0.2,  # 目标策略加噪声，抑制过估计
    target_noise_clip=0.5,  # 裁剪目标噪声，避免动作过激
)

# 训练模型
print("\n开始联合优化训练 (RIS相位 + 功率分配)...")
print("TensorBoard logdir:", os.path.abspath("./ddpg_joint_tensorboard"))
callback = build_callbacks()
noise_callback = NoiseDecayCallback(action_noise, initial_sigma=initial_sigma, min_sigma_ratio=min_sigma_ratio)
callback = CallbackList(callback.callbacks + [noise_callback])
model.learn(total_timesteps=20000, callback=callback, tb_log_name="main_run", log_interval=1)
print("训练完成！")

# 测试优化结果
obs, info = env.reset()
best_rate = 0
best_info = None

print("\n评估最优结果...")
for _ in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if info['total_rate'] > best_rate and info['amdep'] >= 0.8:
        best_rate = info['total_rate']
        best_info = info

if best_info:
    print("\n优化结果 (满足AMDEP >= 0.8):")
    print(f"最优总速率: {best_info['total_rate']:.4f} bit/s/Hz")
    print(f"AMDEP值: {best_info['amdep']:.4f}")
    print(f"公共功率分配系数 alpha_c: {best_info['alpha_c']:.4f}")
    print(f"Jammer功率: {best_info['power_jammer']:.4f} W")
    print(f"公共功率: {best_info['power_common']:.4f} W")
    print(f"私有总功率: {best_info['power_private_total']:.4f} W")
    print(f"单用户私有功率: {best_info['power_private_per_user']:.4f} W")
    print(f"公共消息速率: {best_info['common_rate']:.4f}")
    print(f"私有消息总速率: {best_info['private_rate_sum']:.4f}")
else:
    print("\n未找到满足AMDEP >= 0.8的解。请尝试增加训练步数或调整奖励函数。")

# 保存模型
model.save("ddpg_joint_optimization")

print("\n优化完成！TensorBoard 日志保存在 ./ddpg_joint_tensorboard/")
print("你可以运行 'tensorboard --logdir ./ddpg_joint_tensorboard/' 来查看训练曲线。")
