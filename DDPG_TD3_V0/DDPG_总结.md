# DDPG 代码思路与优化思路总结

## 代码思路（DDPG流程）

- **环境建模**：实现 `JointOptimizationEnv`，把联合优化问题包装为 Gym 环境，定义动作空间与状态空间，并在 `step` 中完成动作映射、信道更新、速率计算与奖励计算。[main.py](file:///home/lh/projects/github/DDPGOPT/main.py#L213-L356)
- **动作设计**：动作包含 `RIS 相位增量 (N)`、`公共功率系数 alpha_c` 与 `Jammer 功率`，统一归一化到 `[-1, 1]`，再映射回实际范围以保证动作合法。[main.py](file:///home/lh/projects/github/DDPGOPT/main.py#L226-L277)
- **状态设计**：状态由 `RIS 相位`、`alpha_c`、`power_jammer` 构成，全部归一化到 `[0, 1]`，便于策略网络学习稳定表征。[main.py](file:///home/lh/projects/github/DDPGOPT/main.py#L230-L265)
- **速率与约束评估**：每一步根据当前 RIS、功率分配与预编码计算公共/私有 SINR 与总速率，同时计算 AMDEP 作为安全约束指标。[main.py](file:///home/lh/projects/github/DDPGOPT/main.py#L279-L314)
- **奖励设计**：以总速率为基础奖励；若 AMDEP 不满足阈值（0.8），施加比例惩罚与阶梯惩罚，形成软约束引导策略学习可行解。[main.py](file:///home/lh/projects/github/DDPGOPT/main.py#L316-L323)
- **训练管线**：使用 Stable-Baselines3 的 DDPG，OU 噪声探索，设置网络结构与超参数，并通过回调记录 TensorBoard 指标与周期性评估，保存最优模型。[main.py](file:///home/lh/projects/github/DDPGOPT/main.py#L370-L435)
- **后验评估**：训练结束后用策略评估最优可行解（满足 AMDEP 阈值），并输出对应的速率与功率分配指标。[main.py](file:///home/lh/projects/github/DDPGOPT/main.py#L438-L465)

## 优化思路（问题建模与求解逻辑）

- **联合优化目标**：最大化系统总速率（公共速率 + 私有速率和），同时满足 AMDEP 安全约束与功率约束。通过强化学习把组合优化问题转为策略搜索。[main.py](file:///home/lh/projects/github/DDPGOPT/main.py#L289-L323)
- **变量耦合处理**：RIS 相位、公共功率分配、Jammer 功率共同决定信道增益与速率，使用连续动作空间直接输出联合决策，避免显式求解非凸优化。[main.py](file:///home/lh/projects/github/DDPGOPT/main.py#L266-L306)
- **软约束惩罚**：对 AMDEP 约束采用“比例惩罚 + 阶梯惩罚”的形式，使策略在优化速率时持续被约束拉回可行域。[main.py](file:///home/lh/projects/github/DDPGOPT/main.py#L316-L323)
- **功率分配逻辑**：总功率固定，Jammer 功率由动作决定，Alice 功率由剩余功率决定；公共/私有功率按 `alpha_c` 与均分规则计算，保证功率守恒。[main.py](file:///home/lh/projects/github/DDPGOPT/main.py#L275-L333)
- **预编码与速率计算**：从等效信道构造共形与零迫预编码，计算各用户的公共/私有 SINR，从而得到速率作为奖励核心。[main.py](file:///home/lh/projects/github/DDPGOPT/main.py#L280-L311)
- **评估与保存策略**：定期评估平均回报，并保存最佳模型，用于后续验证可行解与性能上界探索。[main.py](file:///home/lh/projects/github/DDPGOPT/main.py#L406-L435)

## DDPG 原理（核心机制）

- **确定性策略梯度**：策略网络直接输出连续动作，通过最大化动作价值函数的梯度更新策略，适合连续控制任务。
- **Actor–Critic 架构**：Actor 产生动作，Critic 估计 Q(s, a)；Actor 用 Critic 的梯度信号优化策略，Critic 用 TD 误差训练。
- **经验回放**：将交互样本存入回放池，随机采样打破相关性，提高样本效率与训练稳定性。
- **目标网络**：为 Actor 与 Critic 维护慢更新的目标网络，使用软更新降低训练抖动与过估计风险。
- **探索噪声**：在动作输出上叠加 OU 噪声实现连续空间探索，避免策略过早收敛到局部最优。

## 双Q网络的数据流程（以TD3为例）

- **输入状态**：环境在时刻 t 输出观测 s_t（本项目为 RIS 相位、alpha_c、Jammer 功率的归一化状态）。[main.py](file:///home/lh/projects/github/DDPGOPT/main.py#L230-L265)
- **Actor 产生动作**：Actor 根据 s_t 输出确定性动作 a_t，再叠加探索噪声得到执行动作 a_t^noise，用于与环境交互。
- **环境反馈**：环境接收 a_t^noise，执行一步得到 (s_{t+1}, r_t, done, info)，其中 r_t 为基于速率与 AMDEP 约束的奖励。
- **经验回放**：将 (s_t, a_t^noise, r_t, s_{t+1}, done) 存入回放池，后续从中随机采样小批量用于训练。
- **双Q评估**：Critic 包含两个独立 Q 网络 Q1、Q2，对同一 (s, a) 计算两路 Q 值以抑制过估计。
- **目标值计算**：使用目标 Actor 产生 a_{t+1}^target，并在目标动作上加噪声与裁剪，分别送入目标 Q1、Q2，取更小的 Q 作为目标 y_t。
- **Critic 更新**：最小化 Q1(s_t, a_t) 与 Q2(s_t, a_t) 对目标 y_t 的均方误差，稳定价值估计。
- **Actor 延迟更新**：每隔若干步（policy_delay）才更新 Actor，使策略更新依赖更稳定的 Critic。
- **软更新目标网络**：使用 τ 进行软更新，使目标网络缓慢跟随主网络，降低训练抖动。

## 与环境交互步骤 vs TimeLimit 包装的区别

- **与环境交互的每一步**：是指策略产生动作后调用 `env.step(action)`，环境执行物理/数学更新并返回下一状态、奖励与终止标志，这是训练与学习的核心数据来源。
- **TimeLimit 包装**：`env = gym.wrappers.TimeLimit(env, max_episode_steps=200)` 只是给环境加上“最大回合步数”的上限，当步数达到上限时强制截断回合，保证训练回合长度可控。[main.py](file:///home/lh/projects/github/DDPGOPT/main.py#L186-L188)
- **本质差异**：交互步骤决定学习样本与策略更新；TimeLimit 只改变回合结束条件，不参与状态转移与奖励计算。
