import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
from torch.utils.tensorboard import SummaryWriter


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0, image_every=200):
        super(TensorboardCallback, self).__init__(verbose)
        self.rew_window = deque(maxlen=1000)
        self.amdep_satisfy = deque(maxlen=1000)
        self.writer = None
        self._img_every = image_every

    def _on_training_start(self) -> None:
        try:
            tb_dir = self.logger.get_dir()
            if tb_dir is not None:
                custom_dir = os.path.join(tb_dir, "custom")
                os.makedirs(custom_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir=custom_dir)
        except Exception:
            self.writer = None

    def _on_training_end(self) -> None:
        if self.writer is not None:
            try:
                self.writer.flush()
                self.writer.close()
            except Exception:
                pass

    def _record_scalar_metrics(self, info, step, actions):
        self.writer.add_scalar('env/total_rate', float(info['total_rate']), step)
        self.writer.add_scalar('env/amdep', float(info['amdep']), step)
        self.writer.add_scalar('env/reward_penalized', float(info['penalized_reward']), step)
        self.writer.add_scalar('env/power_jammer', float(info['power_jammer']), step)
        self.writer.add_scalar('env/power_alice', float(info['power_alice']), step)
        self.writer.add_scalar('env/power_common', float(info['power_common']), step)
        self.writer.add_scalar('env/power_private_total', float(info['power_private_total']), step)
        self.writer.add_scalar('env/power_private_per_user', float(info['power_private_per_user']), step)
        if actions is not None:
            try:
                actions_np = actions if isinstance(actions, np.ndarray) else actions.cpu().numpy()
                self.writer.add_histogram('actions/alpha', actions_np[..., -2], step)
                self.writer.add_histogram('actions/pj', actions_np[..., -1], step)
            except Exception:
                pass

    def _record_images(self, info, step):
        if (step % self._img_every) != 0:
            return
        try:
            coeffs = [
                float(info['coeff_common_total']),
                float(info['coeff_private_total']),
                float(info['coeff_jammer']),
            ]
            labels = ['common', 'private', 'jammer']
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(labels, coeffs, color=['tab:blue', 'tab:orange', 'tab:red'])
            ax.set_ylim(0.0, 1.0)
            ax.set_title('Power Coefficients')
            self.writer.add_figure('images/power_coefficients', fig, step, close=True)
        except Exception:
            pass

    def _record_bilingual_metrics(self, info, step):
        self.writer.add_scalar('环境/总速率 TotalRate', float(info['total_rate']), step)
        self.writer.add_scalar('环境/AMDEP', float(info['amdep']), step)
        self.writer.add_scalar('环境/罚后奖励 PenalizedReward', float(info['penalized_reward']), step)
        self.writer.add_scalar('功率/干扰器功率 JammerPower', float(info['power_jammer']), step)
        self.writer.add_scalar('功率/Alice总功率 AlicePower', float(info['power_alice']), step)
        self.writer.add_scalar('功率/公共功率 CommonPower', float(info['power_common']), step)
        self.writer.add_scalar('功率/私有总功率 PrivatePowerTotal', float(info['power_private_total']), step)
        self.writer.add_scalar('功率/单用户私有功率 PrivatePowerPerUser', float(info['power_private_per_user']), step)
        self.writer.add_scalar('诊断/罚后奖励均值 PenalizedRewardAvg', sum(self.rew_window) / len(self.rew_window), step)
        self.writer.add_scalar('诊断/AMDEP满足率 AMDEPSatisfyRate', sum(self.amdep_satisfy) / len(self.amdep_satisfy), step)
        if 'train/actor_loss' in self.model.logger.name_to_value:
            self.writer.add_scalar('训练/演员损失 Actor Loss', self.model.logger.name_to_value['train/actor_loss'], step)
        if 'train/critic_loss' in self.model.logger.name_to_value:
            self.writer.add_scalar('训练/评论家损失 Critic Loss', self.model.logger.name_to_value['train/critic_loss'], step)

    def _on_step(self) -> bool:
        if 'total_rate' in self.locals['infos'][0]:
            info = self.locals['infos'][0]
            self.logger.record('env/total_rate', info['total_rate'])
            self.logger.record('env/amdep', info['amdep'])
            self.logger.record('env/alpha_c', info['alpha_c'])
            self.logger.record('env/power_jammer', info['power_jammer'])
            self.logger.record('env/power_alice', info['power_alice'])
            self.logger.record('env/power_common', info['power_common'])
            self.logger.record('env/power_private_total', info['power_private_total'])
            self.logger.record('env/power_private_per_user', info['power_private_per_user'])
            self.logger.record('env/coeff_common_total', info['coeff_common_total'])
            self.logger.record('env/coeff_private_total', info['coeff_private_total'])
            self.logger.record('env/coeff_private_per_user', info['coeff_private_per_user'])
            self.logger.record('env/coeff_jammer', info['coeff_jammer'])
            self.logger.record('env/coeff_alice_total', info['coeff_alice_total'])
            self.logger.record('env/common_rate', info['common_rate'])
            self.logger.record('env/private_rate_sum', info['private_rate_sum'])
            self.logger.record('env/reward_penalized', info['penalized_reward'])
            self.rew_window.append(float(info['penalized_reward']))
            self.amdep_satisfy.append(1.0 if float(info['amdep']) >= 0.8 else 0.0)
            if len(self.rew_window) > 0:
                self.logger.record('diagnostics/reward_penalized_avg', sum(self.rew_window) / len(self.rew_window))
                self.logger.record('diagnostics/amdep_satisfy_rate', sum(self.amdep_satisfy) / len(self.amdep_satisfy))
            actions = self.locals.get('actions', None)
            if actions is not None:
                try:
                    actions_np = actions if isinstance(actions, np.ndarray) else actions.cpu().numpy()
                    alpha_part = actions_np[..., -2]
                    pj_part = actions_np[..., -1]
                    self.logger.record('actions/alpha_action_mean', float(np.mean(alpha_part)))
                    self.logger.record('actions/alpha_action_std', float(np.std(alpha_part)))
                    self.logger.record('actions/pj_action_mean', float(np.mean(pj_part)))
                    self.logger.record('actions/pj_action_std', float(np.std(pj_part)))
                except Exception:
                    pass
            if self.writer is not None:
                try:
                    step = int(self.num_timesteps)
                    self._record_scalar_metrics(info, step, actions)
                    self._record_images(info, step)
                except Exception:
                    pass
            try:
                step = int(self.num_timesteps)
                self._record_bilingual_metrics(info, step)
            except Exception:
                pass
            if (self.n_calls % 100) == 0:
                print(
                    f"step={self.num_timesteps} "
                    f"R={info['total_rate']:.4f} "
                    f"R_pen={info['penalized_reward']:.4f} "
                    f"AMDEP={info['amdep']:.4f} "
                    f"alpha_c={info['alpha_c']:.3f} "
                    f"P_A={info['power_alice']:.3f} "
                    f"P_J={info['power_jammer']:.3f} "
                    f"P_c={info['power_common']:.3f} "
                    f"P_p_tot={info['power_private_total']:.3f} "
                    f"P_p/u={info['power_private_per_user']:.3f} "
                    f"coeff(c, p, j)=({info['coeff_common_total']:.3f}, {info['coeff_private_total']:.3f}, {info['coeff_jammer']:.3f})"
                )
        return True


def build_callbacks():
    tensorboard_callback = TensorboardCallback()
    return CallbackList([tensorboard_callback])
