import math
from typing import Tuple
import numpy as np


def generate_rayleigh_channel(N: int, M: int, Omega: float=1) -> np.ndarray:
    """
    生成瑞利衰落信道矩阵（幅度服从瑞利分布，相位均匀分布）
    输入参数：
        N      : 矩阵行数（天线数/用户数）
        M      : 矩阵列数（时间样本/子载波数）
        Omega  : 平均功率（E[|h|²] = Omega）
    输出：
        H_rayleigh : N x M 的复信道矩阵（复数形式包含幅度和相位）
    """
    # 生成独立的高斯实部和虚部（均值为0，方差为Omega/2）
    real_part = np.random.normal(0, np.sqrt(Omega/2), (N, M))
    imag_part = np.random.normal(0, np.sqrt(Omega/2), (N, M))
    
    # 合成复信道矩阵
    return real_part + 1j * imag_part



def generate_nakagami_channel(N: int, M: int, m: float, Omega: float, seed: int=None) -> np.ndarray:
    """
    生成Nakagami-m衰落信道矩阵（幅度服从Nakagami-m分布，相位均匀分布）
    输入参数：
        N      : 矩阵行数（天线数/用户数）
        M      : 矩阵列数（时间样本/子载波数）
        m      : Nakagami形状参数（m >= 0.5）
        Omega  : 平均功率（E[|h|²] = Omega）
    输出：
        H_nakagami : N x M 的复信道矩阵（复数形式包含幅度和相位）
    """
    if m < 0.5:
        raise ValueError("Nakagami形状参数m必须 >= 0.5")
    rng = np.random.default_rng(seed)
    # 生成伽马分布随机变量（形状参数=m，尺度参数=Omega/m）
    gamma_samples = rng.gamma(shape=m, scale=Omega/m, size=(N, M))
    
    # 幅度 = sqrt(伽马变量)
    amplitude = np.sqrt(gamma_samples)
    
    # 生成均匀相位（0到2π）
    phase = rng.uniform(0, 2*np.pi, (N, M))
    
    # 合成复信道矩阵
    return amplitude * np.exp(1j * phase)


def generate_RIS_random(N, seed=None) -> np.ndarray:
    """
    生成 NxN 随机的 RIS 对角矩阵
    :param N: 矩阵维度
    :return: NxN 的对角矩阵，对角线元素为幅度为1的复数
    """
    # 生成均匀分布的相位
    np.random.seed(seed)
    phase = 2 * np.pi * np.random.rand(N, 1)
    
    # 生成复数，幅度为1
    h = 1*np.exp(1j * phase)
    
    # 创建对角矩阵
    ris = np.diag(np.squeeze(h))
    return ris


def precoder_normalization(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算预编码向量和矩阵
    :param H: Alice-User 的信道增益矩阵 [N_t, K] (天线数 x 用户数)
    :return: p_c (共形预编码向量), p_k (归一化的零迫预编码矩阵)
    """
    # 共形预编码计算
    precoding_c = np.sum(H, axis=1)  # 按列求和
    p_c = precoding_c / np.linalg.norm(precoding_c)  # 归一化
    
    # 计算零迫预编码矩阵
    procoding_uk = np.linalg.pinv(H.T @ H) @ H.T  # H/(H'H) 的等价实现
    
    # 归一化每一列
    for k in range(procoding_uk.shape[1]):
        norm_k = np.linalg.norm(procoding_uk[:, k])
        if norm_k != 0:
            procoding_uk[:, k] /= norm_k
    
    p_k = procoding_uk
    return p_c, p_k

def rate_uk(
    power_common_cof: float,  
    power_private_cof: np.ndarray, 
    power_alice: float,  
    power_jammer: float,  
    channal_ruk: np.ndarray, 
    channal_ar: np.ndarray,  
    channal_jr: np.ndarray, 
    ris: np.ndarray, 
    distance_loss_cof_aru: np.ndarray,  
    distance_loss_cof_jru: np.ndarray,  
    self_interference_cof: float, 
    noise_variance: float, 
    user_number: int,  
    procoder_common: np.ndarray, 
    procoder_private: np.ndarray 
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    计算公共消息速率和私有消息速率。
    :param power_common_cof: 公共消息功率分配系数
    :param power_private_cof: 私有消息功率分配系数 [K,1]
    :param power_alice: Alice的功率
    :param power_jammer: Jammer的功率
    :param channal_ruk: RIS-Uk的信道增益 [N,K]
    :param channal_ar: Alice-RIS的信道增益 [N,M]
    :param channal_jr: Jammer-RIS的信道增益 [N,1]
    :param ris: RIS反射单元 [N,N]
    :param distance_loss_cof_aru: Alice-RIS-Uk的路径损失 [K,1]
    :param distance_loss_cof_jru: Jammer-RIS-Uk的路径损失 [K,1]
    :param self_interference_cof: 自干扰系数
    :param noise_variance: 噪声方差
    :param user_number: 用户数量 K
    :param procoder_common: 公共消息预编码 [M,1]
    :param procoder_private: 私有消息预编码 [M,K]
    :return: 公共消息速率和私有消息速率 [K,1]
    """
    # 运行时检查，确保矩阵参数是 np.ndarray 类型
    for param in [channal_ruk, channal_ar, channal_jr, ris, 
                  distance_loss_cof_aru, distance_loss_cof_jru, 
                  procoder_common, procoder_private]:
        assert isinstance(param, np.ndarray), f"{param} must be a numpy ndarray"
    
    rate_common_temp = np.zeros(user_number)
    
    for k in range(user_number):
        rate_common_up = power_common_cof * power_alice * \
            np.abs(channal_ruk[:, k].conj().T @ ris @ channal_ar @ procoder_common) ** 2 * distance_loss_cof_aru[k]
        temp = 0
        for j in range(user_number):
            temp += power_private_cof[j] * power_alice * distance_loss_cof_aru[k] * \
                   np.abs(channal_ruk[:, k].conj().T @ ris @ channal_ar @ procoder_private[:, j]) ** 2
        
        rate_common_down = temp + self_interference_cof * power_jammer * distance_loss_cof_jru[k] * \
                           np.abs(channal_ruk[:, k].conj().T @ ris @ channal_jr) ** 2 + noise_variance
        
        rate_common_temp[k] = rate_common_up / rate_common_down
    
    rate_private_temp = np.zeros(user_number)
    
    for k in range(user_number):
        rate_private_up = power_private_cof[k] * power_alice * \
            np.abs(channal_ruk[:, k].conj().T @ ris @ channal_ar @ procoder_private[:, k]) * distance_loss_cof_aru[k]
        temp = 0
        for j in range(user_number):
            if j == k:
                continue
            temp += power_private_cof[j] * power_alice * distance_loss_cof_aru[k] * \
                   np.abs(channal_ruk[:, k].conj().T @ ris @ channal_ar @ procoder_private[:, j]) ** 2
        
        rate_private_down = temp + self_interference_cof * power_jammer * distance_loss_cof_jru[k] * \
                            np.abs(channal_ruk[:, k].conj().T @ ris @ channal_jr) ** 2 + noise_variance
        
        rate_private_temp[k] = rate_private_up / rate_private_down
    
    return rate_common_temp, rate_private_temp

def AMDEP(user_number,
          power_Jammer_max,
          power_Alice,
          power_common_cof,
          L3,
          L4)->float:
    '''
    计算AMDEP
    :param user_number: 用户数量
    :param power_Jammer_max: Jammer的最大功率
    :param power_Alice: Alice的功率
    :param power_common_cof: 公共消息功率分配系数
    :param L3: Jammer-RIS-Willie的路径损失
    :param L4: Alice-RIS-Willie的路径损失
    :return: AMDEP
    '''
    n = 100
    mu = user_number * power_Jammer_max * L3
    nu = (1-power_common_cof) * power_Alice * L4
    sumf = 0
    for i in range(1, n+1):
        omega_i = math.pi / n
        t_i = math.cos(math.pi * (2*i - 1) / (2 * n))
        F_t_i = math.sqrt(1 - t_i**2) * (((1 + t_i) * (mu + nu)) / ((1 + t_i) * mu + 2 * nu)) ** user_number
        sumf += omega_i * F_t_i
    xi_a_star = ((mu / (mu + nu)) ** user_number) * 0.5 * sumf

    return xi_a_star



def generate_RIS_from_phase(phase_array: np.ndarray) -> np.ndarray:
    """
    根据输入的相位数组生成NxN的RIS对角矩阵（幅度恒为1）
    
    :param phase_array: 一维相位数组（维度为N），单位弧度
    :return: NxN的RIS对角矩阵，对角线元素为复数exp(j*phase)
    """

    # 生成复数（幅度为1）
    diag_elements = np.exp(1j * phase_array)
    
    # 构造对角矩阵
    ris_matrix = np.diag(diag_elements)
    return ris_matrix

def procoder(antenna_number,
             user_number,
             amplitude:np.ndarray,
             phase:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    生成预编码器
    :param antenna_number: 天线数
    :param user_number: 用户数
    :param amplitude: 预编码器幅度
    :param phase: 预编码器相位
    :return: 预编码器
    '''
    assert len(amplitude)== len(phase), "预编码器幅度和相位长度不匹配"
    assert antenna_number*(user_number+1) == len(amplitude), "预编码器幅度长度不匹配"
    common_amplitude = amplitude[:antenna_number]
    common_amplitude = common_amplitude/np.sum(common_amplitude)
    common_phase = phase[:antenna_number]
    common_procoder = common_amplitude * np.exp(1j * common_phase)
    
    private_amplitude = amplitude[antenna_number:]
    private_phase = phase[antenna_number:]
    # 转换为列优先矩阵（Fortran顺序）
    private_amplitude = private_amplitude.reshape((antenna_number, user_number), order='F')
    private_phase = private_phase.reshape((antenna_number, user_number), order='F')
    
    # L1归一化每列幅度（保证Σ|amplitude|=1）[3,6](@ref)
    col_sums = private_amplitude.sum(axis=0, keepdims=True)
    private_amplitude = private_amplitude / col_sums
    
    # 生成复数矩阵（欧拉公式法）[1,2](@ref)
    private_procoder = private_amplitude * np.exp(1j * private_phase)
    
    return common_procoder, private_procoder


if __name__ == "__main__":
    for i in range(2):
        h1 = generate_nakagami_channel(3, 3, 2, 1)
        print(h1)
    h1 = generate_nakagami_channel(3, 3, 2, 1, seed=0)
    print(h1)