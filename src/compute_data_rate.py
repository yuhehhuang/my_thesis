import numpy as np

def compute_data_rate(sat_pos_km, user_pos_km,
                      freq_hz=14.5e9, EIRP_dBW=22.5, bandwidth_Hz=10e6):
    """
    根據衛星與使用者的三維距離，估算傳輸速率（Mbps）
    sat_pos_km, user_pos_km: [x, y, z] in kilometers
    """
    # 計算距離（m）
    d_km = np.linalg.norm(np.array(sat_pos_km) - np.array(user_pos_km))
    d_m = d_km * 1000

    # 自由空間損耗（FSPL）
    c = 3e8  # 光速 m/s
    PL_dB = 20 * np.log10(d_km) + 20 * np.log10(freq_hz) - 147.55

    # 接收功率 Pr = EIRP - PL
    P_r_dBW = EIRP_dBW - PL_dB
    P_r_dBm = P_r_dBW + 30
    # 熵噪聲功率（dBm）:Reference:https://zhuanlan.zhihu.com/p/24332783
    noise_power_dBm = -162 + 10 * np.log10(bandwidth_Hz) 
    # SNR (linear)
    SNR_linear = 10 ** ((P_r_dBm - noise_power_dBm) / 10)
    # Shannon capacity
    capacity_bps = bandwidth_Hz * np.log2(1 + SNR_linear)
    capacity_mbps = capacity_bps / 1e6

    return capacity_mbps 

# def lookup_se_from_cn(cn_dB):
#     if cn_dB < 0:
#         return 0.0
#     elif cn_dB < 3:
#         return 0.5
#     elif cn_dB < 6:
#         return 1.0
#     elif cn_dB < 10:
#         return 2.0
#     elif cn_dB < 13:
#         return 3.0
#     elif cn_dB < 16:
#         return 4.0
#     elif cn_dB < 19.57:
#         return 5.0
#     else:
#         return 5.9

# def compute_cn_and_spectral_efficiency(
#     sat_pos_km, user_pos_km,
#     freq_hz=14.5e9,
#     EIRP_dBW=22.5,
#     bandwidth_Hz=10e6,
#     receiver_gain_dBi=0,
#     L_atm_dB=0.5,
#     T_K=290
# ):
#     # 計算距離（m）
#     d_km = np.linalg.norm(np.array(sat_pos_km) - np.array(user_pos_km))
#     d_m = d_km * 1000

#     # FSPL (dB)
#     PL_dB = 20 * np.log10(d_m) + 20 * np.log10(freq_hz) - 147.55

#     # EIRP per Hz
#     EIRP_dBW_per_Hz = EIRP_dBW - 10 * np.log10(bandwidth_Hz)

#     # 接收功率密度 Pr (dBW/Hz)
#     Pr_dBW_per_Hz = EIRP_dBW_per_Hz - PL_dB - L_atm_dB + receiver_gain_dBi

#     # 熱噪聲密度 N (dBW/Hz)
#     k_B = 1.38e-23  # J/K
#     noise_power_W_per_Hz = k_B * T_K
#     noise_dBW_per_Hz = 10 * np.log10(noise_power_W_per_Hz)

#     # C/N in dB
#     CN_dB = Pr_dBW_per_Hz - noise_dBW_per_Hz

#     # 對應 spectral efficiency
#     SE_bps_per_Hz = lookup_se_from_cn(CN_dB)

#     return CN_dB, SE_bps_per_Hz