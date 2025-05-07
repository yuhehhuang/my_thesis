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
    PL_dB = 20 * np.log10(d_m) + 20 * np.log10(freq_hz) - 147.55

    # 接收功率 Pr = EIRP - PL
    P_r_dBW = EIRP_dBW - PL_dB
    P_r_dBm = P_r_dBW + 30
    # 熵噪聲功率（dBm）
    noise_power_dBm = -174 + 10 * np.log10(bandwidth_Hz)
    # SNR (linear)
    SNR_linear = 10 ** ((P_r_dBm - noise_power_dBm) / 10)
    # Shannon capacity
    capacity_bps = bandwidth_Hz * np.log2(1 + SNR_linear)
    capacity_mbps = capacity_bps / 1e6

    return capacity_mbps 