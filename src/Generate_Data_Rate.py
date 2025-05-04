import pandas as pd
import pickle
from skyfield.api import load, wgs84
from numpy.linalg import norm
import ast

def compute_data_rate(sat_pos, user_pos):
    distance = norm(sat_pos - user_pos)
    if distance > 2000:
        return 0
    return max(10, 3000 / (distance + 1))

# === 載入資料 ===
user_df = pd.read_csv("data/user_info_with_Ks.csv")
access_df = pd.read_csv("data/access_matrix.csv")
access_df["visible_sats"] = access_df["visible_sats"].apply(ast.literal_eval)
access_matrix = access_df.to_dict("records")

with open("data/satellite_positions.pkl", "rb") as f:
    satellite_positions = pickle.load(f)

# === Skyfield 時間軸 ===
ts = load.timescale()
SLOTS = len(access_matrix)
times = [ts.utc(2025, 4, 21, 0, i, 0) for i in range(SLOTS)]

# === 使用者位置
user_locations = {}
for _, row in user_df.iterrows():
    user_id = row["user_id"]
    lat = row["lat"]
    lon = row["lon"]
    user_locations[user_id] = [wgs84.latlon(lat, lon).at(t).position.km for t in times]

# === 建立兩種 dict
data_rate_dict_user = {}  # (sat, user_id, t) → float (for Yens)
data_rate_dict_avg = {}   # (t, sat) → avg float   (for DP)

for t_idx in range(SLOTS):
    visible_sats = access_matrix[t_idx]["visible_sats"]
    for sat in visible_sats:
        key_sat_time = (sat, t_idx)
        if key_sat_time not in satellite_positions:
            continue
        sat_pos = satellite_positions[key_sat_time]

        rates = []
        for user_id in user_locations:
            user_pos = user_locations[user_id][t_idx]
            rate = compute_data_rate(sat_pos, user_pos)
            data_rate_dict_user[(sat, user_id, t_idx)] = rate  # 個別記錄
            rates.append(rate)

        avg_rate = sum(rates) / len(rates) if rates else 0.0
        data_rate_dict_avg[(t_idx, sat)] = avg_rate

# === 存檔
with open("data/data_rate_dict_user.pkl", "wb") as f:
    pickle.dump(data_rate_dict_user, f)
with open("data/data_rate_dict_avg.pkl", "wb") as f:
    pickle.dump(data_rate_dict_avg, f)

print(f"✅ user version saved: {len(data_rate_dict_user)} entries")
print(f"✅ avg  version saved: {len(data_rate_dict_avg)} entries")