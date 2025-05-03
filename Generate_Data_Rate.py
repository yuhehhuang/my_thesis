import pandas as pd
import pickle
from skyfield.api import load, wgs84
from compute_data_rate import compute_data_rate  # 你必須先有這個函式
import ast

# === 載入資料 ===
user_df = pd.read_csv("user_info_with_Ks.csv")  # 需包含 user_id, lat, lon
access_df = pd.read_csv("access_matrix.csv")
access_df["visible_sats"] = access_df["visible_sats"].apply(ast.literal_eval)
access_matrix = access_df.to_dict("records")

with open("satellite_positions.pkl", "rb") as f:
    satellite_positions = pickle.load(f)

# === Skyfield 時間軸 ===
ts = load.timescale()
START_TIME = ts.utc(2025, 4, 21, 0, 0, 0)
SLOTS = len(access_matrix)
times = [ts.utc(2025, 4, 21, 0, i, 0) for i in range(SLOTS)]

# === 準備使用者位置 ===
user_locations = {}
for _, row in user_df.iterrows():
    user_id = row["user_id"]
    lat = row["lat"]
    lon = row["lon"]
    user_locations[user_id] = [wgs84.latlon(lat, lon).at(t).position.km for t in times]

# === 建立 data_rate_dict ===
data_rate_dict = {}

for t_idx in range(SLOTS):
    visible_sats = access_matrix[t_idx]["visible_sats"]
    for user_id in user_locations:
        user_pos = user_locations[user_id][t_idx]
        for sat in visible_sats:
            key = (sat, t_idx)
            if key not in satellite_positions:
                continue
            sat_pos = satellite_positions[key]
            rate = compute_data_rate(sat_pos, user_pos)
            data_rate_dict[(sat, user_id, t_idx)] = rate

# === 輸出 ===
with open("data_rate_dict.pkl", "wb") as f:
    pickle.dump(data_rate_dict, f)

print("Saved data_rate_dict.pkl with {} entries.".format(len(data_rate_dict)))
