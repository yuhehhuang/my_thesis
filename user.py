import pandas as pd
import random
from geopy.distance import distance
from geopy.point import Point

T = 96
NUM_USERS = 500
CENTER_LAT = 40.0386
CENTER_LON = -75.5966

center = Point(CENTER_LAT, CENTER_LON)
users = []

for user_id in range(NUM_USERS):
    # --- 使用時間設定 ---
    duration = random.randint(5, 25)
    t_start = random.randint(0, T - duration)
    t_end = t_start + duration

    # --- handover 限制 ---
    row = {
        "user_id": user_id,
        "t_start": t_start,
        "t_end": t_end
    }
    for K in range(6):
        row[f"K{K}"] = K + (duration // 3) + 5

    # --- 隨機經緯度產生（半徑 0.8km 內） ---
    angle = random.uniform(0, 360)
    radius = random.uniform(0, 0.8)
    point = distance(kilometers=radius).destination(center, bearing=angle)
    row["lat"] = point.latitude
    row["lon"] = point.longitude

    users.append(row)

df = pd.DataFrame(users)
df.to_csv("user_info_with_Ks.csv", index=False)
print("✅ user_info_with_Ks.csv 產生成功，包含 lat/lon")
