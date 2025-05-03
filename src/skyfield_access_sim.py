from skyfield.api import load, Topos, utc
from datetime import datetime, timedelta
from geopy.distance import distance
from geopy.point import Point
import pandas as pd
import pickle
# === 參數設定 ===
TLE_URL = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle'
USER_LAT = 40.0386
USER_LON = -75.5966 
START_TIME_UTC = datetime(2025, 4, 21, 0, 0, 0, tzinfo=utc)
SLOTS = 96
SLOT_INTERVAL_MIN = 1
MAX_SAT = 7204
OUTPUT_CSV = "access_matrix.csv"

# === 載入 TLE ===
print("Loading TLE data...")

#satellites	回傳的是一個 Python list，每個元素都是 EarthSatellite 物件 (.name, .model,.at(t),.altaz());
satellites = load.tle_file(TLE_URL)
# === 擷取所有衛星名稱（不論可見與否）===
all_satellite_names = [sat.name for sat in satellites]
print(f" Starlink 衛星總數：{len(satellites)} 顆")

ts = load.timescale()
observer = Topos(latitude_degrees=USER_LAT, longitude_degrees=USER_LON)
t0 = ts.utc(START_TIME_UTC)



# ====== 模擬時間序列 ======
times = [ts.utc(START_TIME_UTC + timedelta(minutes=i)) for i in range(SLOTS)]

# ====== 模擬可視性 ======
access_matrix = []
satellite_positions = {}
for t_idx, t in enumerate(times):
    visible_sats = []
    for sat in satellites:
        difference = sat - observer
        topocentric = difference.at(t)
        #把 topocentric 轉換成「仰角（altitude）、方位角（azimuth）、距離」
        #仰角介於-90 到 90 度之間 ,我們設定 45 度為可見衛星的門檻
        alt, az, dist = topocentric.altaz()
        if alt.degrees >= 45 : 
        #根據A Multi-slot Load Balancing Scheme for LEO Satellite Communication Handover Target Selection Fig4.回推是45度
            visible_sats.append(sat.name)
        # 儲存每個 (sat.name, t_idx) 的位置
        sat_xyz = sat.at(t).position.km.tolist()
        satellite_positions[(sat.name, t_idx)] = sat_xyz
    access_matrix.append({
        "time_slot": t_idx,
        "visible_sats": visible_sats
    })
    print(f"Slot {t_idx}: {len(visible_sats)} sats visible")

# ====== 輸出 CSV ======
df = pd.DataFrame(access_matrix)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved access matrix to {OUTPUT_CSV}")

# ====== 輸出 satellite_positions.pkl ======
with open("satellite_positions.pkl", "wb") as f:
    pickle.dump(satellite_positions, f)
print("Saved satellite_positions.pkl")
# ====== 輸出 all_satellite_names.csv ======
all_satellite_names = [sat.name for sat in satellites]
df_names = pd.DataFrame({"sat_name": all_satellite_names})
df_names.to_csv("all_satellite_names.csv", index=False)
print("Saved all_satellite_names.csv")