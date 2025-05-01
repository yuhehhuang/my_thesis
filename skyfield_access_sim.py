from skyfield.api import load, Topos, utc
from datetime import datetime, timedelta
import pandas as pd

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
print(f" Starlink 衛星總數：{len(satellites)} 顆")

ts = load.timescale()
observer = Topos(latitude_degrees=USER_LAT, longitude_degrees=USER_LON)
t0 = ts.utc(START_TIME_UTC)



# ====== 模擬時間序列 ======
times = [ts.utc(START_TIME_UTC + timedelta(minutes=i)) for i in range(SLOTS)]

# ====== 模擬可視性 ======
access_matrix = []
for t_idx, t in enumerate(times):
    visible_sats = []
    for sat in satellites:
        difference = sat - observer
        topocentric = difference.at(t)
        #把 topocentric 轉換成「仰角（altitude）、方位角（azimuth）、距離」
        #仰角介於-90 到 90 度之間
        alt, az, dist = topocentric.altaz()
        if alt.degrees >= 45 :
            visible_sats.append(sat.name)
    access_matrix.append({
        "time_slot": t_idx,
        "visible_sats": visible_sats
    })
    print(f"Slot {t_idx}: {len(visible_sats)} sats visible")

# ====== 輸出 CSV ======
df = pd.DataFrame(access_matrix)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved access matrix to {OUTPUT_CSV}")
