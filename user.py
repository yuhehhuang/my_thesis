#to set user information:(1) user_id, (2) begin, (3) end (4) legal handover count(have K value)
#save at user_schedule.csv
#only need to run once
import pandas as pd
import random

T = 96
NUM_USERS = 500

users = []
for user_id in range(NUM_USERS):
    duration = random.randint(5, 25)
    t_start = random.randint(0, T - duration)
    t_end = t_start + duration
    row = {
        "user_id": user_id,
        "t_start": t_start,
        "t_end": t_end
    }
    # K是為了實驗結果有不同的handover次數，顯示成功率
    for K in range(6):
        row[f"K{K}"] = K + (duration // 3)+5
    users.append(row)

df = pd.DataFrame(users)
df.to_csv("user_info_with_Ks.csv", index=False)