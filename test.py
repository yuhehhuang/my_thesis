import sys
sys.path.append("src")
import pandas as pd
import ast  # 為了安全轉換字串成 list
import random
import yens_algo
import pickle
from collections import defaultdict
import performance_calculate
df_names = pd.read_csv("data/all_satellite_names.csv")
all_satellite_names = df_names["sat_name"].tolist()
print(all_satellite_names[:5])                # 印出前 5 顆衛星名稱
