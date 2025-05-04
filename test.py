import sys
sys.path.append("src")
import pandas as pd
import ast  # 為了安全轉換字串成 list
import random
import yens_algo
import pickle
from collections import defaultdict
import performance_calculate
with open("data/data_rate_dict.pkl", "rb") as f:
    data_rate_dict = pickle.load(f)

print("✅ 總共有 keys:", len(data_rate_dict))
print("🔍 隨機挑幾個 keys：")
for i, key in enumerate(data_rate_dict):
    if i < 20:
        print(key)