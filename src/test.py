import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import ast  # 為了安全轉換字串成 list
import random
import yens_algo
import pickle
from collections import defaultdict
import performance_calculate
def load_satellite_names_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df["visible_sats"] = df["visible_sats"].apply(ast.literal_eval)
    all_sats = set()
    for sats in df["visible_sats"]:
        all_sats.update(sats)
    return sorted(list(all_sats))

satellites = load_satellite_names_from_csv("access_matrix.csv")
print(len(satellites)) #1026
for k in list(sat_load_dict.keys())[:10]:
    print(k, sat_load_dict[k])