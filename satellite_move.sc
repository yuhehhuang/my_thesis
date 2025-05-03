import win32com.client
import os

app = win32com.client.Dispatch("STK11.Application")
app.Visible = True
app.UserControl = True
stkRoot = app.Personality2

# 設定檔案儲存路徑
output_dir = r"C:\Users\YourName\Documents\STK_Access_CSV"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 開啟你的 Scenario（也可以是 New）
stkRoot.LoadScenario(r"C:\Users\YourName\Documents\STK 11\Scenarios\MyHandover.sc")
scenario = stkRoot.CurrentScenario

# 假設你有一顆衛星群 LEO1 ~ LEO10
N = 10
user_names = ["User1", "User2"]  # 你可以根據自己建立的地面站名調整

for user_name in user_names:
    user = scenario.Children.GetElements("eFacility").Item(user_name)
    
    for i in range(1, N + 1):
        sat = scenario.Children.GetElements("eSatellite").Item(f"LEO{i}")
        
        access = sat.GetAccessToObject(user)
        access.ComputeAccess()

        # 匯出 Report（CSV格式）
        accessReport = access.DataProviders.GetDataPrvTimeVarFromPath("Access Data")
        result = accessReport.Exec(scenario.StartTime, scenario.StopTime, 60)  # 每60秒

        start_times = result.DataSets.GetDataSetByName("Start Time").GetValues()
        stop_times = result.DataSets.GetDataSetByName("Stop Time").GetValues()

        # 寫入 CSV
        with open(os.path.join(output_dir, f"{user_name}_LEO{i}_access.csv"), "w") as f:
            f.write("Start,Stop\n")
            for st, et in zip(start_times, stop_times):
                f.write(f"{st},{et}\n")