import win32com.client

app = win32com.client.Dispatch("STK11.Application")
app.Visible = True
app.UserControl = True

stkRoot = app.Personality2
scenario = stkRoot.Children.New(18, "HandoverScenario")  # 18 = eScenario

# 設定開始結束時間
scenario2 = scenario.QueryInterface(win32com.client.constants.IAgScenario)
scenario2.SetTimePeriod("Today", "+10min")
scenario2.StartTime = "Today"
scenario2.StopTime = "+10min"
scenario2.Epoch = "Today"
scenario2.Animation.AnimStepValue = 60  # 每分鐘一個 time slot