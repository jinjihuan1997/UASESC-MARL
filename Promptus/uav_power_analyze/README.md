一, 获取无人机启动状态能耗与电量的关系

步骤1: 从无人机中拷贝 .DAT 原始数据
步骤2: 运行 DatCon 软件得到 CSV 文件(勾选All Signal Groups)
步骤3: 将得到的 CSV 文件放入到 raw_data 文件夹

代码文件: dji_energy_plot.py
功能: 绘制无人机能耗对应图像, 数据(功耗,SoC) 图像显示
代码文件: dji_prune_csv.py 
功能: 精简原数据文件的代码, 只保留需要的几列内容

运行指令:
只剪枝 (生成 *_slim.csv, 不画图)
python uav_power_analyze/dji_prune_csv.py --csv uav_power_analyze/raw_data/condition_1/FLY967.csv
剪枝后自动调用你的原脚本 dji_energy_plot.py 作图
python uav_power_analyze/dji_prune_csv.py --csv uav_power_analyze/raw_data/condition_1/FLY967.csv --auto_plot --plot_args "--lang zh --dpi 200"

二, 获取无人机启动+信号发射状态能耗与电量的关系

三, 获取无人机启动+信号接收+信号发射状态能耗与电量的关系

四, 无人机搭载的无线通信 wifi 为 RT-AX86U PRO
后台管理地址为: http://router.asus.com
默认 IP 地址: http://192.168.50.1
SSID:    UAV1-2.4G    UAV1-5G
连接密码: enes2025     enes2025
后台账号: admin
后台密码: enes2025

五, 原始数据的采集也是基于一定条件的, 采集条件为:
condition_1: 无人机与遥控器相连接, 无人机无任何外接负载 (wifi板,边缘计算设备), 采集运行20分钟内的能源消耗信息 (已完成, 2025年10月30日)
condition_2: 无人机与遥控器相连接, 无人机外接wifi板,推流路线为1, 采集运行20分钟内的能源消耗信息 (已完成, 2025年10月30日)
condition_3: 无人机与遥控器相连接, 无人机外接wifi板,推流路线为2, 采集运行20分钟内的能源消耗信息
condition_4: 无人机与遥控器相连接, 无人机外接wifi板,推流路线为3, 采集运行20分钟内的能源消耗信息
condition_5: 无人机与遥控器相连接, 无人机外接wifi板,推流路线为3, 采集运行20分钟内的能源消耗信息
