import pandas as pd
import matplotlib.pyplot as plt
import glob

for csv_filename in glob.glob('/home/data_disk/user5/cdata/battery/*.csv'):
    # 读取数据文件
    file_path = csv_filename  # 请将路径替换为你的实际文件路径
    data = pd.read_csv(file_path, usecols=[0, 1, 7], header=0,
                       names=['index', 'charge_status', 'INSULATION_RESISTANCE'])

    # 重置索引
    data['index'] = pd.RangeIndex(start=1, stop=len(data) + 1, step=1)

    max_value = data['INSULATION_RESISTANCE'].max()
    min_value = data['INSULATION_RESISTANCE'].min()
    data_range = max_value - min_value

    # 初始化failure of insulation列
    data['failure of insulation'] = 0

    # 对绝缘失效情况进行标记
    for i in range(len(data)):
        if data.loc[i, 'charge_status'] == 1 or data.loc[i, 'charge_status'] == 2:
            if data.loc[i, 'INSULATION_RESISTANCE'] < 41:
                data.loc[i, 'failure of insulation'] = 1

        elif data.loc[i, 'charge_status'] == 3 or data.loc[i, 'charge_status'] == 4:
            if data.loc[i, 'INSULATION_RESISTANCE'] < 205:
                data.loc[i, 'failure of insulation'] = 1
            if i < 2 or i + 2 > len(data):
                continue
            else:
                if ((data.loc[i + 1, 'charge_status'] == 1 or data.loc[i + 1, 'charge_status'] == 2) or (
                        data.loc[i - 1, 'charge_status'] == 1 or data.loc[i - 1, 'charge_status'] == 2)) and data.loc[
                    i, 'INSULATION_RESISTANCE'] > 41:
                    data.loc[i, 'failure of insulation'] = 0

    # 补充情况
    i = 0
    while i < len(data):
        if data.loc[i, 'failure of insulation'] == 1:
            j = i + 1
            while j < len(data):
                diff = abs(data.loc[j, 'INSULATION_RESISTANCE'] - data.loc[j - 1, 'INSULATION_RESISTANCE'])
                if diff < 0.003 * data_range:
                    data.loc[j, 'failure of insulation'] = 1
                    j += 1
                elif diff > 0.003 * data_range:
                    i = j - 1  # 更新i以跳过已经处理的点
                    break
                else:
                    j += 1
            i = j  # 更新i以跳过已经处理的点
        else:
            i += 1

    # 将failure of insulation中的标签写回文件
    # 读取原始数据
    original_data = pd.read_csv(file_path)

    # 添加failure of insulation列
    original_data['failure_of_insulation'] = data['failure of insulation']

    # 将处理后的数据写回原文件
    original_data.to_csv(file_path, index=False)
    print(csv_filename.split('/')[-1], "数据已成功写回原文件")
