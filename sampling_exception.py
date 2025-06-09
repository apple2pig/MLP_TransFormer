import glob
import pandas as pd
import numpy as np


def normalize(series):
    return (series - series.min()) / (series.max() - series.min())


def extract_features(discharge_data):
    # S201: 计算电压矩阵中每个电压值与同行的中位数电压值的差值，得到目标矩阵
    for i in range(rows):
        median = np.median(discharge_data[i, :])
        discharge_data[i, :] -= median

    return discharge_data


def calculate_distance_accumulation_matrix(target_matrix, window_length):
    # S202: 设定滑动窗口的时间长度，并通过滑动窗口划分目标矩阵
    num_records = target_matrix.shape[0]
    num_windows = num_records - window_length + 1

    # S203: 计算距离累积矩阵
    distance_accumulation_matrix = []

    for start_idx in range(num_records):
        window_length = min(window_length, num_records - window_length)
        window = target_matrix[start_idx:start_idx + window_length, :]
        distance_accumulation_vector = np.sum(window, axis=0)
        distance_accumulation_matrix.append(distance_accumulation_vector)

    distance_accumulation_matrix = np.array(distance_accumulation_matrix)

    return distance_accumulation_matrix


for csv_filename in glob.glob('/home/data_disk/user5/cdata/battery2/*.csv'):
    # 读取CSV文件
    data = pd.read_csv(csv_filename)

    # 获取放电状态的电压数据  loc[data['CHARGE_STATUS'] == 3].
    discharge_data = data.loc[data['CHARGE_STATUS'] == 3, 'U_1':'U_92'].to_numpy()
    discharge_data = discharge_data.astype(float)
    # avg = data.loc[:, 'U_1':'U_92'].apply(normalize).min(axis=1)

    # 获取电压矩阵的行数和列数
    rows = discharge_data.shape[0]
    cols = discharge_data.shape[1]

    # 设定滑动窗口的时间长度
    window_length = 3

    # 执行特征提取和距离累积矩阵计算
    target_matrix = extract_features(discharge_data)
    distance_accumulation_matrix = calculate_distance_accumulation_matrix(target_matrix, window_length)

    # S301-S302: 获取各个电芯的第一分位数和第二分位数
    first_quartiles = []
    second_quartiles = []

    for j in range(distance_accumulation_matrix.shape[1]):
        sorted_col = sorted(distance_accumulation_matrix[:, j])
        first_quartile = np.percentile(sorted_col, 5)
        second_quartile = np.percentile(sorted_col, 95)
        first_quartiles.append(first_quartile)
        second_quartiles.append(second_quartile)

    # S4: 获取新能源汽车电池组整体的第一全局分位数和第二全局分位数
    global_first_quartile = np.percentile(np.array(sorted(first_quartiles)), 5)
    global_second_quartile = np.percentile(np.array(sorted(second_quartiles)), 95)

    # S5: 遍历距离累积矩阵中的每个电芯的特征向量，判断是否发生采样异常
    y = np.zeros((rows, cols))
    for i in range(distance_accumulation_matrix.shape[0]):
        for j in range(distance_accumulation_matrix.shape[1]):
            if distance_accumulation_matrix[i, j] < global_first_quartile:
                if (j < distance_accumulation_matrix.shape[1] - 1 and distance_accumulation_matrix[
                    i, j + 1] > global_second_quartile) or (
                        j > 0 and (distance_accumulation_matrix[i, j - 1] > global_second_quartile)):
                    y[i][j] = 1

    result = np.sum(y, axis=0)
    threshold = rows * 0.01
    data['sampling_exception'] = 0
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if result[j] < threshold:
                break
            if y[i, j] == 1 or data.loc[i, 'sampling_exception'] == 1:
                data.loc[i, 'sampling_exception'] = 1
            else:
                data.loc[i, 'sampling_exception'] = 0

    # original_data = pd.read_csv(csv_filename)
    # # 添加列
    # original_data['sampling_exception'] = data['sampling_exception']
    # # 将处理后的数据写回原文件
    # original_data.to_csv(csv_filename, index=False)
    # print(csv_filename.split('/')[-1].split('.')[0], "数据已成功写回原文件")

    # result = [1 if result[p] > 1 else 0 for p in range(len(result))]
    # 输出标记的时刻及发生故障的电芯号
    # from matplotlib import pyplot as plt
    #
    # plt.figure(figsize=(15, 6))
    # plt.plot([rows * 0.01] * cols, c='b')
    # plt.plot(result, c='r')
    # plt.title(csv_filename.split('/')[-1].split('.')[0])
    # plt.show()
