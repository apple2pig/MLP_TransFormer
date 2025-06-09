import numpy as np
import math
import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathos.multiprocessing import ThreadPool as Pool  # 多线程
import torch

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")


def normalize(series):
    return (series - series.min()) / (series.max() - series.min())


def ApEn2(s: list | np.ndarray, r: float, m: int = 2):
    s = np.squeeze(s)
    th = r * np.std(s)  # 容限阈值

    def phi(m):
        n = len(s)
        x = s[np.arange(n - m + 1).reshape(-1, 1) + np.arange(m)]
        ci = lambda xi: ((np.abs(x - xi).max(1) <= th).sum()) / (n - m + 1)  # 构建一个匿名函数
        c = Pool().map(ci, x)  # 所传递的参数格式: 函数名,函数参数
        return np.sum(np.log(c)) / (n - m + 1)

    x = Pool().map(phi, [m, m + 1])
    H = x[0] - x[1]
    return H


if __name__ == '__main__':
    csv_file = '/home/data_disk/user5/cdata/battery2/WCVT1000000000099.csv'
    name = csv_file.split('/')[-1].split('.')[0]

    data = pd.read_csv(csv_file)
    charge_status = data.loc[:, 'CHARGE_STATUS'].to_numpy()
    # max_v = all_u.max(axis=1).to_numpy()
    # min_v = all_u.min(axis=1).to_numpy()
    all_u = data.loc[:, 'U_1':'U_92'].to_numpy()
    min_v = data.loc[:, 'MIN_CELL_VOLT'].to_numpy()

    for i in range(all_u.shape[0]):
        for j in range(all_u.shape[1]):
            if all_u[i, j] == 0:
                all_u[i, j] = min_v[i]

    w = 10
    status = charge_status[0]
    node = 0
    min_v_result = []
    m = all_u.shape[0]
    for j in range(0, all_u.shape[1]):
        tem = []
        for i in range(0, m - w, w):
            tem.append(ApEn2(all_u[i:i + w, j],0.2))
        min_v_result.append(tem)
        plt.figure(figsize=(15, 6))
        plt.plot(tem, c='r', label=f'U_{j + 1}')
        plt.legend()
        plt.title(name)
        plt.show()

    plt.figure(figsize=(15, 6))
    for th in min_v_result:
        plt.plot(th, label=f'U_{min_v_result.index(th) + 1}')
    # plt.plot(max_v_result, c='r', label='short')
    # plt.plot(max_v, label='max_v')
    # plt.plot(min_v, c='b', label='min_v')
    plt.legend()
    plt.title(name)
    plt.show()
