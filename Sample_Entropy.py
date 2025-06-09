import numpy as np
import math
import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathos.multiprocessing import ThreadPool as Pool  # 多线程
import EntropyHub as EH


def normalize(series):
    return (series - series.min()) / (series.max() - series.min())


def SampleEntropy(Datalist, r, m=2000):
    list_len = len(Datalist)  # 总长度
    th = r * np.std(Datalist)  # 容限阈值

    def Phi(k):
        list_split = [Datalist[i:i + k] for i in range(0, list_len - k + (k - m))]  # 将其拆分成多个子列表
        # 这里需要注意，2维和3维分解向量时的方式是不一样的！！！
        Bm = 0.0
        for i in range(0, len(list_split)):  # 遍历每个子向量
            Bm += ((np.abs(list_split[i] - list_split).max(1) <= th).sum() - 1) / (len(list_split) - 1)  # 注意分子和分母都要减1
        return Bm

    ## 多线程
    # x = Pool().map(Phi, [m,m+1])
    # H = - math.log(x[1] / x[0])
    H = - math.log(Phi(m + 1) / Phi(m))
    return H



if __name__ == '__main__':
    # for csv_file in glob.glob('/home/data_disk/user5/cdata/battery/' + '*.csv'):
    csv_file = '/home/data_disk/user5/cdata/battery2/WCVT1000000000099.csv'
    name = csv_file.split('/')[-1].split('.')[0]
    plt.figure(figsize=(15, 6))
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

    max_v_result = []
    min_v_result = []
    w = 10
    status = charge_status[0]
    node = 0
    for j in range(0, all_u.shape[1]):
        tem = all_u[:, j]
        print(tem)
        min_v_SE = SampleEntropy(tem, 3)
        min_v_result.append(min_v_SE)
    plt.plot(min_v_result, c='r', label='short')
    # plt.plot(min_v, c='b', label='min_v')
    plt.legend()
    plt.title(name)
    plt.show()
