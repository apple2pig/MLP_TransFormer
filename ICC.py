from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import glob


def normalize(series):
    return (series - series.min()) / (series.max() - series.min())


icc_all = []
for csv_file in glob.glob('/home/data_disk/user5/cdata/battery2/' + '*.csv'):
    # csv_file = '/home/data_disk/user5/cdata/battery2/WCVT1000000000301.csv'
    name = csv_file.split('/')[-1].split('.')[0]

    data = pd.read_csv(csv_file)
    all_u = np.nan_to_num(data.loc[:, 'U_1':'U_92'].to_numpy())
    m = all_u.shape[0]
    for i in range(all_u.shape[0]):
        for j in range(all_u.shape[1]):
            if all_u[i, j] == 0:
                all_u[i, j] = np.median(all_u[:, j])
    if all_u[10, 10] > 1000:
        all_u = all_u / 1000

    n = all_u.shape[0]
    icc = []
    for j in range(all_u.shape[1] - 1):
        x1 = all_u[:, j]
        x2 = all_u[:, j + 1]
        _x = (np.sum(x1) + np.sum(x2)) / (2 * n)
        S2 = (np.sum(np.square(x1 - _x)) + np.sum(np.square(x2 - _x))) / (2 * n)
        fm = 0
        for i in range(all_u.shape[0]):
            fm += (x1[i].item() - _x) * (x2[i].item() - _x)
        icc.append(fm / (n * S2))

    plt.figure(figsize=(15, 6))
    plt.plot(icc, c='b')
    plt.title(name)
    plt.legend()
    plt.grid(True)

    # 坐标轴的实例
    ax = plt.gca()

    # 取消突起刻度线
    ax.tick_params(axis='both', which='both', length=0)

    # 设置边框颜色为灰色
    ax.spines['top'].set_color('gray')
    ax.spines['right'].set_color('gray')
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    # plt.savefig()
    plt.show()
