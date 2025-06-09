import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import glob
from scipy.spatial import distance as dis
from matplotlib.pyplot import MultipleLocator
from matplotlib import ticker
import imageio.v2 as imageio


def normalize(series):
    return (series - series.min()) / (series.max() - series.min())


def delta(v):
    m = v.shape[0]
    result = [0]
    for i in range(1, m):
        result.append((v[i] - v[i - 1]) / 10)
    return result


def calculate_ICC(v1, v2):
    n = len(v1)
    x1 = np.array(v1)
    x2 = np.array(v2)
    _x = (np.sum(x1) + np.sum(x2)) / (2 * n)
    S2 = (np.sum(np.square(x1 - _x)) + np.sum(np.square(x2 - _x))) / (2 * n)
    fm = 0
    for i in range(len(x1)):
        fm += (x1[i].item() - _x) * (x2[i].item() - _x)
    icc = fm / (n * S2)
    return icc


if __name__ == '__main__':
    for csv_file in glob.glob('/home/data_disk/user5/cdata/temporary_files/battery2/' + '*.csv'):
        csv_file = '/home/data_disk/user5/cdata/temporary_files/battery2/WCVT1000000000099.csv'
        name = csv_file.split('/')[-1].split('.')[0]

        data = pd.read_csv(csv_file)
        all_u = np.nan_to_num(data.loc[:, 'U_1':'U_92'].to_numpy())
        time = np.nan_to_num(data.loc[:, 'TIME'].to_numpy())
        m = all_u.shape[0]
        for i in range(all_u.shape[0]):
            for j in range(all_u.shape[1]):
                if all_u[i, j] == 0:
                    all_u[i, j] = np.median(all_u[:, j])
        if max(all_u[0]) > 1000:
            all_u = all_u / 1000

        Delta = []
        t1_list = []
        # 画左边异常图的
        plt.figure(figsize=(10, 3))
        plt.title(name + ' Delta U')
        plt.grid(True)
        for j in range(all_u.shape[1]):
            tem = delta(all_u[:, j])
        #     if j==85:
        #         plt.plot(all_u[:, j], linewidth=3, c='r', alpha=0.5)
        #     else:   
        #         plt.plot(all_u[:, j], linewidth=1)
        # plt.plot(all_u[:, 84], linewidth=2, c='b', label='U_85')
        # plt.plot(all_u[:, 86], linewidth=2, c='b', label='U_87')
        # plt.plot(all_u[:, 85], linewidth=2, c='r', label='U_86')
        # plt.legend()
        # plt.savefig(f'/home/data_disk/user5/cdata/temporary_files/Us/battery/pic/tem.png')
            Delta.append(tem)
        #     im = imageio.imread('/home/data_disk/user5/cdata/temporary_files/Us/battery/pic/tem.png')
        #     t1_list.append(im)
        # imageio.mimsave('/home/data_disk/user5/cdata/temporary_files/Us/battery/pic/t4.gif', t1_list, loop=0)
        # plt.plot([0.015] * m, c='b')
        # plt.plot([-0.015] * m, c='b')
        
        


        ICC = []
        exception = []
        y = []
        for j in range(len(Delta) - 1):
            icc = calculate_ICC(Delta[j], Delta[j + 1])
            if icc < 0.8:
                # print(f'{name} connection exception between U{j + 1} and U{j + 2}')
                # exception.append(j + 1)
                exception.append(j)
                y.append(icc)
                
            ICC.append(icc)
        # 画右边ICC的
        plt.figure(figsize=(10, 8))
        plt.plot(ICC, c='b', label='ICC')
        plt.scatter(x=np.array(exception), y=np.array(y), c='r',s=100)
        print(y,exception)
        plt.title(name + ' ICC')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'/home/data_disk/user5/cdata/temporary_files/Us/battery/pic/{name}+ICC.png')
        plt.show()

        break
        # data['connection_exception'] = pd.DataFrame([0 for _ in range(all_u.shape[0])])
        # Delta = normalize(np.array(Delta))
        # for j in range(Delta.shape[0]):
        #     if exception.count(j) > 1:
        #         for i in range(Delta.shape[1]):
        #             if Delta[j, i] > 0.7 or Delta[j, i] < -0.7:
        #                 print(f'{name} connection exception Time: {time[i]}')
        #                 data.loc[i:, 'connection_exception'] = 1
        #                 break
        # #
        # # original_data = pd.read_csv(csv_file)
        # # original_data['connection_exception'] = data['connection_exception']
        # # original_data.to_csv(csv_file, index=False)
        # # print(csv_file.split('/')[-1].split('.')[0], "数据已成功写回原文件")

        # # plt.plot(data['connection_exception'], label='result', color='r')
        # # plt.title(name)
        # # plt.legend()
        # # plt.grid(True, linewidth=2)
        # # ax = plt.gca()
        # # plt.show()
