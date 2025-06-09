import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import glob
from scipy.spatial import distance as dis
from matplotlib.pyplot import MultipleLocator
import seaborn as sns
from matplotlib import ticker

sns.set_style("darkgrid")


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
    for csv_file in glob.glob('/home/data_disk/user5/cdata/battery2/' + '*.csv'):
        csv_file = '/home/data_disk/user5/cdata/battery2/WCVT1000000000218.csv'
        name = csv_file.split('/')[-1].split('.')[0]

        data = pd.read_csv(csv_file)
        all_u = np.nan_to_num(data.loc[:, 'U_1':'U_92'].to_numpy())
        time = np.nan_to_num(data.loc[:, 'TIME'].to_numpy())
        m = all_u.shape[0]
        for i in range(all_u.shape[0]):
            for j in range(all_u.shape[1]):
                if all_u[i, j] == 0:
                    all_u[i, j] = np.median(all_u[:, j])
        if np.argmax(all_u) > 1000:
            all_u = all_u / 1000

        Delta = []
        plt.figure(figsize=(15, 6))
        for j in range(all_u.shape[1]):
            tem = delta(all_u[:, j])
            Delta.append(tem)

        ICC = []
        exception = []
        for j in range(len(Delta) - 1):
            icc = calculate_ICC(Delta[j], Delta[j + 1])
            if icc < 0.8:
                print(f'{name} connection exception between U{j + 1} and U{j + 2}')
                exception.append(j + 1)
                exception.append(j)
            ICC.append(icc)
        # plt.plot(ICC, c='b', label='ICC')

        Delta = normalize(np.array(Delta))
        result = np.array([0 for _ in range(Delta.shape[1])])
        for j in range(Delta.shape[0]):
            if exception.count(j) > 1:
                for i in range(Delta.shape[1]):
                    if Delta[j, i] > 0.7 or Delta[j, i] < -0.7:
                        print(f'{name} connection exception Time: {time[i]}')
                        result[i:] = 1
                        break

        plt.plot(result, label='result', color='r')
        plt.title(name)
        plt.legend()
        plt.grid(True, linewidth=2)
        ax = plt.gca()
        plt.show()
