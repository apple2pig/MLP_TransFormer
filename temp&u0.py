import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import glob
from matplotlib.pyplot import MultipleLocator

if __name__ == '__main__':
    u = [(1, 2, 3, 4, 5), (6, 7, 8, 9, 10), (11, 12, 13, 14, 15), (16, 17, 18, 19, 20), (21, 22, 23, 24, 25),
         (26, 27, 28, 29, 30), (31, 32, 33, 34, 35), (36, 37, 38, 39, 40), (41, 42, 43, 44, 45), (46, 47, 48, 49, 50),
         (51, 52, 53, 54, 55, 56), (57, 58, 59, 60, 61, 62), (63, 64, 65, 66, 67, 68), (69, 70, 71, 72, 73, 74),
         (75, 76, 77, 78, 79, 80), (81, 82, 83, 84, 85, 86), (87, 88, 89, 90, 91, 92)]
    t = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22), (23, 24),
         (25, 26), (27, 28), (29, 30), (31, 32), (33, 34)]
    for csv_file in glob.glob('/home/data_disk/user5/cdata/battery2/' + '*.csv'):
        csv_file = '/home/data_disk/user5/cdata/battery2/WCVT1000000000480.csv'
        name = csv_file.split('/')[-1].split('.')[0]

        data = pd.read_csv(csv_file)
        all_u = np.nan_to_num(data.loc[:, 'U_1':'U_92'].to_numpy())
        all_t = np.nan_to_num(data.loc[:, 'T_1':'T_34'].to_numpy())
        if np.argmax(all_u) > 1000:
            all_u = all_u / 1000

        plt.figure(figsize=(15, 6))
        for temperature in t:
            plt.plot(all_t[29430:29450, temperature[0] - 1])
            plt.plot(all_u[29430:29450, temperature[1] - 1])
            plt.grid()
            plt.show()
            break

        plt.figure(figsize=(15, 6))
        for voltage in u:
            for i in range(len(voltage)):
                plt.plot(all_u[29430:29450, voltage[i] - 1])
            plt.plot(all_u[29430:29450, 5],c='b')
            plt.grid()
            plt.show()
            break
        break
