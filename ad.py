import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imageio.v2 as imageio
import datetime

def trans_time(timestamps):
    dates = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]
    formatted_dates = [date.strftime('%Y-%m-%d %H:%M:%S') for date in dates]
    return formatted_dates


image_list = []
csv_file = "/home/data_disk/user5/cdata/temporary_files/battery2/WCVT1000000000353.csv"
# 初始化数据

name = csv_file.split('/')[-1].split('.')[0]
data = pd.read_csv(csv_file)
time = np.nan_to_num(data.loc[:, 'TIME'].to_numpy())
all_u = data.loc[:, 'U_1':'U_92'].to_numpy()
print(all_u[500:1000].min(),all_u[500:1000].max())

if type(time[0]) == np.int64:
    time = trans_time(time)

t_list = []
for k in range(500,1000):
    plt.figure(figsize=(8, 6))
    for j in range(all_u.shape[1]):
        if j == 56:
            plt.plot(time[k:k + 8], all_u[k:k + 8, j], c='r', label='U_57', linewidth=2)
            continue
        plt.plot(time[k:k + 8], all_u[k:k + 8, j], c='b', linewidth=0.5, alpha=0.5)
    plt.legend(loc=1)
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.ylim((3620,3644))
    plt.savefig('/home/data_disk/user5/cdata/temporary_files/Us/battery/pic/temp.png')
    image_list.append(imageio.imread('/home/data_disk/user5/cdata/temporary_files/Us/battery/pic/temp.png'))
    plt.close()

imageio.mimsave('/home/data_disk/user5/cdata/temporary_files/Us/battery/pic/pic.gif', image_list, fps=50,loop=0)
