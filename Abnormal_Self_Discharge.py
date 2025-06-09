import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import time
import imageio.v2 as imageio

def normalize(series):
    return (series - series.min()) / (series.max() - series.min())


if __name__ == '__main__':
    for csv_file in glob.glob('/home/data_disk/user5/cdata/temporary_files/battery2/' + '*.csv'):
        csv_file = '/home/data_disk/user5/cdata/temporary_files/battery2/WCVT1000000000353.csv'
        name = csv_file.split('/')[-1].split('.')[0]

        data = pd.read_csv(csv_file)
        all_u = np.nan_to_num(data.loc[:, 'U_1':'U_92'].to_numpy())
        time_pd = np.nan_to_num(data.loc[:, 'TIME'].to_numpy())
        for i in range(all_u.shape[0]):
            for j in range(all_u.shape[1]):
                if all_u[i, j] == 0:
                    all_u[i, j] = np.median(all_u[:, j])
        if max(all_u[0]) > 1000:
            all_u = all_u / 1000

        l = all_u.shape[0]
        w_l = 300
        all_uf = []

        avg_tem = np.median(all_u, axis=0)
        all_u_centered = all_u - avg_tem

        # 使用NumPy的滚动窗口平均函数，如果需要的话，需要安装scipy库
        from scipy.signal import convolve

        weights = np.ones(w_l) / w_l

        # 扩展weights以匹配all_u_centered的形状
        weights_2d = weights[:, np.newaxis]

        # 使用卷积计算滚动平均
        all_u_rolling_avg = convolve(all_u_centered, weights_2d, mode='valid')

        for j in range(all_u.shape[1]):
            uf = np.empty(l)
            valid_length = l - w_l + 1
            uf[:valid_length] = all_u_centered[:valid_length, j] - all_u_rolling_avg[:, j]
            if valid_length < l:
                remaining_avg = np.mean(all_u_centered[valid_length - 1:, j])
                uf[valid_length:] = all_u_centered[valid_length:, j] - remaining_avg
            all_uf.append(uf.tolist())

        all_uf_flat = np.concatenate(all_uf)
        f1 = np.percentile(all_uf_flat, 25)
        f2 = np.percentile(all_uf_flat, 75)
        iqr = f2 - f1

        all_th_bottom = f1 - 3 * iqr
        all_th_top = f2 + 3 * iqr
        
        t_list = []
        image_list = []
        # 画右边特征值的
        plt.figure(figsize=(8, 6))
        plt.rcParams['font.family'] = 'KaiTi'
        plt.rcParams['axes.unicode_minus'] = False

        plt.plot([all_th_top]*l, c='r',linewidth=2, label='threshold')
        plt.plot((np.array(all_uf[56])-0.12), c='b',linewidth=1)
        plt.plot([all_th_bottom]*l, c='r',linewidth=2)
        
        plt.legend(loc=1)
        plt.grid(True)
        plt.savefig('/home/data_disk/user5/cdata/temporary_files/Us/battery/pic/pic.png')

        # for k in range(500):
        #     plt.figure(figsize=(8, 6))
        #     plt.rcParams['font.family'] = 'KaiTi'
        #     plt.rcParams['axes.unicode_minus'] = False

        #     plt.plot([all_th_top]*8, c='r',linewidth=5, label='上阈值')
        #     plt.plot([all_th_bottom]*8, c='r',linewidth=5, label='下阈值')
        #     plt.plot((np.array(all_uf[56])-0.12)[k+10000:k+10008], c='b',linewidth=5)
        #     plt.legend(loc=1)
        #     plt.grid(True)
        #     plt.savefig('/home/data_disk/user5/cdata/temporary_files/Us/battery/pic/temp.png')
        #     image_list.append(imageio.imread('/home/data_disk/user5/cdata/temporary_files/Us/battery/pic/temp.png'))
        #     plt.close()

        # imageio.mimsave('/home/data_disk/user5/cdata/temporary_files/Us/battery/pic/pic.gif', image_list, duration=15,loop=0)
        # break

        # min_all_uf = [min(col) for col in zip(*all_uf)]

        # y = np.zeros((l, len(all_uf)))
        # for j in range(len(all_uf)):
        #     for i in range(l):
        #         if (all_uf[j][i] < all_th_bottom or all_uf[j][i] > all_th_top) and all_uf[j][i] == min_all_uf[i]:
        #             y[i][j] = 1

        # result = np.sum(y, axis=0)
        # threshold = max(result) * 0.7
        # for j in range(len(all_uf)):
        #     if result[j] > threshold:
        #         index = j
        #         for i in range(l):
        #             if y[i][index] == 1:
        #                 if type(time_pd[i]) == np.int64:
        #                     t = time.localtime(time_pd[i])
        #                     print(
        #                         f'{name} Abnormal_self_discharge: U_{index + 1},TIME:{t.tm_year}-{t.tm_mon}-{t.tm_mday} {t.tm_hour}:{t.tm_min}:{t.tm_sec}')
        #                 else:
        #                     print(f'{name} Abnormal_self_discharge: U_{index + 1},TIME:{time_pd[i]}')
        #                 break
        # # plt.figure(figsize=(15, 6))
        # # plt.plot(result, c='b')
        # # plt.title(name)
        # # plt.show()
