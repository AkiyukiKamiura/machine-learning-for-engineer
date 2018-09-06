#! python3

import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas
import PIL
import sklearn

def least_square_error(data_x, data_t, M):
    t_mat = np.array(data_t)
    x_powered_mat = np.array([[pow(x, m) for m in range(M+1)] for x in data_x])
    coefficient_w = np.linalg.inv(np.dot(x_powered_mat.transpose(), x_powered_mat)).dot(x_powered_mat.transpose()).dot(t_mat)[::-1]
    return coefficient_w

# 学習データの数と近似曲線の次元数を設定
N = 10
M = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
display_M = [0, 1, 3, 9]
display_count = 0
rmse_test_list = []
rmse_training_list = []

# データセット作成
data_x = np.linspace(0.0, 1.0, N)
continuity_x = np.linspace(0.0, 1.0, 201)
data_t_true = np.sin(2*np.pi*data_x)
data_t_training = data_t_true + np.random.normal(0.0, 0.3, len(data_x))
data_t_test = data_t_true + np.random.normal(0.0, 0.3, len(data_x))

fig, ax = plt.subplots(ncols=2, nrows=math.ceil(len(display_M)/2), figsize=(10,7), sharex=True)

# 最小二乗法
for i, m in enumerate(M):
    coefficient_w = least_square_error(data_x, data_t_training, m)
    approximate_curve = np.polyval(coefficient_w, continuity_x)

    # 平方根平均二条誤差 (Root Mean Square Error) = sqrt(2Ed/N)
    error_training = np.sum(np.power(np.polyval(coefficient_w, data_x) - data_t_training, 2))/2
    rmse_training = math.sqrt(2*error_training/N)
    rmse_training_list.append(rmse_training)

    error_test = np.sum(np.power(np.polyval(coefficient_w, data_x) - data_t_test, 2))/2
    rmse_test = math.sqrt(2*error_test/N)
    rmse_test_list.append(rmse_test)

    # 可視化
    if i in display_M:
        row, col = math.floor(display_count/2), math.floor(display_count%2)
        ax[row][col].plot(continuity_x, np.sin(2*np.pi*continuity_x), '--')
        ax[row][col].plot(data_x, data_t_training, 'ob')
        ax[row][col].plot(continuity_x, np.array(approximate_curve), label='rsme:'+str(round(rmse_training, 2)))
        ax[row][col].set_title('m = ' + str(m))
        ax[row][col].set_xlabel('t')
        ax[row][col].set_ylabel('x')
        ax[row][col].grid(True)
        ax[row][col].legend()
        display_count += 1

plt.savefig('least_square/graph.png')

# RMSを可視化 = オーバーフィッティングの検出
plt.figure()
plt.plot(M, rmse_training_list, label='Training Set')
plt.plot(M, rmse_test_list, label='Test Set')
plt.title('RMS Error')
plt.xlabel('M')
plt.grid(True)
plt.legend()
plt.savefig('least_square/rms.png')
plt.show()
