#! python3

import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas
import PIL
import sklearn

"""
最尤推定法は多項式の係数{w}に加えて、標準偏差{σ}をパラメータとして
トレーニングデータのセットが得られる確率の対数を最大化する問題を解析的に解く
結果的に、多項式の係数{w}は最小二乗法と同じ、
標準偏差{σ}は、平方根平均二乗誤差に等しい
ことがわかる
"""

def least_square_error(data_x, data_t, M):
    t_mat = np.array(data_t)
    x_powered_mat = np.array([[pow(x, m) for m in range(M+1)] for x in data_x])
    coefficient_w = np.linalg.inv(np.dot(x_powered_mat.transpose(), x_powered_mat)).dot(x_powered_mat.transpose()).dot(t_mat)[::-1]
    return coefficient_w

# 学習データの数と近似曲線の次元数を設定
N = 100
M = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
display_M = [0, 1, 3, 9]
display_count = 0
log_likelihood_training_list = []
log_likelihood_test_list = []

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
    log_likelihood_training = (N/2)*(math.log(N) - math.log(2*rmse_training) - math.log(2*math.pi) - 1)
    log_likelihood_training_list.append(log_likelihood_training)

    error_test = np.sum(np.power(np.polyval(coefficient_w, data_x) - data_t_test, 2))/2
    rmse_test = math.sqrt(2*error_test/N)
    log_likelihood_test = (N/2)*(math.log(N) - math.log(2*rmse_test) - math.log(2*math.pi) - 1)
    log_likelihood_test_list.append(log_likelihood_test)

    # 標準偏差 standard deviation
    upper_sd = approximate_curve + rmse_training
    lower_sd = approximate_curve - rmse_training

    # 可視化
    if i in display_M:
        row, col = math.floor(display_count/2), math.floor(display_count%2)
        ax[row][col].plot(continuity_x, np.sin(2*np.pi*continuity_x), '--')
        ax[row][col].plot(data_x, data_t_training, 'ob')
        ax[row][col].plot(continuity_x, np.array(approximate_curve), label='sigma:'+str(round(rmse_training, 2)))
        ax[row][col].plot(continuity_x, upper_sd, '--', color='gray', lw=1)
        ax[row][col].plot(continuity_x, lower_sd, '--', color='gray', lw=1)
        ax[row][col].set_title('m = ' + str(m))
        ax[row][col].set_xlabel('t')
        ax[row][col].set_ylabel('x')
        ax[row][col].grid(True)
        ax[row][col].legend()
        display_count += 1

plt.savefig('maximum_likelihood/graph.png')

# RMSを可視化 = オーバーフィッティングの検出
plt.figure()
plt.plot(M, log_likelihood_training_list, label='Training Set')
plt.plot(M, log_likelihood_test_list, label='Test Set')
plt.title('Log likelihood')
plt.xlabel('M')
plt.grid(True)
plt.legend()
plt.savefig('maximum_likelihood/log_likelihood.png')
plt.show()
