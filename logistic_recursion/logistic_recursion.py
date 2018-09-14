#! python3

import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas
import PIL
from sklearn import datasets

"""
ロジスティック回帰 # 分類アルゴリズムの一つ
確率を用いた最尤推定法
パーセプトロンより優位

尤度関数を解析的に最大化することはできないため,
ニュートン法の多次元拡張: ニュートンラフソン法 を用いる

平面上のすべての点において、クラスに割り当てられる確率が計算されるため、
ROC曲線を利用し、性能評価を定量的に行うことができる

ROC曲線: Receiver Operating Characteristic 曲線
真陽性と偽陽性の関係グラフ
"""

def logistic_function(x):
    return 1/(1 + np.exp(-x))

def roc(param, phi, t):
    true_pos_list, false_pos_list = [], []
    n = np.sum(t == 1)
    p = logistic_function(np.dot(phi, param))
    p_thres = np.linspace(0, 1)
    for thr in p_thres:
        true_pos = np.sum(np.logical_and(p >= thr, t == 1))
        false_pos = np.sum(np.logical_and(p >= thr, t == 0))
        true_pos_list.append(true_pos/n)
        false_pos_list.append(false_pos/n)
    return true_pos_list, false_pos_list

# データセット作成
N = 100
data, t = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, n_samples=N, n_classes=2)
x, y = data.transpose()
class_1_x, class_1_y = [x[i] for i in range(N) if t[i] == 1], [y[i] for i in range(N) if t[i] == 1]
class_2_x, class_2_y = [x[i] for i in range(N) if t[i] == 0], [y[i] for i in range(N) if t[i] == 0]

phi = np.array([[1, x[i], y[i]] for i in range(N)])

# パラメータの初期値
param = np.array([0., 0., 0.])
param_hist = [param]

# IRLS法
loop = 30

for l in range(loop):
    param_prev = np.copy(param)
    z = np.array([logistic_function(np.dot(param_prev, [1, x[i], y[i]])) for i in range(N)])
    R = np.diag(z)
    param = param_prev - np.linalg.inv(phi.T.dot(R).dot(phi)).dot(phi.T).dot(z-t)
    param_hist.append(param)

    if np.linalg.norm(param_prev) != 0 and pow(np.linalg.norm(param - param_prev)/np.linalg.norm(param_prev), 2) < 0.001: break

# 可視化
x_line = np.linspace(-5, 5, 2)
y_line = (param[0] + param[1]*x_line)/(-param[2])
plt.plot(class_1_x, class_1_y, 'o')
plt.plot(class_2_x, class_2_y, 'x')
plt.plot(x_line, y_line, color='gray')
plt.xlim(-3, 5)
plt.ylim(-4, 2)
plt.grid()
plt.savefig('logistic_recursion/classification.png')

plt.figure()

l = range(len(param_hist))
w0, w1, w2 = np.array(param_hist).transpose()
plt.plot(l, w0, label='w0')
plt.plot(l, w1, label='w1')
plt.plot(l, w2, label='w2')
plt.xlim(0, len(param_hist)-1)
plt.grid()
plt.legend()
plt.savefig('logistic_recursion/parameters.png')

plt.figure()

true_pos, false_pos = roc(param, phi, t)
plt.plot(false_pos, true_pos, 'o--')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.grid()
plt.savefig('logistic_recursion/roc.png')

plt.show()
