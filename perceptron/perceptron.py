#! python3

import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas
import PIL
from sklearn import datasets

"""
パーセプトロンの分類問題は、解析的に解くことができないため
確率的勾配降下法で数値解析
"""

# データセット作成
N = 100
data, t = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=5, n_clusters_per_class=1, n_samples=N, n_classes=2)
t = [1 if ele == 1 else -1 for ele in t]
x, y = data.transpose()
class_1_x, class_1_y = [x[i] for i in range(N) if t[i] == 1], [y[i] for i in range(N) if t[i] == 1]
class_2_x, class_2_y = [x[i] for i in range(N) if t[i] == -1], [y[i] for i in range(N) if t[i] == -1]
c = (np.sum(x) + np.sum(y))/(2*N)
print(c)

# パラメータの初期値 [w0, w1, w2]
param = np.array([0., 0., 0.])
param_hist = [list(param)]

# 確率的勾配降下法
loop = 40

for l in range(loop):
    print('loop: ', l)
    count = 0
    for i in range(N):
        phi = np.array([c, x[i], y[i]])
        if np.dot(phi, param)*t[i] <= 0: # 誤分類を行った場合
            param += phi*t[i]
            count += 1
    param_hist.append(list(param))
    print(count)

# 可視化
x_line = np.linspace(-5, 5, 2)
y_line = (c*param[0] + param[1]*x_line)/(-param[2])
plt.plot(class_1_x, class_1_y, 'o')
plt.plot(class_2_x, class_2_y, 'x')
plt.plot(x_line, y_line, color='gray')
plt.xlim(-3, 5)
plt.ylim(-4, 2)
plt.grid()
plt.savefig('perceptron/classification.png')

plt.figure()

l = range(loop+1)
w0, w1, w2 = np.array(param_hist).transpose()
plt.plot(l, w0, label='w0')
plt.plot(l, w1, label='w1')
plt.plot(l, w2, label='w2')
plt.xlim(0, 30)
plt.grid()
plt.legend()

plt.savefig('perceptron/parameters.png')
plt.show()
