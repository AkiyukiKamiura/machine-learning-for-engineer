#! python3

import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from PIL import Image
from sklearn import datasets
import glob

# mnistの手書き数字データを用いて分類
mnist = datasets.fetch_mldata('MNIST original')
X, y = mnist.data / 255,  mnist.target
bin_X = np.array([[0 if d <= 0.5 else 1 for d in data] for data in X])
train_df = pd.DataFrame(bin_X)

N = len(train_df)
K = 10
mix = [1/K for i in range(K)]
mu = np.random.rand(K, 784)

def bern(x, mu):
    r = 1.0
    for xi, mui in zip(x, mu):
        if xi == 1:
            r *= mui
        else:
            r *= (1.0 - mui)
    return r

def run_em(mix, mu):
    # Estimation Phase
    resp = pd.DataFrame()
    for ind, line in train_df.iterrows():
        tmp = []
        for k in range(K):
            a = mix[k]*bern(line, mu[k])
            sum_a = sum([mix[kk]*bern(line, mu[kk]) for kk in range(K)])
            tmp.append(a/sum_a)
        resp = resp.append([tmp], ignore_index=True)

    # Maximization Phase
    new_mix = np.zeros(K)
    new_mu = np.zeros((K, 28*28))
    for k in range(K):
        nk = resp[k].sum()
        new_mix[k] = nk/N
        for index, line in train_df.iterrows():
            new_mu[k] += line*resp[k][index]
        new_mu[k] /= nk

    clst = []
    for index, line in resp.iterrows():
        clst.append(np.argmax(line[0:]))

    return new_mix, new_mu, clst

def show_images(mu):
    fig = plt.figure()
    for k in range(K):
        subplot = fig.add_subplot(1, K, k+1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.imshow(mu[k].reshape(28, 28), cmap=plt.cm.gray_r)

for i in range(10):
    mix, mu, cls = run_em(mix, mu)
    show_images(mu)
    plt.show()
