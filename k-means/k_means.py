#! python3

import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas
from PIL import Image
from sklearn import datasets
import glob

"""
k平均法 - クラスタリングの一手法

クラスタリングをカラー写真の色情報に適応し、
代表色を決定することで、減色処理を行う
"""

# 減色対象の画像の読み込み
img = Image.open(glob.glob('k-means/img/fruits.jpeg')[0])
img = np.array(img)
width, height = img.shape[:2]

# 減色後の色の数とセントロイドの設定
K = 3
centroids = np.random.rand(K, 3) * 255

print('=====================================')
print('Number of clusters: K =', K)
print('Initial centers:', centroids)
print('=====================================')

for i in range(100):
    n_elem = [0 for i in range(K)]
    new_centroids = [np.array([0, 0, 0]) for i in range(K)]
    for w in range(width):
        for h in range(height):
            i = np.argmin(np.linalg.norm(centroids - img[w, h], axis=(1,)))
            new_centroids[i] += img[w, h]
            n_elem[i] += 1

    for i in range(K):
        if n_elem[i] != 0:
            new_centroids[i] = new_centroids[i]/n_elem[i]
        else:
            new_centroids[i] = centroids[i]
    new_centroids = np.array(new_centroids)

    if (new_centroids == centroids).all(): break
    centroids = new_centroids

rst = np.copy(img)
for w in range(width):
    for h in range(height):
        i = np.argmin(np.linalg.norm(centroids - img[w, h], axis=(1,)))
        rst[w, h] = centroids[i]

plt.imshow(rst)
plt.savefig('k-means/img/reduced_color_' + str(K) + '.png')
plt.show()
