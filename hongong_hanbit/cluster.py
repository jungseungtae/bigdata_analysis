import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = r'C:\Users\jstco\Downloads\hg-mldl-master\hg-mldl-master'

fruits = np.load(file_path + '/fruits_300.npy')
# print(fruits.shape)
# print(fruits)

## 이미지 1행
# print(fruits[0, 0, :])

# 이미지, 반전
# plt.imshow(fruits[0], cmap = 'gray')
# plt.show()

# plt.imshow(fruits[0], cmap = 'gray_r')
# plt.show()

# fig, axs = plt.subplots(1, 2)
# axs[0].imshow(fruits[100], cmap = 'gray_r')
# axs[1].imshow(fruits[200], cmap = 'gray_r')
# plt.show()

apple = fruits[0 : 100].reshape(-1, 100 * 100)
pine = fruits[100 : 200].reshape(-1, 100 * 100)
banana = fruits[200 : 300].reshape(-1, 100 * 100)

# print(apple.shape)
# print(apple.mean(axis = 1))

fruit_names = ["apple", "pine", "banana"]
fruit_data = [apple, pine, banana]

## 행으로 데이터 표현
# for data, name in zip(fruit_data, fruit_names):
#     plt.hist(np.mean(data, axis=1), alpha=0.8, label=name)
#
# plt.legend()
# plt.show()

## 열로 데이터 표현
# fig, axs = plt.subplots(1, 3, figsize = (20, 5))
#
# for i in range(3):
#     axs[i].bar(range(10000), np.mean(fruit_data[i], axis = 0))
#     axs[i].set_title(fruit_names[i])
#
# plt.show()

## 평균으로 이미지 만들기
# fig, axs = plt.subplots(1, 3, figsize = (20, 5))

apple_mean = np.mean(apple, axis = 0).reshape(100, 100)
pine_mean = np.mean(pine, axis = 0).reshape(100, 100)
banana_mean = np.mean(banana, axis = 0).reshape(100, 100)

mean_list = [apple_mean, pine_mean, banana_mean]

# for i in range(3):
#     axs[i].imshow(mean_list[i], cmap = 'gray_r')
#
# plt.show()

## 평균과 가까운 사과 사진 고르기
abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis = (1, 2))
# print(abs_diff)

apple_index = np.argsort(abs_mean)[: 100]
# print(apple_index)
# fig, axs = plt.subplots(10, 10, figsize = (10, 10))
#
# for i in range(10):
#     for j in range(10):
#         axs[i, j].imshow(fruits[apple_index[i * 10 + j]], cmap = 'gray_r')
#         axs[i, j].axis('off')
#
# plt.show()

## 바나나 찾기
abs_diff = np.abs(fruits - banana_mean)
abs_mean = np.mean(abs_diff, axis = (1, 2))
# print(abs_diff)

apple_index = np.argsort(abs_mean)[: 100]
# print(apple_index)
# fig, axs = plt.subplots(10, 10, figsize = (10, 10))
#
# for i in range(10):
#     for j in range(10):
#         axs[i, j].imshow(fruits[apple_index[i * 10 + j]], cmap = 'gray_r')
#         axs[i, j].axis('off')
#
# plt.show()

from sklearn.cluster import KMeans

fruits_2d = fruits.reshape(-1, 100 * 100)
# print(fruits_2d)

## 3개로 그룹
km = KMeans(n_clusters = 3, n_init=10, random_state = 42)
km.fit(fruits_2d)

# print(km.labels_)
# print(np.unique(km.labels_, return_counts = True))

def draw_fruits(arr, ratio = 1):
    n = len(arr)

    rows = int(np.ceil(n / 10))
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols, figsize=(cols * ratio, rows * ratio), squeeze = False)

    for i in range(rows):
        for j in range(cols):
            if i * 10 + j < n:
                axs[i, j].imshow(arr[i * 10 + j], cmap = 'gray_r')
            axs[i, j].axis('off')

    plt.show()

# 분류 0 : 사과, 1 : 바나나, 2 : 파인애플
# draw_fruits(fruits[km.labels_ == 0])
# draw_fruits(fruits[km.labels_ == 1])
# draw_fruits(fruits[km.labels_ == 2])

## 클러스터 중심 데이터로 이미지 만들기
# draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio = 3)

## 데이터 1개 거리 보기
# print(km.transform(fruits_2d[100:101]))
# print(km.predict(fruits_2d[100:101]))
# draw_fruits(fruits[100:101])
# print(km.n_iter_)

## 최적의 k 값 찾기
inertia = []

# for k in range(2, 7):
#     km = KMeans(n_clusters = k, n_init = 'auto', random_state = 42)
#     km.fit(fruits_2d)
#     inertia.append(km.inertia_)
#
# plt.plot(range(2, 7), inertia)
# plt.xlabel('k')
# plt.ylabel('inertia')
# plt.show()

## 주성분 분석 PCA
from sklearn.decomposition import PCA

pca = PCA(n_components = 50)
pca.fit(fruits_2d)

# print(pca.components_.shape)

# draw_fruits(pca.components_.reshape(-1, 100, 100))

fruits_pca = pca.transform(fruits_2d)
# print(fruits_pca.shape)

fruits_inverse = pca.inverse_transform(fruits_pca)
# print(fruits_inverse.shape)

# fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)

# for start in [0, 100, 200]:
#     draw_fruits(fruits_reconstruct[start:start + 100])
#     print('\n')

# print(np.sum(pca.explained_variance_ratio_))
#
# plt.plot(pca.explained_variance_ratio_)
# plt.show()


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

target = np.array([0] * 100 + [1] * 100 + [2] * 100)
# print(target)

from sklearn.model_selection import cross_validate
scores = cross_validate(lr, fruits_2d, target)
print(scores)

scores = cross_validate(lr, fruits_pca, target)
print(scores['test_score'])

pca = PCA(n_components=0.5)
pca.fit(fruits_2d)
# print(pca.n_components_)

fruits_pca = pca.transform(fruits_2d)
# print(fruits_pca.shape)

# scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))

km.fit(fruits_pca)
# print(np.unique(km.labels_, return_counts = True))

for label in range(3):
    data = fruits_pca[km.labels_ == label]
    plt.scatter(data[:, 0], data[:, 1])

plt.legend(['apple', 'banana', 'pine'])
plt.show()