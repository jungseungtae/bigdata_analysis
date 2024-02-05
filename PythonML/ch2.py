##### 제 2장. 시각화와 분석 테크닉 #####

import pandas as pd
from global_functions import DataAnalyzer

file_path = r'C:\Users\jstco\Downloads\PythonML-main\PythonML-main\chapter02'

order_data = pd.read_csv(file_path + '/order_data.csv')
# print(order_data.head())

## 1. 데이터 전처리
order_data = order_data.loc[(order_data['status'] == 1) |
                            (order_data['status'] == 2)]
# print(len(order_data))
# print(order_data.columns)

# analyze_data = order_data.columns
# analyze_data = list(analyze_data)
# # print(analyze_data)
#
# print(analyze_data.shape)
# print(analyze_data.head())

analyze_data = order_data[[
    'store_id', 'customer_id', 'coupon_cd',
    'order_accept_date', 'delivered_date', 'total_amount',
    'store_name', 'wide_area', 'narrow_area',
    'takeout_name', 'status_name']]
# print(analyze_data.shape)
# print(analyze_data.head())

## 2. 데이터 파악하기
# analyzer = DataAnalyzer(analyze_data)
# analyzer.summarize_basic()

import warnings

warnings.filterwarnings("ignore")

analyze_data[['store_id', 'coupon_cd']] = analyze_data[['store_id', 'coupon_cd']].astype(str)
# print(analyze_data.dtypes)
# print(analyze_data.describe())

## 3. 월별 매출 집계하기
# print(analyze_data['order_accept_date'].head())

analyze_data['order_accept_date'] = pd.to_datetime(analyze_data['order_accept_date'])
analyze_data['order_accept_month'] = analyze_data['order_accept_date'].dt.strftime('%Y%m')
# print(analyze_data[['order_accept_date', 'order_accept_month']].head())

analyze_data['delivered_date'] = pd.to_datetime(analyze_data['delivered_date'])
analyze_data['delivered_month'] = analyze_data['delivered_date'].dt.strftime('%Y%m')
# print(analyze_data[['delivered_date', 'delivered_month']].head())

month_data = analyze_data.groupby('order_accept_month')['total_amount']
# print(month_data.head())
# print(month_data.describe())
# print(month_data['total_amount'].sum())

import matplotlib.pyplot as plt
import os

if os.name == 'nt':
    plt.rc('font', family='Malgun Gothic')
elif os.name == 'posix':
    plt.rc('font', family='AllieGothic')
#
# plt.rc('axes', unicode_minus=False)

# month_data.sum().plot()
# plt.show()
#
# month_data.mean().plot()
# plt.show()

# plt.hist(analyze_data['total_amount'])
# plt.show()

# plt.hist(analyze_data['total_amount'], bins = 21)
# plt.show()

# pre_data = pd.pivot_table(
#     analyze_data,
#     index = 'order_accept_month',
#     columns = 'narrow_area',
#     values = 'total_amount',
#     aggfunc = 'mean'
# )

# print(pre_data)

# regions = ['서울', '부산', '대전', '광주', '세종', '경기남부', '경기북부']
#
# for region in regions:
#     plt.plot(pre_data.index, pre_data[region], label = region)
#
# plt.legend()
# plt.hist()
# plt.show()

## 4. 클러스터링을 위한 데이터 가공
store_clustering = analyze_data.groupby('store_id')['total_amount'].agg([
    'size', 'mean', 'median', 'max', 'min'])
store_clustering.reset_index(inplace=True, drop = True)

# print(len(store_clustering))
# print(store_clustering.head())

import seaborn as sns
# hexbin = sns.jointplot(x = 'mean', y = 'size', data = store_clustering, kind = 'hex')

## 5. 클러스터링을 이용해 매장을 그룹화
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# 표준화
sc = StandardScaler()
store_clustering_sc = sc.fit_transform(store_clustering)

# K-means 적용(4개 클러스터로 구분)
kmeans = KMeans(n_clusters = 4, random_state = 0)
clusters = kmeans.fit(store_clustering_sc)
# 수행결과 저장
store_clustering['cluster'] = clusters.labels_

# print(store_clustering['cluster'].unique())
# print(store_clustering)

# store_clustering.columns = ['월 건수', '월 평균', '월 중앙',
#                             '월 최댓값', '월 최솟값', 'cluster']
# s_cluster_mean = store_clustering.groupby('cluster').mean()
# print(s_cluster_mean)

## t-SNE로 시각화
from sklearn.manifold import TSNE
tsne = TSNE(n_components = 2, random_state = 0)
x = tsne.fit_transform(store_clustering_sc)
tsne_df = pd.DataFrame(x)
tsne_df['cluster'] = store_clustering['cluster']
tsne_df.columns = ['axis_0', 'axis_1', 'cluster']
# print(tsne_df.head())

# tsne_graph = sns.scatterplot(x = 'axis_0', y = 'axis_1', hue = 'cluster', data = tsne_df)
sns.scatterplot(x = 'axis_0', y = 'axis_1', hue = 'cluster', data = tsne_df)

# 그래프의 제목과 축 레이블을 추가합니다.
plt.title('t-SNE Scatter Plot')
plt.xlabel('Axis 0')
plt.ylabel('Axis 1')

# 범례를 추가합니다.
plt.legend(title='Cluster', loc='best')

# 그래프를 표시합니다.
# plt.show()