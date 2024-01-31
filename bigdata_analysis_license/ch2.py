###########################################

## 빅데이터분석기사 실기 ##

## Chapter 2. 데이터 전처리

###########################################


## 2. 데이터 변환

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
file_path = r'C:\Users\jstco\Downloads\6768\csv'

# ## 국적별 성인 1000명 육류 소비량 데이터 생성
# meat_consumption_korean = 5 * np.random.randn(1000) + 53.9
# meat_consumption_japan = 4 * np.random.randn(1000) + 32.7
#
# # df
# meat_consumption = pd.DataFrame({'koreans': meat_consumption_korean, 'japanese': meat_consumption_japan})
#
# # print(meat_consumption.head())
#
# ## histogram
# # plt.hist(meat_consumption_japan)
# # plt.xlabel('Japanese')
# # plt.show()
#
# # plt.hist(meat_consumption_korean)
# # plt.xlabel('koreans')
# # plt.show()
#
# ## 표준정규화함수를 이용하여 표준화
# import scipy.stats as ss
#
# meat_consumption['Koreans Standardzation'] = ss.zscore(meat_consumption_korean)
# meat_consumption['japanese Standardzation'] = ss.zscore(meat_consumption_japan)
#
# # print(meat_consumption.head())
#
# ## 수식을 직접 입력하여 표준화하기
# # (데이터 - 평균) / 표준편차
# meat_consumption['Koreans Standardzation2'] = \
#     (meat_consumption_korean - np.mean(meat_consumption_korean)) / np.std(meat_consumption_korean)
#
# meat_consumption['japanese Standardzation2'] = \
#     (meat_consumption_japan - np.mean(meat_consumption_japan)) / np.std(meat_consumption_japan)
#
# # print(meat_consumption.head())
#
# ## 표준화 데이터 히스토그램
# # plt.hist(meat_consumption['japanese Standardzation'])
# # plt.xlabel('Japanese Standardzation')
# # plt.show()
# #
# # plt.hist(meat_consumption['Koreans Standardzation'])
# # plt.xlabel('Koreans Standardzation')
# # plt.show()
#
# ## 사이킷런 스케일러를 이용한 정규화
# from sklearn.preprocessing import StandardScaler
#
# scaler = StandardScaler()
#
# meat_consumption['Koreans Standardzation3'] = scaler.fit_transform(meat_consumption[['koreans']])
# meat_consumption['Japanese Standardzation3'] = scaler.fit_transform(meat_consumption[['japanese']])
#
# # print(meat_consumption[['Koreans Standardzation3', 'Japanese Standardzation3']].head()
#
# ## Min-Max 추가
# from sklearn.preprocessing import MinMaxScaler
#
# scaler = MinMaxScaler()
#
# meat_consumption['Koreans Min-Max'] = scaler.fit_transform(meat_consumption[['koreans']])
# meat_consumption['Japanese Min-Max'] = scaler.fit_transform(meat_consumption[['japanese']])
#
# # print(meat_consumption[['Koreans Min-Max', 'Japanese Min-Max']].head())
#
# # Min-Max 수식으로 변환하기
# # 컬럼 최대값 최솟값을 추출하여 (데이터 - 최솟값) / (최대값 - 최솟값)
# Min = np.min(meat_consumption_korean)
# Max = np.max(meat_consumption_korean)
# meat_consumption['koreans MM2'] = (meat_consumption['koreans'] - Min) / (Max - Min)
#
# Min = np.min(meat_consumption_japan)
# Max = np.max(meat_consumption_japan)
# meat_consumption['japanese MM2'] = (meat_consumption['japanese'] - Min) / (Max - Min)
#
# # print(meat_consumption[['koreans MM2', 'japanese MM2']].head())
#
# ## 미국 판사 데이터
# # CONT: 변호사와 판사의 연락 횟수
# # INTG: 판사의 청렴성
# # DMNR: 판사의 판결 능력
# # DILG: 판사의 의사 소통 능력
# # CFMG: 판사의 법률 지식
# # DECI: 판사의 판결의 일관성
# # PREP: 판사의 준비성
# # FAMI: 판사의 법률 가족
# # ORAL: 판사의 구두 변론 능력
# # WRIT: 판사의 글쓰기 능력
# # PHYS: 판사의 외모
# # RTEN: 판사의 판결 속도
#

# df = pd.read_csv(file_path + '/USJudgeRatings.csv')
#
# # print(df.head())
#
# # print(ss.skew(df['CONT']))
# # print(ss.skew(df['PHYS']))
#
# df['CONT1'] = np.log(df['CONT'])
# df['PHYS1'] = np.log(np.max(df['PHYS'] + 1) - df['PHYS'])
# # print(df.head())
#
# # print(ss.skew(df['CONT1']))
# # print(ss.skew(df['PHYS1']))
#
#
# # 수학점수 (Math_score)
# data = [["철수",52], ["영희",92], ["미영",84], ["시완",71], ["미경",65], ["영환",81], ["숙경",66], ["부영",77], ["민섭",73], ["보연",74]]
# df = pd.DataFrame(data, columns=['name', 'score'])
#
# # print(np.mean(df['score']))
# #
# # plt.hist(df['score'], bins = 5, range = [50, 100], rwidth = 0.9)
# # plt.show()
# # print(df)
#
# df['grade'] = ''
# df.loc[(df['score'] < 60), 'grade'] = 'F'
# df.loc[(df['score'] >= 60) & (df['score'] < 70), 'grade'] = 'D'
# df.loc[(df['score'] >= 70) & (df['score'] < 80), 'grade'] = 'C'
# df.loc[(df['score'] >= 80) & (df['score'] < 90), 'grade'] = 'B'
# df.loc[(df['score'] >= 90) & (df['score'] <= 100), 'grade'] = 'A'
#
# # print(df.head())
#
# df['grade'] = pd.cut(x = df['score'], bins = [0, 60, 70, 80, 90, 100], labels = ['F', 'D', 'C', 'B', 'A'], include_lowest = True)
# # print(df)
#
# df['grade_qcut'] = pd.qcut(x = df['score'], q = 5, labels = ['F', 'D', 'C', 'B', 'A'],)
# # print(df)


## 3. 차원축소 PCA(Principal Component Analysis, 주성분분석)
iris = pd.read_csv(file_path + '/iris.csv')
print(iris.head())

df = iris.drop(['species'], axis = 1)
df_species = iris['species']

# print(df.head())

from sklearn.preprocessing import StandardScaler
col = df.columns

for i in range(3):
    df[col[i]] = StandardScaler().fit_transform(df[[col[i]]])

from sklearn.decomposition import PCA
pca = PCA(n_components = 4)
p_score = pca.fit_transform(df)
# print(p_score.shape)
# print(pca.explained_variance_ratio_)