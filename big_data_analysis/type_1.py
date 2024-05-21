import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

file_path = r'C:\Users\jstco\Downloads\6768\csv'

titanic = pd.read_csv(file_path + '/titanic.csv')

# print(df.head())
# print(df.info())
# print(df.describe())

## 1. 데이터 탐색(EDA)

# 좌석 클래스별
grouped = titanic.groupby('Pclass')
# print(grouped.size())

# plt.hist(titanic['Fare'])
# plt.show()

# 요금별 생존
data_0 = titanic[titanic['Survived'] == 0]['Fare']
data_1 = titanic[titanic['Survived'] == 1]['Fare']

# fig, ax = plt.subplots()
# ax.boxplot([data_0, data_1])
# plt.show()


# 성별 생존율
grouped_sex = titanic.groupby('Sex')
# print(grouped_sex.size())

survived_female = titanic[titanic['Sex'] == 'female']['Survived']
survived_male = titanic[titanic['Sex'] == 'male']['Survived']

grouped_female_survived = pd.DataFrame(survived_female).groupby('Survived')
# print(grouped_female_survived.size())

grouped_male_survived = pd.DataFrame(survived_male).groupby('Survived')
# print(grouped_male_survived.size())


## 2. 데이터 전처리

# 표준정규분포
meat_consumption_kor = 5 * np.random.randn(100) + 53.9
meat_consumption_jap = 5 * np.random.randn(100) + 32.7

meat_consumption = pd.DataFrame({'korean' : meat_consumption_kor, 'japanese' : meat_consumption_jap})
# print(meat_consumtion)

# plt.hist(meat_consumption_kor)
# plt.xlabel('Korean')
# plt.show()
#
# plt.hist(meat_consumption_jap)
# plt.xlabel('Japanese')
# plt.show()

# fig, axs = plt.subplots(1, 2, figsize = (10, 5))
#
# axs[0].hist(meat_consumption_kor, bins = 30)
# axs[0].set_xlabel('Korean')
#
# axs[1].hist(meat_consumption_kor, bins = 30)
# axs[1].set_xlabel('Japanese')
#
# plt.tight_layout()
# plt.show()

mean_kor = meat_consumption_kor.mean()
mean_jap = meat_consumption_jap.mean()

std_kor = meat_consumption_kor.std()
std_jap = meat_consumption_jap.std()

# np.zscore
# print(meat_consumption.head())
# meat_consumption['z_kor'] = (meat_consumption['korean'] - mean_kor) / std_kor
# meat_consumption['z_jap'] = (meat_consumption['japanese'] - mean_jap) / std_jap

# print(meat_consumption.head())

# Z-score
import scipy.stats as ss

# meat_consumption['z_kor'] = ss.zscore(meat_consumption_kor)
# meat_consumption['z_jap'] = ss.zscore(meat_consumption_jap)

# print(meat_consumption.head())


# fig, axs = plt.subplots(1, 2, figsize = (10, 5))
#
# axs[0].hist(meat_consumption['z_kor'])
# axs[0].set_xlabel = 'Koreans'
#
# axs[1].hist(meat_consumption['z_jap'])
# axs[1].set_xlabel = 'japanese'

# plt.show()


# sklearn
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# meat_consumption['z_kor'] = scaler.fit_transform(meat_consumption[['korean']])
# meat_consumption['z_jap'] = scaler.fit_transform(meat_consumption[['japanese']])
#
# print(meat_consumption.head())

# min-max 정규화
# print(meat_consumption.head())

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
meat_consumption['kor_mm'] = scaler.fit_transform(meat_consumption[['korean']])
meat_consumption['jap_mm'] = scaler.fit_transform(meat_consumption[['japanese']])
# print(meat_consumption.head())

test_score = pd.read_csv(file_path + '/df_sample.csv')
# print(test_score)

fin_min = np.min(test_score['기말'])
fin_max = np.max(test_score['기말'])

test_score['MM_scaler'] = (test_score['기말'] - fin_min) / ((fin_max - fin_min))
# print(test_score.head())


## 왜도
judge_ratings = pd.read_csv(file_path + '/USJudgeRatings.csv')
# print(judge_ratings.head())

import scipy.stats as ss
# print(ss.skew(judge_ratings['CONT']))
# print(ss.skew(judge_ratings['PHYS']))

# 왜도 로그값 변환 - 데이터를 대칭적인 분포로 만듬
judge_ratings['log_cont'] = np.log10(judge_ratings['CONT'])
judge_ratings['log_phys'] = np.log10(np.max(judge_ratings['PHYS'] + 1) - judge_ratings['PHYS'])
# print(judge_ratings.head())


## 범주화
names = ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Hannah', 'Ian', 'Jack']

np.random.seed(42)
# 랜덤으로 점수 생성
scores = np.random.randint(0, 101, size=10)

# 데이터 프레임 생성
data = {'이름': names, '점수': scores}
df = pd.DataFrame(data)

# print(df.head())
# print(np.mean(df['점수']))

# plt.hist(df['점수'], bins = 5, range = [50, 100], rwidth = 0.9)
# plt.show()

# 등급주기
df['grade_cut'] = pd.qcut(x = df['점수'], q = 5, labels = ['F', 'D', 'C', 'B', 'A'])
# print(df.head())

## PCA 주성분분석
iris = pd.read_csv(file_path + '/iris.csv')
# iris.info()

df = iris.drop(['species'], axis = 1)
df_species = iris['species']

# print(df.head())

sclaer = StandardScaler()
x = scaler.fit_transform(df)

df_scaler = pd.DataFrame(x, columns = df.columns)
# print(df_scaler.head())

# score
from sklearn.decomposition import PCA
pca = PCA(n_components = 4)
p_score = pca.fit_transform(df_scaler)
# print(p_score.shape)
# print(pca.explained_variance_ratio_)


# 결측치 처리
x = [14, 15, 13, 14, None, None, 19, 11, 12, 18]
test_score['토론'] = x
# test_score.info()

# print(titanic.isnull().sum())

# 결측치 대체
# titanic.info()
titanic_dropna = titanic.dropna(axis = 0)
# titanic_dropna.info()

# 평균값으로 대체
age_mean = titanic['Age'].mean()

# inplace = True : 새로운 객체를 생성하는 것이 아닌 현재 데이터셋을 변환
titanic['Age'].fillna(age_mean, inplace = True)
# print(titanic['Age'].isnull().sum())

# 최빈값으로 대체 mode
age_mode = titanic['Age'].mode()
# print(age_mode)
# titanic['Age'].fillna(age_mode[0], inplace = True)
# print(titanic['Age'].isnull().sum())

# 인접한 값, 결측치 행의 이전 값으로 대체
# titanic['Embarked'].fillna(method = 'ffill', inplace = True)
# print(titanic['Embarked'].isnull().sum())

# 그룹별로 평균값 대체 : 속성별로 평균값이 다르므로 그룹별 평균을 구하여 결측치를 대체
sex_age_mean = titanic.groupby('Sex')['Age'].mean()
class_age_mean = titanic.groupby('Pclass')['Age'].mean()

# print(sex_age_mean, class_age_mean)

# print(titanic.tail())

# titanic['Age'].fillna(titanic.groupby('Pclass')['Age'].transform('mean'), inplace = True)
# print(titanic.tail())


# 이상치 처리
age_quantiles = titanic['Age'].quantile([0.25, 0.5, 0.75, 1.0])
titanic['Age_Quartiles'] = pd.cut(titanic['Age'], bins = age_quantiles.values, labels = ['Q1', 'Q2', 'Q3'])
# print(titanic[['Age', 'Age_Quartiles']])

# 4분위수 구하기
q1 = titanic['Age'].describe()['25%']
q2 = titanic['Age'].describe()['50%']
q3 = titanic['Age'].describe()['75%']
IQR = q3 - q1

# print(q1, q2, q3, IQR)

upper = titanic['Age'] > (q3 + IQR * 1.5)
titanic['upper'] = upper
under = titanic['Age'] < (q1 - IQR * 1.5)
titanic['under'] = under

up_true = titanic[titanic['upper']]
# print(up_true)

## 평활화
lynx = pd.read_csv(file_path + '/lynx.csv')
# lynx.info()
# print(lynx.describe())

lynx['sma'] = lynx['value'].rolling(10).mean()

# plt.plot(lynx['value'])
# plt.plot(lynx['sma'])
# plt.show()

lynx['ewm'] = lynx['value'].ewm(10).mean()

# plt.plot(lynx['ewm'])
# plt.show()


