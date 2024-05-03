
### 1유형 ###

import pandas as pd
import numpy as np

file_path = r'C:\Users\jstco\Downloads\6768\csv'

df = pd.read_csv(file_path + '/airquality.csv')
# print(df.head())
# print(df.info())

## 결측치 처리
# 37
# print(df['Ozone'].isnull().sum())
Ozone_mean = df['Ozone'].mean()

## 0처리
df['Ozone'].fillna(0, inplace = True)

## 평균처리
# df['Ozone'].fillna(Ozone_mean, inplace = True)

# print(df['Ozone'].isnull().sum())

Ozone_mean2 = df['Ozone'].mean()

# print(f'널 처리 전 : {Ozone_mean}', f'널 처리 후 : {Ozone_mean2}')
# print(f'{Ozone_mean - Ozone_mean2}')

## Min-Max 표준화, Z 표준화 변환 뒤 컬럼 추가

# Min-Max 표준화
Min = np.min(df['Wind'])
Max = np.max(df['Wind'])
df['min_max'] = round((df['Wind'] - Min) / (Max - Min), 2)

# Z 표준화
Mean = np.mean(df['Wind'])
Std = np.std(df['Wind'])
df['z'] = round((df['Wind'] - Mean) / Std, 2)

# print(df.head())
# print(df.describe())

## 월 평균기온 구하기
mean_temp = df.groupby('Month')['Temp'].mean()
# print(mean_temp)



### 2유형 ###

import sklearn

ploan = pd.read_csv(file_path + '/Bank_Personal_Loan_Modeling.csv')
# print(ploan.shape)
# print(ploan.info())

# 결측치 삭제, 변수 제거
ploan_processed = ploan.dropna().drop(['ID', 'ZIP Code'], axis = 1, inplace = False)

X = ploan_processed.drop(['Personal Loan'], axis = 1)
y = ploan_processed['Personal Loan']

from sklearn.model_selection import train_test_split

# 데이터 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, train_size = 0.7, test_size = 0.3, random_state = 1234)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

import sklearn.preprocessing as preprocessing

preprocessor = preprocessing.Normalizer()

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 25)


# K-근접이웃 분류기
for n_neighbors in neighbors_settings:
    # 모델생성
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    # 훈련세트의 정확도
    training_accuracy.append(clf.score(X_train, y_train))
    # 일반화 정확도
    test_accuracy.append(clf.score(X_test, y_test))

# print(test_accuracy)

import scipy.stats as stats

X = df['Temp'].mean()
# print(round(X, 2))

t_score, p_value = stats.ttest_1samp(df['Temp'], 77)
print(round(t_score, 2))
print(round(p_value, 4))

pv = round(p_value, 4)
print(pv)

if pv < 0.05:
    print('기각')
else:
    print('채택')