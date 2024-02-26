import numpy as np
import pandas as pd
import sklearn

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from global_functions import DataAnalyzer

pd.set_option('display.max_columns', None)
file_path = r'C:\Users\jstco\Downloads\6768\csv'

## ch 3.2 분석과정 이해
# df = pd.read_csv(file_path + '/titanic.csv')
#
# # analyzer = DataAnalyzer(df)
# # analyzer.summarize_basic()
#
# # ---------- COLUMNS --------------------------------------------------
# # Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
# #        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
# #       dtype='object')
#
# # Age 결측치 평균 대치
# d_mean = df['Age'].mean()
# df['Age'].fillna(d_mean, inplace = True)
#
# # Embarked(승선) 결측치 최빈값으로 대치
# d_mode = df['Embarked'].mode()[0]
# df['Embarked'].fillna(d_mode, inplace = True)
#
# from sklearn.preprocessing import LabelEncoder
#
# # 성별을 0, 1로 변환
# df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
# # print(df.head())
#
# # Embarked 레이블 인코딩
# df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
# df['FamilySize'] = df['SibSp'] + df['Parch']
#
# # print(df.head())
#
# X = df[["Pclass","Sex","Age","Fare","Embarked","FamilySize"]]
# y = df["Survived"]
#
# # print(X.head())
# # print(y.head())
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)
#
# dt = DecisionTreeClassifier(random_state = 11)
# dt.fit(X_train, y_train)
#
# pred = dt.predict(X_test)
#
# from sklearn.metrics import accuracy_score
# acc = accuracy_score(y_test, pred)
# # print(acc)
#
# from sklearn.metrics import confusion_matrix
# mat = confusion_matrix(y_test, pred)
# # print(mat)
#
# from sklearn.metrics import classification_report
# rpt = classification_report(y_test, pred)
# # print(rpt)


###################################

## 2. KNN 분류

###################################

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
#
# df = pd.read_csv(file_path + '/iris.csv')
#
# # analyzer = DataAnalyzer(df)
# # analyzer.summarize_basic()
#
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
#
# col_names = df.columns
#
# # for i in range(0, 4):
# #     print(col_names[i])
#
# for i in range(0, 4):
#     df[[col_names[i]]] = scaler.fit_transform(df[[col_names[i]]])
#
# X = df.iloc[:, :4]
# # print(X)
#
# y = df.iloc[:, 4]
# # print(y)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)
#
# # print(X_train.shape)     # 학습 데이터(독립변수)
# # print(X_test.shape)      # 테스트 데이터(독립변수)
# # print(y_train.shape)     # 학습 데이터(종속변수)
# # print(y_test.shape)      # 테스트 데이터(종속변수)
#
#
# # k 근접이웃 객체생성
# # knn = KNeighborsClassifier(n_neighbors = 3)
# knn = KNeighborsClassifier(n_neighbors = 5)
# knn.fit(X_train, y_train)
#
# pred = knn.predict(X_test)
#
# from sklearn.metrics import accuracy_score
# acc = accuracy_score(y_test, pred)
# print(acc)
#
# from sklearn.metrics import confusion_matrix
# mat = confusion_matrix(y_test, pred)
# print(mat)
#
# from sklearn.metrics import classification_report
# rpt = classification_report(y_test, pred)
# print(rpt)


###################################

## 3. SVM 분류

###################################

# from sklearn import svm
#
# df = pd.read_csv(file_path + '/titanic.csv')
#
# # analysis = DataAnalyzer(df)
# # analysis.summarize_basic()
#
# # 결측치 처리
# d_mean = df['Age'].mean()
# df['Age'].fillna(d_mean, inplace = True)
#
# d_mode = df['Embarked'].mode()[0]
# df['Embarked'].fillna(d_mode, inplace = True)
#
# df['FamilySize'] = df['SibSp'] + df['Parch']
#
# onehot_sex = pd.get_dummies(df['Sex'])
# df = pd.concat([df, onehot_sex], axis = 1)
#
# onehot_embarked = pd.get_dummies(df['Embarked'])
# df = pd.concat([df, onehot_embarked], axis = 1)
#
# # print(df.head())
#
# X = df[["Pclass","Age","Fare","FamilySize","female","male","C","Q","S"]]
# y = df["Survived"]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 10)
#
# # print(X_train.shape)     # 학습 데이터(독립변수)
# # print(X_test.shape)      # 테스트 데이터(독립변수)
# # print(y_train.shape)     # 학습 데이터(종속변수)
# # print(y_test.shape)      # 테스트 데이터(종속변수)
#
# sv = svm.SVC(kernel = 'rbf')
# sv.fit(X_train, y_train)
#
# pred = sv.predict(X_test)
#
# from sklearn.metrics import accuracy_score
# acc = accuracy_score(y_test, pred)
# # print(acc)
#
# from sklearn.metrics import classification_report
# rpt = classification_report(y_test, pred)
# # print(rpt)


###################################

## 4. 로지스틱회귀 분류

###################################

# from sklearn.linear_model import LogisticRegression
#
# df = pd.read_csv(file_path + '/iris.csv')
# # print(df.info)
# # print(df.describe())
#
# # 각 독립변수별 NX 정규화
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
#
# df[["sepal_length"]] = scaler.fit_transform(df[["sepal_length"]])
# df[["sepal_width"]] = scaler.fit_transform(df[["sepal_width"]])
# df[["petal_length"]] = scaler.fit_transform(df[["petal_length"]])
# df[["petal_width"]] = scaler.fit_transform(df[["petal_width"]])
#
# # 분석 데이터셋 준비
# X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
# y = df['species']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)
#
# # print(X_train.shape)
# # print(X_test.shape)
# # print(y_train.shape)
# # print(y_test.shape)
#
# # print(X_train.head())
# # print(y_train.head())
#
# # Logistic regression 객체생성
# lr = LogisticRegression()
# lr.fit(X_train, y_train)
#
# pred = lr.predict(X_test)
#
# from sklearn.metrics import accuracy_score
# acc = accuracy_score(y_test, pred)
# # print(acc)


###################################

## 5. 랜덤 포레스트 분류

###################################

import numpy as np
import pandas as pd
import sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv(file_path + '/titanic.csv')
# print(df.head())
# print(df.info())

# 결측치 = 평균
d_mean = df['Age'].mean()
df['Age'].fillna(d_mean, inplace=True)

# 결측치 = 최빈값
d_moed = df['Embarked'].mode()[0]
df['Embarked'].fillna(d_moed, inplace = True)

# 성별 인코딩
from sklearn.preprocessing import LabelEncoder
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])

# Embarked(티켓 클래스)
from sklearn.preprocessing import LabelEncoder
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

# SibSp + Parch = Family size
df['Family'] = df['SibSp'] + df['Parch']

# 데이터셋 준비
X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Family']]
y = df['Survived']
# print(X.head())
# print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)
# print(X_train.head())
# print(y_test.shape)

dt = DecisionTreeClassifier(random_state= 11)
dt.fit(X_train, y_train)

pred = dt.predict(X_test)
# print(pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, pred)
# print(acc)

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test,pred)
print(mat)