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

