import numpy as np
import pandas as pd
import sklearn

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

file_path = r'C:\Users\jstco\Downloads\6768\csv'

iris = pd.read_csv(file_path + '/iris.csv')
# iris.info()

X = iris.drop('species', axis = 1)
y = iris['species']
# print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# print(X_train.shape, X_test.shape)

dt = DecisionTreeClassifier(random_state = 42)
dt.fit(X_train, y_train)

pred = dt.predict(X_test)

## 분석모델 성능 평가방법
from sklearn.metrics import accuracy_score
# acc = accuracy_score(y_test, pred)
# print(acc)

from sklearn.metrics import confusion_matrix
# con_matrix = confusion_matrix(y_test, pred)
# print(con_matrix)

from sklearn.metrics import classification_report

# rpt = classification_report(y_test, pred)
# print(rpt)

### 데이터분석

## 지도학습 - 분류

# 의사결정나무
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split

titanic = pd.read_csv(file_path + '/titanic.csv')
# titanic.info()

# 나이의 결측치를 평균 값으로 대치하시오
age_mean = titanic['Age'].mean()
titanic['Age'].fillna(age_mean, inplace = True)

# Embarked의 결측값을 최빈값으로 입력하시오
embarked_mode = titanic['Embarked'].mode()[0]
titanic['Embarked'].fillna(embarked_mode, inplace = True)

# sex 데이터를 0, 1 레이블인코딩
from sklearn.preprocessing import LabelEncoder

titanic['Sex'] = LabelEncoder().fit_transform(titanic['Sex'])

# Embarked 데이터 레이블 인코딩
titanic['Embarked'] = LabelEncoder().fit_transform(titanic['Embarked'])

# SibSp, Parch 값을 합산하여 FamilySize로 생성
titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch']

# print(titanic.head())


# 분석 데이터셋 준비
X = titanic[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize']]
y = titanic['Survived']

# print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

dt = DecisionTreeClassifier(random_state = 11)
dt.fit(X_train, y_train)

pred = dt.predict(X_test)
# print(pred)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

acc = accuracy_score(y_test, pred)
mat = confusion_matrix(y_test, pred)
rpt = classification_report(y_test, pred)

# print(acc)
# print(mat)
# print(rpt)


## KNN 분류
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# print(iris.describe())

scaler = MinMaxScaler()

# MMX 정규화
iris_X = iris.drop('species', axis = 1)
# print(iris_X.head())
iris_X = scaler.fit_transform(iris_X)

columns = iris.columns[:4]
iris_X = pd.DataFrame(iris_X, columns = columns)
# print(iris_X.head())

X = iris_X
y = iris['species']

y = LabelEncoder().fit_transform(iris['species'])

# print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)

# knn = KNeighborsClassifier(n_neighbors = 3)
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)

pred = knn.predict(X_test)
# print(pred)

acc = accuracy_score(y_test, pred)
mat = confusion_matrix(y_test, pred)
rpt = classification_report(y_test, pred)

# print(acc)
# print(mat)
# print(rpt)


## SVM 분류
from sklearn import svm

titanic = pd.read_csv(file_path + '/titanic.csv')

onehot_sex = pd.get_dummies(titanic['Sex'])
onehot_embarked = pd.get_dummies(titanic['Embarked'])
titanic = pd.concat([titanic, onehot_sex, onehot_embarked], axis = 1)
# print(titanic.head())

titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch']

age_mean = titanic['Age'].mean()
age_mode = titanic['Embarked'].mode()[0]

titanic['Age'].fillna(age_mean, inplace = True)
titanic['Embarked'].fillna(age_mode, inplace = True)

X = titanic[['Pclass', 'Age', 'Fare', 'FamilySize', 'female', 'male', 'C', 'Q', 'S']]
y = titanic['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 10)

sv = svm.SVC(kernel = 'rbf')
sv.fit(X_train, y_train)

pred = sv.predict(X_test)

# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report

acc = accuracy_score(y_test, pred)
mat = confusion_matrix(y_test, pred)
rpt = classification_report(y_test, pred)

# print(acc)
# print(mat)
# print(rpt)


## 로지스틱 회귀 분류 문제
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# iris.info()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

iris_X = iris.drop('species', axis = 1)
# print(iris_X.head())

iris_X = scaler.fit_transform(iris_X)
iris_X = pd.DataFrame(iris_X)
# print(iris_X.head())

y = iris['species']

X_train, X_test, y_train, y_test = train_test_split(iris_X, y, test_size = 0.2, random_state = 11)

lr = LogisticRegression()
lr.fit(X_train, y_train)

pred = lr.predict(X_test)
# print(pred)

# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report

acc = accuracy_score(y_test, pred)
mat = confusion_matrix(y_test, pred)
rpt = classification_report(y_test, pred)

# print(acc)
# print(mat)
# print(rpt)

## 랜덤포레스트
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

titanic = pd.read_csv(file_path + '/titanic.csv')

# print(titanic.head())
# print(titanic.describe())

d_mean = titanic['Age'].mean()
titanic['Age'].fillna(d_mean, inplace = True)

d_mode = titanic['Embarked'].mode()[0]
titanic['Embarked'].fillna(d_mode, inplace = True)

# print(titanic[['Embarked', 'Sex']].head())

le = LabelEncoder()

titanic['Sex'] = le.fit_transform(titanic['Sex'])
titanic['Embarked'] = le.fit_transform(titanic['Embarked'])

# print(titanic[['Embarked', 'Sex']].head())
print(titanic.info())

# titanic['FamilySize'] =