import numpy as np
import pandas as pd
import sklearn

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from global_functions import DataAnalyzer

pd.set_option('display.max_columns', None)
file_path = r'C:\Users\jstco\Downloads\6768\csv'


## ch 3. 분석과정 이해
df = pd.read_csv(file_path + '/iris.csv')

# analyzer = DataAnalyzer(df)
# analyzer.summarize_basic()    # info, describe, head, columns, type, shape, null

# 데이터 변환
df['species'].replace({'setosa': 0,
                       'versicolor': 1,
                       'virginica': 2},
                      inplace = True)
# print(df.head())

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 11)

# print(X_train.shape, X_test.shape)
# print(y_train.shape, y_test.shape)

dt = DecisionTreeClassifier(random_state = 11)
dt.fit(X_train, y_train)

pred = dt.predict(X_test)
# print(pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, pred)
# print(acc)

## 3. 분석모델 성능 평가방법
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, pred)

from sklearn.metrics import classification_report
rpt = classification_report(y_test, pred)
# print(rpt)