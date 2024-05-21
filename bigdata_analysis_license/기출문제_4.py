import numpy as np
import pandas as pd

file_path = r'C:\Users\jstco\Downloads\6768\csv'


### 1유형 ###
lst = [10, 11, 11.2, 13, 15.5, 18, 19.8, 20, 31, 33, 39.5, 42]

# 1, 3사분위수
q1 = np.percentile(lst, 25)
q3 = np.percentile(lst, 75)

diff = abs(q1 - q3)
# print(diff)

result = int(diff)
# print(result)


df = pd.read_csv(file_path + '/facebook.csv')
# df.info()
# print(df.head())

df['positive'] = (df['num_loves'] + df['num_wows']) / df['num_reactions']
# print(df['positive'].describe())

result_df = df[(df['positive'] > 0.4) & (df['positive'] < 0.5) & (df['status_type'] == 'video')]
# result_df = df[(df['positive'] > 0.4) & (df['positive'] < 0.5) & (df['status_type'] == 'video')]
# print(result_df)
result = len(result_df)
# print(result)



df = pd.read_csv(file_path + '/netflix.csv')
# df.info()

# print(df['date_added'])

result_df = df[df['date_added'].str.contains('January') & df['date_added'].str.contains('2018')]
# print(result_df)

result_df_UK = result_df[result_df['country'] == 'United Kingdom']
# print(result_df_UK)

result = len(result_df_UK)
# print(result)


### 2유형 ###

# 데이터 불러오기
X_test = pd.read_csv(file_path + '/CS_Seg_X_test.csv')
X_train = pd.read_csv(file_path + '/CS_Seg_X_train.csv')
y_train = pd.read_csv(file_path + '/CS_Seg_y_train.csv')

# 데이터 나누기
# 범주형 데이터
X_train_word = X_train[['Gender','Ever_Married','Graduated','Profession' ,'Spending_Score']]
X_test_word = X_test[['Gender','Ever_Married','Graduated','Profession' ,'Spending_Score']]
print(X_train_word)

# 수치형 데이터
X_train_num= X_train.drop(columns=['ID','Gender','Ever_Married','Graduated','Profession','Spending_Score'])
X_test_num= X_test.drop(columns=['ID','Gender','Ever_Married','Graduated','Profession','Spending_Score'])
print(X_train_num)

# 스케일링
from sklearn.preprocessing import MinMaxScaler

# MinMax 스케일러 생성
scaler = MinMaxScaler()

# 선택한 특성에 MinMax 스케일러를 적용하고 데이터 변환
X_train_num_scale = scaler.fit_transform(X_train_num)
X_test_num_scale = scaler.transform(X_test_num)

# 데이터 프레임 설정
df_train_num = pd.DataFrame(X_train_num_scale , columns = X_train_num.columns)
df_test_num = pd.DataFrame(X_test_num_scale , columns = X_test_num.columns)

# 원핫 인코딩
df_train_word = pd.get_dummies(X_train_word)
df_test_word = pd.get_dummies(X_test_word)

# 데이터 결합
df_train = pd.concat([df_train_num, df_train_word], axis = 1)
df_test = pd.concat([df_test_num, df_train_word], axis = 1)
print(df_train)


from pandas.core.common import random_state
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# target 값 변경
y_train[y_train['Segmentation'] =='A'] = 0
y_train[y_train['Segmentation'] =='B'] = 1
y_train[y_train['Segmentation'] =='C'] = 2
y_train[y_train['Segmentation'] =='D'] = 3

# 모델 생성
model = xgb.XGBClassifier(random_state=77)

# train, validation 데이터 설정
X_train, X_val, y_train_t, y_val = train_test_split(df_train.values, y_train['Segmentation'].values, test_size=0.3)

# 모델 학습
model.fit(X_train, y_train_t)

# vaidation 데이터로 성능 평가
y_pred = model.predict(X_val)
# print(classification_report(y_val.astype(int), y_pred))

# 예측
y_pred = model.predict(df_test)
df = pd.DataFrame(y_train['ID'], columns=['ID'])
df['Segmentation'] = y_pred

print(df_test.shape)

# Segmentation 데이터를 숫자에서 문자로 수정
df['Segmentation'] = df['Segmentation'].map({0 : 'A', 1 : 'B', 2 : 'C', 3 : 'D'})
print(df.head())