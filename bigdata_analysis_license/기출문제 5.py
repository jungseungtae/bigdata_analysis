import numpy as np
import pandas as pd

file_path = r'C:\Users\jstco\Downloads\6768\csv'

## 특정 데이터 구하기
df = pd.read_csv(file_path + '/trash_bag.csv', encoding='cp949')
# df.info()

# print(df['용도'].unique())
# print(df['사용대상'].unique())

df_home_trash = df[(df['용도'] == '음식물쓰레기') & (df['사용대상'] == '가정용')]
# print(df_home_trash)

result_df = df_home_trash[df_home_trash['2L가격'] != 0]
# print(result_df.head())

result = int(result_df['2L가격'].mean())
# print(result)

## BMI 구하기
df = pd.read_csv(file_path + '/BMI.csv')
# df.info()

df['Height_M'] = df['Height'] / 100
df['BMI'] = df['Weight'] / (df['Height_M'] ** 2)
# print(df.head())

normal = len(df[(df['BMI'] >= 18.5) & (df['BMI'] < 23)])
# print(normal)

overweight = len(df[(df['BMI'] >= 23) & (df['BMI'] < 25)])
# print(overweight)

result = abs(normal - overweight)
# print(result)


df = pd.read_csv(file_path + '/students.csv', encoding = 'cp949')
# df.info()
# print(df.head())

df['순전입학생수'] = df['총 전입학생'] - df['총 전출학생']
# print(df.head())

result_df = df.groupby(['학교'])[['순전입학생수', '전체학생 수']].sum()
# print(result_df)

max_value = result_df['순전입학생수'].max()
max_row = result_df[result_df['순전입학생수'] == max_value]
result = max_row['전체학생 수'].values[0]
# print(result)


## type 2 ##
X_test = pd.read_csv(file_path + '/used_car_X_test.csv')
X_train = pd.read_csv(file_path + '/used_car_X_train.csv')
y_train = pd.read_csv(file_path + '/used_car_y_train.csv')

# print(X_train.head())
# X_train.info()

## 명목형
X_train_word = X_train.select_dtypes('object')
X_test_word = X_test.select_dtypes('object')
# X_train_word.info()

## 수치형
X_train_num = X_train.select_dtypes(['int64', 'float64'])
X_test_num = X_test.select_dtypes(['int64', 'float64'])

X_train_num.drop('id', axis = 1, inplace = True)
X_test_num.drop('id', axis = 1, inplace = True)

# X_train_num.info()


## 스케일링
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train_num_scale = scaler.fit_transform(X_train_num)
X_test_num_scale = scaler.fit_transform(X_test_num)

df_train_num = pd.DataFrame(X_train_num_scale, columns = X_train_num.columns)
df_test_num = pd.DataFrame(X_test_num_scale, columns = X_test_num.columns)

df_train_word = pd.get_dummies(X_train_word)
df_test_word = pd.get_dummies(X_test_word)

## 데이터프레임 컬럼확인
train_col = set(df_train_word.columns)
test_col = set(df_test_word.columns)
# print(train_col)
# print(test_col)
# print(train_col - test_col)
# print(test_col - train_col) # 0

miss_in_test_col = train_col - test_col
# print(miss_in_test_col)

for col in miss_in_test_col:
    df_test_word[col] = 0

# print(df_test_word.info())

df_train = pd.concat([df_train_num, df_train_word], axis = 1)
df_test = pd.concat([df_test_num, df_test_word], axis = 1)
# df_train.info()

## 모델학습
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

model = xgb.XGBRegressor(random_state = 77)

X_train, X_val, y_train_t, y_val = train_test_split(df_train.values, y_train['price'].values, test_size = 0.3)

model.fit(X_train, y_train_t)

y_pred = model.predict(X_val)
# print(y_pred)
# print(np.sqrt(mean_squared_error(y_val, y_pred)))

y_pred = model.predict(df_test)
df = pd.DataFrame(X_test['id'], columns = ['id'])
df['price'] = y_pred
# print(df.head())

# df.to_csv(file_path + '/result.csv', index = False)