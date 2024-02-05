###########################################

## 빅데이터분석기사 실기 ##

## Chapter 1. 데이터 탐색

###########################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

## 데이터 불러오기
df = pd.read_csv("https://raw.githubusercontent.com/YoungjinBD/dataset/main/titanic.csv")

# print(df.head())
# print(df.info())

## 변수 타입 변환
df['Survived'] = df['Survived'].astype(str)
df['Pclass'] = df['Pclass'].astype(str)

# print(df.info())

# print(df.describe(include = 'all'))
grouped = df.groupby('Pclass')
# print(grouped.size())

# plt.hist(df['Fare'])
# plt.show()

## 사망자와 생존자의 요금 데이터
data_0 = df[df['Survived'] == '0']['Fare']
data_1 = df[df['Survived'] == '1']['Fare']

# fig, ax = plt.subplots()
# ax.boxplot([data_0, data_1])
# plt.show()

grouped = df.groupby('Sex')
print(grouped.size())

data_0 = df[df['Sex'] == 'female']['Survived'] # 여성생존자
grouped = pd.DataFrame(data_0).groupby('Survived')
print('female Survived :')
print(grouped.size())
print('-' * 20)

data_1 = df[df['Sex'] == 'male']['Survived'] # 여성생존자
grouped = pd.DataFrame(data_1).groupby('Survived')
print('male Survived :')
print(grouped.size())
print('-' * 20)