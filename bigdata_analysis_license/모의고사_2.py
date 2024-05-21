### 제 1유형 ###

from global_functions import DataAnalyzer
import sys
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import pearsonr
from itertools import combinations
from statsmodels.stats.proportion import proportion_confint

# pd.set_option('display.max_columns', None)

file_path = r'C:\Users\jstco\Downloads\6768\csv'
df = pd.read_csv(file_path + '/HR-Employee-Attrition.csv')
analysis = DataAnalyzer(df)
# analysis.summarize_basic()

from sklearn.preprocessing import LabelEncoder

# df['Attrition_numerical'] = LabelEncoder().fit_transform(df['Attrition'])
# df['Attrition_numerical'].value_counts()
# print(df.head())

# 오브젝트 데이터를 범주 데이터로 바꾸고 범주의 갯수를 카운트
# cat_feat = df.select_dtypes('object', 'category').columns.values
# df_cat = df[cat_feat].copy()
# print(df_cat.nunique().sort_values())

# df_cat = df_cat.drop(['Over18'], axis = 1, errors = 'ignore')
# print(df_cat)

# 수치형 컬럼만으로 데이터셋 만들기
num_feat = df.select_dtypes('number').columns.values
df_num = df[num_feat].copy()

corr = df_num.corr(method = 'pearson')
# print(corr.shape)

# 상관계수가 0.9보다 높은 2개의 변수와 상관계수
# for i in range(0, 24):
#     for j in range(i + 1, 25):
#         if(corr.iloc[i, j] >= 0.9):
#             print(i, j, corr.iloc[i, j])

df_num = df_num.drop(['JobLevel'], axis = 1, errors = 'ignore')
# df_num.info()


### 제 2유형 ###

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
from sklearn import metrics
from sklearn.metrics import f1_score

dat = pd.read_csv(file_path + '/Parkinsons.csv')
# print(dat.head(10))

# name 컬럼 제거 후 표준화
dat_processing = dat.drop(['name'], axis = 1, inplace = False)
dat_processing_norm = preprocessing.minmax_scale(dat_processing)
dat_processed = pd.DataFrame(dat_processing_norm)
dat_processed.columns = dat_processing.columns

# 상수항 추가
dat_processed = sm.add_constant(dat_processed, has_constant = 'add')
# print(dat_processed.head(10))


feature_columns = list(dat_processed.columns.difference(['status']))
# print(feature_columns)

X = dat_processed[feature_columns]
y = dat_processed['status'].astype('category')

train_x, test_x, train_y, test_y = train_test_split(X, y, stratify = y, test_size = 0.1, random_state = 2017010500)
model = sm.Logit(train_y, train_x)
results = model.fit(method = 'bfgs', maxiter = 1000)

# print(results.summary())

# cut-off 정의
def cut_off(y, threshold):
    Y = y.copy()
    Y[Y > threshold] = 1
    Y[Y <= threshold] = 0
    return(Y.astype(int))

test_y_pred_prob = results.predict(test_x)
test_y_pred = cut_off(test_y_pred_prob, 0.8)
# print(f1_score(test_y, test_y_pred))


### 제 3유형 ###

