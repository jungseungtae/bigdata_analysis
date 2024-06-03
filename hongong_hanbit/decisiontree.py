import pandas as pd
import numpy as np
from global_functions import DataAnalyzer

## 데이터 불러오기
wine = pd.read_csv('./data/wine.csv')
# print(wine.head())

## 탐색적 데이터탐색
analysis = DataAnalyzer(wine)
# analysis.summarize_basic()

## 데이터타입 변환
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

train_col = wine[['alcohol', 'sugar', 'pH']]

## 데이터 분리
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size = 0.2, random_state = 42)
# print(train_input.shape, test_input.shape)

## 표준화
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)

train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 표준화 후 데이터값
# train_scaled = pd.DataFrame(train_scaled, columns = train_col.columns)
# print(train_scaled.head())

## 로지스틱 회귀분류
from sklearn.linear_model import LogisticRegression

# lr = LogisticRegression()
# lr.fit(train_scaled, train_target)

## 정확도
# print('LogisticRegression train', lr.score(train_scaled, train_target))
# print('LogisticRegression test', lr.score(test_scaled, test_target))

## 기울기, 절편
# print(lr.coef_, lr.intercept_)


## 결정트리
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state = 42)
dt.fit(train_scaled, train_target)

# print('의사결정나무 train score : ', dt.score(train_scaled, train_target))
# print('의사결정나무 test score : ', dt.score(test_scaled, test_target))


## 결정트리 plot
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# plt.figure(figsize = (10, 7))
# plot_tree(dt)
# plt.show()

# plot_tree(dt, max_depth = 1, filled = True, feature_names = ['alcohol', 'sugar', 'pH'])
# plt.show()

## 가지치기
# dt = DecisionTreeClassifier(max_depth = 3, random_state = 3)
# dt.fit(train_scaled, train_target)

# print(dt.score(train_scaled, train_target))
# print(dt.score(test_scaled, test_target))

# plt.figure(figsize = (20, 15))
# plot_tree(dt, filled = True, feature_names = ['alcohol', 'sugar', 'pH'])
# plt.show()

## 표준화 전 데이터
# dt = DecisionTreeClassifier(max_depth = 3, random_state = 3)
# dt.fit(train_input, train_target)

# print(dt.score(train_input, train_target))
# print(dt.score(test_input, test_target))

## 특성별 중요도 점수
# print(dt.feature_importances_)

## 검증세트로 분류하기

sub_input, val_input, sub_target, val_target = train_test_split(
    train_input, train_target, test_size = 0.2, random_state = 42
)
# print(sub_input.shape, val_input.shape)

# dt = DecisionTreeClassifier(random_state = 42)
# dt.fit(sub_input, sub_target)

## 검증세트 score
# print(dt.score(sub_input, sub_target))
# print(dt.score(val_input, val_target))


## 교차검증
from sklearn.model_selection import cross_validate

# 검증시간, test점수
scores = cross_validate(dt, train_input, train_target)
# print(scores)

# print(np.mean(scores['test_score']))


## K-Fold 검증
from sklearn.model_selection import StratifiedKFold

# scores = cross_validate(dt, train_input, train_target, cv = StratifiedKFold())
# print(np.mean(scores['test_score']))

# splitter = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)
# scores = cross_validate(dt, train_input, train_target, cv = splitter)
# print(np.mean(scores['test_score']))


## 하이퍼파라미터 튜닝
from sklearn.model_selection import GridSearchCV

# params = {'min_impurity_decrease' : [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
# gs = GridSearchCV(DecisionTreeClassifier(random_state = 42), params, n_jobs = -1)
# gs.fit(train_input, train_target)
# dt = gs.best_estimator_

# print(dt.score(train_input, train_target))
# # 최적 하이퍼파라미터
# print(gs.best_params_)
# # 각 파라미터별 score
# print(gs.cv_results_['mean_test_score'])
#
# best_index = np.argmax(gs.cv_results_['mean_test_score'])
# print(gs.cv_results_['params'][best_index])

# params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),
#           'max_depth' : range(5, 20, 1),
#           'min_samples_split' : range(2, 100, 10)}

# gs = GridSearchCV(DecisionTreeClassifier(random_state = 42), params, n_jobs = -1)
# gs.fit(train_input, train_target)
# print(gs.best_params_)

# print(np.max(gs.cv_results_['mean_test_score']))


## 랜덤 서치
from scipy.stats import uniform, randint
rgen = randint(0, 10)
rgen.rvs(10)


params = {'min_impurity_decrease': uniform(0.0001, 0.001),
          'max_depth' : randint(20, 50),
          'min_samples_split' : randint(2, 25),
          'min_samples_leaf' : randint(1, 25),
          }

from sklearn.model_selection import RandomizedSearchCV

# n_iter : 샘플갯수, n_jobs : CPU 사용범위 -1 전부사용
# gs = RandomizedSearchCV(DecisionTreeClassifier(random_state = 42),
#                         params, n_iter = 100, n_jobs = -1, random_state = 42)
#
# gs.fit(train_input, train_target)
#
# print(gs.best_params_)
# print(np.max(gs.cv_results_['mean_test_score']))
#
# dt = gs.best_estimator_
# print(dt.score(test_input, test_target))


## 랜덤포레스트

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs = -1, random_state = 42)
scores = cross_validate(rf, train_input, train_target, return_train_score = True, n_jobs = -1)

print(scores)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))

rf.fit(train_input, train_target)
print(rf.feature_importances_)

## Out of barge score : 추출 후 남은 데이터로 계산하기
rf = RandomForestClassifier(oob_score = True, n_jobs = -1, random_state = 42)
rf.fit(train_input, train_target)
# print(rf.oob_score_)

## 엑스트라트리

from sklearn.ensemble import ExtraTreesClassifier

# et = ExtraTreesClassifier(n_jobs = -1, random_state = 42)
# scores = cross_validate(et, train_input, train_target, return_train_score = True, n_jobs = -1)
# print(np.mean(scores['train_score']), np.mean(scores['test_score']))
#
# et.fit(train_input, train_target)
# print(et.feature_importances_)


## 그레디언트 부스팅
from sklearn.ensemble import GradientBoostingClassifier

# gb = GradientBoostingClassifier(random_state = 42)
# scores = cross_validate(gb, train_input, train_target, return_train_score = True, n_jobs = -1)
# print(np.mean(scores['train_score']), np.mean(scores['test_score']))
#
# gb = GradientBoostingClassifier(random_state = 42, n_estimators = 500, learning_rate = 0.2)
# scores = cross_validate(gb, train_input, train_target, return_train_score = True, n_jobs = -1)
# print(np.mean(scores['train_score']), np.mean(scores['test_score']))
#
# gb.fit(train_input, train_target)
# print(gb.feature_importances_)


## 히스토그램 기반 부스팅
from sklearn.ensemble import HistGradientBoostingClassifier
