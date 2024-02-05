###

## 분류 뉴런 만들기
## 퍼셉트론 : 이진분류 문제에서 최적의 가중치를 학습하는 알고리즘

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

### 로지스틱회귀에서 사용하는 시그모이드 함수에 대하여 알아봅시다옹
## 시그모이드 함수가 만들어지는 과정

# odds > logit > sigmoid

## odds ratio
# odds함수는 성공, 실패 확률의 비율을 나타내는 통계입니다.
# probs = np.arange(0, 1, 0.01)
# odds = [p/(1-p) for p in probs]
# plt.plot(probs, odds)
# plt.xlabel("p")
# plt.ylabel("p/(1-p)")
# plt.title("odds ratio")
# plt.show()


## logit
# logit은 odds함수에 log를 씌운 함수입니다.
# probs = np.arange(0.001, 0.999, 0.001)
# logit = [np.log(p/(1-p)) for p in probs]
# plt.plot(probs, logit)
# plt.xlabel('p')
# plt.ylabel('log(p/(1-p))')
# plt.title('logit function')
# plt.show()


## 로지스틱 함수
# zs = np.arange(-10., 10., 0.1)
# gs = [1/(1+np.exp(-z)) for z in zs]
# plt.plot(zs, gs)
# plt.xlabel('z')
# plt.ylabel('1/(1+e^-z)')
# plt.title('logistic regression')
# plt.show()


from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

# print(cancer.data.shape, cancer.target.shape)
# print(cancer)
# print(cancer.data[:3])

# plt.boxplot(cancer.data)
# plt.xlabel('feature')
# plt.ylabel('value')
# plt.show()

# print(cancer.feature_names[[3, 13, 23]])
# print(np.unique(cancer.target, return_counts=True))

x = cancer.data
y = cancer.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    stratify=y,
                                                    test_size=0.2,
                                                    random_state=42)

# print(x_train.shape, x_test.shape)
np.uniique(y_train, return_counts = True)

class LogisticNeuron:
    def __init__(self):
        self.w = None
        self.b = None

    # 직선 방정식 계산
    def forpass(self, x):
        z = np.sum(x * self.w) + self.b
        return z

    # 역전파 계산(예측결과와 실제결과값의 오차로 가중치와 편향을 예측)
    def backward(self, x, err):
        w_grad = x * err        # 가중치의 경사도 계산
        b_grad = 1 * err        #
        return w_grad, b_grad

    # 시그모이드 계산(활성화함수)
    def activation(self, z):
        z = np.clip(z, -100, None)
        a = 1 / (1 + np.exp(-z))
        return a

    def fit(self, x, y, epochs = 100):
        self.w = np.ones(x.shape[1])
        self.b = 0
        for i in range(epochs):
            for x_i, y_i in zip(x, y):
                z = self.forpass(x_i)
                a = self.activation()
                err = -(y_i - a)
                w_grad, b_grad = self.backward(x_i, err)
                self.w -= w_grad
                self.b -= b_grad

    # def predict(self, x):
