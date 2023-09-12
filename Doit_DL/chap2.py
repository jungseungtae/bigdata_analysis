from sklearn.datasets import load_diabetes

diabetes = load_diabetes()

# print(diabetes.data.shape, diabetes.target.shape)
# print(diabetes.data[0:3])
# print(diabetes.target[:3])

import matplotlib.pyplot as plt

plt.scatter(diabetes.data[:, 2], diabetes.target)
plt.xlabel("x")
plt.ylabel("y")
# plt.show()

x = diabetes.data[:, 2]
y = diabetes.target

x_sample = x[99:109]
# print(x_sample, x_sample.shape)

w = b = 1.0

# 경사하강법
# 무작위로 w와 b값을 정한 후 입력값 x를 선택하였을 때 결과값(y_hat)을 예측하는 것
# y = ax + b -> y_hat = wx + b
# x(w) = 입력데이터, y(y_hat) = 타겟 데이터

# w 값을 조정하여 예측값 바꾸기
# x[0]값을 대입하여 y값과 y_hat값의 차이를 비교
y_hat = x[0] * w + b
# print(y_hat)  # 예측값
# print(y[0])   # 실제값

# w의 값을 올려 예측값과 실제값의 차이를 조절
w_inc = w + 0.1
y_hat_inc = w_inc * x[0] + b
# print(y_hat_inc)

# 예측값 변동확인
w_rate = (y_hat_inc - y_hat) / (w_inc - w)
# print(w_rate)

# 가중치 상승
w_new = w + w_rate
# print(w_new)

# 변화율로 절편 업데이트(b)
b_inc = b + 0.1
y_hat_inc = x[0] * w + b_inc
# print(y_hat_inc)

b_rate = (y_hat_inc - y_hat) / (b_inc - b)
# print(b_rate)

## 오차 역전파로 가중치와 절편 업데이트

# 첫 번째 샘플
# err = 실제 y - 예측 y값
err = y[0] - y_hat
w_new = w + w_rate * err
b_new = b + 1 * err
# print(w_new, b_new)

# 두 번째 샘플
y_hat = x[1] * w_new + b_new
err = y[1] - y_hat
w_rate = x[1]

w_new = w_new + w_rate * err
b_new = b_new + 1 * err
# print(w_new, b_new)

for x_i, y_i in zip(x, y):
    y_hat = x_i * w + b
    err = y_i - y_hat
    w_rate = x_i
    w = w + w_rate * err
    b = b + 1 * err
# print(w, b)

plt.scatter(x, y)
pt1 = (-0.1, -0.1 * w + b)
pt2 = (0.15, 0.15 * w + b)
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
plt.xlabel("x")
plt.ylabel("y")
# plt.show()

## 적합한 모델 찾기
## y_hat : 예측값, err = 실제 y - 예측 y, w_rate = 가중치
for i in range(1, 100):
    for x_i, y_i in zip(x, y):
        y_hat = x_i * w + b
        err = y_i - y_hat
        w_rate = x_i
        w = w + w_rate * err
        b = b + 1 * err
        # print(x_i, y_i, y_hat, err, w_rate, w, b)
# print(w, b)

## 반복작업 후 w,b값 적용
plt.scatter(x, y)
pt1 = (-0.1, -0.1 * w + b)
pt2 = (0.15, 0.15 * w + b)
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
plt.xlabel("x")
plt.ylabel("y")
# plt.show()

## 모델로 예측하기(x_new를 입력하였을 때 y값을 예측)
x_new = 0.18
y_pred = x_new * w + b

plt.scatter(x, y)
plt.scatter(x_new, y_pred)
plt.xlabel('x')
plt.ylabel('y')
# plt.show()