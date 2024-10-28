
'''
    ARIMA (AutoRegressive Integrated Moving Average): 전통적인 통계 모델로, 비정상성을 처리하기 위해 차분을 사용합니다.
    LSTM (Long Short-Term Memory): 순환 신경망(RNN)의 일종으로, 긴 시퀀스 데이터를 잘 처리할 수 있어 시계열 데이터에 적합합니다.
    GRU (Gated Recurrent Unit): LSTM과 유사하지만 구조가 간단해 계산 효율성이 좋습니다.
    Transformer: 최근에는 Transformer 모델도 시계열 데이터 예측에 활용되고 있습니다. 특히 자기회귀적 모델과 결합하여 사용할 수 있습니다.
'''

import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU 비활성화
# print(tf.__version__)

'''
    x_train : 총 60,000개의 28x28 크기의 이미지
    y_train : 총 60,000개의 이미지 레이블
    x_test : 총 10,000개의 이미지
    y_test : 총 10,000개의 이미지 레이블

	수기로 작성된 0-9까지 데이터
	0이면 가장 검은색 255면 가장 밝은 값    
'''

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# ##  이미지 데이터를 255로 나누어 0과 1 사이로 변환하는 것은 신경망 모델이 효과적으로 학습할 수 있도록 돕는 중요한 전처리 단계입니다.
# x_train, x_test = x_train / 255.0, x_test / 255.0

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape = (28,28)), ## 2D 이미지를 1D 벡터로 변환
#     tf.keras.layers.Dense(239, activation='relu'), ## 239개의 노드를 가진 완전 연결(Dense) 레이어
#     tf.keras.layers.Dropout(0.2), ## 20%의 노드를 무작위로 제거하여 과적합(overfitting)을 방지
#     tf.keras.layers.Dense(32, activation='relu'),  ## 32개의 노드를 가진 완전 연결(Dense) 레이어 (RELU 활성화 함수)
#     tf.keras.layers.Dropout(0.2), ## 또 다른 20%의 노드를 무작위로 제거
#     tf.keras.layers.Dense(10) ## 최종 레이어는 10개의 노드를 가진 완전 연결(Dense) 레이어입니다. 이 레이어는 MNIST 데이터셋의 10개의 클래스를 예측하기 위한 출력 레이어입니다. 
# ])
# loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
# model.compile(optimizer = 'adam', loss = loss_func, metrics = ['accuracy'])
# model.fit(x_train, y_train, epochs = 10)
# print(model.evaluate(x_test, y_test))

# a= np.array([[[1,2,3],[4,5,6]],[[0.1,0.2,0.3],[0.4,0.5,0.6]]])
# b = np.array([[[10,20],[30,40],[50,60]],[[1,2],[3,4],[5,6]]])
# print(np.matmul(a,b))
# print(np.dot(a,b))


def logistic_regression(X, b, b0):
    return 1. / (1. + np.exp(-np.dot(X, b) - b0))

def accuracy(y_pred, y_true):
    correction_prediction = np.equal(np.round(y_pred), y_true.astype(np.int64))
    return np.mean(correction_prediction.astype(np.float32))

def logistic_regression_wo_vectorization(x_test, b, b0):
    pred = list()
    for t in x_test:
        pred.append(logistic_regression(t, b, b0))
    return pred

num_classes = 10 # 0~9까지 숫자
num_feacture = 784 # 28* 28

#Training parameters.
learning_rate = 0.0001 ## 넘어가는 사이즈
training_step = 50 ## 오차가 0이 되는 지점을 탐색하면 좋겠지만 불가능하기 떄문에 충분히 큰수로 지정
batch_size = 256
display_step = 50

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, y_train = map(list, zip(*[(x,y) for x, y in zip(x_train, y_train) if y ==0 or y== 1]))
x_test, y_test = map(list, zip(*[[x,y] for x, y in zip(x_train, y_train) if y ==0 or y== 1]))

x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
y_train, y_test = np.array(y_train, np.int64), np.array(y_test, np.int64)

x_train, x_test = x_train.reshape([-1, num_feacture]), x_test.reshape([-1, num_feacture])
x_train, x_test = x_train / 255.0, x_test / 255.0

b= np.random.uniform(-1, 1, num_feacture)
b0 = np.random.uniform(-1, 1)

# for step in range(training_step): 
#     db = np.zeros(num_feacture, dtype='float32')
#     db0 = 0.
#     for x, y in zip(x_train, y_train):
#         a = logistic_regression(x, b, b0)
#         db += (y - a) * x  # 수정된 부분
#         db0 += y - a
#     b += learning_rate * db
#     b0 += learning_rate * db0

# pred = logistic_regression_wo_vectorization(x_test, b, b0)
# print("Accuracy : ", accuracy(pred, y_test))

for step in range(training_step):
    a = logistic_regression(x_train, b, b0)
    
    # 벡터화된 업데이트
    db = np.dot(x_train.T, (y_train - a))  # (num_feature,)
    db0 = np.sum(y_train - a)  # 스칼라
    
    b += learning_rate * db
    b0 += learning_rate * db0

pred = logistic_regression(x_test, b, b0)
print("Accuracy: ", accuracy(pred, y_test))

