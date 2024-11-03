
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
print(tf.__version__)

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
# 데이터 자체의 프레임을 조절해야함.

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

'''
    단항 분리 0과 1만 분리
'''
num_classes = 10 # 0~9까지 숫자
num_feacture = 784 # 28* 28

#Training parameters.
learning_rate = 0.0001 ## 넘어가는 사이즈
training_step = 50 ## 오차가 0이 되는 지점을 탐색하면 좋겠지만 불가능하기 떄문에 충분히 큰수로 지정
batch_size = 256
display_step = 50

'''
    x_train : 총 60,000개의 28x28 크기의 이미지
    y_train : 총 60,000개의 이미지 레이블(x_train에 대한 정답)
    x_test : 총 10,000개의 이미지
    y_test : 총 10,000개의 이미지 레이블(x_test에 대한 정답)

	수기로 작성된 0-9까지 데이터
	0이면 가장 검은색 255면 가장 밝은 값    
'''

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, y_train = map(list, zip(*[(x,y) for x, y in zip(x_train, y_train) if y ==0 or y== 1]))
x_test, y_test = map(list, zip(*[[x,y] for x, y in zip(x_train, y_train) if y ==0 or y== 1]))

x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
y_train, y_test = np.array(y_train, np.float32), np.array(y_test, np.float32)

x_train, x_test = x_train.reshape([-1, num_feacture]), x_test.reshape([-1, num_feacture])
x_train, x_test = x_train / 255.0, x_test / 255.0

b= np.random.uniform(-1, 1, num_feacture) ## 784개의 원소값을 가지는 벡터 생성
b0 = np.random.uniform(-1, 1)

# (1)
for step in range(training_step): 
    db = np.zeros(num_feacture, dtype='float32')
    db0 = 0.
    for x, y in zip(x_train, y_train):
        a = logistic_regression(x, b, b0)
        db += (y - a) * x  # 수정된 부분
        db0 += y - a
    b += learning_rate * db
    b0 += learning_rate * db0

pred = logistic_regression_wo_vectorization(x_test, b, b0)
print("Accuracy : ", accuracy(pred, y_test))

# (2)
# for step in range(training_step):
#     a = logistic_regression(x_train, b, b0)
    
#     # 벡터화된 업데이트 
#     db = np.dot(x_train.T, (y_train - a))  # (num_feature,)
#     db0 = np.sum(y_train - a)  # 스칼라
    
#     b += learning_rate * db
#     b0 += learning_rate * db0

# pred = logistic_regression(x_test, b, b0)
# print("Accuracy: ", accuracy(pred, y_test))

# (3)
# pred = np.vectorize(logistic_regression, signature='(n),(n),()->()')(x_test, b, b0)
# print("Accuracy: ", accuracy(pred, y_test))

'''
    다항 분리
'''

def logistic_regression_multi(X, b, b0):
    return 1. / (1. + np.exp(-np.dot(b, X) - b0))

num_classes = 10 # 0~9까지 숫자
num_feacture = 784 # 28* 28

#Training parameters.
learning_rate = 0.0001 ## 넘어가는 사이즈
training_step = 50 ## 오차가 0이 되는 지점을 탐색하면 좋겠지만 불가능하기 떄문에 충분히 큰수로 지정
batch_size = 256
display_step = 50

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = x_train.reshape([-1, num_feacture]), x_test.reshape([-1, num_feacture])
x_train, x_test = x_train /255. , x_test / 255.

b= np.random.uniform(-1, 1, num_feacture*num_classes).reshape((num_classes, num_feacture))
b0 = np.random.uniform(-1, 1, num_classes)

for step in range(training_step):
    db = np.zeros((num_classes, num_feacture), dtype= 'float32')
    db0 = np.zeros(num_classes, dtype='float32')
    
    for x, y in zip(x_train, y_train):
        yy = tf.one_hot(y, depth=num_classes).numpy()
        a= logistic_regression_multi(x, b, b0)
        db += np.matmul(np.expand_dims(yy-a, axis = 1), np.expand_dims(x, axis = 0))
        db0 += yy -a
    
    b += learning_rate * db
    b0 += learning_rate * db0
    
pred = np.argmax(np.array([logistic_regression_multi(x, b, b0) for x in x_test]), axis=1)
print("Accuracy: ", np.mean(pred == y_test))


# import numpy as np

# # 예시 관절 데이터 (타임스탬프와 관절 좌표)
# # pose_data는 [(timestamp, keypoints), ...] 형태로 저장되어 있다고 가정
# pose_data = [
#     (0.0, np.array([x1, y1, x2, y2, ...])),  # 프레임 1
#     (0.1, np.array([x1, y1, x2, y2, ...])),  # 프레임 2
#     # ... (더 많은 프레임)
#     (10.0, np.array([x1, y1, x2, y2, ...]))  # 프레임 n
# ]

# # 운동 구간을 설정 (t초부터 t+10초까지)
# t = 0.0  # 시작 시간
# duration = 10.0  # 구간 길이 (초)

# # 구간별 데이터 필터링
# def filter_motion_data(pose_data, start_time, duration):
#     filtered_data = []
#     end_time = start_time + duration
    
#     for timestamp, keypoints in pose_data:
#         if start_time <= timestamp < end_time:
#             filtered_data.append((timestamp, keypoints))
    
#     return filtered_data

# # t초부터 t+10초까지의 운동 데이터 필터링
# motion_set = filter_motion_data(pose_data, t, duration)

# # 운동 데이터 출력
# for timestamp, keypoints in motion_set:
#     print(f"Timestamp: {timestamp}, Keypoints: {keypoints}")

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset

# # 예시 데이터셋 클래스
# class MotionDataset(Dataset):
#     def __init__(self, motion_data):
#         self.motion_data = motion_data
    
#     def __len__(self):
#         return len(self.motion_data)
    
#     def __getitem__(self, idx):
#         timestamp, keypoints = self.motion_data[idx]
#         # 여기서는 keypoints를 텐서로 변환하고 레이블을 준비해야 합니다.
#         return torch.tensor(keypoints, dtype=torch.float32), 1  # 레이블은 예시로 1로 설정

# # 모델 정의
# class SimpleNN(nn.Module):
#     def __init__(self, input_size):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(input_size, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 1)  # 회귀 문제의 경우 1개의 출력

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# # 모델 학습 함수
# def train_model(model, dataset, num_epochs=100):
#     dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
#     criterion = nn.MSELoss()  # 회귀 문제의 경우 MSE
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     for epoch in range(num_epochs):
#         for inputs, labels in dataloader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels.view(-1, 1))  # 레이블 차원 맞추기
#             loss.backward()
#             optimizer.step()
        
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# # 사용 예시
# input_size = len(motion_set[0][1])  # keypoints의 길이
# dataset = MotionDataset(motion_set)
# model = SimpleNN(input_size)

# # 모델 학습
# train_model(model, dataset)